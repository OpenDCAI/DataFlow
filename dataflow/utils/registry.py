import importlib
import importlib.util
import sys
import types
import os
import pkgutil
from contextlib import contextmanager
from dataflow.logger import get_logger
from pathlib import Path

from rich.console import Console
from rich.table import Table

import ast
from pathlib import Path

def generate_import_structure_from_type_checking(source_file: str, base_path: str) -> dict:
    source = Path(source_file).read_text(encoding="utf-8")
    tree = ast.parse(source)

    import_structure = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.If) and getattr(node.test, 'id', '') == 'TYPE_CHECKING':
            for subnode in node.body:
                if isinstance(subnode, ast.ImportFrom):
                    module_rel = subnode.module.replace(".", "/")
                    for alias in subnode.names:
                        name = alias.name
                        module_file = str(Path(base_path) / f"{module_rel}.py")
                        import_structure[name] = (module_file, name)

    return import_structure


class Registry():
    """
    The registry that provides name -> object mapping, to support third-party
    users' custom modules.

    To create a registry (e.g. a backbone registry):

    .. code-block:: python

        BACKBONE_REGISTRY = Registry('BACKBONE')

    To register an object:

    .. code-block:: python

        @BACKBONE_REGISTRY.register()
        class MyBackbone():
            ...

    Or:

    .. code-block:: python

        BACKBONE_REGISTRY.register(MyBackbone)
    """

    def __init__(self, name):
        """
        Args:
            name (str): the name of this registry
        """
        self._name = name
        self._obj_map = {}
        # Avoid re-parsing the source tree for repeated lookups of the same class.
        self._verified_precedence = {}
        self._base_folder = Path(__file__).resolve().parents[1]
        self._sub_modules = sorted(os.listdir(self._base_folder / name))
        if "__init__.py" in self._sub_modules:
            self._sub_modules.remove("__init__.py")
        if "__pycache__" in self._sub_modules:
            self._sub_modules.remove("__pycache__")
        if len(self._sub_modules) > 0:
            self.loader_map = dict(zip(self._sub_modules, [None] * len(self._sub_modules)))
        
    @contextmanager
    def _discovery_transaction(self):
        """Rollback discovery-owned state if an import fails.

        This includes import side effects in the other global registry. Previously
        loaded modules and manually registered objects are left intact.
        """
        registries = [
            value for value in globals().values()
            if isinstance(value, Registry)
        ]
        if self not in registries:
            registries.append(self)
        snapshots = [
            (registry, dict(registry._obj_map),
             dict(getattr(registry, "loader_map", {})),
             dict(registry._verified_precedence))
            for registry in registries
        ]
        loaded = set(sys.modules)
        lazy_cache = {
            id(module): (module, dict(module._loaded_classes))
            for module in sys.modules.values()
            if isinstance(module, LazyLoader)
        }
        try:
            yield
        except BaseException:
            for registry, objects, loaders, verified in snapshots:
                registry._obj_map.clear()
                registry._obj_map.update(objects)
                registry._verified_precedence.clear()
                registry._verified_precedence.update(verified)
                if hasattr(registry, "loader_map"):
                    registry.loader_map.clear()
                    registry.loader_map.update(loaders)
            for module, cache in lazy_cache.values():
                module._loaded_classes.clear()
                module._loaded_classes.update(cache)
            # Python normally removes only the failing module. Dependencies that
            # imported successfully during the attempt must also be discarded.
            for module_name in sorted(set(sys.modules) - loaded, reverse=True):
                module = sys.modules.pop(module_name, None)
                parent_name, separator, child_name = module_name.rpartition(".")
                if separator:
                    parent = sys.modules.get(parent_name)
                    if parent is not None and parent.__dict__.get(child_name) is module:
                        del parent.__dict__[child_name]
            raise

    def _load_module(self, module_name):
        entry = Path(module_name).stem if module_name.endswith(".py") else module_name
        module = importlib.import_module(f"dataflow.{self._name}.{entry}")
        self.loader_map[module_name] = module
        return module

    def _init_loaders(self):
        with self._discovery_transaction():
            for module_name in self.loader_map:
                if self.loader_map[module_name] is None:
                    self._load_module(module_name)

    def _do_register(self, name, obj):
        existing = self._obj_map.get(name)
        if existing is None:
            self._obj_map[name] = obj
            self._verified_precedence.pop(name, None)
            return
        if existing is obj:
            return
        # Preserve the public first-registration-wins contract for user
        # extensions. Built-in discovery conflicts use a stable source ranking.
        namespace = f"dataflow.{self._name}."
        old_module = getattr(existing, "__module__", "")
        new_module = getattr(obj, "__module__", "")
        if old_module.startswith(namespace) and new_module.startswith(namespace):
            if (new_module, getattr(obj, "__qualname__", name)) < (
                old_module, getattr(existing, "__qualname__", name)
            ):
                self._obj_map[name] = obj
                self._verified_precedence.pop(name, None)

    def register(self, obj=None):
        """Register an object under its class/function name."""
        if obj is None:
            def deco(func_or_class):
                self._do_register(func_or_class.__name__, func_or_class)
                return func_or_class
            return deco
        self._do_register(obj.__name__, obj)

    def _source_declares_registration(self, source, name):
        """Recognize statically attributable registrations for this registry.

        This rejects comments, string mentions, unrelated decorated classes,
        and registrations targeting a different registry. Dynamic aliases and
        computed registration names require explicit full discovery.
        """
        try:
            tree = ast.parse(source.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, SyntaxError):
            return False

        # Canonical names of this exact registry object in this module.
        registry_names = {
            symbol for symbol, value in globals().items() if value is self
        }
        imported_names = set(registry_names)
        registry_modules = set()
        for statement in tree.body:
            if isinstance(statement, ast.ImportFrom):
                if statement.module in ("dataflow.utils.registry", "dataflow.utils"):
                    for alias in statement.names:
                        if alias.name in registry_names:
                            imported_names.add(alias.asname or alias.name)
                        if alias.name == "registry" and statement.module == "dataflow.utils":
                            registry_modules.add(alias.asname or alias.name)
            elif isinstance(statement, ast.Import):
                for alias in statement.names:
                    if alias.name == "dataflow.utils.registry" and alias.asname:
                        registry_modules.add(alias.asname)

        def dotted(node):
            if isinstance(node, ast.Name):
                return node.id
            if isinstance(node, ast.Attribute):
                return f"{dotted(node.value)}.{node.attr}"
            return ""

        def owned_registration(node):
            # Registry.register supports both @registry.register and
            # @registry.register() in addition to registry.register(Class).
            target_call = node.func if isinstance(node, ast.Call) else node
            if not (
                isinstance(target_call, ast.Attribute)
                and target_call.attr == "register"
            ):
                return False
            target = target_call.value
            if isinstance(target, ast.Name):
                return target.id in imported_names
            if isinstance(target, ast.Attribute) and target.attr in registry_names:
                qualifier = dotted(target.value)
                return qualifier in registry_modules or qualifier in (
                    "dataflow.utils.registry", "dataflow.utils"
                )
            return False

        def expression_registers(node):
            """A direct registration call evaluated by this statement."""
            return any(
                isinstance(child, ast.Call)
                and child.args
                and isinstance(child.args[0], ast.Name)
                and child.args[0].id == name
                and owned_registration(child)
                for child in ast.walk(node)
            )

        def static_truth(node):
            """Evaluate only literal conditions; do not execute arbitrary code."""
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
                operand = static_truth(node.operand)
                return None if operand is None else not operand
            if isinstance(node, ast.BoolOp):
                values = [static_truth(value) for value in node.values]
                if isinstance(node.op, ast.And):
                    if False in values:
                        return False
                    if all(value is True for value in values):
                        return True
                elif isinstance(node.op, ast.Or):
                    if True in values:
                        return True
                    if all(value is False for value in values):
                        return False
                return None
            try:
                return bool(ast.literal_eval(node))
            except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
                return None

        def block_registers(statements):
            # Unlike ast.walk, this respects known-dead branches and does not
            # descend into functions whose bodies run only if called.
            for statement in statements:
                if isinstance(statement, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                    if statement.name == name and any(
                        owned_registration(decorator)
                        for decorator in statement.decorator_list
                    ):
                        return True
                    if isinstance(statement, ast.ClassDef) and block_registers(statement.body):
                        return True
                elif isinstance(statement, ast.If):
                    condition = static_truth(statement.test)
                    if condition is not False and block_registers(statement.body):
                        return True
                    if condition is not True and block_registers(statement.orelse):
                        return True
                elif isinstance(statement, (ast.While, ast.For, ast.AsyncFor)):
                    if isinstance(statement, ast.While) and static_truth(statement.test) is False:
                        if block_registers(statement.orelse):
                            return True
                    elif block_registers(statement.body) or block_registers(statement.orelse):
                        return True
                elif isinstance(statement, (ast.Try, getattr(ast, "TryStar", ast.Try))):
                    if block_registers(statement.body):
                        return True
                    if any(block_registers(handler.body) for handler in statement.handlers):
                        return True
                    if block_registers(statement.orelse) or block_registers(statement.finalbody):
                        return True
                elif isinstance(statement, (ast.With, ast.AsyncWith)):
                    if block_registers(statement.body):
                        return True
                elif isinstance(statement, ast.Match):
                    for case in statement.cases:
                        if case.guard is not None and static_truth(case.guard) is False:
                            continue
                        if block_registers(case.body):
                            return True
                else:
                    if expression_registers(statement):
                        return True
                if isinstance(statement, (ast.Raise, ast.Return, ast.Break, ast.Continue)):
                    break
            return False

        return block_registers(tree.body)

    def _resolve_prior_builtin_candidates(self, name, current):
        """Find higher-ranked built-in registrations without full discovery.

        A top-level package being initialized does not imply that nested
        modules have been imported, so this check is independent of loader_map.
        """
        prefix = f"dataflow.{self._name}."
        current_module = getattr(current, "__module__", "")
        if not current_module.startswith(prefix):
            return

        package = importlib.import_module(f"dataflow.{self._name}")
        for location in getattr(package, "__path__", ()):
            root = Path(location)
            for source in sorted(root.rglob("*.py")):
                parts = source.relative_to(root).with_suffix("").parts
                if parts[-1] == "__init__":
                    parts = parts[:-1]
                if not parts:
                    continue
                contender = prefix + ".".join(parts)
                if contender >= current_module or contender in sys.modules:
                    continue
                if self._source_declares_registration(source, name):
                    importlib.import_module(contender)

    def get(self, name):
        ret = self._obj_map.get(name)
        if ret is not None:
            if getattr(ret, "__module__", "").startswith(
                f"dataflow.{self._name}."
            ):
                if self._verified_precedence.get(name) is ret:
                    return ret
                with self._discovery_transaction():
                    self._resolve_prior_builtin_candidates(name, ret)
                resolved = self._obj_map[name]
                self._verified_precedence[name] = resolved
                return resolved
            return ret
        logger = get_logger()
        with self._discovery_transaction():
            if None in self.loader_map.values():
                self._init_loaders()
            # Built-in registration may have populated the map during import.
            ret = self._obj_map.get(name)
            if ret is not None:
                return ret
            for module_lib in self.loader_map.values():
                try:
                    clss = getattr(module_lib, name)
                except AttributeError:
                    continue
                self._do_register(name, clss)
            ret = self._obj_map.get(name)
            if ret is not None:
                return ret
        logger.error(f"No object named '{name}' found in '{self._name}' registry!")
        raise KeyError(f"No object named '{name}' found in '{self._name}' registry!")

    def apply_whitelist(self, names, *, verbose: bool = True):
        """
        在已调用 _get_all() 之后执行：
        仅保留 `names`（白名单）中的对象，其他对象从 _obj_map 中移除。
        同时返回一份详尽报告，并（可选）输出日志摘要。

        :param names: Iterable[str]，白名单
        :param verbose: 是否用 logger 输出摘要信息
        :return: dict 报告，字段如下：
            {
                "requested_whitelist": [str, ...],        # 传入的白名单（去重、排序后）
                "missing_in_registry": [str, ...],        # 在白名单里但不在 obj_map 的
                "kept": [str, ...],                       # 实际保留
                "removed": [str, ...],                    # 实际移除
                "total_before": int,                      # 裁剪前数量
                "total_after": int,                       # 裁剪后数量
                "trimmed_by": int                         # 裁掉的数量（= before - after）
            }
        """
        logger = get_logger()

        # 1) 规范化输入
        names = [] if names is None else list(names)
        keep_set = set(map(str, names))

        # 2) 快照当前注册表
        before_keys = set(self._obj_map.keys())
        total_before = len(before_keys)

        # 3) 计算三类集合
        missing_in_registry = sorted(keep_set - before_keys)       # 白名单中缺失的
        to_keep = sorted(before_keys & keep_set)                    # 真正保留的
        to_remove = sorted(before_keys - keep_set)                  # 将被移除的（不在白名单）

        # 4) 执行就地裁剪
        for k in to_remove:
            self._obj_map.pop(k, None)
            self._verified_precedence.pop(k, None)

        # 5) 统计结果
        total_after = len(self._obj_map)
        trimmed_by = total_before - total_after

        # 6) 构造报告
        report = {
            "requested_whitelist": sorted(keep_set),
            "missing_in_registry": missing_in_registry,
            "kept": to_keep,
            "removed": to_remove,
            "total_before": total_before,
            "total_after": total_after,
            "trimmed_by": trimmed_by,
        }

        # 7) 可读日志摘要（尽可能合理且完整）
        if verbose:
            logger.info(
                f"[Registry:{self._name}] whitelist applied: "
                f"before={total_before}, after={total_after}, trimmed_by={trimmed_by}"
            )
            if to_keep:
                logger.info(f"[Registry:{self._name}] kept ({len(to_keep)}): {to_keep}")
            else:
                logger.info(f"[Registry:{self._name}] kept (0): []")

            if to_remove:
                logger.info(f"[Registry:{self._name}] removed ({len(to_remove)}): {to_remove}")
            else:
                logger.info(f"[Registry:{self._name}] removed (0): []")

            if missing_in_registry:
                logger.warning(
                    f"[Registry:{self._name}] in-whitelist-but-missing "
                    f"({len(missing_in_registry)}): {missing_in_registry}"
                )

        return report
    
    def __contains__(self, name):
        return name in self._obj_map

    def __iter__(self):
        return iter(self._obj_map.items())

    def keys(self):
        return self._obj_map.keys()

    def __repr__(self):
        table = Table(title=f'Registry of {self._name}')
        table.add_column('Names', justify='left', style='cyan')
        table.add_column('Objects', justify='left', style='green')

        for name, obj in sorted(self._obj_map.items()):
            table.add_row(name, str(obj))

        console = Console()
        with console.capture() as capture:
            console.print(table, end='')

        return capture.get()

    def _get_all(self):
        with self._discovery_transaction():
            for module_name in self.loader_map:
                loader = self.loader_map[module_name]
                if loader is None:
                    loader = self._load_module(module_name)
                import_all = getattr(loader, "_import_all", None)
                if callable(import_all):
                    import_all()
                elif hasattr(loader, "__path__"):
                    for info in sorted(
                        pkgutil.walk_packages(
                            loader.__path__, prefix=f"{loader.__name__}."
                        ), key=lambda item: item.name
                    ):
                        importlib.import_module(info.name)

    def get_obj_map(self):
        """
        Get the object map of the registry.
        """
        return self._obj_map
    
    def get_type_of_objects(self):
        """
        Classify the object type by its path of registration.
        This is used to classify objects into different categories.
        :return: A dictionary with object type as keys and their name as values.
        """
        object_types_dict = {}
        for name, obj in self._obj_map.items():
            module_str = obj.__module__
            # print(obj.__name__, module_str)
            parts = module_str.split(".")
            object_types_dict[name] = parts[0:]
        return object_types_dict

OPERATOR_REGISTRY = Registry(name='operators')

PROMPT_REGISTRY = Registry(name='prompts')

class LazyLoader(types.ModuleType):

    def __init__(self, name, path, import_structure):
        """
        初始化 LazyLoader 模块。

        :param name: 模块名称
        :param import_structure: 定义类名到文件路径的映射字典
        """
        super().__init__(name)
        self._import_structure = import_structure
        self._loaded_classes = {}
        self._base_folder = Path(__file__).resolve().parents[2]
        self.__path__ = [path]
        self.__all__ = list(import_structure.keys())
        
    def _import_all(self):
        for cls_name in self.__all__:
            self.__getattr__(cls_name)

    def _load_class_from_file(self, file_path, class_name):
        """
        从指定文件中加载类。

        :param file_path: 脚本文件的路径
        :param class_name: 类的名字
        :return: 类对象
        """
        p = Path(file_path)
        if p.is_absolute():
            abs_file_path = str(p)
        else:
            abs_file_path = str(Path(self._base_folder) / p)
        if not os.path.exists(abs_file_path):
            raise FileNotFoundError(abs_file_path)
        rel_path = Path(abs_file_path).relative_to(self._base_folder)
        # 去掉后缀得到 ('dataflow', 'operators', 'generate', ... , 'question_generator')
        rel_parts = rel_path.with_suffix('').parts
        prefix_parts = tuple(self.__name__.split('.'))
        if rel_parts[:len(prefix_parts)] == prefix_parts:
            rel_parts = rel_parts[len(prefix_parts):]
        mod_name = '.'.join((*prefix_parts, *rel_parts))
        logger = get_logger()
        # 动态加载模块

        try:
            parts = mod_name.split(".")
            for i in range(1, len(parts)):
                parent = ".".join(parts[:i])
                if parent not in sys.modules:
                    dummy_mod = importlib.util.module_from_spec(
                        importlib.util.spec_from_loader(parent, loader=None)
                    )
                    dummy_mod.__path__ = [os.path.dirname(abs_file_path)]
                    sys.modules[parent] = dummy_mod

            spec = importlib.util.spec_from_file_location(mod_name, abs_file_path)
            logger.debug(f"LazyLoader {self.__path__} successfully imported spec {spec.__str__()}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[mod_name] = module
            logger.debug(f"LazyLoader {self.__path__} successfully imported module {module.__str__()} from spec {spec.__str__()}")
            logger.debug(f"Module name: {module.__name__}")
            logger.debug(f"Module file: {module.__file__}")
            logger.debug(f"Module package: {module.__package__}")
            spec.loader.exec_module(module)

        except BaseException as e:
            # Never leave a partially executed module in sys.modules.
            if "mod_name" in locals() and sys.modules.get(mod_name) is locals().get("module"):
                sys.modules.pop(mod_name, None)
            import traceback
            logger.exception("Import failed for %s from %s", mod_name, abs_file_path)
            # 可选：再打印一份，CI 日志更清楚
            print("=== Lazy import traceback ===")
            print("Module:", mod_name)
            print("File:", abs_file_path)
            print("".join(traceback.format_exception(e)))
            raise


        # 提取类
        if not hasattr(module, class_name):
            raise AttributeError(f"Class {class_name} not found in {abs_file_path}")
        return getattr(module, class_name)

    def __getattr__(self, item):
        """
        动态加载类。

        :param item: 类名
        :return: 动态加载的类对象
        """
        logger = get_logger()
        if item in self._loaded_classes:
            cls = self._loaded_classes[item]
            logger.debug(f"Lazyloader {self.__path__} got cached class {cls}")
            return cls
        # 从映射结构中获取文件路径和类名
        if item in self._import_structure:
            file_path, class_name = self._import_structure[item]
            logger.info(f"Lazyloader {self.__path__} trying to import {item} ")
            # Direct attribute access bypasses Registry.get()/_get_all();
            # it must still rollback decorators executed by a failed module.
            with OPERATOR_REGISTRY._discovery_transaction():
                cls = self._load_class_from_file(file_path, class_name)
            logger.debug(f"Lazyloader {self.__path__} got and cached class {cls}")
            self._loaded_classes[item] = cls
            return cls
        logger.debug(f"Module {self.__name__} has no attribute {item}")
        raise AttributeError(f"Module {self.__name__} has no attribute {item}")
