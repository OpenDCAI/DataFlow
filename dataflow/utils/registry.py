import importlib
import importlib.util
import sys
import types
import os
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
        self._base_folder = Path(__file__).resolve().parents[1]
        self._sub_modules = list(os.listdir(self._base_folder / name))
        if "__init__.py" in self._sub_modules:
            self._sub_modules.remove("__init__.py")
        if "__pycache__" in self._sub_modules:
            self._sub_modules.remove("__pycache__")
        if len(self._sub_modules) > 0:
            self.loader_map = dict(zip(self._sub_modules, [None] * len(self._sub_modules)))
        
    def _init_loaders(self):
        for module_name in self.loader_map.keys():
            module_path = f"dataflow.{self._name}.{module_name}"
            self.loader_map[module_name] = importlib.import_module(module_path)

    def _do_register(self, name, obj):
        if name not in self._obj_map:
            self._obj_map[name] = obj

    def register(self, obj=None):
        """
        Register the given object under the the name `obj.__name__`.
        Can be used as either a decorator or not.
        See docstring of this class for usage.
        """
        if obj is None:
            # used as a decorator
            def deco(func_or_class):
                name = func_or_class.__name__
                self._do_register(name, func_or_class)
                return func_or_class

            return deco

        # used as a function call
        name = obj.__name__
        self._do_register(name, obj)

    def get(self, name):
        ret = self._obj_map.get(name)
        logger = get_logger()
        if ret is None:
            if None in self.loader_map.values():
                self._init_loaders()
            for module_lib in self.loader_map.values():
                # module_path = "dataflow.operators." + x
                try:
                    # module_lib = importlib.import_module(module_path)
                    clss = getattr(module_lib, name)
                    self._obj_map[name] = clss
                    return clss
                except AttributeError as e:
                    logger.debug(f"{str(e)}")
                    continue
                except Exception as e:
                    raise e
            logger.error(f"No object named '{name}' found in '{self._name}' registry!")
            raise KeyError(f"No object named '{name}' found in '{self._name}' registry!")

        if ret is None:
            logger.error(f"No object named '{name}' found in '{self._name}' registry!")
        assert ret is not None, f"No object named '{name}' found in '{self._name}' registry!"
        
        return ret
    
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
        if None in self.loader_map.values():
            self._init_loaders()
        for loader in self.loader_map.values():
            loader._import_all()

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
            object_types_dict[name] = parts[1:]
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

        except Exception as e:
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
            cls = self._load_class_from_file(file_path, class_name)
            logger.debug(f"Lazyloader {self.__path__} got and cached class {cls}")
            self._loaded_classes[item] = cls
            return cls
        logger.debug(f"Module {self.__name__} has no attribute {item}")
        raise AttributeError(f"Module {self.__name__} has no attribute {item}")
