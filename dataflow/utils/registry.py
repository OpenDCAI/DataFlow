import importlib
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
        if ret is None:
            if self._name == 'model': 
                for x in ['Text', 'image', 'video']:
                    module_path = "dataflow.Eval." + x
                    try:
                        module_lib = importlib.import_module(module_path)
                        clss = getattr(module_lib, name)
                        self._obj_map[name] = clss
                        return clss
                    except AttributeError as e:
                        print(e)
                        continue
                    except Exception as e:
                        raise e
                raise KeyError(f"No object named '{name}' found in '{self._name}' registry!")
            elif self._name == "processor":
                for x in ['text.refiners', 'text.filters', 'text.deduplicators', 'text.reasoners', 'image.filters', 'image.deduplicators', 'video.filters']:
                # for x in ['image.filters', 'image.refiners']:
                    module_path = "dataflow.process." + x
                    try:
                        module_lib = importlib.import_module(module_path)
                        clss = getattr(module_lib, name)
                        self._obj_map[name] = clss
                        return clss
                    except AttributeError as e:
                        print(e)
                        continue
                    except Exception as e:
                        raise e
                raise KeyError(f"No object named '{name}' found in '{self._name}' registry!")
        
        assert ret is not None, f"No object named '{name}' found in '{self._name}' registry!"
        
        return ret

    def __contains__(self, name):
        return name in self._obj_map

    def __iter__(self):
        return iter(self._obj_map.items())

    def keys(self):
        return self._obj_map.keys()


MODEL_REGISTRY = Registry('model')
FORMATTER_REGISTRY = Registry('formatter')
PROCESSOR_REGISTRY = Registry('processor')

import importlib.util
import types
import os

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
        self.__path__ = [path]

    def _load_class_from_file(self, file_path, class_name):
        """
        从指定文件中加载类。

        :param file_path: 脚本文件的路径
        :param class_name: 类的名字
        :return: 类对象
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} does not exist")
        
        # 动态加载模块
        spec = importlib.util.spec_from_file_location(class_name, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # 提取类
        if not hasattr(module, class_name):
            raise AttributeError(f"Class {class_name} not found in {file_path}")
        return getattr(module, class_name)

    def __getattr__(self, item):
        """
        动态加载类。

        :param item: 类名
        :return: 动态加载的类对象
        """
        if item in self._loaded_classes:
            return self._loaded_classes[item]
        # 从映射结构中获取文件路径和类名
        if item in self._import_structure:
            file_path, class_name = self._import_structure[item]
            cls = self._load_class_from_file(file_path, class_name)
            self._loaded_classes[item] = cls
            return cls
        raise AttributeError(f"Module {self.__name__} has no attribute {item}")
