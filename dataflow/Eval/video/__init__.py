# from .video_aesthetic_scorer import VideoAestheticScorer
# from .video_motion_scorer import VideoMotionScorer
# from .video_resolution_scorer import VideoResolutionScorer
# from .fastvqa_scorer import FastVQAScorer, FasterVQAScorer
# from .dover_scorer import DOVERScorer
# from .emscorer import EMScorer
# from .pacscorer import PACScorer

# __all__ = [
#     'VideoAestheticScorer',
#     'VideoMotionScorer',
#     'VideoResolutionScorer',
#     'FastVQAScorer',
#     'FasterVQAScorer',
#     'DOVERScorer',
#     'EMScorer',
#     'PACScorer'
# ]
# import os
# import importlib
# from types import ModuleType

# package_prefix = 'dataflow.Eval.video'
# class_mapping = {
#     'VideoAestheticScorer': '.video_aesthetic_scorer',
#     'VideoMotionScorer': '.video_motion_scorer',
#     'VideoResolutionScorer': '.video_resolution_scorer',
#     'FastVQAScorer': '.fastvqa_scorer',
#     'FasterVQAScorer': '.fastvqa_scorer',
#     'DOVERScorer': '.dover_scorer',
#     'EMScorer': '.emscorer',
#     'PACScorer': '.pacscorer'
# }


# class LazyLoader(ModuleType):
#     def __init__(self, name, package_prefix):
#         super().__init__(name)
#         self._package_prefix = package_prefix
#         self._loaded_modules = {}
#         self.__path__ = []

#     def __getattr__(self, item):
#         if item in self._loaded_modules:
#             return self._loaded_modules[item]
#         print(item)
#         # for file_name in os.listdir(self._base_path):
#         #     if file_name.endswith(".py") and file_name != "__init__.py":
#         #         module_name = file_name[:-3]
#         #         module_path = f"{self._package_prefix}.{module_name}"
#         #         module = importlib.import_module(module_path)
#         #         if hasattr(module, item):
#         #             self._loaded_modules[item] = getattr(module, item)
#         #             return self._loaded_modules[item]
#         module = importlib.import_module(package_prefix + class_mapping[item])
#         if hasattr(module, item):
#             self._loaded_modules[item] = getattr(module, item)
#             return self._loaded_modules[item]

#         raise AttributeError(f"Module {self._package_prefix} has no attribute {item}")

# import sys
# base_path = os.path.dirname(__file__)
# sys.modules[__name__] = LazyLoader(__name__, package_prefix)
import sys
from dataflow.utils.registry import LazyLoader


# 定义类名和文件路径的映射
_import_structure = {
    "VideoMotionScorer": ("dataflow/Eval/video/video_motion_scorer.py", "VideoMotionScorer"),
    "FastVQAScorer": ("dataflow/Eval/video/fastvqa_scorer.py", "FastVQAScorer"),
    "FasterVQAScorer": ("dataflow/Eval/video/fastvqa_scorer.py", "FasterVQAScorer"),
    "DOVERScorer": ("dataflow/Eval/video/dover_scorer.py", "DOVERScorer"),
    "EMScorer": ("dataflow/Eval/video/emscorer.py", "EMScorer"),
    "PACScorer": ("dataflow/Eval/video/pacscorer.py", "PACScorer"),
    "VideoResolutionScorer": ("dataflow/Eval/video/video_resolution_scorer.py", "VideoResolutionScorer")
}

# 替换当前模块为 LazyLoader
sys.modules[__name__] = LazyLoader(__name__, "dataflow/Eval/video", _import_structure)
