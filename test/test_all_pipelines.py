import importlib
import os
import inspect
from typing import Generic, List, Optional, Protocol, TypeVar, Union, Dict
from dataclasses import dataclass
from dataflow.logger import get_logger
from dataflow.cli_funcs.paths import DataFlowPath 

import importlib.util
import os
import ast
import inspect
from dataclasses import dataclass
from typing import Optional
@dataclass
class PipelineInfo:
    name: str
    category: str
    class_obj: type
    file_path: str
    description: str = ""

class TestAllPipelines:
    def __init__(self):
        self.pipelines = {}
        self.category_list = [
            "api_pipelines",
            "gpu_pipelines",
            "cpu_pipelines",
            os.path.join("..", "playground", "playground"),
        ]
        self.llm_serving
        
        
        self._discover_pipelines()
        print(f"Discovered {len(self.pipelines)} pipelines.")
        for name, info in self.pipelines.items():
            print(f"- {name} ({info.category}): {info.description}")

    def _discover_pipelines(self):
        pipeline_base = os.path.join(DataFlowPath.get_dataflow_statics_dir(), "pipelines")
        for category in self.category_list:
            full_dir = os.path.join(pipeline_base, category)
            if not os.path.isdir(full_dir):
                continue
            for fn in os.listdir(full_dir):
                if not fn.endswith(".py") or fn.startswith("__"):
                    continue
                try:
                    info = self._analyze_pipeline(category, fn, full_dir)
                    if info:
                        self.pipelines[info.name] = info
                except Exception as e:
                    print(f"[Warning] analyze pipeline `{fn}` failed: {e}")

    def _extract_class_name(self, file_path: str) -> str:
        """用 AST 找到文件中唯一的 class 名称，保证文件里只有一个 class。"""
        with open(file_path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read())
        classes = [n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
        if len(classes) != 1:
            raise ImportError(f"Expected exactly one class in {file_path}, but found: {classes}")
        return classes[0]

    def count_by_category(self) -> Dict[str, int]:
        """
        返回一个 dict，key 为 category，value 为该 category 下的 pipelines 数量
        """
        stats: Dict[str, int] = {}
        for info in self.pipelines.values():
            stats[info.category] = stats.get(info.category, 0) + 1
        return stats

    def _analyze_pipeline(self, category: str, file_name: str, dir_path: str) -> Optional[PipelineInfo]:
        file_path = os.path.join(dir_path, file_name)

        # 1) 先拿到这个文件里唯一的 class 名
        class_name = self._extract_class_name(file_path)

        # 2) 动态 load module
        spec = importlib.util.spec_from_file_location(class_name, file_path)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # 3) 直接通过 getattr 拿到 class
        cls = getattr(module, class_name, None)
        if cls is None or not inspect.isclass(cls):
            return None

        # 4) 你可以用 docstring 或者 file_name 作为描述
        desc = cls.__doc__.strip() if cls.__doc__ else f"{class_name} pipeline"

        return PipelineInfo(
            name=class_name,
            category=category,
            class_obj=cls,
            file_path=file_path,
            description=desc
        )

if __name__ == "__main__":
    tester = TestAllPipelines()
    static_cate = tester.count_by_category()
    print(static_cate)
