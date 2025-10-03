# dataflow/dataflowagent/toolkits/pipeline_assembler.py
from __future__ import annotations

import importlib
import inspect
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

from dataflow import get_logger
from dataflow.utils.registry import OPERATOR_REGISTRY

log = get_logger()


def snake_case(name: str) -> str:
    """
    Convert CamelCase (with acronyms) to snake_case.
    Examples:
        SQLGenerator -> sql_generator
        HTTPRequest -> http_request
    """
    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    s2 = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1)
    return s2.replace("__", "_").lower()


def try_import(module_path: str) -> bool:
    try:
        importlib.import_module(module_path)
        return True
    except Exception as e:
        log.warning(f"[pipeline_assembler] import {module_path} failed: {e}")
        return False


def build_stub(cls_name: str, module_path: str) -> str:
    return (
        f"# Fallback stub for {cls_name}, original module '{module_path}' not found\n"
        f"class {cls_name}:  # type: ignore\n"
        f"    def __init__(self, *args, **kwargs):\n"
        f"        import warnings; warnings.warn(\n"
        f"            \"Stub operator {cls_name} used, module '{module_path}' missing.\"\n"
        f"        )\n"
        f"    def run(self, *args, **kwargs):\n"
        f"        return kwargs.get(\"storage\")  # 透传\n"
    )


def group_imports(op_names: List[str]) -> Tuple[List[str], List[str], Dict[str, type]]:
    """
    Returns:
        imports: list of import lines
        stubs: list of stub class code blocks
        op_classes: mapping from provided operator name -> actual class object
    """
    imports: List[str] = []
    stubs: List[str] = []
    op_classes: Dict[str, type] = {}

    module2names: Dict[str, List[str]] = defaultdict(list)

    for name in op_names:
        cls = OPERATOR_REGISTRY.get(name)
        if cls is None:
            raise KeyError(f"Operator <{name}> not in OPERATOR_REGISTRY")

        op_classes[name] = cls
        mod = cls.__module__
        if try_import(mod):
            module2names[mod].append(cls.__name__)
        else:
            stubs.append(build_stub(cls.__name__, mod))

    for m in sorted(module2names.keys()):
        names = sorted(set(module2names[m]))
        imports.append(f"from {m} import {', '.join(names)}")

    return imports, stubs, op_classes


def _format_default(val: Any) -> str:
    """
    Produce a code string for a default value.
    If default is missing (inspect._empty), we return 'None' to keep code runnable.
    """
    if val is inspect._empty:
        return "None"
    if isinstance(val, str):
        return repr(val)
    return repr(val)


def extract_op_params(cls: type) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], bool]:
    """
    Inspect 'cls' for __init__ and run signatures.

    Returns:
        init_kwargs: list of (param_name, code_str_default) for __init__ (excluding self)
        run_kwargs: list of (param_name, code_str_default) for run (excluding self and storage)
        run_has_storage: whether run(...) has 'storage' parameter
    """
    # ---- __init__
    init_kwargs: List[Tuple[str, str]] = []
    try:
        init_sig = inspect.signature(cls.__init__)
        for p in list(init_sig.parameters.values())[1:]:  # skip self
            if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                continue
            init_kwargs.append((p.name, _format_default(p.default)))
    except Exception as e:
        log.warning(f"[pipeline_assembler] inspect __init__ of {cls.__name__} failed: {e}")

    # ---- run
    run_kwargs: List[Tuple[str, str]] = []
    run_has_storage = False
    if hasattr(cls, "run"):
        try:
            run_sig = inspect.signature(cls.run)
            params = list(run_sig.parameters.values())[1:]  # skip self
            for p in params:
                if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                    continue
                if p.name == "storage":
                    run_has_storage = True
                    continue
                run_kwargs.append((p.name, _format_default(p.default)))
        except Exception as e:
            log.warning(f"[pipeline_assembler] inspect run of {cls.__name__} failed: {e}")

    return init_kwargs, run_kwargs, run_has_storage


def render_operator_blocks(op_names: List[str], op_classes: Dict[str, type]) -> Tuple[str, str]:
    """
    Render operator initialization lines and forward-run lines without leading indentation.
    Indentation will be applied by build_pipeline_code when inserting into the template.
    """
    init_lines: List[str] = []
    forward_lines: List[str] = []

    for name in op_names:
        cls = op_classes[name]
        var_name = snake_case(cls.__name__)

        init_kwargs, run_kwargs, run_has_storage = extract_op_params(cls)

        # Inject pipeline context where appropriate
        rendered_init_args: List[str] = []
        for k, v in init_kwargs:
            if k == "llm_serving":
                rendered_init_args.append(f"{k}=self.llm_serving")
            else:
                rendered_init_args.append(f"{k}={v}")

        init_line = f"self.{var_name} = {cls.__name__}(" + ", ".join(rendered_init_args) + ")"
        init_lines.append(init_line)

        # Build run call
        run_args: List[str] = []
        if run_has_storage:
            run_args.append("storage=self.storage.step()")
        run_args.extend([f"{k}={v}" for k, v in run_kwargs])

        if run_args:
            call = (
                f"self.{var_name}.run(\n"
                f"    " + ", ".join(run_args) + "\n"
                f")"
            )
        else:
            call = f"self.{var_name}.run()"
        forward_lines.append(call)

    return "\n".join(init_lines), "\n".join(forward_lines)


def indent_block(code: str, spaces: int) -> str:
    """
    Indent every line of 'code' by 'spaces' spaces. Keeps internal structure.
    """
    import textwrap as _tw
    code = _tw.dedent(code or "").strip("\n")
    if not code:
        return ""
    prefix = " " * spaces
    return "\n".join(prefix + line if line else "" for line in code.splitlines())


def write_pipeline_file(
    code: str,
    file_name: str = "recommend_pipeline.py",
    overwrite: bool = True,
) -> Path:
    """
    把生成的 pipeline 代码写入当前文件同级目录下的 `file_name`。
    """
    target_path = Path(__file__).resolve().parent / file_name

    if target_path.exists() and not overwrite:
        raise FileExistsError(f"{target_path} already exists. Set overwrite=True to replace it.")

    target_path.write_text(code, encoding="utf-8")
    log.info(f"[pipeline_assembler] code written to {target_path}")

    return target_path


def build_pipeline_code(
    op_names: List[str],
    *,
    cache_dir: str = "./cache_local",
    llm_local: bool = False,
    local_model_path: str = "",
    chat_api_url: str = "",
    model_name: str = "gpt-4o",
    file_path: str = "",
) -> str:
    # 1) 收集导入与类
    import_lines, stub_blocks, op_classes = group_imports(op_names)
    import_section = "\n".join(import_lines)
    stub_section = "\n\n".join(stub_blocks)  # 用空行隔开多个 stub

    # 2) 渲染 operator 代码片段（无缩进）
    ops_init_block_raw, forward_block_raw = render_operator_blocks(op_names, op_classes)

    # 3) LLM-Serving 片段（无缩进，统一在模板中缩进）
    if llm_local:
        llm_block_raw = f"""
# -------- LLM Serving (Local) --------
self.llm_serving = LocalModelLLMServing_vllm(
    hf_model_name_or_path="{local_model_path}",
    vllm_tensor_parallel_size=1,
    vllm_max_tokens=8192,
    hf_local_dir="local",
    model_name="{model_name}",
)
"""
    else:
        llm_block_raw = f"""
# -------- LLM Serving (Remote) --------
self.llm_serving = APILLMServing_request(
    api_url="{chat_api_url}chat/completions",
    key_name_of_api_key="DF_API_KEY",
    model_name="{model_name}",
    max_workers=100,
)
"""

    # 4) 统一缩进（先缩进，再插入；占位符行保证顶格）
    llm_block = indent_block(llm_block_raw, 8)           # 位于 __init__ 内
    ops_init_block = indent_block(ops_init_block_raw, 8) # 位于 __init__ 内
    forward_block = indent_block(forward_block_raw, 8)   # 位于 forward 内

    # 5) 模板（占位符行顶格，无任何前导空格）
    template = '''"""
Auto-generated by pipeline_assembler
"""
from dataflow.pipeline import PipelineABC
from dataflow.utils.storage import FileStorage
from dataflow.serving import APILLMServing_request, LocalModelLLMServing_vllm

{import_section}

{stub_section}

class RecommendPipeline(PipelineABC):
    def __init__(self):
        super().__init__()
        # -------- FileStorage --------
        self.storage = FileStorage(
            first_entry_file_name="{file_path}",
            cache_path="{cache_dir}",
            file_name_prefix="dataflow_cache_step",
            cache_type="jsonl",
        )
{llm_block}

{ops_init_block}

    def forward(self):
{forward_block}

if __name__ == "__main__":
    pipeline = RecommendPipeline()
    pipeline.compile()
    pipeline.forward()
'''

    # 6) 格式化并返回（不再使用全局 dedent，避免破坏已计算的缩进）
    code = template.format(
        file_path = file_path,
        import_section=import_section,
        stub_section=stub_section,
        cache_dir=cache_dir,
        llm_block=llm_block,
        ops_init_block=ops_init_block,
        forward_block=forward_block,
    )
    return code


def pipeline_assembler(recommendation: List[str], **kwargs) -> Dict[str, Any]:
    code = build_pipeline_code(recommendation, **kwargs)
    return {"pipe_code": code}


async def apipeline_assembler(recommendation: List[str], **kwargs) -> Dict[str, Any]:
    return pipeline_assembler(recommendation, **kwargs)


if __name__ == "__main__":
    test_ops = [
        "SQLGenerator",
        "SQLExecutionFilter",
        "SQLComponentClassifier",
    ]
    result = pipeline_assembler(
        test_ops,
        cache_dir="./cache_local",
        llm_local=False,
        chat_api_url="",
        model_name="gpt-4o",
        file_path = " "
    )
    code_str = result["pipe_code"]
    write_pipeline_file(code_str, file_name="my_recommend_pipeline.py", overwrite=True)
    print("Generated pipeline code written to my_recommend_pipeline.py")