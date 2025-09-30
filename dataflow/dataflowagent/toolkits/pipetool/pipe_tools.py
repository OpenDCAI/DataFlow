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

EXTRA_IMPORTS: set[str] = set()  

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

    for m in sorted(module2names.keys()):
        names = sorted(set(module2names[m]))
        imports.append(f"from {m} import {', '.join(names)}")

    # 追加 choose_prompt_template 过程中收集的额外 import
    imports.extend(sorted(EXTRA_IMPORTS))

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

def choose_prompt_template(op_name: str) -> str:
    """
    返回 prompt_template 的代码字符串。
    规则：
      1. 若类有 ALLOWED_PROMPTS 且非空 → 取第一个并实例化；
      2. 否则回退到 __init__ 默认值；若仍不可用则返回 None。
    """
    from dataflow.utils.registry import OPERATOR_REGISTRY
    import inspect, json

    cls = OPERATOR_REGISTRY.get(op_name)
    if cls is None:
        raise KeyError(f"Operator {op_name} not found in registry")

    # 优先使用 ALLOWED_PROMPTS
    if getattr(cls, "ALLOWED_PROMPTS", None):
        prompt_cls = cls.ALLOWED_PROMPTS[0]
        EXTRA_IMPORTS.add(f"from {prompt_cls.__module__} import {prompt_cls.__qualname__}")
        return f"{prompt_cls.__qualname__}()"

    # -------- 无 ALLOWED_PROMPTS，兜底处理 --------
    sig = inspect.signature(cls.__init__)
    p = sig.parameters.get("prompt_template")
    if p is None:
        # 理论上不会走到这里，因为调用方只在存在该参数时才进来
        return "None"

    default_val = p.default
    if default_val in (inspect._empty, None):
        return "None"

    # 基础类型可直接 repr
    if isinstance(default_val, (str, int, float, bool)):
        return repr(default_val)

    # 类型对象 → 加 import 然后实例化
    if isinstance(default_val, type):
        EXTRA_IMPORTS.add(f"from {default_val.__module__} import {default_val.__qualname__}")
        return f"{default_val.__qualname__}()"

    # UnionType / 其它复杂对象 → 字符串化再 repr，保证可写入代码
    return repr(str(default_val))


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
            elif k == "prompt_template":
                p_t = choose_prompt_template(name)
                rendered_init_args.append(f'{k}={p_t}')
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


# def build_pipeline_code(
#     op_names: List[str],
#     *,
#     cache_dir: str = "./cache_local",
#     llm_local: bool = False,
#     local_model_path: str = "",
#     chat_api_url: str = "",
#     model_name: str = "gpt-4o",
#     file_path: str = "",
# ) -> str:
#     # 1) 收集导入与类
#     import_lines, stub_blocks, op_classes = group_imports(op_names)


#     # 2) 渲染 operator 代码片段（无缩进）
#     ops_init_block_raw, forward_block_raw = render_operator_blocks(op_names, op_classes)

#     import_lines.extend(sorted(EXTRA_IMPORTS))
    
#     import_section = "\n".join(import_lines)
#     stub_section = "\n\n".join(stub_blocks)  # 用空行隔开多个 stub

#     # 3) LLM-Serving 片段（无缩进，统一在模板中缩进）
#     if llm_local:
#         llm_block_raw = f"""
# # -------- LLM Serving (Local) --------
# self.llm_serving = LocalModelLLMServing_vllm(
#     hf_model_name_or_path="{local_model_path}",
#     vllm_tensor_parallel_size=1,
#     vllm_max_tokens=8192,
#     hf_local_dir="local",
#     model_name="{model_name}",
# )
# """
#     else:
#         llm_block_raw = f"""
# # -------- LLM Serving (Remote) --------
# self.llm_serving = APILLMServing_request(
#     api_url="{chat_api_url}chat/completions",
#     key_name_of_api_key="DF_API_KEY",
#     model_name="{model_name}",
#     max_workers=100,
# )
# """

#     # 4) 统一缩进（先缩进，再插入；占位符行保证顶格）
#     llm_block = indent_block(llm_block_raw, 8)           # 位于 __init__ 内
#     ops_init_block = indent_block(ops_init_block_raw, 8) # 位于 __init__ 内
#     forward_block = indent_block(forward_block_raw, 8)   # 位于 forward 内

#     # 5) 模板（占位符行顶格，无任何前导空格）
#     template = '''"""
# Auto-generated by pipeline_assembler
# """
# from dataflow.pipeline import PipelineABC
# from dataflow.utils.storage import FileStorage
# from dataflow.serving import APILLMServing_request, LocalModelLLMServing_vllm

# {import_section}

# {stub_section}

# class RecommendPipeline(PipelineABC):
#     def __init__(self):
#         super().__init__()
#         # -------- FileStorage --------
#         self.storage = FileStorage(
#             first_entry_file_name="{file_path}",
#             cache_path="{cache_dir}",
#             file_name_prefix="dataflow_cache_step",
#             cache_type="jsonl",
#         )
# {llm_block}

# {ops_init_block}

#     def forward(self):
# {forward_block}

# if __name__ == "__main__":
#     pipeline = RecommendPipeline()
#     pipeline.compile()
#     pipeline.forward()
# '''

#     # 6) 格式化并返回（不再使用全局 dedent，避免破坏已计算的缩进）
#     code = template.format(
#         file_path = file_path,
#         import_section=import_section,
#         stub_section=stub_section,
#         cache_dir=cache_dir,
#         llm_block=llm_block,
#         ops_init_block=ops_init_block,
#         forward_block=forward_block,
#     )
#     return code

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
    # 1) 根据 file_path 后缀判断 cache_type
    file_suffix = Path(file_path).suffix.lower() if file_path else ""
    if file_suffix == ".jsonl":
        cache_type = "jsonl"
    elif file_suffix == ".json":
        cache_type = "json"
    elif file_suffix == ".csv":
        cache_type = "csv"  
    else:
        cache_type = "jsonl" 
        log.warning(f"[pipeline_assembler] Unknown file suffix '{file_suffix}', defaulting to 'jsonl'")

    # 2) 收集导入与类
    import_lines, stub_blocks, op_classes = group_imports(op_names)

    # 3) 渲染 operator 代码片段（无缩进）
    ops_init_block_raw, forward_block_raw = render_operator_blocks(op_names, op_classes)

    import_lines.extend(sorted(EXTRA_IMPORTS))
    
    import_section = "\n".join(import_lines)
    stub_section = "\n\n".join(stub_blocks)

    # 4) LLM-Serving 片段（无缩进，统一在模板中缩进）
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

    # 5) 统一缩进
    llm_block = indent_block(llm_block_raw, 8)
    ops_init_block = indent_block(ops_init_block_raw, 8)
    forward_block = indent_block(forward_block_raw, 8)

    # 6) 模板（使用 {cache_type} 占位符）
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
            cache_type="{cache_type}",
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

    # 7) 格式化并返回
    code = template.format(
        file_path=file_path,
        import_section=import_section,
        stub_section=stub_section,
        cache_dir=cache_dir,
        cache_type=cache_type, 
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

# ===================================================================通过my pipline的 py文件，拿到结构化的输出信息
import ast
from pathlib import Path
from typing import Any, Dict, List, Tuple

from dataflow.utils.registry import OPERATOR_REGISTRY 

# --------------------------------------------------------------------
# config
# --------------------------------------------------------------------
SKIP_CLASSES: set[str] = {
    "FileStorage",
    "APILLMServing_request",
    "LocalModelLLMServing_vllm",
}

_IN_PREFIXES = ("input", "input_")
_OUT_PREFIXES = ("output", "output_")

def _is_input(name: str)  -> bool: return name.startswith(_IN_PREFIXES)
def _is_output(name: str) -> bool: return name.startswith(_OUT_PREFIXES)

def _guess_type(cls_obj: "type | None", cls_name: str) -> str:
    # ---------- rule 1 ----------
    if cls_obj is not None:
        parts = cls_obj.__module__.split(".")
        if len(parts) >= 2:
            candidate = parts[-2]
            if candidate not in {"__init__", "__main__"}:
                return candidate

    # ---------- rule 2  (后缀启发) ----------
    lower = cls_name.lower()
    if lower.endswith("parser"):
        return "parser"
    if lower.endswith("generator"):
        return "generate"
    if lower.endswith("filter"):
        return "filter"
    if lower.endswith("evaluator"):
        return "eval"
    if lower.endswith("refiner"):
        return "refine"

    # ---------- rule 3 ----------
    return "other"

# --------------------------------------------------------------------
# safe literal eval
# --------------------------------------------------------------------
def _literal_eval_safe(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant):                
        return node.value
    try:
        import ast as _ast
        return _ast.literal_eval(node)
    except Exception:
        return ast.unparse(node) if hasattr(ast, "unparse") else repr(node)

# --------------------------------------------------------------------
# main
# --------------------------------------------------------------------
def parse_pipeline_file(file_path: str | Path) -> Dict[str, Any]:
    """
    Parameters
    ----------
    file_path : str | Path
        Path to the generated pipeline python file (e.g. mypipeline.py).

    Returns
    -------
    Dict[str, Any]
        {
          "nodes": [...],
          "edges": [...]
        }
    """
    file_path = Path(file_path)
    src = file_path.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(file_path))

    # ----------------------------------------
    # collect operators in __init__            self.x = Xxx(...)
    # ----------------------------------------
    init_ops: Dict[str, Tuple[str, Dict[str, Any]]] = {}  # var -> (cls_name, init_kwargs)
    forward_calls: Dict[str, List[Dict[str, Any]]] = {}   # var -> [run_kwargs]

    class PipelineVisitor(ast.NodeVisitor):
        def visit_ClassDef(self, node: ast.ClassDef):
            for body_item in node.body:
                if isinstance(body_item, ast.FunctionDef):
                    if body_item.name == "__init__":
                        self._parse_init(body_item)
                    elif body_item.name == "forward":
                        self._parse_forward(body_item)

        def _parse_init(self, func: ast.FunctionDef):
            for stmt in func.body:
                if (
                    isinstance(stmt, ast.Assign)
                    and isinstance(stmt.targets[0], ast.Attribute)
                    and isinstance(stmt.value, ast.Call)
                ):
                    attr: ast.Attribute = stmt.targets[0]
                    if not (isinstance(attr.value, ast.Name) and attr.value.id == "self"):
                        continue
                    var_name = attr.attr
                    call: ast.Call = stmt.value
                    if isinstance(call.func, ast.Name):
                        cls_name = call.func.id
                    elif isinstance(call.func, ast.Attribute):
                        cls_name = call.func.attr
                    else:
                        continue

                    # skip unwanted classes
                    if cls_name in SKIP_CLASSES:
                        continue

                    kwargs = {
                        kw.arg: _literal_eval_safe(kw.value)
                        for kw in call.keywords
                        if kw.arg is not None
                    }
                    init_ops[var_name] = (cls_name, kwargs)

        def _parse_forward(self, func: ast.FunctionDef):
            for call in ast.walk(func):
                if (
                    isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Attribute)
                    and call.func.attr == "run"
                ):
                    obj = call.func.value
                    if not (isinstance(obj, ast.Attribute) and isinstance(obj.value, ast.Name) and obj.value.id == "self"):
                        continue
                    var_name = obj.attr
                    kw_dict = {
                        kw.arg: _literal_eval_safe(kw.value)
                        for kw in call.keywords
                        if kw.arg is not None
                    }
                    forward_calls.setdefault(var_name, []).append(kw_dict)

    PipelineVisitor().visit(tree)

    # ----------------------------------------
    # nodes
    # ----------------------------------------
    nodes: List[Dict[str, Any]] = []
    var2node_id: Dict[str, str] = {}
    produced_ports: Dict[str, Tuple[str, str]] = {}

    for idx, (var, (cls_name, init_kwargs)) in enumerate(init_ops.items(), 1):
        node_id = f"node{idx}"
        var2node_id[var] = node_id
        run_cfg = forward_calls.get(var, [{}])[0]  # use first run
        for k, v in run_cfg.items():
            if _is_output(k) and isinstance(v, str):
                produced_ports[v] = (node_id, k)

        cls_obj = None
        try:
            cls_obj = OPERATOR_REGISTRY.get(cls_name)
        except Exception:
            pass
        
        nodes.append({
            "id": node_id,
            "name": cls_name,
            "type": _guess_type(cls_obj, cls_name),
            "config": {
                "init": init_kwargs,
                "run":  run_cfg,
            },
        })

    # ----------------------------------------
    # edges
    # ----------------------------------------
    edges: List[Dict[str, Any]] = []
    for var, runs in forward_calls.items():
        if var not in var2node_id:           # run() belongs to skipped class
            continue
        tgt_id = var2node_id[var]
        for run_cfg in runs:
            for k, v in run_cfg.items():
                if _is_input(k) and isinstance(v, str) and v in produced_ports:
                    src_id, src_port = produced_ports[v]
                    edges.append({
                        "source": src_id,
                        "target": tgt_id,
                        "source_port": src_port,
                        "target_port": k,
                    })

    return {"nodes": nodes, "edges": edges}


























if __name__ == "__main__":
    # test_ops = [
    #     "SQLGenerator",
    #     "SQLExecutionFilter",
    #     "SQLComponentClassifier",
    # ]
    # result = pipeline_assembler(
    #     test_ops,
    #     cache_dir="./cache_local",
    #     llm_local=False,
    #     chat_api_url="",
    #     model_name="gpt-4o",
    #     file_path = " "
    # )
    # code_str = result["pipe_code"]
    # write_pipeline_file(code_str, file_name="my_recommend_pipeline.py", overwrite=True)
    # print("Generated pipeline code written to my_recommend_pipeline.py")
    graph = parse_pipeline_file("/mnt/DataFlow/lz/proj/DataFlow/dataflow/dataflowagent/tests/my_pipeline.py")
    import json, pprint
    pprint.pprint(graph, width=120)
    # 或者保存
    Path("pipeline_graph.json").write_text(json.dumps(graph, indent=2), "utf-8")