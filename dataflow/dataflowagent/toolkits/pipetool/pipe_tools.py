# dataflow/dataflowagent/toolkits/pipeline_assembler.py
from __future__ import annotations

import ast
import json
import itertools
from pathlib import Path
from typing import Any, Dict, List, Tuple, DefaultDict
import requests
from dataflow.dataflowagent.state import DFState,DFRequest
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

def call_llm_for_selection(
    system_prompt: str,
    user_message: str,
    api_url: str,
    api_key: str,
    model: str,
    temperature: float = 0.3,
    max_tokens: int = 100
) -> str:
    """
    调用 LLM API 进行选择决策
    
    Args:
        system_prompt: 系统提示词
        user_message: 用户消息
        api_url: API 地址（OpenAI 兼容格式）
        api_key: API 密钥
        model: 模型名称
        temperature: 温度参数
        max_tokens: 最大 token 数
    
    Returns:
        LLM 返回的文本内容
    """
    if not api_url.endswith('/chat/completions'):
        if api_url.endswith('/'):
            api_url = api_url + 'chat/completions'
        else:
            api_url = api_url + '/chat/completions'
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    
    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()
        
        # 提取返回的内容
        content = result.get('choices', [{}])[0].get('message', {}).get('content', '').strip()
        log.info(f"[pipeline_assembler] LLM selection result: {content}")
        return content
        
    except Exception as e:
        log.error(f"[pipeline_assembler] LLM API call failed: {e}")
        raise


def extract_prompt_info(prompt_cls: type) -> Dict[str, Any]:
    """
    提取 prompt 类的详细信息，包括示例提示词
    
    Args:
        prompt_cls: Prompt 类对象
    
    Returns:
        包含类名、模块、文档字符串和示例提示词的字典
    """
    prompt_info = {
        'class_name': prompt_cls.__qualname__,
        'module': prompt_cls.__module__,
        'docstring': (prompt_cls.__doc__ or '').strip(),
    }
    
    # 尝试实例化并获取示例提示词
    try:
        instance = prompt_cls()
        
        # 如果有 build_prompt 方法
        if hasattr(instance, 'build_prompt'):
            sig = inspect.signature(instance.build_prompt)
            params = list(sig.parameters.keys())
            
            # 构造示例参数
            example_args = {}
            for param in params:
                if param == 'self':
                    continue
                # 使用占位符
                example_args[param] = f"<example_{param}>"
            
            try:
                # 调用 build_prompt 获取完整的提示词模板
                example_prompt = instance.build_prompt(**example_args)
                # 截取前 800 字符避免过长
                prompt_info['full_prompt_template'] = example_prompt[:800]
                if len(example_prompt) > 800:
                    prompt_info['full_prompt_template'] += "\n...[truncated]"
            except Exception as e:
                log.warning(f"[pipeline_assembler] Failed to get example prompt for {prompt_cls.__name__}: {e}")
                prompt_info['full_prompt_template'] = "Unable to generate example"
        
        # 如果有其他可用的属性，也可以提取
        if hasattr(instance, 'template'):
            prompt_info['template_attr'] = str(instance.template)[:200]
            
    except Exception as e:
        log.warning(f"[pipeline_assembler] Failed to instantiate {prompt_cls.__name__}: {e}")
        prompt_info['full_prompt_template'] = "Unable to instantiate"
    
    return prompt_info


def choose_prompt_template_by_llm(op_name: str, state: DFState) -> str:
    """
    通过 LLM 选择最合适的 prompt_template
    
    规则：
      1. 提取 operator 的所有 ALLOWED_PROMPTS 候选
      2. 获取每个 prompt 的详细信息（包括提示词模板）
      3. 调用 LLM 让它根据 target 任务描述选择最合适的 prompt
      4. 返回选中 prompt 的实例化代码字符串
    
    Args:
        op_name: Operator 名称
        state: DFState 对象，包含 request.target 等信息
    
    Returns:
        选中的 prompt_template 实例化代码字符串
    """
    cls = OPERATOR_REGISTRY.get(op_name)
    if cls is None:
        raise KeyError(f"Operator {op_name} not found in registry")
    
    # 如果没有 ALLOWED_PROMPTS 或为空，回退到原逻辑
    allowed_prompts = getattr(cls, "ALLOWED_PROMPTS", None)
    if not allowed_prompts:
        log.info(f"[pipeline_assembler] No ALLOWED_PROMPTS for {op_name}, using default logic")
        return choose_prompt_template(op_name, state)
    
    # 如果只有一个候选，直接使用
    if len(allowed_prompts) == 1:
        prompt_cls = allowed_prompts[0]
        EXTRA_IMPORTS.add(f"from {prompt_cls.__module__} import {prompt_cls.__qualname__}")
        return f"{prompt_cls.__qualname__}()"
    
    # 收集所有候选 prompt 的详细信息
    log.info(f"[pipeline_assembler] Extracting info from {len(allowed_prompts)} prompt candidates")
    prompt_candidates = []
    for prompt_cls in allowed_prompts:
        prompt_info = extract_prompt_info(prompt_cls)
        prompt_candidates.append(prompt_info)
    
    # 构造 LLM 请求
    target = state.request.target
    system_prompt = """You are an expert at selecting the most appropriate prompt template for a given task.

Your job is to:
1. Analyze the target task description
2. Review all available prompt templates (including their documentation and example prompts)
3. Select the MOST suitable prompt template

IMPORTANT: Respond with ONLY the exact class name of the selected prompt template, nothing else."""
    
    user_message = f"""Target Task Description:
{target}

Available Prompt Templates:
"""
    
    for i, p in enumerate(prompt_candidates, 1):
        user_message += f"\n{'='*60}\n"
        user_message += f"Option {i}: {p['class_name']}\n"
        user_message += f"{'='*60}\n"
        
        if p['docstring']:
            user_message += f"Documentation:\n{p['docstring']}\n\n"
        
        if 'full_prompt_template' in p:
            user_message += f"Prompt Template Example:\n{p['full_prompt_template']}\n"
        
        if 'template_attr' in p:
            user_message += f"Template: {p['template_attr']}\n"
    
    user_message += f"\n{'='*60}\n"
    user_message += "\nBased on the target task, which prompt template is most suitable?\n"
    user_message += "Respond with ONLY the class name (e.g., 'MathAnswerGeneratorPrompt')."
    
    # 调用 LLM
    try:
        selected_class_name = call_llm_for_selection(
            system_prompt=system_prompt,
            user_message=user_message,
            api_url=state.request.chat_api_url,
            api_key=state.request.api_key,
            model=state.request.model
        )
        
        # 清理返回结果（移除可能的引号、空格等）
        selected_class_name = selected_class_name.strip().strip('"\'`')
        
        # 找到对应的 prompt class
        for prompt_cls in allowed_prompts:
            if prompt_cls.__qualname__ == selected_class_name or prompt_cls.__name__ == selected_class_name:
                log.info(f"[pipeline_assembler] LLM selected prompt: {prompt_cls.__qualname__}")
                EXTRA_IMPORTS.add(f"from {prompt_cls.__module__} import {prompt_cls.__qualname__}")
                return f"{prompt_cls.__qualname__}()"
        
        # 如果没找到精确匹配，尝试模糊匹配
        for prompt_cls in allowed_prompts:
            if selected_class_name in prompt_cls.__qualname__ or prompt_cls.__name__ in selected_class_name:
                log.warning(f"[pipeline_assembler] Using fuzzy match for '{selected_class_name}' -> {prompt_cls.__qualname__}")
                EXTRA_IMPORTS.add(f"from {prompt_cls.__module__} import {prompt_cls.__qualname__}")
                return f"{prompt_cls.__qualname__}()"
        
        # 如果还是没找到，使用第一个作为默认
        log.warning(f"[pipeline_assembler] LLM selected unknown prompt '{selected_class_name}', using first available")
        
    except Exception as e:
        log.error(f"[pipeline_assembler] LLM selection failed: {e}, using first available prompt")
    
    # 默认使用第一个
    prompt_cls = allowed_prompts[0]
    EXTRA_IMPORTS.add(f"from {prompt_cls.__module__} import {prompt_cls.__qualname__}")
    return f"{prompt_cls.__qualname__}()"


# ==================================================================================================================================
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

def choose_prompt_template(op_name: str, state: DFState) -> str:
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


def render_operator_blocks(op_names: List[str], op_classes: Dict[str, type], state :DFState) -> Tuple[str, str]:
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
                # p_t = choose_prompt_template(name, state)
                # 用LLM来选择
                p_t = choose_prompt_template_by_llm(name, state)
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



def build_pipeline_code(
    op_names: List[str],
    state: DFState,
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
    ops_init_block_raw, forward_block_raw = render_operator_blocks(op_names, op_classes, state)

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


def pipeline_assembler(recommendation: List[str], state: DFState,**kwargs) -> Dict[str, Any]:
    code = build_pipeline_code(recommendation, state, **kwargs)
    return {"pipe_code": code}


async def apipeline_assembler(recommendation: List[str], **kwargs) -> Dict[str, Any]:
    return pipeline_assembler(recommendation, **kwargs)

# ===================================================================通过my pipline的 py文件，拿到结构化的输出信息
"""
Parse a generated PipelineABC python file and export a graph schema::

    {
      "nodes": [...],
      "edges": [...]
    }

Requirements:
    - 支持 input_key / output_key 既可以是关键字参数也可以是位置参数
    - 允许同一个算子 run 多次
    - nodes.id 直接使用 self.xxx 的变量名
"""
from collections import defaultdict
from dataflow.utils.registry import OPERATOR_REGISTRY

# ----------------------------------------------------- #
# config & helpers
# ----------------------------------------------------- #
SKIP_CLASSES: set[str] = {
    "FileStorage",
    "APILLMServing_request",
    "LocalModelLLMServing_vllm",
}

_IN_PREFIXES = ("input", "input_")
_OUT_PREFIXES = ("output", "output_")


def _is_input(name: str) -> bool:
    return name.startswith(_IN_PREFIXES)


def _is_output(name: str) -> bool:
    return name.startswith(_OUT_PREFIXES)


def _guess_type(cls_obj: type | None, cls_name: str) -> str:
    """
    Guess operator category for front-end icon & color.
    规则:
        1. package 名倒数第二段 (operators.xxx.{filter|parser}.xxx)
        2. 类名后缀启发
        3. 兜底 'other'
    """
    # rule-1
    if cls_obj is not None:
        parts = cls_obj.__module__.split(".")
        if len(parts) >= 2:
            candidate = parts[-2]
            if candidate not in {"__init__", "__main__"}:
                return candidate
    # rule-2
    lower = cls_name.lower()
    for suf, cat in [
        ("parser", "parser"),
        ("generator", "generate"),
        ("filter", "filter"),
        ("evaluator", "eval"),
        ("refiner", "refine"),
    ]:
        if lower.endswith(suf):
            return cat
    # rule-3
    return "other"


def _literal_eval_safe(node: ast.AST) -> Any:
    """ast.literal_eval 的宽松版本，失败就返回反编译字符串"""
    if isinstance(node, ast.Constant):  # fast path
        return node.value
    try:
        return ast.literal_eval(node)
    except Exception:
        return ast.unparse(node) if hasattr(ast, "unparse") else repr(node)


# ----------------------------------------------------- #
# AST 解析主流程
# ----------------------------------------------------- #
def parse_pipeline_file(file_path: str | Path) -> Dict[str, Any]:
    """
    Parameters
    ----------
    file_path : str | Path
        生成的 pipeline python 文件路径

    Returns
    -------
    dict
        {"nodes": [...], "edges": [...]}
    """
    file_path = Path(file_path)
    src = file_path.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(file_path))

    # ------------------------------------------------- #
    # 1. 解析 __init__ 里的 operator 实例
    # ------------------------------------------------- #
    def _parse_init(init_func: ast.FunctionDef) -> Dict[str, Tuple[str, Dict[str, Any]]]:
        """
        Returns
        -------
        var_name -> (cls_name, init_kwargs)
        """
        results: Dict[str, Tuple[str, Dict[str, Any]]] = {}
        for stmt in init_func.body:
            if (
                isinstance(stmt, ast.Assign)
                and stmt.targets
                and isinstance(stmt.targets[0], ast.Attribute)
                and isinstance(stmt.value, ast.Call)
            ):
                attr: ast.Attribute = stmt.targets[0]
                if not (isinstance(attr.value, ast.Name) and attr.value.id == "self"):
                    continue
                var_name = attr.attr

                call: ast.Call = stmt.value
                # 取类名
                if isinstance(call.func, ast.Name):
                    cls_name = call.func.id
                elif isinstance(call.func, ast.Attribute):
                    cls_name = call.func.attr
                else:
                    continue

                if cls_name in SKIP_CLASSES:  # 跳过非算子
                    continue

                kwargs = {
                    kw.arg: _literal_eval_safe(kw.value)
                    for kw in call.keywords
                    if kw.arg is not None
                }
                results[var_name] = (cls_name, kwargs)
        return results

    # ------------------------------------------------- #
    # 2. 解析 forward() 里的 run 调用
    # ------------------------------------------------- #
    def _parse_forward(
        forward_func: ast.FunctionDef,
    ) -> DefaultDict[str, List[Dict[str, Any]]]:
        """
        Returns
        -------
        var_name -> [run_kwargs ...]  (保持出现顺序)
        """
        mapping: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)

        # walk 按源码顺序遍历需借助 ast.iter_child_nodes + 递归
        def _visit(node: ast.AST):
            # 按出现顺序遍历
            for child in ast.iter_child_nodes(node):
                if (
                    isinstance(child, ast.Call)
                    and isinstance(child.func, ast.Attribute)
                    and child.func.attr == "run"
                ):
                    obj = child.func.value
                    if (
                        isinstance(obj, ast.Attribute)
                        and isinstance(obj.value, ast.Name)
                        and obj.value.id == "self"
                    ):
                        var_name = obj.attr

                        # ------- 关键字参数 -------
                        kw_dict = {
                            kw.arg: _literal_eval_safe(kw.value)
                            for kw in child.keywords
                            if kw.arg is not None
                        }

                        # ------- 位置参数 -------
                        # 假设位置顺序为 (storage, input_key, output_key, ...)
                        if len(child.args) >= 2:
                            kw_dict.setdefault("input_key", _literal_eval_safe(child.args[1]))
                        if len(child.args) >= 3:
                            kw_dict.setdefault("output_key", _literal_eval_safe(child.args[2]))

                        mapping[var_name].append(kw_dict)
                _visit(child)

        _visit(forward_func)
        return mapping

    # ------------------------------------------------- #
    # 3. 主 visitor：定位唯一继承 PipelineABC 的类
    # ------------------------------------------------- #
    init_ops, forward_calls = {}, defaultdict(list)

    class PipelineVisitor(ast.NodeVisitor):
        def visit_ClassDef(self, node: ast.ClassDef):  # noqa: N802
            nonlocal init_ops, forward_calls
            # naive 判断: 存在 forward() 方法即认为是 pipeline
            has_forward = any(
                isinstance(b, ast.FunctionDef) and b.name == "forward" for b in node.body
            )
            if not has_forward:
                return
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    if item.name == "__init__":
                        init_ops = _parse_init(item)
                    elif item.name == "forward":
                        forward_calls = _parse_forward(item)

    PipelineVisitor().visit(tree)

    # ------------------------------------------------- #
    # 4. build nodes
    # ------------------------------------------------- #
    def build_nodes() -> tuple[list[dict[str, Any]],
                            dict[str, str],
                            dict[str, tuple[str, str]]]:
        """
        Returns
        -------
        nodes                : list of node-dict
        var2id               : var_name -> node_id        (供后续查表)
        produced_ports       : label(str) -> (node_id, port_name)
        """
        nodes: list[dict[str, Any]] = []
        var2id: dict[str, str] = {}
        produced_ports: dict[str, tuple[str, str]] = {}

        global_counter = itertools.count(1)       

        for var, (cls_name, init_kwargs) in init_ops.items():
            # -------- 生成 node_id -------- #
            node_id = f"node{next(global_counter)}"    # <-- 变成 node1/node2/…

            var2id[var] = node_id

            # forward() 第一次 run 的配置
            first_run_cfg = forward_calls.get(var, [{}])[0]

            # 把首次 run 产生的 output 标记为 “已经产生”
            for k, v in first_run_cfg.items():
                if _is_output(k) and isinstance(v, str):
                    produced_ports[v] = (node_id, k)
            try:
                cls_obj = OPERATOR_REGISTRY.get(cls_name)
            except Exception:
                cls_obj = None

            nodes.append(
                {
                    "id": node_id,
                    "name": cls_name,
                    "type": _guess_type(cls_obj, cls_name),
                    "config": {
                        "init": init_kwargs,
                        "run": first_run_cfg,
                    },
                }
            )
        return nodes, var2id, produced_ports

    # ------------------------------------------------- #
    # 5. build edges (按 forward 执行顺序)
    # ------------------------------------------------- #
    def build_edges(
        produced_ports: dict[str, tuple[str, str]],
        var2id: dict[str, str],
    ) -> list[dict[str, Any]]:
        edges: list[dict[str, Any]] = []
        for var, runs in forward_calls.items():
            tgt_id = var2id.get(var)
            if not tgt_id:
                continue
            for run_cfg in runs:
                for k, v in run_cfg.items():
                    if _is_input(k) and isinstance(v, str) and v in produced_ports:
                        src_id, src_port = produced_ports[v]
                        edges.append(
                            {
                                "source": src_id,
                                "target": tgt_id,
                                "source_port": src_port,
                                "target_port": k,
                            }
                        )
        return edges

    nodes, var2id, produced_ports = build_nodes()
    edges = build_edges(produced_ports, var2id)
    return {"nodes": nodes, "edges": edges}


# ----------------------------------------------------- #
# CLI 方便快速测试（免参数版）
# ----------------------------------------------------- #
if __name__ == "__main__":
    import json
    from pathlib import Path
    import pprint

    PY_PATH = Path("/mnt/DataFlow/lz/proj/DataFlow/dataflow/dataflowagent/tests/my_pipeline.py")

    graph = parse_pipeline_file(PY_PATH)

    pprint.pprint(graph, width=120)

    OUT_PATH = PY_PATH.with_suffix(".json")
    OUT_PATH.write_text(json.dumps(graph, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved to {OUT_PATH}")





















# if __name__ == "__main__":
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
