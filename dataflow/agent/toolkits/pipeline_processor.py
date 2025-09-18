#!/usr/bin/env python3
"""
pipeline_processor.py  ── Operator parameter extraction and pipeline code generator
Author  : [Zhou Liu]
License : MIT
Created : 2024-07-02

This module provides utility functions and classes to:

* introspect operator classes and extract their parameter signatures
* patch operator configuration JSON files with up-to-date parameter info
* generate executable Python pipeline scripts based on recommended graphs
* perform topological sorting of pipeline nodes

The code assumes that operator class registration is handled via OPERATOR_REGISTRY.
It is designed for use in dynamic dataflow and AI workflow systems.

Thread-safety: The module is stateless and thread-safe unless you modify global registry or perform concurrent writes to output files.
"""
from dataflow.utils.registry import OPERATOR_REGISTRY
OPERATOR_REGISTRY._get_all()
import inspect
import json
from typing import List, Dict, Any, Type, Iterable, Tuple,Optional
from dataflow.utils.registry import OPERATOR_REGISTRY
from pathlib import Path
import os
from .tool_factory import TOOL
from dataflow import get_logger
logger = get_logger()

operators_info_path = f"{os.path.dirname(os.path.abspath(__file__))}/resources/Operator_patched.json"

def _parse_params(sig: inspect.Signature) -> List[Dict[str, Any]]:
    """Split a signature object into a list of argument dicts, ignoring self."""
    return [
        {
            "name": p.name,
            "default": None if p.default is inspect.Parameter.empty else p.default,
            "kind": str(p.kind),  # POSITIONAL_ONLY / VAR_POSITIONAL ...
        }
        for p in sig.parameters.values()
        if p.name != "self"
    ]
def get_class_method_params(
    cls: Type,
    method_name: str = "run",
) -> Dict[str, List[Dict[str, Any]]]:
    params: Dict[str, List[Dict[str, Any]] | None] = {}

    params["init"] = _parse_params(inspect.signature(cls.__init__))

    if hasattr(cls, method_name):
        params[method_name] = _parse_params(
            inspect.signature(getattr(cls, method_name))
        )
    else:
        params[method_name] = None

    return params


def collect_operator_params(
    method_name: str = "run",
) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    """
    Traverse OPERATOR_REGISTRY and collect parameter information for all operators.

    :param operator_registry: An iterable like [(name, cls), ...]
    :param method_name:      The business method name to extract, default 'run'
    :return: {
               "OperatorName": {
                   "init": [...],
                   "run":  [...]
               },
               ...
             }
    """
    summary: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for name, cls in OPERATOR_REGISTRY:
        summary[name] = get_class_method_params(cls, method_name=method_name)
    return summary

def patch_operator_json(
    json_path: str | Path,
    *,
    method_name: str = "run",
    save_as: Optional[str | Path] = None
) -> Dict[str, Any]:
    """
    1. Read Operator.json
    2. Replace each operator's `"command"` field with the parameter information
       collected from collect_operator_params()
    3. If **llm_serving** exists in **__init__**, set the `"mode"` field to "GPU";
       for other operators, fill with an empty string if the field does not exist.

    Parameters
    ----------
    json_path : str | Path
        Path to the original Operator.json
    method_name : str
        Business method name to parse (default "run")
    save_as : str | Path | None
        If specified, write the result to disk; otherwise, return the dict in memory

    Returns
    -------
    Dict[str, Any]
        The modified complete Operator description
    """
    # ---------- 1. Collect all operator parameter signatures ---------- #
    param_summary = collect_operator_params(method_name=method_name)

    # ---------- 2. Read JSON ---------- #
    json_path = Path(json_path)
    data: Dict[str, Any] = json.loads(json_path.read_text(encoding="utf-8"))

    # ---------- 3. Traverse pipeline / operator ---------- #
    for pipeline_name, operators in data.items():
        if not isinstance(operators, list):
            continue

        for op in operators:
            op_name = op.get("name")
            if op_name not in param_summary:
                # Operator not registered, skip
                op.setdefault("mode", "")    # Also fill mode with an empty string
                continue

            # 3-a) Replace command
            op["command"] = param_summary[op_name]

            # 3-b) Check for llm_serving
            has_llm_serving = any(
                p["name"] == "llm_serving"
                for p in param_summary[op_name]["init"]
            )
            if has_llm_serving:
                op["mode"] = "GPU"           # Can be changed to "CPU" / "API" later ...
            else:
                op.setdefault("mode", "")    # Ensure the field exists

    # ---------- 4. Write back or return ---------- #
    if save_as is not None:
        save_path = Path(save_as)
        save_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

    return data

def _call_get_desc_static(cls, lang: str) -> str | None:
    """
    Only handles classes where get_desc is explicitly declared as @staticmethod.
    - If parameters are (lang,)        →  get_desc(lang)
    - If parameters are (self, lang)   →  get_desc(None, lang)  (for compatibility with incorrect signatures)
    Returns None for other signatures (skips the class).
    """
    func_obj = cls.__dict__.get("get_desc")
    if not isinstance(func_obj, staticmethod):
        return None

    fn = func_obj.__func__
    params = list(inspect.signature(fn).parameters)

    if params == ["lang"]:
        return fn(lang)
    if params == ["self", "lang"]:
        return fn(None, lang)

    return None


def get_operator_descriptions(lang: str = "zh") -> List[Dict[str, Any]]:
    """
    Iterate over OPERATOR_REGISTRY and return the descriptions of all qualified operators:
        [
          {"node": 1, "name": "QuestionGenerator", "description": "..."},
          ...
        ]
    Only classes with a `get_desc` declared as `@staticmethod` and that can be called successfully are included.
    """
    desc_list: List[Dict[str, Any]] = []
    idx = 1

    for op_name, cls in OPERATOR_REGISTRY:
        desc = _call_get_desc_static(cls, lang)
        if desc is None:
            continue

        desc_list.append(
            {
                "node": idx,
                "name": op_name,     # or cls.__name__, adjust as needed
                "description": desc,
                "command": "",
                "required": "",
                "depends_on": [],
                "mode": ""
            }
        )
        idx += 1

    return desc_list


def update_operator_descriptions(
    json_file_path= operators_info_path,
    key="MIXTURE"
):
    ops = get_operator_descriptions()
    for item in ops:
        desc = item.get("description")
        if isinstance(desc, BaseException):
            item["description"] = str(desc)
    json_path = Path(json_file_path)
    data = json.loads(json_path.read_text(encoding="utf-8"))
    data[key] = ops
    json_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    logger.info(f"[Scanning of the dataflow package completed]: {len(ops)} records have been updated to '{key}'.")

import inspect
import json
import os
import textwrap
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, List, Tuple

from dataflow.utils.registry import OPERATOR_REGISTRY
def _topological_sort(nodes: List[Dict[str, Any]],
                      edges: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    id2node = {n["id"]: n for n in nodes}
    indeg = defaultdict(int)
    graph = defaultdict(list)
    # logger.info(f"[nodes]:{nodes}")
    # logger.info(f"[edges]:{edges}")
    for e in edges:
        src, dst = e["source"], e["target"]
        graph[src].append(dst)
        indeg[dst] += 1
        indeg[src] += 0

    for n in id2node:
        indeg[n] += 0

    q = deque([nid for nid, d in indeg.items() if d == 0])
    order = []
    while q:
        nid = q.popleft()
        order.append(nid)
        for nxt in graph[nid]:
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                q.append(nxt)

    if len(order) != len(id2node):
        raise ValueError("Graph has a cycle")
    return [id2node[n] for n in order]

_NAME2CLS = {name: cls for name, cls in OPERATOR_REGISTRY}

def _snake(name: str) -> str:
    """Simple lower-case variable name"""
    return name.lower()

def generate_pipeline_py(
    pre_task_result: Dict[str, Any],
    py_path: str | Path = "recommend_pipeline.py",
    json_file: str = "",
    *,
    local: bool = False,
    local_model_name_or_path: str= "",
    chat_api_url: str = ""
) -> str:
    """Generate an executable Python pipeline script."""

    # ---------------- 拓扑排序、收集 import ----------------
    nodes, edges = pre_task_result["nodes"], pre_task_result.get("edges", [])
    ordered = _topological_sort(nodes, edges)

    module2names: Dict[str, List[str]] = defaultdict(list)
    for node in ordered:
        cls = _NAME2CLS.get(node["name"])
        if cls is None:
            raise KeyError(f"Operator {node['name']} not registered in OPERATOR_REGISTRY")
        module2names[cls.__module__].append(cls.__name__)

    import_lines = [
        f"from {module} import {', '.join(sorted(set(names)))}"
        for module, names in module2names.items()
    ]
    extra_imports = [
        "from dataflow.utils.storage import FileStorage",
        "from dataflow.serving import APILLMServing_request, LocalModelLLMServing_vllm, LocalModelLLMServing_sglang",
    ]

    def _py_literal(val):
        if isinstance(val, str):
            return f'"{val}"'
        if val is None:
            return '""'
        return repr(val)

    # ---------------- 生成 __init__ 中的算子实例 -------------
    init_operator_lines = []
    for node in ordered:
        cls_name = node["name"]
        var_name = _snake(cls_name)
        parts = []
        for p in node["command"]["init"]:
            pname, pdef = p["name"], p["default"]

            if pname == "llm_serving":
                parts.append("llm_serving=llm_serving")
            elif pname == "prompt_template":
                parts.append("prompt_template=None")
            else:
                parts.append(f"{pname}={_py_literal(pdef)}")
        init_operator_lines.append(f"        self.{var_name} = {cls_name}({', '.join(parts)})")

    # ---------------- 生成 forward() 调用 --------------------
    forward_lines = []
    for node in ordered:
        var_name = _snake(node["name"])
        run_args = ["storage=self.storage.step()"]
        for p in node["command"]["run"]:
            if p["name"] != "storage":
                run_args.append(f"{p['name']}={_py_literal(p['default'])}")
        forward_lines.append(
            f"        self.{var_name}.run(\n            {', '.join(run_args)}\n        )"
        )

    # ---------------- LLM-Serving 代码块 ---------------------
    if local:
        llm_block = f"""
        # -------- LLM Serving (Local) --------
        llm_serving = LocalModelLLMServing_vllm(
            hf_model_name_or_path="{local_model_name_or_path}",
            vllm_tensor_parallel_size=1,
            vllm_max_tokens=8192,
            hf_local_dir="local",
        )
        """
    else:
        llm_block = f"""
        # -------- LLM Serving (Remote) --------
        llm_serving = APILLMServing_request(
            api_url="{chat_api_url}",
            key_name_of_api_key='DF_API_KEY',
            model_name="gpt-4o",
            max_workers=100,
        )
        # For local models, uncomment below
        # llm_serving = LocalModelLLMServing_vllm(
        #     hf_model_name_or_path="{local_model_name_or_path}",
        #     vllm_tensor_parallel_size=1,
        #     vllm_max_tokens=8192,
        #     hf_local_dir="local",
        # )
        """

    code = "\n".join(
        [
            "from dataflow.pipeline import PipelineABC",
            "import pytest",
            *import_lines,
            *extra_imports,
            "",
            "",
            "class RecommendPipeline(PipelineABC):",
            "    def __init__(self):",
            "        super().__init__()",
            textwrap.indent(
                textwrap.dedent(
                    f"""
                    # -------- FileStorage (请根据需要修改参数) --------
                    self.storage = FileStorage(
                        first_entry_file_name="{json_file}",
                        cache_path="./cache_local",
                        file_name_prefix="dataflow_cache_step",
                        cache_type="jsonl",
                    )
                    """
                ),
                " " * 8,
            ),
            textwrap.indent(textwrap.dedent(llm_block), " " * 8),
            *init_operator_lines,
            "",
            "    def forward(self):",
            *forward_lines,
            "",
            "",
            'if __name__ == "__main__":',
            "    pipeline = RecommendPipeline()",
            "    pipeline.compile()",
            "    pipeline.forward()",
            "",
        ]
    )

    py_path = Path(py_path)
    logger.debug("[成功写入]")
    py_path.write_text(code, encoding="utf-8")
    return code

@TOOL(desc="生成pipeline代码并且执行")
def local_tool_for_execute_the_recommended_pipeline(
    pre_task_result: Dict[str, Any],
    request,
    *,
    # py_path: str = "recommend_pipeline.py",
    dry_run: bool = False,
    is_in_debug_process: bool = False,
    current_round: int = 0
):
    """
    Generate and (optionally) execute the recommended pipeline.
    When is_in_debug_process=True and py_path already exists,
    directly read the existing file to avoid overwriting patched code during debugging.
    """
    py_file = Path(request.py_path)
    if is_in_debug_process and py_file.exists():
        code = py_file.read_text(encoding="utf-8")
        if 'if __name__ == "__main__":' not in code:
            main_block = textwrap.dedent(
                """
                if __name__ == "__main__":
                    pipeline = RecommendPipeline()
                    pipeline.compile()
                    pipeline.forward()
                """
            )
            with py_file.open("a", encoding="utf-8") as f:
                f.write(main_block)
            code += main_block
            logger.info(
                "Main entrance not found. Appended the standard "
                '"if __name__ == \'__main__\':" block to the existing file.'
            )
        logger.info(f"[Reusing existing pipeline file] {request.py_path}")
    else:
        code = generate_pipeline_py(
            pre_task_result,
            py_path=request.py_path,
            json_file=request.json_file,
            local=request.use_local_model,
            local_model_name_or_path=request.local_model_name_or_path,
            chat_api_url=request.chat_api_url   
        )
    logger.info(f"[Agent generated Pipeline Code]: {code}")
    if request.execute_the_pipeline and not dry_run:
        import subprocess, sys
        logger.info("\n[............Pipeline is running............]\n")
        run_res = subprocess.run(
            [sys.executable, "-u", str(py_file)],  
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,              
            text=True,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        if run_res.returncode != 0:
            raise RuntimeError(
                f"{request.py_path} exited with {run_res.returncode}\n"
                f"stdout:\n{run_res.stdout}\n"
                f"stderr:\n{run_res.stderr}"
            )
    if dry_run:
        return code
    logger.info(f"\n[............Pipeline Code Has Been Saved in {request.py_path}............]\n")
    return code

def local_tool_for_update_operators_info():
    # 只更新mixture
    update_operator_descriptions()
    
    # 更新init和run的参数
    patch_operator_json(
      json_path = operators_info_path,
      save_as   = operators_info_path
    ) 

if __name__ == "__main__":
    local_tool_for_update_operators_info()
    # result = collect_operator_params()
    # import pprint
    # pprint.pprint(result)
    # patch_operator_json(
    #     "/mnt/h_h_public/lh/lz/DataFlow/dataflow/agent/toolkits/resources/Operator.json",
    #     save_as="/mnt/h_h_public/lh/lz/DataFlow/dataflow/agent/toolkits/resources/Operator_patched.json"
    # )
    # pre_result = {'edges': [{'source': 'node0', 'target': 'node1'}, {'source': 'node1', 'target': 'node2'}, {'source': 'node2', 'target': 'node3'}, {'source': 'node3', 'target': 'node4'}, {'source': 'node4', 'target': 'node5'}, {'source': 'node5', 'target': 'node6'}, {'source': 'node6', 'target': 'node7'}, {'source': 'node7', 'target': 'node8'}], 'reason': "The pipeline is designed to process mathematics and science content with a focus on prediction without deduplication. Starting with QuestionGenerator to create new questions, followed by QuestionFilter to ensure correctness. Then, ReasoningDifficultyDatasetEvaluator and ReasoningCategoryDatasetEvaluator are used to categorize and label the questions. AnswerPipelineRoot separates data with and without answers. PseudoAnswerGenerator generates candidate answers, and AnswerFormatterFilter ensures the answers meet format requirements. Finally, AnswerTokenLengthFilter checks for appropriate answer lengths. This sequence ensures comprehensive processing while adhering to the user's requirements.", 'outputs': [], 'nodes': [{'name': 'QuestionGenerator', 'type': 'generator', 'description': '基于现有的问题数据，每个问题合成1-5个新问题', 'command': {'init': [{'name': 'num_prompts', 'default': 1, 'kind': 'POSITIONAL_OR_KEYWORD'}, {'name': 'llm_serving', 'default': None, 'kind': 'POSITIONAL_OR_KEYWORD'}], 'run': [{'name': 'storage', 'default': None, 'kind': 'POSITIONAL_OR_KEYWORD'}, {'name': 'input_key', 'default': None, 'kind': 'POSITIONAL_OR_KEYWORD'}]}, 'required': True, 'depends_on': [], 'mode': 'GPU', 'id': 'node1'}, {'name': 'QuestionFilter', 'type': 'filter', 'description': '检查每个问题的正确性', 'command': {'init': [{'name': 'system_prompt', 'default': 'You are a helpful assistant.', 'kind': 'POSITIONAL_OR_KEYWORD'}, {'name': 'llm_serving', 'default': None, 'kind': 'POSITIONAL_OR_KEYWORD'}], 'run': [{'name': 'storage', 'default': None, 'kind': 'POSITIONAL_OR_KEYWORD'}, {'name': 'input_key', 'default': 'math_problem', 'kind': 'POSITIONAL_OR_KEYWORD'}]}, 'required': True, 'depends_on': [], 'mode': 'GPU', 'id': 'node2'}, {'name': 'ReasoningDifficultyDatasetEvaluator', 'type': 'generator', 'description': '为每个问题确定一个难度分数标签', 'command': {'init': [{'name': 'llm_serving', 'default': None, 'kind': 'POSITIONAL_OR_KEYWORD'}], 'run': [{'name': 'storage', 'default': None, 'kind': 'POSITIONAL_OR_KEYWORD'}, {'name': 'input_key', 'default': None, 'kind': 'POSITIONAL_OR_KEYWORD'}, {'name': 'output_key', 'default': 'difficulty_score', 'kind': 'POSITIONAL_OR_KEYWORD'}]}, 'required': True, 'depends_on': [], 'mode': 'GPU', 'id': 'node3'}, {'name': 'ReasoningCategoryDatasetEvaluator', 'type': 'generator', 'description': '将所有问题分类到7个大类别，以及每个大类别下的若干的小类别', 'command': {'init': [{'name': 'llm_serving', 'default': None, 'kind': 'POSITIONAL_OR_KEYWORD'}], 'run': [{'name': 'storage', 'default': None, 'kind': 'POSITIONAL_OR_KEYWORD'}, {'name': 'input_key', 'default': 'instruction', 'kind': 'POSITIONAL_OR_KEYWORD'}, {'name': 'output_key', 'default': 'question_category', 'kind': 'POSITIONAL_OR_KEYWORD'}]}, 'required': True, 'depends_on': [], 'mode': 'GPU', 'id': 'node4'}, {'name': 'AnswerPipelineRoot', 'type': 'generator', 'description': '用于检查数据是否包含Answer、groundtruth，并分离有答案和没答案的数据，方便后续分别处理', 'command': {'init': [], 'run': [{'name': 'storage', 'default': None, 'kind': 'POSITIONAL_OR_KEYWORD'}, {'name': 'input_answer_key', 'default': 'output', 'kind': 'POSITIONAL_OR_KEYWORD'}, {'name': 'input_gt_key', 'default': 'golden_answer', 'kind': 'POSITIONAL_OR_KEYWORD'}]}, 'required': True, 'depends_on': [], 'mode': '', 'id': 'node5'}, {'name': 'PseudoAnswerGenerator', 'type': 'generator', 'description': '该算子能够生成多个候选答案，并通过统计方法选择最优解，从而实现伪答案的生成。其输入参数包括：输入文件路径（input_file）、输出文件路径（output_file）、最大生成次数（max_times）以及统计选择模式（selection_mode，可选frequency或consistency）。输出参数包括：最终选择的答案字段（final_answer）和候选答案列表字段（candidate_answers）。该算子能够生成多个候选答案，并通过统计方法选择最优解，从而实现伪答案的生成。其输入参数包括：输入文件路径（input_file）、输出文件路径（output_file）、最大生成次数（max_times）以及统计选择模式（selection_mode，可选frequency或consistency）。输出参数包括：最终选择的答案字段（final_answer）和候选答案列表字段（candidate_answers）。', 'command': {'init': [{'name': 'llm_serving', 'default': None, 'kind': 'POSITIONAL_OR_KEYWORD'}, {'name': 'max_times', 'default': 3, 'kind': 'POSITIONAL_OR_KEYWORD'}], 'run': [{'name': 'storage', 'default': None, 'kind': 'POSITIONAL_OR_KEYWORD'}, {'name': 'input_key', 'default': 'instruction', 'kind': 'POSITIONAL_OR_KEYWORD'}, {'name': 'output_key_answer', 'default': 'pseudo_answers', 'kind': 'POSITIONAL_OR_KEYWORD'}, {'name': 'output_key_answer_value', 'default': 'pseudo_answer_value', 'kind': 'POSITIONAL_OR_KEYWORD'}, {'name': 'output_key_solutions', 'default': 'pseudo_solutions', 'kind': 'POSITIONAL_OR_KEYWORD'}, {'name': 'output_key_correct_solution_example', 'default': 'pseudo_correct_solution_example', 'kind': 'POSITIONAL_OR_KEYWORD'}]}, 'required': True, 'depends_on': [], 'mode': 'GPU', 'id': 'node6'}, {'name': 'AnswerFormatterFilter', 'type': 'processor', 'description': '按照给定的格式，基于规则过滤掉不符合格式要求的数据', 'command': {'init': [], 'run': [{'name': 'storage', 'default': None, 'kind': 'POSITIONAL_OR_KEYWORD'}, {'name': 'input_key', 'default': 'generated_cot', 'kind': 'POSITIONAL_OR_KEYWORD'}]}, 'required': True, 'depends_on': [], 'mode': '', 'id': 'node7'}, {'name': 'AnswerTokenLengthFilter', 'type': 'processor', 'description': '过滤掉Answer长度不合适的数据', 'command': {'init': [{'name': 'max_answer_token_length', 'default': 8192, 'kind': 'POSITIONAL_OR_KEYWORD'}, {'name': 'tokenizer_dir', 'default': 'Qwen/Qwen2.5-0.5B-Instruct', 'kind': 'POSITIONAL_OR_KEYWORD'}], 'run': [{'name': 'storage', 'default': None, 'kind': 'POSITIONAL_OR_KEYWORD'}, {'name': 'input_key', 'default': 'generated_cot', 'kind': 'POSITIONAL_OR_KEYWORD'}]}, 'required': True, 'depends_on': [], 'mode': '', 'id': 'node8'}], 'name': 'fcfa5825-d28b-4eba-ad17-7a46d7cc2fc7_pipeline'}
    # print(local_tool_for_execute_the_recommended_pipeline(pre_result,dry_run=True))