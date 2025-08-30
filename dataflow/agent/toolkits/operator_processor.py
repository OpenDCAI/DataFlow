#!/usr/bin/env python3
"""
operator_codegen.py ── Operator Code Generation and Debugging Utilities
Author  : [Zhou Liu]
License : MIT
Created : 2024-07-10

This module provides utilities for dynamic code generation, discovery, retrieval, and execution of DataFlow operators
within a model-driven workflow (such as LLM-based agent pipelines).

Features:
* Automatic operator main-file generation: Wraps an operator class with a runnable script for quick testing and deployment.
* Operator class discovery: Finds the first valid operator class in a given module based on registration and inheritance.
* Source code extraction: Retrieves and formats the source code of operator classes for inspection or debugging.
* Automated execution: Supports dry-run or actual execution of the auto-generated operator script with logging.
* LLM serving integration: Handles both remote (API) and local (VLLM) language model serving preparation for operator instantiation.
* Error handling and clear logging throughout the code generation and execution process.

Intended for advanced developer workflows where operators are written, debugged, and tested semi-automatically,
especially in agent-driven or AutoML/dataflow systems.

Thread-safety: This module is not inherently thread-safe. Ensure safe usage in concurrent or distributed environments.

Dependencies:
- dataflow (core framework for OperatorABC, storage, and serving)
- Python 3.8+
- Standard library: importlib, inspect, pathlib, subprocess, sys, textwrap, typing, os

Usage:
    # Typical usage scenario
    from operator_codegen import generate_operator_py, local_tool_for_debug_and_exe_operator

    # Generate runnable operator code and optionally execute it
    code = generate_operator_py(request)
    result = local_tool_for_debug_and_exe_operator(request, dry_run=False)

"""
import time
from dataflow.utils.registry import OPERATOR_REGISTRY
OPERATOR_REGISTRY._get_all()
import importlib
import inspect
import os
from pathlib import Path
import subprocess
import sys
import textwrap
from .tools import ChatAgentRequest
from typing import Dict,Any, List
from dataflow import get_logger
logger = get_logger()

def _py_literal(val: Any) -> str:
    if isinstance(val, str):
        return repr(val)         
    if val is None:
        return '""'
    return repr(val)

def _find_first_operator(module) -> type:
    """
    Return the first class in the module that satisfies all of the following:
    (1) is a class,
    (2) inherits from OperatorABC,
    (3) is registered in OPERATOR_REGISTRY.
    """
    from dataflow.core import OperatorABC
    from dataflow.utils.registry import OPERATOR_REGISTRY

    _NAMECLS = {name: cls for name, cls in OPERATOR_REGISTRY}
    logger.debug(f'[_NAMECLS]: {_NAMECLS}')

    for obj in module.__dict__.values():
        if inspect.isclass(obj) and issubclass(obj, OperatorABC) and obj is not OperatorABC:
            if OPERATOR_REGISTRY.get(obj.__name__) is obj:
                return obj
    raise RuntimeError("No eligible operator class found in this file")

def generate_operator_py(
    request: ChatAgentRequest,
) -> str:
    src_path = Path(request.py_path).expanduser().resolve()
    if not src_path.exists():
        raise FileNotFoundError(src_path)
    
    original_code = src_path.read_text(encoding="utf-8")
    spec = importlib.util.spec_from_file_location(src_path.stem, src_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[src_path.stem] = module
    spec.loader.exec_module(module)
    operator_cls = _find_first_operator(module)
    cls_name = operator_cls.__name__
    # 解析 __init__，构造实例化形参（带默认值）
    sig = inspect.signature(operator_cls.__init__)

    init_parts = []
    op_need_llm = False
    llm_block = ' '
    for name, param in list(sig.parameters.items())[1:]:
        if name == "llm_serving":
            init_parts.append("llm_serving=llm_serving")
            op_need_llm = True
        elif param.default is param.empty:
            init_parts.append(f"# TODO: {name}=...")
        else:
            init_parts.append(f"{name}={_py_literal(param.default)}")
    init_call = f"{cls_name}({', '.join(init_parts)})"

    # ---------- LLM-Serving 代码块 ----------
    if op_need_llm:
        logger.debug("[op_need_llm]")
        if request.use_local_model:
            # 暂时不写，等确定的Serving
            llm_block = textwrap.dedent(
                f"""\
                # -------- LLM Serving (Local) --------
                llm_serving = LocalModelLLMServing_vllm(
                    hf_model_name_or_path="{request.local_model_name_or_path}",
                    vllm_tensor_parallel_size=1,
                    vllm_max_tokens=8192,
                    hf_local_dir="local",
                )
                """
            )
                # f"""\
                # # -------- LLM Serving (Local) --------
                # llm_serving = LocalModelLLMServing_vllm(
                #     hf_model_name_or_path="{request.local_model_name_or_path}",
                #     vllm_tensor_parallel_size=1,
                #     vllm_max_tokens=8192,
                #     hf_local_dir="local",
                # )
                # """
        else:
            llm_block = textwrap.dedent(
                f"""\
                # -------- LLM Serving (Remote) --------
                llm_serving = APILLMServing_request(
                    api_url="{request.chat_api_url}",
                    key_name_of_api_key = 'DF_API_KEY',
                    model_name="{request.model}",
                    max_workers=100,
                )
                # 若需本地模型，请改用 LocalModelLLMServing 并设置 local=True
                """
            )

    # ---------- main 代码块 ----------
    main_block = textwrap.dedent(
        f"""
        # ======== Auto-generated runner ========
        from dataflow.utils.storage import FileStorage
        from dataflow.serving import APILLMServing_request, LocalModelLLMServing_vllm, LocalModelLLMServing_sglang
        from dataflow.core import LLMServingABC

        if __name__ == "__main__":
            # 1. FileStorage
            storage = FileStorage(
                first_entry_file_name="{request.json_file}",
                cache_path="./cache_local",
                file_name_prefix="dataflow_cache_step",
                cache_type="jsonl",
            )

            # 2. LLM-Serving
        """
    ) + textwrap.indent(llm_block, " " * 4) + textwrap.dedent(
        f"""
            # 3. Instantiate operator
            operator = {init_call}

            # 4. Run
            operator.run(storage=storage.step())
        """
    )

    full_code = f"{original_code.rstrip()}\n\n{main_block}"
    logger.debug("[full_code]")

    output_path = Path(request.py_path).expanduser().resolve()
    output_path.write_text(full_code, encoding="utf-8")
    return full_code

def local_tool_for_get_match_operator_code(pre_task_result: Dict[str, Any]) -> str:
    start_time = time.time()
    if not pre_task_result or not isinstance(pre_task_result, dict):
        return "# ❗ pre_task_result is empty, cannot extract operator names"

    from dataflow.utils.registry import OPERATOR_REGISTRY
    _NAME2CLS = {name: cls for name, cls in OPERATOR_REGISTRY}

    blocks: list[str] = []
    for op_name in pre_task_result.get("match_operators", [])[:2]:
        cls = _NAME2CLS.get(op_name)
        if cls is None:
            blocks.append(f"# --- {op_name} is not registered in OPERATOR_REGISTRY ---")
            continue
        try:
            cls_src    = inspect.getsource(cls)
            module_src = inspect.getsource(sys.modules[cls.__module__])

            # ---------- 修正版：拿到所有 import ----------
            import_lines = [
                l for l in module_src.splitlines()
                if l.strip().startswith(("import ", "from "))
            ]
            import_block = "\n".join(import_lines)
            # --------------------------------------------
            src_block = f"# === Source of {op_name} ===\n{import_block}\n\n{cls_src}"
            blocks.append(src_block)
        except (OSError, TypeError) as e:
            blocks.append(f"# --- Failed to get the source code of {op_name}: {e} ---")
    elapsed = time.time() - start_time
    logger.warning(f"[local_tool_for_get_match_operator_code] Time used: {elapsed:.4f} seconds")
    return "\n\n".join(blocks)

def local_tool_for_debug_and_exe_operator(
    request: ChatAgentRequest,
    *,
    dry_run: bool = False,
    is_in_debug_process : bool = False,
    current_round: int = 0
):
    py_file = Path(request.py_path)
    logger.info(f'[in local_tool_for_debug_and_exe_operator]:{request.py_path}')
    if not py_file.exists():
        # logger.info('111111111111111111')
        raise FileNotFoundError(f"Operator file does not exist: {py_file}")
    else:
        code = generate_operator_py(request=request)
        logger.info('[执行generate！！！]')
    
    # if current_round == 0 and py_file.exists():
    #     code = generate_operator_py(request=request)
    #     logger.info(f"Reusing existing opetator file {request.py_path}")
    # else:
    #     code = py_file.read_text(encoding="utf-8")
    logger.info(f"[Agent Generated Operator Code in {request.py_path}]:\n{code}")
    if dry_run:
        # Only surface the code to the caller – no execution.
        logger.info("[Dry-run] Operator will not be executed.")
        return code
    if request.execute_the_operator:
        logger.info("\n[............Operator is running............]\n")
        # run_res = subprocess.run(
        #     [sys.executable, str(py_file)],
        #     capture_output=True,
        #     text=True,
        # )
        run_res = subprocess.run(
            [sys.executable, "-u", str(py_file)],  
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,              
            text=True,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        if run_res.returncode != 0:
            raise RuntimeError(
                f"{py_file} exited with {run_res.returncode}\n"
                f"stdout:\n{run_res.stdout}\n"
                f"stderr:\n{run_res.stderr}"
            )
        logger.info(
            f"\n[............Operator {request.py_path} executed successfully............]\n"
            f"stdout:\n{run_res.stdout}"
            f"stdout:\n{run_res.stderr}"
        )
    return code