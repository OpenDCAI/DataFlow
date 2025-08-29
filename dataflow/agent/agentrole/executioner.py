"""
executioner.py  ── Core task-chain execution and LLM-driven code repair engine
Author  : [Zhou Liu]
License : MIT
Created : 2024-07-02

This module enables automated, robust execution of Task-Chain pipelines using generative AI:

* Dynamically generates executable code for each param_func string placeholder via LLM calls.
* Saves, executes, and captures output; on failure, invokes the DebugAgent to let the LLM repair code, repeating until success or max_rounds is reached.
* On successful execution, dynamically imports the resulting .py files and replaces Task object string fields with true callable functions, enabling downstream task continuity.
* Maintains execution context, including dependencies on previous task outputs, debug history, function pools, and conversational memory data.
* Designed to support iterative, context-aware, and resilient agent workflows that require dynamic code generation and seamless error recovery.

Thread-safety is determined by the underlying context and code execution mechanisms; dynamically generated code should be inspected for side effects.

"""
import importlib.util
import inspect
import os
import signal
import subprocess
import tempfile
import traceback
import sys
from ..servicemanager.memory_service import Memory,MemoryClient
from ..promptstemplates.prompt_template import PromptsTemplateGenerator
from .debugger import DebugAgent
from ..taskcenter import Task
from ..toolkits import (
    ChatResponse,
    ChatAgentRequest,
    local_tool_for_sample
)
from typing import Dict, List, Any,Iterable,Sequence
import yaml
import json
import re
from json import JSONDecoder, JSONDecodeError
import ast
import decimal
import datetime as dt
from pathlib import Path
from dataflow import get_logger
logger = get_logger()
import io
import os
import sys
import time
import psutil          
import cloudpickle     
import traceback
import contextlib
import multiprocessing as mp
from types import FunctionType
from dataclasses import dataclass
from typing import Any, Dict
import textwrap
import shutil
from rich.console import Console
from rich.live import Live
from rich.spinner import Spinner
from rich.text import Text
console = Console()
from dataflow.cli_funcs.paths import DataFlowPath
DATAFLOW_PATH = DataFlowPath.get_dataflow_dir()
@dataclass
class SandboxResult:
    success: bool
    result: Any = None
    stdout: str = ""
    stderr: str = ""
    traceback: str = ""
    exc_repr: str = ""
    duration: float = 0.0
# wanted_ref_file
def _normalise_wanted_ref_files(
    wanted_ref_files: str | Path | Iterable[str | Path] | None,
) -> list[str]:
    if not wanted_ref_files:      # None, '' 等
        return []
    if isinstance(wanted_ref_files, (str, Path)):
        raw: Sequence[str | Path]
        if isinstance(wanted_ref_files, str) and wanted_ref_files.strip().startswith(("[", "(")):
            try:
                raw = ast.literal_eval(wanted_ref_files)
                if isinstance(raw, (str, bytes)):    
                    raw = [raw]
            except Exception:
                raw = [wanted_ref_files]
        else:
            raw = [wanted_ref_files]
    else:
        raw = list(wanted_ref_files)          
    basenames: list[str] = []
    for item in raw:
        if not item:
            continue
        basenames.append(Path(str(item)).name)
    seen: set[str] = set()
    return [x for x in basenames if not (x in seen or seen.add(x))]

def _extract_deepest_frame_source(
    wanted_ref_files: Iterable[str] | str | Path | None = None,
) -> str:
    # ---------- 兼容性增强部分 ----------
    wanted_basenames: list[str] = _normalise_wanted_ref_files(wanted_ref_files)
    if not wanted_basenames:
        return ""
    dataflow_path = DATAFLOW_PATH.expanduser().resolve()
    merged_source: list[str] = []
    for basename in wanted_basenames:
        try:
            file_path = next(dataflow_path.rglob(basename))
        except StopIteration:
            continue

        try:
            merged_source.append(file_path.read_text(encoding="utf-8"))
        except Exception:
            continue
    if merged_source:
        separator = "\n\n# " + "\n\n# ".join("=" * 80 for _ in merged_source) + "\n\n"
        return separator.join(merged_source)
    return ""

class Tee(io.TextIOBase):
    def __init__(self, *targets):
        self.targets = targets
    def write(self, s):
        for t in self.targets:
            t.write(s)
        return len(s)
    def flush(self):
        for t in self.targets:
            t.flush()
    def isatty(self):
        return getattr(self.targets[0], "isatty", lambda: False)()

def _sandbox_worker(func_ser: bytes, args: tuple,
                    kwargs: Dict[str, Any], q: mp.Queue):
    start = time.perf_counter()

    # Set the process group if supported (for easier child process management)
    if hasattr(os, "setpgrp"):
        os.setpgrp()

    out_buf, err_buf = io.StringIO(), io.StringIO()
    tee_out = Tee(sys.__stdout__, out_buf)
    tee_err = Tee(sys.__stderr__, err_buf)

    try:
        func: FunctionType = cloudpickle.loads(func_ser)
        with contextlib.redirect_stdout(tee_out), \
             contextlib.redirect_stderr(tee_err):
            ret = func(*args, **kwargs)

        q.put(SandboxResult(
            success=True,
            result=ret,
            stdout=out_buf.getvalue(),
            stderr=err_buf.getvalue(),
            duration=time.perf_counter() - start,
        ))
    except Exception as e:
        q.put(SandboxResult(
            success=False,
            result=None,
            stdout=out_buf.getvalue(),
            stderr=err_buf.getvalue(),
            traceback=traceback.format_exc(),
            exc_repr=repr(e),
            duration=time.perf_counter() - start,
        ))

def run_function_isolated(
    func: FunctionType,
    *,
    args: tuple = (),
    kwargs: Dict[str, Any] | None = None,
    timeout: int = 120,
) -> SandboxResult:
    """
    Run `func` in a **fully isolated** subprocess. Even if the function internally forks or spawns
    subprocesses, it is safe. Always returns a SandboxResult and never raises exceptions to the caller.
    """
    kwargs = kwargs or {}
    func_ser = cloudpickle.dumps(func)
    q: mp.Queue = mp.Queue(maxsize=1)
    proc = mp.Process(target=_sandbox_worker, args=(func_ser, args, kwargs, q))
    proc.start()
    proc.join(timeout)

    def _kill_tree(pid: int):
        try:
            parent = psutil.Process(pid)
            for child in parent.children(recursive=True):
                child.kill()
            parent.kill()
        except psutil.NoSuchProcess:
            pass

    # ── Timeout: kill the process group / child processes ──────────────────────────────
    if proc.is_alive():
        if hasattr(os, "killpg"):
            try:
                os.killpg(proc.pid, signal.SIGKILL)   # POSIX
            except Exception:
                pass
        _kill_tree(proc.pid)
        proc.join(1)
        return SandboxResult(
            success=False,
            exc_repr="TimeoutError",
            traceback=f"TimeoutError: execution exceeded {timeout}s",
        )

    # ── Read execution result ─────────────────────────────────────────
    try:
        return q.get_nowait()
    except Exception:
        # In theory, this only happens if the child process crashes and fails to write to the queue
        return SandboxResult(
            success=False,
            exc_repr="EmptyResult",
            traceback="Child process exited abnormally and returned no result.",
        )

class ExecutionAgent:
    def __init__(
        self,
        # config_path: str,
        request: ChatAgentRequest,
        memory_entity: Memory,
        prompt_template: PromptsTemplateGenerator,
        debug_agent: DebugAgent,
        task_chain: list,
    ):
        # with open(config_path, "r", encoding="utf-8") as f:
        #     cfg = yaml.safe_load(f)
        # self.api_key = cfg["API_KEY"]
        # self.api_url = cfg["CHAT_API_URL"]
        # self.model = cfg["MODEL"]
        self.api_key = request.api_key
        self.api_url = request.chat_api_url
        self.model = request.model

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        self.generated_code: Dict[str, str] = {}
        self.generated_funcs_execution_results: Dict[str, dict] = {}
        self.request = request
        self.memory_entity = memory_entity
        self.session_id = self.memory_entity.get_session_id(request.sessionKEY)
        self.prompt_template = prompt_template
        self.client = MemoryClient(self.memory_entity)
        self.debug_agent = debug_agent
        self.task_chain = task_chain
        self.output_dir = "/mnt/h_h_public/lh/lz/DataFlow/dataflow/agent/agentrole/test/TempFiles"
        self.debug_history: Dict[str, list] = {}
        self.latest_code = ""
        self.functions_pool: Dict[str, dict] = {}
        self.pre_tasks_context: List[dict] = []

    def register_functions_pool(self, result: dict, task_name: str):
        func_name = result.get("function_name", "")
        func_desc = result.get("description", "")
        func_key = func_name

        if func_key not in self.functions_pool:
            self.functions_pool[func_key] = {
                "description": func_desc,
                "parameters": result.get("parameters", []),
                "return": result.get("return", {}),
                "code": result.get("code", ""),
                "py_file": "",
                "txt_file": "",
                "success": False,
                "last_exec_result": "",
                "tasks_used_by": [],
            }

        if task_name not in self.functions_pool[func_key]["tasks_used_by"]:
            self.functions_pool[func_key]["tasks_used_by"].append(task_name)
        return func_key

    async def generate_function_code(
        self,
        function_name: str,
        task: Task,
        debug_info: str,
        latest_code: str,
    ) -> Dict[str, Any]:
        # ------- 构造 prompt -------
        all_dep_param_funcs: List[str] = []
        if task.depends_on:
            logger.debug(f"任务 {task.task_name} 依赖于 {task.depends_on}")
            dep_param_funcs = None
            for dep_task_name in task.depends_on:
                dep_task = next(
                    (t for t in self.task_chain if getattr(t, "task_name", None) == dep_task_name), None
                )
                if dep_task is not None:
                    dep_param_funcs = getattr(dep_task, "param_funcs", [])
                    dep_task_template_key = getattr(dep_task, "task_template", "")
                    dep_task_template = self.prompt_template.templates.get(dep_task_template_key, "")
                    dep_func_outputs = {}
                    for dep_func in dep_param_funcs:
                        output_key = f"{dep_task_name}_{dep_func}"
                        logger.debug(f"dep_func ==== {dep_func},output_key ==== {output_key}")
                        output_value = self.memory_entity.get_session_data(self.session_id, output_key)
                        logger.debug(f"依赖函数返回值 {output_key}: {output_value}")
                        dep_func_outputs[output_key] = output_value

                    self.pre_tasks_context.append(
                        {
                            "task_name": dep_task_name,
                            "task_prompt": dep_task_template,
                            "task_func_outputs": dep_func_outputs,
                            "task_final_result": "",
                        }
                    )
                    all_dep_param_funcs.extend(list(dep_func_outputs.keys()))
                else:
                    self.pre_tasks_context.append(
                        {"task_name": dep_task_name, "task_prompt": "", "task_func_outputs": None}
                    )
            logger.debug("分支1")
            logger.warning(f"==== all_dep_param_funcs ==== {all_dep_param_funcs}")
            user_prompt = self.prompt_template.render(
                "task_prompt_for_executioner_with_dep",
                pre_tasks_context=self.pre_tasks_context,
                task_info=self.prompt_template.templates.get(f"{task.task_template}", ""),
                function_name=function_name,
                latest_code=latest_code,
                dep_param_funcs=all_dep_param_funcs,
                debug_info=debug_info,
            )
        else:
            if debug_info and debug_info.strip():
                logger.debug("分支2")
                prompt_key = "task_prompt_for_executioner_debug"
                user_prompt = self.prompt_template.render(
                    prompt_key,
                    task_info=self.prompt_template.templates[f"{task.task_template}"],
                    function_name=function_name,
                    latest_code=latest_code,
                    debug_info=debug_info,
                )
            else:
                logger.debug("分支3")
                prompt_key = "task_prompt_for_executioner"
                user_prompt = self.prompt_template.render(
                    prompt_key,
                    task_info=self.prompt_template.templates[f"{task.task_template}"],
                    function_name=function_name,
                )

        system_prompt = self.prompt_template.templates["system_prompt_for_executioner"]
        json_data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }

        # ------- 调用 LLM -------
        response_content = await self.client.post(
            url=self.api_url,
            headers=self.headers,
            json_data=json_data,
            session_key=self.request.sessionKEY,
        )
        result = self.robust_parse_json(response_content)
        self.register_functions_pool(result=result, task_name=task.task_name)
        logger.debug(f"生成的代码 JSON：\n {result}")
        return result

    def save_and_execute_code(
        self,
        task_func_content: dict,
        task: Task,
        output_dir: str = None,
        test_cases: list = None,
        extra_info: int = 0,
    ) -> Dict[str, str]:
        function_name = task_func_content.get("function_name", "my_func")
        parameters = task_func_content.get("parameters", [])
        base_name = f"{task.task_name}_{function_name}_current_round_{extra_info}"
        if output_dir is None:
            output_dir = tempfile.gettempdir()
        py_file = os.path.join(output_dir, f"{base_name}.py")
        txt_file = os.path.join(output_dir, f"{base_name}.txt")

        func_code = task_func_content.get("code", "").strip()
        py_content = func_code + "\n\n"
        if not test_cases:
            logger.debug(f"---------------parameters-------------:\n {parameters}")
            input_args = self.collect_func_args(parameters,task)
            main_code = self.build_main_test_code(function_name, input_args)
            py_content = re.sub(
                r'if\s+__name__\s*==\s*[\'"]__main__[\'"]\s*:[\s\S]*', "", py_content
            ).strip() + "\n\n" + main_code + "\n"
        else:
            py_content = re.sub(
                r'if\s+__name__\s*==\s*[\'"]__main__[\'"]\s*:[\s\S]*', "", py_content
            ).strip() + "\n\n" + "if __name__ == '__main__':\n"
            for case in test_cases:
                py_content += f"    {case.strip()}\n"
            py_content += "\n"
        with open(py_file, "w", encoding="utf-8") as f:
            f.write(py_content)
        try:
            with open(txt_file, "w", encoding="utf-8") as outfile:
                subprocess.run(
                    ["python", py_file],
                    stdout=outfile,
                    stderr=subprocess.STDOUT,
                    timeout=60,
                )
        except Exception as e:
            with open(txt_file, "a", encoding="utf-8") as outfile:
                outfile.write(f"\n执行出错: {e}\n")
        return {"py_file": py_file, "txt_file": txt_file}

    async def execute(self, max_rounds: int = 5):
        try:
            if os.path.exists(self.output_dir):
                shutil.rmtree(self.output_dir)
            os.makedirs(self.output_dir, exist_ok=True)
            logger.debug("已清空并重新创建临时目录: %s", self.output_dir)
        except Exception as e:
            logger.warning("清空临时目录 %s 失败: %s", self.output_dir, e)

        for task in self.task_chain:
            task_name = getattr(task, "task_name", "default_task")
            param_funcs = getattr(task, "param_funcs", {})

            for function_name, func_info in param_funcs.items():
                if not isinstance(func_info, str):
                    continue  # 已经是可调用对象
                logger.info(f"开始执行任务 {task_name} -> 函数 {function_name}")
                if function_name in self.functions_pool:
                    logger.debug(f"------ 函数 {function_name} 已存在于函数池，跳过生成 ------")
                    continue

                current_round = 0
                current_code = ""
                debug_info = ""
                debug_key = f"{task_name}_{function_name}"
                self.debug_history[debug_key] = []
                py_file = ""

                while current_round < max_rounds:
                    logger.debug(f"[{function_name}] Round {current_round + 1} 生成 & 执行中...")
                    try:
                        result = await self.generate_function_code(
                            function_name=function_name,
                            task=task,
                            debug_info=debug_info,
                            latest_code=current_code,
                        )
                        current_code = result.get("code", "")
                        self.latest_code = current_code
                        logger.debug(f"生成代码:\n{current_code}")
                    except Exception as e:
                        last_error = f"LLM 生成代码失败: {e}"
                        logger.error(last_error)
                        break
                    try:
                        files = self.save_and_execute_code(
                            task_func_content=result,
                            task=task,
                            output_dir=self.output_dir,
                            extra_info=current_round,
                        )
                        txt_file, py_file = files["txt_file"], files["py_file"]
                        with open(txt_file, "r", encoding="utf-8") as f:
                            exec_result = f.read()
                        logger.debug(f"[执行结果]\n{exec_result}")
                        # 保存执行结果到 memory
                        self.memory_entity.set_session_data(self.session_id, debug_key, exec_result)
                        with open(py_file, "r", encoding="utf-8") as f:
                            code_content = f.read()
                    except Exception as e:
                        exec_result = f"代码执行失败: {e}\n {traceback.format_exc()}"
                        logger.error(exec_result)
                    # ---------------- 调用 DebugAgent ---------------- #
                    debug_success = True
                    if any(err in exec_result for err in ["Traceback", "错误", "Error"]):
                        debug_success = False
                        logger.warning(f"检测到执行错误，调用 DebugAgent 修复...")
                        try:
                            debug_info = await self.debug_agent.debug_simple_tool_code(
                                func_name=function_name,
                                code=code_content,
                                error=exec_result,
                                history="",
                            )
                            self.debug_history[debug_key].append(
                                {
                                    "func_name": function_name,
                                    "current_round": current_round + 1,
                                    "debug_info": debug_info,
                                }
                            )
                        except Exception as e:
                            debug_info = f"debug_agent 调用失败: {e}"
                            logger.error(debug_info)

                    # ---------------- 回合结果处理 ---------------- #
                    if debug_success:
                        logger.debug(f"-----------函数 {function_name} 运行成功！------------")
                        self.generated_funcs_execution_results[debug_key] = {
                            "result": exec_result,
                            "code": current_code,
                            "success": True,
                            "final_code_file_path": py_file,
                        }
                        break
                    else:
                        current_round += 1
                        self.generated_funcs_execution_results[debug_key] = {
                            "result": exec_result,
                            "code": current_code,
                            "success": False,
                        }
                        if current_round == max_rounds:
                            logger.error(f"函数 {function_name} 修复 {max_rounds} 次仍未成功，已放弃。")
                            logger.error("任务无法继续执行，正在退出......")
                            sys.exit(1)

        logger.info("=== 所有任务执行完毕 ===")
        logger.debug(f"Debug 历史记录: {self.debug_history}")
        return self.generated_funcs_execution_results

    @staticmethod
    def _strip_json_comments(s: str) -> str:
        s = re.sub(r"/\*.*?\*/", "", s, flags=re.DOTALL)           # 块注释
        s = re.sub(r"^\s*//.*$", "", s, flags=re.MULTILINE)        # 行首 // 注释
        s = re.sub(r",\s*([}\]])", r"\1", s)                       # 尾逗号
        return s

    def robust_parse_json(self, s: str) -> dict:
        clean = self._strip_json_comments(s)
        decoder = JSONDecoder()
        idx = 0
        dicts = []
        length = len(clean)
        while True:
            idx = clean.find("{", idx)
            if idx < 0 or idx >= length:
                break
            try:
                obj, end = decoder.raw_decode(clean, idx)
                if isinstance(obj, dict):
                    dicts.append(obj)
                idx = end
            except JSONDecodeError:
                idx += 1

        if not dicts:
            raise ValueError("在输入中未提取到任何 JSON 对象")
        if len(dicts) == 1:
            return dicts[0]

        merged: Dict[str, Any] = {}
        for d in dicts:
            merged.update(d)
        return merged

    async def update_task_chain_with_executed_functions(self):
        logger.info("开始更新任务链中的函数引用 ...")
        for task_idx, task in enumerate(self.task_chain):
            task_name = getattr(task, "task_name", f"unnamed_task_{task_idx}")
            if not hasattr(task, "param_funcs") or not isinstance(task.param_funcs, dict):
                logger.debug(f"任务 {task_name} 没有可更新的 param_funcs，跳过。")
                continue

            updated_param_funcs = task.param_funcs.copy()
            for func_key_name in task.param_funcs.keys():
                result_lookup_key = f"{task_name}_{func_key_name}"
                if result_lookup_key not in self.generated_funcs_execution_results:
                    logger.debug(f"{result_lookup_key} 未找到执行结果，跳过。")
                    continue

                exec_info = self.generated_funcs_execution_results[result_lookup_key]
                if not (exec_info.get("success") and "final_code_file_path" in exec_info):
                    logger.debug(f"{result_lookup_key} 执行不成功或缺少文件路径，跳过。")
                    continue

                file_path = exec_info["final_code_file_path"]
                if not os.path.exists(file_path):
                    logger.error(f"文件路径不存在: {file_path}")
                    continue

                try:
                    sanitized_task = "".join(c if c.isalnum() else "_" for c in task_name)
                    sanitized_func = "".join(c if c.isalnum() else "_" for c in func_key_name)
                    module_name = f"agent_executed_{sanitized_task}_{sanitized_func}"
                    file_dir = os.path.dirname(file_path)
                    if file_dir not in sys.path:
                        sys.path.insert(0, file_dir)
                        remove_path_after_import = True
                    else:
                        remove_path_after_import = False

                    spec = importlib.util.spec_from_file_location(module_name, file_path)
                    if spec is None or spec.loader is None:
                        logger.error(f"无法创建模块规范: {file_path}")
                        if remove_path_after_import:
                            sys.path.pop(0)
                        continue

                    imported_module = importlib.util.module_from_spec(spec)
                    sys.modules[module_name] = imported_module
                    spec.loader.exec_module(imported_module)

                    if hasattr(imported_module, func_key_name):
                        actual_function = getattr(imported_module, func_key_name)
                        if callable(actual_function):
                            updated_param_funcs[func_key_name] = actual_function
                            logger.debug(f"成功导入 {func_key_name} 并更新到任务 {task_name}")
                        else:
                            logger.warning(f"{func_key_name} 不是可调用对象")
                    else:
                        logger.warning(f"{func_key_name} 未在模块中找到")

                    if remove_path_after_import and sys.path[0] == file_dir:
                        sys.path.pop(0)
                except Exception as e:
                    logger.error(
                        f"动态导入函数 {func_key_name} 失败: {e}\n{traceback.format_exc()}"
                    )
            task.param_funcs = updated_param_funcs
        logger.info("函数引用更新完毕。")

    def collect_func_args(self, parameters, task=None):
        input_args = {}
        for param in parameters:
            name = param["name"] # 这个name就是任务+函数的名
            ptype = param.get("type", "str").lower()
            desc = param.get("description", "")

            # 1) 依赖任务产物
            cached_val = None
            if task and getattr(task, "depends_on", None):
                for dep in task.depends_on:
                    cached_val = self.memory_entity.get_session_data(
                        self.session_id, name
                    )
                    cached_val = self._cast(cached_val,ptype)
                    # logger.debug(f"分支1 ： {dep}_{name}")
                    if cached_val is not None:
                        break
            # 2) 仍然没有 -> 交互式输入
            if cached_val is None:
                while True:
                    user_input = self.safe_input(f"请输入参数【{name}】（{ptype}，{desc}）：")
                    try:
                        cached_val = self._cast(user_input, ptype)
                        break
                    except Exception as e:
                        logger.warning(f"输入有误：{e}，请重新输入。")
            input_args[name] = cached_val
        logger.info(f"收集到调用参数：{input_args}")
        return input_args

    def _cast(self, value: Any, ptype: str) -> Any:
        """
        将 value 转换成指定 ptype 所代表的 Python 对象。

        支持的 ptype（大小写不敏感）：
            int / integer
            float / double
            decimal
            bool / boolean
            str / string
            list
            dict
            date       (返回 datetime.date)
            datetime   (返回 datetime.datetime)
            path       (返回 pathlib.Path)

        如果 ptype 未识别，则保持原样返回。
        """

        # 预处理
        if value is None or ptype is None:
            return value
        ptype = ptype.lower()

        # ---------- 标量类型 ----------
        try:
            if ptype in ("int", "integer"):
                return int(value)

            if ptype in ("float", "double"):
                return float(value)

            if ptype == "decimal":
                return decimal.Decimal(value)

            if ptype in ("bool", "boolean"):
                if isinstance(value, bool):
                    return value
                if isinstance(value, (int, float)):
                    return value != 0
                return str(value).strip().lower() in ("1", "true", "yes", "y", "ok")

            if ptype in ("str", "string"):
                return str(value)

            # ---------- 序列 / 映射 ----------
            if ptype == "list":
                return self._parse_container(value, list)

            if ptype == "dict":
                return self._parse_container(value, dict)

            # ---------- 日期时间 ----------
            if ptype == "date":
                if isinstance(value, dt.date) and not isinstance(value, dt.datetime):
                    return value
                return dt.date.fromisoformat(str(value))

            if ptype == "datetime":
                if isinstance(value, dt.datetime):
                    return value
                return dt.datetime.fromisoformat(str(value))

            # ---------- 文件 / 路径 ----------
            if ptype == "path":
                return Path(value)

        except Exception as e:
            # 统一包装异常，方便日志排查
            raise ValueError(f"Failed to cast {value!r} to {ptype}: {e}") from e

        # ---------- 默认回退 ----------
        return value

    def safe_input(slef, prompt=""):
        try:
            return input(prompt)
        except UnicodeDecodeError:
            sys.stdout.write(prompt)
            raw = sys.stdin.buffer.readline()
            import locale
            encodings = [locale.getpreferredencoding(False), "gbk", "cp936", "latin1"]
            for enc in encodings:
                try:
                    return raw.decode(enc).rstrip("\r\n")
                except UnicodeDecodeError:
                    continue
            return repr(raw)

    def _parse_container(self, raw: Any, expect_type: type):
        """
        将 raw 解析成 list 或 dict。
        解析顺序：
            1. 已经是目标类型 → 直接返回
            2. 尝试 json.loads
            3. 尝试 ast.literal_eval（允许 {'a':1} 这类 Python 字面量）
        """
        if isinstance(raw, expect_type):
            return raw

        if not isinstance(raw, str):
            raise TypeError(f"Need str to parse into {expect_type.__name__}, got {type(raw)}")

        try:
            obj = json.loads(raw)
            if isinstance(obj, expect_type):
                return obj
        except json.JSONDecodeError:
            pass

        obj = ast.literal_eval(raw) 
        if isinstance(obj, expect_type):
            return obj

        raise ValueError(f"String does not represent a {expect_type.__name__}: {raw}")

    @staticmethod
    def build_main_test_code(function_name: str, input_args: Dict[str, Any]) -> str:
        import pprint
        params_literal = pprint.pformat(input_args, indent=4, width=80)
        main_code = (
            "if __name__ == '__main__':\n"
            f"    params = {params_literal}\n"
            f"    result = {function_name}(**params)\n"
            "    print(result)\n"
        )
        return main_code

    async def execute_step(self, step_index: int):
        pass
    
    async def safe_call_tool(
        self,
        tool_func,
        *,
        func_name: str,
        task_name: str,
        call_args: Dict[str, Any],
        history: str = "",
        max_debug_rounds: int = 10,
        timeout: int = 120,
        is_debug_subprocess_code: bool = True,
    ) -> "SandboxResult":
        """
        Isolated execution + LLM Debug.
        When is_debug_subprocess_code=True, prioritize debugging the script
        launched internally by tool_func (call_args["py_path"]). 
        If the script does not exist yet, run as a normal function first,
        wait for the script to be generated, and then switch to script debugging mode.
        """

        # ───────── 0. Preprocess script path ─────────
        py_file: Path | None = None
        if is_debug_subprocess_code:
            import secrets
            orig_path = Path(call_args["request"].py_path)          # recommend_pipeline.py
            token     = secrets.token_hex(8)                        # 8-char hex
            new_path  = orig_path.with_name(f"{orig_path.stem}_{token}{orig_path.suffix}")
            if orig_path.exists():
                shutil.copy(orig_path, new_path) 
                logger.info(f'[copy to new_path]: {new_path}')
                call_args["request"].py_path = new_path
            py_path: str | None = call_args.get("request").py_path
            py_file = Path(py_path)
            if not py_path:
                raise ValueError("call_args must include the 'py_path' field")
            if py_file.exists():
                code_src = py_file.read_text(encoding="utf-8")
            else:
                # File does not exist → run as a normal function first
                logger.warning(
                    f"[safe_call_tool] Subprocess script {py_file} does not exist yet, "
                    "will debug the tool function itself first, and switch after the script is generated."
                )
                code_src = inspect.getsource(tool_func)
                is_debug_subprocess_code = False   # downgrade temporarily
        else:
            code_src = inspect.getsource(tool_func)

        logger.info(f"[first code src (truncated)]: {code_src[:1000]} ...")
        wanted_ref_file = []
        current_round = 0
        while current_round <= max_debug_rounds:
            logger.info(f"[current_round]: in debug round {current_round} ......")

            spinner = Spinner("dots", text=Text(f" Debugging round {current_round}  ", style="bold green"))
            with Live(spinner, console=console, refresh_per_second=12):
                # ───────── 1. Isolated execution ─────────
                call_args["is_in_debug_process"] = True      # Inject sandbox mark
                call_args['current_round'] = current_round
                result = run_function_isolated(
                    tool_func,
                    kwargs=call_args,
                    timeout=timeout,
                )

            logger.info(
                "[run_function_isolated result]:\n"
                f"[traceback]: {result.traceback}\n"
                f"[stdout]: {result.stdout}\n"
                f"[stderr]: {result.stderr}"
            )
            if result.success:
                console.print("✅  Debug finished, function succeeded\n", style="bold green")
                return result

            # ───────── 2. If the script was just generated, switch debug target ─────────
            if (
                not is_debug_subprocess_code          # still in function mode
                and py_file is not None
                and py_file.exists()
            ):
                logger.info(f"[safe_call_tool] Script generated, switching to script debug mode: {py_file}")
                code_src = py_file.read_text(encoding="utf-8")
                is_debug_subprocess_code = True
                logger.debug('[in ogger.info(f"[safe_call_tool] Script generated, switching to script debug mode: {py_file}")]')

            # ───────── 3. Call LLM to fix ─────────
            try:
                cls_detail_code = _extract_deepest_frame_source(wanted_ref_file)
                data_keys = local_tool_for_sample(
                    call_args["request"], sample_size=1, use_file_sys=1, only_keys=True
                )
                response_content = await self.debug_agent.debug_pipeline_tool_code(
                    template_name   = "task_prompt_for_pipeline_code_debug",
                    code            = textwrap.dedent(code_src),
                    error           = f"{result.traceback},{result.stdout},{result.stderr}",
                    cls_detail_code = cls_detail_code,
                    history         = history,
                    data_keys       = data_keys,
                )
                logger.debug(f'[response_content]: 返回原文{response_content}')
                json_content = self.robust_parse_json(response_content)
                fixed_code: str | None = json_content.get("code")
                wanted_ref_file = json_content.get("wanted_ref_file")
                logger.info(f"[fixed_code]:\n{fixed_code}")
                logger.warning(f"\n -------[debug agent origin resp]------- \n  {json_content.get('reason')} \n {json_content.get('wanted_ref_file')}")
                if not fixed_code:
                    raise ValueError("No 'code' field in LLM response content")
            except Exception as e:
                logger.error(f"DebugAgent call failed: {e}")
                return result

            # ───────── 4. Apply the LLM-repaired code ─────────
            try:
                if is_debug_subprocess_code:
                    compile(fixed_code, str(py_file), "exec")
                    backup_path = py_file.with_suffix(".bak.py")
                    shutil.copy(py_file, backup_path)
                    py_file.write_text(fixed_code, encoding="utf-8")
                    logger.info(
                        f"Repaired subprocess script written: {py_file} (Backup: {backup_path})"
                    )
                else:
                    local_ns: dict[str, Any] = {}
                    exec(fixed_code, {}, local_ns)
                    tool_func = local_ns.get(func_name)
                    if not callable(tool_func):
                        raise ValueError(f"Function {func_name} not found in LLM fixed code")
                code_src = fixed_code
            except Exception as e:
                logger.error(f"Failed to apply LLM fixed code: {e}")
                return result

            current_round += 1

        logger.error("[!!!Debug Error!!!]:\n Reached max rounds of repair, still failed, please intervene manually.")
        return result
    
# if __name__=="__main__":
#     print(_extract_deepest_frame_source("['tools.py','task_definitions.py']"))