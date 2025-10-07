from __future__ import annotations

from dataflow.dataflowagent.state import DFState
import re
from dataflow.dataflowagent.graghbuilder.gragh_builder import GenericGraphBuilder
from dataflow.dataflowagent.toolkits.optool.op_tools import (
    local_tool_for_get_purpose,
    get_operator_content_str,
)
from dataflow.dataflowagent.toolkits.basetool.file_tools import (
    local_tool_for_sample,
)
from dataflow.cli_funcs.paths import DataFlowPath
from dataflow.agent.toolkits.operator_processor import (
    local_tool_for_get_match_operator_code,
)
from dataflow.dataflowagent.agentroles.match import create_match
from dataflow.dataflowagent.agentroles.writer import create_writer
from dataflow.dataflowagent.agentroles.debugger import create_code_debugger
from dataflow.dataflowagent.agentroles.oprewriter import create_rewriter
from dataflow.dataflowagent.agentroles.append_llm_serving import create_llm_append_serving
from dataflow.dataflowagent.agentroles.instantiator import create_llm_instantiator


def create_operator_write_graph() -> GenericGraphBuilder:
    """Build the operator write workflow graph.

    Flow: match_operator -> write_the_operator -> operator_executor
          -> (code_debugger -> op_rewriter -> after_rewrite -> operator_executor)*
    """
    builder = GenericGraphBuilder(state_model=DFState, entry_point="match_operator")

    # ---------------- 前置工具：match_operator ----------------
    @builder.pre_tool("get_operator_content", "match_operator")
    def pre_get_operator_content(state: DFState):
        cat = state.category.get("category") or state.request and getattr(state.request, "category", None)
        data_type = cat or state.temp_data.get("category") or "Default"
        return get_operator_content_str(data_type=data_type)

    @builder.pre_tool("purpose", "match_operator")
    def pre_get_purpose(state: DFState):
        return local_tool_for_get_purpose(state.request)

    # ---------------- 前置工具：write_the_operator ----------------
    @builder.pre_tool("example", "write_the_operator")
    def pre_example_from_matched(state: DFState):
        """
        为写算子提供更强的 in-context 示例：
        将匹配到的所有算子源码（含 import + 类定义）拼接为示例，让 LLM 模仿项目风格。
        优先从 DFState.matched_ops 读取；若为空则回退读取 agent_results。
        """
        names: list[str] = []
        try:
            if isinstance(state.matched_ops, list) and state.matched_ops:
                names = list(dict.fromkeys(state.matched_ops))
            else:
                res = state.agent_results.get("match_operator", {}).get("results", {})
                names = list(dict.fromkeys(res.get("match_operators", []) or []))
        except Exception:
            names = []

        if not names:
            return ""

        blocks = []
        chunk = 3  # 分批聚合，避免极长提示一次性超长
        for i in range(0, len(names), chunk):
            part = names[i:i+chunk]
            try:
                blocks.append(local_tool_for_get_match_operator_code({"match_operators": part}))
            except Exception:
                continue
        code_examples = "\n\n".join([b for b in blocks if b])
        # 写阶段保持泛化，不再注入样例与可用键说明
        return code_examples

    @builder.pre_tool("target", "write_the_operator")
    def pre_target(state: DFState):
        return state.request.target

    #（移除）写算子阶段不再注入 data_sample / available_keys，保持生成阶段泛化

    # ---------------- 调试相关前置工具（对齐 pipeline 复用） ----------------
    @builder.pre_tool("pipeline_code", "code_debugger")
    def dbg_get_code(state: DFState):
        return state.temp_data.get("pipeline_code", "") or getattr(state, "draft_operator_code", "")

    @builder.pre_tool("error_trace", "code_debugger")
    def dbg_get_err(state: DFState):
        return state.execution_result.get("stderr", "") or state.execution_result.get("traceback", "")

    @builder.pre_tool("pipeline_code", "op_rewriter")
    def rw_get_code(state: DFState):
        return state.temp_data.get("pipeline_code", "") or getattr(state, "draft_operator_code", "")

    @builder.pre_tool("error_trace", "op_rewriter")
    def rw_get_err(state: DFState):
        return state.execution_result.get("stderr", "") or state.execution_result.get("traceback", "")

    @builder.pre_tool("debug_reason", "op_rewriter")
    def rw_get_reason(state: DFState):
        return state.code_debug_result.get("reason", "")

    # 为 op_rewriter 注入数据上下文，辅助其在重写阶段完善自动选键逻辑
    @builder.pre_tool("data_sample", "op_rewriter")
    def rw_get_data_sample(state: DFState):
        try:
            # 使用有效数据路径，避免取不到样例
            from types import SimpleNamespace as _SN
            default_test_file = f"{DataFlowPath.get_dataflow_agent_dir()}/test_data.jsonl"
            eff_path = getattr(state.request, "json_file", "") or default_test_file
            stats = local_tool_for_sample(_SN(json_file=eff_path), sample_size=2)
            return stats.get("samples", []) if isinstance(stats, dict) else []
        except Exception:
            return []

    @builder.pre_tool("available_keys", "op_rewriter")
    def rw_get_available_keys(state: DFState):
        try:
            # 优先使用运行期调试收集到的 available_keys
            dbg = state.temp_data.get("debug_runtime", {})
            if isinstance(dbg, dict):
                dbg_keys = dbg.get("available_keys", []) or []
                if dbg_keys:
                    return dbg_keys
            from types import SimpleNamespace as _SN
            default_test_file = f"{DataFlowPath.get_dataflow_agent_dir()}/test_data.jsonl"
            eff_path = getattr(state.request, "json_file", "") or default_test_file
            stats = local_tool_for_sample(_SN(json_file=eff_path), sample_size=2)
            return stats.get("available_keys", []) if isinstance(stats, dict) else []
        except Exception:
            return []

    # 为 op_rewriter 额外提供目标与预选输入键，便于其进行键修复
    @builder.pre_tool("target", "op_rewriter")
    def rw_get_target(state: DFState):
        return getattr(state.request, "target", "")

    @builder.pre_tool("preselected_input_key", "op_rewriter")
    def rw_get_preselected_key(state: DFState):
        try:
            from types import SimpleNamespace as _SN
            default_test_file = f"{DataFlowPath.get_dataflow_agent_dir()}/test_data.jsonl"
            eff_path = getattr(state.request, "json_file", "") or default_test_file
            stats = local_tool_for_sample(_SN(json_file=eff_path), sample_size=2)
            samples = stats.get("samples", []) if isinstance(stats, dict) else []
            keys = stats.get("available_keys", []) if isinstance(stats, dict) else []
            if not samples or not keys:
                return ""
            import numpy as _np
            best_k, best_len = "", -1.0
            for k in keys:
                try:
                    vals = [str(s.get(k, "")) for s in samples]
                    avg_len = _np.mean([len(v) for v in vals]) if vals else 0.0
                except Exception:
                    avg_len = 0.0
                if avg_len > best_len:
                    best_k, best_len = k, avg_len
            return best_k
        except Exception:
            return ""

    # ---------------- LLM前置：Append LLM Serving ----------------
    @builder.pre_tool("pipeline_code", "llm_append_serving")
    def pre_llm_append_code(state: DFState):
        return state.temp_data.get("pipeline_code", "") or getattr(state, "draft_operator_code", "")

    @builder.pre_tool("llm_serving_snippet", "llm_append_serving")
    def pre_llm_serving_snippet(state: DFState):
        return (
            "# -------- LLM Serving (Remote) --------\n"
            "self.llm_serving = APILLMServing_request(\n"
            '    api_url="http://123.129.219.111:3000/v1/chat/completions",\n'
            '    key_name_of_api_key="DF_API_KEY",\n'
            '    model_name="gpt-4o",\n'
            "    max_workers=100,\n"
            ")\n"
        )

    # 追加：Append 阶段也传入上下文（仅作提示，不得用于运行逻辑）
    @builder.pre_tool("example_data", "llm_append_serving")
    def pre_llm_append_example(state: DFState):
        try:
            from types import SimpleNamespace as _SN
            default_test_file = f"{DataFlowPath.get_dataflow_agent_dir()}/test_data.jsonl"
            eff_path = getattr(state.request, "json_file", "") or default_test_file
            stats = local_tool_for_sample(_SN(json_file=eff_path), sample_size=2)
            return stats.get("samples", []) if isinstance(stats, dict) else []
        except Exception:
            return []

    @builder.pre_tool("available_keys", "llm_append_serving")
    def pre_llm_append_keys(state: DFState):
        try:
            from types import SimpleNamespace as _SN
            default_test_file = f"{DataFlowPath.get_dataflow_agent_dir()}/test_data.jsonl"
            eff_path = getattr(state.request, "json_file", "") or default_test_file
            stats = local_tool_for_sample(_SN(json_file=eff_path), sample_size=2)
            return stats.get("available_keys", []) if isinstance(stats, dict) else []
        except Exception:
            return []

    @builder.pre_tool("target", "llm_append_serving")
    def pre_llm_append_target(state: DFState):
        return getattr(state.request, "target", "")

    # ---------------- LLM前置：Instantiate ----------------
    @builder.pre_tool("pipeline_code", "llm_instantiate")
    def pre_inst_code(state: DFState):
        return state.temp_data.get("pipeline_code", "") or getattr(state, "draft_operator_code", "")

    @builder.pre_tool("target", "llm_instantiate")
    def pre_inst_target(state: DFState):
        return getattr(state.request, "target", "")

    @builder.pre_tool("example_data", "llm_instantiate")
    def pre_inst_example(state: DFState):
        try:
            from types import SimpleNamespace as _SN
            default_test_file = f"{DataFlowPath.get_dataflow_agent_dir()}/test_data.jsonl"
            eff_path = getattr(state.request, "json_file", "") or default_test_file
            stats = local_tool_for_sample(_SN(json_file=eff_path), sample_size=2)
            return stats.get("samples", []) if isinstance(stats, dict) else []
        except Exception:
            return []

    @builder.pre_tool("available_keys", "llm_instantiate")
    def pre_inst_keys(state: DFState):
        try:
            from types import SimpleNamespace as _SN
            default_test_file = f"{DataFlowPath.get_dataflow_agent_dir()}/test_data.jsonl"
            eff_path = getattr(state.request, "json_file", "") or default_test_file
            stats = local_tool_for_sample(_SN(json_file=eff_path), sample_size=2)
            return stats.get("available_keys", []) if isinstance(stats, dict) else []
        except Exception:
            return []

    @builder.pre_tool("preselected_input_key", "llm_instantiate")
    def pre_inst_preselected_key(state: DFState):
        try:
            from types import SimpleNamespace as _SN
            default_test_file = f"{DataFlowPath.get_dataflow_agent_dir()}/test_data.jsonl"
            eff_path = getattr(state.request, "json_file", "") or default_test_file
            stats = local_tool_for_sample(_SN(json_file=eff_path), sample_size=2)
            samples = stats.get("samples", []) if isinstance(stats, dict) else []
            keys = stats.get("available_keys", []) if isinstance(stats, dict) else []
            if not samples or not keys:
                return ""
            # 计算各列的平均字符串长度（基于前2条样例）
            import numpy as _np
            best_k, best_len = "", -1.0
            for k in keys:
                try:
                    vals = [str(s.get(k, "")) for s in samples]
                    avg_len = _np.mean([len(v) for v in vals]) if vals else 0.0
                except Exception:
                    avg_len = 0.0
                if avg_len > best_len:
                    best_k, best_len = k, avg_len
            return best_k
        except Exception:
            return ""

    @builder.pre_tool("test_data_path", "llm_instantiate")
    def pre_inst_test_path(state: DFState):
        try:
            default_test_file = f"{DataFlowPath.get_dataflow_agent_dir()}/test_data.jsonl"
            return getattr(state.request, 'json_file', '') or default_test_file
        except Exception:
            return ""

    # ---------------- 节点实现 ----------------
    async def match_node(s: DFState) -> DFState:
        agent = create_match()
        return await agent.execute(s, use_agent=False)

    async def write_node(s: DFState) -> DFState:
        agent = create_writer()
        return await agent.execute(s, use_agent=False)

    # 移除单纯执行器节点的硬依赖，真实测试在实例化节点完成
    async def executor_node(s: DFState) -> DFState:
        return s

    async def inject_llm_serving_node(s: DFState) -> DFState:
        from dataflow.dataflowagent.toolkits.tool_manager import get_tool_manager
        # 若代码已包含 llm_serving/APILLMServing_request，则可跳过或交给 LLM保持不变
        code_str = s.temp_data.get("pipeline_code", "") or getattr(s, "draft_operator_code", "")
        if code_str and ("self.llm_serving" in code_str or "APILLMServing_request" in code_str):
            return s
        agent = create_llm_append_serving(tool_manager=get_tool_manager(), model_name="gpt-4o")
        s2 = await agent.execute(s, use_agent=True)
        # 若 LLM 产出不可用，回退一次硬注入（保底）
        code_str2 = s2.temp_data.get("pipeline_code", "") or getattr(s2, "draft_operator_code", "")
        if not code_str2:
            # 保留原有硬注入逻辑：仅在缺失时补齐，避免重复
            try:
                # 复用原注入策略：若已有即跳过
                def _hard_inject(code: str) -> str:
                    if (not code) or ("self.llm_serving" in code) or ("APILLMServing_request" in code):
                        return code
                    return code + "\nfrom dataflow.serving import APILLMServing_request\n"
                new_code = _hard_inject(code_str or "")
                if new_code and new_code != (code_str or ""):
                    s2.temp_data["pipeline_code"] = new_code
                    s2.draft_operator_code = new_code
            except Exception:
                pass
        return s2

    async def debugger_node(s: DFState) -> DFState:
        from dataflow.dataflowagent.toolkits.tool_manager import get_tool_manager

        debugger = create_code_debugger(tool_manager=get_tool_manager())
        return await debugger.execute(s, use_agent=True)

    async def rewriter_node(s: DFState) -> DFState:
        from dataflow.dataflowagent.toolkits.tool_manager import get_tool_manager

        rewriter = create_rewriter(tool_manager=get_tool_manager(), model_name="o3")
        return await rewriter.execute(s, use_agent=True)

    def after_rewrite_node(s: DFState) -> DFState:
        from dataflow.dataflowagent.toolkits.tool_manager import get_tool_manager

        rewriter = create_rewriter(tool_manager=get_tool_manager(), model_name="o3")
        return rewriter.after_rewrite(s)

    # ---------------- 新增：实例化节点（LLM 生成可运行入口 + 执行验证） ----------------
    async def instantiate_operator_main_node(s: DFState) -> DFState:
        from dataflow.dataflowagent.toolkits.tool_manager import get_tool_manager
        try:
            agent = create_llm_instantiator(tool_manager=get_tool_manager(), model_name="gpt-4o")
            s2 = await agent.execute(s, use_agent=True)
            code_str = s2.temp_data.get("pipeline_code", "") or getattr(s2, "draft_operator_code", "")
            if not code_str:
                # 回退一次硬注入入口（保底），如果 LLM 未返回代码
                return s2

            import io, contextlib
            buf_out, buf_err = io.StringIO(), io.StringIO()
            try:
                with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
                    exec(code_str, {"__name__": "__main__"})
            except SystemExit:
                pass
            except Exception as e:
                s2.temp_data.setdefault("debug_runtime", {})
                s2.temp_data["debug_runtime"]["exec_error"] = str(e)

            out_s, err_s = buf_out.getvalue(), buf_err.getvalue()
            selected_key = None
            try:
                import re as _re
                for line in (out_s or "").splitlines():
                    m = _re.search(r"\[selected_input_key\]\s*(.+)", line)
                    if m:
                        selected_key = m.group(1).strip()
                        break
            except Exception:
                selected_key = None

            # 成功判定（若未解析到 selected_input_key，则视为失败，触发重写修复入口）
            success = False
            try:
                import pandas as _pd
                from pathlib import Path as _Path
                p = _Path("./cache_local/dataflow_cache_step_step1.jsonl")
                if p.exists():
                    df = _pd.read_json(str(p), lines=True)
                    success = (not df.empty)
            except Exception:
                success = False

            if not selected_key:
                success = False

            # 二次校验：selected_key 必须在真实 available_keys 中
            scanned_keys = []
            try:
                from types import SimpleNamespace as _SN
                from dataflow.dataflowagent.toolkits.basetool.file_tools import local_tool_for_sample as _lts
                default_test_file = f"{DataFlowPath.get_dataflow_agent_dir()}/test_data.jsonl"
                eff_path = getattr(s2.request, "json_file", "") or default_test_file
                stats = _lts(_SN(json_file=eff_path), sample_size=2)
                scanned_keys = stats.get("available_keys", []) if isinstance(stats, dict) else []
            except Exception:
                scanned_keys = []

            if selected_key and scanned_keys and (selected_key not in scanned_keys):
                success = False
            if not scanned_keys:
                success = False

            s2.temp_data.setdefault("debug_runtime", {})
            s2.temp_data["debug_runtime"].update({
                "stdout": out_s[:2000] if out_s else "",
                "stderr": err_s[:2000] if err_s else "",
                "input_key": selected_key,
                "available_keys": scanned_keys or s2.temp_data.get("available_keys", []),
                "reason": ("NO_SELECTED_INPUT_KEY" if not selected_key else ""),
            })

            s2.execution_result = {
                "success": bool(success),
                "stdout": out_s,
                "stderr": err_s or s2.temp_data.get("debug_runtime", {}).get("exec_error", ""),
                "file_path": s2.temp_data.get("pipeline_file_path", ""),
            }
            return s2
        except Exception:
            return s

    # ---------------- 条件边（复用 pipeline 的循环思路） ----------------
    def exec_condition(s: DFState):
        if s.request.need_debug:
            if s.execution_result.get("success"):
                return "__end__"
            if s.temp_data.get("round", 0) >= s.request.max_debug_rounds:
                return "__end__"
            return "code_debugger"
        else:
            return "__end__"

    nodes = {
        "match_operator": match_node,
        "write_the_operator": write_node,
        "llm_append_serving": inject_llm_serving_node,
        "llm_instantiate": instantiate_operator_main_node,
        "code_debugger": debugger_node,
        "rewriter": rewriter_node,
        "after_rewrite": after_rewrite_node,
    }
    edges = [
        ("match_operator", "write_the_operator"),
        ("write_the_operator", "llm_append_serving"),
        ("llm_append_serving", "llm_instantiate"),
        ("code_debugger", "rewriter"),
        ("rewriter", "after_rewrite"),
        ("after_rewrite", "llm_append_serving"),
        ("llm_append_serving", "llm_instantiate"),
    ]

    builder.add_nodes(nodes).add_edges(edges).add_conditional_edges({"llm_instantiate": exec_condition})
    return builder
