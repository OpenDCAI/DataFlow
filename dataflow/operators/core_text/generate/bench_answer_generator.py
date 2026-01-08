import json
import inspect
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
import pandas as pd

from dataflow import get_logger
from dataflow.core import OperatorABC, LLMServingABC
from dataflow.core.prompt import DIYPromptABC, prompt_restrict
from dataflow.prompts.core_text import FormatStrPrompt
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.storage import DataFlowStorage


@prompt_restrict(
    FormatStrPrompt
    )

@OPERATOR_REGISTRY.register()
class BenchAnswerGenerator(OperatorABC):
    """
    用于 bench 评测的统一生成算子, 与 UnifiedBenchDatasetEvaluator 参数对齐

    输入:
      - eval_type: 评测类型, 取值同 evaluator
      - 运行时通过 input_xxx_key 传入各字段名（未传默认 None）
      - input_context_key: 可选, 上下文字段名, 不传则 None
    输出:
      - output_key: 生成结果列, 默认 generated_ans
      - 对于不需要生成的类型, 默认不写 output_key, 直接返回空列表
    """

    def __init__(
        self,
        eval_type: Literal[
                "key1_text_score",
                "key2_qa",
                "key2_q_ma",
                "key3_q_choices_a",
                "key3_q_choices_as",
                "key3_q_a_rejected",
            ] = "key2_qa",
        llm_serving: LLMServingABC = None,
        prompt_template: Union[FormatStrPrompt, DIYPromptABC] = FormatStrPrompt,
        system_prompt: str = "You are a helpful assistant specialized in generating answers to questions.",
        allow_overwrite: bool = False,
        force_generate: bool = False,
    ):
        self.logger = get_logger()
        self.llm_serving = llm_serving
        self.prompt_template = prompt_template
        self.system_prompt = system_prompt
        self.allow_overwrite = allow_overwrite
        self.force_generate = force_generate
        self.eval_type = eval_type

    # ---------- 工具函数 ----------
    def _normalize_context(self, ctx: Any) -> Optional[str]:
        if ctx is None:
            return None
        if isinstance(ctx, float) and np.isnan(ctx):
            return None
        if isinstance(ctx, list):
            parts = []
            for x in ctx:
                if x is None:
                    continue
                s = str(x).strip()
                if s:
                    parts.append(s)
            return "\n".join(parts) if parts else None
        s = str(ctx).strip()
        return s if s else None

    def _ensure_list(self, v: Any) -> Optional[List[str]]:
        if v is None:
            return None
        if isinstance(v, float) and np.isnan(v):
            return None
        if isinstance(v, list):
            return [str(x) for x in v]
        s = str(v).strip()
        if not s:
            return None
        # 尝试 json list
        if s.startswith("[") and s.endswith("]"):
            try:
                obj = json.loads(s)
                if isinstance(obj, list):
                    return [str(x) for x in obj]
            except Exception:
                pass
        return None

    def _format_choices_text(self, choices: List[str]) -> str:
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        lines = []
        for i, c in enumerate(choices):
            tag = letters[i] if i < len(letters) else str(i)
            lines.append(f"{tag}. {c}")
        return "\n".join(lines)

    def _build_prompt_fallback(
        self,
        *,
        eval_type: str,
        question: Optional[str],
        context: Optional[str],
        choices: Optional[List[str]],
    ) -> str:
        ctx_block = f"Context:\n{context}\n\n" if context else ""
        q_block = f"Question:\n{(question or '').strip()}\n\n"

        if eval_type in ("key2_qa", "key2_q_ma"):
            return f"{ctx_block}{q_block}Answer:"
        if eval_type == "key3_q_choices_a":
            ch = self._format_choices_text(choices or [])
            return f"{ctx_block}{q_block}Choices:\n{ch}\n\nChoose exactly one option. Output only the option letter (e.g., A).\nAnswer:"
        if eval_type == "key3_q_choices_as":
            ch = self._format_choices_text(choices or [])
            return (
                f"{ctx_block}{q_block}Choices:\n{ch}\n\n"
                "This is a multi-select question. Output JSON only, format: {\"choices\": [\"A\",\"C\"]}.\nAnswer:"
            )
        # key1_text_score / key3_q_a_rejected 默认不需要生成
        return f"{ctx_block}{q_block}Answer:"

    def _build_prompt(
        self,
        *,
        eval_type: str,
        question: Optional[str],
        context: Optional[str],
        choices: Optional[List[str]],
    ) -> str:
        if self.prompt_template is not None and hasattr(self.prompt_template, "build_prompt"):
            try:
                fn = getattr(self.prompt_template, "build_prompt")

                if eval_type in ("key3_q_choices_a", "key3_q_choices_as"):
                    need_fields = {"question", "choices"}
                else:
                    need_fields = {"question"}

                kwargs = {
                    "eval_type": eval_type,
                    "question": question,
                    "context": context or "",
                    "choices": choices,
                    "choices_text": self._format_choices_text(choices) if choices else "",
                }

                sig = inspect.signature(fn)
                params = list(sig.parameters.values())
                has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params)

                accepted = {p.name for p in params if p.name != "self"}
                expects_need_fields = "need_fields" in accepted

                if has_varkw:
                    if expects_need_fields:
                        return fn(need_fields, **kwargs)
                    return fn(**kwargs)

                filtered = {k: v for k, v in kwargs.items() if k in accepted}
                if expects_need_fields:
                    return fn(need_fields, **filtered)
                return fn(**filtered)
            except Exception as e:
                self.logger.error(f"prompt_template.build_prompt 失败, fallback 默认模板: {e}")
        return self._build_prompt_fallback(eval_type=eval_type, question=question, context=context, choices=choices)

    def _call_generate(self, prompts: List[str]) -> List[str]:
        if not hasattr(self.llm_serving, "generate_from_input"):
            self.logger.error("llm_serving 缺少 generate_from_input 接口")
            return [""] * len(prompts)
        try:
            # 兼容有无 system_prompt 参数
            try:
                return self.llm_serving.generate_from_input(user_inputs=prompts, system_prompt=self.system_prompt)
            except TypeError:
                return self.llm_serving.generate_from_input(prompts)
        except Exception as e:
            self.logger.error(f"generate_from_input 执行失败: {e}")
            return [""] * len(prompts)

    def _need_generation(self, eval_type: str) -> bool:
        # evaluator 当前实现里:
        # - key1_text_score: 不需要 generated_ans
        # - key2_qa / key2_q_ma: 需要 generated_ans
        # - key3_q_choices_a: evaluator 可用 ll 做选择题评估 -> 默认不生成
        # - key3_q_choices_as: evaluator 当前用解析 generated_ans -> 需要
        # - key3_q_a_rejected: evaluator 用 ll 比较 better vs rejected -> 不需要
        if self.force_generate:
            return eval_type != "key1_text_score"
        return eval_type in ("key2_qa", "key2_q_ma", "key3_q_choices_as")

    # ---------- 主入口 ----------
    def run(
        self,
        storage: DataFlowStorage,
        input_text_key: Optional[str] = None,
        input_question_key: Optional[str] = None,
        input_target_key: Optional[str] = None,
        input_targets_key: Optional[str] = None,
        input_choices_key: Optional[str] = None,
        input_label_key: Optional[str] = None,
        input_labels_key: Optional[str] = None,
        input_better_key: Optional[str] = None,
        input_rejected_key: Optional[str] = None,
        input_context_key: Optional[str] = None,
        output_key: str = "generated_ans",
    ) -> List[str]:

        df = storage.read("dataframe")
        eval_type = self.eval_type

        if not self._need_generation(eval_type):
            self.logger.info(f"[BenchAnswerGenerator] eval_type={eval_type} 默认不需要生成, 跳过")
            storage.write(df)
            return []

        if (output_key in df.columns) and (not self.allow_overwrite):
            self.logger.error(f"输出列已存在且不允许覆盖: {output_key}")
            storage.write(df)
            return []

        # 读取字段
        q_col = input_question_key
        if not q_col or q_col not in df.columns:
            self.logger.error(f"缺少 question 列, input_question_key={q_col}")
            storage.write(df)
            return []

        ch_col = input_choices_key
        need_choices = eval_type in ("key3_q_choices_a", "key3_q_choices_as")
        if need_choices and (not ch_col or ch_col not in df.columns):
            self.logger.error(f"缺少 choices 列, input_choices_key={ch_col}")
            storage.write(df)
            return []

        ctx_series = None
        if input_context_key:
            if input_context_key in df.columns:
                ctx_series = df[input_context_key]
            else:
                self.logger.error(f"context_key 不存在: {input_context_key}, 视为 None")

        prompts: List[str] = []
        for idx, row in df.iterrows():
            q = row[q_col]
            ctx = self._normalize_context(ctx_series.loc[idx]) if ctx_series is not None else None

            choices = None
            if need_choices:
                choices = self._ensure_list(row[ch_col])
                if not choices:
                    # choices 为空, 仍然生成一个可追踪的输出, 避免整体崩
                    choices = [""]

            prompts.append(
                self._build_prompt(
                    eval_type=eval_type,
                    question=str(q) if q is not None else "",
                    context=ctx,
                    choices=choices,
                )
            )

        answers = self._call_generate(prompts)
        df[output_key] = answers
        out_file = storage.write(df)
        self.logger.info(f"[BenchAnswerGenerator] 生成完成, 保存到 {out_file}")
        return [output_key]

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "该算子用于 bench 评测的统一答案生成，根据 eval_type + keys_map 从 DataFrame 取字段构造 prompt 并批量调用 LLM 生成答案。\n"
                "对于默认不需要生成的类型会跳过生成（可用 force_generate 强制）。\n\n"
                "初始化参数：\n"
                "- eval_type：评测类型（key1_text_score / key2_qa / key2_q_ma / key3_q_choices_a / key3_q_choices_as / key3_q_a_rejected）\n"
                "- llm_serving：LLM 服务对象（需提供 generate_from_input）\n"
                "- prompt_template：提示模板对象（可选，需提供 build_prompt；否则使用内置 fallback 模板）\n"
                "- system_prompt：系统提示词\n"
                "- allow_overwrite：输出列已存在时是否允许覆盖\n"
                "- force_generate：是否强制对可生成类型都生成\n\n"
                "运行参数：\n"
                "- storage：DataFlowStorage\n"
                "- input_text_key：文本列名（key1_text_score）\n"
                "- input_question_key：问题列名（key2/key3）\n"
                "- input_target_key：单个参考答案列名（key2_qa）\n"
                "- input_targets_key：多个参考答案列名（key2_q_ma）\n"
                "- input_choices_key：选项列名（key3_q_choices_a/key3_q_choices_as）\n"
                "- input_label_key：单个标签列名（key3_q_choices_a）\n"
                "- input_labels_key：多个标签列名（key3_q_choices_as）\n"
                "- input_better_key：优选答案列名（key3_q_a_rejected）\n"
                "- input_rejected_key：劣选答案列名（key3_q_a_rejected）\n"
                "- input_context_key：可选，上下文字段名\n"
                "- output_key：生成结果列名（默认 generated_ans）\n\n"
                "输出：\n"
                "- 写回 DataFrame 的 output_key 列（若跳过生成则不写）\n"
                "- 返回新增/写入的列名列表（通常为 [output_key] 或 []）"
            )
        return (
            "This operator generates answers for unified bench evaluation by building prompts from a dataframe and calling an LLM.\n\n"
            "Input Parameters:\n"
            "- eval_type: Evaluation type (key1_text_score/key2_qa/key2_q_ma/key3_q_choices_a/key3_q_choices_as/key3_q_a_rejected)\n"
            "- llm_serving: LLM serving object (must provide generate_from_input)\n"
            "- prompt_template: Prompt template object (optional; must provide build_prompt; falls back to an internal template)\n"
            "- system_prompt: System prompt passed to the serving (if supported)\n"
            "- allow_overwrite: Whether to overwrite an existing output column\n"
            "- force_generate: Whether to force generation for types that can be skipped by default\n\n"
            "Run Parameters:\n"
            "- storage: DataFlowStorage\n"
            "- input_text_key: Text column name (key1_text_score)\n"
            "- input_question_key: Question column name (key2/key3)\n"
            "- input_target_key: Single reference answer column name (key2_qa)\n"
            "- input_targets_key: Multiple reference answers column name (key2_q_ma)\n"
            "- input_choices_key: Choices column name (key3_q_choices_a/key3_q_choices_as)\n"
            "- input_label_key: Single label column name (key3_q_choices_a)\n"
            "- input_labels_key: Multiple labels column name (key3_q_choices_as)\n"
            "- input_better_key: Better answer column name (key3_q_a_rejected)\n"
            "- input_rejected_key: Rejected answer column name (key3_q_a_rejected)\n"
            "- input_context_key: Optional context column name\n"
            "- output_key: Output column name for generated answers (default: generated_ans)\n\n"
            "Output Parameters:\n"
            "- Writes output_key into the dataframe when generation is performed\n"
            "- Returns a list of written keys (usually [output_key] or [])"
        )
