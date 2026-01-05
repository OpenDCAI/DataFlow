from __future__ import annotations

import json
import inspect
import re
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
import pandas as pd

from dataflow import get_logger
from dataflow.core import OperatorABC, LLMServingABC
from dataflow.core.prompt import DIYPromptABC, prompt_restrict
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.storage import DataFlowStorage


@prompt_restrict()  # 保持通用, 不强绑固定 prompt 类

@OPERATOR_REGISTRY.register()
class BenchAnswerGenerator(OperatorABC):
    """
    用于 bench 评测的统一生成算子, 与 UnifiedBenchDatasetEvaluator 参数对齐

    输入:
      - eval_type: 评测类型, 取值同 evaluator
      - keys_map: 指定各字段名, 同 evaluator
      - context_key: 可选, 上下文字段名, 不传则 None
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
        llm_serving: Optional[LLMServingABC] = None,
        prompt_template: Optional[Union[DIYPromptABC, Any]] = None,
        system_prompt: str = "You are a helpful assistant specialized in generating answers to questions.",
        allow_overwrite: bool = False,
        # 是否强制对所有类型都生成, 默认只对需要 pred 的类型生成
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
                kwargs = {
                    "eval_type": eval_type,
                    "question": question,
                    "context": context,
                    "choices": choices,
                    "choices_text": self._format_choices_text(choices) if choices else None,
                }
                kwargs = {k: v for k, v in kwargs.items() if v is not None}

                sig = inspect.signature(fn)
                params = sig.parameters.values()
                has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params)
                if has_varkw:
                    return fn(**kwargs)

                accepted = {p.name for p in params if p.name != "self"}
                filtered = {k: v for k, v in kwargs.items() if k in accepted}
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
        keys_map: Dict[str, str],
        context_key: Optional[str] = None,
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
        q_col = keys_map.get("question")
        if not q_col or q_col not in df.columns:
            self.logger.error(f"缺少 question 列, keys_map.question={q_col}")
            storage.write(df)
            return []

        ch_col = keys_map.get("choices")
        need_choices = eval_type in ("key3_q_choices_a", "key3_q_choices_as")
        if need_choices and (not ch_col or ch_col not in df.columns):
            self.logger.error(f"缺少 choices 列, keys_map.choices={ch_col}")
            storage.write(df)
            return []

        ctx_series = None
        if context_key:
            if context_key in df.columns:
                ctx_series = df[context_key]
            else:
                self.logger.error(f"context_key 不存在: {context_key}, 视为 None")

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
                "用于 bench 评测的统一生成算子, 与 evaluator 的 eval_type + keys_map 对齐。\n"
                "默认只对需要生成输出的类型生成 output_key=generated_ans, 并支持 context_key 作为可选上下文。\n"
                "可通过 allow_overwrite 控制是否覆盖已存在的输出列。"
            )
        return "Unified bench answer generator aligned with evaluator eval_type and keys_map."
