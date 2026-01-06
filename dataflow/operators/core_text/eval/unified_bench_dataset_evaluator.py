import json
import os
import re
import time
import unicodedata
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
import numpy as np
import pandas as pd
from math_verify import parse, verify
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.core.prompt import DIYPromptABC
from dataflow.core.prompt import prompt_restrict
from dataflow.prompts.model_evaluation.general import AnswerJudgePrompt
from dataflow.core import LLMServingABC
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.storage import DataFlowStorage
from dataflow.utils.reasoning.AnswerExtraction import StringCleaner, UnitTextManager, AnswerExtractor


@prompt_restrict(
    AnswerJudgePrompt
)

@OPERATOR_REGISTRY.register()
class UnifiedBenchDatasetEvaluator(OperatorABC):
    """
    统一 Bench 评测算子：支持 6 类 keys-type + metric。

    评测类型 (bench_dataflow_eval_type): (详见doc)
      - key1_text_score
      - key2_qa
      - key2_q_ma
      - key3_q_choices_a
      - key3_q_choices_as
      - key3_q_a_rejected

    核心思想：
      只需要传 bench_dataflow_eval_type + metric_type + keys_map + (可选) context_key
      - evaluator 内部负责：
          1) 读取 dataframe
          2) 取 keys
          3) 组装 prompt（用 prompt_template 或默认模板）
          4) 计算 metric
          5) 写回结果列 + 统计落盘
    """

    # -----------------------------
    # 构造
    # -----------------------------
    def __init__(
        self,
        eval_result_path: Optional[str] = None,
        eval_type: Literal[
                "key1_text_score",
                "key2_qa",
                "key2_q_ma",
                "key3_q_choices_a",
                "key3_q_choices_as",
                "key3_q_a_rejected",
            ] = "key2_qa",
        llm_serving: Optional[LLMServingABC] = None,
        prompt_template: Union[AnswerJudgePrompt, DIYPromptABC] = AnswerJudgePrompt,
        system_prompt: str = "You are a helpful assistant specialized in evaluating answer correctness.",
        metric_type: Optional[str] = None,
        use_semantic_judge: bool = False,
    ):
        if eval_result_path is None:
            timestamp = int(time.time())
            eval_result_path = f"result_bencheval/UnifiedBenchDatasetEvaluator_result_{timestamp}.json"

        self.eval_result_path = eval_result_path
        self.eval_type = eval_type
        self.llm_serving = llm_serving
        self.prompt_template = prompt_template
        self.system_prompt = system_prompt
        self.metric_type = metric_type
        self.use_semantic_judge = use_semantic_judge

        unit_manager = UnitTextManager()
        string_cleaner = StringCleaner(unit_manager)
        self.answer_extractor = AnswerExtractor(string_cleaner)

        self.logger = get_logger()
        self.empty_responses_count = 0

    # -----------------------------
    # 工具函数：列检查
    # -----------------------------
    def _check_columns(self, dataframe: pd.DataFrame, cols: List[str]) -> bool:
        ok = True
        for c in cols:
            if c not in dataframe.columns:
                self.logger.error(f"Required column '{c}' not found in dataframe")
                ok = False
        return ok

    # -----------------------------
    # 工具函数：context 统一拼接
    # -----------------------------
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

    # -----------------------------
    # 工具函数：默认 prompt（当 prompt_template 不存在或 build_prompt 不可用）
    # -----------------------------
    def _default_prompt(
        self,
        *,
        question: Optional[str] = None,
        context: Optional[str] = None,
        text: Optional[str] = None,
        choices: Optional[List[str]] = None,
        task: str = "",
    ) -> str:
        if task == "text_score":
            return (text or "").strip()

        ctx_block = f"Context:\n{context}\n\n" if context else ""
        q_block = f"Question:\n{(question or '').strip()}\n\n"

        if choices is not None:
            # 标准化成 A./B./C. 格式，便于模板替换 & 也便于 fallback 解析
            letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            formatted = []
            for i, ch in enumerate(choices):
                tag = letters[i] if i < len(letters) else str(i)
                formatted.append(f"{tag}. {str(ch)}")
            choices_block = "Choices:\n" + "\n".join(formatted) + "\n\n"
            return f"{ctx_block}{q_block}{choices_block}Answer:"
        else:
            return f"{ctx_block}{q_block}Answer:"

    def _build_prompt(
        self,
        *,
        question: Optional[str] = None,
        context: Optional[str] = None,
        text: Optional[str] = None,
        choices: Optional[List[str]] = None,
        task: str = "",
    ) -> str:
        # 兼容你的 prompt_template（通常有 build_prompt）
        if self.prompt_template is not None and hasattr(self.prompt_template, "build_prompt"):
            try:
                # 给模板更丰富的变量，模板不用可以忽略
                choices_text = None
                if choices is not None:
                    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                    formatted = []
                    for i, ch in enumerate(choices):
                        tag = letters[i] if i < len(letters) else str(i)
                        formatted.append(f"{tag}. {str(ch)}")
                    choices_text = "\n".join(formatted)

                return self.prompt_template.build_prompt(
                    question=question,
                    context=context,
                    text=text,
                    choices=choices,
                    choices_text=choices_text,
                    task=task,
                )
            except Exception as e:
                self.logger.error(f"prompt_template.build_prompt failed, fallback to default. err={e}")

        return self._default_prompt(question=question, context=context, text=text, choices=choices, task=task)

    # -----------------------------
    # math_verify compare
    # -----------------------------
    def _try_math_verify_compare(self, answer: Any, ground_truth: Any) -> Optional[bool]:
        try:
            return verify(parse(str(ground_truth)), parse(str(answer)))
        except Exception:
            try:
                return verify(parse(ground_truth), parse(answer))
            except Exception:
                return None

    def _math_verify_compare(self, answer: Any, ground_truth: Any) -> bool:
        res = self._try_math_verify_compare(answer, ground_truth)
        return bool(res) if res is not None else False

    def _normalize_text_for_match(self, text: Any) -> str:
        if text is None:
            return ""
        s = unicodedata.normalize("NFKC", str(text))
        s = s.translate(str.maketrans({
            "₀": "0",
            "₁": "1",
            "₂": "2",
            "₃": "3",
            "₄": "4",
            "₅": "5",
            "₆": "6",
            "₇": "7",
            "₈": "8",
            "₉": "9",
        }))
        s = s.strip()
        s = re.sub(r"\s+", " ", s)
        if s.endswith((".", "。", "!", "！", "?", "？")):
            s = s[:-1].strip()
        return s.casefold()

    def _text_contains_match(self, pred: Any, ref: Any) -> bool:
        p = self._normalize_text_for_match(pred)
        r = self._normalize_text_for_match(ref)
        if not p or not r:
            return False
        return (r in p) or (p in r)

    # -----------------------------
    # 多参考答案：把 targets 解析成 List[str]
    # -----------------------------
    def _normalize_targets(self, targets: Any) -> List[str]:
        if targets is None:
            return []
        if isinstance(targets, float) and np.isnan(targets):
            return []
        if isinstance(targets, list):
            return [str(x) for x in targets if str(x).strip()]

        s = str(targets).strip()
        if not s:
            return []

        # 尝试 json list
        if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")):
            try:
                obj = json.loads(s)
                if isinstance(obj, list):
                    return [str(x) for x in obj if str(x).strip()]
            except Exception:
                pass

        # 常见分隔
        if "||" in s:
            parts = [p.strip() for p in s.split("||")]
        elif "|" in s:
            parts = [p.strip() for p in s.split("|")]
        elif ";" in s:
            parts = [p.strip() for p in s.split(";")]
        else:
            parts = [s]
        return [p for p in parts if p]

    # -----------------------------
    # choice 解析（fallback 用）
    # -----------------------------
    def _parse_choice_from_text(self, text: str, num_choices: int) -> Optional[int]:
        if text is None:
            return None
        t = str(text).strip()
        if not t:
            return None

        # 先找 A/B/C...
        m = re.search(r"\b([A-Za-z])\b", t)
        if m:
            idx = ord(m.group(1).upper()) - ord("A")
            if 0 <= idx < num_choices:
                return idx

        # 再找数字（1-based 或 0-based 都兼容）
        m = re.search(r"\b(\d+)\b", t)
        if m:
            val = int(m.group(1))
            if 0 <= val < num_choices:
                return val
            if 1 <= val <= num_choices:
                return val - 1

        return None

    def _parse_multiselect_set(self, text: str, num_choices: int) -> Optional[set]:
        if text is None:
            return None
        s = str(text).strip()
        if not s:
            return None

        # json list
        if s.startswith("[") and s.endswith("]"):
            try:
                obj = json.loads(s)
                if isinstance(obj, list):
                    res = set()
                    for x in obj:
                        if isinstance(x, str):
                            x = x.strip()
                            if len(x) == 1 and x.isalpha():
                                idx = ord(x.upper()) - ord("A")
                                if 0 <= idx < num_choices:
                                    res.add(idx)
                            elif x.isdigit():
                                v = int(x)
                                if 0 <= v < num_choices:
                                    res.add(v)
                                elif 1 <= v <= num_choices:
                                    res.add(v - 1)
                        elif isinstance(x, int):
                            if 0 <= x < num_choices:
                                res.add(x)
                            elif 1 <= x <= num_choices:
                                res.add(x - 1)
                    return res
            except Exception:
                pass

        # 字母集合：如 "A,C,D" / "B D"
        letters = re.findall(r"\b([A-Za-z])\b", s)
        if letters:
            res = set()
            for ch in letters:
                idx = ord(ch.upper()) - ord("A")
                if 0 <= idx < num_choices:
                    res.add(idx)
            return res if res else None

        # 数字集合：如 "1,3,4"
        nums = re.findall(r"\b(\d+)\b", s)
        if nums:
            res = set()
            for n in nums:
                v = int(n)
                if 0 <= v < num_choices:
                    res.add(v)
                elif 1 <= v <= num_choices:
                    res.add(v - 1)
            return res if res else None

        return None

    # -----------------------------
    # micro-F1 / Jaccard
    # -----------------------------
    def _set_metrics(self, pred: set, gold: set) -> Dict[str, float]:
        if pred is None or gold is None:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "jaccard": 0.0, "exact_set": 0.0}
        inter = len(pred & gold)
        p = inter / len(pred) if len(pred) > 0 else 0.0
        r = inter / len(gold) if len(gold) > 0 else 0.0
        f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
        j = inter / len(pred | gold) if len(pred | gold) > 0 else 0.0
        exact = 1.0 if pred == gold else 0.0
        return {"precision": float(p), "recall": float(r), "f1": float(f1), "jaccard": float(j), "exact_set": float(exact)}

    # -----------------------------
    # LLM loglikelihood 适配（尽量兼容不同 serving 实现）
    # -----------------------------
    def _ll_batch(self, prompts: List[str], continuations: List[str]) -> Optional[List[float]]:
        if self.llm_serving is None:
            return None

        # 尝试常见方法名
        cand_names = [
            "loglikelihood_batch",
            "loglikelihood",
            "get_loglikelihood_batch",
            "get_loglikelihood",
            "score_batch",
            "score",
        ]
        for name in cand_names:
            if hasattr(self.llm_serving, name):
                fn = getattr(self.llm_serving, name)
                try:
                    # 兼容多种签名： (prompts, continuations) / (pairs)
                    try:
                        return fn(prompts=prompts, continuations=continuations)  # type: ignore
                    except TypeError:
                        try:
                            return fn(prompts, continuations)  # type: ignore
                        except TypeError:
                            pairs = list(zip(prompts, continuations))
                            return fn(pairs)  # type: ignore
                except Exception as e:
                    self.logger.error(f"llm_serving.{name} failed: {e}")
                    return None

        model_id = getattr(self.llm_serving, "real_model_path", None) or getattr(self.llm_serving, "hf_model_name_or_path", None)
        hf_cache_dir = getattr(self.llm_serving, "hf_cache_dir", None)
        trust_remote_code = getattr(self.llm_serving, "trust_remote_code", True)

        if model_id is None:
            self.logger.error("llm_serving does not expose real_model_path/hf_model_name_or_path; cannot compute loglikelihood.")
            return None

        try:
            tokenizer = getattr(self, "_ll_hf_tokenizer", None)
            model = getattr(self, "_ll_hf_model", None)
            loaded_id = getattr(self, "_ll_hf_model_id", None)
            if tokenizer is None or model is None or loaded_id != model_id:
                tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=hf_cache_dir, trust_remote_code=trust_remote_code)
                model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=hf_cache_dir, trust_remote_code=trust_remote_code)
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model.to(device)
                model.eval()
                self._ll_hf_tokenizer = tokenizer
                self._ll_hf_model = model
                self._ll_hf_model_id = model_id
        except Exception as e:
            self.logger.error(f"failed to load hf model/tokenizer for loglikelihood: {e}")
            return None

        try:
            device = next(model.parameters()).device
            pad_id = tokenizer.pad_token_id
            if pad_id is None:
                pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

            batch_size = 4
            lls: List[float] = []

            def _safe_ids(text: str) -> List[int]:
                return tokenizer(text, add_special_tokens=False).input_ids

            for start in range(0, len(prompts), batch_size):
                ps = ["" if p is None else str(p) for p in prompts[start:start + batch_size]]
                cs = ["" if c is None else str(c) for c in continuations[start:start + batch_size]]

                full_ids_list: List[List[int]] = []
                prompt_lens: List[int] = []
                cont_lens: List[int] = []

                for p, c in zip(ps, cs):
                    full_ids = _safe_ids(p + c)
                    p_ids = _safe_ids(p)
                    if len(p_ids) <= len(full_ids) and full_ids[:len(p_ids)] == p_ids:
                        prompt_len = len(p_ids)
                    else:
                        c_ids = _safe_ids(c)
                        prompt_len = max(0, len(full_ids) - len(c_ids))
                    cont_len = max(0, len(full_ids) - prompt_len)
                    full_ids_list.append(full_ids)
                    prompt_lens.append(prompt_len)
                    cont_lens.append(cont_len)

                max_len = max((len(x) for x in full_ids_list), default=0)
                if max_len == 0:
                    lls.extend([0.0] * len(full_ids_list))
                    continue

                input_ids = torch.full((len(full_ids_list), max_len), pad_id, dtype=torch.long, device=device)
                attention_mask = torch.zeros((len(full_ids_list), max_len), dtype=torch.long, device=device)
                for i, ids in enumerate(full_ids_list):
                    if not ids:
                        continue
                    t = torch.tensor(ids, dtype=torch.long, device=device)
                    input_ids[i, : t.numel()] = t
                    attention_mask[i, : t.numel()] = 1

                with torch.no_grad():
                    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                    log_probs = F.log_softmax(logits, dim=-1)

                shift_log_probs = log_probs[:, :-1, :].contiguous()
                shift_labels = input_ids[:, 1:].contiguous()
                token_ll = shift_log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)

                for i in range(len(full_ids_list)):
                    cont_len = cont_lens[i]
                    prompt_len = prompt_lens[i]
                    if cont_len <= 0:
                        lls.append(0.0)
                        continue
                    start_pos = max(prompt_len, 1)
                    end_pos = prompt_len + cont_len
                    start_idx = start_pos - 1
                    end_idx = end_pos - 1
                    if end_idx <= start_idx:
                        lls.append(0.0)
                        continue
                    ll_val = float(token_ll[i, start_idx:end_idx].sum().detach().cpu())
                    lls.append(ll_val)

            return lls
        except Exception as e:
            self.logger.error(f"hf loglikelihood computation failed: {e}")
            return None

    def _ppl_batch(self, texts: List[str]) -> Optional[List[float]]:
        if self.llm_serving is None:
            return None

        model_id = getattr(self.llm_serving, "real_model_path", None) or getattr(self.llm_serving, "hf_model_name_or_path", None)
        hf_cache_dir = getattr(self.llm_serving, "hf_cache_dir", None)
        trust_remote_code = getattr(self.llm_serving, "trust_remote_code", True)

        if model_id is None:
            self.logger.error("llm_serving does not expose real_model_path/hf_model_name_or_path; cannot compute ppl.")
            return None

        try:
            tokenizer = getattr(self, "_ppl_hf_tokenizer", None)
            model = getattr(self, "_ppl_hf_model", None)
            loaded_id = getattr(self, "_ppl_hf_model_id", None)
            if tokenizer is None or model is None or loaded_id != model_id:
                tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=hf_cache_dir, trust_remote_code=trust_remote_code)
                model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=hf_cache_dir, trust_remote_code=trust_remote_code)
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model.to(device)
                model.eval()
                self._ppl_hf_tokenizer = tokenizer
                self._ppl_hf_model = model
                self._ppl_hf_model_id = model_id
        except Exception as e:
            self.logger.error(f"failed to load hf model/tokenizer for ppl: {e}")
            return None

        try:
            device = next(model.parameters()).device
            batch_size = 4
            ppls: List[float] = []
            max_len = getattr(getattr(model, "config", None), "max_position_embeddings", None)

            for start in range(0, len(texts), batch_size):
                batch_texts = ["" if t is None else str(t) for t in texts[start:start + batch_size]]
                enc = tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_len,
                )
                input_ids = enc["input_ids"].to(device)
                attention_mask = enc.get("attention_mask", None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)

                with torch.no_grad():
                    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = input_ids[:, 1:].contiguous()

                if attention_mask is None:
                    shift_mask = torch.ones_like(shift_labels, dtype=torch.float32, device=device)
                else:
                    shift_mask = attention_mask[:, 1:].to(dtype=torch.float32)

                vocab_size = shift_logits.size(-1)
                token_nll = F.cross_entropy(
                    shift_logits.view(-1, vocab_size),
                    shift_labels.view(-1),
                    reduction="none",
                ).view(shift_labels.size(0), -1)

                nll_sum = (token_nll * shift_mask).sum(dim=1)
                denom = shift_mask.sum(dim=1).clamp_min(1.0)
                ppl_batch = torch.exp(nll_sum / denom).detach().cpu().tolist()
                ppls.extend([float(x) for x in ppl_batch])

            return ppls
        except Exception as e:
            self.logger.error(f"hf ppl computation failed: {e}")
            return None

    # -----------------------------
    # 统计落盘
    # -----------------------------
    def _save_stats(self, bench_name_or_prefix: str, stats: Dict[str, Any]) -> None:
        os.makedirs(os.path.dirname(self.eval_result_path), exist_ok=True)
        df = pd.DataFrame([stats])
        df.to_json(self.eval_result_path, orient="records", force_ascii=False, indent=2)
        self.logger.success(f"Statistics saved to {self.eval_result_path}")

    # -----------------------------
    # 主入口
    # -----------------------------
    def run(
        self,
        storage: DataFlowStorage,
        input_keys_map: Optional[Dict[str, str]] = None,
        input_context_key: Optional[str] = None,
        input_pred_key: str = "generated_ans",
        output_eval_valid_key: str = "eval_valid",
        output_eval_error_key: str = "eval_error",
        output_eval_pred_key: str = "eval_pred",
        output_eval_score_key: str = "eval_score",
    ) -> List[str]:
        """
        keys_map 示例：
          - key1_text_score: {"text": "text"}
          - key2_qa: {"question":"question", "target":"golden_answer"}
          - key2_q_ma: {"question":"question", "targets":"gold_answers"}
          - key3_q_choices_a: {"question":"question", "choices":"choices", "label":"label"}
          - key3_q_choices_as: {"question":"question", "choices":"choices", "labels":"labels"}
          - key3_q_a_rejected: {"question":"question", "better":"chosen", "rejected":"rejected"}
        """
        df = storage.read("dataframe")
        eval_type = self.eval_type

        # 输出列统一
        if output_eval_valid_key not in df.columns:
            df[output_eval_valid_key] = True
        df[output_eval_error_key] = ""
        df[output_eval_pred_key] = None
        df[output_eval_score_key] = np.nan  # 数值型评分（accuracy 类用 0/1）

        # 默认 metric
        metric_type = self.metric_type
        if metric_type is None:
            metric_type = self._default_metric_for_type(eval_type, self.use_semantic_judge)

        if input_keys_map is None:
            self.logger.error("keys_map is required.")
            storage.write(df)
            return [output_eval_valid_key, output_eval_error_key, output_eval_pred_key, output_eval_score_key]

        # context 处理：统一读一列（可无）
        ctx_series = None
        if input_context_key is not None:
            if input_context_key not in df.columns:
                self.logger.error(f"context_key '{input_context_key}' not found; treat as None.")
            else:
                ctx_series = df[input_context_key]

        # 分发
        if eval_type == "key1_text_score":
            required = [input_keys_map.get("text", "")]
            if not self._check_columns(df, required):
                storage.write(df)
                return required

            text_col = input_keys_map["text"]
            texts = [str(x) if x is not None else "" for x in df[text_col].tolist()]
            ppl = self._ppl_batch(texts)
            if ppl is None:
                df[output_eval_valid_key] = False
                df[output_eval_error_key] = "ppl_unavailable"
                storage.write(df)
                self._save_stats(storage.file_name_prefix, {
                    "bench_name_or_prefix": storage.file_name_prefix,
                    "type": eval_type,
                    "metric": metric_type,
                    "total_samples": len(df),
                    "valid_samples": 0,
                    "note": "ppl unavailable in llm_serving",
                })
                return [text_col, output_eval_score_key, output_eval_valid_key, output_eval_error_key]

            df[output_eval_score_key] = ppl
            df[output_eval_pred_key] = None
            df[output_eval_valid_key] = True
            storage.write(df)

            stats = {
                "bench_name_or_prefix": storage.file_name_prefix,
                "type": eval_type,
                "metric": metric_type,
                "total_samples": int(len(df)),
                "valid_samples": int(len(df)),
                "ppl_mean": float(np.mean(ppl)) if len(ppl) else 0.0,
            }
            self._save_stats(storage.file_name_prefix, stats)
            return [text_col, output_eval_score_key, output_eval_valid_key, output_eval_error_key]

        elif eval_type in ("key2_qa", "key2_q_ma"):
            # QA：默认走 math_verify 抽取+对比（可选 semantic_judge）
            # 单参考：target
            # 多参考：targets
            question_col = input_keys_map.get("question", "")
            if eval_type == "key2_qa":
                target_col = input_keys_map.get("target", "")
                required = [question_col, target_col, input_pred_key]
                if not self._check_columns(df, required):
                    storage.write(df)
                    return required

                self._eval_qa_single(
                    df=df,
                    question_col=question_col,
                    target_col=target_col,
                    pred_col=input_pred_key,
                    ctx_series=ctx_series,
                    metric_type=metric_type,
                )
                storage.write(df)

                stats = self._stats_for_binary(df)
                stats.update({
                    "bench_name_or_prefix": storage.file_name_prefix,
                    "type": eval_type,
                    "metric": metric_type,
                })
                self._save_stats(storage.file_name_prefix, stats)
                return [question_col, target_col, input_pred_key, output_eval_score_key, output_eval_valid_key, output_eval_error_key]

            else:
                targets_col = input_keys_map.get("targets", "")
                required = [question_col, targets_col, input_pred_key]
                if not self._check_columns(df, required):
                    storage.write(df)
                    return required

                self._eval_qa_multi(
                    df=df,
                    question_col=question_col,
                    targets_col=targets_col,
                    pred_col=input_pred_key,
                    ctx_series=ctx_series,
                    metric_type=metric_type,
                )
                storage.write(df)

                stats = self._stats_for_binary(df)
                stats.update({
                    "bench_name_or_prefix": storage.file_name_prefix,
                    "type": eval_type,
                    "metric": metric_type,
                })
                self._save_stats(storage.file_name_prefix, stats)
                return [question_col, targets_col, input_pred_key, output_eval_score_key, output_eval_valid_key, output_eval_error_key] 

        elif eval_type == "key3_q_choices_a":
            question_col = input_keys_map.get("question", "")
            choices_col = input_keys_map.get("choices", "")
            label_col = input_keys_map.get("label", "")
            required = [question_col, choices_col, label_col]
            # 若没有 llm_serving，则 fallback 需要 pred_col
            if self.llm_serving is None:
                required.append(input_pred_key)

            if not self._check_columns(df, required):
                storage.write(df)
                return required

            self._eval_mc_single(
                df=df,
                question_col=question_col,
                choices_col=choices_col,
                label_col=label_col,
                ctx_series=ctx_series,
                metric_type=metric_type,
                pred_col=input_pred_key,
            )
            storage.write(df)

            stats = self._stats_for_binary(df)
            stats.update({
                "bench_name_or_prefix": storage.file_name_prefix,
                "type": eval_type,
                "metric": metric_type,
            })
            self._save_stats(storage.file_name_prefix, stats)
            return [question_col, choices_col, label_col, output_eval_score_key, output_eval_valid_key, output_eval_error_key]

        elif eval_type == "key3_q_choices_as":
            question_col = input_keys_map.get("question", "")
            choices_col = input_keys_map.get("choices", "")
            labels_col = input_keys_map.get("labels", "")
            required = [question_col, choices_col, labels_col, input_pred_key]  # 先按“解析模型输出集合”实现
            if not self._check_columns(df, required):
                storage.write(df)
                return required

            self._eval_mc_multi(
                df=df,
                question_col=question_col,
                choices_col=choices_col,
                labels_col=labels_col,
                pred_col=input_pred_key,
                metric_type=metric_type,
            )
            storage.write(df)

            stats = self._stats_for_multiselect(df)
            stats.update({
                "bench_name_or_prefix": storage.file_name_prefix,
                "type": eval_type,
                "metric": metric_type,
            })
            self._save_stats(storage.file_name_prefix, stats)
            return [question_col, choices_col, labels_col, input_pred_key, output_eval_score_key, output_eval_valid_key, output_eval_error_key]

        elif eval_type == "key3_q_a_rejected":
            question_col = input_keys_map.get("question", "")
            better_col = input_keys_map.get("better", "")
            rejected_col = input_keys_map.get("rejected", "")
            required = [question_col, better_col, rejected_col]
            if not self._check_columns(df, required):
                storage.write(df)
                return required

            if self.llm_serving is None:
                # 这个类型没有 pred_col 可 fallback，只能报错
                self.logger.error("llm_serving is required for pairwise evaluation")
                df[output_eval_valid_key] = False
                df[output_eval_error_key] = "llm_serving_required_for_pairwise"
                storage.write(df)
                stats = {
                    "bench_name_or_prefix": storage.file_name_prefix,
                    "type": eval_type,
                    "metric": metric_type,
                    "total_samples": int(len(df)),
                    "valid_samples": 0,
                    "note": "pairwise requires llm_serving loglikelihood",
                }
                self._save_stats(storage.file_name_prefix, stats)
                return required + [output_eval_score_key, output_eval_valid_key, output_eval_error_key]

            self._eval_pairwise(
                df=df,
                question_col=question_col,
                better_col=better_col,
                rejected_col=rejected_col,
                ctx_series=ctx_series,
                metric_type=metric_type,
            )
            storage.write(df)

            stats = self._stats_for_binary(df)
            stats.update({
                "bench_name_or_prefix": storage.file_name_prefix,
                "type": eval_type,
                "metric": metric_type,
            })
            self._save_stats(storage.file_name_prefix, stats)
            return required + [output_eval_score_key, output_eval_valid_key, output_eval_error_key]

        else:
            self.logger.error(f"Unknown bench_dataflow_eval_type: {eval_type}")
            storage.write(df)
            return [output_eval_valid_key, output_eval_error_key, input_pred_key, output_eval_score_key]

    # -----------------------------
    # 默认 metric
    # -----------------------------
    def _default_metric_for_type(self, t: str, use_semantic_judge: bool) -> str:
        if t == "key1_text_score":
            return "ppl"
        if t == "key2_qa":
            return "semantic_judge" if use_semantic_judge else "math_verify"
        if t == "key2_q_ma":
            return "any_math_verify"
        if t == "key3_q_choices_a":
            return "ll_choice_acc"
        if t == "key3_q_choices_as":
            return "micro_f1"
        if t == "key3_q_a_rejected":
            return "pairwise_ll_winrate"
        return "unknown"

    # -----------------------------
    # 统计：binary（0/1）
    # -----------------------------
    def _stats_for_binary(self, df: pd.DataFrame) -> Dict[str, Any]:
        total = len(df)
        valid_mask = df[output_eval_valid_key] == True
        valid = int(valid_mask.sum())
        # eval_score: 0/1
        if valid > 0:
            acc = float(df.loc[valid_mask, output_eval_score_key].mean())
        else:
            acc = 0.0
        return {
            "total_samples": int(total),
            "valid_samples": int(valid),
            "accuracy": float(acc),
        }

    # -----------------------------
    # 统计：多选（f1/jaccard 等）
    # -----------------------------
    def _stats_for_multiselect(self, df: pd.DataFrame) -> Dict[str, Any]:
        total = len(df)
        valid_mask = df[output_eval_valid_key] == True
        valid = int(valid_mask.sum())
        # eval_score 默认存 f1
        if valid > 0:
            f1_mean = float(df.loc[valid_mask, output_eval_score_key].mean())   
        else:
            f1_mean = 0.0
        # 如果你想要更多维度（jaccard/exact_set），可以从 eval_pred 里扩展存 dict，这里先给最小
        return {
            "total_samples": int(total),
            "valid_samples": int(valid),
            "micro_f1_mean": float(f1_mean),
        }

    # -----------------------------
    # key2_qa：单参考
    # -----------------------------
    def _eval_qa_single(
        self,
        df: pd.DataFrame,
        question_col: str,
        target_col: str,
        pred_col: str,
        ctx_series: Optional[pd.Series],
        metric_type: str,
    ) -> None:
        if metric_type == "semantic_judge":
            # 语义 judge 需要 llm_serving.generate_from_input
            if self.llm_serving is None or not hasattr(self.llm_serving, "generate_from_input"):
                self.logger.error("semantic_judge requires llm_serving.generate_from_input")
                df[output_eval_valid_key] = False
                df[output_eval_error_key] = "semantic_judge_unavailable"
                return

            # 默认用“预测 vs 标准”直接 judge（这里只做通用；可自行替换 AnswerJudgePrompt）
            inputs = []
            row_indices = []
            for idx, row in df.iterrows():
                gt = row[target_col]
                pred = row[pred_col]
                if gt is None or (isinstance(gt, str) and gt.strip() == ""):
                    df.at[idx, output_eval_valid_key] = False
                    df.at[idx, output_eval_error_key] = "empty_reference"
                    continue
                if pred is None or (isinstance(pred, str) and pred.strip() == ""):
                    df.at[idx, output_eval_valid_key] = False
                    df.at[idx, output_eval_error_key] = "empty_prediction"
                    continue

                prompt = (
                    "You are an evaluator. Decide if the prediction is correct given the reference.\n"
                    f"Reference:\n{gt}\n\nPrediction:\n{pred}\n\n"
                    'Return JSON: {"judgement_result": true/false}'
                )
                inputs.append(prompt)
                row_indices.append(idx)

            if not inputs:
                return

            try:
                responses = self.llm_serving.generate_from_input(user_inputs=inputs, system_prompt=self.system_prompt)
            except Exception as e:
                self.logger.error(f"semantic_judge generate_from_input failed: {e}")
                for idx in row_indices:
                    df.at[idx, output_eval_valid_key] = False
                    df.at[idx, output_eval_error_key] = "semantic_judge_failed"
                return

            for idx, resp in zip(row_indices, responses):
                ok = self._resolve_judge_response(resp)
                df.at[idx, output_eval_score_key] = 1.0 if ok else 0.0
                df.at[idx, output_eval_pred_key] = None
                df.at[idx, output_eval_valid_key] = True    
                df.at[idx, output_eval_error_key] = ""

            return

        # 默认：math_verify
        for idx, row in df.iterrows():
            gt = row[target_col]
            pred_raw = row[pred_col]
            if gt is None or (isinstance(gt, str) and gt.strip() == ""):
                df.at[idx, output_eval_valid_key] = False
                df.at[idx, output_eval_error_key] = "empty_reference"
                continue
            if pred_raw is None or (isinstance(pred_raw, str) and pred_raw.strip() == ""):
                df.at[idx, output_eval_valid_key] = False
                df.at[idx, output_eval_error_key] = "empty_prediction"
                continue

            final_answer = self.answer_extractor.extract_answer(pred_raw, None)
            text_ok = self._text_contains_match(pred_raw, gt) or self._text_contains_match(final_answer, gt)
            math_res = self._try_math_verify_compare(final_answer, gt)
            ok = text_ok or (math_res is True)
            df.at[idx, output_eval_score_key] = 1.0 if ok else 0.0
            df.at[idx, output_eval_pred_key] = str(final_answer) if (math_res is True) else str(pred_raw)
            df.at[idx, output_eval_valid_key] = True
            df.at[idx, output_eval_error_key] = ""

    # -----------------------------
    # key2_q_ma：多参考
    # -----------------------------
    def _eval_qa_multi(
        self,
        df: pd.DataFrame,
        question_col: str,
        targets_col: str,
        pred_col: str,
        ctx_series: Optional[pd.Series],
        metric_type: str,
    ) -> None:
        # 默认：any_math_verify
        for idx, row in df.iterrows():
            targets_raw = row[targets_col]
            pred_raw = row[pred_col]
            targets = self._normalize_targets(targets_raw)

            if len(targets) == 0:
                df.at[idx, output_eval_valid_key] = False
                df.at[idx, output_eval_error_key] = "empty_references"
                continue
            if pred_raw is None or (isinstance(pred_raw, str) and pred_raw.strip() == ""):
                df.at[idx, output_eval_valid_key] = False
                df.at[idx, output_eval_error_key] = "empty_prediction"
                continue

            final_answer = self.answer_extractor.extract_answer(pred_raw, None)
            ok_any = False
            matched_by_text = False
            for gt in targets:
                text_ok = self._text_contains_match(pred_raw, gt) or self._text_contains_match(final_answer, gt)
                math_res = self._try_math_verify_compare(final_answer, gt)
                if text_ok or (math_res is True):
                    ok_any = True
                    matched_by_text = matched_by_text or text_ok
                    break

            df.at[idx, output_eval_score_key] = 1.0 if ok_any else 0.0
            df.at[idx, output_eval_pred_key] = str(pred_raw) if matched_by_text else str(final_answer)
            df.at[idx, output_eval_valid_key] = True
            df.at[idx, output_eval_error_key] = ""  

    # -----------------------------
    # key3_q_choices_a：单选
    # -----------------------------
    def _eval_mc_single(
        self,
        df: pd.DataFrame,
        question_col: str,
        choices_col: str,
        label_col: str,
        ctx_series: Optional[pd.Series],
        metric_type: str,
        pred_col: str,
    ) -> None:
        # 优先：loglikelihood
        if metric_type == "ll_choice_acc" and self.llm_serving is not None:
            # 批量做：每行要对 choices 逐个算 ll，先实现清晰版（你后面可优化 batching）
            for idx, row in df.iterrows():
                q = row[question_col]
                choices = row[choices_col]
                label = row[label_col]

                if choices is None or (isinstance(choices, float) and np.isnan(choices)):
                    df.at[idx, output_eval_valid_key] = False
                    df.at[idx, output_eval_error_key] = "empty_choices"
                    continue
                if not isinstance(choices, list):
                    # 尝试 json
                    try:
                        choices = json.loads(str(choices))
                    except Exception:
                        df.at[idx, output_eval_valid_key] = False
                        df.at[idx, output_eval_error_key] = "choices_not_list"
                        continue
                if len(choices) == 0:
                    df.at[idx, output_eval_valid_key] = False
                    df.at[idx, output_eval_error_key] = "empty_choices"
                    continue

                ctx = None
                if ctx_series is not None:
                    ctx = self._normalize_context(ctx_series.loc[idx])

                prompt = self._build_prompt(question=str(q), context=ctx, choices=[str(c) for c in choices], task="mc_single")

                # label 规范化为 idx
                gold_idx = self._normalize_label_to_index(label, len(choices))
                if gold_idx is None:
                    df.at[idx, output_eval_valid_key] = False
                    df.at[idx, output_eval_error_key] = "invalid_label"
                    continue

                prompts = [prompt] * len(choices)
                conts = []
                for c in choices:
                    c_str = str(c)
                    # 常见做法：continuation 前补空格，避免直接拼在 Answer: 后面太粘连
                    conts.append((" " + c_str) if (len(prompt) > 0 and not prompt.endswith((" ", "\n"))) else c_str)

                lls = self._ll_batch(prompts, conts)
                if lls is None:
                    df.at[idx, output_eval_valid_key] = False
                    df.at[idx, output_eval_error_key] = "ll_unavailable"
                    continue

                pred_idx = int(np.argmax(np.array(lls)))
                df.at[idx, output_eval_pred_key] = int(pred_idx)
                df.at[idx, output_eval_score_key] = 1.0 if pred_idx == gold_idx else 0.0
                df.at[idx, output_eval_valid_key] = True
                df.at[idx, output_eval_error_key] = ""

            return

        # fallback：从 pred_col 解析（generation 输出里抓 A/B/C 或数字）
        self.logger.warning("ll_choice_acc unavailable; fallback to parse generated output for single-choice.")
        for idx, row in df.iterrows():
            choices = row[choices_col]
            label = row[label_col]
            pred_text = row[pred_col] if pred_col in df.columns else None

            if choices is None:
                df.at[idx, output_eval_valid_key] = False
                df.at[idx, output_eval_error_key] = "empty_choices"
                continue
            if not isinstance(choices, list):
                try:
                    choices = json.loads(str(choices))
                except Exception:
                    df.at[idx, output_eval_valid_key] = False
                    df.at[idx, output_eval_error_key] = "choices_not_list"
                    continue

            gold_idx = self._normalize_label_to_index(label, len(choices))
            pred_idx = self._parse_choice_from_text(str(pred_text), len(choices)) if pred_text is not None else None
            if gold_idx is None or pred_idx is None:
                df.at[idx, output_eval_valid_key] = False
                df.at[idx, output_eval_error_key] = "parse_failed"
                continue

            df.at[idx, output_eval_pred_key] = int(pred_idx)
            df.at[idx, output_eval_score_key] = 1.0 if pred_idx == gold_idx else 0.0
            df.at[idx, output_eval_valid_key] = True
            df.at[idx, output_eval_error_key] = ""

    def _normalize_label_to_index(self, label: Any, n: int) -> Optional[int]:
        if label is None:
            return None
        # 若 label 本身是 int
        if isinstance(label, (int, np.integer)):
            v = int(label)
            if 0 <= v < n:
                return v
            if 1 <= v <= n:
                return v - 1
            return None
        s = str(label).strip()
        if not s:
            return None
        # A/B/C
        if len(s) == 1 and s.isalpha():
            idx = ord(s.upper()) - ord("A")
            return idx if 0 <= idx < n else None
        # 数字
        if s.isdigit():
            v = int(s)
            if 0 <= v < n:
                return v
            if 1 <= v <= n:
                return v - 1
        return None

    # -----------------------------
    # key3_q_choices_as：多选
    # -----------------------------
    def _eval_mc_multi(
        self,
        df: pd.DataFrame,
        question_col: str,
        choices_col: str,
        labels_col: str,
        pred_col: str,
        metric_type: str,
    ) -> None:
        # 这里按你说的“先最小落地”：从 pred_col 解析集合 -> micro_f1
        for idx, row in df.iterrows():
            choices = row[choices_col]
            gold = row[labels_col]
            pred_text = row[pred_col]

            if choices is None:
                df.at[idx, output_eval_valid_key] = False
                df.at[idx, output_eval_error_key] = "empty_choices"
                continue
            if not isinstance(choices, list):
                try:
                    choices = json.loads(str(choices))
                except Exception:
                    df.at[idx, output_eval_valid_key] = False
                    df.at[idx, output_eval_error_key] = "choices_not_list"
                    continue

            n = len(choices)
            gold_set = self._normalize_multilabel_to_set(gold, n)
            pred_set = self._parse_multiselect_set(str(pred_text), n)

            if gold_set is None or pred_set is None:
                df.at[idx, output_eval_valid_key] = False
                df.at[idx, output_eval_error_key] = "parse_failed"
                continue

            m = self._set_metrics(pred_set, gold_set)
            # eval_score 默认存 f1（你的上层聚合最常用）
            df.at[idx, output_eval_score_key] = float(m["f1"])
            # eval_pred 存更丰富的信息，便于 debug
            df.at[idx, output_eval_pred_key] = json.dumps(
                {"pred_set": sorted(list(pred_set)), "gold_set": sorted(list(gold_set)), **m},
                ensure_ascii=False,
            )
            df.at[idx, output_eval_valid_key] = True
            df.at[idx, output_eval_error_key] = ""

    def _normalize_multilabel_to_set(self, labels: Any, n: int) -> Optional[set]:
        if labels is None:
            return None
        if isinstance(labels, float) and np.isnan(labels):
            return None
        if isinstance(labels, list):
            s = set()
            for x in labels:
                idx = self._normalize_label_to_index(x, n)
                if idx is None:
                    continue
                s.add(idx)
            return s if len(s) > 0 else set()

        s = str(labels).strip()
        if not s:
            return None
        # json list
        if s.startswith("[") and s.endswith("]"):
            try:
                obj = json.loads(s)
                if isinstance(obj, list):
                    res = set()
                    for x in obj:
                        idx = self._normalize_label_to_index(x, n)
                        if idx is not None:
                            res.add(idx)
                    return res
            except Exception:
                pass

        # 分隔符
        parts = re.split(r"[,\s;/|]+", s)
        res = set()
        for p in parts:
            p = p.strip()
            if not p:
                continue
            idx = self._normalize_label_to_index(p, n)
            if idx is not None:
                res.add(idx)
        return res if len(res) > 0 else set()

    # -----------------------------
    # key3_q_a_rejected：偏好对比
    # -----------------------------
    def _eval_pairwise(
        self,
        df: pd.DataFrame,
        question_col: str,
        better_col: str,
        rejected_col: str,
        ctx_series: Optional[pd.Series],
        metric_type: str,
    ) -> None:
        # 默认：pairwise_ll_winrate
        for idx, row in df.iterrows():
            q = row[question_col]
            better = row[better_col]
            rej = row[rejected_col]

            if better is None or (isinstance(better, str) and better.strip() == ""):
                df.at[idx, output_eval_valid_key] = False
                df.at[idx, output_eval_error_key] = "empty_better"
                continue
            if rej is None or (isinstance(rej, str) and rej.strip() == ""):
                df.at[idx, output_eval_valid_key] = False
                df.at[idx, output_eval_error_key] = "empty_rejected"
                continue

            ctx = None
            if ctx_series is not None:
                ctx = self._normalize_context(ctx_series.loc[idx])

            prompt = self._build_prompt(question=str(q), context=ctx, task="pairwise")

            prompts = [prompt, prompt]
            conts = []
            better_s = str(better)
            rej_s = str(rej)
            conts.append((" " + better_s) if (len(prompt) > 0 and not prompt.endswith((" ", "\n"))) else better_s)
            conts.append((" " + rej_s) if (len(prompt) > 0 and not prompt.endswith((" ", "\n"))) else rej_s)

            lls = self._ll_batch(prompts, conts)
            if lls is None or len(lls) != 2:
                df.at[idx, output_eval_valid_key] = False
                df.at[idx, output_eval_error_key] = "ll_unavailable"
                continue

            win = 1.0 if float(lls[0]) > float(lls[1]) else 0.0
            df.at[idx, output_eval_score_key] = win
            df.at[idx, output_eval_pred_key] = json.dumps({"ll_better": float(lls[0]), "ll_rejected": float(lls[1])}, ensure_ascii=False)
            df.at[idx, output_eval_valid_key] = True
            df.at[idx, output_eval_error_key] = ""

    # -----------------------------
    # 语义 judge 响应解析（兼容你旧逻辑）
    # -----------------------------
    def _resolve_judge_response(self, response: Any) -> bool:
        if response is None or (isinstance(response, str) and response.strip() == ""):
            self.empty_responses_count += 1
            return False
        try:
            s = str(response)
            # 尝试 json
            try:
                obj = json.loads(s)
                if isinstance(obj, dict) and "judgement_result" in obj:
                    return bool(obj["judgement_result"])
            except Exception:
                pass

            pattern = re.compile(r'"judgement_result"\s*:\s*(true|false)', re.IGNORECASE)
            m = pattern.search(s)
            if m:
                return m.group(1).lower() == "true"
            # fallback
            return ("true" in s.lower()) and ("false" not in s.lower())
        except Exception as e:
            self.logger.error(f"Response format error: {response}. Error: {e}")
            return False

    # -----------------------------
    # 描述
    # -----------------------------
    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "该算子用于统一 Bench 评测，支持多种任务范式并将评测结果写回 DataFrame，同时输出整体统计到 eval_result_path。\n\n"
                "支持类型与默认 metric：\n"
                "- key1_text_score：ppl\n"
                "- key2_qa：math_verify（或 use_semantic_judge=True 时 semantic_judge）\n"
                "- key2_q_ma：any_math_verify（多参考）\n"
                "- key3_q_choices_a：ll_choice_acc（基于 loglikelihood；无 serving 接口时使用 HF forward 计算 ll）\n"
                "- key3_q_choices_as：micro_f1（解析多选集合后计算）\n"
                "- key3_q_a_rejected：pairwise_ll_winrate（基于 ll 比较 better vs rejected）\n\n"
                "初始化参数：\n"
                "- eval_result_path：统计结果落盘路径\n"
                "- eval_type：评测类型（同上）\n"
                "- llm_serving：可选；用于 semantic_judge 或提供模型路径信息以进行 PPL/LL 的 HF 计算\n"
                "- prompt_template：提示模板对象（可选；需提供 build_prompt；默认使用 AnswerJudgePrompt）\n"
                "- system_prompt：语义评测/judge 的系统提示词\n"
                "- metric_type：可选；不传则使用 eval_type 的默认 metric\n"
                "- use_semantic_judge：仅对 key2_qa 有效；是否使用语义评测\n\n"
                "运行参数：\n"
                "- storage：DataFlowStorage\n"
                "- input_keys_map：字段映射（不同 eval_type 需要不同 key：text/question/target/targets/choices/label/labels/better/rejected）\n"
                "- input_context_key：可选，上下文字段名\n"
                "- input_pred_key：预测答案字段名（默认 generated_ans）\n\n"
                "输出：\n"
                "- output_eval_score_key（数值分数）\n"
                "- output_eval_pred_key（解析后的预测）\n"
                "- output_eval_valid_key（是否有效）\n"
                "- output_eval_error_key（错误信息）\n"
                "- 保存统计：total_samples/valid_samples/accuracy 或 ppl_mean 等到 eval_result_path\n"
                "- 返回本次评测涉及/产出的列名列表"
            )
        return (
            "This operator evaluates unified bench datasets across multiple task archetypes. It writes per-sample results back to the dataframe and saves aggregated statistics to eval_result_path.\n\n"
            "Supported Types (default metric):\n"
            "- key1_text_score (ppl)\n"
            "- key2_qa (math_verify or semantic_judge)\n"
            "- key2_q_ma (any_math_verify)\n"
            "- key3_q_choices_a (ll_choice_acc)\n"
            "- key3_q_choices_as (micro_f1)\n"
            "- key3_q_a_rejected (pairwise_ll_winrate)\n\n"
            "Input Parameters:\n"
            "- eval_result_path: Path to save aggregated statistics\n"
            "- eval_type: Evaluation type (one of the supported types)\n"
            "- llm_serving: Optional; required for semantic_judge and used as model source for HF-based PPL/LL computation\n"
            "- prompt_template: Prompt template object (optional; must provide build_prompt; default is AnswerJudgePrompt)\n"
            "- system_prompt: System prompt for semantic judging\n"
            "- metric_type: Optional; overrides the default metric for the given eval_type\n"
            "- use_semantic_judge: Only for key2_qa; whether to use LLM-based semantic judging\n\n"
            "Run Parameters:\n"
            "- storage: DataFlowStorage\n"
            "- keys_map: Column mapping; depends on eval_type (text/question/target/targets/choices/label/labels/better/rejected)\n"
            "- context_key: Optional context column name\n"
            "- input_pred_key: Prediction column name (default: generated_ans)\n\n"
            "Output Parameters:\n"
            f"- output_eval_score_key: Numeric score (accuracy classes use 0/1)\n"
            f"- output_eval_pred_key: Parsed prediction\n"
            f"- output_eval_valid_key: Whether the sample is valid\n"
            f"- output_eval_error_key: Error message if any\n"
            "- Saves aggregated stats to eval_result_path\n"
            "- Returns a list of involved/output keys"
        )