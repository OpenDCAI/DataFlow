import pandas as pd
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger

from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC
from dataflow.core import LLMServingABC
import json

@OPERATOR_REGISTRY.register()
class EvaluateSmilesEquivalence(OperatorABC):
    """
    对每个块（row）里的 golden_label 与 synth_smiles 进行 SMILES 等价性评估：
    - 以 abbreviation 对齐
    - RDKit 规范 SMILES 后比较是否相等（相等=1，否则=0；不存在对应项=0）
    - 输出到 dataframe 的新列：final_result（list[dict]）、block_score、block_total、block_accuracy
    - 统计 overall 汇总并保存到 self.overall_summary
    """
    def __init__(self, llm_serving: LLMServingABC = None):
        self.logger = get_logger()
        self.overall_summary = None  # 评估完成后填充

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "评估 golden_label 与 synth_smiles 的 SMILES 等价性并计算分数。"
                "逐块输出 final_result、块内得分与准确率，并统计全局总分。"
            )
        elif lang == "en":
            return (
                "Evaluate SMILES equivalence between golden_label and synth_smiles per block, "
                "emit final_result, block-level scores/accuracy, and compute overall summary."
            )
        else:
            return "Evaluate SMILES equivalence for each block."

    # ========= 内部工具函数（不对用户暴露） =========
    def _are_smiles_equivalent(self, s1: str, s2: str) -> bool:
        """
        使用 RDKit 将 SMILES 规范化后判断是否等价。
        任意一侧解析失败，返回 False。
        """
        try:
            from rdkit import Chem
        except ImportError:
            raise Exception(
                """
        rdkit is not installed in this environment yet.
        Please use pip install rdkit.
        """
            )
        try:
            m1, m2 = Chem.MolFromSmiles(s1), Chem.MolFromSmiles(s2)
            if m1 is None or m2 is None:
                return False
            c1 = Chem.MolToSmiles(m1, canonical=True)
            c2 = Chem.MolToSmiles(m2, canonical=True)
            return c1 == c2
        except Exception as e:
            self.logger.warning(f"SMILES 解析异常: {e}")
            return False

    def _score_one_block(self, golden_label, synth_smiles):
        """
        对单个块计算：
        - final_result: list[ {abbreviation, full_name, smiles, score} ]
        - block_score, block_total, block_accuracy
        """
        golden = golden_label or []
        synth = synth_smiles or []

        # 用 abbreviation 建索引（synth_smiles 可能为空/缺字段）
        synth_map = {}
        for item in synth:
            abbr = (item or {}).get("abbreviation")
            smi  = (item or {}).get("smiles")
            if abbr is not None and smi is not None:
                synth_map[abbr] = smi

        final_result = []
        block_score = 0

        for g in golden:
            abbr = (g or {}).get("abbreviation", "")
            gold_smi = (g or {}).get("smiles", "")
            synth_smi = synth_map.get(abbr)  # 可能 None

            score = 0
            if synth_smi is not None:
                score = 1 if self._are_smiles_equivalent(gold_smi, synth_smi) else 0

            block_score += score
            final_result.append({
                "abbreviation": abbr,
                "full_name": (g or {}).get("full_name", ""),
                "smiles": gold_smi,
                "score": score
            })

        block_total = len(golden)
        block_accuracy = (block_score / block_total) if block_total else None

        return final_result, block_score, block_total, block_accuracy

    # ========= 对外主流程 =========
    def run(
        self,
        storage: DataFlowStorage,
        golden_key: str = "golden_label",
        synth_key: str = "synth_smiles",
        output_key: str = "final_result",
    ):
        """
        dataframe 视作读取到的 JSON（每行一个块），按照给定 key 取出 golden/synth，写回评估结果。
        返回 output_key（供后续算子引用）。
        """
        self.logger.info("Running EvaluateSmilesEquivalence...")

        # 读取 dataframe
        dataframe = storage.read('dataframe')
        self.logger.info(f"Loaded dataframe with {len(dataframe)} rows")

        # 为结果准备列
        block_scores = []
        block_totals = []
        block_accuracies = []
        final_results_all = []

        # 逐行（块）评估
        for idx, row in dataframe.iterrows():
            golden_label = row.get(golden_key, [])  # list[dict]
            synth_smiles = row.get(synth_key, [])   # list[dict]（可能为空）

            final_result, block_score, block_total, block_accuracy = \
                self._score_one_block(golden_label, synth_smiles)

            final_results_all.append(final_result)
            block_scores.append(block_score)
            block_totals.append(block_total)
            block_accuracies.append(block_accuracy)

        # 写回列
        dataframe[output_key] = final_results_all
        dataframe["block_score"] = block_scores
        dataframe["block_total"] = block_totals
        dataframe["block_accuracy"] = block_accuracies

        # 汇总总体分数
        overall_score = int(sum(block_scores))
        overall_total = int(sum(block_totals))
        overall_accuracy = (overall_score / overall_total) if overall_total else None

        self.overall_summary = {
            "overall_score": overall_score,
            "overall_total": overall_total,
            "overall_accuracy": overall_accuracy
        }
        self.logger.info(f"Overall summary: {self.overall_summary}")

        # 保存更新后的 dataframe
        storage.write(dataframe)

        # 返回给后续算子使用的列名（与 ExtractSmilesFromText 一致风格）
        return output_key
