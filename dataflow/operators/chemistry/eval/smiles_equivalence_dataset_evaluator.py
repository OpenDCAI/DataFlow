import pandas as pd
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger

from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC
from dataflow.core import LLMServingABC
import json

@OPERATOR_REGISTRY.register()
class SmilesEquivalenceDatasetEvaluator(OperatorABC):
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
                "逐块输出 final_result、块内得分与准确率，并统计全局总分。\n"
                "输入参数：\n"
                "- input_golden_key：标准答案字段名（list[dict]，含abbreviation/full_name/smiles），默认为'golden_label'\n"
                "- input_synth_key：合成结果字段名（list[dict]，含abbreviation/smiles），默认为'synth_smiles'\n"
                "- output_key：写入逐项评估结果的字段名，默认为'final_result'\n"
                "输出参数：\n"
                "- output_key：list[dict]，每个golden项一条{abbreviation, full_name, smiles, score}，按abbreviation对齐，RDKit规范化后等价得1，否则或无对应项得0\n"
                "- block_score：该行得分之和（整数）\n"
                "- block_total：该行golden项数量（整数）\n"
                "- block_accuracy：block_score / block_total，block_total为0时为空\n"
                "- 不过滤任何行；全局汇总仅保存在算子的overall_summary属性中，不写入数据\n"
                "- 返回output_key（字符串）"
            )
        elif lang == "en":
            return (
                "Evaluate SMILES equivalence between golden_label and synth_smiles per block, "
                "emit final_result, block-level scores/accuracy, and compute overall summary.\n"
                "Input Parameters:\n"
                "- input_golden_key: gold label field (list[dict] with abbreviation/full_name/smiles), default 'golden_label'\n"
                "- input_synth_key: synthesized field (list[dict] with abbreviation/smiles), default 'synth_smiles'\n"
                "- output_key: name of the field that receives the per-item results, default 'final_result'\n"
                "Output Parameters:\n"
                "- output_key: list[dict], one {abbreviation, full_name, smiles, score} per gold item, aligned by abbreviation; score is 1 if the RDKit canonical SMILES match, else 0 (also 0 when no matching item)\n"
                "- block_score: sum of scores in the row (integer)\n"
                "- block_total: number of gold items in the row (integer)\n"
                "- block_accuracy: block_score / block_total, empty when block_total is 0\n"
                "- No rows are filtered; the overall summary is kept only in the operator's overall_summary attribute, not written to the data\n"
                "- Returns output_key (a string)"
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
        input_golden_key: str = "golden_label",
        input_synth_key: str = "synth_smiles",
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
            golden_label = row.get(input_golden_key, [])  # list[dict]
            synth_smiles = row.get(input_synth_key, [])   # list[dict]（可能为空）

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
