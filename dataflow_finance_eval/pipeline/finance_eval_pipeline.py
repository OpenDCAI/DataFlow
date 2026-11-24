"""Finance eval pipeline: QA eval → SQL eval → hallucination eval → report.

PyTorch 风格 Pipeline，遵循 DataFlow 主仓库 Pipeline 约定。
"""
import logging

from dataflow_finance_eval.operator import (
    EvalReportGenerator,
    FinanceHallucinationSampleEvaluator,
    FinanceQASampleEvaluator,
    FinanceSQLSampleEvaluator,
)


class FinanceEvalPipeline:
    """端到端金融任务评测流水线。"""

    def __init__(self, llm_serving=None, use_ast: bool = False, **kwargs):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.storage = kwargs.get("storage")
        self.qa_evaluator = FinanceQASampleEvaluator()
        self.sql_evaluator = FinanceSQLSampleEvaluator(use_ast=use_ast)
        self.hallucination_evaluator = FinanceHallucinationSampleEvaluator(llm_serving=llm_serving)
        self.report_generator = EvalReportGenerator()

    def run(self):
        self.logger.info("FinanceEvalPipeline start: qa → sql → hallucination → report")

        storage = self.storage.step()
        self.qa_evaluator.run(
            storage,
            input_answer="answer",
            input_gold_answer="gold_answer",
            output_qa_score="qa_score",
            output_qa_correct="qa_correct",
        )

        storage = self.storage.step()
        self.sql_evaluator.run(
            storage,
            input_sql="sql",
            input_gold_sql="gold_sql",
            output_sql_score="sql_score",
            output_sql_correct="sql_correct",
        )

        storage = self.storage.step()
        self.hallucination_evaluator.run(
            storage,
            input_text="text",
            input_question="question",
            input_answer="answer",
            output_hallucination_flag="hallucination",
            output_hallucination_reason="hallucination_reason",
        )

        storage = self.storage.step()
        self.report_generator.run(
            storage,
            input_qa_correct="qa_correct",
            input_sql_correct="sql_correct",
            input_hallucination_flag="hallucination",
            output_summary="eval_summary",
        )

        self.logger.info("FinanceEvalPipeline done")
