"""Finance QA pipeline: PDF → QA 生成 → 幻觉过滤 → 数值一致性精炼。

PyTorch 风格 Pipeline，遵循 DataFlow 主仓库 Pipeline 约定。
"""
import logging

from dataflow_finance_qa.operator import (
    FinancialHallucinationFilter,
    FinancialPDFParser,
    FinancialQAGenerator,
    NumericConsistencyRefiner,
)


class FinanceQAPipeline:
    """端到端金融 QA 数据制备流水线。"""

    def __init__(self, llm_serving, max_pages: int | None = None, **kwargs):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.storage = kwargs.get("storage")
        self.pdf_parser = FinancialPDFParser(max_pages=max_pages)
        self.qa_generator = FinancialQAGenerator(llm_serving=llm_serving)
        self.hallucination_filter = FinancialHallucinationFilter(llm_serving=llm_serving)
        self.refiner = NumericConsistencyRefiner(llm_serving=llm_serving)

    def run(self):
        self.logger.info("FinanceQAPipeline start: parse → generate → filter → refine")

        storage = self.storage.step()
        self.pdf_parser.run(storage, input_pdf_path="pdf_path", output_text="text")

        storage = self.storage.step()
        self.qa_generator.run(
            storage, input_text="text", output_question="question", output_answer="answer"
        )

        storage = self.storage.step()
        self.hallucination_filter.run(
            storage,
            input_text="text",
            input_question="question",
            input_answer="answer",
        )

        storage = self.storage.step()
        self.refiner.run(
            storage, input_text="text", input_answer="answer", output_answer="answer_refined"
        )

        self.logger.info("FinanceQAPipeline done")
