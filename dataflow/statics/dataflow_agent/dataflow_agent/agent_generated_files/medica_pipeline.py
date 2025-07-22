# ============================================================
# file: agent_generated_pipelines/medical_full_pipeline.py
# ============================================================
"""
Pipeline 顺序：
1. MedicalQuestionParaphraser      （同义改写，列 -> questionPARA）
2. MedicalQuestionContextualizer   （上下文扩充，列 -> enhanced_question）
3. ReasonStepAugment               （生成思维链，列 -> reasoning_steps）
4. MedicalDistractorGenerator      （生成干扰项，列 -> distractors）
"""

import os
from dataflow.utils.storage import FileStorage
from dataflow.serving import APILLMServing_request
from dataflow import get_logger
from operator_q_para import MedicalQuestionParaphraser
from operator_q_context_expand import MedicalQuestionContextualizer
from operator_q_cot import ReasonStepAugment
from operator_a_distractor_gen import MedicalDistractorGenerator
logger = get_logger()


class MedicalFullPipeline:
    def __init__(
        self,
        first_entry_file: str = "./example/medical_data.jsonl",
        cache_dir: str = "./cache_local",
        llm_api_url: str = "http://123.129.219.111:3000/v1/chat/completions",
        model_name: str = "gpt-4o",
    ):
        # ---------- DataFlow Storage ----------
        self.storage = FileStorage(
            first_entry_file_name=first_entry_file,
            cache_path=cache_dir,
            file_name_prefix="dataflow_cache_step",
            cache_type="jsonl",
        )

        # ---------- LLM Serving ----------
        self.llm_serving = APILLMServing_request(
            api_url=llm_api_url,
            key_name_of_api_key="DF_API_KEY",
            model_name=model_name,
            max_workers=64,
        )

        # ---------- Operators ----------
        self.step1_paraphraser = MedicalQuestionParaphraser(
            llm_serving=self.llm_serving,
            n_paraphrases=1,
            system_prompt=(
                "You are a medical question rewriting expert. "
                "For the given medical scenario question, provide a concise, semantically equivalent paraphrase."
            ),
            separator="\n",
        )

        self.step2_context = MedicalQuestionContextualizer(
            llm_serving=self.llm_serving,
            difficulty="medium",          # easy / medium / hard
            output_key="enhanced_question"
        )

        self.step3_cot = ReasonStepAugment(
            llm_serving=self.llm_serving,
            prompt_template=(
                "You are an expert in medical ethics. "
                "Explain in 3 short steps how to arrive at the given answer.\n\n"
                "Question: {question}\nAnswer: {answer}\n\nReasoning:"
            ),
        )

        self.step4_distractor = MedicalDistractorGenerator(
            llm_serving=self.llm_serving
        )

    # ----------------------------------------------------
    # 主执行函数
    # ----------------------------------------------------
    def forward(self):
        logger.info("STEP-1  同义改写 (Q-PARA)")
        self.step1_paraphraser.run(
            storage=self.storage.step(),
            input_key="question",
            output_key="questionPARA",
        )

        logger.info("STEP-2  上下文扩充 (Q-CONTEXT-EXPAND)")
        self.step2_context.run(
            storage=self.storage.step(),
            input_key="questionPARA",        # 使用刚生成的列作为输入
        )

        logger.info("STEP-3  思维链生成 (QA-COT)")
        self.step3_cot.run(
            storage=self.storage.step(),
            input_question_key="enhanced_question",
            input_answer_key="golden_answers",   # 数据集中正确答案列
            output_key="reasoning_steps",
        )

        logger.info("STEP-4  干扰项生成 (A-DISTRACTOR-GEN)")
        self.step4_distractor.run(
            storage=self.storage.step(),
            question_key="enhanced_question",
            answer_key="golden_answers",
            output_distractor_key="distractors",
        )

        logger.info("Pipeline 已全部执行完毕！")


# ------------------- CLI -------------------
if __name__ == "__main__":
    # 根据实际数据路径修改 first_entry_file
    pipeline = MedicalFullPipeline(
        first_entry_file="/mnt/h_h_public/lh/lz/DataFlow/dataflow/example/DataflowAgent/mq_test_data.jsonl"
    )
    pipeline.forward()