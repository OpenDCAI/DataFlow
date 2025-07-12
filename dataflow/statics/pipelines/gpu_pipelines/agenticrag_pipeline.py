from dataflow.operators.generate import (
    AutoPromptGenerator,
    QAGenerator,
    QAScorer
)

from dataflow.operators.filter import (
    ContentChooser
)

from dataflow.utils.storage import FileStorage
from dataflow.serving import LocalModelLLMServing_vllm


class AgenticRAGPipeline():
    def __init__(self):

        self.storage = FileStorage(
            first_entry_file_name="../example_data/AgenticRAGPipeline/pipeline_small_chunk.json",
            cache_path="./cache_local",
            file_name_prefix="dataflow_cache_step",
            cache_type="json",
        )

        # use API server as LLM serving
        llm_serving = LocalModelLLMServing_vllm(
            hf_model_name_or_path="Qwen2.5-7B-Instruct", # set to your own model path
            vllm_tensor_parallel_size=4,
            vllm_max_tokens=8192,
        )

        embedding_serving = LocalModelLLMServing_vllm(hf_model_name_or_path="Alibaba-NLP/gte-Qwen2-7B-instruct", vllm_max_tokens=8192)
        self.content_chooser_step1 = ContentChooser(embedding_serving=embedding_serving, num_samples=5, method="random")

        self.prompt_generator_step2 = AutoPromptGenerator(llm_serving)

        self.qa_generator_step3 = QAGenerator(llm_serving)

        self.qa_scorer_step4 = QAScorer(llm_serving)
        
    def forward(self):

        self.content_chooser_step1.run(
            storage = self.storage.step(),
            input_key = "text"
        )

        self.prompt_generator_step2.run(
            storage = self.storage.step(),
            input_key = "text"
        )

        self.qa_generator_step3.run(
            storage = self.storage.step(),
            input_key="text",
            prompt_key="generated_prompt",
            output_quesion_key="generated_question",
            output_answer_key="generated_answer"
        )

        self.qa_scorer_step4.run(
            storage = self.storage.step(),
            input_question_key="generated_question",
            input_answer_key="generated_answer",
            output_question_quality_key="question_quality_grades",
            output_question_quality_feedback_key="question_quality_feedbacks",
            output_answer_alignment_key="answer_alignment_grades",
            output_answer_alignment_feedback_key="answer_alignment_feedbacks",
            output_answer_verifiability_key="answer_verifiability_grades",
        )
        
if __name__ == "__main__":
    model = AgenticRAGPipeline()
    model.forward()
