import pytest
from dataflow.operators.generate.question_generator import QuestionGenerator
from dataflow.operators.filter.question_filter import QuestionFilter
from dataflow.operators.generate.question_difficulty_classifier import QuestionDifficultyClassifier
from dataflow.operators.generate.question_category_classifier import QuestionCategoryClassifier
from dataflow.operators.filter.answer_pipeline_root import AnswerPipelineRoot
from dataflow.operators.generate.pseudo_answer_generator import PseudoAnswerGenerator
from dataflow.operators.filter.answer_formatter_filter import AnswerFormatterFilter
from dataflow.operators.filter.answer_token_length_filter import AnswerTokenLengthFilter
from dataflow.utils.storage import FileStorage
from dataflow.serving import APILLMServing_request, LocalModelLLMServing_vllm, LocalModelLLMServing_sglang


class RecommendPipeline():
    def __init__(self):

        # -------- FileStorage (请根据需要修改参数) --------
        self.storage = FileStorage(
            first_entry_file_name="/data1/lianghao/lh/DataFlow/dataflow/example/ReasoningPipeline/pipeline_math_short.json",
            cache_path="./cache_local",
            file_name_prefix="dataflow_cache_step",
            cache_type="jsonl",
        )


        # -------- LLM Serving (Local) --------
        llm_serving = LocalModelLLMServing_vllm(
            hf_model_name_or_path="/mnt/public/model/huggingface/Qwen2.5-7B-Instruct",
            vllm_tensor_parallel_size=1,
            vllm_max_tokens=8192,
            hf_local_dir="local",
        )

        self.questiongenerator = QuestionGenerator(num_prompts=1, llm_serving=llm_serving)
        self.questionfilter = QuestionFilter(system_prompt="You are a helpful assistant.", llm_serving=llm_serving)
        self.questiondifficultyclassifier = QuestionDifficultyClassifier(llm_serving=llm_serving)
        self.questioncategoryclassifier = QuestionCategoryClassifier(llm_serving=llm_serving)
        self.answerpipelineroot = AnswerPipelineRoot()
        self.pseudoanswergenerator = PseudoAnswerGenerator(llm_serving=llm_serving, max_times=3)
        self.answerformatterfilter = AnswerFormatterFilter()
        self.answertokenlengthfilter = AnswerTokenLengthFilter(max_answer_token_length=8192, tokenizer_dir="Qwen/Qwen2.5-0.5B-Instruct")

    def forward(self):
        self.questiongenerator.run(
            storage=self.storage.step(), input_key=""
        )
        self.questionfilter.run(
            storage=self.storage.step(), input_key="math_problem"
        )
        self.questiondifficultyclassifier.run(
            storage=self.storage.step(), input_key="", output_key="difficulty_score"
        )
        self.questioncategoryclassifier.run(
            storage=self.storage.step(), input_key="instruction", output_key="question_category"
        )
        self.answerpipelineroot.run(
            storage=self.storage.step(), input_answer_key="output", input_gt_key="golden_answer"
        )
        self.pseudoanswergenerator.run(
            storage=self.storage.step(), input_key="instruction", output_key_answer="pseudo_answers", output_key_answer_value="pseudo_answer_value", output_key_solutions="pseudo_solutions", output_key_correct_solution_example="pseudo_correct_solution_example"
        )
        self.answerformatterfilter.run(
            storage=self.storage.step(), input_key="generated_cot"
        )
        self.answertokenlengthfilter.run(
            storage=self.storage.step(), input_key="generated_cot"
        )


if __name__ == "__main__":
    pipeline = RecommendPipeline()
    pipeline.forward()
