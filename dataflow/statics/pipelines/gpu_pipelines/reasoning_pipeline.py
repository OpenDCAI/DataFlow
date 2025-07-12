from dataflow.operators.generate import (
    QuestionCategoryClassifier,
    QuestionDifficultyClassifier,
    QuestionGenerator,
    AnswerGenerator,
)

from dataflow.operators.filter import (
    QuestionFilter,
    AnswerPipelineRoot,
    AnswerFormatterFilter,
    AnswerTokenLengthFilter,
    AnswerGroundTruthFilter,
    AnswerNgramFilter, 
)
from dataflow.utils.storage import FileStorage
from dataflow.serving import LocalModelLLMServing_vllm

class ReasoningPipeline():
    def __init__(self):

        self.storage = FileStorage(
            first_entry_file_name="../example_data/ReasoningPipeline/pipeline_math_short.json",
            cache_path="./cache_local",
            file_name_prefix="dataflow_cache_step",
            cache_type="jsonl",
        )

        llm_serving = LocalModelLLMServing_vllm(
            hf_model_name_or_path="Qwen/Qwen2.5-7B-Instruct", # set to your own model path
            vllm_tensor_parallel_size=1,
            vllm_max_tokens=8192,
        )

        self.question_filter_step1 = QuestionFilter(
            system_prompt="You are an expert in evaluating mathematical problems. Follow the user's instructions strictly and output your final judgment in the required JSON format.",
            llm_serving=llm_serving
        )
        self.question_gen_step2 =  QuestionGenerator(
            num_prompts=3,
            llm_serving=llm_serving
        )
        self.question_filter_step3 = QuestionFilter(
            system_prompt="You are an expert in evaluating mathematical problems. Follow the user's instructions strictly and output your final judgment in the required JSON format.",
            llm_serving=llm_serving
        )
        self.question_difficulty_classifier_step4 = QuestionDifficultyClassifier(
            llm_serving=llm_serving
        )
        self.question_category_classifier_step5 = QuestionCategoryClassifier(
            llm_serving=llm_serving
        )
        ########################## branch ############################
        # self.answer_pipeline_root_step6 = AnswerPipelineRoot()
        ########################## answer ############################
        self.answer_generator_step7 = AnswerGenerator(
            llm_serving=llm_serving
        )
        
        self.answer_format_filter_step8 = AnswerFormatterFilter()
        
        self.answer_token_length_filter_step9 = AnswerTokenLengthFilter(
            max_answer_token_length = 8192,
            tokenizer_dir = "Qwen/Qwen2.5-0.5B-Instruct"
        )
        
        self.answer_groundtruth_filter_step10 = AnswerGroundTruthFilter()
        
        self.answer_ngram_filter_step11 = AnswerNgramFilter(
            min_score = 0.1,
            max_score = 1.0,
            ngrams = 5
        )

    def forward(self):

        self.question_filter_step1.run(
            storage = self.storage.step(),
            input_key = "instruction",
        )

        self.question_gen_step2.run(
            storage = self.storage.step(),
            input_key = "instruction",
        )

        self.question_filter_step3.run(
            storage = self.storage.step(),
            input_key = "instruction",
        )

        self.question_difficulty_classifier_step4.run(
            storage = self.storage.step(),
            input_key = "instruction",
            output_key = "question_difficulty"
        )

        self.question_category_classifier_step5.run(
            storage = self.storage.step(),
            input_key = "instruction",
            output_key = "question_category"
        )
        ############# branch #############
        # self.answer_pipeline_root_step6.run(
        #     storage = self.storage.step(),
        #     input_answer_key = "output",
        #     input_gt_key = "golden_answer"
        # )
        ############## answer #############
        self.answer_generator_step7.run(
            storage = self.storage.step(),
            input_key = "instruction", 
            output_key = "generated_cot"
        )
        self.answer_format_filter_step8.run(
            storage = self.storage.step(),
            input_key = "generated_cot",
        )
        self.answer_token_length_filter_step9.run(
            storage = self.storage.step(),
            input_key =  "generated_cot"
        )
        self.answer_groundtruth_filter_step10.run(
            storage = self.storage.step(),
            test_answer_key = "generated_cot",
            gt_answer_key =  "golden_answer"
        )
        self.answer_ngram_filter_step11.run(
            storage = self.storage.step(),
            question_key = "instruction",
            answer_key = "generated_cot"
        )

if __name__ == "__main__":
    model = ReasoningPipeline()
    model.forward()
