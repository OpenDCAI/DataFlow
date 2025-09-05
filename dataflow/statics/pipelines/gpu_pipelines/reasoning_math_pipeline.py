from dataflow.operators.reasoning import (
    ReasoningCategoryDatasetEvaluator,
    ReasoningDifficultyDatasetEvaluator,
    ReasoningQuestionGenerator,
    ReasoningAnswerGenerator,
)

from dataflow.operators.reasoning import (
    ReasoningQuestionFilter,
    ReasoningAnswerPipelineRootFilter,
    ReasoningAnswerFormatterFilter,
    ReasoningAnswerTokenLengthFilter,
    ReasoningAnswerGroundTruthFilter,
    ReasoningAnswerNgramFilter, 
)

from dataflow.prompts.reasoning.math import (
    MathQuestionFilterPrompt,
    MathAnswerGeneratorPrompt,
    MathQuestionSynthesisPrompt
)

from dataflow.utils.storage import FileStorage
from dataflow.serving import LocalModelLLMServing_vllm, LocalModelLLMServing_sglang

class ReasoningMath_GPUPipeline():
    def __init__(self):

        self.storage = FileStorage(
            first_entry_file_name="../example_data/ReasoningPipeline/pipeline_math_short.json",
            cache_path="./cache_local",
            file_name_prefix="dataflow_cache_step",
            cache_type="jsonl",
        )
        # use vllm as LLM serving
        self.llm_serving = LocalModelLLMServing_vllm(
            hf_model_name_or_path="Qwen/Qwen2.5-7B-Instruct", # set to your own model path
            vllm_tensor_parallel_size=1,
            vllm_max_tokens=8192,
        )
        # use SGLang as LLM serving
        # llm_serving = LocalModelLLMServing_sglang(
        #     hf_model_name_or_path="Qwen/Qwen2.5-7B-Instruct",
        #     sgl_dp_size=1, # data parallel size
        #     sgl_tp_size=1, # tensor parallel size
        #     sgl_max_tokens=1024,
        #     sgl_tensor_parallel_size=4
        # )

        self.question_filter_step1 = ReasoningQuestionFilter(
            system_prompt="You are an expert in evaluating mathematical problems. Follow the user's instructions strictly and output your final judgment in the required JSON format.",
            llm_serving=self.llm_serving,
            prompt_template=MathQuestionFilterPrompt()
        )
        self.question_gen_step2 =  ReasoningQuestionGenerator(
            num_prompts=3,
            llm_serving=self.llm_serving,
            prompt_template=MathQuestionSynthesisPrompt()
        )
        self.question_filter_step3 = ReasoningQuestionFilter(
            system_prompt="You are an expert in evaluating mathematical problems. Follow the user's instructions strictly and output your final judgment in the required JSON format.",
            llm_serving=self.llm_serving,
            prompt_template=MathQuestionFilterPrompt()
        )
        self.question_difficulty_classifier_step4 = ReasoningDifficultyDatasetEvaluator(
            llm_serving=self.llm_serving
        )
        self.question_category_classifier_step5 = ReasoningCategoryDatasetEvaluator(
            llm_serving=self.llm_serving
        )
        ########################## branch ############################
        # self.answer_pipeline_root_step6 = AnswerPipelineRoot()
        ########################## answer ############################
        self.answer_generator_step7 = ReasoningAnswerGenerator(
            llm_serving=self.llm_serving,
            prompt_template=MathAnswerGeneratorPrompt()
        )
        
        self.answer_format_filter_step8 = ReasoningAnswerFormatterFilter()
        
        self.answer_token_length_filter_step9 = ReasoningAnswerTokenLengthFilter(
            max_answer_token_length = 8192,
            tokenizer_dir = "Qwen/Qwen2.5-0.5B-Instruct"
        )
        
        self.answer_groundtruth_filter_step10 = ReasoningAnswerGroundTruthFilter()
        
        self.answer_ngram_filter_step11 = ReasoningAnswerNgramFilter(
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
            input_test_answer_key = "generated_cot",
            input_gt_answer_key =  "golden_answer"
        )
        self.answer_ngram_filter_step11.run(
            storage = self.storage.step(),
            input_question_key = "instruction",
            input_answer_key = "generated_cot"
        )

if __name__ == "__main__":
    model = ReasoningMath_GPUPipeline()
    model.forward()
