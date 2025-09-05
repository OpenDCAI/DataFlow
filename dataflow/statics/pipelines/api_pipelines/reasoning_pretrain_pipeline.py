from dataflow.operators.reasoning import (
    ReasoningQuestionGenerator,
    ReasoningAnswerGenerator,
    ReasoningPretrainFormatConvertGenerator
)
from dataflow.prompts.reasoning.math import (
    MathQuestionFilterPrompt,
    MathQuestionSynthesisPrompt,
    MathAnswerGeneratorPrompt
)
from dataflow.operators.reasoning import ReasoningQuestionFilter, ReasoningAnswerNgramFilter, ReasoningAnswerPipelineRootFilter
from dataflow.utils.storage import FileStorage
from dataflow.serving import APILLMServing_request

# 这里或许未来可以有个pipeline基类
class Reasoning_APIPipeline_Pretrain():
    def __init__(self, llm_serving=None):

        self.storage = FileStorage(
            first_entry_file_name="../example_data/ReasoningPipeline/pipeline_math_short.json",
            cache_path="./cache_local",
            file_name_prefix="dataflow_cache_step",
            cache_type="jsonl",
        )
        if llm_serving is None:
            # use API server as LLM serving
            self.llm_serving = APILLMServing_request(
                    api_url="http://api.openai.com/v1/chat/completions",
                    model_name="gpt-4o",
                    max_workers=100
            )

            ## use local model as LLM serving
            # llm_serving = LocalModelLLMServing(
            #     # model_name_or_path="/data0/models/Qwen2.5-7B-Instruct", # set to your own model path
            #     model_name_or_path="/mnt/public/model/huggingface/Qwen2.5-7B-Instruct",
            #     tensor_parallel_size=4,
            #     max_tokens=1024,
            #     model_source="local"
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
        
        ########################## branch ############################
        self.answer_pipeline_root_step3 = ReasoningAnswerPipelineRootFilter()
        ########################## answer ############################
        self.answer_generator_step4 = ReasoningAnswerGenerator(
            llm_serving=self.llm_serving,
            prompt_template=MathAnswerGeneratorPrompt()
        )
        
        self.answer_ngram_filter_step5 = ReasoningAnswerNgramFilter(
            min_score = 0.1,
            max_score = 1.0,
            ngrams = 5
        )
        
        self.sft_to_pretrain_step6 = ReasoningPretrainFormatConvertGenerator()
                
        # 未来或许可以维护一个类似nn.sequential的容器，方便添加并实例化多个算子
    def forward(self):

        self.question_filter_step1.run(
            storage = self.storage.step(),
            input_key = "instruction",
        )

        self.question_gen_step2.run(
            storage = self.storage.step(),
            input_key = "instruction",
        )

        ############# branch #############
        self.answer_pipeline_root_step3.run(
            storage = self.storage.step(),
            input_answer_key = "output",
            input_gt_key = "golden_answer"
        )
        ############## answer #############
        self.answer_generator_step4.run(
            storage = self.storage.step(),
            input_key = "instruction", 
            output_key = "generated_cot"
        )
        self.answer_ngram_filter_step5.run(
            storage = self.storage.step(),
            input_question_key = "instruction",
            input_answer_key = "generated_cot"
        )
        self.sft_to_pretrain_step6.run(
            storage = self.storage.step(),
            input_read_key_question="instruction",
            input_read_key_answer="generated_cot",
            output_key="text",
            )

if __name__ == "__main__":
    pipeline = Reasoning_APIPipeline_Pretrain()
    pipeline.forward()

