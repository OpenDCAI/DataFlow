from dataflow.operators.generate import *
from dataflow.operators.filter import *
from dataflow.utils.storage import FileStorage
from dataflow.serving import APILLMServing_request, LocalModelLLMServing

# 这里或许未来可以有个pipeline基类
class ReasoningPipeline_Pretrain():
    def __init__(self, llm_serving=None):

        self.storage = FileStorage(
            first_entry_file_name="../dataflow/example/ReasoningPipeline/pipeline_math_short.json",
            cache_path="./cache_local",
            file_name_prefix="dataflow_cache_step",
            cache_type="jsonl",
        )
        if llm_serving is None:
            # use API server as LLM serving
            llm_serving = APILLMServing_request(
                    api_url="http://123.129.219.111:3000/v1/chat/completions",
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
        
        self.question_filter_step1 = QuestionFilter(
            system_prompt="You are an expert in evaluating mathematical problems. Follow the user's instructions strictly and output your final judgment in the required JSON format.",
            llm_serving=llm_serving
        )
        self.question_gen_step2 =  QuestionGenerator(
            num_prompts=3,
            llm_serving=llm_serving
        )
        
        ########################## branch ############################
        self.answer_pipeline_root_step3 = AnswerPipelineRoot()
        ########################## answer ############################
        self.answer_generator_step4 = AnswerGenerator(
            llm_serving=llm_serving
        )
        
        self.answer_ngram_filter_step5 = AnswerNgramFilter(
            min_score = 0.1,
            max_score = 1.0,
            ngrams = 5
        )
        
        self.sft_to_pretrain_step6 = PretrainFormatConverter()
                
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
            question_key = "instruction",
            answer_key = "generated_cot"
        )
        self.sft_to_pretrain_step6.run(
            storage = self.storage.step(),
            read_key_question="instruction",
            read_key_answer="generated_cot",
            output_key="text",
            )

if __name__ == "__main__":
    # For testing the ReasoningPipeline_Pretrain
    pipeline = ReasoningPipeline_Pretrain()
    pipeline.forward()

