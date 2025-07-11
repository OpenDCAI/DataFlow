from dataflow.operators.generate import (
    AtomicTaskGenerator,
    DepthQAGenerator,
    WidthQAGenerator
)

from dataflow.operators.filter import *
from dataflow.utils.storage import FileStorage
from dataflow.serving import APILLMServing_request, LocalModelLLMServing
from dataflow.core import LLMServingABC

# 这里或许未来可以有个pipeline基类
class TaskCraftPipeline():
    def __init__(self, llm_serving:LLMServingABC=None):

        self.storage = FileStorage(
            first_entry_file_name="../dataflow/example/AgenticRAGPipeline/pipeline_small_chunk.json",
            cache_path="./cache_local",
            file_name_prefix="taskcraft_test",
            cache_type="jsonl",
        )

        # use API server as LLM serving
        llm_serving = APILLMServing_request(
                api_url="http://123.129.219.111:3000/v1/chat/completions",
                model_name="gpt-4o-mini",
                max_workers=50
        )

        self.task_step1 = AtomicTaskGenerator(
            llm_serving=llm_serving
        )

        self.task_step2 = DepthQAGenerator(
            llm_serving = llm_serving
        )
        

        ### Either use DepthQAGenerator or WidthQAGenerator
        
        # self.task_step3 = WidthQAGenerator(
        #     llm_serving = llm_serving
        # )
        
    # 未来或许可以维护一个类似nn.sequential的容器，方便添加并实例化多个算子
    def forward(self):

        self.task_step1.run(
            storage = self.storage.step(),
            input_key = "text",
        )

        self.task_step2.run(
            storage = self.storage.step(),
            input_key= "question",
            output_key="depth_question"
        )
        
        # self.task_step3.run(
        #     storage = self.storage.step(),
        #     input_question_key = "question",
        #     input_identifier_key= "identifier",
        #     input_answer_key = "refined_answer"
        # )
        
if __name__ == "__main__":
    model = TaskCraftPipeline()
    model.forward()
