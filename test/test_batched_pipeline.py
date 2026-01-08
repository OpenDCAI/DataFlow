import re
from dataflow.pipeline import BatchedPipelineABC
from dataflow.operators.core_text import PromptedGenerator
from dataflow.serving import APILLMServing_request
from dataflow.utils.storage import BatchedFileStorage

class AutoOPPipeline(BatchedPipelineABC):
    
    def __init__(self):
        super().__init__()
        self.storage = BatchedFileStorage(
            first_entry_file_name="./dataflow/example/GeneralTextPipeline/pt_input.jsonl",
            cache_path="./cache_autoop",
            file_name_prefix="dataflow_cache_auto_run",
            cache_type="jsonl",
        )
        self.llm_serving1 = APILLMServing_request(
                api_url="http://123.129.219.111:3000/v1/chat/completions",
                model_name="gpt-5-mini",
                max_workers=100,
        )
        self.op1 = PromptedGenerator(
            llm_serving=self.llm_serving1,
            system_prompt="请将以下内容翻译成中文：",
        )
        self.op2 = PromptedGenerator(
            llm_serving=self.llm_serving1,
            system_prompt="请将以下内容翻译成韩文：",
        )
        self.op3 = PromptedGenerator(
            llm_serving=self.llm_serving1,
            system_prompt="请将以下内容翻译成日语："
        )
        
    def forward(self):
        self.op1.run(
            self.storage.step(),
            input_key='raw_content',
            output_key='content_cn1'
        )
        self.op2.run(
            self.storage.step(),
            input_key='raw_content',
            output_key='content_cn2'
        )
        self.op3.run(
            self.storage.step(),
            input_key='raw_content',
            output_key='content_cn3'
        )
        
if __name__ == "__main__":
    pipeline = AutoOPPipeline()
    pipeline.compile()
    pipeline.forward(batch_size=2, resume_from_last=True)
    pipeline.forward(batch_size=2, resume_from_last=False)  # should overwrite