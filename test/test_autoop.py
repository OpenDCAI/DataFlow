from dataflow.pipeline import PipelineABC
from dataflow.operators.general_text import (
    LLMLanguageFilter,
)
from dataflow.operators.text_pt import MetaSampleEvaluator
from dataflow.operators.core_text import PromptedGenerator
from dataflow.serving import APILLMServing_request, LocalModelLLMServing_vllm, LocalHostLLMAPIServing_vllm
from dataflow.utils.storage import FileStorage

class AutoOPPipeline(PipelineABC):
    
    def __init__(self):
        super().__init__()
        self.storage = FileStorage(
            first_entry_file_name="./dataflow/example/GeneralTextPipeline/pt_input.jsonl",
            cache_path="./cache_autoop",
            file_name_prefix="dataflow_cache_auto_run",
            cache_type="jsonl",
        )
        self.llm_serving1 = LocalHostLLMAPIServing_vllm(
            hf_model_name_or_path="/mnt/public/model/Qwen/Qwen2.5-0.5B-Instruct/"
        )
        self.llm_serving2 = LocalHostLLMAPIServing_vllm(
            hf_model_name_or_path="/mnt/public/model/Qwen/Qwen2.5-0.5B-Instruct/"
        )
        self.op1 = PromptedGenerator(
            llm_serving=self.llm_serving1,
            system_prompt="请将以下内容翻译成中文：",
        )
        self.op2 = PromptedGenerator(
            llm_serving=self.llm_serving2,
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
    print(pipeline.op_runtimes)
    pipeline.forward()