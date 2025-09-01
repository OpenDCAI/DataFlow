from dataflow.pipeline import PipelineABC
# from dataflow.operators.filter import (
    # LLMLanguageFilter,
# )
# from dataflow.operators.eval import MetaScorer
from dataflow.operators.core_text import PromptedGenerator
from dataflow.serving import APILLMServing_request, LocalModelLLMServing_vllm, LocalHostLLMAPIServing_vllm
from dataflow.utils.storage import FileStorage

class AutoOPPipeline(PipelineABC):
    
    def __init__(self):
        super().__init__()
        self.storage = FileStorage(
            first_entry_file_name="../dataflow/example/GeneralTextPipeline/pt_input.jsonl",
            cache_path="./cache",
            file_name_prefix="dataflow_cache_auto_run",
            cache_type="jsonl",
        )
        self.llm_serving = LocalModelLLMServing_vllm(
            hf_model_name_or_path="/mnt/public/model/huggingface/Qwen3-0.6B"
        )
        # self.llm_serving = LocalHostLLMAPIServing_vllm(
        #     hf_model_name_or_path="/mnt/public/model/Qwen/Qwen2.5-0.5B-Instruct/",
        #     vllm_gpu_memory_utilization=0.8
        # )
        self.op1 = PromptedGenerator(
            llm_serving=self.llm_serving,
            system_prompt="请将以下内容翻译成中文：",
        )
        self.op2 = PromptedGenerator(
            llm_serving=self.llm_serving,
            system_prompt="请将以下内容翻译成韩文：",
        )
        self.op3 = PromptedGenerator(
            llm_serving=self.llm_serving,
            system_prompt="请将以下内容翻译成日语："
        )
        
    def forward(self):
        self.op1.run(
            self.storage.step(),
            input_key='raw_content',
            # output_key='content_CN'
            output_key="raw_content"
        )
        self.op2.run(
            self.storage.step(),
            input_key='raw_content',
            # input_key="raw_content",
            output_key='content_JA'
        )
        self.op3.run(
            self.storage.step(),
            input_key='raw_content',
            output_key='content_KR'
        )
        
if __name__ == "__main__":
    pipeline = AutoOPPipeline()
    pipeline.compile()
    print(pipeline.llm_serving_list)
    print(pipeline.llm_serving_counter)
    pipeline.draw_graph(
        port=8081,
        hide_no_changed_keys=True
    )
    # print(pipeline.op_runtimes)
    # pipeline.forward()