from dataflow.operators.generate import (
    CorpusTextSplitterBatch,
    FileOrURLToMarkdownConverterBatch,
    KnowledgeCleanerBatch,
    MultiHopQAGeneratorBatch,
)
from dataflow.utils.storage import FileStorage
from dataflow.serving import LocalModelLLMServing_vllm, LocalModelLLMServing_sglang


class KBCleaning_batchvllm_GPUPipeline():
    def __init__(self):
        self.storage = FileStorage(
            first_entry_file_name="./input/pdf_list.jsonl",  # 输入：PDF列表文件
            cache_path="./.cache/gpu",  # 缓存路径
            file_name_prefix="batch_cleaning_step",  # 文件前缀
            cache_type="json",  # 缓存格式
        )

        self.knowledge_cleaning_step1 = FileOrURLToMarkdownConverterBatch(
            intermediate_dir="./input",  # 中间文件目录
            lang="en",
            mineru_backend="pipeline",
        )

        self.knowledge_cleaning_step2 = CorpusTextSplitterBatch(
            split_method="token",
            chunk_size=512,
            tokenizer_name="Qwen/Qwen2.5-7B-Instruct",
        )

    def forward(self):
        """执行完整的Pipeline流程"""
        print("🔄 Step 1: File/URL to Markdown conversion...")
        self.knowledge_cleaning_step1.run(
            storage=self.storage.step(),
        )

        print("🔄 Step 2: Text splitting into chunks...")
        self.knowledge_cleaning_step2.run(
            storage=self.storage.step(),
        )

        print("🔄 Starting LLM serving...")
        self.llm_serving = LocalModelLLMServing_vllm(
            hf_model_name_or_path="Qwen/Qwen2.5-7B-Instruct",
            vllm_max_tokens=2048,
            vllm_tensor_parallel_size=1,
            vllm_gpu_memory_utilization=0.6,
            vllm_repetition_penalty=1.2
        )

        self.knowledge_cleaning_step3 = KnowledgeCleanerBatch(
            llm_serving=self.llm_serving,
            lang="en"
        )

        self.knowledge_cleaning_step4 = MultiHopQAGeneratorBatch(
            llm_serving=self.llm_serving,
            lang="en"
        )

        print("🔄 Step 3: Knowledge cleaning...")
        self.knowledge_cleaning_step3.run(
            storage=self.storage.step(),
        )

        print("🔄 Step 4: Multi-hop QA generation...")
        self.knowledge_cleaning_step4.run(
            storage=self.storage.step(),
        )

        print("✅ Pipeline completed! Output saved to: ./.cache/gpu/batch_cleaning_step_step4.json")


if __name__ == "__main__":
    print("🚀 Starting KB Cleaning Pipeline...")
    print("📄 Input: ./input/pdf_list.jsonl")
    print("💾 Cache: ./.cache/gpu/")
    print("📤 Output: ./.cache/gpu/batch_cleaning_step_step4.json")
    print("-" * 60)

    model = KBCleaning_batchvllm_GPUPipeline()
    model.forward()