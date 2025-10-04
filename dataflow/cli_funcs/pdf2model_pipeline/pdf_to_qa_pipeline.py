#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import os
from pathlib import Path
from dataflow.operators.knowledge_cleaning import (
    KBCChunkGeneratorBatch as CorpusTextSplitterBatch,
    FileOrURLToMarkdownConverterBatch,
    KBCTextCleanerBatch as KnowledgeCleanerBatch,
    KBCMultiHopQAGeneratorBatch as MultiHopQAGeneratorBatch,
)
from dataflow.utils.storage import FileStorage
from dataflow.serving import LocalModelLLMServing_vllm, LocalModelLLMServing_sglang


class KBCleaning_batchvllm_GPUPipeline():
    def __init__(self, cache_base="./"):
        # 处理cache_base相对路径
        cache_path = Path(cache_base)
        if not cache_path.is_absolute():
            caller_cwd = Path(os.environ.get('PWD', os.getcwd()))
            cache_path = caller_cwd / cache_path

        self.storage = FileStorage(
            first_entry_file_name=str(cache_path / ".cache" / "gpu" / "pdf_list.jsonl"),
            cache_path=str(cache_path / ".cache" / "gpu"),
            file_name_prefix="batch_cleaning_step",
            cache_type="json",
        )

        self.knowledge_cleaning_step1 = FileOrURLToMarkdownConverterBatch(
            intermediate_dir=str(cache_path / ".cache"),
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


def main():
    parser = argparse.ArgumentParser(description="PDF to QA Pipeline")
    parser.add_argument("--cache", default="./", help="Cache directory path")
    args = parser.parse_args()

    print("🚀 Starting KB Cleaning Pipeline...")
    print(f"📄 Input: {args.cache}.cache/gpu/pdf_list.jsonl")
    print(f"💾 Cache: {args.cache}.cache/gpu/")
    print(f"📤 Output: {args.cache}.cache/gpu/batch_cleaning_step_step4.json")
    print("-" * 60)

    model = KBCleaning_batchvllm_GPUPipeline(cache_base=args.cache)
    model.forward()


if __name__ == "__main__":
    main()