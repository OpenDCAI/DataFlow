#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import os
from pathlib import Path
from dataflow.operators.knowledge_cleaning import (
    KBCChunkGeneratorBatch as CorpusTextSplitterBatch,
    KBCTextCleanerBatch as KnowledgeCleanerBatch,
    KBCMultiHopQAGeneratorBatch as MultiHopQAGeneratorBatch,
)
# from dataflow.operators.knowledge_cleaning import (
#     CorpusTextSplitterBatch,
#     KnowledgeCleanerBatch,
#     MultiHopQAGeneratorBatch,
# )
from dataflow.utils.storage import FileStorage
from dataflow.serving import LocalModelLLMServing_vllm
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['ONNXRUNTIME_THREAD_AFFINITY'] = 'false'

class Text2QAPipeline:
    def __init__(self, cache_base="./"):
        # 处理cache_base相对路径
        cache_path = Path(cache_base)
        if not cache_path.is_absolute():
            caller_cwd = Path(os.environ.get('PWD', os.getcwd()))
            cache_path = caller_cwd / cache_path

        self.storage = FileStorage(
            first_entry_file_name=str(cache_path / ".cache" / "gpu" / "text_input.jsonl"),
            cache_path=str(cache_path / ".cache" / "gpu"),
            file_name_prefix="text2qa_step",
            cache_type="json",
        )

        self.text_splitting_step = CorpusTextSplitterBatch(
            split_method="token",
            chunk_size=512,
            tokenizer_name="Qwen/Qwen2.5-7B-Instruct",
        )

    def forward(self):
        """执行完整的Pipeline流程"""
        print("Step 1: Text splitting into chunks...")
        self.text_splitting_step.run(
            storage=self.storage.step(),
        )

        print("Starting LLM serving...")
        self.llm_serving = LocalModelLLMServing_vllm(
            hf_model_name_or_path="Qwen/Qwen2.5-7B-Instruct",
            vllm_max_tokens=2048,
            vllm_tensor_parallel_size=1,
            vllm_gpu_memory_utilization=0.6,
            vllm_repetition_penalty=1.2
        )

        self.knowledge_cleaning_step = KnowledgeCleanerBatch(
            llm_serving=self.llm_serving,
            lang="en"
        )

        self.qa_generation_step = MultiHopQAGeneratorBatch(
            llm_serving=self.llm_serving,
            lang="en"
        )

        print("Step 2: Knowledge cleaning...")
        self.knowledge_cleaning_step.run(
            storage=self.storage.step(),
        )

        print("Step 3: Multi-hop QA generation...")
        self.qa_generation_step.run(
            storage=self.storage.step(),
        )

        print("Pipeline completed! Output saved to: ./.cache/gpu/text2qa_step_step3.json")


def main():
    parser = argparse.ArgumentParser(description="Text to QA Pipeline")
    parser.add_argument("--cache", default="./", help="Cache directory path")
    args = parser.parse_args()

    print("Starting Text to QA Pipeline...")
    print(f"Input: {args.cache}.cache/gpu/text_input.jsonl")
    print(f"Cache: {args.cache}.cache/gpu/")
    print(f"Output: {args.cache}.cache/gpu/text2qa_step_step3.json")
    print("-" * 60)

    model = Text2QAPipeline(cache_base=args.cache)
    model.forward()


if __name__ == "__main__":
    main()