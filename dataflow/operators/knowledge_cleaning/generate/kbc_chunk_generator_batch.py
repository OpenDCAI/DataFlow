import os
import json
from typing import Dict, List, Optional
from chonkie import (
    TokenChunker,
    SentenceChunker,
    SemanticChunker,
    RecursiveChunker
)
from tokenizers import Tokenizer
from transformers import AutoTokenizer
import pandas as pd
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC


@OPERATOR_REGISTRY.register()
class KBCChunkGeneratorBatch(OperatorABC):
    def __init__(self,
                 chunk_size: int = 512,
                 chunk_overlap: int = 50,
                 split_method: str = "token",
                 min_tokens_per_chunk: int = 100,
                 tokenizer_name: str = "bert-base-uncased",
                 ):
        # 必需参数检查
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.split_method = split_method
        self.min_tokens_per_chunk = min_tokens_per_chunk
        tokenizer_name = tokenizer_name
        # 初始化tokenizer和chunker
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.chunker = self._initialize_chunker()
        self.logger = get_logger()

    @staticmethod
    def get_desc(lang: str = "zh"):
        if (lang == "zh"):
            return (
                "CorpusTextSplitter是轻量级文本分割工具，",
                "支持词/句/语义/递归分块，",
                "可配置块大小、重叠和最小块长度",
            )
        elif (lang == "en"):
            return (
                "CorpusTextSplitter is a lightweight text segmentation tool",
                "that supports multiple chunking methods",
                "(token/sentence/semantic/recursive) with configurable size and overlap,",
                "optimized for RAG applications."
            )

    def _initialize_chunker(self):
        """Initialize the appropriate chunker based on method"""
        if self.split_method == "token":
            return TokenChunker(
                tokenizer=self.tokenizer,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
        elif self.split_method == "sentence":
            return SentenceChunker(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
        elif self.split_method == "semantic":
            return SemanticChunker(
                chunk_size=self.chunk_size,
            )
        elif self.split_method == "recursive":
            return RecursiveChunker(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
        else:
            raise ValueError(f"Unsupported split method: {self.split_method}")

    def _load_text(self, text_paths: List[str]) -> List[str]:
        """Load text from file list"""
        texts = []
        for text_path in text_paths:
            if not os.path.exists(text_path):
                self.logger.error(f"Input file not found: {text_path}")
                texts.append("")

            elif text_path.endswith('.txt') or text_path.endswith('.md') or text_path.endswith('.xml'):
                with open(text_path, 'r', encoding='utf-8') as f:
                    texts.append(f.read())

            elif text_path.endswith(('.json', '.jsonl')):
                with open(text_path, 'r', encoding='utf-8') as f:
                    data = json.load(f) if text_path.endswith(
                        '.json') else [json.loads(line) for line in f]
                text_fields = ['text', 'content', 'body']
                for field in text_fields:
                    if isinstance(data, list) and len(data) > 0 and field in data[0]:
                        texts.append("\n".join([item[field] for item in data]))
                    elif isinstance(data, dict) and field in data:
                        texts.append(data[field])
                if (field not in text_fields):
                    raise ValueError("No text field found in JSON input")
            else:
                raise ValueError(f"Unsupported file format for {text_path}")

        return texts

    def _validate_dataframe(self, dataframe: pd.DataFrame):
        forbidden_keys = [self.output_key]
        conflict = [k for k in forbidden_keys if k in dataframe.columns]
        if conflict:
            raise ValueError(
                f"The following column(s) already exist and would be overwritten: {conflict}")

    def run(self, storage: DataFlowStorage, input_key: str = "text_path", output_key: str = "chunk_path"):
        """Perform text splitting and save results"""
        # try:
        self.input_key = input_key
        self.output_key = output_key
        dataframe = storage.read("dataframe")
        self._validate_dataframe(dataframe)

        text_paths = dataframe[self.input_key].tolist()
        texts = self._load_text(text_paths)
        output_paths = []

        for i, text in enumerate(texts):
            if(text):
                # 计算总token数和最大限制
                tokens = self.tokenizer.encode(text)
                total_tokens = len(tokens)
                max_tokens = self.tokenizer.model_max_length  # 假设这是tokenizer的最大token限制
                print("max_tokens: ", self.tokenizer.model_max_length)

                if total_tokens <= max_tokens:
                    chunks = self.chunker(text)
                else:
                    # 计算需要分割的份数x（向上取整）
                    x = (total_tokens + max_tokens - 1) // max_tokens

                    # 按词数等分文本（近似分割）
                    words = text.split()  # 按空格分词
                    words_per_chunk = (len(words) + x - 1) // x  # 每份的词数

                    chunks = []
                    for j in range(0, len(words), words_per_chunk):
                        chunk_text = ' '.join(words[j:j+words_per_chunk])
                        chunks.extend(self.chunker(chunk_text))
                json_chunks = [{
                    "raw_chunk": chunk.text,
                } for chunk in chunks]

                output_dir = "/".join([os.path.dirname(text_paths[i]), "extract"])
                os.makedirs(output_dir, exist_ok=True)
                file_name = os.path.splitext(os.path.basename(text_paths[i]))[0]+'_chunk.json'
                output_path = os.path.join(output_dir, file_name)
                output_paths.append(output_path)

                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(json_chunks, f, ensure_ascii=False, indent=4)
                self.logger.info(
                    f"Successfully split {text_paths[i]} into {len(chunks)} chunks. Saved to {output_path}")
            else:
                output_paths.append("")

        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>")
        print(output_paths)
        print("<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        dataframe[self.output_key] = output_paths
        output_file = storage.write(dataframe)
        self.logger.info(
            f"Successfully split text into {len(chunks)} chunks. Saved to {output_file}")

        return [output_key]
