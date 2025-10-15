import os
import time
import re
from typing import List, Optional, Dict, Any, Union, Tuple

from dataflow.core import LLMServingABC
from dataflow.logger import get_logger

from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache
from lightrag.llm.ollama import ollama_embed
from lightrag.utils import EmbeddingFunc
from lightrag.kg.shared_storage import initialize_pipeline_status
import asyncio
from tqdm.asyncio import tqdm_asyncio

WORKING_DIR = "./LightRAG"

async def initialize_rag(
        llm_model_name, api_url, api_key, embed_model_name, embed_binding_host, embedding_dim, max_embed_tokens):

    async def llm_model_func(
        prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
    ) -> str:
        return await openai_complete_if_cache(
            llm_model_name,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            api_key=api_key,
            base_url=api_url,
            **kwargs,
        )

    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=embedding_dim,
            max_token_size=max_embed_tokens,
            func=lambda texts: ollama_embed(
                texts,
                embed_model=embed_model_name,
                host=embed_binding_host
            ),
        ),
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()
    return rag

class LightRAGServing(LLMServingABC):
    def __init__(self,
                 api_url: str = "https://api.openai.com/v1",
                 key_name_of_api_key: str = "DF_API_KEY",
                 llm_model_name: str = "gpt-4o",
                 embed_model_name: str = "bge-m3:latest",
                 embed_binding_host: str = "http://localhost:11434",
                 embedding_dim: int = 1024,
                 max_embed_tokens: int = 8192,
                 document_list: List[str] = []
                 ):
        self.rag: LightRAG = None
        self.api_url = api_url
        self.llm_model_name = llm_model_name
        self.embed_model_name = embed_model_name
        self.embed_binding_host = embed_binding_host
        self.embedding_dim = embedding_dim
        self.max_embed_tokens = max_embed_tokens
        self.logger = get_logger()
        self.document_list = document_list

        # config api_key in os.environ global, since safty issue.
        self.api_key = os.environ.get(key_name_of_api_key)
        if self.api_key is None:
            error_msg = f"Lack of `{key_name_of_api_key}` in environment variables. Please set `{key_name_of_api_key}` as your api-key to {api_url} before using APILLMServing_request."
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
    @classmethod
    async def create(cls, *args, **kwargs) -> "LightRAGServing":
        instance = cls(*args, **kwargs)
        if instance.rag is None:
            instance.rag = await initialize_rag(
                instance.llm_model_name,
                instance.api_url,
                instance.api_key,
                instance.embed_model_name,
                instance.embed_binding_host,
                instance.embedding_dim,
                instance.max_embed_tokens
            )
            try:
                instance.logger.info("Loading documents...")
                await instance.load_documents(instance.document_list)
                instance.logger.info("Documents processing completed.")
            except Exception as e:
                instance.logger.error(f"Error during documents processing: {e}\n")
                return
            
        return instance
    
    def start_serving(self):
        pass

    async def cleanup(self):
        if self.rag:
            storages = [
                self.rag.text_chunks,
                self.rag.fukk_docs,
                self.rag.entities_vdb,
                self.rag.relationships_vdb,
                self.rag.chunks_vdb,
                self.rag.chunk_entity_relation_graph,
                self.rag.doc_status,
                self.rag.llm_response_cache
            ]
            await asyncio.gather(*[s.drop() for s in storages], return_exceptions=True)

        await self.rag.finalize_storages()
    
    async def load_documents(self, document_paths: List[str]):
        tasks = []
        for path in document_paths:
            with open(path, "r", encoding="utf-8") as f:
                tasks.append(self.rag.ainsert(f.read()))
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def generate_from_input(self, user_inputs: List[str], system_prompt: str) -> List[str]:
        tasks = [
            self.rag.aquery(question, system_prompt=system_prompt, param=QueryParam(mode="hybrid"))
            for question in user_inputs
        ]
        responses = await tqdm_asyncio.gather(*tasks)
        return list(responses)
    