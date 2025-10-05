from __future__ import annotations
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import asyncio
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type
from huggingface_hub import HfApi
from datasets import load_dataset
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from langchain_core.tools import Tool

from dataflow.dataflowagent.promptstemplates.prompt_template import PromptsTemplateGenerator
from dataflow.dataflowagent.state import DataCollectionState
from dataflow.dataflowagent.utils import robust_parse_json
from dataflow.dataflowagent.toolkits.tool_manager import ToolManager
from dataflow import get_logger

log = get_logger()

class DataCollector:

    def __init__(self, 
                 model_name: Optional[str] = None,
                 temperature: float = 0.0,
                 max_tokens: int = 4096):
        """
        初始化Agent
        
        Args:
            model_name: 模型名称
            temperature: 模型温度
            max_tokens: 最大token数
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

        self.tool_mode = "auto"  # 默认工具选择模式，可扩展为 "auto", "required", "none"

        # Initialize HuggingFace API client
        self.hf_endpoint = 'https://hf-mirror.com'
        self.hf_api = HfApi(endpoint=self.hf_endpoint)

    @property
    def role_name(self) -> str:
        return "data_collector"
    
    @property
    def system_prompt_template_name(self) -> str:
        return "system_prompt_for_data_collection"
    
    @property
    def task_prompt_template_name(self) -> str:
        return "task_prompt_for_data_collection"
    

    def build_messages(self, state: DataCollectionState) -> List[BaseMessage]:
        """构建消息列表"""
        log.info("构建提示词消息...")
        
        ptg = PromptsTemplateGenerator(state.request.language)
        sys_prompt = ptg.render(self.system_prompt_template_name)
        
        task_params = {'user_query': state.request.target}
        task_prompt = ptg.render(self.task_prompt_template_name, **task_params)
        
        log.info(f"系统提示词: {sys_prompt}")
        log.info(f"任务提示词: {task_prompt}")
        
        messages = [
            SystemMessage(content=sys_prompt),
            HumanMessage(content=task_prompt),
        ]
        
        log.info("提示词消息构建完成")
        return messages
    
    def create_llm(self, state: DataCollectionState) -> ChatOpenAI:
        """创建LLM实例"""
        actual_model = self.model_name or state.request.model
        log.info(f"创建LLM实例，模型: {actual_model}")
        
        llm = ChatOpenAI(
            openai_api_base=state.request.chat_api_url,
            openai_api_key=state.request.api_key,
            model_name=actual_model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        
        return llm
    
    async def invoke(self, state: DataCollectionState) -> list:
        """调用LLM并处理响应"""
        log.info(f"{self.role_name} 调用LLM并处理响应...")
        
        messages = self.build_messages(state)
        llm = self.create_llm(state)
        
        try:
            answer_msg = await llm.ainvoke(messages)
            answer_text = answer_msg.content.strip()
            log.info(f'LLM调用成功并返回结果: {answer_text}')

            if answer_text == 'No valid keyword':
                raise Exception("No valid keyword")
            # Parse comma-separated keywords
            keywords = [k.strip() for k in answer_text.split(",")]
            return keywords
        except Exception as e:
            log.exception("提取关键词失败: %s", e)
            return []

    async def search_datasets(self, keywords: List[str], state: DataCollectionState) -> Dict[str, list]:
        """
        Asynchronously search for datasets on Hugging Face based on keywords.
        
        Args:
            keywords: List of keywords to search for
            
        Returns:
            List of dictionaries
        """
        results = {}

        tasks = []
        for keyword in keywords:
            tasks.append(self._search_for_keyword(keyword, results, state))

        await asyncio.gather(*tasks)
        
        return results

    async def _search_for_keyword(self, keyword: str, results: Dict[str, list], state: DataCollectionState):
        try:
            results[keyword] = []
            log.info(f"查找关键词 '{keyword}' 相关的数据集...")

            # Use the Hugging Face API to search for datasets
            datasets = await asyncio.to_thread(self.hf_api.list_datasets, search=keyword, limit=state.request.dataset_num_limit, size_categories=state.request.dataset_size_category)

            for dataset in datasets:
                results[keyword].append({
                    "id": dataset.id
                })

        except Exception as e:
            log.error(f"搜索关键词 '{keyword}' 相关数据集时出错: {e}")

    async def download_datasets(self, state: DataCollectionState) -> Dict[str, list]:
        download_results = {}
        tasks = []

        for keyword, dataset_infos in state.datasets.items():
            if not dataset_infos:
                log.info(f"未找到关键词 '{keyword}' 相关的数据集，跳过下载")
                continue
            
            log.info(f"开始下载关键词 '{keyword}' 相关的数据集...")
            download_dir = os.path.join(state.request.download_dir, keyword.replace(" ", "_"), 'tmp')
            os.makedirs(download_dir, exist_ok=True)
            download_results[keyword] = []

            for dataset in dataset_infos:
                dataset_id = dataset["id"]
                tasks.append(self._download_dataset(download_dir, dataset_id, keyword, download_results))

        await asyncio.gather(*tasks)

        return download_results

    async def _download_dataset(self, download_dir: str, dataset_id: str, keyword: str, download_results: Dict[str, list]):
        try:
            log.info(f"下载数据集 {dataset_id}...")
            dataset_dir = os.path.join(download_dir, dataset_id.replace("/", "_"))
            os.makedirs(dataset_dir, exist_ok=True)

            await asyncio.to_thread(load_dataset, dataset_id, cache_dir=dataset_dir)

            download_results[keyword].append({
                "dataset_id": dataset_id,
                "success": True,
                "message": f"Successfully downloaded {dataset_id} to {dataset_dir}"
            })
            log.info(f"数据集 {dataset_id} 下载成功，保存至 {dataset_dir}")

        except Exception as e:
            download_results[keyword].append({
                "dataset_id": dataset_id,
                "success": False,
                "message": str(e)
            })
            log.error(f"下载数据集 {dataset_id} 失败: {e}")


    async def execute(self, state: DataCollectionState, **kwargs) -> DataCollectionState:
        """
        执行入口
        
        Args:
            state: DataCollectionState实例
            **kwargs: 额外参数
            
        Returns:
            更新后的DataCollectionState
        """
        log.info(f"开始执行 {self.role_name}")

        # Step 1: Extract keywords from the query
        keywords = await self.invoke(state)
        if not keywords:
            log.warning("未提取到有效关键词，结束执行")
            return state
        state.keywords = keywords
        log.info(f"提取关键词: {', '.join(keywords)}")

        # Step 2: Search for datasets
        datasets_found = await self.search_datasets(keywords, state)
        state.datasets = datasets_found
        
        if all(len(lst) == 0 for lst in datasets_found.values()):
            log.warning("未找到相关数据集")
            return state

        # Step 3: Download datasets
        os.makedirs(state.request.download_dir, exist_ok=True)
        download_results = await self.download_datasets(state)
        state.downloads = download_results

        # 统计下载结果
        for keyword, results in download_results.items():
            success_count = sum(1 for r in results if r["success"])
            fail_count = sum(1 for r in results if not r["success"])
            log.info(f"关键词 '{keyword}' 下载结果: 成功 {success_count} 个，失败 {fail_count} 个")

        log.info(f"{self.role_name} 执行完成")
        return state
    
async def data_collection(
    state: DataCollectionState,
    model_name: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    **kwargs,
) -> DataCollectionState:
    data_collector = DataCollector(
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return await data_collector.execute(state, **kwargs)
