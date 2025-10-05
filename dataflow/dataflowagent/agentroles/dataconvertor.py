from __future__ import annotations

import asyncio
import os
import re
import json
from abc import ABC, abstractmethod
from collections import Counter
from typing import Any, Dict, List, Optional, Type
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



class DataConvertor:

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

    @property
    def role_name(self) -> str:
        return "data_convertor"
    
    @property
    def system_prompt_template_name(self) -> str:
        return "system_prompt_for_data_conversion"
    
    @property
    def task_prompt_template_name(self) -> str:
        return "task_prompt_for_data_conversion"
    

    def build_messages(self, state: DataCollectionState, column_names: List[str], sample_record: Dict[str, Any]) -> List[BaseMessage]:
        """构建消息列表"""
        log.info("构建提示词消息...")
        
        ptg = PromptsTemplateGenerator(state.request.language)
        sys_prompt = ptg.render(self.system_prompt_template_name)
        
        task_params = {'column_names': column_names, 'first_row': sample_record}
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
    
    async def invoke(self, state: DataCollectionState, column_names: List[str], sample_record: Dict[str, Any]) -> Dict[str, Any]:
        """调用LLM并处理响应"""
        log.info(f"{self.role_name} 调用LLM并处理响应...")
        
        messages = self.build_messages(state, column_names, sample_record)
        llm = self.create_llm(state)
        
        try:
            answer_msg = await llm.ainvoke(messages)
            answer_text = answer_msg.content.strip()
            log.info(f'LLM调用成功并返回结果: {answer_text}')

            pattern = r'```json([\s\S]*?)```'
            match = re.search(pattern, answer_text).group(1).strip() if re.search(pattern, answer_text) else answer_text

            try:
                annotation_result = json.loads(match)
                return annotation_result
            except json.JSONDecodeError as e:
                log.exception(f"解析GPT响应为JSON失败: 内容为{match}")
                raise ValueError(f"Failed to parse GPT response as JSON: {e}")
            
        except Exception as e:
            log.exception("数据集标注失败: %s", e)
            raise Exception(f"Error during dataset annotation: {e}")

    
    def record_summary(self, state):
        info = ""

        if not state.keywords:
            info += "Sorry, I couldn't extract any valid keywords from your request."
            return info
        info += f"Extracted keywords: {', '.join(state.keywords)}\n\n"

        if all(len(lst) == 0 for lst in state.datasets.values()):
            info += "No datasets were found matching your keywords."
            return info
        
        info += "Datasets found:\n"
        for keyword, dataset_infos in state.datasets.items():
            if not dataset_infos:
                info += f"- No datasets found for keyword: {keyword}\n"
                continue
            download_infos = state.downloads[keyword]
            info += f"- {len(download_infos)} datasets found for keyword: {keyword}\n"
            for download_info in download_infos:
                status = "Download succeeded" if download_info['success'] else "Download failed"
                info += f"  - {download_info['dataset_id']}: {status}\n"
        info += "\n"

        info += "Post-processing summary:\n"
        for keyword, sources in state.sources.items():
            info += f"- Keyword: {keyword}\n"
            info += f"-- Total count:\tPT {sum(item[1] for item in sources['PT'])} records, SFT {sum(item[1] for item in sources['SFT'])} records\n"
            if not sources['PT'] and not sources['SFT']:
                info += "  No datasets were successfully post-processed.\n"
                continue
            info += "-- Source details:\n"
            for category, datasets in sources.items():
                if datasets:
                    info += f"-- {category}:\t"
                    for dataset_id, record_count in datasets:
                        info += f"{dataset_id}: {record_count} records\t"
                    info += "\n"
                else:
                    info += f"-- {category}: No datasets processed\n"
        

        info = info.strip()
        log.info("处理结果汇总:\n" + info)
        with open(os.path.join(state.request.download_dir, "summary.txt"), 'w') as f:
            f.write(info)

    async def execute(self, state: DataCollectionState, **kwargs) -> DataCollectionState:
        """
        执行入口
        
        Args:
            state: DataCollectionState实例
            **kwargs: 额外参数
            
        Returns:
            更新后的DataCollectionState
        """
        log.info(f"{self.role_name} 开始执行...")

        # Step 1: Convert datasets
        for keyword in state.keywords:
            if keyword not in state.downloads.keys() or not Counter([res['success'] for res in state.downloads[keyword]])[True]:
                state.sources[keyword] = {'PT': [], 'SFT': []}
                continue
            
            data_sources = {'PT': [], 'SFT': []}

            data_dir = os.path.join(state.request.download_dir, keyword.replace(" ", "_"))
            for dataset in state.downloads[keyword]:
                if not dataset['success']:
                    continue
                dataset_id = dataset['dataset_id']
                try:
                    data = load_dataset(os.path.join(data_dir, 'tmp', dataset_id.replace("/", "_")))
                    for split, data_content in data.items():
                        annotation_result = await self.invoke(state, data_content.column_names, data_content[0])
                        category = annotation_result.get('category', 'Unknown')
                        if category == 'PT':
                            data_file = os.path.join(data_dir, 'PT.jsonl')
                            with open(data_file, 'a') as f:
                                for row in data_content:
                                    text = row[annotation_result['text']]
                                    json_obj = {'text': text}
                                    f.write(json.dumps(json_obj) + '\n')
                            data_sources['PT'].append((f'{dataset_id}_({split})', len(data_content)))
                            log.info(f"数据集 {dataset_id}, split {split} 被分类为 PT，包含 {len(data_content)} 条记录。")

                        elif category == 'SFT':
                            data_file = os.path.join(data_dir, 'SFT.jsonl')
                            with open(data_file, 'a') as f:
                                for row in data_content:
                                    question = row[annotation_result['question']]
                                    output = row[annotation_result['output']] if annotation_result['output'] else None
                                    answer = row[annotation_result['answer']]
                                    json_obj = {
                                        'question': question,
                                        'output': output,
                                        'answer': answer
                                    }
                                    f.write(json.dumps(json_obj) + '\n')
                            data_sources['SFT'].append((f'{dataset_id}_({split})', len(data_content)))
                            log.info(f"数据集 {dataset_id}, split {split} 被分类为 SFT，包含 {len(data_content)} 条记录。")

                        else:
                            raise ValueError(f"Invalid category '{category}'")
                            
                except Exception as e:
                    log.error(f"处理数据集 {dataset_id} 时出错: {e}, 跳过该数据集")
                    continue

            state.sources[keyword] = data_sources
        
        # Step 2: Record summary
        self.record_summary(state)

        log.info(f"{self.role_name} 执行完成")
        return state
    

async def data_conversion(
    state: DataCollectionState,
    model_name: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    **kwargs,
) -> DataCollectionState:
    data_collector = DataConvertor(
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return await data_collector.execute(state, **kwargs)