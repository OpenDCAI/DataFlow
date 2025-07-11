import random
import json
import os
from dataflow.serving import APILLMServing_request
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage
import pandas as pd
from dataflow.core import LLMServingABC
from dataflow.prompts.general_text import ConsistentChatPrompt

@OPERATOR_REGISTRY.register()
class ConsistentChatGenerator(OperatorABC):
    def __init__(self, llm_serving: LLMServingABC = None, num_dialogs_per_intent = 20, num_turns_per_dialog = 6, temperature = 0.9):
        self.logger = get_logger()
        self.logger.info(f'Initializing {self.__class__.__name__}...')
        self.llm_serving = llm_serving
        self.num_dialogs_per_intent = num_dialogs_per_intent
        self.num_turns_per_dialog = num_turns_per_dialog
        self.temperature = temperature
        self.prompt = ConsistentChatPrompt()
        self.logger.info(f'{self.__class__.__name__} initialized.')
            
    def run(self):
        all_query_prompts = []
        
        # Step 1: Generate all queries using LLM
        for intent, info_flows in self.prompt.get_intent_categories().items():
            for _ in range(self.num_dialogs_per_intent):
                info_flow = random.choice(info_flows)
                topic = random.choice(self.prompt.topic_dict[intent])
                query_prompt = self.prompt.get_query_prompt(info_flow, topic)
                all_query_prompts.append(query_prompt)
        # Step 2: Generate queries by calling llm_serving once
        queries_list = self.llm_serving.generate_from_input(user_inputs=all_query_prompts)
        valid_queries = []
        cnt = 0
        for queries_str in queries_list:
            try:
                clean_queries_str = queries_str.replace("```json", "").replace("```", "").strip()  
                queries = json.loads(clean_queries_str)  # 解析成字典格式
                valid_queries.append(queries)
            except json.JSONDecodeError:
                cnt += 1
                self.logger.debug(f'Json parse failed counts: {cnt} (Model generation error)')
                continue
        all_response_prompts = []
        for queries in valid_queries:
            category = queries.get("category")
            turns = queries.get("turns")
            all_response_prompts.append(self.prompt.get_response_prompt(topic=category, queries=turns))
        responses_list = self.llm_serving.generate_from_input(user_inputs=all_response_prompts)

        final_queries = []
        final_responses = []
        cnt = 0
        for query, responses_str in zip(valid_queries, responses_list):
            try:
                clean_responses_str = responses_str.replace("```json", "").replace("```", "").strip()  
                responses = json.loads(clean_responses_str) 
                final_queries.append(query)
                final_responses.append(responses)
            except json.JSONDecodeError:
                cnt += 1
                self.logger.debug(f'Json parse failed counts: {cnt} (Model generation error)')
                continue
        formatted_data = []

        for query_data, response_data in zip(final_queries, final_responses):
            if isinstance(response_data, dict):
                response_data = response_data.get('responses', [])
            try:
                category = query_data['category']
                turns = query_data['turns']
                conversation = []
                for i in range(len(turns)):
                    conversation.append({"role": "user", "value": turns[i]})
                    if i < len(response_data):
                        conversation.append({"role": "assistant", "value": response_data[i]['response']})
                formatted_data.append({
                    "category": category,
                    "conversation": conversation
                })
            except Exception as e:
                self.logger.debug(f"Error processing category '{query_data.get('category', 'Unknown')}': {e}")
                continue 
        self.logger.info(f'Number of synthesized dialogues: {len(formatted_data)}')
        
        output_filename = "generated_dialogs.json"
        with open(output_filename, "w") as f:
            json.dump(formatted_data, f, indent=4)

        print(f"Data generated and saved to {output_filename}")
        return formatted_data
