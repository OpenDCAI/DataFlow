import random
import json
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
        self.num_dialogs_per_intent = num_dialogs_per_intent # Based on the topic_dict in the existing prompt, it is recommended to set the value to below 1000 (which can generate 9000 conversation data). Otherwise, it is recommended to add more topic_dict in dataflow.prompts.general_text.ConsistentChatPrompt to increase data richness
        self.num_turns_per_dialog = num_turns_per_dialog
        self.temperature = temperature
        self.prompt = ConsistentChatPrompt()
        self.logger.info(f'{self.__class__.__name__} initialized.')
    
    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "根据预置主题和人类意图，两阶段从0合成多轮对话格式数据（合成数量大于9000时建议增加标签数量）。\n"
                "输入参数：\n"
                "- llm_serving：LLM服务对象，需实现LLMServingABC接口\n"
                "- num_dialogs_per_intent：每个意图生成的对话数量，默认20\n"
                "- num_turns_per_dialog：每个对话的轮次数量，默认6\n"
                "- temperature：生成温度，控制输出随机性，默认0.9\n"
                "输出参数：\n"
                "- 包含category和conversation字段的DataFrame，其中conversation为多轮对话列表"
            )
        elif lang == "en":
            return (
                "Two-stage generation of multi-turn dialogue data from scratch based on predefined topics and human intents (for over 9000 samples, consider increasing the number of tags).\n"
                "Input Parameters:\n"
                "- llm_serving: LLM serving object implementing LLMServingABC interface\n"
                "- num_dialogs_per_intent: Number of dialogs generated per intent, default 20\n"
                "- num_turns_per_dialog: Number of turns per dialog, default 6\n"
                "- temperature: Sampling temperature for generation, default 0.9\n"
                "Output Parameters:\n"
                "- DataFrame containing 'category' and 'conversation' fields, where conversation is a list of multi-turn dialogues"
            )
        else:
            return "Two-stage generation of multi-turn dialogue data from scratch based on predefined topics and human intents."

    def run(self, storage: DataFlowStorage):
        all_query_prompts = []
        
        # Step 1: Generate all queries using LLM
        for intent, info_flows in self.prompt.get_intent_categories().items():
            for _ in range(self.num_dialogs_per_intent):
                info_flow = random.choice(info_flows)
                topic = random.choice(self.prompt.topic_dict[intent])
                query_prompt = self.prompt.get_query_prompt(info_flow, topic)
                all_query_prompts.append(query_prompt)
        # Step 2: Generate queries by calling llm_serving once
        self.logger.info("Generating queries...")
        queries_list = self.llm_serving.generate_from_input(user_inputs=all_query_prompts)
        valid_queries = []
        cnt = 0
        for queries_str in queries_list:
            try:
                if not isinstance(queries_str, str):
                    raise ValueError("Invalid response type") 
                clean_queries_str = queries_str.replace("```json", "").replace("```", "").strip()  
                queries = json.loads(clean_queries_str)
                valid_queries.append(queries)
            except (json.JSONDecodeError, ValueError) as e:
                cnt += 1
                self.logger.debug(f'Json parse failed counts: {cnt} (Model generation error)')
                continue
        all_response_prompts = []
        for queries in valid_queries:
            category = queries.get("category")
            turns = queries.get("turns")
            all_response_prompts.append(self.prompt.get_response_prompt(topic=category, queries=turns))
        self.logger.info("Generating responses...")
        responses_list = self.llm_serving.generate_from_input(user_inputs=all_response_prompts)

        final_queries = []
        final_responses = []
        cnt = 0
        for query, responses_str in zip(valid_queries, responses_list):
            try:
                if not isinstance(responses_str, str):
                    raise ValueError("Invalid response type")  
                clean_responses_str = responses_str.replace("```json", "").replace("```", "").strip()
                responses = json.loads(clean_responses_str) 
                final_queries.append(query)
                final_responses.append(responses)
            except (json.JSONDecodeError, ValueError) as e:
                cnt += 1
                self.logger.debug(f'Json parse failed counts: {cnt} (Model generation error): {str(e)}')
                continue

        formatted_data = []

        for query_data, response_data in zip(final_queries, final_responses):
            if isinstance(response_data, dict):
                response_data = response_data.get('responses', [])
            try:
                category = query_data['category']
                turns = query_data['turns']
                
                # Ensure the number of turns matches the number of responses
                num_user_turns = len(turns)
                num_assistant_responses = len(response_data)
                
                if num_user_turns > num_assistant_responses:
                    turns = turns[:num_assistant_responses]

                conversation = []
                for i in range(len(turns)):
                    conversation.append({"role": "user", "value": turns[i]})
                    if i < len(response_data):
                        conversation.append({"role": "assistant", "value": response_data[i]['response']})
                
                # Ensure conversation does not end with a user message
                if conversation and conversation[-1]["role"] == "user":
                    conversation.pop()

                if not conversation:
                    continue

                formatted_data.append({
                    "category": category,
                    "conversation": conversation
                })
            except Exception as e:
                self.logger.debug(f"Error processing category '{query_data.get('category', 'Unknown')}': {e}")
                continue 
        self.logger.info(f'Number of synthesized dialogues: {len(formatted_data)}')
        
        df = pd.DataFrame(formatted_data)
        storage.write(df)
        self.logger.info(f'Number of synthesized dialogues: {len(df)} written to storage as DataFrame')
        return df
