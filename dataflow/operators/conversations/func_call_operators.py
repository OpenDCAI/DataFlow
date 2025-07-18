import re
import pandas as pd
import numpy as np
from tqdm import tqdm
from dataflow.core import OperatorABC, LLMServingABC
from dataflow.utils.storage import DataFlowStorage
from dataflow.prompts.func_call import FuncCallPrompt
from dataflow.logger import get_logger
from dataflow.utils.registry import OPERATOR_REGISTRY

@OPERATOR_REGISTRY.register()
class ScenarioExtractor(OperatorABC):
    
    def __init__(self, llm_serving: LLMServingABC):
        self.logger = get_logger()
        self.prompt = FuncCallPrompt()
        self.llm_serving = llm_serving
        self.logger.info(f"Initializing {self.__class__.__name__}...")
    
    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "从对话内容中提取场景信息，使用LLM服务分析对话并生成场景描述。\n"
                "输入参数：\n"
                "- llm_serving：LLM服务对象，需实现LLMServingABC接口\n"
                "- input_chat_key：对话内容字段名\n"
                "- output_key：输出场景字段名，默认'scenario'\n"
                "输出参数：\n"
                "- 包含提取场景信息的DataFrame\n"
                "- 包含输出字段名的列表"
            )
        elif lang == "en":
            return (
                "Extract scenario information from conversation content using LLM service to analyze dialogues and generate scenario descriptions.\n"
                "Input Parameters:\n"
                "- llm_serving: LLM serving object implementing LLMServingABC interface\n"
                "- input_chat_key: Field name for conversation content\n"
                "- output_key: Field name for output scenario, default 'scenario'\n"
                "Output Parameters:\n"
                "- DataFrame containing extracted scenario information\n"
                "- List containing output field name"
            )
        else:
            return "Extract scenario information from conversation content using LLM service."
    
    def _reformat_prompt(self, dataframe: pd.DataFrame):
        formatted_prompts = [self.prompt.extract_scenario_prompt(conversation=item) for item in tqdm(dataframe[self.input_chat_key], desc=f"Reformatting prompts...")]

        return formatted_prompts

    def run(self, storage: DataFlowStorage, input_chat_key: str, output_key: str = "scenario"):
        self.input_chat_key = input_chat_key
        self.output_key = output_key
        dataframe = storage.read("dataframe")
        llm_inputs = self._reformat_prompt(dataframe)
        llm_outputs = self.llm_serving.generate_from_input(llm_inputs)
        dataframe[self.output_key] = llm_outputs
        output_file = storage.write(dataframe)
        self.logger.info(f"Results saved to {output_file}")
        return [self.output_key]


@OPERATOR_REGISTRY.register()
class ScenarioExpander(OperatorABC):

    def __init__(self, llm_serving: LLMServingABC):
        self.logger = get_logger()
        self.prompt = FuncCallPrompt()
        self.llm_serving = llm_serving
        self.logger.info(f"Initializing {self.__class__.__name__}...")

    def _reformat_prompt(self, dataframe: pd.DataFrame):
        formatted_prompts = [self.prompt.expand_scenario_prompt(scenario=item) for item in tqdm(dataframe[self.input_scenario_key], desc=f"Reformatting prompts...")]
        return formatted_prompts

    def run(self, storage: DataFlowStorage, input_scenario_key: str, output_key: str = "modified_scenario"):
        self.input_scenario_key = input_scenario_key
        self.output_key = output_key
        dataframe = storage.read("dataframe")
        llm_inputs = self._reformat_prompt(dataframe)
        llm_outputs = self.llm_serving.generate_from_input(llm_inputs)
        dataframe[self.output_key] = llm_outputs
        output_file = storage.write(dataframe)
        self.logger.info(f"Results saved to {output_file}")
        return [self.output_key]  

@OPERATOR_REGISTRY.register()
class AtomTaskGenerator(OperatorABC):

    def __init__(self, llm_serving: LLMServingABC):
        self.logger = get_logger()
        self.prompt = FuncCallPrompt()
        self.llm_serving = llm_serving
        self.logger.info(f"Initializing {self.__class__.__name__}...")

    def _reformat_prompt(self, dataframe: pd.DataFrame):
        formatted_prompts = [self.prompt.atomic_task_generate_prompt(scenario=item) for item in tqdm(dataframe[self.input_scenario_key], desc=f"Reformatting prompts...")]
        return formatted_prompts

    def run(self, storage: DataFlowStorage, input_scenario_key: str, output_key: str = "atom_task"):
        self.input_scenario_key = input_scenario_key
        self.output_key = output_key
        dataframe = storage.read("dataframe")
        llm_inputs = self._reformat_prompt(dataframe)
        llm_outputs = self.llm_serving.generate_from_input(llm_inputs)
        dataframe[self.output_key] = llm_outputs
        output_file = storage.write(dataframe)
        self.logger.info(f"Results saved to {output_file}")
        return [self.output_key]  

@OPERATOR_REGISTRY.register()
class SequentialTaskGenerator(OperatorABC):

    def __init__(self, llm_serving: LLMServingABC):
        self.logger = get_logger()
        self.prompt = FuncCallPrompt()
        self.llm_serving = llm_serving
        self.logger.info(f"Initializing {self.__class__.__name__}...")

    def _reformat_prompt(self, dataframe: pd.DataFrame):
        formatted_prompts = [self.prompt.sequential_task_generate_prompt(task=item) for item in tqdm(dataframe[self.input_task_key], desc=f"Reformatting prompts...")]
        return formatted_prompts

    def run(self, storage: DataFlowStorage, input_task_key: str, output_subsequent_task_key: str = "subsequent_task", output_composition_task_key: str = "composition_task"):
        self.input_task_key = input_task_key
        self.output_subsequent_task_key = output_subsequent_task_key
        self.output_composition_task_key = output_composition_task_key
        dataframe = storage.read("dataframe")
        llm_inputs = self._reformat_prompt(dataframe)
        llm_outputs = self.llm_serving.generate_from_input(llm_inputs)
        subsequent_tasks, composition_tasks = [], []
        for item in llm_outputs:
            # 正则表达式提取
            match_subsequent = re.search(r"### Subsequent Task: (.*?)\n", item)
            match_composition = re.search(r"### Composition Task: (.*?)$", item)
            if match_subsequent:
                subsequent_task = match_subsequent.group(1)
            else:
                subsequent_task = None
            if match_composition:
                composition_task = match_composition.group(1)
            else:
                composition_task = None
            subsequent_tasks.append(subsequent_task)
            composition_tasks.append(composition_task)

        dataframe[self.output_subsequent_task_key] = subsequent_tasks
        dataframe[self.output_composition_task_key] = composition_tasks
        output_file = storage.write(dataframe)
        self.logger.info(f"Results saved to {output_file}")
        return [self.output_subsequent_task_key, output_composition_task_key]  

@OPERATOR_REGISTRY.register()
class ParaSeqTaskGenerator(OperatorABC):

    def __init__(self, llm_serving: LLMServingABC):
        self.logger = get_logger()
        self.prompt = FuncCallPrompt()
        self.llm_serving = llm_serving
        self.logger.info(f"Initializing {self.__class__.__name__}...")

    def _reformat_prompt(self, dataframe: pd.DataFrame):
        formatted_prompts = [self.prompt.parallel_then_sequential_task_generate_prompt(task=item) for item in tqdm(dataframe[self.input_task_key], desc=f"Reformatting prompts...")]
        return formatted_prompts

    def run(self, storage: DataFlowStorage, input_task_key: str, output_parallel_task_key: str = "parallel_task",  output_subsequent_task_key: str = "subsequent_task", output_composition_task_key: str = "composition_task"):
        self.input_task_key = input_task_key
        self.output_parallel_task_key = output_parallel_task_key
        self.output_subsequent_task_key = output_subsequent_task_key
        self.output_composition_task_key = output_composition_task_key
        dataframe = storage.read("dataframe")
        llm_inputs = self._reformat_prompt(dataframe)
        llm_outputs = self.llm_serving.generate_from_input(llm_inputs)
        parallel_tasks, subsequent_tasks, composition_tasks = [], [], []
        for item in llm_outputs:
            # 正则表达式提取
            match_parallel = re.search(r"### Parallel Task: (.*?)\n", item)
            match_subsequent = re.search(r"### Subsequent Task: (.*?)\n", item)
            match_composition = re.search(r"### Composition Task: (.*?)$", item)
            if match_parallel:
                parallel_task = match_parallel.group(1)
            else:
                parallel_tasks = None
            if match_subsequent:
                subsequent_task = match_subsequent.group(1)
            else:
                subsequent_task = None
            if match_composition:
                composition_task = match_composition.group(1)
            else:
                composition_task = None
            parallel_tasks.append(parallel_task)
            subsequent_tasks.append(subsequent_task)
            composition_tasks.append(composition_task)
        dataframe[self.output_parallel_task_key] = parallel_tasks
        dataframe[self.output_subsequent_task_key] = subsequent_tasks
        dataframe[self.output_composition_task_key] = composition_tasks
        output_file = storage.write(dataframe)
        self.logger.info(f"Results saved to {output_file}")
        return [self.output_parallel_task_key, self.output_subsequent_task_key, output_composition_task_key]  

@OPERATOR_REGISTRY.register()
class CompositionTaskFilter(OperatorABC):
    def __init__(self, llm_serving: LLMServingABC):
        self.logger = get_logger()
        self.prompt = FuncCallPrompt()
        self.llm_serving = llm_serving
        self.logger.info(f"Initializing {self.__class__.__name__}...")

    def _reformat_prompt(self, dataframe: pd.DataFrame):
        formatted_prompts = []
        for task, sub_tasks in tqdm(zip(dataframe[self.input_composition_task_key], dataframe[self.input_sub_tasks_keys].to_dict(orient='records')), desc="Reformatting prompts..."):
            formatted_prompts.append(self.prompt.filter_composition_task_prompt(task=task, sub_tasks=sub_tasks))
        # formatted_prompts = [self.prompt.filter_composition_task(task=item, sub_tasks=sub_tasks) for item, sub_tasks in tqdm(zip(dataframe[self.input_composition_task_key], dataframe[self.input_sub_tasks_key]), desc=f"Reformatting prompts...")]
        return formatted_prompts

    def run(self, storage: DataFlowStorage, input_composition_task_key: str, input_sub_tasks_keys: list[str], output_key: str = "runable_label"):
        self.input_composition_task_key = input_composition_task_key
        self.input_sub_tasks_keys = input_sub_tasks_keys
        self.output_key = output_key
        dataframe = storage.read("dataframe")
        llm_inputs = self._reformat_prompt(dataframe)
        self.logger.debug(f"One of formatted prompts: {llm_inputs[0]}")
        llm_outputs = self.llm_serving.generate_from_input(llm_inputs)
        self.logger.debug(f"One of LLM outputs: {llm_outputs[0]}")
        labels = []
        for item in llm_outputs:
            match = re.search(r"<ans>(yes|no)</ans>", item.strip(), re.IGNORECASE)
            if match:
                labels.append(1 if match.group(1).lower() == "yes" else 0)
            else:
                labels.append(0)
        dataframe[self.output_key] = labels
        dataframe = dataframe[dataframe[self.output_key] > 0]
        storage.write(dataframe)
        output_file = storage.write(dataframe)
        self.logger.info(f"Results saved to {output_file}")
        return [self.output_key]
    
@OPERATOR_REGISTRY.register()
class FunctionGenerator(OperatorABC):
    def __init__(self, llm_serving: LLMServingABC):
        self.logger = get_logger()
        self.prompt = FuncCallPrompt()
        self.llm_serving = llm_serving
        self.logger.info(f"Initializing {self.__class__.__name__}...")

    def _reformat_prompt(self, dataframe: pd.DataFrame):
        formatted_prompts = []
        for task, sub_tasks in tqdm(zip(dataframe[self.input_composition_task_key], dataframe[self.input_sub_tasks_keys].to_dict(orient='records')), desc="Reformatting prompts..."):
            formatted_prompts.append(self.prompt.function_generate_prompt(task=task, sub_tasks=sub_tasks))
        # formatted_prompts = [self.prompt.filter_composition_task(task=item, sub_tasks=sub_tasks) for item, sub_tasks in tqdm(zip(dataframe[self.input_composition_task_key], dataframe[self.input_sub_tasks_key]), desc=f"Reformatting prompts...")]
        return formatted_prompts

    def run(self, storage: DataFlowStorage, input_composition_task_key: str, input_sub_tasks_keys: list[str], output_key: str = "functions"):
        self.input_composition_task_key = input_composition_task_key
        self.input_sub_tasks_keys = input_sub_tasks_keys
        self.output_key = output_key
        dataframe = storage.read("dataframe")
        llm_inputs = self._reformat_prompt(dataframe)
        # self.logger.info(f"One of formatted prompts: {llm_inputs[0]}")
        llm_outputs = self.llm_serving.generate_from_input(llm_inputs)
        # self.logger.info(f"One of LLM outputs: {llm_outputs[0]}")
        dataframe[self.output_key] = llm_outputs
        storage.write(dataframe)
        output_file = storage.write(dataframe)
        self.logger.info(f"Results saved to {output_file}")
        return [self.output_key]
    
@OPERATOR_REGISTRY.register()
class MultiTurnConversationGenerator(OperatorABC):
    def __init__(self, llm_serving: LLMServingABC):
        self.llm_serving = llm_serving
        self.prompt = FuncCallPrompt()
        self.logger = get_logger()
        
    def _reformat_user_agent_prompt(self, dataframe: pd.DataFrame):
        user_agent_prompts = []
        for item in tqdm(dataframe[self.input_task_key], desc="Reformatting prompts..."):
            user_agent_prompts.append(self.prompt.user_agent_prompt(task=item))
        return user_agent_prompts

    def _reformat_assistant_agent_prompt(self, user_agent_prompts: list[str], dataframe: pd.DataFrame):
        assistant_agent_sys_prompts = []
        for sub_tasks, functions in zip(dataframe[self.input_sub_tasks_keys].to_dict(orient='records'), dataframe[self.input_functions_key]):
            assistant_agent_sys_prompts.append(self.prompt.assistant_agent_prompt(sub_task=sub_tasks, sub_task_func=functions))
        assistant_agent_user_inputs = user_agent_prompts
        inputs = [[{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_input}] for sys_prompt, user_input in zip(assistant_agent_sys_prompts, assistant_agent_user_inputs)]   
        return inputs
        
    def _reformat_tool_agent_prompt(self, func_calls: list[str]):
        tool_agent_prompts = []
        for func_call in func_calls:
            tool_agent_prompts.append(self.prompt.tool_agent_prompt(function=func_call))
        return tool_agent_prompts
        
    def run(self, storage: DataFlowStorage, input_task_key: str, input_sub_tasks_keys: list[str], input_functions_key: list[str], output_conversations_key: str = "conversations"):
        self.input_task_key = input_task_key
        self.input_sub_tasks_keys = input_sub_tasks_keys
        self.input_functions_key = input_functions_key
        self.output_user_agent_response_key = "user_response"
        self.output_key = output_conversations_key
        dataframe = storage.read("dataframe")
        
        user_agent_prompts = self._reformat_user_agent_prompt(dataframe)
        user_agent_responses = self.llm_serving.generate_from_input(user_agent_prompts)
        
        dataframe[self.output_user_agent_response_key] = user_agent_responses
        turns = 0
        completed_label = [0] * len(dataframe)
        valid_label = [0] * len(dataframe)
        cur_conversations = self._reformat_assistant_agent_prompt(user_agent_responses, dataframe)
        while True:
            assistant_agent_inputs = cur_conversations
            not_completed_idxs = np.where(np.array(completed_label) == 0)[0]
            valid_idxs = np.where(np.array(valid_label) == 0)[0]
            cur_chatting_idxs = np.intersect1d(not_completed_idxs, valid_idxs)
            cur_chatting_conversations = [assistant_agent_inputs[idx] for idx in cur_chatting_idxs]
            assistant_agent_outputs = self.llm_serving.generate_from_conversations(cur_chatting_conversations)
            new_assistant_agent_outputs = list(zip(cur_chatting_idxs, assistant_agent_outputs))
            
            func_call_pattern = r"<func_call>(.*?)</func_call>"
            func_calls = []
            final_answer_pattern = r"<final>(.*?)</final>"
            for idx, text in new_assistant_agent_outputs:
                if isinstance(text, str):
                    final_match = re.search(final_answer_pattern, text, re.DOTALL)
                else:
                    self.logger.warning("Warning: 'text' is not a string:", text)
                    final_match = None
                    valid_label[idx] = 1
                    continue
                final_match = re.search(final_answer_pattern, text, re.DOTALL)
                if final_match:
                    completed_label[idx] = 1
                    self.logger.info(f"Final answer found: {idx}")
                    continue
                func_match = re.search(func_call_pattern, text, re.DOTALL)
                if func_match:
                    result = func_match.group(1)
                    func_calls.append(f"<func_call>{result}</func_call>")
                else:
                    func_calls.append("")
                    

            for item, text in zip(cur_chatting_conversations, assistant_agent_outputs):
                item.append({"role": "assistant", "content": text})
                
            not_completed_idxs = np.where(np.array(completed_label) == 0)[0]
            valid_idxs = np.where(np.array(valid_label) == 0)[0]
            cur_chatting_idxs = np.intersect1d(not_completed_idxs, valid_idxs)
            cur_chatting_conversations = [assistant_agent_inputs[idx] for idx in cur_chatting_idxs]
            tool_agent_inputs = self._reformat_tool_agent_prompt(func_calls)
            tool_agent_outputs = self.llm_serving.generate_from_input(tool_agent_inputs)
            for item, text in zip(cur_chatting_conversations, tool_agent_outputs):
                item.append({"role": "assistant", "content": text})
            turns += 1
            if turns >= 5:
                break
        self.logger.info(f"Bad answer {np.where(np.array(completed_label) == 0)[0]}")
        dataframe[self.output_key] = cur_conversations
        dataframe = dataframe[np.array(completed_label) == 1]
        storage.write(dataframe)
        return self.output_key
        
        
