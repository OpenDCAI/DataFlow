import re
import pandas as pd
import numpy as np
from tqdm import tqdm
from dataflow.core import OperatorABC, LLMServingABC
from dataflow.utils.storage import DataFlowStorage
from dataflow.prompts.func_call import CompositionTaskFilterPrompt
from dataflow.logger import get_logger
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.core.prompt import prompt_restrict

@prompt_restrict(
    CompositionTaskFilterPrompt
)

@OPERATOR_REGISTRY.register()
class CompositionTaskFilter(OperatorABC):
    def __init__(self, llm_serving: LLMServingABC):
        self.logger = get_logger()
        self.prompt = CompositionTaskFilterPrompt()
        self.llm_serving = llm_serving
        self.logger.info(f"Initializing {self.__class__.__name__}...")

    @staticmethod  
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "根据组合任务及其子任务，使用LLM服务判断组合任务是否具备可行性与完备性，从而进行可运行任务的筛选。\n"
                "输入参数：\n"
                "- llm_serving：LLM服务对象，需实现LLMServingABC接口\n"
                "- input_composition_task_key：组合任务字段名\n"
                "- input_sub_tasks_keys：子任务字段名列表（如原子任务、并行任务、后继任务等）\n"
                "- output_key：可运行标签的输出字段名，默认'runable_label'\n"
                "输出参数：\n"
                "- 仅包含可运行组合任务的数据DataFrame\n"
                "- 包含输出字段名的列表（可运行标签字段）"
            )
        elif lang == "en":
            return (
                "Evaluate the feasibility and completeness of a composition task based on its sub-tasks using an LLM service, and filter out unexecutable tasks.\n"
                "Input Parameters:\n"
                "- llm_serving: LLM serving object implementing LLMServingABC interface\n"
                "- input_composition_task_key: Field name for the composition task\n"
                "- input_sub_tasks_keys: List of field names for sub-tasks (e.g., atomic, parallel, subsequent tasks)\n"
                "- output_key: Field name for the executability label, default 'runable_label'\n"
                "Output Parameters:\n"
                "- DataFrame containing only executable composition tasks\n"
                "- List containing the output field name (executability label)"
            )
        else:
            return "Filter composition tasks for feasibility and completeness using LLM service."


    def _reformat_prompt(self, dataframe: pd.DataFrame):
        formatted_prompts = []
        for task, sub_tasks in tqdm(zip(dataframe[self.input_composition_task_key], dataframe[self.input_sub_tasks_keys].to_dict(orient='records')), desc="Reformatting prompts..."):
            formatted_prompts.append(self.prompt.build_prompt(task=task, sub_tasks=sub_tasks))
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
    
