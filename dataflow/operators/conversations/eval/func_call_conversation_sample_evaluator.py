import re
import pandas as pd
import numpy as np
from tqdm import tqdm
from dataflow.core import OperatorABC, LLMServingABC
from dataflow.core.prompt import prompt_restrict
from dataflow.utils.storage import DataFlowStorage
from dataflow.prompts.func_call import ConversationEvalPrompt
from dataflow.logger import get_logger
from dataflow.utils.registry import OPERATOR_REGISTRY

@prompt_restrict(
    ConversationEvalPrompt
)

@OPERATOR_REGISTRY.register()
class FuncCallConversationSampleEvaluator(OperatorABC):
    
    def __init__(self, llm_serving: LLMServingABC):
        self.llm_serving = llm_serving
        self.prompt = ConversationEvalPrompt()
        self.logger = get_logger()      
        
    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "对对话样本进行打分评估：使用 LLM 服务根据预设评分提示词对每条对话进行评分，并将结果写回数据流。\n"
                "输入参数：\n"
                "- llm_serving：LLM 服务对象，需实现 LLMServingABC 接口\n"
                "- input_conversation_key：DataFrame 中对话内容字段名，默认 'conversations'\n"
                "- output_score_key：评分结果输出字段名，默认 'score'\n"
                "处理流程：\n"
                "- 读取存储中的 DataFrame\n"
                "- 将每条对话重组为评分提示词并调用 LLM 生成评分（JSON）\n"
                "- 解析 JSON，提取 'score' 字段写入 DataFrame；解析失败则回退为 0\n"
                "输出参数：\n"
                "- 包含评分结果列的 DataFrame\n"
                "- 包含输出字段名的列表（仅 'score' 或自定义的输出列名）"
            )
        elif lang == "en":
            return (
                "Evaluate conversation samples with an LLM-based scorer: the operator formats each "
                "conversation into a scoring prompt, calls the LLM, parses the JSON response, and writes the score back.\n"
                "Input Parameters:\n"
                "- llm_serving: LLM serving object implementing LLMServingABC\n"
                "- input_conversation_key: column name for conversations in the DataFrame, default 'conversations'\n"
                "- output_score_key: column name for the score output, default 'score'\n"
                "Process:\n"
                "- Read the DataFrame from storage\n"
                "- Reformat each conversation into a scoring prompt and call the LLM (expects JSON)\n"
                "- Parse the JSON to extract 'score'; fallback to 0 on parse errors\n"
                "Output:\n"
                "- DataFrame with a score column added\n"
                "- A list containing the output field name (e.g., 'score')"
            )
        else:
            return "Evaluate conversation samples with an LLM-based scorer and write the parsed 'score' back to the DataFrame."    
    
    def _reformat_prompt(self, dataframe: pd.DataFrame):
        formatted_prompts = []
        for conversation in tqdm(dataframe[self.input_conversation_key],  desc="Reformatting prompts..."):
            formatted_prompts.append(self.prompt.build_prompt(conversation=conversation))
        return formatted_prompts
    
    def clean_json_block(self, s: str) -> str:
        s = s.strip()
        if s.startswith("```"):
            # 去掉首尾 ```json 或 ``` 包裹
            s = s.strip("`")
            s = s.replace("json\n", "", 1)  # 去掉开头的 json\n
        if s.endswith("```"):
            s = s[:-3]
        return s.strip()
    
    def json_validate(self, llm_outputs):
        import json
        scores = []
        for item in llm_outputs:
            score = 0
            try:
                data = json.loads(self.clean_json_block(item))
                score = data['score']
            except Exception as e:
                self.logger.debug(f"json loading error in item {item}")
            scores.append(score)
        return scores
    
    def run(self, storage: DataFlowStorage, input_conversation_key: str = "conversations", output_score_key = "score"):
        self.input_conversation_key = input_conversation_key
        self.output_score_key = output_score_key
        dataframe = storage.read("dataframe")
        llm_inputs = self._reformat_prompt(dataframe)
        llm_outputs = self.llm_serving.generate_from_input(llm_inputs)
        dataframe[self.output_score_key] = self.json_validate(llm_outputs)
        storage.write(dataframe)
        output_file = storage.write(dataframe)
        self.logger.info(f"Results saved to {output_file}")
        return [self.output_score_key]
