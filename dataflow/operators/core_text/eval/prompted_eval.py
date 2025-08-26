import pandas as pd
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger

from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC
from dataflow.core import LLMServingABC
import re
@OPERATOR_REGISTRY.register()
class PromptedEvaluator(OperatorABC):
    '''
    Answer Generator is a class that generates answers for given questions.
    '''
    def __init__(self, llm_serving: LLMServingABC, system_prompt: str = "Please evaluate the quality of this data on a scale from 1 to 5."):
        self.logger = get_logger()
        self.llm_serving = llm_serving
        self.system_prompt = system_prompt
    
    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "PromptedEvaluator：使用 LLM 根据系统提示词对数据质量进行评分，并将评分写回 DataFrame（同时通过 "
                "storage 持久化）。模型应只输出分数（整数）。\n"
                "功能：对每行输入文本生成一个评分。\n"
                "输入参数：\n"
                "- llm_serving：LLM 服务对象，需实现 LLMServingABC 接口。\n"
                "- system_prompt：系统提示词（默认：'Please evaluate the quality of this data on a scale from 1 to 5.'）。\n"
                "- input_key：输入文本所在列名（默认：'raw_content'）。\n"
                "- output_key：评分结果写入的列名（默认：'eval'）。\n"
                "输出：\n"
                "- 返回输出列名（用于后续算子引用），评分结果已写回并保存。"
            )
        elif lang == "en":
            return (
                "PromptedEvaluator: uses an LLM to rate data quality and writes the score back to the "
                "DataFrame (persisted via storage). The model is expected to output only the integer score.\n"
                "Purpose: for each input row, produce an score.\n"
                "Input Parameters:\n"
                "- llm_serving: LLM serving object implementing LLMServingABC.\n"
                "- system_prompt: system prompt (default: 'Please evaluate the quality of this data on a scale from 1 to 5.').\n"
                "- input_key: column name containing input text (default: 'raw_content').\n"
                "- output_key: column name to store scores (default: 'eval').\n"
                "Output:\n"
                "- Returns the output column name for downstream operators; the scored DataFrame is saved."
            )
        else:
            return "PromptedEvaluator rates data quality (1–5) from input text and stores the integer score."

    def _parse_scores(self, outputs: list[str]) -> list[int]:
        """
        将模型输出的分数字符串转为整数。
        - 成功提取到 1–5 范围内的分数 → 返回该分数
        - 提取失败或不合法 → 返回 0
        """
        results = []
        for out in outputs:
            score = 0
            try:
                if out is None:
                    results.append(0)
                    continue

                text = str(out).strip()

                # 用正则找第一个数字
                match = re.search(r"\d+", text)
                if match:
                    val = int(match.group())
                    if 1 <= val <= 5:
                        score = val
                # 否则默认 0
            except Exception:
                score = 0

            results.append(score)
        return results

    def eval(self, dataframe, input_key):
        llm_inputs = []
        for index, row in dataframe.iterrows():
            raw_content = row.get(input_key, '')
            if raw_content:
                llm_input = self.system_prompt + str(raw_content) + 'Please only output the score!'
                llm_inputs.append(llm_input)
        
        try:
            self.logger.info("Generating text using the model...")
            generated_outputs = self.llm_serving.generate_from_input(llm_inputs)
            scores = self._parse_scores(generated_outputs)
            self.logger.info("Text generation completed.")
        except Exception as e:
            self.logger.error(f"Error during text generation: {e}")
            return
        return scores

    def run(self, storage: DataFlowStorage, input_key: str = "raw_content", output_key: str = "eval"):
        self.logger.info("Running PromptGenerator...")

        # Load the raw dataframe from the input file
        dataframe = storage.read('dataframe')
        self.logger.info(f"Loading, number of rows: {len(dataframe)}")

        # Create a list to hold all generated questions and answers
        generated_outputs = self.eval(dataframe, input_key)

        # Add the generated content back to the dataframe
        dataframe[output_key] = generated_outputs

        # Save the updated dataframe to the output file
        output_file = storage.write(dataframe)
        return output_key
