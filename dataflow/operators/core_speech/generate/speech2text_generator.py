from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger

from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC
from dataflow.core import LLMServingABC

import os
import math
import warnings
import base64
from io import BytesIO
from typing import List, Optional, Union, Dict, Tuple

@OPERATOR_REGISTRY.register()
class Speech2TextGenerator(OperatorABC):
    def __init__(
        self,
        llm_serving: LLMServingABC,
        system_prompt: str = "You are a helpful assistant",
    ):
        self.logger = get_logger()
        self.llm_serving = llm_serving
        self.system_prompt = system_prompt
    
    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "该算子用于将语音内容转录为文本。它接收语音文件路径或URL，使用大语言模型进行转录，"
                "并将转录结果保存到数据框中。\n"
                "输入参数：\n"
                "- llm_serving：LLM服务对象，需实现LLMServingABC接口\n"
                "- system_prompt：系统提示词，用于定义模型行为，默认为'You are a helpful assistant'\n"
                "- input_key：输入语音文件路径或URL的字段名，默认为'raw_content'\n"
                "- output_key：输出转录文本的字段名，默认为'generated_content'\n"
                "输出参数：\n"
                "- 返回输出字段名，用于后续算子引用\n"
                "- 在数据框中添加包含转录文本的新列"
            )
        elif lang == "en":
            return (
                "This operator transcribes speech content into text. It receives paths or URLs to speech files, "
                "uses a large language model for transcription, and saves the transcription results to the dataframe.\n"
                "Input Parameters:\n"
                "- llm_serving: LLM serving object implementing LLMServingABC interface\n"
                "- system_prompt: System prompt to define model behavior, default is 'You are a helpful assistant'\n"
                "- input_key: Field name for input speech file paths or URLs, default is 'raw_content'\n"
                "- output_key: Field name for output transcription text, default is 'generated_content'\n\n"
                "Output Parameters:\n"
                "- Returns output field name for subsequent operator reference\n"
                "- Adds a new column containing transcription text to the dataframe"
            )
        else:
            return (
                "SpeechTranscriptor converts speech files to text using a large language model and saves results to a dataframe."
            )
    
    def run(self, storage: DataFlowStorage, input_key: str = "raw_content", output_key: str = "generated_content"):
        self.input_key, self.output_key = input_key, output_key
        self.logger.info("Running Speech Transcriptor...")

        dataframe = storage.read('dataframe')
        self.logger.info(f"Loading, number of rows: {len(dataframe)}")

        llm_inputs = []
        for index, row in dataframe.iterrows():
            path_or_url = row.get(self.input_key, '')
            llm_inputs.append(path_or_url)
        
        transcriptions = self.llm_serving.generate_from_input(
            user_inputs=llm_inputs,
            system_prompt=self.system_prompt
        )

        dataframe[self.output_key] = transcriptions
        output_file = storage.write(dataframe)
        self.logger.info(f"Saving to {output_file}")
        self.logger.info("Speech Transcriptor done")

        return output_key
