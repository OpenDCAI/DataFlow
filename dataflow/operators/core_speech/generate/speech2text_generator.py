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
