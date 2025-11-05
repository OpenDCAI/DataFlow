import pandas as pd
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
import string

from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC
from dataflow.core import LLMServingABC
from dataflow.core.prompt import prompt_restrict, PromptABC, DIYPromptABC
from typing import Union, Any, Set

from dataflow.prompts.core_text import StrFormatPrompt
@prompt_restrict(
    StrFormatPrompt
)
@OPERATOR_REGISTRY.register()
class PromptTemplatedGenerator(OperatorABC):
    def __init__(
            self,
            llm_serving: LLMServingABC, 
            prompt_template: Union[StrFormatPrompt, DIYPromptABC] = None,
        ):
        self.logger = get_logger()
        self.llm_serving = llm_serving
        self.prompt_template = prompt_template
        if prompt_template is None:
            raise ValueError("prompt_template cannot be None")


    def run(
            self, 
            storage: DataFlowStorage,
            output_key: str = "generated_content",
            **input_keys: Any
        ):
        self.storage: DataFlowStorage = storage
        self.output_key = output_key
        self.logger.info("Running PromptTemplatedGenerator...")
        self.input_keys = input_keys

        need_fields = getattr(self.prompt_template, "fields", None)

    # Load the raw dataframe from the input file
        dataframe = storage.read('dataframe')
        self.logger.info(f"Loading, number of rows: {len(dataframe)}")
        llm_inputs = []

        for idx, row in dataframe.iterrows():
            key_dict = {}
            for key in need_fields:
                key_dict[key] = row[input_keys[key]]
            prompt_text = self.prompt_template.build_prompt(**key_dict)
            llm_inputs.append(prompt_text)
        self.logger.info(f"Prepared {len(llm_inputs)} prompts for LLM generation.")
        # Create a list to hold all generated contents
        # Generate content using the LLM serving
        generated_outputs = self.llm_serving.generate_from_input(llm_inputs)

        dataframe[self.output_key] = generated_outputs

        output_file = self.storage.write(dataframe)
        return output_key