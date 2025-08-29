import numpy as np
from tqdm import tqdm
from dataflow.core import OperatorABC, LLMServingABC
from dataflow.prompts.general_text import LanguageFilterPrompt
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.logger import get_logger
from dataflow.utils.storage import DataFlowStorage

@OPERATOR_REGISTRY.register()
class LLMLanguageFilter(OperatorABC):
    """
    Operator for filtering text based on language using LLM.
    Argument allowed_languages is a list of allowed languages, using the ISO 639-1 two-letter language code to specify the language (for example, 'en' for English, 'zh' for Chinese, etc.).
    """
    def __init__(self, llm_serving: LLMServingABC = None, allowed_languages: list[str] = ['en']):
        self.logger = get_logger()
        self.prompt = LanguageFilterPrompt()
        self.llm_serving = llm_serving
        self.allowed_languages = allowed_languages
        self.logger.info(f"Initializing {self.__class__.__name__}...")

    @staticmethod
    def get_desc(lang: str = "zh"):
        return "使用大语言模型识别语言并过滤数据" if lang == "zh" else "Using large language models to identify languages and filter data."
    
    def _reformat_prompt(self, dataframe):
        formatted_prompts = [self.prompt.language_filter_prompt(text=item) for item in tqdm(dataframe[self.input_key], desc="Reformatting Prompt...")]
        return formatted_prompts

    def run(self, storage: DataFlowStorage, input_key: str, output_key: str = 'language_label'):
        self.input_key, self.output_key = input_key, output_key
        dataframe = storage.read("dataframe")
        formatted_prompts = self._reformat_prompt(dataframe)
        llm_outputs = self.llm_serving.generate_from_input(formatted_prompts)
        dataframe[self.output_key] = llm_outputs
        filtered_dataframe = dataframe[dataframe[self.output_key].isin(self.allowed_languages)]
        storage.write(filtered_dataframe)
        self.logger.info(f"Filtering completed. Total records passing filter: {len(filtered_dataframe)}.")
        return dataframe