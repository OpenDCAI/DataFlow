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
        return (
            "使用大语言模型识别语言并过滤数据。\n"
            "输入参数：\n"
            "- input_key：输入文本字段名（必填）\n"
            "- output_key：写入LLM语言识别结果的输出字段名，默认为'language_label'\n"
            "输出参数：\n"
            "- 过滤后的DataFrame，仅保留output_key值在allowed_languages中的行；"
            "每行新增output_key字段，值为LLM返回的原始字符串（预期为ISO 639-1语言代码，如'en'），该字段保留在输出中\n"
            "- run()返回的是过滤前的完整DataFrame（含output_key列），而不是字段名列表"
        ) if lang == "zh" else (
            "Using large language models to identify languages and filter data.\n"
            "Input Parameters:\n"
            "- input_key: Input text field name (required)\n"
            "- output_key: Name of the output field that receives the LLM language label, default is 'language_label'\n"
            "Output Parameters:\n"
            "- Filtered DataFrame containing only rows whose output_key value is in allowed_languages; "
            "each row gets a new output_key field holding the raw string returned by the LLM (expected to be an ISO 639-1 code such as 'en'), which is kept in the output\n"
            "- run() returns the full unfiltered DataFrame (including the output_key column), not a list of field names"
        )
    
    def _reformat_prompt(self, dataframe):
        formatted_prompts = [self.prompt.build_prompt(text=item) for item in tqdm(dataframe[self.input_key], desc="Reformatting Prompt...")]
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