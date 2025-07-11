from dataflow.prompts.kbcleaning import KnowledgeCleanerPrompt
import pandas as pd
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger

from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC
from dataflow.core import LLMServingABC

@OPERATOR_REGISTRY.register()
class KnowledgeCleaner(OperatorABC):
    '''
        KnowledgeCleaner is a class that cleans knowledge for RAG to make them more accurate, reliable and readable.
    '''
    def __init__(self, llm_serving: LLMServingABC, lang="en"):
        self.logger = get_logger()
        self.prompts = KnowledgeCleanerPrompt(lang=lang)    
        self.llm_serving = llm_serving
    
    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "知识清洗算子：对原始知识内容进行标准化处理，包括HTML标签清理、特殊字符规范化、"
                "链接处理和结构优化，提升RAG知识库的质量。主要功能：\n"
                "1. 移除冗余HTML标签但保留语义化标签\n"
                "2. 标准化引号/破折号等特殊字符\n"
                "3. 处理超链接同时保留文本\n"
                "4. 保持原始段落结构和代码缩进\n"
                "5. 确保事实性内容零修改"
            )
        elif lang == "en":
            return (
                "Knowledge Cleaning Operator: Standardizes raw content for RAG by:\n"
                "1. Removing redundant HTML tags while preserving semantic markup\n"
                "2. Normalizing special characters (quotes/dashes)\n"
                "3. Processing hyperlinks with text preservation\n"
                "4. Maintaining original paragraph structure and code indentation\n"
                "5. Guaranteeing zero modification of factual content"
            )
        else:
            return "Knowledge cleaning operator for RAG content standardization"

    def _validate_dataframe(self, dataframe: pd.DataFrame):
        required_keys = [self.input_key]
        forbidden_keys = [self.output_key]

        missing = [k for k in required_keys if k not in dataframe.columns]
        conflict = [k for k in forbidden_keys if k in dataframe.columns]

        if missing:
            raise ValueError(f"Missing required column(s): {missing}")
        if conflict:
            raise ValueError(f"The following column(s) already exist and would be overwritten: {conflict}")

    def _reformat_prompt(self, dataframe):
        """
        Reformat the prompts in the dataframe to generate questions.
        """
        raw_contents = dataframe[self.input_key].tolist()
        inputs = [self.prompts.Classic_COT_Prompt(raw_content) for raw_content in raw_contents]

        return inputs

    def run(
        self, 
        storage: DataFlowStorage, 
        input_key:str = "raw_content", 
        output_key:str = "cleaned"
        ):
        '''
        Runs the knowledge cleaning process, reading from the input key and saving results to output key.
        '''
        self.input_key, self.output_key = input_key, output_key
        dataframe = storage.read("dataframe")
        self._validate_dataframe(dataframe)
        formatted_prompts = self._reformat_prompt(dataframe)
        cleaned = self.llm_serving.generate_from_input(formatted_prompts,"")

        #for each in cleaned, only save the content in <cleaned_start> and <cleaned_end>
        cleaned_extracted = [
            text.split('<cleaned_start>')[1].split('<cleaned_end>')[0].strip() 
            if '<cleaned_start>' in text and '<cleaned_end>' in text 
            else text.strip()
            for text in cleaned
        ]
        dataframe[self.output_key] = cleaned_extracted
        output_file = storage.write(dataframe)
        self.logger.info(f"Results saved to {output_file}")

        return [output_key]