from dataflow.prompts.kbcleaning import KnowledgeCleanerPrompt
import pandas as pd
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
import json
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC
from dataflow.core import LLMServingABC


@OPERATOR_REGISTRY.register()
class KBCTextCleanerBatch(OperatorABC):
    '''
        KnowledgeCleaner is a class that cleans knowledge for RAG to make them more accurate, reliable and readable.
    '''

    def __init__(self, llm_serving: LLMServingABC, lang="en", prompt_template = None):
        self.logger = get_logger()
        self.prompts = KnowledgeCleanerPrompt(lang=lang)    
        self.llm_serving = llm_serving
        if prompt_template:
            self.prompt_template = prompt_template
        else:
            self.prompt_template = KnowledgeCleanerPrompt()

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
            raise ValueError(
                f"The following column(s) already exist and would be overwritten: {conflict}")

    def _reformat_prompt(self, dataframe):
        """
        Reformat the prompts in the dataframe to generate questions.
        """
        raw_contents = dataframe[self.input_key].tolist()
        inputs = [self.prompt_template.build_prompt(
            raw_content) for raw_content in raw_contents]

        return inputs
    

    def _reformat_prompt_from_path(self, chunk_path: str) -> list:
        """
        Reformat the prompts in the file (JSON or JSONL) to generate question prompts.

        Args:
            chunk_path (str): Path to the .json or .jsonl file containing raw chunks.

        Returns:
            list: A list of formatted prompt strings.
        """
        if chunk_path.endswith(".json"):
            dataframe = pd.read_json(chunk_path)
        elif chunk_path.endswith(".jsonl"):
            dataframe = pd.read_json(chunk_path, lines=True)
        else:
            raise ValueError(
                "Unsupported file format. Only .json and .jsonl are supported.")

        if "raw_chunk" not in dataframe.columns:
            raise KeyError("'raw_chunk' field not found in the input file.")

        raw_contents = dataframe["raw_chunk"].tolist()
        inputs = [self.prompts.build_prompt(
            raw_content) for raw_content in raw_contents]

        return raw_contents, inputs


    def run(
        self,
        storage: DataFlowStorage,
        input_key: str = "chunk_path",
        output_key: str = "cleaned_chunk_path"
    ):
        '''
        Runs the knowledge cleaning process, reading from the input key and saving results to output key.
        '''
        self.input_key, self.output_key = input_key, output_key
        dataframe = storage.read("dataframe")
        self._validate_dataframe(dataframe)
        chunk_paths = dataframe[self.input_key].tolist()

        for chunk_path in chunk_paths:
            if(chunk_path):
                raw_chunks, formatted_prompts = self._reformat_prompt_from_path(chunk_path)
                cleaned = self.llm_serving.generate_from_input(formatted_prompts, "")

                # for each in cleaned, only save the content in <cleaned_start> and <cleaned_end>
                cleaned_extracted = [
                    text.split('<cleaned_start>')[1].split('<cleaned_end>')[0].strip()
                    if '<cleaned_start>' in str(text) and '<cleaned_end>' in str(text)
                    else str(text).strip()
                    for text in cleaned
                ]
                json_items=[{
                    "raw_chunk": raw_chunk,
                    "cleaned_chunk": cleaned_chunk
                } for raw_chunk, cleaned_chunk in zip(raw_chunks, cleaned_extracted)]

                with open(chunk_path, "w", encoding="utf-8") as f:
                    json.dump(json_items, f, ensure_ascii=False, indent=4)
                self.logger.info(f"Successfully cleaned contents in {chunk_path}")
                
        dataframe[self.output_key] = chunk_paths
        output_file = storage.write(dataframe)
        self.logger.info(f"Results saved to {output_file}")

        return [output_key]
