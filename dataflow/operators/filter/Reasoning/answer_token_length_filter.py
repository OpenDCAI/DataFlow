from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC
from transformers import AutoTokenizer

from tqdm import tqdm
import pandas as pd

@OPERATOR_REGISTRY.register()
class AnswerTokenLengthFilter(OperatorABC):
    def __init__(self,
                max_answer_token_length: int = 8192,
                tokenizer_dir: str = "Qwen/Qwen2.5-0.5B-Instruct"):
        
        self.max_answer_token_length = max_answer_token_length
        self.tokenizer_dir = tokenizer_dir
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_dir)
        self.logger = get_logger()
       
    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "该算子根据token数量过滤过长的答案。\n\n"
                "输入参数：\n"
                "- max_answer_token_length：最大token数\n"
                "- tokenizer_dir：分词器路径\n"
                "- read_min/max_score：分数范围\n\n"
                "输出参数：\n"
                "- 长度合规返回1，否则返回0"
            )
        elif lang == "en":
            return (
                "Filters answers exceeding specified token length limit.\n\n"
                "Input Parameters:\n"
                "- max_answer_token_length: Maximum allowed tokens\n"
                "- tokenizer_dir: Tokenizer directory\n"
                "- read_min/max_score: Score range\n\n"
                "Output Parameters:\n"
                "- Returns 1 if within limit, 0 otherwise"
            )
        else:
            return "AnswerTokenLengthFilter enforces answer length constraints"


    def _validate_dataframe(self, dataframe: pd.DataFrame):
        required_keys = [self.input_key]
        forbidden_keys = []

        missing = [k for k in required_keys if k not in dataframe.columns]
        conflict = [k for k in forbidden_keys if k in dataframe.columns]

        if missing:
            self.logger.error(f"Missing required column(s): {missing}")
        if conflict:
            self.logger.error(f"The following column(s) already exist and would be overwritten: {conflict}")
        missing_keys = [key for key in required_keys if key not in dataframe.columns]

        if missing_keys:
            self.logger.error(f"The following required columns are missing from the dataframe: {missing_keys}")

    def run(
            self,
            storage:DataFlowStorage,
            input_key: str = "generated_cot"
            ) -> list:
        
        dataframe = storage.read("dataframe")
        
        self.input_key = input_key
        self.logger.info(f"Found {len(dataframe)} rows in the dataframe")
        self._validate_dataframe(dataframe)

        def get_token_count(input_string):
            tokens = self.tokenizer.encode(input_string, add_special_tokens=False)
            return len(tokens)

        output = []
        for i, text in tqdm(enumerate(dataframe[self.input_key]), desc="Checking token lengths"):
            is_valid = get_token_count(text) <= self.max_answer_token_length
            if is_valid:
                output.append(dataframe.iloc[i])
        
        dataframe = pd.DataFrame(output)

        output_file = storage.write(dataframe)
        self.logger.info(f"Saved {len(dataframe)} filtered rows to {output_file}")