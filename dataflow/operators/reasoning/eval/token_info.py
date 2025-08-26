from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage
import pandas as pd
from transformers import AutoTokenizer

@OPERATOR_REGISTRY.register()
class ToKenInfo(OperatorABC):
    def __init__(self):
        self.logger = get_logger()
        self.logger.info(f'Initializing {self.__class__.__name__}...')
        self.logger.info(f'{self.__class__.__name__} initialized.')
        self.information_name = "Token Information"

    def get_token_info(self, samples, input_question_key, input_answer_key, model_name_or_path):
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        questions = [sample.get(input_question_key, '') for sample in samples]
        answers = [sample.get(input_answer_key, '') for sample in samples]

        questions_tokens_length = [len(tokenizer.encode(question, add_special_tokens=False)) for question in questions]
        answers_tokens_length = [len(tokenizer.encode(answer, add_special_tokens=False)) for answer in answers]

        # count zeros in questions_tokens_length and answers_tokens_length
        questions_zeros_count = questions_tokens_length.count(0)
        answers_zeros_count = answers_tokens_length.count(0)

        # count min,max,mean, median of questions_tokens_length and answers_tokens_length
        questions_min = min(questions_tokens_length) if questions_tokens_length else 0
        questions_max = max(questions_tokens_length) if questions_tokens_length else 0
        questions_mean = sum(questions_tokens_length) / len(questions_tokens_length) if questions_tokens_length else 0
        questions_median = sorted(questions_tokens_length)[len(questions_tokens_length) // 2] if questions_tokens_length else 0
        answers_min = min(answers_tokens_length) if answers_tokens_length else 0
        answers_max = max(answers_tokens_length) if answers_tokens_length else 0
        answers_mean = sum(answers_tokens_length) / len(answers_tokens_length) if answers_tokens_length else 0
        answers_median = sorted(answers_tokens_length)[len(answers_tokens_length) // 2] if answers_tokens_length else 0
        token_info = {
            "questions_zeros_count": questions_zeros_count,
            "answers_zeros_count": answers_zeros_count,
            "questions_min": questions_min,
            "questions_max": questions_max,
            "questions_mean": questions_mean,
            "questions_median": questions_median,
            "answers_min": answers_min,
            "answers_max": answers_max,
            "answers_mean": answers_mean,
            "answers_median": answers_median
        }
        self.logger.info(f"Token information: {token_info}")
        return token_info
    
    def run(self,storage: DataFlowStorage, input_question_key: str, input_answer_key: str, model_name_or_path: str):
        self.input_question_key = input_question_key
        self.input_answer_key = input_answer_key
        self.model_name_or_path = model_name_or_path
        dataframe = storage.read("dataframe")
        if self.input_question_key not in dataframe.columns:
            self.logger.error(f"Input key {self.input_question_key} not found in dataframe columns.")
            return {}
        if self.input_answer_key not in dataframe.columns:
            self.logger.warning(f"Input key {self.input_answer_key} not found in dataframe columns")
        samples = dataframe.to_dict(orient='records')
        token_info = self.get_token_info(samples, self.input_question_key, self.input_answer_key, self.model_name_or_path)
        return token_info