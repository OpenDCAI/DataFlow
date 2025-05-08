from dataflow.core import ReasonerFilter
import numpy as np
from dataflow.utils.registry import PROCESSOR_REGISTRY
from transformers import AutoTokenizer
from dataflow.utils.utils import get_logger

@PROCESSOR_REGISTRY.register()
class AnswerTokenLengthFilter(ReasonerFilter):
    def __init__(self, args_dict: dict):
        super().__init__(args_dict)
        self.filter_name = 'AnswerTokenLengthFilter'
        self.max_answer_token_length = args_dict['max_answer_token_length']
        self.tokenizer = AutoTokenizer.from_pretrained(args_dict['tokenizer_dir'])
        self.logger = get_logger()
        
    def filter_func(self, dataset):
        def get_token_count(input_string):
            tokens = self.tokenizer.encode(input_string, add_special_tokens=False)
            return len(tokens)

        return np.array([get_token_count(item[self.keys]) <= self.max_answer_token_length for item in dataset]).astype(int)