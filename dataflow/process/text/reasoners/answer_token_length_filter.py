from dataflow.core import TextFilter
import numpy as np
from dataflow.utils.registry import PROCESSOR_REGISTRY
from transformers import AutoTokenizer

@PROCESSOR_REGISTRY.register()
class AnswerTokenLengthFilter(TextFilter):
    def __init__(self, args_dict: dict):
        super().__init__(args_dict)
        self.filter_name = 'AnswerTokenLengthFilter'
        self.max_answer_token_length = args_dict['max_answer_token_length']
        self.tokenizer = AutoTokenizer.from_pretrained(args_dict['tokenizer_dir'])

    def filter_func(self, dataset):
        def get_token_count(input_string):
            # 使用 tokenizer 对字符串进行编码，并获取 token 数目
            tokens = self.tokenizer.encode(input_string, add_special_tokens=False)
            return len(tokens)

        return np.array([get_token_count(item['answer']) <= self.max_answer_token_length for item in dataset]).astype(int)