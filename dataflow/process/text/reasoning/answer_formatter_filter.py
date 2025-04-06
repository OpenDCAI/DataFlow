from dataflow.core import TextFilter
import numpy as np
from dataflow.utils.registry import PROCESSOR_REGISTRY
import re

@PROCESSOR_REGISTRY.register()
class AnswerFormatterFilter(TextFilter):
    def __init__(self, args_dict: dict):
        super().__init__(args_dict)
        self.filter_name = 'AnswerFormatterFilter'

    def is_valid_answer(answer: str) -> bool:
        # start with "Solution:"
        if not answer.startswith("Solution:"):
            return False
        
        # check that every step start with "→" or not
        #steps = answer.split("\n")
        #for step in steps:
        #    if step.strip() and not step.strip().startswith("→"):
        #        return False
        
        # check final answer in \boxed{} or not 
        if not re.search(r'\\boxed{.*}', answer):
            return False
        
        return True 
    
    def filter_func(self, dataset):
        indexes =  np.zeros(len(dataset)).astype(int)

        for i, item in enumerate(dataset):
            answer = item['answer']
            if AnswerFormatterFilter.is_valid_answer(answer):
                indexes[i] = 1

        return indexes