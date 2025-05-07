import pandas as pd
from tqdm import tqdm
from math_verify import parse, verify, LatexExtractionConfig
import logging
from dataflow.utils.registry import GENERATOR_REGISTRY
from dataflow.utils.utils import get_logger

@GENERATOR_REGISTRY.register()
class AnswerJudger_mathverify:
    def __init__(self, config: dict):
        self.config = config
        self._check_config()
        self.input_file = self.config['input_file']
        self.output_file = self.config['output_file']
        self.answer_key = self.config['answer_key']
        self.gt_key = self.config['gt_key']
        self.result_key = self.config['result_key']
        self.logger = get_logger()

    def _check_config(self):
        required_keys = [
            'input_file', 'output_file',
            'answer_key', 'gt_key', 'result_key',
        ]
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Key {key} is not in the config")
    
    def run(self):
        raw_dataframe = pd.read_json(self.input_file, lines=True)
        key_list = raw_dataframe.columns.to_list()
        if self.answer_key not in key_list:
            raise ValueError(f"answer_key: {self.answer_key} not found in the dataframe, please check the input_key: {key_list}")
        if self.gt_key not in key_list:
            raise ValueError(f"gt_key: {self.gt_key} not found in the dataframe, please check the input_key: {key_list}")
        self.logger.info(f"Found {len(raw_dataframe)} rows in the dataframe")
        results = []
        for answer, gt in tqdm(zip(raw_dataframe[self.answer_key], raw_dataframe[self.gt_key]), total=len(raw_dataframe), desc='processed'):
            results.append(float(verify(parse(answer), parse(gt))) > 0)
        raw_dataframe[self.result_key] = results
        raw_dataframe.to_json(self.output_file, orient='records', lines=True)
