from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage
import pandas as pd
from dataflow.core import LLMServingABC
from dataflow.prompts.general_text import MetaPrompt  
import ast

@OPERATOR_REGISTRY.register()
class MetaScorer(OperatorABC):
    def __init__(self, llm_serving: LLMServingABC = None):
        self.logger = get_logger()
        self.logger.info(f'Initializing {self.__class__.__name__}...')
        self.llm_serving = llm_serving
        self.score_name = 'MetaScore'
        self.prompt = MetaPrompt()
        self.logger.info(f'{self.__class__.__name__} initialized.')

        self.output_columns = [
            "Text Structure",
            "Diversity & Complexity",
            "Fluency & Understandability",
            "Safety",
            "Educational Value",
            "Content Accuracy & Effectiveness"
        ]

    def get_score(self, samples, input_key):
        system_prompt = self.prompt.build_system_prompt()
        user_prompts = []
        for sample in samples:
            input_text = sample.get(input_key, '')
            user_prompt = self.prompt.build_user_prompt(input_text)
            full_prompt = system_prompt + "\n" + user_prompt
            user_prompts.append(full_prompt)

        responses = self.llm_serving.generate_from_input(user_inputs=user_prompts)
        scores = []

        for i, response in enumerate(responses):
            try:
                lines = response.strip().split("\n")
                last_line = lines[-1].strip()
                parsed_scores = ast.literal_eval(last_line)
                if isinstance(parsed_scores, list) and len(parsed_scores) == 6:
                    scores.append(parsed_scores)
                else:
                    raise ValueError("Score format invalid")
            except Exception as e:
                self.logger.warning(f"Failed to extract score from response {i}: {e}")
                scores.append([float('nan')] * 6)

        return scores

    def eval(self, dataframe: pd.DataFrame, input_key: str):
        samples = dataframe.to_dict(orient='records')
        self.logger.info(f"Evaluating {self.score_name}...")
        scores = self.get_score(samples, input_key)
        self.logger.info("Evaluation complete!")
        return scores

    def run(self, storage: DataFlowStorage, input_key: str):
        self.input_key = input_key
        dataframe = storage.read("dataframe")
        scores = self.eval(dataframe, self.input_key)
        # 展开6列固定命名
        score_df = pd.DataFrame(scores, columns=self.output_columns)
        dataframe = pd.concat([dataframe, score_df], axis=1)
        storage.write(dataframe)
