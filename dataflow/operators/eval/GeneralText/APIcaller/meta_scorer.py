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

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "通过LLM评估文本的多个元属性，包括文本结构、多样性与复杂性、流畅性与可理解性、安全性、教育价值以及内容准确性与有效性。\n"
                "输入参数：\n"
                "- llm_serving：LLM服务对象，需实现LLMServingABC接口\n"
                "- input_key：输入文本字段名\n"
                "输出参数：\n"
                "- 包含6个评估维度得分的DataFrame，列名为：Text Structure, Diversity & Complexity, Fluency & Understandability, Safety, Educational Value, Content Accuracy & Effectiveness"
            )
        elif lang == "en":
            return (
                "Evaluate multiple meta attributes of text using LLM, including Text Structure, Diversity & Complexity, Fluency & Understandability, Safety, Educational Value, and Content Accuracy & Effectiveness.\n"
                "Input Parameters:\n"
                "- llm_serving: LLM serving object implementing LLMServingABC interface\n"
                "- input_key: Field name for input text\n"
                "Output Parameters:\n"
                "- DataFrame containing scores for 6 evaluation dimensions with columns: Text Structure, Diversity & Complexity, Fluency & Understandability, Safety, Educational Value, Content Accuracy & Effectiveness"
            )
        else:
            return "Evaluate multiple meta attributes of text using LLM."
    
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
