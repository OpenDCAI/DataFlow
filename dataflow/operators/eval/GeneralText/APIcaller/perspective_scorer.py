from googleapiclient import discovery
from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
import pandas as pd
from dataflow.core import LLMServingABC
from dataflow.serving import PerspectiveAPIServing

@OPERATOR_REGISTRY.register()
class PerspectiveScorer(OperatorABC):
    """Operator that assigns Perspective API toxicity scores to text inputs."""
    def __init__(self, serving: PerspectiveAPIServing = None):
        self.logger = get_logger()
        self.logger.info(f"Initializing {self.__class__.__name__}...")
        self.serving = serving
        self.score_name = 'PerspectiveScore'
        self.logger.info(f"{self.__class__.__name__} initialized.")
    
    @staticmethod
    def get_desc(lang: str = "zh"):
        return "使用 PerspectiveAPI 评估文本的毒性，返回毒性概率，得分越高表明文本毒性越高。" if lang == "zh" else "Assess text toxicity using PerspectiveAPI; higher scores indicate more toxicity."

    def get_score(self, samples: list[dict], input_key: str) -> list[float]:
        # Extract texts, truncate if needed
        texts = []
        max_bytes = 20480
        for sample in samples:
            text = sample.get(input_key, '') or ''
            encoded = text.encode('utf-8')
            if len(encoded) > max_bytes:
                text = encoded[:max_bytes].decode('utf-8', errors='ignore')
            texts.append(text)
        # Delegate to serving
        return self.serving.generate_from_input(texts)

    def eval(self, dataframe: pd.DataFrame, input_key: str) -> list[float]:
        self.logger.info(f"Evaluating {self.score_name}...")
        samples = dataframe.to_dict(orient='records')
        scores = self.get_score(samples, input_key)
        self.logger.info("Evaluation complete!")
        return scores

    def run(self,
            storage: DataFlowStorage,
            input_key: str,
            output_key: str = 'PerspectiveScore'):
        self.input_key = input_key
        self.output_key = output_key
        dataframe = storage.read("dataframe")
        dataframe[self.output_key] = self.eval(dataframe, self.input_key)
        storage.write(dataframe)
