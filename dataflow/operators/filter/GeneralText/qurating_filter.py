from dataflow.operators.eval import QuratingScorer
import numpy as np
from dataflow.core import OperatorABC
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.utils import get_logger
from dataflow.utils.storage import DataFlowStorage

@OPERATOR_REGISTRY.register()
class QuratingFilter(OperatorABC):

    def __init__(self, min_scores: dict = {'writing_style': 0,'required_expertise': 0,'facts_and_trivia': 0,'educational_value': 0}, max_scores: dict = {'writing_style': 9,'required_expertise': 9,'facts_and_trivia': 9,'educational_value': 9}, 
                 map_batch_size: int = 512, num_workers: int = 1, device_batch_size: int = 16, device: str = 'cuda', 
                 labels: list = ['writing_style', 'required_expertise', 'facts_and_trivia', 'educational_value'], model_cache_dir: str = './dataflow_cache'):
        self.logger = get_logger()
        self.min_scores = min_scores
        self.max_scores = max_scores

        # Initialize the QuratingScorer with the passed parameters
        self.scorer = QuratingScorer(map_batch_size=map_batch_size, 
                                     num_workers=num_workers, device_batch_size=device_batch_size, device=device, 
                                     labels=labels, model_cache_dir=model_cache_dir)
        
        self.logger.info(f"Initializing {self.__class__.__name__} with min_scores = {self.min_scores} and max_scores = {self.max_scores}...")

    @staticmethod
    def get_desc(lang: str = "zh"):
        return "使用Qurating评分器过滤掉低质量数据" if lang == "zh" else "Filter out low-quality data using the Qurating scorer."

    def run(self, storage: DataFlowStorage, input_key: str):
        self.input_key = input_key
        dataframe = storage.read("dataframe")
        self.logger.info(f"Running {self.filter_name}...")

        # Get the scores for filtering
        scores = self.scorer.eval(dataframe, self.input_key)

        # Initialize results to all valid (1)
        results = np.ones(len(dataframe), dtype=int)

        # Iterate over each label to apply the filter and add a column
        for label in self.min_scores.keys():
            min_score = self.min_scores[label]
            max_score = self.max_scores[label]
            score_key = f"Qurating{''.join([word.capitalize() for word in label.split('_')])}Score"
            metric_scores = np.array(scores[score_key])

            # Apply score filter for the current label
            metric_filter = (min_score <= metric_scores) & (metric_scores <= max_score)
            results = results & metric_filter.astype(int)

            # Add a new column with the name '{label}_filter' containing 0 or 1 based on the filter
            dataframe[f"{label}_label"] = metric_filter.astype(int)

        # Filter the dataframe based on the results
        filtered_dataframe = dataframe[results == 1]
        # Write the filtered dataframe back to storage
        storage.write(filtered_dataframe)

        self.logger.info(f"Filtering completed. Total records passing filter: {len(filtered_dataframe)}.")

        result = [f"{label}_label" for label in self.min_scores.keys()]
        
        return result
