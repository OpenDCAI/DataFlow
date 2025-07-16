from dataflow.core import OperatorABC
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.storage import DataFlowStorage
from datasets import Dataset
from tqdm import tqdm
from dataflow import get_logger
from dataflow.operators.eval.GeneralText.models.Qurating.qurater_annotate import ModelAnnotator
from dataflow.operators.eval.GeneralText.models.Qurating.qurater_annotate import TokenizeAndChunk
import torch

@OPERATOR_REGISTRY.register()
class QuratingScorer(OperatorABC):
    def __init__(self, map_batch_size: int = 512, num_workers: int = 1, device_batch_size: int = 16, device: str = 'cuda', 
                 labels: list = ['writing_style', 'required_expertise', 'facts_and_trivia', 'educational_value'], model_cache_dir: str = './dataflow_cache'):
        self.logger = get_logger()
        self.logger.info(f'Initializing {self.__class__.__name__}...')
        self.model = 'princeton-nlp/QuRater-1.3B'
        self.tokens_field = 'input_ids'
        self.tokens = 512
        self.map_batch_size = map_batch_size
        self.batch_size = -1 
        self.num_workers = num_workers
        self.model_cache_dir = model_cache_dir
        self.labels = labels or []
        self.device_batch_size = device_batch_size
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.score_type = float 
        self.data_type = 'text'  
        self.score_name = 'QuratingScore'
        self.logger.info(f'{self.__class__.__name__} initialized.')

    @staticmethod
    def get_desc(lang: str = "zh"):
        return "通过 Qurating 模型从 '写作风格'、'所需专业程度'、'事实与趣闻' 和 '教育价值' 四个维度评估文本质量。" if lang == "zh" else "Evaluate text quality across four dimensions using the Qurating model: 'writing_style', 'required_expertise', 'facts_and_trivia', and 'educational_value'."

    def _score_func(self, sample):
        """Process a single sample and return the score."""
        batch_dict = {'text': [sample]}  # Wrap sample into a list for processing
        dataset = Dataset.from_dict(batch_dict)
        
        # Tokenize and chunk
        dataset = dataset.map(
            TokenizeAndChunk(self.model, 'text', self.tokens_field, self.tokens, self.model_cache_dir),
            batched=True,
            batch_size=self.map_batch_size,
            num_proc=self.num_workers,
            remove_columns=dataset.column_names
        )
        
        # Annotate the model results
        dataset = dataset.map(
            ModelAnnotator(self.model, self.labels, self.device_batch_size, self.device, self.model_cache_dir),
            batched=True,
            with_indices=True,
            batch_size=self.map_batch_size,
            remove_columns=dataset.column_names
        )

        results_dict = dataset.to_dict()
        result_filtered = {}

        for key in results_dict:
            for label in self.labels:
                average_key = f"{label}_average"
                if average_key in results_dict[key]:
                    new_key = f"Qurating{''.join([word.capitalize() for word in label.split('_')])}Score"
                    result_filtered[new_key] = results_dict[key]

        return result_filtered

    def eval(self, dataframe, input_key):
        self.logger.info(f"Evaluating {self.score_name}...")
        batch_dict = {'text': dataframe[input_key]}  # Wrap sample into a list for processing
        dataset = Dataset.from_dict(batch_dict)
        # Tokenize and chunk
        dataset = dataset.map(
            TokenizeAndChunk(self.model, 'text', self.tokens_field, self.tokens, self.model_cache_dir),
            batched=True,
            batch_size=self.map_batch_size,
            num_proc=self.num_workers,
            remove_columns=dataset.column_names
        )
        
        # Annotate the model results
        dataset = dataset.map(
            ModelAnnotator(self.model, self.labels, self.device_batch_size, self.device, self.model_cache_dir),
            batched=True,
            with_indices=True,
            batch_size=self.map_batch_size,
            remove_columns=dataset.column_names
        )
        results_dict = dataset.to_dict()
        result_filtered = {}
        for label in self.labels:
            average_key = f"{label}_average"
            if average_key in results_dict:
                new_key = f"Qurating{''.join([word.capitalize() for word in label.split('_')])}Score"
                result_filtered[new_key] = results_dict[average_key]  # Use the average values

        self.logger.info("Evaluation complete!")
        return result_filtered

    def run(self, storage: DataFlowStorage, input_key: str, output_key: str):
        dataframe = storage.read("dataframe")
        scores = self.eval(dataframe, input_key)
        for score_dict in scores:
            for key, value in score_dict.items():
                if key not in dataframe:
                    dataframe[key] = value
        
        storage.write(dataframe)
