from dataflow.core import OperatorABC
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.storage import DataFlowStorage
from datasets import Dataset
from tqdm import tqdm
from dataflow import get_logger
from dataflow.operators.text_pt.eval.Qurating.qurater_annotate import ModelAnnotator
from dataflow.operators.text_pt.eval.Qurating.qurater_annotate import TokenizeAndChunk
import torch

@OPERATOR_REGISTRY.register()
class QuratingSampleEvaluator(OperatorABC):
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
        if lang == "zh":
            return (
                "通过Qurating模型(princeton-nlp/QuRater-1.3B)从四个维度评估文本质量：写作风格(writing_style)、所需专业程度(required_expertise)、" 
                "事实与趣闻(facts_and_trivia)和教育价值(educational_value)。每个维度返回0-1之间的分数，综合评估文本的整体质量。\n" 
                "输入参数：\n" 
                "- text: 待评估的文本字符串\n" 
                "- labels: 评估维度列表，默认为['writing_style', 'required_expertise', 'facts_and_trivia', 'educational_value']\n" 
                "输出参数：\n" 
                "- dict: 包含各维度分数的字典，键为维度名称，值为0-1之间的分数"
            )
        else:
            return (
                "Evaluate text quality across four dimensions using the Qurating model (princeton-nlp/QuRater-1.3B): writing_style, required_expertise, " 
                "facts_and_trivia, and educational_value. Each dimension returns a score between 0 and 1, providing a comprehensive assessment of overall text quality.\n" 
                "Input parameters:\n" 
                "- text: Text string to be evaluated\n" 
                "- labels: List of evaluation dimensions, default ['writing_style', 'required_expertise', 'facts_and_trivia', 'educational_value']\n" 
                "Output parameters:\n" 
                "- dict: Dictionary containing scores for each dimension, with keys as dimension names and values as scores between 0 and 1"
            )

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
