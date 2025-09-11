from dataflow.operators.text_pt import QuratingSampleEvaluator
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
        self.scorer = QuratingSampleEvaluator(map_batch_size=map_batch_size, 
                                     num_workers=num_workers, device_batch_size=device_batch_size, device=device, 
                                     labels=labels, model_cache_dir=model_cache_dir)
        
        self.logger.info(f"Initializing {self.__class__.__name__} with min_scores = {self.min_scores} and max_scores = {self.max_scores}...")

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "基于QuratingScorer打分器的得分对数据进行过滤。通过Qurating模型从四个维度评估文本质量：写作风格、所需专业知识、事实与 trivia 内容、教育价值。\n"
                "每个维度评分范围为0-9分，综合判断文本质量，可用于筛选高质量教育类或知识类内容。\n"
                "输入参数：\n"
                "- min_scores：各维度保留样本的最小分数阈值，默认为{'writing_style':0,'required_expertise':0,'facts_and_trivia':0,'educational_value':0}\n"
                "- max_scores：各维度保留样本的最大分数阈值，默认为{'writing_style':9,'required_expertise':9,'facts_and_trivia':9,'educational_value':9}\n"
                "- map_batch_size：映射批次大小，默认为512\n"
                "- num_workers：数据加载工作进程数，默认为1\n"
                "- device_batch_size：设备批次大小，默认为16\n"
                "- device：模型运行设备，默认为'cuda'\n"
                "- labels：评估维度列表，默认为['writing_style', 'required_expertise', 'facts_and_trivia', 'educational_value']\n"
                "- model_cache_dir：模型缓存目录，默认为'./dataflow_cache'\n"
                "输出参数：\n"
                "- 过滤后的DataFrame，仅保留所有维度分数均在对应阈值范围内的样本\n"
                "- 返回包含各维度过滤结果字段名的列表，用于后续算子引用"
            )
        elif lang == "en":
            return (
                "Filter data using scores from the QuratingScorer. Evaluate text quality across four dimensions using Qurating model: writing style, required expertise, facts and trivia content, and educational value.\n"
                "Each dimension is scored from 0-9, providing comprehensive quality assessment for filtering high-quality educational or knowledge-based content.\n"
                "Input Parameters:\n"
                "- min_scores: Minimum score thresholds for each dimension, default is {'writing_style':0,'required_expertise':0,'facts_and_trivia':0,'educational_value':0}\n"
                "- max_scores: Maximum score thresholds for each dimension, default is {'writing_style':9,'required_expertise':9,'facts_and_trivia':9,'educational_value':9}\n"
                "- map_batch_size: Mapping batch size, default is 512\n"
                "- num_workers: Number of data loading workers, default is 1\n"
                "- device_batch_size: Device batch size, default is 16\n"
                "- device: Model running device, default is 'cuda'\n"
                "- labels: List of evaluation dimensions, default is ['writing_style', 'required_expertise', 'facts_and_trivia', 'educational_value']\n"
                "- model_cache_dir: Model cache directory, default is './dataflow_cache'\n"
                "Output Parameters:\n"
                "- Filtered DataFrame containing only samples with all dimension scores within corresponding threshold ranges\n"
                "- List containing field names of each dimension's filtering results for subsequent operator reference"
            )
        else:
            return "Filter data based on multi-dimensional quality assessment using Qurating model."


    def run(self, storage: DataFlowStorage, input_key: str):
        self.input_key = input_key
        dataframe = storage.read("dataframe")
        self.logger.info(f"Running {self.__class__.__name__}...")

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
