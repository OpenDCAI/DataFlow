from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.operators.text_pt import TextbookSampleEvaluator

@OPERATOR_REGISTRY.register()
class TextbookFilter(OperatorABC):

    def __init__(self, min_score=0.99, max_score=1, model_cache_dir:str='./dataflow_cache'):
        self.logger = get_logger()
        self.min_score = min_score
        self.max_score = max_score
        self.scorer = TextbookSampleEvaluator(model_cache_dir=model_cache_dir)
        self.logger.info(f"Initializing {self.__class__.__name__} with min_score = {min_score} and max_score = {max_score}")
    
    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "基于TextbookScorer打分器的得分对数据进行过滤。使用FastText分类器评估文本的教育价值，判断文本是否适合作为教材内容。\n"
                "分类器经过训练可识别具有教育意义、结构清晰、知识准确的文本，适用于构建教育类数据集。\n"
                "输入参数：\n"
                "- min_score：保留样本的最小教育价值分数阈值，默认为0.99\n"
                "- max_score：保留样本的最大教育价值分数阈值，默认为1.0\n"
                "- model_cache_dir：模型缓存目录，默认为'./dataflow_cache'\n"
                "- input_key：输入文本字段名\n"
                "- output_key：教育价值分数字段名，默认为'TextbookScore'\n"
                "输出参数：\n"
                "- 过滤后的DataFrame，仅保留教育价值分数在[min_score, max_score]范围内的样本\n"
                "- 返回包含教育价值分数字段名的列表，用于后续算子引用"
            )
        elif lang == "en":
            return (
                "Filter data using scores from the TextbookScorer. Assess educational value of text using FastText classifier to determine if text is suitable as educational material.\n"
                "Classifier is trained to identify text with educational significance, clear structure, and accurate knowledge, suitable for building educational datasets.\n"
                "Input Parameters:\n"
                "- min_score: Minimum educational value score threshold for retaining samples, default is 0.99\n"
                "- max_score: Maximum educational value score threshold for retaining samples, default is 1.0\n"
                "- model_cache_dir: Model cache directory, default is './dataflow_cache'\n"
                "- input_key: Input text field name\n"
                "- output_key: Educational value score field name, default is 'TextbookScore'\n"
                "Output Parameters:\n"
                "- Filtered DataFrame containing only samples with educational value scores within [min_score, max_score] range\n"
                "- List containing educational value score field name for subsequent operator reference"
            )
        else:
            return "Filter data based on educational value assessment using FastText textbook classifier."

    def run(self, storage: DataFlowStorage, input_key: str, output_key: str='TextbookScore'):
        self.input_key = input_key
        self.output_key = output_key
        dataframe = storage.read("dataframe")
        scores = self.scorer.eval(dataframe, self.input_key)
        dataframe[self.output_key] = scores
        filtered_dataframe = dataframe[(dataframe[self.output_key] >= self.min_score) & (dataframe[self.output_key] <= self.max_score)]
        output_file = storage.write(filtered_dataframe)
        self.logger.info(f"Filtering completed. Total records passing filter: {len(filtered_dataframe)}.")
        return [self.output_key]
        
        