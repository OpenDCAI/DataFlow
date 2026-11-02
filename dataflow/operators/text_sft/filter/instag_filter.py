from dataflow.operators.text_sft import InstagSampleEvaluator
from dataflow.core import OperatorABC
import numpy as np
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.utils import get_logger
from dataflow.utils.storage import DataFlowStorage

@OPERATOR_REGISTRY.register()
class InstagFilter(OperatorABC):

    def __init__(self, min_score=0.0, max_score=1.0, model_cache_dir='./dataflow_cache', device='cuda', max_new_tokens=1024):
        self.logger = get_logger()
        self.min_score = min_score
        self.max_score = max_score
        
        # Initialize the scorer
        self.scorer = InstagSampleEvaluator(
            model_cache_dir=model_cache_dir,
            device=device,
            max_new_tokens=max_new_tokens
        )
        self.logger.info(f"Initializing {self.__class__.__name__} with min_score = {self.min_score} and max_score = {self.max_score}...")

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return "基于InstagScorer打分器的过滤算子。使用预训练的Instag模型对指令进行分析，返回标签的数量来评估指令的内容多样性。参数包括模型缓存目录(model_cache_dir)、计算设备(device)和最大新生成标记数(max_new_tokens)。过滤范围由min_score（默认0.0）和max_score（默认1.0）参数控制，保留 min_score <= 标签数量 <= max_score 的行，标签越多表示内容多样性越大。run()参数：input_instruction_key为读取的指令字段名，默认为'instruction'；output_key为写入标签数量的输出字段名，默认为'InstagScore'。输出：过滤后的DataFrame，每行新增output_key字段，值为该行的整数标签数量，该字段保留在输出中；返回包含output_key的列表。"
        else:
            return "Filter operator based on InstagScorer. Uses pre-trained Instag model to analyze instructions, returning the number of tags to evaluate content diversity. Parameters include model cache directory (model_cache_dir), computing device (device), and maximum new tokens (max_new_tokens). Filter range is controlled by min_score (default 0.0) and max_score (default 1.0) parameters; rows with min_score <= tag count <= max_score are kept, with more tags indicating greater content diversity. run() parameters: input_instruction_key is the instruction field to read, default is 'instruction'; output_key is the output field that receives the tag count, default is 'InstagScore'. Output: Filtered DataFrame where each row gets a new output_key field holding its integer tag count, which is kept in the output; returns a list containing output_key."

    def run(self, storage: DataFlowStorage, input_instruction_key: str = 'instruction', output_key: str = 'InstagScore'):
        self.input_instruction_key = input_instruction_key
        self.output_key = output_key
        dataframe = storage.read("dataframe")
        self.logger.info(f"Running {self.__class__.__name__}...")

        scores = self.scorer.eval(dataframe, self.input_instruction_key)
        dataframe[self.output_key] = scores
        filtered_dataframe = dataframe[(dataframe[self.output_key] >= self.min_score) & (dataframe[self.output_key] <= self.max_score)]
        storage.write(filtered_dataframe)

        self.logger.info(f"Filtering completed. Total records passing filter: {len(filtered_dataframe)}.")

        return [self.output_key]
