import numpy as np
from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.operators.text_pt import PerplexitySampleEvaluator

@OPERATOR_REGISTRY.register()
class PerplexityFilter(OperatorABC):
    def __init__(self, min_score: float = 10.0, max_score: float = 500.0, model_name: str = 'gpt2', device='cuda'):
        self.logger = get_logger()
        self.min_score = min_score
        self.max_score = max_score
        self.scorer = PerplexitySampleEvaluator(model_name=model_name, device=device)
        self.logger.info(f"Initializing {self.__class__.__name__} with min_score = {self.min_score} and max_score = {self.max_score}")
    
    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "基于PerplexityScorer打分器的得分对数据进行过滤。基于Huggingface模型计算文本的困惑度，困惑度越低，文本的流畅性和可理解性越高。\n"
                "输入参数：\n"
                "- min_score：最小困惑度阈值（包含），默认为10.0\n"
                "- max_score：最大困惑度阈值（包含），默认为500.0\n"
                "- model_name：Huggingface模型路径或名称，默认为'gpt2'\n"
                "- device：模型运行设备，默认为'cuda'\n"
                "- input_key：输入文本字段名，无默认值，必须指定\n"
                "- output_key：写入困惑度的输出字段名，默认为'PerplexityScore'\n"
                "输出参数：\n"
                "- 过滤后的DataFrame，仅保留 min_score <= 困惑度 <= max_score 的文本；"
                "每行新增output_key字段，值为该行的浮点困惑度，该字段保留在输出中\n"
                "- 返回包含output_key的列表"
            )
        else:
            return (
                "Filter data using scores from the PerplexityScorer. Uses Huggingface model to calculate text perplexity; lower scores indicate better fluency and understandability.\n"
                "Input Parameters:\n"
                "- min_score: Minimum perplexity threshold (inclusive), default is 10.0\n"
                "- max_score: Maximum perplexity threshold (inclusive), default is 500.0\n"
                "- model_name: Huggingface model path or name, default is 'gpt2'\n"
                "- device: Model device, default is 'cuda'\n"
                "- input_key: Input text field name; no default, must be given\n"
                "- output_key: Name of the output field that receives the perplexity, default is 'PerplexityScore'\n\n"
                "Output Parameters:\n"
                "- Filtered DataFrame containing only texts with min_score <= perplexity <= max_score; "
                "each row gets a new output_key field holding its float perplexity, which is kept in the output\n"
                "- List containing output_key"
            )
        
    def run(self, storage: DataFlowStorage, input_key: str, output_key: str = 'PerplexityScore'):
        self.input_key = input_key
        self.output_key = output_key
        dataframe = storage.read("dataframe")
        self.logger.info(f"Running {self.__class__.__name__} with input_key = {self.input_key} and output_key = {self.output_key}...")

        # 获取得分并过滤
        scores = np.array(self.scorer.eval(dataframe, self.input_key))
        dataframe[self.output_key] = scores
        filtered_dataframe = dataframe[(scores >= self.min_score) & (scores <= self.max_score)]

        # 将过滤后的数据框写回存储
        storage.write(filtered_dataframe)

        self.logger.info(f"Filtering completed. Total records passing filter: {len(filtered_dataframe)}.")

        return [self.output_key]
