from dataflow.operators.text_sft import SuperfilteringSampleEvaluator
import numpy as np
from dataflow.core import OperatorABC
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.utils import get_logger
from dataflow.utils.storage import DataFlowStorage

@OPERATOR_REGISTRY.register()
class SuperfilteringFilter(OperatorABC):

    def __init__(self, min_score=0.0, max_score=1.0, device='cuda', model_cache_dir='./dataflow_cache', max_length=512):
        self.logger = get_logger()
        self.min_score = min_score
        self.max_score = max_score
        
        self.scorer = SuperfilteringSampleEvaluator(
            device=device,
            model_cache_dir=model_cache_dir,
            max_length=max_length
        )
        self.logger.info(f"Initializing {self.__class__.__name__} with min_score = {self.min_score} and max_score = {self.max_score}...")

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "使用Superfiltering评分器过滤掉低质量数据。基于GPT-2模型计算困惑度比值来评估指令跟随难度，比值越低表示指令越容易被模型理解和执行。\n"
                "适用于筛选适合特定模型能力的指令数据，提高模型训练效率和效果。\n"
                "输入参数：\n"
                "- min_score：保留样本的最小分数阈值，默认为0.0\n"
                "- max_score：保留样本的最大分数阈值，默认为1.0\n"
                "- device：模型运行设备，默认为'cuda'\n"
                "- model_cache_dir：模型缓存目录，默认为'./dataflow_cache'\n"
                "- max_length：文本最大长度，默认为512\n"
                "- input_instruction_key：指令字段名，默认为'instruction'\n"
                "- input_input_key：输入字段名，默认为'input'\n"
                "- input_output_key：输出字段名，默认为'output'\n"
                "- output_key：过滤结果分数字段名，默认为'SuperfilteringScore'\n"
                "输出参数：\n"
                "- 过滤后的DataFrame，仅保留分数在[min_score, max_score]范围内的样本\n"
                "- 返回包含过滤结果分数字段名的列表，用于后续算子引用"
            )
        elif lang == "en":
            return (
                "Filter out low-quality data using the Superfiltering scorer. Evaluate instruction following difficulty by calculating perplexity ratio with GPT-2 model; lower ratios indicate instructions are easier for models to understand and execute.\n"
                "Suitable for selecting instruction data appropriate for specific model capabilities, improving model training efficiency and effectiveness.\n"
                "Input Parameters:\n"
                "- min_score: Minimum score threshold for retaining samples, default is 0.0\n"
                "- max_score: Maximum score threshold for retaining samples, default is 1.0\n"
                "- device: Model running device, default is 'cuda'\n"
                "- model_cache_dir: Model cache directory, default is './dataflow_cache'\n"
                "- max_length: Maximum text length, default is 512\n"
                "- input_instruction_key: Instruction field name, default is 'instruction'\n"
                "- input_input_key: Input field name, default is 'input'\n"
                "- input_output_key: Output field name, default is 'output'\n"
                "- output_key: Filter result score field name, default is 'SuperfilteringScore'\n"
                "Output Parameters:\n"
                "- Filtered DataFrame containing only samples with scores within [min_score, max_score] range\n"
                "- List containing filter result score field name for subsequent operator reference"
            )
        else:
            return "Filter low-quality data using perplexity ratio calculated with GPT-2 model."

    def run(self, storage: DataFlowStorage, input_instruction_key: str = 'instruction', input_input_key: str = 'input', input_output_key: str = 'output', output_key: str = "SuperfilteringScore"):
        self.input_instruction_key = input_instruction_key
        self.input_input_key = input_input_key
        self.input_response_key = input_output_key
        self.output_key = output_key
        dataframe = storage.read("dataframe")
        self.logger.info(f"Running {self.__class__.__name__ } with input_instruction_key = {self.input_instruction_key}, intput_output_key = {self.input_output_key}, output_key = {self.output_key}...")

        # Get the scores for filtering
        scores = self.scorer.eval(dataframe, input_instruction_key, input_input_key, input_output_key)
        dataframe[self.output_key] = scores
        filtered_dataframe = dataframe[(dataframe[self.output_key] >= self.min_score) & (dataframe[self.output_key] <= self.max_score)]
        storage.write(filtered_dataframe)

        self.logger.info(f"Filtering completed. Total records passing filter: {len(filtered_dataframe)}.")

        return [self.output_key]
