import numpy as np
from dataflow import get_logger
from dataflow.core import OperatorABC, LLMServingABC
from dataflow.utils.storage import DataFlowStorage
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.operators.text_sft import TreeinstructSampleEvaluator

@OPERATOR_REGISTRY.register()
class TreeinstructFilter(OperatorABC):

    def __init__(self, min_score: int = 7, max_score: int = 100, llm_serving: LLMServingABC = None):
        self.logger = get_logger()
        self.min_score = min_score
        self.max_score = max_score
        self.scorer = TreeinstructSampleEvaluator(llm_serving=llm_serving)
        self.logger.info(f"Initializing {self.__class__.__name__} with min_score = {min_score} and max_score = {max_score}")
    
    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "基于TreeinstructScore打分器的得分对数据进行过滤。通过生成语法树的节点数来衡量指令复杂性，节点越多表示指令越复杂。\n"
                "适用于筛选特定复杂度范围内的指令数据，平衡数据集难度分布，优化模型训练效果。\n"
                "输入参数：\n"
                "- min_score：保留样本的最小语法树节点数阈值，默认为7\n"
                "- max_score：保留样本的最大语法树节点数阈值，默认为100\n"
                "- llm_serving：LLM服务对象，需实现LLMServingABC接口\n"
                "- input_key：输入指令字段名\n"
                "- output_key：语法树节点数字段名，默认为'TreeinstructScore'\n"
                "输出参数：\n"
                "- 过滤后的DataFrame，仅保留语法树节点数在[min_score, max_score]范围内的样本\n"
                "- 返回包含语法树节点数字段名的列表，用于后续算子引用"
            )
        elif lang == "en":
            return (
                "Filter data using scores from the TreeinstructScore. Measure instruction complexity by the number of nodes in the generated syntax tree; more nodes indicate more complex instructions.\n"
                "Suitable for selecting instruction data within specific complexity ranges, balancing dataset difficulty distribution and optimizing model training effectiveness.\n"
                "Input Parameters:\n"
                "- min_score: Minimum syntax tree node count threshold for retaining samples, default is 7\n"
                "- max_score: Maximum syntax tree node count threshold for retaining samples, default is 100\n"
                "- llm_serving: LLM serving object implementing LLMServingABC interface\n"
                "- input_key: Input instruction field name\n"
                "- output_key: Syntax tree node count field name, default is 'TreeinstructScore'\n"
                "Output Parameters:\n"
                "- Filtered DataFrame containing only samples with syntax tree node count within [min_score, max_score] range\n"
                "- List containing syntax tree node count field name for subsequent operator reference"
            )
        else:
            return "Filter data based on instruction complexity measured by syntax tree node count."
        
    def run(self, storage: DataFlowStorage, input_key: str, output_key: str = 'TreeinstructScore'):
        self.input_key = input_key
        self.output_key = output_key
        dataframe = storage.read("dataframe")
        self.logger.info(f"Running {self.__class__.__name__} with input_key = {self.input_key} and output_key = {self.output_key}...")

        # Get the scores for filtering
        scores = np.array(self.scorer.eval(dataframe, self.input_key))

        dataframe[self.output_key] = scores
        filtered_dataframe = dataframe[(scores >= self.min_score) & (scores <= self.max_score)]

        # Write the filtered dataframe back to storage
        storage.write(filtered_dataframe)

        self.logger.info(f"Filtering completed. Total records passing filter: {len(filtered_dataframe)}.")

        return [self.output_key]
