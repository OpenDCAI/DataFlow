from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.core import LLMServingABC
from dataflow.operators.text_sft import AlpagasusSampleEvaluator

@OPERATOR_REGISTRY.register()
class AlpagasusFilter(OperatorABC):

    def __init__(self, min_score=3, max_score=5, llm_serving: LLMServingABC = None, dimension='quality'):
        self.logger = get_logger()
        self.min_score = min_score
        self.max_score = max_score
        self.logger.info(f"Initializing {self.__class__.__name__} with min_score = {self.min_score} and max_score = {self.max_score}...")
        self.scorer = AlpagasusSampleEvaluator(llm_serving, dimension)
    
    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "基于AlpagasusScorer打分器的得分对数据进行过滤。通过调用GPT模型评估指令的质量，返回一个质量得分。\n\n"
                "初始化参数：\n"
                "- min_score: 最低分数阈值，默认为3\n"
                "- max_score: 最高分数阈值，默认为5\n"
                "- llm_serving: LLM服务实例\n"
                "- dimension: 评估维度，默认为'quality'（质量）\n\n"
                "运行参数：\n"
                "- input_instruction_key: 输入指令字段名\n"
                "- input_input_key: 输入内容字段名\n"
                "- input_output_key: 输出内容字段名\n"
                "- output_key: 输出分数字段名，默认为'AlpagasusScore'\n\n"
                "过滤逻辑：保留分数在[min_score, max_score]范围内的数据"
            )
        else:
            return (
                "Filter data using scores from the AlpagasusScorer. Evaluate instruction quality using GPT model and return a quality score.\n\n"
                "Initialization Parameters:\n"
                "- min_score: Minimum score threshold, default is 3\n"
                "- max_score: Maximum score threshold, default is 5\n"
                "- llm_serving: LLM serving instance\n"
                "- dimension: Evaluation dimension, default is 'quality'\n\n"
                "Run Parameters:\n"
                "- input_instruction_key: Input instruction field name\n"
                "- input_input_key: Input content field name\n"
                "- input_output_key: Output content field name\n"
                "- output_key: Output score field name, default is 'AlpagasusScore'\n\n"
                "Filter Logic: Keep data with scores in [min_score, max_score] range"
            )


    def run(self, storage: DataFlowStorage, input_instruction_key: str, input_input_key: str, input_output_key: str, output_key: str='AlpagasusScore'):
        self.input_instruction_key = input_instruction_key
        self.input_input_key = input_input_key
        self.input_output_key = input_output_key
        self.output_key = output_key
        self.logger.info(f"Running {self.__class__.__name__} with input_instruction_key = {self.input_instruction_key}, input_input_key = {self.input_input_key}, input_output_key = {self.input_output_key} and output_key = {self.output_key}...")    
        dataframe = storage.read("dataframe")
        scores = self.scorer.eval(dataframe, self.input_instruction_key, self.input_input_key, self.input_output_key)
        dataframe[self.output_key] = scores
        filtered_dataframe = dataframe[(dataframe[self.output_key] >= self.min_score) & (dataframe[self.output_key] <= self.max_score)]
        output_file = storage.write(filtered_dataframe)
        self.logger.info(f"Filtering completed. Total records passing filter: {len(filtered_dataframe)}.")
        return [self.output_key]
        
        