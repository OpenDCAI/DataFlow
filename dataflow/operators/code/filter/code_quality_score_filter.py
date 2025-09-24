import pandas as pd
import numpy as np
from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.core import LLMServingABC
from dataflow.operators.code.eval.code_quality_sample_evaluator import CodeQualitySampleEvaluator

@OPERATOR_REGISTRY.register()
class CodeQualityScoreFilter(OperatorABC):
    """
    CodeQualityScoreFilter filters code samples based on LLM-generated quality scores
    from CodeQualityEvaluator. It evaluates code correctness, completeness, clarity,
    best practices, and efficiency, then filters out samples below the specified threshold.
    
    This filter uses evaluator scores to filter:
    - Removes code with syntax errors or logical issues
    - Removes incomplete or poorly structured code
    - Removes code that doesn't follow best practices
    - Keeps code with quality scores within specified range
    """

    def __init__(self, llm_serving: LLMServingABC, min_score: int = 7, max_score: int = 10):
        """
        Initializes the operator with LLM serving and evaluator.
        """
        self.min_score = min_score
        self.max_score = max_score
        self.logger = get_logger()
        self.logger.info(f"Initializing {self.__class__.__name__} with min_score: {self.min_score} and max_score: {self.max_score}...")
        self.scorer = CodeQualityEvaluator(llm_serving)
    
    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "基于LLM生成的代码质量分数过滤代码样本，评估正确性、完整性、清晰度、最佳实践和效率。\n\n"
                "评估维度：\n"
                "- 正确性：代码语法和逻辑是否正确\n"
                "- 完整性：代码是否完整实现功能\n"
                "- 清晰度：代码是否清晰易懂\n"
                "- 最佳实践：是否遵循编程最佳实践\n"
                "- 效率：代码执行效率如何\n\n"
                "输入参数：\n"
                "- input_key: 输入字段名（需要包含指令和代码列）\n"
                "- output_key: 输出标签字段名 (默认: 'quality_score')\n"
                "- min_score: 最小质量分数阈值 (默认: 7)\n"
                "- max_score: 最大质量分数阈值 (默认: 10)\n\n"
                "输出参数：\n"
                "- 过滤后的DataFrame，仅保留质量分数在指定范围内的代码样本\n"
                "- 返回包含质量分数标签字段名的列表"
            )
        else:
            return (
                "Filter code samples based on LLM-generated quality scores evaluating correctness, completeness, clarity, best practices, and efficiency.\n\n"
                "Evaluation Dimensions:\n"
                "- Correctness: syntax and logic accuracy\n"
                "- Completeness: functional completeness\n"
                "- Clarity: code readability and understandability\n"
                "- Best Practices: adherence to programming standards\n"
                "- Efficiency: execution performance\n\n"
                "Input Parameters:\n"
                "- input_key: Input field name (requires instruction and code columns)\n"
                "- output_key: Output label field name (default: 'quality_score')\n"
                "- min_score: Minimum quality score threshold (default: 7)\n"
                "- max_score: Maximum quality score threshold (default: 10)\n\n"
                "Output Parameters:\n"
                "- Filtered DataFrame containing only code samples with quality scores within specified range\n"
                "- List containing quality score label field names"
            )

    def _validate_dataframe(self, dataframe: pd.DataFrame):
        """
        Validates the DataFrame to ensure the required columns exist.
        """
        required_keys = [self.input_key]

        missing = [k for k in required_keys if k not in dataframe.columns]

        if missing:
            raise ValueError(f"Missing required column(s) for CodeQualityScoreFilter: {missing}")

    def _apply_score_filter(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the score-based filtering logic.
        
        Args:
            dataframe: Input DataFrame
            
        Returns:
            Filtered DataFrame
        """
        # Filter based on score range
        score_filter = (dataframe["quality_score"] >= self.min_score) & (dataframe["quality_score"] <= self.max_score)
        # Also keep samples with failed parsing (score = 0)
        nan_filter = dataframe["quality_score"] == 0
        final_filter = score_filter | nan_filter
        
        return dataframe[final_filter]


    def run(self, storage: DataFlowStorage, input_key: str, output_key: str = 'quality_score_filter_label'):
        self.input_key = input_key
        self.output_key = output_key
        self.logger.info(f"Running {self.__class__.__name__} with input_key: {self.input_key} and output_key: {self.output_key}...")
        
        dataframe = storage.read("dataframe")
        
        # Use existing quality_score if available, otherwise evaluate
        if "quality_score" not in dataframe.columns:
            scores, feedbacks = self.scorer.eval(dataframe, self.input_key)
            dataframe["quality_score"] = scores
            dataframe["quality_feedback"] = feedbacks
        
        # Apply filtering based on existing quality_score
        results = np.ones(len(dataframe), dtype=int)
        score_filter = (dataframe["quality_score"] >= self.min_score) & (dataframe["quality_score"] <= self.max_score)
        nan_filter = dataframe["quality_score"] == 0  # Keep failed parsing samples
        metric_filter = score_filter | nan_filter
        results = results & metric_filter.astype(int)
        
        self.logger.debug(f"Filtered by quality score, {np.sum(results)} data remained")
        dataframe[self.output_key] = metric_filter.astype(int)
        
        filtered_dataframe = dataframe[results == 1]
        output_file = storage.write(filtered_dataframe)
        self.logger.info(f"Filtering completed. Total records passing filter: {len(filtered_dataframe)}.")
        
        return [self.output_key]
