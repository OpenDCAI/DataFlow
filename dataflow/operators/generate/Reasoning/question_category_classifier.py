from dataflow.prompts.reasoning import QuestionCategoryPrompt
import pandas as pd
import json
import re
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger

from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC

from dataflow.utils.reasoning.CategoryFuzz import CategoryUtils
from dataflow.core import LLMServingABC

@OPERATOR_REGISTRY.register()
class QuestionCategoryClassifier(OperatorABC):
    def __init__(self, llm_serving: LLMServingABC = None):
        """
        Initialize the QuestionCategoryClassifier with the provided configuration.
        """
        self.logger = get_logger()
        self.prompts = QuestionCategoryPrompt()
        self.llm_serving = llm_serving

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "该算子用于对用户问题进行多级分类（主分类和子分类）。"
                "通过大语言模型对输入问题进行语义分析，输出分类编码结果。\n\n"
                "输入参数：\n"
                "- db_port/db_name/table_name：数据库连接参数（存储模式）\n"
                "- input_file/output_file：文件路径（文件模式）\n"
                "- input_key：输入数据中问题字段的键名\n"
                "- generator_type：模型调用方式（aisuite/request）\n\n"
                "输出参数：\n"
                "- classification_result：包含主分类和子分类的编码结果"
            )
        elif lang == "en":
            return (
                "Performs hierarchical classification (primary and secondary) on user questions. "
                "Utilizes LLM for semantic analysis and outputs category codes.\n\n"
                "Input Parameters:\n"
                "- db_port/db_name/table_name: Database connection params (storage mode)\n"
                "- input_file/output_file: File paths (file mode)\n"
                "- input_key: Key for question field in input data\n"
                "- generator_type: Model invocation method (aisuite/request)\n\n"
                "Output Parameters:\n"
                "- classification_result: Combined category code"
            )
        
    def _validate_dataframe(self, dataframe: pd.DataFrame):
        required_keys = [self.input_key]
        forbidden_keys = [self.output_key]

        missing = [k for k in required_keys if k not in dataframe.columns]
        conflict = [k for k in forbidden_keys if k in dataframe.columns]

        if missing:
            raise ValueError(f"Missing required column(s): {missing}")
        if conflict:
            raise ValueError(f"The following column(s) already exist and would be overwritten: {conflict}")
        
    def _reformat_prompt(self, dataframe):
        """
        Reformat the prompts in the dataframe to generate questions.
        """
        # Check if input_key is in the dataframe
        formatted_prompts = []
        for text in dataframe[self.input_key]:
            used_prompt = self.prompts.question_synthesis_prompt(text)
            formatted_prompts.append(used_prompt.strip())

        return formatted_prompts

    def run(self, storage: DataFlowStorage, input_key:str = "instruction", output_key:str="question_category") -> None:
        """
        Run the question category classification process.
        """
        self.input_key, self.output_key = input_key, output_key
        dataframe = storage.read("dataframe")
        self._validate_dataframe(dataframe)
        formatted_prompts = self._reformat_prompt(dataframe)
        responses = self.llm_serving.generate_from_input(formatted_prompts)

        for (idx, row), classification_str in zip(dataframe.iterrows(), responses):
            try:
                if not classification_str:
                    raise ValueError("空字符串")

                # 去除 Markdown 代码块包裹
                cleaned_str = re.sub(r"^```json\s*|\s*```$", "", classification_str.strip(), flags=re.DOTALL)

                # 去除非 ASCII 字符（可选）
                cleaned_str = re.sub(r"[^\x00-\x7F]+", "", cleaned_str)

                classification = json.loads(cleaned_str)

                primary_raw = classification.get("primary_category", "")
                secondary_raw = classification.get("secondary_category", "")

                category_info = CategoryUtils().normalize_categories(raw_primary=primary_raw, raw_secondary=secondary_raw)

                dataframe.at[idx, "primary_category"] = category_info["primary_category"]
                dataframe.at[idx, "secondary_category"] = category_info["secondary_category"]

            except json.JSONDecodeError:
                self.logger.warning(f"[警告] JSON 解析失败，收到的原始数据: {repr(classification_str)}")
            except Exception as e:
                self.logger.error(f"[错误] 解析分类结果失败: {e}")
                self.logger.debug(f"[DEBUG] 原始字符串：{repr(classification_str)}")

        output_file = storage.write(dataframe)
        self.logger.info(f"Classification results saved to {output_file}")

        return ["primary_category", "secondary_category"]