import pandas as pd
import random
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC, LLMServingABC

# The prompt template remains the same.
REASON_TEMPLATE = '''# Scenario
{}

# Question
{}

# Retrieved Documents
{}

# Instructions:
1. Identify the essential problem.
2. Identify the helpful information to address the questions. Not all retrieved documents are relevant.
3. Think step by step to reason and draft an answer with as many thoughts as you have.
'''

@OPERATOR_REGISTRY.register()
class ReasonDistill(OperatorABC):
    '''
    ReasonDistill distills reasoning capabilities from LLMs by generating a step-by-step thought process.
    It takes a question, a scenario, a positive document, and several hard negative documents as input.
    '''

    def __init__(self, llm_serving: LLMServingABC):
        """
        Initializes the operator with a dependency-injected LLM serving instance.
        Args:
            llm_serving: An object that conforms to the LLMServingABC interface for making LLM calls.
        """
        self.logger = get_logger()
        self.llm_serving = llm_serving

    @staticmethod
    def get_desc(lang: str = "zh") -> str:
        if lang == "zh":
            return (
                "RAREPipeline: ReasonDistill 算子通过组合正负示例文档，提示大语言模型生成详细的推理过程。\n\n"
                "输入参数：\n"
                "- input_text_key: 包含正面文档的字段名。\n"
                "- input_question_key: 包含问题的字段名。\n"
                "- input_scenario_key: 包含情景的字段名。\n"
                "- input_hardneg_key: 包含困难负样本列表的字段名。\n"
                "- output_key: 用于存储生成推理过程的字段名。\n"
            )
        elif lang == "en":
            return (
                "RAREPipeline: ReasonDistill operator distills reasoning capabilities from LLMs by generating a step-by-step thought process.\n\n"
                "Input Parameters:\n"
                "- input_text_key: Field name for the positive document.\n"
                "- input_question_key: Field name for the question.\n"
                "- input_scenario_key: Field name for the scenario.\n"
                "- input_hardneg_key: Field name for the list of hard negatives.\n"
                "- output_key: Field name for storing the generated reasoning.\n"
            )
        else:
            return "RAREPipeline: ReasonDistill operator distills reasoning capabilities from LLMs."

    def _validate_dataframe(self, dataframe: pd.DataFrame, required_keys: list, output_key: str):
        """Validates that the input dataframe has the required columns and the output column doesn't exist."""
        missing = [k for k in required_keys if k not in dataframe.columns]
        if missing:
            raise ValueError(f"Missing required column(s): {missing}")

        if output_key in dataframe.columns:
            raise ValueError(f"The output column '{output_key}' already exists and would be overwritten.")

    def _build_prompts(
        self,
        dataframe: pd.DataFrame,
        text_key: str,
        question_key: str,
        scenario_key: str,
        hardneg_key: str
    ) -> list:
        """Builds a list of prompts for the LLM based on the dataframe rows."""
        prompts = []
        for _, row in dataframe.iterrows():
            text = row[text_key]
            question = row[question_key]
            scenario = row[scenario_key]
            hard_negatives = row[hardneg_key]

            # Combine the positive document with hard negatives and shuffle them
            documents = [text] + hard_negatives
            random.shuffle(documents)
            documents_str = "\n\n".join(documents)

            # Format the final prompt
            prompt = REASON_TEMPLATE.format(scenario.strip(), question.strip(), documents_str.strip())
            prompts.append(prompt)
        return prompts

    def run(
        self,
        storage: DataFlowStorage,
        input_text_key: str = "text",
        input_question_key: str = "question",
        input_scenario_key: str = "scenario",
        input_hardneg_key: str = "hard_negatives",
        output_key: str = "reasoning"
    ) -> list:
        """
        Executes the reasoning distillation process.
        """
        self.logger.info("Starting reasoning distillation process.")

        # 1. Read data using the storage abstraction
        dataframe = storage.read("dataframe")
        
        # 2. Validate input dataframe
        required_keys = [input_text_key, input_question_key, input_scenario_key, input_hardneg_key]
        self._validate_dataframe(dataframe, required_keys, output_key)
        
        # 3. Build prompts for each row
        self.logger.info(f"Building prompts for {len(dataframe)} rows...")
        prompts = self._build_prompts(
            dataframe,
            input_text_key,
            input_question_key,
            input_scenario_key,
            input_hardneg_key
        )
        
        # 4. Generate reasoning from LLM
        self.logger.info("Generating reasoning from LLM...")
        responses = self.llm_serving.generate_from_input(user_inputs=prompts, system_prompt="")
        
        # 5. Add responses to the dataframe
        dataframe[output_key] = responses
        
        # 6. Write data back using the storage abstraction
        output_file = storage.write(dataframe)
        self.logger.info(f"Reasoning distillation complete. Results saved to {output_file}")

        # 7. Return the name of the newly created column
        return [output_key]
