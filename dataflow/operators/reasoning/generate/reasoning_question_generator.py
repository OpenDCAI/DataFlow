from dataflow.prompts.reasoning.diy import DiyQuestionSynthesisPrompt
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC
from dataflow.core import LLMServingABC
import pandas as pd
import random
from dataflow.prompts.reasoning.math import MathQuestionSynthesisPrompt

@OPERATOR_REGISTRY.register()
class ReasoningQuestionGenerator(OperatorABC):
    def __init__(self,
                num_prompts: int = 1,
                llm_serving: LLMServingABC = None,
                prompt_template = None,
                ):
        """
        Initialize the QuestionGenerator with the provided configuration.
        """
        self.logger = get_logger()
        
        if prompt_template is None:
            prompt_template = MathQuestionSynthesisPrompt()
        self.prompts = prompt_template
        self.num_prompts = num_prompts
        self.llm_serving = llm_serving

        if self.num_prompts not in range(1,6):
            self.logger.debug("num_prompts must be an integer between 1 and 5 (inclusive)")

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "该算子用于基于现有问题生成新问题。\n"
                "输入参数：\n"
                "- num_prompts：生成问题的数量，整数，范围1-5（含），默认1\n"
                "- llm_serving：LLM服务实例，用于生成问题\n"
                "- prompt_template：提示模板对象，用于构建生成提示词\n"
                "输出参数：\n"
                "- 原始输入列（由input_key指定）：新增生成的问题\n"
                "- Synth_or_Input：标识问题来源，'input'表示原始问题，'synth'表示生成的新问题"
            )
        elif lang == "en":
            return (
                "Generates new questions based on existing ones. \n"
                "Input Parameters:\n"
                "- num_prompts: Number of questions to generate per input, integer between 1-5 (inclusive), default 1\n"
                "- llm_serving: LLM serving instance for question generation\n"
                "- prompt_template: Prompt template object for constructing generation prompts\n"
                "Output Parameters:\n"
                "- Original input column (specified by input_key): Contains newly generated questions\n"
                "- Synth_or_Input: Indicates question source, 'input' for original questions, 'synth' for generated questions"
            )
        elif lang == "en":
            return (
                "Generates new questions based on existing ones. "
                "Produces 1-5 new questions per original question.\n"
                "Input Parameters:\n"
                "- eval_stage: Evaluation stage identifier\n"
                "- read_min/max_score: Score filtering thresholds\n"
                "- Other params same as base classifier\n"
                "Output Parameters:\n"
                "- generated_questions: List of newly generated questions"
            )
        
    def _validate_dataframe(self, dataframe: pd.DataFrame):
        required_keys = [self.input_key]
        forbidden_keys = ["Synth_or_Input"]

        missing = [k for k in required_keys if k not in dataframe.columns]
        conflict = [k for k in forbidden_keys if k in dataframe.columns]

        if missing:
            raise ValueError(f"Missing required column(s): {missing}")
        if conflict:
            raise ValueError(f"The following column(s) already exist and would be overwritten: {conflict}")

        
    def _reformat_prompt(self, dataframe):
        """
        Reformat the prompts in the dataframe to generate questions based on num_prompts.
        """
        diversity_mode = [
            "1, 2, 3",
            "1, 2, 4",
            "1, 2, 5",
            "1, 4, 5",
            "1, 2, 3, 4, 5"
        ]

        formatted_prompts = []

        for question in dataframe[self.input_key]:
            if self.num_prompts == 0:
                formatted_prompts.append("")  # Skip generating for this question
            else:
                if not isinstance(self.prompts, DiyQuestionSynthesisPrompt):
                # Randomly choose the required number of transformations from diversity_mode
                    selected_items = random.sample(diversity_mode, self.num_prompts)
                    for selected_item in selected_items:
                        used_prompt = self.prompts.build_prompt(selected_item, question)
                        formatted_prompts.append(used_prompt.strip())
                else: ### diy prompt
                    try:
                        used_prompt = self.prompts.build_prompt(question=question)
                        formatted_prompts.append(used_prompt.strip())
                    except:
                        self.logger.debug(f"Please check if the symbol {{question}} in prompt is missing.")

        return formatted_prompts

    def run(
            self, 
            storage: DataFlowStorage, 
            input_key: str,
            output_synth_or_input_flag: str = "Synth_or_Input"
            ):
        """
        Run the question generation process.
        """
        self.input_key = input_key
        dataframe = storage.read("dataframe")
        self._validate_dataframe(dataframe)
        formatted_prompts = self._reformat_prompt(dataframe)
        responses = self.llm_serving.generate_from_input(formatted_prompts)

        new_rows = pd.DataFrame({
            input_key: responses,
        })
        new_rows[output_synth_or_input_flag] = "synth"
        dataframe[output_synth_or_input_flag] = "input"

        dataframe = pd.concat([dataframe, new_rows], ignore_index=True)
        dataframe = dataframe[dataframe[input_key].notna()]
        dataframe = dataframe[dataframe[input_key] != ""]

        output_file = storage.write(dataframe)
        self.logger.info(f"Generated questions saved to {output_file}")

        return [input_key, output_synth_or_input_flag]