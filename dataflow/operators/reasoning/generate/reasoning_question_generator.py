from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC
from dataflow.core import LLMServingABC
from dataflow.prompts.reasoning.math import MathQuestionSynthesisPrompt
from dataflow.prompts.reasoning.general import GeneralQuestionSynthesisPrompt
from dataflow.prompts.reasoning.diy import DiyQuestionSynthesisPrompt
from dataflow.core.prompt import prompt_restrict, DIYPromptABC

import pandas as pd
import random
import re
from typing import Union

@prompt_restrict(
    MathQuestionSynthesisPrompt,
    GeneralQuestionSynthesisPrompt,
    DiyQuestionSynthesisPrompt
)

@OPERATOR_REGISTRY.register()
class ReasoningQuestionGenerator(OperatorABC):
    def __init__(self,
                num_prompts: int = 1,
                llm_serving: LLMServingABC = None,
                prompt_template: Union[MathQuestionSynthesisPrompt, GeneralQuestionSynthesisPrompt, DiyQuestionSynthesisPrompt, DIYPromptABC] = None
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
                "- input_key：输入问题字段名（run参数），必填，无默认值\n"
                "- output_synth_or_input_flag：写入问题来源标记的字段名，默认为'Synth_or_Input'\n"
                "输出参数：\n"
                "- 生成的问题作为新行追加到原始行之后，写在input_key字段中；新行的其他字段为空\n"
                "- output_synth_or_input_flag：标识问题来源，'input'表示原始问题，'synth'表示生成的新问题\n"
                "- 删除input_key为空或空字符串的行；若输入中已存在'Synth_or_Input'字段则报错\n"
                "- 返回[input_key, output_synth_or_input_flag]"
            )
        elif lang == "en":
            return (
                "Generates new questions based on existing ones. \n"
                "Input Parameters:\n"
                "- num_prompts: Number of questions to generate per input, integer between 1-5 (inclusive), default 1\n"
                "- llm_serving: LLM serving instance for question generation\n"
                "- prompt_template: Prompt template object for constructing generation prompts\n"
                "- input_key: Input question field name (run parameter), required, no default\n"
                "- output_synth_or_input_flag: Name of the field that receives the question-source flag, default 'Synth_or_Input'\n"
                "Output Parameters:\n"
                "- Generated questions are appended as new rows after the original rows, in the input_key field; other fields of the new rows are empty\n"
                "- output_synth_or_input_flag: Indicates question source, 'input' for original questions, 'synth' for generated questions\n"
                "- Rows whose input_key is missing or an empty string are dropped; raises an error if a 'Synth_or_Input' field already exists in the input\n"
                "- Returns [input_key, output_synth_or_input_flag]"
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
                    except Exception:
                        self.logger.debug(f"Please check if the symbol {{question}} in prompt is missing.")

        return formatted_prompts

    def _parse_response(self, response: str):
        # useful for reasoning llms. If response is in format of Deepseek thinking: <think>...</think><answer>...</answer>, keep only the answer part.
        pattern = r"<think>(.*?)</think><answer>(.*?)</answer>"
        matches = re.findall(pattern, response)
        if matches:
            return matches[0][1]
        else:
            return response
    
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
        responses = [self._parse_response(response) for response in responses]

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
