import pandas as pd
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger

from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC
from dataflow.core import LLMServingABC

@OPERATOR_REGISTRY.register()
class QAGenerator:
    '''
    SeedQAGenerator is a class that uses LLMs to generate QA pairs based on seed input.
    '''

    def __init__(self, llm_serving: LLMServingABC):
        self.logger = get_logger()
        self.llm_serving = llm_serving
    
    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "该算子用于生成对应文档片段的QA对。\n\n"
                "输入参数：\n"
                "- input_key: 包含文档片段的字段名\n"
                "- prompt_key: 包含提示词的字段名\n"
                "- output_quesion_key: 包含生成问题的字段名\n"
                "- output_answer_key: 包含生成答案的字段名\n"
            )
        elif lang == "en":
            return (
                "This operator generates QA pairs for given document fragments.\n\n"
                "Input Parameters:\n"
                "- input_key: Field name containing the content\n"
                "- prompt_key: Field name containing the generated prompt\n"
                "- output_quesion_key: Field name containing the generated question\n"
                "- output_answer_key: Field name containing the generated answer\n"
            )
        else:
            return "QAGenerator generates QA pairs for given document fragments."

    def _validate_dataframe(self, dataframe: pd.DataFrame):
        required_keys = [self.input_key]
        forbidden_keys = [self.output_question_key, self.output_answer_key]

        missing = [k for k in required_keys if k not in dataframe.columns]
        conflict = [k for k in forbidden_keys if k in dataframe.columns]

        if missing:
            raise ValueError(f"Missing required column(s): {missing}")
        if conflict:
            raise ValueError(f"The following column(s) already exist and would be overwritten: {conflict}")

    def _build_prompt(self, df):
        prompts = []
        for index, row in df.iterrows():
            prompts.append(row[self.prompt_key] + "Format:\nQ: ...\nA: ..." + "\nSeed data:\n" + row[self.input_key])
        return prompts

    def _parse_qa(self, response: str) -> tuple:
        lines = response.strip().split('\n')
        q = next((line[2:].strip() for line in lines if line.lower().startswith("q:")), "")
        a = next((line[2:].strip() for line in lines if line.lower().startswith("a:")), "")
        return q, a

    def run(
        self, 
        storage: DataFlowStorage, 
        input_key:str = "text", 
        prompt_key:str = "generated_prompt",
        output_quesion_key:str = "generated_question",
        output_answer_key:str = "generated_answer"
        ):
        '''
        Runs the answer generation process, reading from the input file and saving results to output.
        '''

        self.input_key, self.prompt_key, self.output_question_key, self.output_answer_key = input_key, prompt_key, output_quesion_key, output_answer_key

        dataframe = storage.read("dataframe")
        self._validate_dataframe(dataframe)
        formatted_prompts = self._build_prompt(dataframe)
        responses = self.llm_serving.generate_from_input(user_inputs=formatted_prompts, system_prompt="")

        questions, answers = zip(*[self._parse_qa(r) for r in responses])

        dataframe[self.output_question_key] = questions
        dataframe[self.output_answer_key] = answers

        output_file = storage.write(dataframe)
        self.logger.info(f"Results saved to {output_file}")

        return [self.output_question_key, self.output_answer_key]