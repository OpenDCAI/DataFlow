from dataflow.prompts.agenticrag import QAScorerPrompt
import pandas as pd
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
import re
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC
from dataflow.core import LLMServingABC

@OPERATOR_REGISTRY.register()
class QAScorer(OperatorABC):
    '''
    Answer Generator is a class that generates answers for given questions.
    '''
    def __init__(self, llm_serving: LLMServingABC):
        self.logger = get_logger()
        self.prompts = QAScorerPrompt()    
        self.llm_serving = llm_serving
    
    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "该算子用于为给的的文档片段生成种子QA对打分\n\n"
                "输入参数：\n"
                "- input_question_key: Field name containing the generated question\n"
                "- input_answer_key: Field name containing the generated answer\n"
                "- output_question_quality_key: Field name containing the question quality grade\n"
                "- output_question_quality_feedback_key: Field name containing the question quality feedback\n"
                "- output_answer_alignment_key: Field name containing the answer alignment grade\n"
                "- output_answer_alignment_feedback_key: Field name containing the answer alignment feedback\n"
                "- output_answer_verifiability_key: Field name containing the answer verifiability grade\n"
                "- output_downstream_value_key: Field name containing the downstream value grade\n"
                "- output_downstream_value_feedback_key: Field name containing the downstream value feedback\n"
            )
        elif lang == "en":
            return (
                "This operator generates prompts for given document fragments to generate seed QA pairs.\n\n"
                "Input Parameters:\n"
                "- input_question_key: Field name containing the generated question\n"
                "- input_answer_key: Field name containing the generated answer\n"
                "- output_question_quality_key: Field name containing the question quality grade\n"
                "- output_question_quality_feedback_key: Field name containing the question quality feedback\n"
                "- output_answer_alignment_key: Field name containing the answer alignment grade\n"
                "- output_answer_alignment_feedback_key: Field name containing the answer alignment feedback\n"
                "- output_answer_verifiability_key: Field name containing the answer verifiability grade\n"
                "- output_downstream_value_key: Field name containing the downstream value grade\n"
                "- output_downstream_value_feedback_key: Field name containing the downstream value feedback\n"
            )
        else:
            return "QAScorer scores QA pairs for given document fragments."

    def _validate_dataframe(self, dataframe: pd.DataFrame):
        required_keys = [self.input_question_key, self.input_answer_key]
        forbidden_keys = [self.output_question_quality_key, self.output_question_quality_feedback_key, self.output_answer_alignment_key, self.output_answer_alignment_feedback_key, self.output_answer_verifiability_key, self.output_answer_verifiability_feedback_key, self.output_downstream_value_key, self.output_downstream_value_feedback_key]

        missing = [k for k in required_keys if k not in dataframe.columns]
        conflict = [k for k in forbidden_keys if k in dataframe.columns]

        if missing:
            raise ValueError(f"Missing required column(s): {missing}")
        if conflict:
            raise ValueError(f"The following column(s) already exist and would be overwritten: {conflict}")

    def _build_prompts(self, dataframe):
        """
        Reformat the prompts in the dataframe to generate questions.
        """
        question_quality_inputs = []
        question_quality_prompt = self.prompts.question_quality_prompt()
        answer_alignment_inputs = []
        answer_alignment_prompt = self.prompts.answer_alignment_prompt()
        answer_verifiability_inputs = []
        answer_verifiability_prompt = self.prompts.answer_verifiability_prompt()
        downstream_value_inputs = []
        downstream_value_prompt = self.prompts.downstream_value_prompt()

        for index, row in dataframe.iterrows():
            question_quality_content = question_quality_prompt + "Question: " + row[self.input_question_key] + "\n" + "Answer: " + row[self.input_answer_key]
            question_quality_inputs.append(question_quality_content)
            answer_alignment_content = answer_alignment_prompt + "Question: " + row[self.input_question_key] + "\n" + "Answer: " + row[self.input_answer_key]
            answer_alignment_inputs.append(answer_alignment_content)
            answer_verifiability_content = answer_verifiability_prompt + "Question: " + row[self.input_question_key] + "\n" + "Answer: " + row[self.input_answer_key]
            answer_verifiability_inputs.append(answer_verifiability_content)
            downstream_value_content = downstream_value_prompt + "Question: " + row[self.input_question_key] + "\n" + "Answer: " + row[self.input_answer_key]
            downstream_value_inputs.append(downstream_value_content)

        return question_quality_inputs, answer_alignment_inputs, answer_verifiability_inputs, downstream_value_inputs

    def _parse_grade_and_feedback(self, response: str) -> tuple:
        grading_match = re.search(r"\*\*Grading\*\*:\s*(\d+)", response)
        feedback_match = re.search(r"\*\*Feedback\*\*:\s*(.+)", response, re.DOTALL)
        grading = float(grading_match.group(1)) if grading_match else 0
        feedback = feedback_match.group(1).strip() if feedback_match else ''

        return grading, feedback

    def run(
        self,
        storage: DataFlowStorage,
        input_question_key: str = "generated_question",
        input_answer_key: str = "generated_answer",
        output_question_quality_key: str = "question_quality_grades",
        output_question_quality_feedback_key: str = "question_quality_feedbacks",
        output_answer_alignment_key: str = "answer_alignment_grades",
        output_answer_alignment_feedback_key: str = "answer_alignment_feedbacks",
        output_answer_verifiability_key: str = "answer_verifiability_grades",
        output_answer_verifiability_feedback_key: str = "answer_verifiability_feedbacks",
        output_downstream_value_key: str = "downstream_value_grades",
        output_downstream_value_feedback_key: str = "downstream_value_feedbacks"
    ):
        self.input_question_key, self.input_answer_key, self.output_question_quality_key, self.output_question_quality_feedback_key, self.output_answer_alignment_key, self.output_answer_alignment_feedback_key, self.output_answer_verifiability_key, self.output_answer_verifiability_feedback_key, self.output_downstream_value_key, self.output_downstream_value_feedback_key = input_question_key, input_answer_key, output_question_quality_key, output_question_quality_feedback_key, output_answer_alignment_key, output_answer_alignment_feedback_key, output_answer_verifiability_key, output_answer_verifiability_feedback_key, output_downstream_value_key, output_downstream_value_feedback_key

        dataframe = storage.read("dataframe")
        self._validate_dataframe(dataframe)

        # 构建prompt
        q_inputs, a_inputs, v_inputs, d_inputs = self._build_prompts(dataframe)

        # 生成四类分数和反馈
        self.logger.info("Scoring question quality...")
        q_scores = self.llm_serving.generate_from_input(user_inputs=q_inputs, system_prompt="")
        q_grades, q_feedbacks = zip(*[self._parse_grade_and_feedback(r) for r in q_scores])

        self.logger.info("Scoring answer alignment...")
        a_scores = self.llm_serving.generate_from_input(user_inputs=a_inputs, system_prompt="")
        a_grades, a_feedbacks = zip(*[self._parse_grade_and_feedback(r) for r in a_scores])

        self.logger.info("Scoring answer verifiability...")
        v_scores = self.llm_serving.generate_from_input(user_inputs=v_inputs, system_prompt="")
        v_grades, v_feedbacks = zip(*[self._parse_grade_and_feedback(r) for r in v_scores])

        self.logger.info("Scoring downstream value...")
        d_scores = self.llm_serving.generate_from_input(user_inputs=d_inputs, system_prompt="")
        d_grades, d_feedbacks = zip(*[self._parse_grade_and_feedback(r) for r in d_scores])

        # 写回结果
        dataframe[self.output_question_quality_key] = q_grades
        dataframe[self.output_question_quality_feedback_key] = q_feedbacks
        dataframe[self.output_answer_alignment_key] = a_grades
        dataframe[self.output_answer_alignment_feedback_key] = a_feedbacks
        dataframe[self.output_answer_verifiability_key] = v_grades
        dataframe[self.output_answer_verifiability_feedback_key] = v_feedbacks
        dataframe[self.output_downstream_value_key] = d_grades
        dataframe[self.output_downstream_value_feedback_key] = d_feedbacks

        output_file = storage.write(dataframe)
        self.logger.info(f"Results saved to {output_file}")

        return [
            output_question_quality_key, output_question_quality_feedback_key,
            output_answer_alignment_key, output_answer_alignment_feedback_key,
            output_answer_verifiability_key, output_answer_verifiability_feedback_key,
            output_downstream_value_key, output_downstream_value_feedback_key
        ]