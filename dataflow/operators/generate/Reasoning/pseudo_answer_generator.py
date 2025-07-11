from collections import defaultdict, Counter
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
import pandas as pd
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC

from dataflow.prompts.reasoning import AnswerGeneratorPrompt
from dataflow.core import LLMServingABC
from dataflow.utils.reasoning.AnswerExtraction import StringCleaner, UnitTextManager, AnswerExtractor

@OPERATOR_REGISTRY.register()
class PseudoAnswerGenerator(OperatorABC):
    '''
    Pseudo Answer Generator is a class that generates answers for given questions, then choose the most frequent answer.
    '''
    def __init__(self, llm_serving: LLMServingABC = None, max_times: int = 3):
        self.logger = get_logger()
        self.prompts = AnswerGeneratorPrompt()
        self.llm_serving = llm_serving
        self.max_times = max_times
        
    def check_config(self):
        required_keys = ["input_file", "output_file", "input_key", "output_key_answer", "output_key_answer_value", "output_key_solutions", "output_key_correct_solution_example", "max_times"]
        missing_keys = [key for key in required_keys if key not in self.config]
        if missing_keys:
            raise ValueError(f"Missing required config keys: {missing_keys}")
    
    def get_extractor(self):
        unit_manager = UnitTextManager()
        string_cleaner = StringCleaner(unit_manager)
        answer_extractor = AnswerExtractor(string_cleaner)
        return answer_extractor
        
    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "该算子生成多个候选答案并通过统计选择最优解，实现伪答案生成。\n\n"
                "输入参数：\n"
                "- input_file：输入文件路径\n"
                "- output_file：输出文件路径\n"
                "- max_times：最大生成次数\n"
                "- selection_mode：统计选择模式（frequency/consistency）\n\n"
                "输出参数：\n"
                "- final_answer：最终选择答案字段\n"
                "- candidate_answers：候选答案列表字段"
            )
        elif lang == "en":
            return (
                "This operator generates multiple candidate answers and selects the optimal solution "
                "through statistical analysis.\n\n"
                "Input Parameters:\n"
                "- input_file: Input file path\n"
                "- output_file: Output file path\n"
                "- max_times: Maximum generation times\n"
                "- selection_mode: Statistical selection mode (frequency/consistency)\n\n"
                "Output Parameters:\n"
                "- final_answer: Selected answer field\n"
                "- candidate_answers: Candidate answers list field"
            )
        else:
            return "PseudoAnswerGenerator produces pseudo-answers through multi-round generation and selection."

    def _validate_dataframe(self, dataframe: pd.DataFrame):
        required_keys = [self.input_key]
        forbidden_keys = [
            self.output_key_answer,
            self.output_key_answer_value,
            self.output_key_solutions,
            self.output_key_correct_solution_example,
        ]

        missing = [k for k in required_keys if k not in dataframe.columns]
        conflict = [k for k in forbidden_keys if k in dataframe.columns]

        if missing:
            key_list = dataframe.columns.tolist()
            raise ValueError(
                f"read_key: {missing[0]} not found in the dataframe, "
                f"please check the read_key: {key_list}"
            )
        if conflict:
            key_list = dataframe.columns.tolist()
            raise ValueError(
                f"Found {conflict} in the dataframe, which leads to overwriting the existing column(s), "
                f"please check the output_key: {key_list}"
            )

    def run(
        self,
        storage: DataFlowStorage,
        input_key: str = "instruction",
        output_key_answer: str = "pseudo_answers",
        output_key_answer_value: str = "pseudo_answer_value",
        output_key_solutions: str = "pseudo_solutions",
        output_key_correct_solution_example: str = "pseudo_correct_solution_example",
        ):

        self.input_key, self.output_key_answer, self.output_key_answer_value = input_key, output_key_answer, output_key_answer_value
        self.output_key_solutions, self.output_key_correct_solution_example = output_key_solutions, output_key_correct_solution_example
        self.extractor = self.get_extractor()
        dataframe = storage.read("dataframe")
        self._validate_dataframe(dataframe)

        input_data_number = dataframe.shape[0]
        user_prompts = dataframe[self.input_key].tolist()
        answer_dict = defaultdict(list)
        solution_dict = defaultdict(list)
        self.logger.info(f"Generating answers for {len(user_prompts)} questions")
        for i in range(self.max_times):
            self.logger.info(f"Generating: {i+1} times")
            solutions = self.llm_serving.generate_from_input(user_prompts)
            answers = [self.extractor.extract_answer(solution, None) for solution in solutions]
            for idx, answer in enumerate(answers):
                answer_dict[idx].append(answer)
                solution_dict[idx].append((answer, solutions[idx]))
        self.logger.info(f"Generating final answers")
        dataframe[self.output_key_answer] = dataframe.get(self.output_key_answer, None) 
        dataframe[self.output_key_solutions] = dataframe.get(self.output_key_solutions, None) 
        dataframe[self.output_key_correct_solution_example] = dataframe.get(self.output_key_correct_solution_example, None) 
        for key, value in answer_dict.items():
            count = Counter(value)
            final_answer = count.most_common(1)[0][0]
            dataframe.at[int(key),self.output_key_answer] = value
            dataframe.at[int(key),self.output_key_solutions] = final_answer
            correct_contents = [content for ans, content in solution_dict[key] if ans == final_answer]
            dataframe.at[int(key), self.output_key_solutions] = correct_contents
            correct_solution_example = correct_contents[0] if correct_contents else None
            dataframe.at[int(key), self.output_key_correct_solution_example] = correct_solution_example
            dataframe.at[int(key), self.output_key_answer_value] = final_answer
        # 过滤掉没有答案的行
        dataframe = dataframe[dataframe[self.output_key_answer_value].notna()]
        dataframe = dataframe[dataframe[self.output_key_correct_solution_example].notna()]
        self.logger.info(f"Data number {input_data_number} -> {dataframe.shape[0]}")
        
        output_file = storage.write(dataframe)
        self.logger.info(f"PsedoAnswerGenerator's results saved to {output_file}")

        return [output_key_answer, output_key_answer_value, output_key_solutions, output_key_correct_solution_example]