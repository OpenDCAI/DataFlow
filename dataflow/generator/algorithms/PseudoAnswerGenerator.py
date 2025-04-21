from dataflow.generator.utils.Prompts import AnswerGeneratorPrompt
from dataflow.generator.utils.LocalModelGenerator import LocalModelGenerator
from dataflow.generator.utils.APIGenerator_aisuite import APIGenerator_aisuite
from dataflow.generator.utils.APIGenerator_request import APIGenerator_request
from collections import defaultdict, Counter
from .AnswerExtraction_qwenmatheval import AnswerExtraction_qwenmatheval
import yaml
import logging
import pandas as pd
from dataflow.utils.registry import GENERATOR_REGISTRY

@GENERATOR_REGISTRY.register()
class PseudoAnswerGenerator:
    '''
    Pseudo Answer Generator is a class that generates answers for given questions, then choose the most frequent answer.
    '''
    def __init__(self,config: dict):
        self.config = config
        self.prompt = AnswerGeneratorPrompt()
        self.input_file = self.config["input_file"]
        self.output_file = self.config["output_file"]
        self.input_key = self.config["input_key"]
        self.output_key_answer = self.config["output_key_answer"]
        self.output_key_answer_value = self.config["output_key_answer_value"]
        self.output_key_solutions = self.config["output_key_solutions"]
        self.output_key_correct_solution_example = self.config["output_key_correct_solution_example"]
        self.max_times = self.config["max_times"]
        self.model_generator = self.__init_model__()
        self.extractor = AnswerExtraction_qwenmatheval(self.config) 
    
    def __init_model__(self):
        if self.config["generator_type"] == "local":
            return LocalModelGenerator(self.config)
        elif self.config["generator_type"] == "aisuite":
            return APIGenerator_aisuite(self.config)
        elif self.config["generator_type"] == "request":
            return APIGenerator_request(self.config)
        else:
            raise ValueError(f"Invalid generator type: {self.config['generator_type']}")
    
    def run(self):
        # read input file : accept jsonl file only
        logging.info(f"Reading input file: {self.input_file}")
        dataframe = pd.read_json(self.input_file,lines=True)
        input_data_number = dataframe.shape[0]
        # check if input_prompt_key are in the dataframe
        if self.input_key not in dataframe.columns:
            key_list = dataframe.columns.tolist()
            raise ValueError(f"input_key: {self.input_key} not found in the dataframe, please check the input_key: {key_list}")
        # check if output_text_key are in the dataframe
        if self.output_key_answer in dataframe.columns:
            key_list = dataframe.columns.tolist()
            raise ValueError(f"Found {self.output_key_answer} in the dataframe, which leads to overwriting the existing column, please check the output_key: {key_list}")
        if self.output_key_solutions in dataframe.columns:
            key_list = dataframe.columns.tolist()
            raise ValueError(f"Found {self.output_key_solutions} in the dataframe, which leads to overwriting the existing column, please check the output_key: {key_list}")
        if self.output_key_correct_solution_example in dataframe.columns:
            key_list = dataframe.columns.tolist()
            raise ValueError(f"Found {self.output_key_correct_solution_example} in the dataframe, which leads to overwriting the existing column, please check the output_key: {key_list}")
        if self.output_key_answer_value in dataframe.columns:
            key_list = dataframe.columns.tolist()
            raise ValueError(f"Found {self.output_key_answer_value} in the dataframe, which leads to overwriting the existing column, please check the output_key: {key_list}")
        # generate text
        user_prompts = dataframe[self.input_key].tolist()
        answer_dict = defaultdict(list)
        solution_dict = defaultdict(list)
        logging.info(f"Generating answers for {len(user_prompts)} questions")
        for i in range(self.max_times):
            logging.info(f"Generating: {i+1} times")
            solutions = self.model_generator.generate_text_from_input(user_prompts)
            answers = [self.extractor.answer_extractor.extract_answer(solution, self.extractor.data_name) for solution in solutions]
            for idx, answer in enumerate(answers):
                answer_dict[idx].append(answer)
                solution_dict[idx].append((answer, solutions[idx]))
        logging.info(f"Generating final answers")
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
        logging.info(f"Data number {input_data_number} -> {dataframe.shape[0]}")
        dataframe.to_json(self.output_file,orient="records",lines=True)