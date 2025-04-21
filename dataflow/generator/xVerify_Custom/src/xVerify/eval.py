import os
import copy
import json
import datetime

from pathlib import Path
from multiprocessing import Pool
from typing import List

from tqdm import tqdm
from loguru import logger
from .model import Model
from .dataset_loader import DatasetLoader
from .prompts import PROMPT


class Evaluator:
    """
    Evaluator class for processing evaluation tasks using the xVerify model.
    """

    def __init__(
            self,
            model: Model,
            process_num: int = 5
    ):
        """
        Initialize the Evaluator instance.

        Args:
            model (LLMs): An instance of the xVerify model.
            process_num (int): Number of parallel processes to use for evaluation.
        """

        self.model = copy.deepcopy(model)
        self.model_name = model.model_name
        self.process_num = process_num
        self.prompt = PROMPT

    def load_data(self, data_path: str, data_size: int = None) -> List[dict]:
        """
        Load evaluation data from the specified path.
        
        Args:
            data_path (str): Path to the dataset file.
            data_size (int, optional): Number of data entries to load. If None, load all data.
        
        Returns:
            list: Loaded dataset entries.
        """

        data_size = data_size if data_size is not None else -1
        data = DatasetLoader.fixed_load(data_path, data_size)
        return data

    def construct_prompt(self, data: List[dict]) -> None:
        """
        Construct the prompt for each data item using the loaded template.
        
        Each data item is expected to be a dictionary containing 'question', 'llm_output',
        and 'correct_answer' keys. The constructed prompt is added to the data item.
        
        Args:
            data (list): List of data items (dictionaries) to be processed.
        """
        for item in data:
            user_input = self.prompt.format(
                question=item['question'],
                output=item['llm_output'],
                answer=item['correct_answer']
            )
            item['prompt'] = user_input
    
    def gen(self, data_point: dict) -> dict:
        """
        Generate the evaluation result for a single data point.
        
        This method sends the constructed prompt to the language model and appends the
        resulting judgment to the data point.
        
        Args:
            data_point (dict): A single data item containing a 'prompt' key.
        
        Returns:
            dict: The data item with an added key for the model's judgment result.
        """

        result = self.model.request(data_point['prompt'])
        data_point[f'{self.model_name}_judgment_result'] = result

        return data_point

    def batch_gen(self, data: List[dict], data_name: str) -> List[dict]:
        """
        Process a batch of data points in parallel and generate evaluation results.
        
        Uses a multiprocessing pool to apply the 'gen' method to each data point.
        
        Args:
            data (list): List of data items to process.
            data_name (str): Name identifier for the dataset (used in progress bar description).
        
        Returns:
            list: List of data items with their evaluation results.
        """

        with Pool(self.process_num) as pool:
            results = list(tqdm(
                pool.imap(self.gen, data),
                total=len(data),
                desc=f'{self.model_name}_{data_name}'
            ))
        return results
    
    def stat_results(self, results: List[dict]) -> dict:
        """
        Compute statistics based on the evaluation results.
        
        Evaluates each data point's judgment result and counts the number of valid,
        correct, and incorrect responses. Also computes the overall accuracy.
        
        Args:
            results (list): List of evaluated data items.
        
        Returns:
            dict: Statistics including valid count, correct count, incorrect count, and accuracy.
        """

        valid_label = ['correct', 'incorrect']
        valid_num = 0
        correct_num = 0
        incorrect_num = 0
        for item in results:
            item_label = item[f'{self.model_name}_judgment_result']
            item_label = item_label.strip().lower()
        
            if item_label in valid_label:
                valid_num += 1
                item['judge_valid'] = 'True'
            else:
                item['judge_valid'] = 'False'
            
            if item_label == 'correct':
                correct_num += 1
            elif item_label == 'incorrect':
                incorrect_num += 1
        
        return {
            "Valid_num": valid_num,
            "Correct_num": correct_num,
            "Incorrect_num": incorrect_num,
            "Accuracy": correct_num / len(results)
        }
    
    def save_output(self, output: dict) -> None:
        """
        Save the evaluation output to a JSON file.
        
        The output is saved to the directory specified by the instance's output_path.
        Creates the directory if it does not exist.
        
        Args:
            output (dict): Dictionary containing evaluation information, statistics, and results.
        """
        
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=4)
        
        logger.info(f"Output saved to {self.output_path}!")

    def single_evaluate(self, question: str, llm_output: str, correct_answer: str) -> str:
        """
        Evaluate a single data point.
        
        Constructs a prompt from the given question, model output, and correct answer,
        sends the prompt to the model, and returns the judgment result.
        
        Args:
            question (str): The question to evaluate.
            llm_output (str): The model's output for the question.
            correct_answer (str): The correct answer for comparison.
        
        Returns:
            str: The judgment result from the model.
        """

        data_point = {
            'question': question,
            'llm_output': llm_output,
            'correct_answer': correct_answer
        }

        user_input = self.prompt.format(
            question=data_point['question'],
            output=data_point['llm_output'],
            answer=data_point['correct_answer']
        )
        data_point['prompt'] = user_input

        result = self.gen(data_point)

        return result[f'{self.model_name}_judgment_result']

    def evaluate(self, data_path: str, output_path: str,data_size: int = None) -> dict:
        """
        Evaluate an entire dataset.
        
        This method loads the dataset, constructs prompts for each data point,
        processes the data points in parallel, computes statistics, and saves the
        output to a JSON file.
        
        Args:
            data_path (str): Path to the dataset file.
            data_size (int, optional): Number of data points to evaluate. If None, evaluate all.
        
        Returns:
            dict: Statistics of the evaluation results.
        """
        
        data = self.load_data(data_path, data_size)
        data_name = Path(data_path).stem
        data_size = len(data)

        info = {
            'llm': {
                "model_name": self.model_name,
                "temperature": self.model.temperature,
                "max_tokens": self.model.max_tokens,
                "top_p": self.model.top_p
            },
            'dataset': data_name,
            'data_num': data_size,
            'datetime': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }
        
        self.construct_prompt(data)
        results = self.batch_gen(data, data_name)
        stat_info = self.stat_results(results)

        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        output_name = f'Eval_Judge_{self.model_name}_{data_name}_{data_size}_{timestamp}.json'
        self.output_path = os.path.join(output_path, output_name)

        self.save_output({
            'info': info, 
            'stat_info': stat_info,
            'results': results
        })

        return stat_info
