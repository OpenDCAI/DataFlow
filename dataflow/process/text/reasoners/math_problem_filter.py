"""
A filter class for evaluating mathematical problems using an API.
This class constructs prompts for mathematical problem evaluation,
sends them to the API, and returns a list of 0s and 1s indicating
whether each problem is correctly formatted and solvable.

Features:
- Constructs evaluation prompts with four progressive checks
- Handles API call failures gracefully
- Processes datasets in parallel for efficiency
- Returns results as a list of 0s and 1s

Usage:
1. Initialize the filter with appropriate parameters
2. Call filter_func with a dataset (list of JSONL lines)
3. Get results as a list of 0s and 1s
"""

from dataflow.utils.api_utils import api_chat
import json
import time
import re
from dataflow.core import ReasonerFilter
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from dataflow.utils.registry import PROCESSOR_REGISTRY

@PROCESSOR_REGISTRY.register()
class MathProblemFilter(ReasonerFilter):
    def __init__(self, args_dict: dict):
        super().__init__(args_dict)
        self.system_prompt = args_dict.get("system_prompt", 
            "You are an expert in evaluating mathematical problems. Follow the user's instructions strictly and output your final judgment in the required JSON format."
        )
        # need to set api_key first
        self.model = self.model_name
        self.api_key = args_dict.get("api_key", "")
        self.filter_name = 'MathProblemFilter'

    def build_prompt(self, question):
        """Constructs an evaluation prompt with four progressive checks"""
        prompt = f"""You are given a mathematical problem. Follow these four steps in order and stop at the first failure:
1. Check only for spelling, grammar, and LaTeX formatting correctness. Do not interpret semantic meaning.
2. For each minimal condition stated in the problem (that cannot be further decomposed), check if it violates the mathematical domain or objective facts (for example, 'half a person' is incorrect). Note: Magical operations are acceptable if the necessary assumption is explicitly stated. Average values (e.g., 15.5 items per minute) are acceptable.
3. Check whether the problem-solving process contains any contradictions. This includes any two minimal conditions contradicting each other or if the final solution would be unreasonable (including unsolvable).
4. If steps 1-3 pass, check whether the problem is fully solvable by verifying that all necessary conditions to answer the question are provided. This check should be based solely on the question.
    
After performing these steps in sequence, output your final judgment in JSON format with exactly the following keys:
{{
    "judgement_test": true/false,
    "error_type": "<error description or null>"
}}
You may include your chain-of-thought, but the final answer must be the JSON object above.
    
Here is the problem to evaluate:
-------------------------------
{question}
-------------------------------
"""
        return prompt

    def process_problem(self, problem):
        """Processes a single problem by calling the API"""
        full_prompt = self.build_prompt(problem)
        try:
            response = api_chat(
                system_info=self.system_prompt,
                messages=full_prompt,
                model=self.model,
                api_url=self.api_url,
                api_key=self.api_key
            )
        except Exception as e:
            # API call failed, return 0
            print(f"API call failed for problem: {problem}. Error: {e}")
            return 0
        else:
            try:
                pattern = re.compile(r'"judgement_test"\s*:\s*(true|false)', re.IGNORECASE)
                match = pattern.search(response)
                # print("---match---",match)
                if match:
                    test_value = match.group(1).lower()
                return 1 if test_value == 'true' else 0
            except Exception as e:
                # Response format error, return 0
                print(f"Response format error for problem: {problem}. Error: {e}")
                return 0

    def filter_func(self, dataset):
        """Main filtering function that processes the entire dataset and returns a list of 0s and 1s"""
        results = []
        max_workers = 1  # Adjust based on your needs

        dataset = dataset.to_list()
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for record in dataset:
                try:
                    problem = record.get("problem", "")
                    if problem:
                        futures.append(executor.submit(self.process_problem, problem))
                except json.JSONDecodeError:
                    print(f"Invalid JSON format in line: {line}")
                    results.append(0)

            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
                result = future.result()
                results.append(result)
                # Add slight delay
                time.sleep(0.1)

        return results

    def handle_api_error(self, error_message):
        """Handles API errors and provides guidance"""
        print(f"API Error: {error_message}")
        print("Possible reasons:")
        print("1. Network connection issue. Please check your internet connection.")
        print("2. Invalid API URL. Please verify the URL format and accessibility.")
        print("3. API service unavailable. Please check if the service is running properly.")
        print("4. API key issue. Please ensure your API key is valid and has proper permissions.")
        print("Suggestion: Try again after checking the above issues.")