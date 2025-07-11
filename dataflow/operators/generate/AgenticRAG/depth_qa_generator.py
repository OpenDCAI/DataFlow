from dataflow.prompts.agenticrag import DepthQAGeneratorPrompt
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow  import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC
from dataflow.core import LLMServingABC

import pandas as pd
import json

@OPERATOR_REGISTRY.register()
class DepthQAGenerator(OperatorABC):
    def __init__(self,
                 llm_serving: LLMServingABC = None
                 ):
        self.logger= get_logger()
        self.prompts = DepthQAGeneratorPrompt()
        self.llm_serving = llm_serving

    @staticmethod
    def get_desc(self, lang):
        if lang == "zh":
            return (
            )
        elif lang == "en":
            return (
            )
        else:
            return
    
    def _validate_dataframe(self, dataframe: pd.DataFrame):
        required_keys = [self.input_key]
        forbidden_keys = [self.output_key]

        missing = [k for k in required_keys if k not in dataframe.columns]
        conflict = [k for k in forbidden_keys if k in dataframe.columns]

        if missing:
            raise ValueError(f"Missing required column(s): {missing}")
        if conflict:
            raise ValueError(f"The following column(s) already exist and would be overwritten: {conflict}")

    def _reformat_prompt(self, dataframe, prompt_type:str = None):
        """
        Reformat the prompts in the dataframe to generate questions.
        """
        if prompt_type == "get_identifier":
            input_prompts = dataframe[self.input_key].tolist()
            system_prompts = self.prompts.get_identifier_system_prompt()
            prompts = [self.prompts.get_identifier_prompt(input_prompts) for input_prompts in input_prompts]
        elif prompt_type == "get_backward":
            input_prompts = dataframe[self.identifier_key].tolist()
            system_prompts = ""
            prompts = [self.prompts.get_backward_task_prompt(input_prompts) for input_prompts in input_prompts]
        elif prompt_type == "check_superset":
            new_identifiers = dataframe[self.new_identifier_key].tolist()
            relations = dataframe[self.relation_key].tolist()
            identifiers = dataframe[self.identifier_key].tolist()
            system_prompts = self.prompts.check_superset_system_prompt()
            prompts = [self.prompts.check_superset_prompt(new_id, relation, identifier) for new_id, relation, identifier in zip(new_identifiers, relations, identifiers)]
        elif prompt_type == "get_new_question":
            new_identifiers = dataframe[self.new_identifier_key].tolist()
            relations = dataframe[self.relation_key].tolist()
            identifiers = dataframe[self.identifier_key].tolist()
            system_prompts = self.prompts.get_question_system_prompt()
            prompts = [self.prompts.get_question_prompt(new_id, relation, identifier) for new_id, relation, identifier in zip(new_identifiers, relations, identifiers)]
        elif prompt_type == "llm_answer":
            questions = dataframe[self.input_key].tolist()
            system_prompts = ""
            prompts = [
                self.prompts.llm_answer_prompt(question) for question in questions
            ]
        elif prompt_type == "get_recall_score":
            golden_answers = dataframe["refined_answer"].tolist()
            llm_answers = dataframe["llm_answer"]
            system_prompts = self.prompts.recall_system_prompt()
            prompts = [
                self.prompts.recall_prompt(golden_answer, llm_answer) for golden_answer, llm_answer in zip(golden_answers, llm_answers)
            ]
        else:
            raise ValueError(f"Unknown prompt_type: {prompt_type}")
        return system_prompts, prompts
    
    def _clean_json_block(self, item: str) -> str:
        return item.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    def recall_score(self, dataframe):
        sys_prompts, user_prompts = self._reformat_prompt(dataframe, "get_recall_score")
        recall_scores = self.llm_serving.generate_from_input(user_prompts, sys_prompts)
        valid_scores = []
        for score_str in recall_scores:
            if score_str is not None:
                try:
                    score_dict = json.loads(score_str)
                    valid_scores.append(score_dict["answer_score"])
                except (json.JSONDecodeError, KeyError):
                    print(score_str)
                    valid_scores.append(0)
                    continue
        return valid_scores

    def run(
            self,
            storage: DataFlowStorage,
            input_key:str = "question",
            output_key:str = "depth_question",
            n_rounds:int = 2
            ):
        self.input_key, self.output_key = input_key, output_key
        dataframe = storage.read("dataframe")
        self._validate_dataframe(dataframe)

        if "identifier" not in dataframe.columns:
            sys_prompts, user_prompts = self._reformat_prompt(dataframe, "get_identifier")
            identifiers = self.llm_serving.generate_from_input(user_prompts, sys_prompts)
            dataframe["identifier"] = identifiers

        # step0: get identifier
        for round_id in range(1, n_rounds + 1):
            self.logger.info(f"=== Iteration Round {round_id} ===")

            # Use identifier from previous round
            self.identifier_key = "identifier" if round_id == 1 else f"new_identifier_{round_id - 1}"
            self.new_identifier_key, self.relation_key =f"new_identifier_{round_id}", f"relation_{round_id}"
            # Backward Step:
            # step1: Generate relation and superset
            sys_prompts, user_prompts = self._reformat_prompt(dataframe, "get_backward")
            backward_results = self.llm_serving.generate_from_input(user_prompts, sys_prompts)
            
            identifiers = []
            relations = []
            valid_indices = []

            for idx, result in enumerate(backward_results):
                try:
                    if isinstance(result, str):
                        result = json.loads(self._clean_json_block(result))

                    if isinstance(result, dict) and "identifier" in result and "relation" in result:
                        identifiers.append(result["identifier"])
                        relations.append(result["relation"])
                        valid_indices.append(idx)
                    else:
                        self.logger.warning(f"[Skipped]: Result at index {idx} is invalid: {result}")
                except Exception as e:
                    self.logger.warning(f"[Error]: Failed to parse backward result at index {idx}: {e}")
                    continue

            dataframe = dataframe.iloc[valid_indices].copy()
            dataframe[self.new_identifier_key] = identifiers
            dataframe[self.relation_key] = relations

            # step2: Check if superset is valid
            sys_prompts, user_prompts = self._reformat_prompt(dataframe, "check_superset")
            check_results = self.llm_serving.generate_from_input(user_prompts, sys_prompts)
            
            valid_indices = []
            for idx, result in enumerate(check_results):
                try:
                    if isinstance(result, str):
                        result = json.loads(self._clean_json_block(result))

                    if isinstance(result, dict) and "new_query" in result:
                        if result["new_query"] == "valid":
                            valid_indices.append(idx)
                    else:
                        self.logger.warning(f"[Skipped]: Result at index {idx} is invalid: {result}")
                except Exception as e:
                    self.logger.warning(f"[Error]: Failed to check superset result at index {idx}: {e}")
                    continue

            dataframe = dataframe.iloc[valid_indices].copy()

            # step3: Generate question based on superset and relation
            sys_prompts, user_prompts = self._reformat_prompt(dataframe, "get_new_question")
            check_results = self.llm_serving.generate_from_input(user_prompts, sys_prompts)

            new_queries = []
            valid_indices = []
            for idx, result in enumerate(check_results):
                try:
                    if isinstance(result, str):
                        result = json.loads(self._clean_json_block(result))

                    if isinstance(result, dict):
                        new_queries.append(result["new_query"])
                        valid_indices.append(idx)
                    else:
                        self.logger.warning(f"[Skipped]: Result at index {idx} is invalid: {result}")
                except Exception as e:
                    self.logger.warning(f"[Error]: Failed to check superset result at index {idx}: {e}")
                    continue

            dataframe = dataframe.iloc[valid_indices].copy()
            question_key = f"{output_key}_{round_id}"
            dataframe[question_key] = new_queries
            
            # Verify module
            sys_prompts, user_prompts = self._reformat_prompt(dataframe, "llm_answer")
            llm_answer_results = self.llm_serving.generate_from_input(user_prompts, sys_prompts)

            dataframe["llm_answer"] = llm_answer_results
            llm_score = self.recall_score(dataframe)
            dataframe["llm_score"] = llm_score
            dataframe = dataframe[dataframe["llm_score"] < 1].drop(columns=["llm_score"]).reset_index(drop=True)
            dataframe = dataframe.drop(columns="llm_answer")

        output_file = storage.write(dataframe)
        self.logger.info(f"Results saved to {output_file}")
        
        return [f"new_identifier_{i}" for i in range(1, n_rounds + 1)] + \
               [f"relation_{i}" for i in range(1, n_rounds + 1)] + \
               [f"{output_key}_{i}" for i in range(1, n_rounds + 1)]