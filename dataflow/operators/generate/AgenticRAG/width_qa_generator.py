from dataflow.prompts.agenticrag import WidthQAGeneratorPrompt
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow  import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC
from dataflow.core import LLMServingABC

import pandas as pd
import json

@OPERATOR_REGISTRY.register()
class WidthQAGenerator(OperatorABC):
    def __init__(self,
                 llm_serving: LLMServingABC = None
                 ):
        self.logger= get_logger()
        self.prompts = WidthQAGeneratorPrompt()
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
        required_keys = [self.input_question_key, self.input_answer_key, self.input_identifier_key]
        forbidden_keys = [self.output_question_key, "content_identifier", "qa_index", "index", "original_answer", "original_question", "state"]

        missing = [k for k in required_keys if k not in dataframe.columns]
        conflict = [k for k in forbidden_keys if k in dataframe.columns]

        if missing:
            raise ValueError(f"Missing required column(s): {missing}")
        if conflict:
            raise ValueError(f"The following column(s) already exist and would be overwritten: {conflict}")

    def _reformat_prompt(self, dataframe, prompt_type:str = None, input_batch: list[dict] = [None]):
        """
        Reformat the prompts in the dataframe to generate questions.
        """
        if prompt_type == "merge_prompt":
            system_prompts = self.prompts.merge_prompt_system_prompt()
            prompts = [
                self.prompts.merge_prompt_prompt([input_batch[i], input_batch[i + 1]])
                for i in range(len(input_batch) - 1)
            ]
        elif prompt_type == "check_origin":
            input_batch = []
            for idx, q, ori_q in zip(dataframe["index"], dataframe["question"], dataframe["original_question"]):
                input_batch.append({
                    "index": idx,
                    "complex_question": q,
                    "original_questions": ori_q if isinstance(ori_q, list) else [ori_q]
                })
            system_prompts = self.prompts.check_origin_system_prompt()
            prompts = [self.prompts.check_origin_prompt(input) for input in input_batch]
            return system_prompts, prompts
        elif prompt_type == "question_verify":
            input_batch = []
            for idx, q in zip(dataframe["index"], dataframe[self.output_question_key]):
                input_batch.append({
                    "index": idx,
                    "complex_question": q,
                })
            system_prompts = self.prompts.question_verify_system_prompt()
            prompts = [self.prompts.question_verify_prompt(input) for input in input_batch]
        elif prompt_type == "get_recall_score":
            golden_answers = dataframe["original_answer"].tolist()
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
            input_question_key:str = "question",
            input_identifier_key:str = "identifier",
            input_answer_key:str = "answer",
            output_question_key:str = "generated_width_task"
            ):
        self.input_question_key, self.input_identifier_key, self.input_answer_key, self.output_question_key = input_question_key, input_identifier_key,input_answer_key, output_question_key
        dataframe = storage.read("dataframe")
        self._validate_dataframe(dataframe)

        # step0: merge_prompt
        input_batch = []
        for i, (_, row) in enumerate(dataframe.iterrows()):
            input_batch.append({
                "index": i,
                "question": row[self.input_question_key],
                "content_identifier": row[self.input_identifier_key],
                "golden_answer": row[self.input_answer_key]
            })

        sys_prompts, user_prompts = self._reformat_prompt(dataframe, "merge_prompt", input_batch)
        merge_results = self.llm_serving.generate_from_input(user_prompts, sys_prompts)

        merged_rows = []
        for idx, result in enumerate(merge_results):
            try:
                if isinstance(result, str):
                    result = json.loads(self._clean_json_block(result))[0]

                if not isinstance(result, dict) or "question" not in result or "index" not in result:
                    self.logger.warning(f"[Skipped]: Invalid result at index {idx}: {result}")
                    continue

                indices = result["index"] if isinstance(result["index"], list) else [result["index"]]
                group_items = [input_batch[i] for i in indices]

                merged_rows.append({
                    "question": result["question"],
                    "content_identifier": result["content_identifier"],
                    "qa_index": indices,
                    "index": idx,
                    "original_answer": [item["golden_answer"] for item in group_items],
                    "original_question": [item["question"] for item in group_items],
                })

            except Exception as e:
                self.logger.warning(f"[Error]: Failed to parse merge result at index {idx}: {e}")
                continue

        dataframe = pd.DataFrame(merged_rows)
        
        # check queries
        # Step 1: Check if complex questions can be decomposed to original questions
        sys_prompts, user_prompts = self._reformat_prompt(dataframe, "check_origin")
        check_query_results = self.llm_serving.generate_from_input(user_prompts, sys_prompts)

        states = []
        complex_questions = []

        for idx, result in enumerate(check_query_results):
            try:
                if isinstance(result, str):
                    result = json.loads(self._clean_json_block(result))
                    if isinstance(result, list):
                        result = result[0]

                if isinstance(result, dict):
                    states.append(result.get("state", None))
                    complex_questions.append(result.get("complex_question", None))
                else:
                    self.logger.warning(f"[Skipped]: Invalid result at index {idx}: {result}")
                    states.append(None)
                    complex_questions.append(None)
            except Exception as e:
                self.logger.warning(f"[Error]: Failed to parse result at index {idx}: {e}")
                states.append(None)
                complex_questions.append(None)

        dataframe["state"] = states
        dataframe[self.output_question_key] = complex_questions
        dataframe = dataframe[dataframe["state"] == 1].copy()

        # Step 2: Verify if LLM can answer the complex questions
        sys_prompts, user_prompts = self._reformat_prompt(dataframe, "question_verify")
        question_verify_results = self.llm_serving.generate_from_input(user_prompts, sys_prompts)

        llm_answers = []
        for idx, result in enumerate(question_verify_results):
            try:
                if isinstance(result, str):
                    result = json.loads(self._clean_json_block(result))[0]

                if isinstance(result, dict):
                    llm_answers.append(result.get("llm_answer", None))

                else:
                    self.logger.warning(f"[Skipped]: Invalid result at index {idx}: {result}")
                    llm_answers.append(None)
            except Exception as e:
                self.logger.warning(f"[Error]: Failed to parse result at index {idx}: {e}")
                llm_answers.append(None)

        dataframe["llm_answer"] = llm_answers
        
        llm_score = self.recall_score(dataframe)
        dataframe["llm_score"] = llm_score
        dataframe = dataframe[dataframe["llm_score"] < 1].drop(columns=["llm_score"]).reset_index(drop=True)
        dataframe = dataframe.drop(columns="llm_answer")

        output_file = storage.write(dataframe)
        self.logger.info(f"Results saved to {output_file}")
        return