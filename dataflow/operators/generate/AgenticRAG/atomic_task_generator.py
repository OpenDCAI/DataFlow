from dataflow.prompts.agenticrag import AtomicTaskGeneratorPrompt
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow  import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC
from dataflow.core import LLMServingABC

import pandas as pd
import json
import os
import string
import re
from collections import Counter

@OPERATOR_REGISTRY.register()
class AtomicTaskGenerator(OperatorABC):
    def __init__(self,
                 llm_serving: LLMServingABC = None
                 ):
        self.logger= get_logger()
        self.prompts = AtomicTaskGeneratorPrompt()
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
        forbidden_keys = [self.output_refined_answer_key, self.output_answer_key, self.output_question_key, self.output_optional_answer_key, self.output_golden_doc_answer_key, self.output_f1_score_key]

        missing = [k for k in required_keys if k not in dataframe.columns]
        conflict = [k for k in forbidden_keys if k in dataframe.columns]

        if missing:
            raise ValueError(f"Missing required column(s): {missing}")
        if conflict:
            raise ValueError(f"The following column(s) already exist and would be overwritten: {conflict}")

    def _reformat_prompt(self, dataframe, prompt_type: str = None):
        """
        Reformat the prompts in the dataframe to generate LLM input.
        All input columns are expected to be strings.
        """
        if prompt_type == "get_identifier":
            input_prompts = dataframe[self.input_key].tolist()
            system_prompt = self.prompts.get_identifier_system_prompt()
            prompts = [self.prompts.get_identifier_prompt(p) for p in input_prompts]

        elif prompt_type == "get_conclusion":
            input_prompts = dataframe[self.input_key].tolist()
            system_prompt = self.prompts.initial_conclusion_system_prompt()
            prompts = [self.prompts.initial_conclusion_prompt(p) for p in input_prompts]

        elif prompt_type == "init_question":
            candidate_strs = dataframe["candidate_tasks_str"].tolist()
            system_prompt = self.prompts.initial_question_system_prompt()
            prompts = []

            for s in candidate_strs:
                try:
                    item = json.loads(self._clean_json_block(s))
                    prompts.append(self.prompts.initial_question_prompt(item['conclusion'], item['R']))
                except Exception as e:
                    print(f"[WARN] Failed to parse candidate_tasks_str: {e} | value: {s}")
                    prompts.append("")  # fallback to empty string or you can `continue`

        elif prompt_type == "clean_qa":
            questions = dataframe[self.output_question_key].tolist()
            answers = dataframe[self.output_answer_key].tolist()
            system_prompt = self.prompts.clean_qa_system_prompt()
            prompts = [
                self.prompts.clean_qa_prompt({"question": q, "original_answer": a})
                for q, a in zip(questions, answers)
            ]
        elif prompt_type == "llm_answer":
            questions = dataframe[self.output_question_key].tolist()
            system_prompt = ""
            prompts = [
                self.prompts.llm_answer_prompt(question) for question in questions
            ]
        elif prompt_type == "get_recall_score":
            golden_answers = dataframe[self.output_refined_answer_key].tolist()
            llm_answers = dataframe["llm_answer"]
            system_prompt = self.prompts.recall_system_prompt()
            prompts = [
                self.prompts.recall_prompt(golden_answer, llm_answer) for golden_answer, llm_answer in zip(golden_answers, llm_answers)
            ]
        elif prompt_type == "more_optional_answer":
            answers = dataframe[self.output_refined_answer_key].tolist()
            system_prompt = self.prompts.more_optional_answer_system_prompt()
            prompts = [
                self.prompts.more_optional_answer_prompt(answer) for answer in answers
            ]
        elif prompt_type == "golden_doc_answer":
            golden_docs = dataframe[self.input_key].tolist()
            questions = dataframe[self.output_question_key].tolist()
            system_prompt = ""
            prompts = [
                self.prompts.golden_doc_prompt(golden_doc, question) 
                for golden_doc, question in zip(golden_docs, questions)
            ]
        else:
            raise ValueError(f"Unknown prompt_type: {prompt_type}")

        return system_prompt, prompts

    
    def _clean_json_block(self, item: str) -> str:
        return item.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    def recall_score(self, dataframe):
        sys_prompts, user_prompts = self._reformat_prompt(dataframe, "get_recall_score")
        recall_scores = self.llm_serving.generate_from_input(user_prompts, sys_prompts)
        valid_scores = []
        for score_str in recall_scores:
            try:
                score_dict = json.loads(self._clean_json_block(score_str))
                valid_scores.append(score_dict["answer_score"])
            except Exception as e:
                print("recall score_str error:", score_str, "\nError:", e)
                valid_scores.append(0)
                continue
        # for score_str in recall_scores:
        #     if score_str is not None:
        #         try:
        #             score_dict = json.loads(self._clean_json_block(score_str))
        #             valid_scores.append(score_dict["answer_score"])
        #         except (json.JSONDecodeError, KeyError):
        #             print("recall score_str error:", score_str)
        #             valid_scores.append(0)
        #             continue
        return valid_scores

    def more_optional_answer(self, dataframe):
        original_answer = dataframe[self.output_refined_answer_key]
        system_prompt, user_prompts = self._reformat_prompt(dataframe, "more_optional_answer")
        optional_answers = self.llm_serving.generate_from_input(user_prompts, system_prompt)
        valid_answers = []
        for optional_answer in optional_answers:
            try:
                if isinstance(optional_answers, str):
                    optional_answer = json.loads(self._clean_json_block(optional_answer))
                    if orginal_answer not in optional_answer:
                        optional_answer.insert(0, original_answer)
                    valid_answers.append(optional_answer)
                else:
                    valid_answers.append(original_answer)
            except Exception as e:
                print(f"Error parsing optional answer: {optional_answer} | Error: {e}")
                valid_answers.append(original_answer)
        return valid_answers
        
        # self.logger.info(f"Response from optional answer: {optional_answers}")
        # self.logger.info(f"optional answer type: {type(optional_answers)}")

        # if original_answer not in optional_answers:
        #     optional_answers.insert(0, original_answer)

        return optional_answers

    def normalize_answer(self, s):
        def remove_articles(text):
            return re.sub(r"\b(a|an|the)\b", " ", text)

        def white_space_fix(text):
            return " ".join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))

    def get_f1_score(self, dataframe):

        f1_scores = []
        for idx, row in dataframe.iterrows():
            prediction = row[self.output_golden_doc_answer_key]
            ground_truths = row[self.output_optional_answer_key]

            final_metric = {"f1": 0, "precision": 0, "recall": 0}

            if ground_truths is None or prediction is None:
                f1_scores.append(final_metric['f1'])
                continue
                
            if isinstance(ground_truths, str):
                ground_truths = [ground_truths]

            for ground_truth in ground_truths:
                
                if ground_truth is None:
                    continue

                normalized_prediction = self.normalize_answer(prediction)
                normalized_ground_truth = self.normalize_answer(ground_truth)

                if normalized_prediction in ["yes", "no", "noanswer"] and normalized_prediction != normalized_ground_truth:
                    continue

                if normalized_ground_truth in ["yes", "no", "noanswer"] and normalized_prediction != normalized_ground_truth:
                    continue

                prediction_tokens = normalized_prediction.split()
                ground_truth_tokens = normalized_ground_truth.split()
                common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
                num_same = sum(common.values())
                if num_same == 0:
                    continue

                precision = 1.0 * num_same / len(prediction_tokens)
                recall = 1.0 * num_same / len(ground_truth_tokens)
                f1 = (2 * precision * recall) / (precision + recall)

                final_metric["precision"] = max(precision, final_metric["precision"])
                final_metric["recall"] = max(recall, final_metric["recall"])
                final_metric["f1"] = max(f1, final_metric["f1"])

            f1_scores.append(final_metric['f1'])

        return f1_scores
        
        
    def run(
        self,
        storage: DataFlowStorage,
        input_key: str = "prompts",
        output_question_key: str = "question",
        output_answer_key:str = "answer",
        output_refined_answer_key:str = "refined_answer",
        output_optional_answer_key: str = "optional_answer",
        output_golden_doc_answer_key: str = "golden_doc_answer",
        output_f1_score_key: str = "f1_score",
        data_num: int = 10, # Number of rows to process from the input dataframe 
        max_per_task: int = 10, # Limit the number of candidate tasks per input
        max_question: int = 10  # Limit the question of each doc
    ):
        self.input_key, self.output_question_key = input_key, output_question_key
        self.output_answer_key, self.output_refined_answer_key = output_answer_key, output_refined_answer_key
        self.output_optional_answer_key, self.output_golden_doc_answer_key, self.output_f1_score_key = output_optional_answer_key, output_golden_doc_answer_key, output_f1_score_key
        self.data_num, self.max_per_task, self.max_question = data_num, max_per_task, max_question

        dataframe = storage.read("dataframe").iloc[:self.data_num]
        self._validate_dataframe(dataframe)

        # === Step 0: Get identifier
        self.logger.info("Get identifier...")
        sys_prompts, user_prompts = self._reformat_prompt(dataframe, "get_identifier")
        identifiers = self.llm_serving.generate_from_input(user_prompts, sys_prompts)

        # === Step 1: Get conclusions
        self.logger.info("Get conclusions...")
        sys_prompts, user_prompts = self._reformat_prompt(dataframe, "get_conclusion")
        conclusions = self.llm_serving.generate_from_input(user_prompts, sys_prompts)

        # === Expand each conclusion into multiple candidate tasks (rows)
        expanded_rows = []
        for idx, (row, output_str, identifier) in enumerate(zip(dataframe.itertuples(index=False), conclusions, identifiers)):
            try:
                parsed = json.loads(self._clean_json_block(output_str))
                parsed = parsed[:self.max_per_task] if isinstance(parsed, list) else parsed
            except Exception as e:
                print(f"[WARN] JSON parse failed at idx={idx}: {e} | output: {output_str}")
                continue

            if not isinstance(parsed, list):
                continue

            for item in parsed:
                if isinstance(item, dict) and "conclusion" in item and "R" in item:
                    expanded_rows.append({
                        **row._asdict(),
                        "identifier": str(identifier),
                        "candidate_tasks_str": json.dumps(item, ensure_ascii=False)
                    })

        if not expanded_rows:
            self.logger.warning("No valid candidate tasks extracted.")
            return

        dataframe = pd.DataFrame(expanded_rows)

        # === Step 2: Generate questions based on conclusion + reasoning
        self.logger.info("Generate questions based on conclusion + reasoning...")
        sys_prompts, user_prompts = self._reformat_prompt(dataframe, "init_question")
        question_outputs = self.llm_serving.generate_from_input(user_prompts, sys_prompts)

        questions = []
        answers = []
        valid_rows = []

        for idx, (res, row) in enumerate(zip(question_outputs, dataframe.itertuples(index=False))):
            try:
                parsed = json.loads(self._clean_json_block(res))
            except Exception as e:
                print(f"[WARN] Failed to parse question JSON at idx={idx}: {e} | res: {res}")
                continue

            if isinstance(parsed, dict) and "Q" in parsed:
                question = parsed["Q"]
                try:
                    task = json.loads(self._clean_json_block(row.candidate_tasks_str))
                    answer = task.get("conclusion", "")
                except Exception:
                    answer = ""
                valid_rows.append(row._asdict())
                questions.append(str(question))
                answers.append(str(answer))

        if not valid_rows:
            self.logger.warning("No valid QA pairs generated.")
            return

        dataframe = pd.DataFrame(valid_rows)
        dataframe[self.output_question_key] = questions
        dataframe[self.output_answer_key] = answers

        # === Step 3: Clean QA
        self.logger.info("Clean QA...")
        sys_prompts, user_prompts = self._reformat_prompt(dataframe, "clean_qa")
        clean_outputs = self.llm_serving.generate_from_input(user_prompts, sys_prompts)

        final_answers = []

        for idx, res in enumerate(clean_outputs):
            try:
                parsed = json.loads(self._clean_json_block(res))
                final_answers.append(str(parsed.get("refined_answer", "")))
            except Exception as e:
                print(f"[WARN] Failed to parse cleaned QA at idx={idx}: {e} | res: {res}")
                final_answers.append("")

        dataframe[self.output_refined_answer_key] = final_answers

        # Verify module
        self.logger.info("llm reasoning verify...")
        sys_prompts, user_prompts = self._reformat_prompt(dataframe, "llm_answer")
        llm_answer_results = self.llm_serving.generate_from_input(user_prompts, sys_prompts)

        dataframe["llm_answer"] = llm_answer_results
        llm_score = self.recall_score(dataframe)
        dataframe["llm_score"] = llm_score
        dataframe = dataframe[dataframe["llm_score"] < 1].drop(columns=["llm_score"]).reset_index(drop=True)
        dataframe = dataframe.drop(columns="llm_answer")

        self.logger.info("generate more optional answer...")
        dataframe[self.output_optional_answer_key] = self.more_optional_answer(dataframe)

        self.logger.info("get golden doc answer...")
        sys_prompts, user_prompts = self._reformat_prompt(dataframe, "golden_doc_answer")
        llm_answer_results = self.llm_serving.generate_from_input(user_prompts, sys_prompts)
        dataframe[self.output_golden_doc_answer_key] = llm_answer_results

        self.logger.info("get f1 score...")
        dataframe[self.output_f1_score_key] = self.get_f1_score(dataframe)
        #dataframe = dataframe[(dataframe[self.output_f1_score_key] > 0.1) & (dataframe[self.output_f1_score_key] < 0.9)].reset_index(drop=True)
        
        # os.makedirs("debug_outputs", exist_ok=True)
        # with open("debug_outputs/df1.txt", "w", encoding="utf-8") as f:
        #     for row in dataframe.to_dict(orient="records"):
        #         for k, v in row.items():
        #             f.write(f"{k}: {v}\n")
        #         f.write("\n" + "-"*80 + "\n\n")

        dataframe = (
            dataframe.groupby("id", group_keys=False)
            .apply(lambda x: x.head(self.max_question))
            .reset_index(drop=True)
        )

        output_file = storage.write(dataframe)
        self.logger.info(f"Results saved to {output_file}")
        return "identifier", "candidate_tasks_str", self.output_question_key, self.output_answer_key, self.output_refined_answer_key, self.output_optional_answer_key, self.output_golden_doc_answer_key, self.output_f1_score_key