from dataflow.generator.utils.APIGenerator_request import APIGenerator_request
from dataflow.generator.utils.LocalModelGenerator import LocalModelGenerator
from dataflow.generator.utils.APIGenerator_aisuite import APIGenerator_aisuite
import pandas as pd
import logging
import os
import re
import sqlite3
import sys
from func_timeout import func_timeout, FunctionTimedOut
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import multiprocessing as mp
from dataflow.utils.utils import get_logger
from dataflow.utils.registry import GENERATOR_REGISTRY

@GENERATOR_REGISTRY.register()
class Text2SQLDifficultyClassifier:
    def __init__(self, config: dict):
        '''
        Initialize the TextToSQLDifficultyClassifier with the provided configuration.
        '''
        self.config = config

        # Extract the configurations from the provided dictionary
        self.db_root_path = self.config.get("db_root_path")
        self.num_cpus = self.config.get("num_cpus", 1)
        self.meta_time_out = self.config.get("meta_time_out", 120.0)
        self.model_generator = self.__init_model__()
        self.easy_medium = self.config.get("easy_medium", 9) # easy:9-10
        self.medium_hard = self.config.get("medium_hard", 5) # medium:5-8
        self.hard_extra = self.config.get("hard_extra", 2) # hard:2-4
        

        # Input and output file paths and keys
        self.input_file = self.config.get("input_file")
        self.output_file = self.config.get("output_file")
        self.input_sql_key = self.config.get("input_sql_key", "SQL")
        self.input_prompt_key = self.config.get("input_prompt_key", "final_prompt")
        self.input_dbid_key = self.config.get("input_dbid_key", "db_id")
        self.output_key = self.config.get("output_key", "ex_difficulty")
        self.output_cnt_true_key = self.config.get("output_cnt_true_key", "cnt_true")
        self.output_predicted_sqls_key = self.config.get("output_predicted_sqls_key", "predicted_sqls")

        self.logger = get_logger()

        # Ensure required paths and keys are provided
        if not self.input_file or not self.output_file:
            self.logger.error("Both input_file and output_file must be specified in the config.")
            raise ValueError("Both input_file and output_file must be specified in the config.")
        if not self.db_root_path:
            self.logger.error("db_root_path must be specified in the config.")
            raise ValueError("db_root_path must be specified in the config.")


    def __init_model__(self):
        '''
        Initialize the model generator based on the configuration.
        '''
        generator_type = self.config.get("generator_type", "local").lower()
        if generator_type == "local":
            return LocalModelGenerator(self.config)
        elif generator_type == "aisuite":
            return APIGenerator_aisuite(self.config)
        elif generator_type == "request":
            return APIGenerator_request(self.config)
        else:
            raise ValueError(f"Invalid generator type: {generator_type}")
    
    @staticmethod
    def parse_response(response, logger):
        pattern = r"```sql\s*(.*?)\s*```"
        
        sql_blocks = re.findall(pattern, response, re.DOTALL)

        if sql_blocks:
            # Extract the last SQL query in the response text and remove extra whitespace characters
            last_sql = sql_blocks[-1].strip()
            return last_sql
        else:
            logger.warning(f"No SQL blocks found in {response}.")
            return response
        
    @staticmethod
    def execute_sql(sql, db_path):    
        conn = sqlite3.connect(db_path)
        # Connect to the database
        cursor = conn.cursor()
        cursor.execute(sql)
        result = cursor.fetchall()
        
        return result


    @staticmethod
    def execute_model(predicted_sqls, ground_truth, db_place, idx, meta_time_out, logger):
        results = []
        cnt_true = 0
        res = 0
        # logging.info(f"start execute idx {idx}")
        try:
            ground_truth_res = func_timeout(meta_time_out, Text2SQLDifficultyClassifier.execute_sql,
                                        args=(ground_truth, db_place))
            for predicted_sql in predicted_sqls:
                try:
                    # predicted_sql = Text2SQLDifficultyClassifier.parse_response(predicted_sql)
                    predicted_res = func_timeout(meta_time_out, Text2SQLDifficultyClassifier.execute_sql,
                                            args=(predicted_sql, db_place))
                    if set(predicted_res) == set(ground_truth_res):
                        res = 1
                    result = {'res': res, 'sql': predicted_sql}
                    # logging.info(f"res:{res}", flush=True)
                except KeyboardInterrupt:
                    # logging.info("KeyboardInterrupt")
                    sys.exit(0)
                except FunctionTimedOut:
                    result = [(f'timeout',)]
                    # logging.info("timeout when execute sqls of question {idx}")
                    res = 0
                except Exception as e:
                    # logging.info(f"error: {e} when execute sqls of question {idx}")
                    result = [(f'error',)]  # possibly len(query) > 512 or not executable
                    res = 0
                # logging.info(result)

                results.append(result)
                if res == 1:
                    cnt_true += 1

        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt")
            sys.exit(0)
        except FunctionTimedOut:
            result = [(f'timeout',)]
            logger.warning(f"timeout when execute gold sql of question {idx}")
            res = -1
            cnt_true = -1
        except Exception as e:
            logger.warning(f"error: {e} when execute gold sql of question {idx}")
            result = [(f'error',)]  # possibly len(query) > 512 or not executable
            res = -1
            cnt_true = -1

        # logging.info(f"idx:{idx}, cnt_true:{cnt_true}", flush=True)
        return {"idx": idx, "cnt_true": cnt_true}

    # def run_sqls_parallel(self, datas, db_root_path, num_cpus=1, meta_time_out=30.0):
    #     pbar = tqdm(total=len(datas))
    #     pbar.set_description("Executing SQLs")

    #     pool = mp.Pool(processes=num_cpus)
    #     exec_result = []
    #     def result_callback(result):
    #         pbar.update()
    #         exec_result.append(result)

    #     for i,data_pair in enumerate(datas):
    #         predicted_sqls = data_pair[self.output_predicted_sqls_key]
    #         ground_truth = data_pair[self.input_sql_key]
    #         db_id = data_pair[self.input_dbid_key].replace('\n', '')
    #         db_id = re.sub(r'[^A-Za-z0-9_]', '', db_id)
    #         db_place = os.path.join(db_root_path.rstrip('/'), db_id, f"{db_id}.sqlite")
    #         idx = i
    #         pool.apply_async(Text2SQLDifficultyClassifier.execute_model, args=(predicted_sqls, ground_truth, db_place, idx, meta_time_out), callback=result_callback)
    #     pool.close()
    #     pool.join()
    #     pbar.close()
    #     return exec_result
    
    def run_sqls_parallel(self, datas, db_root_path, num_cpus=1, meta_time_out=30.0):
        pbar = tqdm(total=len(datas), desc="Executing SQLs")
        exec_result = []

        def wrap_task(data_pair, idx):
            predicted_sqls = data_pair[self.output_predicted_sqls_key]
            ground_truth = data_pair[self.input_sql_key]
            db_id = data_pair[self.input_dbid_key].replace('\n', '')
            db_id = re.sub(r'[^A-Za-z0-9_]', '', db_id)
            db_place = os.path.join(db_root_path.rstrip('/'), db_id, f"{db_id}.sqlite")
            return Text2SQLDifficultyClassifier.execute_model(predicted_sqls, ground_truth, db_place, idx, meta_time_out, self.logger)

        with ThreadPoolExecutor(max_workers=num_cpus) as executor:
            futures = [
                executor.submit(wrap_task, data_pair, i)
                for i, data_pair in enumerate(datas)
            ]

            for future in as_completed(futures):
                try:
                    result = future.result()
                    exec_result.append(result)  # 顺序不保证
                except Exception as e:
                    self.logger.error(f"Error in SQL execution: {e}")
                    exec_result.append(None)
                pbar.update()

        pbar.close()
        return exec_result

    def sort_results(self, list_of_dicts):
        return sorted(list_of_dicts, key=lambda x: x['idx'])
    
    def get_difficulty(self, cnt_true):
        if cnt_true == -1:
            return "gold error"
        elif cnt_true >= self.easy_medium:
            return "easy"
        elif cnt_true >= self.medium_hard:
            return "medium"
        elif cnt_true >= self.hard_extra:
            return "hard"
        else:
            return "extra"
        
    def report_statistics(self, dataframe: pd.DataFrame):
        '''
        print the statistics of the SQL difficulty.
        '''
        counts = dataframe[self.output_key].value_counts()
        self.logger.info("SQL Difficulty Statistics")
        stats = [f"{difficulty.title()}: {counts.get(difficulty, 0)}" for difficulty in ['easy', 'medium', 'hard', 'extra']]
        self.logger.info(", ".join(stats))

    def process_single_question(self, question):
        try:
            # 注意：这里传入的是单个元素的列表，因为 generate_text_from_input 是 batch 接口
            result = self.model_generator.generate_text_from_input([question])
            return result[0] if result else ""
        except Exception as e:
            self.logger.error(f"Error processing question: {e}")
            return ""
        
    def run(self):
        '''
        Runs the TextSQLDifficultyClassifier, reading from the input file and saving results to output.
        '''
        # Read input file: only accept jsonl format
        dataframe = pd.read_json(self.input_file, lines=True)
        
        # Ensure the input and output keys are correctly set
        self._validate_dataframe(dataframe)

        # Extract prompts and repeat each question 10 times
        input_prompts = dataframe[self.input_prompt_key].tolist()
        repeated_questions = [q for q in input_prompts for _ in range(10)]

        # Generate model responses in batch
        responses = []
        # responses = self.model_generator.generate_text_from_input(repeated_questions)
        with ThreadPoolExecutor(max_workers=self.num_cpus) as executor:
            futures = {
                executor.submit(self.process_single_question, question): idx
                for idx, question in enumerate(tqdm(repeated_questions, desc="Submitting tasks"))
            }

            results_buffer = [None] * len(repeated_questions)

            for future in tqdm(as_completed(futures), total=len(repeated_questions), desc="Collecting responses"):
                idx = futures[future]
                try:
                    result = future.result()
                    results_buffer[idx] = result
                except Exception as e:
                    self.logger.error(f"Error retrieving future result: {e}")
                    results_buffer[idx] = ""

        responses = results_buffer
        
        # Group responses for each original input (10 responses per input)
        num_data = len(input_prompts)
        if len(responses) != num_data * 10:
            error_msg = f"Expected {num_data * 10} responses, but got {len(responses)}"
            self.logger.error(error_msg)
            raise AssertionError(error_msg)
        grouped_responses = [
            responses[i * 10:(i + 1) * 10] for i in range(num_data)
        ]
        grouped_parsed_responses = [
            [Text2SQLDifficultyClassifier.parse_response(sql, self.logger) for sql in group]
            for group in grouped_responses
        ]

        # Add results to dataframe and convert to list of dictionaries
        dataframe[self.output_predicted_sqls_key] = grouped_parsed_responses
        datas = dataframe.to_dict(orient='records')

        exec_result = self.run_sqls_parallel(datas, self.db_root_path, num_cpus=self.num_cpus, meta_time_out=self.meta_time_out)
        exec_result = self.sort_results(exec_result)

        if len(datas) != len(exec_result):
            error_msg = f"Length mismatch: exec_result has {len(exec_result)}, but datas has {len(datas)}"
            self.logger.error(error_msg)
            raise AssertionError(error_msg)

        for execres in exec_result:
            datas[execres["idx"]][self.output_cnt_true_key] = execres["cnt_true"]
            datas[execres["idx"]][self.output_key] = self.get_difficulty(execres["cnt_true"])

        if len(datas) != len(dataframe):
            error_msg = f"Length mismatch: original dataframe has {len(dataframe)}, but datas has {len(datas)}"
            self.logger.error(error_msg)
            raise AssertionError(error_msg)
            
        dataframe = pd.DataFrame(datas)

        # Ensure output directory exists
        output_dir = os.path.dirname(self.output_file)
        os.makedirs(output_dir, exist_ok=True)

        # Save DataFrame to JSON file
        self.report_statistics(dataframe)
        dataframe.to_json(self.output_file, orient="records", lines=True, force_ascii=False)

        
    def _validate_dataframe(self, dataframe: pd.DataFrame):
        '''
        Helper method to validate the input dataframe columns.
        '''
        # Check if the input sql key exists in the dataframe
        if self.input_sql_key not in dataframe.columns:
            self.logger.error(f"input_sql_key: {self.input_sql_key} not found in the dataframe.")
            raise ValueError(f"input_sql_key: {self.input_sql_key} not found in the dataframe.")
        
        # Check if the input prompt key exists in the dataframe
        if self.input_prompt_key not in dataframe.columns:
            self.logger.error(f"input_prompt_key: {self.input_prompt_key} not found in the dataframe.")
            raise ValueError(f"input_prompt_key: {self.input_prompt_key} not found in the dataframe.")
        
        # Check if the input dbid key exists in the dataframe
        if self.input_dbid_key not in dataframe.columns:
            self.logger.error(f"input_dbid_key: {self.input_dbid_key} not found in the dataframe.")
            raise ValueError(f"input_dbid_key: {self.input_dbid_key} not found in the dataframe.")
        
        # Check if the output key already exists in the dataframe
        if self.output_key in dataframe.columns:
            self.logger.warning(f"Found {self.output_key} in the dataframe, which would overwrite an existing column. Please use a different output_key.")
            raise ValueError(f"Found {self.output_key} in the dataframe, which would overwrite an existing column. Please use a different output_key.")

        # Check if the output cnt_true key already exists in the dataframe
        if self.output_cnt_true_key in dataframe.columns:
            self.logger.warning(f"Found {self.output_cnt_true_key} in the dataframe, which would overwrite an existing column. Please use a different output_key.")
            raise ValueError(f"Found {self.output_cnt_true_key} in the dataframe, which would overwrite an existing column. Please use a different output_key.")

        # Check if the output predicted_sqls key already exists in the dataframe
        if self.output_predicted_sqls_key in dataframe.columns:
            self.logger.warning(f"Found {self.output_predicted_sqls_key} in the dataframe, which would overwrite an existing column. Please use a different output_key.")
            raise ValueError(f"Found {self.output_predicted_sqls_key} in the dataframe, which would overwrite an existing column. Please use a different output_key.")
