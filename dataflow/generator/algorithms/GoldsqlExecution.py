from func_timeout import func_timeout, FunctionTimedOut
from tqdm import tqdm
import logging
import sqlite3
import sys
import re
import pandas as pd
import os
import multiprocessing as mp
from dataflow.utils.registry import GENERATOR_REGISTRY

@GENERATOR_REGISTRY.register()
class GoldsqlExecution:
    def __init__(self, config: dict):
        '''
        Initialize the GoldsqlExecution with the provided configuration.
        '''
        self.config = config

        # Extract the configurations from the provided dictionary
        self.db_root_path = self.config.get("db_root_path")
        self.num_cpus = self.config.get("num_cpus", 1)
        self.meta_time_out = self.config.get("meta_time_out", 120.0)

        # Input and output file paths and keys
        self.input_file = self.config.get("input_file")
        self.output_file = self.config.get("output_file")
        self.input_sql_key = self.config.get("input_sql_key", "SQL")
        self.input_dbid_key = self.config.get("input_dbid_key", "db_id")
        self.output_key = self.config.get("output_key", "is_correct")
        self.output_result_key = self.config.get("output_result_key", "exec_result")

        # Ensure required paths and keys are provided
        if not self.input_file or not self.output_file:
            raise ValueError("Both input_file and output_file must be specified in the config.")

    def execute_sql(self, sql, db_path):
        '''
        Execute the given SQL statement and return the results.
        '''
        # Connect to the database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(sql)
        result = cursor.fetchall()
        return result
    
    def execute_model(self, ground_truth, db_place, idx, meta_time_out):
        '''
        Execute SQL model with timeout and error handling.
        '''
        is_correct = True
        logging.info(f"start execute idx {idx}")
        try:
            results = func_timeout(meta_time_out, self.execute_sql,
                        args=(ground_truth, db_place))
            return {"idx": idx,"is_correct": is_correct, "results": results}
        except KeyboardInterrupt:
            logging.info("KeyboardInterrupt")
            sys.exit(0)
        except FunctionTimedOut:
            logging.info("timeout")
            result = (f'timeout')
            is_correct = False
            return {"idx": idx,"is_correct": is_correct, "results": result}
        except Exception as e:
            logging.info(f"error: {e}")
            result = (f'error:{e}')  # possibly len(query) > 512 or not executable
            is_correct = False
            return {"idx": idx,"is_correct": is_correct, "results": result}
        
    def run_sqls_parallel(self, datas, db_root_path, num_cpus, meta_time_out, exec_result=[]):
        '''
        Execute the given SQL statement and return the results.
        '''
        pbar = tqdm(total=len(datas))
        pbar.set_description("Executing SQLs")
        pool = mp.Pool(processes=num_cpus)

        def result_callback(result):
            pbar.update()
            exec_result.append(result)

        for i,data_pair in enumerate(datas):
            ground_truth = data_pair[self.input_sql_key]
            db_id = data_pair[self.input_dbid_key].replace('\n', '')
            db_id = re.sub(r'[^A-Za-z0-9_]', '', db_id)
            db_place = os.path.join(db_root_path.rstrip('/'), db_id, f"{db_id}.sqlite")
            pool.apply_async(self.execute_model, args=(ground_truth, db_place, i, meta_time_out), callback=result_callback)
        
        pool.close()
        pool.join()
        pbar.close()
        return sorted(exec_result, key=lambda x: x['idx'])

    def run(self):
        '''
        Main execution method, used to read the input file, execute SQL statements, and save the results to the output file.
        '''
        # Read input file: only accept jsonl format
        dataframe = pd.read_json(self.input_file, lines=True)
        
        # Ensure the input and output keys are correctly set
        self._validate_dataframe(dataframe)

        # Execute gold sqls from datas
        selected_columns = [self.input_sql_key, self.input_dbid_key]
        datas = dataframe[selected_columns].to_dict('records')
        exec_result = self.run_sqls_parallel(datas, self.db_root_path, self.num_cpus, self.meta_time_out, exec_result=[])
        dataframe[self.output_key] = [item["is_correct"] for item in exec_result]
        dataframe[self.output_result_key] = [item["results"] for item in exec_result]

        # Ensure output directory exists
        output_dir = os.path.dirname(self.output_file)
        os.makedirs(output_dir, exist_ok=True)

        # Save DataFrame to JSON file
        dataframe.to_json(self.output_file, orient="records", lines=True, force_ascii=False)

        
    def _validate_dataframe(self, dataframe: pd.DataFrame):
        '''
        Helper method to validate the input dataframe columns.
        '''
        # Check if the input sql key exists in the dataframe
        if self.input_sql_key not in dataframe.columns:
            raise ValueError(f"input_prompt_key: {self.input_sql_key} not found in the dataframe.")
        
        # Check if the input dbid key exists in the dataframe
        if self.input_dbid_key not in dataframe.columns:
            raise ValueError(f"input_dbid_key: {self.input_dbid_key} not found in the dataframe.")
        
        # Check if the output key already exists in the dataframe
        if self.output_key in dataframe.columns:
            raise ValueError(f"Found {self.output_key} in the dataframe, which would overwrite an existing column. Please use a different output_key.")

        # Check if the output result key already exists in the dataframe
        if self.output_result_key in dataframe.columns:
            raise ValueError(f"Found {self.output_result_key} in the dataframe, which would overwrite an existing column. Please use a different output_key.")
