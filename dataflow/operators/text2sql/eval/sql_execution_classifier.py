import pandas as pd
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.core import LLMServingABC
from dataflow.utils.storage import DataFlowStorage
from dataflow.utils.text2sql.database_manager import DatabaseManager


@OPERATOR_REGISTRY.register()
class SQLExecutionClassifier(OperatorABC):
    def __init__(self, 
                llm_serving: LLMServingABC, 
                database_manager: DatabaseManager, 
                difficulty_config: dict | None = None, 
                num_generations: int = 10,
                timeout: float = 5.0):
        self.llm_serving = llm_serving     
        self.database_manager = database_manager
        if difficulty_config is None:
            self.difficulty_config = {
                "num_generations": 10,
                'thresholds': [2, 5, 9],
                'labels': ['extra', 'hard', 'medium', 'easy']
            }
        else:
            self.difficulty_config = difficulty_config
        self.num_generations = num_generations
        self.timeout = timeout
        self.logger = get_logger()
        
        if self.num_generations <= self.difficulty_config["thresholds"][-1]:
            nearest_multiple = ((self.difficulty_config["thresholds"][-1] // 5) + 1) * 5
            self.logger.warning(f"num_generations is less than the last threshold, will be set to {nearest_multiple}")
            self.num_generations = nearest_multiple
        if len(self.difficulty_config['thresholds']) != len(self.difficulty_config['labels']) - 1:
            raise ValueError("Thresholds and labels configuration mismatch")

    @staticmethod
    def get_desc(lang):
        if lang == "zh":
            return (
                "让模型根据自然语言问题、数据库Schema和提示词，多次生成SQL，通过生成SQL的准确率，评估该问题对于模型的难度。\n\n"
                "输入参数：\n"
                "- input_db_id_key: 输入数据库ID列名\n"
                "- input_sql_key: 输入SQL列名\n"
                "- input_prompt_key: 输入prompt列名\n\n"
                "输出参数：\n"
                "- output_difficulty_key: 输出难度列名"
            )
        elif lang == "en":
            return (
                "This operator evaluates the difficulty of SQL generation for a question based on the accuracy of generated SQLs.\n\n"
                "Input parameters:\n"
                "- input_db_id_key: The name of the input database ID column\n"
                "- input_sql_key: The name of the input SQL column\n"
                "- input_prompt_key: The name of the input prompt column\n\n"
                "Output parameters:\n"
                "- output_difficulty_key: The name of the output difficulty column"
            )
        else:
            return "SQL execution difficulty evaluator for Text2SQL tasks."

    def check_column(self, dataframe):
        required_columns = [self.input_db_id_key, self.input_sql_key, self.input_prompt_key]
        missing_columns = [col for col in required_columns if col not in dataframe.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
    
    @staticmethod
    def parse_response(response, logger):
        pattern = r"```sql\s*(.*?)\s*```"
        
        sql_blocks = re.findall(pattern, response, re.DOTALL)

        if sql_blocks:
            last_sql = sql_blocks[-1].strip()
            return last_sql
        else:
            return ""

    @staticmethod
    def execute_model_batch(predicted_sqls_list, ground_truth_list, database_manager, db_ids, idxs, meta_time_out, logger):
        comparisons = []
        sql_mapping = {}
        
        comparison_idx = 0
        for i, (predicted_sqls, ground_truth, db_id, idx) in enumerate(zip(predicted_sqls_list, ground_truth_list, db_ids, idxs)):
            for j, predicted_sql in enumerate(predicted_sqls):
                comparison = {
                    'db_id': db_id,
                    'sql1': predicted_sql,
                    'sql2': ground_truth,
                    'timeout': meta_time_out,
                    'operation_id': f"compare_{comparison_idx}"
                }
                comparisons.append(comparison)
                sql_mapping[comparison_idx] = {
                    'original_idx': i,
                    'sql_idx': j,
                    'idx': idx,
                    'sql': predicted_sql
                }
                comparison_idx += 1
        
        try:
            batch_results = database_manager.batch_compare_sql(comparisons)
        except Exception as e:
            logger.error(f"Batch comparison failed: {e}")
            results = []
            for i, (predicted_sqls, _, _, idx) in enumerate(zip(predicted_sqls_list, ground_truth_list, db_ids, idxs)):
                result_data = []
                for predicted_sql in predicted_sqls:
                    result_data.append({'res': 0, 'sql': predicted_sql, 'error': 'batch_execution_failed'})
                results.append({"idx": idx, "cnt_true": -1, "results": result_data})
            return results
        
        results = {}
        for i, (predicted_sqls, _, _, idx) in enumerate(zip(predicted_sqls_list, ground_truth_list, db_ids, idxs)):
            results[i] = {
                "idx": idx,
                "cnt_true": 0,
                "results": [None] * len(predicted_sqls)
            }
        
        for batch_idx, batch_result in enumerate(batch_results):
            if batch_idx in sql_mapping:
                mapping = sql_mapping[batch_idx]
                original_idx = mapping['original_idx']
                sql_idx = mapping['sql_idx']
                
                if batch_result.success:
                    res = 1 if batch_result.data else 0
                    if res == 1:
                        results[original_idx]["cnt_true"] += 1
                    result_item = {'res': res, 'sql': mapping['sql']}
                else:
                    result_item = {'res': 0, 'sql': mapping['sql'], 'error': batch_result.error or 'unknown_error'}
                
                results[original_idx]["results"][sql_idx] = result_item
        
        return [results[i] for i in sorted(results.keys())]

    def run_sqls_parallel(self, datas, database_manager, num_cpus, meta_time_out):
        # pbar = tqdm(total=len(datas), desc="Executing SQLs")
        exec_result = []

        predicted_sqls_list = []
        ground_truth_list = []
        db_ids = []
        idxs = []
        
        for i, data_pair in enumerate(datas):
            predicted_sqls = data_pair[self.output_predicted_sqls_key]
            ground_truth = data_pair[self.input_sql_key]
            db_id = data_pair[self.input_db_id_key].replace('\n', '')
            db_id = re.sub(r'[^A-Za-z0-9_]', '', db_id)
            
            predicted_sqls_list.append(predicted_sqls)
            ground_truth_list.append(ground_truth)
            db_ids.append(db_id)
            idxs.append(i)
        
        # batch_size = max(1, len(datas) // num_cpus) if num_cpus > 1 else len(datas)
        batch_size = len(datas)
        
        def process_batch(batch_data):
            batch_predicted_sqls, batch_ground_truth, batch_db_ids, batch_idxs = batch_data
            return SQLExecutionClassifier.execute_model_batch(
                batch_predicted_sqls, batch_ground_truth, database_manager, 
                batch_db_ids, batch_idxs, self.timeout, self.logger
            )
        
        batches = []
        for i in range(0, len(datas), batch_size):
            end_idx = min(i + batch_size, len(datas))
            batch = (
                predicted_sqls_list[i:end_idx],
                ground_truth_list[i:end_idx],
                db_ids[i:end_idx],
                idxs[i:end_idx]
            )
            batches.append(batch)
        
        with ThreadPoolExecutor(max_workers=num_cpus) as executor:
            futures = [executor.submit(process_batch, batch) for batch in batches]
            
            for future in as_completed(futures):
                try:
                    batch_results = future.result()
                    exec_result.extend(batch_results)
                    # pbar.update(len(batch_results))
                except Exception as e:
                    self.logger.warning(f"Error in batch SQL execution: {e}")
                    # pbar.update(batch_size)

        # pbar.close()
        return exec_result

    def sort_results(self, list_of_dicts):
        return sorted(list_of_dicts, key=lambda x: x['idx'])
        
    def report_statistics(self, dataframe: pd.DataFrame):
        counts = dataframe[self.output_difficulty_key].value_counts()
        self.logger.info("SQL Difficulty Statistics")
        stats = [f"{difficulty.title()}: {counts.get(difficulty, 0)}" for difficulty in ['easy', 'medium', 'hard', 'extra']]
        self.logger.info(", ".join(stats))
                
    def classify_difficulty(self, score):
        if score == -1:
            return "gold error"
        thresholds = self.difficulty_config['thresholds']
        labels = self.difficulty_config['labels']
        
        for i, threshold in enumerate(thresholds):
            if score <= threshold:
                return labels[i]
        return labels[-1]

    def run(self, storage: DataFlowStorage,
            input_db_id_key: str = "db_id",
            input_sql_key: str = "SQL",
            input_prompt_key: str = "rl_prompt",
            output_difficulty_key: str = "sql_execution_difficulty"
        ):
        self.input_sql_key = input_sql_key
        self.input_prompt_key = input_prompt_key
        self.input_db_id_key = input_db_id_key
        self.output_difficulty_key = output_difficulty_key
        
        self.output_predicted_sqls_key = "_temp_predicted_sqls"
        self.output_cnt_true_key = "_temp_cnt_true"
        
        dataframe = storage.read("dataframe")
        self.check_column(dataframe)
        input_prompts = dataframe[self.input_prompt_key].tolist()
        
        self.logger.info(f"Processing {len(input_prompts)} questions, generating {self.num_generations} SQLs each...")
        prompts = [q for q in input_prompts for _ in range(self.num_generations)]
        responses = self.llm_serving.generate_from_input(prompts, system_prompt="You are a helpful assistant.")
        datas = dataframe.to_dict(orient='records')
        for i, data in enumerate(datas):
            start_idx = i * self.num_generations
            end_idx = start_idx + self.num_generations
            question_responses = responses[start_idx:end_idx]
            parsed_sqls = []
            for response in question_responses:
                if response:
                    parsed_sql = self.parse_response(response, self.logger)
                    parsed_sqls.append(parsed_sql)
                else:
                    parsed_sqls.append("")
            
            data[self.output_predicted_sqls_key] = parsed_sqls

        exec_result = self.run_sqls_parallel(datas, self.database_manager, 
                                            num_cpus=os.cpu_count(), 
                                            meta_time_out=5.0)
        exec_result = self.sort_results(exec_result)
        
        for execres in exec_result:
            if execres is not None:
                idx = execres["idx"]
                cnt_true = execres["cnt_true"]
                datas[idx][self.output_difficulty_key] = self.classify_difficulty(cnt_true)
                datas[idx][self.output_cnt_true_key] = cnt_true
        
        for data in datas:
            data.pop(self.output_predicted_sqls_key, None)
            data.pop(self.output_cnt_true_key, None)
        
        dataframe = pd.DataFrame(datas)
        
        self.report_statistics(dataframe)
        
        difficulty_counts = dataframe[self.output_difficulty_key].value_counts()
        self.logger.info("\nDifficulty Distribution:")
        for difficulty in ['easy', 'medium', 'hard', 'extra', 'gold error']:
            count = difficulty_counts.get(difficulty, 0)
            dataframe_len = len(dataframe) if dataframe is not None else 0
            if dataframe_len > 0:
                percentage = count / dataframe_len * 100
                self.logger.info(f"  {difficulty}: {count} ({percentage:.1f}%)")
            else:
                self.logger.info(f"  {difficulty}: {count} (0.0%)")
        
        output_file = storage.write(dataframe)
        self.logger.info(f"Difficulty classification results saved to {output_file}")

        return [self.output_difficulty_key]