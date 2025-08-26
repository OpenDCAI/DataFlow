import re
import os
import pandas as pd
from tqdm import tqdm
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage
from dataflow.utils.text2sql.database_manager import DatabaseManager


@OPERATOR_REGISTRY.register()
class SQLExecutionFilter(OperatorABC):
    def __init__(self, database_manager: DatabaseManager):
        self.database_manager = database_manager
        self.logger = get_logger()

    @staticmethod
    def get_desc(lang):
        if lang == "zh":
            return (
                "对条目进行过滤，在数据库中执行SQL，筛选掉不可执行的条目。\n\n"
                "输入参数：\n"
                "- input_sql_key: 输入SQL列名\n"
                "- input_db_id_key: 输入数据库ID列名\n\n"
            )
        elif lang == "en":
            return (
                "This operator filters items based on whether the SQL can be executed in the database.\n\n"
                "Input parameters:\n"
                "- input_sql_key: The name of the input SQL column\n"
                "- input_db_id_key: The name of the input database ID column\n\n"
            )
        else:
            return "SQL execution filter for Text2SQL tasks."

    def filter_select_sql(self, sql):
        '''
            remain SELECT-type queries
        '''
        sql_wo_comments = re.sub(r'/\*.*?\*/', '', sql, flags=re.DOTALL)
        sql_wo_comments = re.sub(r'--.*', '', sql_wo_comments)
        sql_wo_comments = sql_wo_comments.strip()

        if sql_wo_comments.lower().startswith("select") or \
            sql_wo_comments.lower().startswith("with"):
            return True
        return False
    
    def check_column(self, dataframe):
        required_columns = [self.input_sql_key, self.input_db_id_key]
        missing_columns = [col for col in required_columns if col not in dataframe.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

    def run(self, storage: DataFlowStorage,
            input_sql_key: str = "sql",
            input_db_id_key: str = "db_id"
        ):
        self.input_sql_key = input_sql_key
        self.input_db_id_key = input_db_id_key
        dataframe = storage.read("dataframe")
        self.check_column(dataframe)
        
        db_id_need_to_check = dataframe[input_db_id_key].unique()
        for db_id in db_id_need_to_check:
            if not self.database_manager.database_exists(db_id):
                self.logger.warning(f"Database {db_id} not found in registry, please check the database folder")
                continue
        
        self.logger.info(f"Start to filter {len(dataframe)} SQLs")

        self.logger.info("Filtering SQLs using select component")
        phase1_passed_indices = []
        for idx, row in dataframe.iterrows():
            sql = row[input_sql_key]
            if self.filter_select_sql(sql):
                phase1_passed_indices.append(idx)

        self.logger.info(f"Phase 1 completed: {len(phase1_passed_indices)}/{len(dataframe)} SQLs passed initial filter")

        db_ids = dataframe[input_db_id_key]
        sql_list = dataframe[input_sql_key]
        execution_results = self.database_manager.batch_execute_queries(db_ids, sql_list)

        final_indices = []
        for idx, exec_result in enumerate(execution_results):
            if exec_result.success:
                final_indices.append(idx)

        self.logger.info(f"Filter completed, remaining {len(final_indices)} SQLs out of {len(dataframe)} original SQLs")

        result_df = dataframe.loc[final_indices]
        
        output_file = storage.write(result_df)
        return []