import re
import os
import pandas as pd
import time
from tqdm import tqdm
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage
from dataflow.utils.text2sql.database_manager import DatabaseManager


@OPERATOR_REGISTRY.register()
class SQLExecutabilityFilter(OperatorABC):
    def __init__(self, database_manager: DatabaseManager):
        self.database_manager = database_manager
        self.logger = get_logger()

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
            input_sql_key: str = "SQL",
            input_db_id_key: str = "db_id"
        ):
        self.input_sql_key = input_sql_key
        self.input_db_id_key = input_db_id_key
        dataframe = storage.read("dataframe")
        self.check_column(dataframe)
        
        # Record input data count
        input_count = len(dataframe)
        
        # Phase 1: Database check
        db_id_need_to_check = dataframe[input_db_id_key].unique()
        for db_id in db_id_need_to_check:
            if not self.database_manager.database_exists(db_id):
                self.logger.warning(f"Database {db_id} not found in registry, please check the database folder")
                continue
        
        self.logger.info(f"Start to filter {len(dataframe)} SQLs")

        # Phase 2: SQL filtering
        phase1_mask = dataframe[input_sql_key].apply(self.filter_select_sql)
        phase1_df = dataframe[phase1_mask].copy()

        self.logger.info(f"Phase 1 completed: {len(phase1_df)}/{len(dataframe)} SQLs passed initial filter")

        if phase1_df.empty:
            output_file = storage.write(phase1_df)
            return []

        sql_triples = list(zip(phase1_df[input_db_id_key].tolist(),
                            phase1_df[input_sql_key].tolist()))

        # Phase 3: SQL execution
        exec_start = time.time()
        execution_results = self.database_manager.batch_explain_queries(sql_triples)

        phase2_mask = [r.success for r in execution_results]
        result_df = phase1_df[phase2_mask].copy()

        self.logger.info(f"Filter completed, remaining {len(result_df)} SQLs out of {len(phase1_df)} (phase1)")

        output_file = storage.write(result_df)
        output_count = len(result_df)
        
        return []
