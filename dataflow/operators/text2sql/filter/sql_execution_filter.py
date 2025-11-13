import json
import re
from typing import Dict, List, Tuple, Optional

import pandas as pd
from tqdm import tqdm

from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.core import OperatorABC, LLMServingABC
from dataflow.utils.storage import DataFlowStorage
from dataflow.utils.text2sql.database_manager import DatabaseManager
from dataflow.utils.text2sql.base import QueryResult


@OPERATOR_REGISTRY.register()
class SQLExecutionFilter(OperatorABC):
    LEMBED_PATTERN = re.compile(
        r"lembed\s*\(\s*(['\"]?)[^,]*?\1\s*,\s*(['\"])(.*?)\2\s*\)",
        re.IGNORECASE | re.DOTALL,
    )

    def __init__(self, database_manager: DatabaseManager, embedding_serving: Optional[LLMServingABC] = None):
        self.database_manager = database_manager
        self.logger = get_logger()
        self.embedding_serving = embedding_serving
        self._embedding_cache: Dict[str, List[float]] = {}

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

    def _get_embedding_from_local_serving(self, text: str) -> List[float]:
        if text in self._embedding_cache:
            return self._embedding_cache[text]

        if self.embedding_serving is None:
            raise RuntimeError("Embedding serving is not configured.")

        vectors = self.embedding_serving.generate_embedding_from_input([text])
        embedding = vectors[0]
        self._embedding_cache[text] = embedding
        return embedding

    def _preprocess_with_local_embedding(self, sql: str) -> str:
        def replacer(match: re.Match) -> str:
            text_to_embed = match.group(3)
            embedding = self._get_embedding_from_local_serving(text_to_embed)
            return "'" + json.dumps(embedding, ensure_ascii=False) + "'"

        return self.LEMBED_PATTERN.sub(replacer, sql)

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
        sql_triples = [(db_id, sql) for db_id, sql in zip(db_ids, sql_list)]

        processed_queries: List[Tuple[str, str]] = []
        index_mapping: List[int] = []
        preprocess_failures: Dict[int, QueryResult] = {}

        embedding_enabled = self.embedding_serving is not None

        if embedding_enabled:
            self.logger.info("Local embedding preprocessing enabled for lembed() SQLs.")

        for idx, (db_id, sql) in enumerate(sql_triples):
            processed_sql = sql
            if embedding_enabled and self.LEMBED_PATTERN.search(sql):
                try:
                    processed_sql = self._preprocess_with_local_embedding(sql)
                except Exception as exc:
                    self.logger.error(
                        "Failed to preprocess SQL for db '%s': %s",
                        db_id,
                        exc,
                        exc_info=True,
                    )
                    preprocess_failures[idx] = QueryResult(
                        success=False, error=f"Embedding preprocess failed: {exc}"
                    )
                    continue

            processed_queries.append((db_id, processed_sql))
            index_mapping.append(idx)

        execution_results: List[QueryResult] = [
            preprocess_failures.get(
                idx, QueryResult(success=False, error="Not executed")
            )
            for idx in range(len(sql_triples))
        ]

        if processed_queries:
            raw_results = self.database_manager.batch_execute_queries(processed_queries)
            for exec_idx, result in enumerate(raw_results):
                orig_idx = index_mapping[exec_idx]
                execution_results[orig_idx] = result

        final_indices = []
        for idx, exec_result in enumerate(execution_results):
            if exec_result.success:
                final_indices.append(idx)

        self.logger.info(f"Filter completed, remaining {len(final_indices)} SQLs out of {len(dataframe)} original SQLs")

        result_df = dataframe.loc[final_indices]
        
        output_file = storage.write(result_df)
        return []
