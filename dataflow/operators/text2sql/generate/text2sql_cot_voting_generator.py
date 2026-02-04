import re
import random
from collections import defaultdict
from typing import Dict, List, Optional, Any, Tuple

import pandas as pd

from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage
from dataflow.utils.text2sql.database_manager import DatabaseManager
from dataflow.utils.text2sql.base import QueryResult


def _parse_response(response: str) -> str:
    if not isinstance(response, str):
        return ""
    pattern = r"```sql\s*(.*?)\s*```"
    blocks = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)
    if blocks:
        sql = blocks[-1].strip()
        if sql:
            return sql
    for pat in [
        r"```SQL\s*(.*?)\s*```",
        r"SELECT.*?FROM",
    ]:
        matches = re.findall(pat, response, re.DOTALL | re.IGNORECASE)
        if matches:
            sql = matches[0].strip() if isinstance(matches[0], str) else matches[0]
            if sql:
                return sql
    return ""


def _result_to_signature(result: QueryResult):
    if not result.success or not result.data:
        return None
    columns = result.columns or []
    if not columns and result.data:
        columns = list(result.data[0].keys())
    return frozenset(
        tuple(row.get(c) for c in columns) for row in result.data
    )


def _tie_break(candidates: List[Dict[str, Any]], tie_breaker: str = "shortest_sql") -> Dict[str, Any]:
    if not candidates:
        raise ValueError("tie_break candidates empty")
    if tie_breaker == "random":
        return random.choice(candidates)
    return min(candidates, key=lambda x: len(x.get("sql") or ""))


def _vote_select(
    candidates: List[Dict[str, Any]],
    tie_breaker: str = "shortest_sql",
) -> Optional[Dict[str, Any]]:
    valid = [c for c in candidates if c.get("is_valid")]
    v = len(valid)
    if v == 0:
        return None
    if v == 1:
        return valid[0]

    buckets = defaultdict(list)
    for c in valid:
        buckets[c["signature"]].append(c)

    best_sig = max(buckets, key=lambda s: len(buckets[s]))
    best_bucket = buckets[best_sig]
    max_votes = len(best_bucket)

    if v == 2:
        if max_votes == 2:
            return random.choice(best_bucket)
        return _tie_break(valid, tie_breaker)

    if max_votes >= 2:
        return random.choice(best_bucket)
    return _tie_break(valid, tie_breaker)


@OPERATOR_REGISTRY.register()
class Text2SQLCoTVotingGenerator(OperatorABC):
    """
    After Text2SQLCoTGenerator, execute SQL for each cot_responses and vote by execution consistency to select final cot_reasoning.
    """

    def __init__(self, database_manager: DatabaseManager):
        self.database_manager = database_manager
        self.tie_breaker = "shortest_sql"
        self.logger = get_logger()

    @staticmethod
    def get_desc(lang):
        if lang == "zh":
            return (
                "After Text2SQLCoTGenerator, execute SQL for each cot_responses and vote by execution consistency to select final cot_reasoning.\n\n"
                "输入参数：\n"
                "- input_cot_responses_key: 输入候选 CoT 列表列名（默认 cot_responses）\n"
                "- input_db_id_key: 输入数据库 ID 列名\n\n"
                "输出参数：\n"
                "- output_cot_key: 输出最终 CoT 列名（默认 cot_reasoning）"
            )
        elif lang == "en":
            return (
                "Execute SQL from each cot_responses and vote by execution consistency to select final cot_reasoning. "
                "Input parameters:\n"
                "- input_cot_responses_key: The name of the input cot_responses column\n"
                "- input_db_id_key: The name of the input database ID column\n\n"
                "Output parameters:\n"
                "- output_cot_key: The name of the output cot_reasoning column"
            )
        return "CoT voting by execution-consistency."

    def check_column(self, dataframe: pd.DataFrame) -> None:
        required = [self.input_cot_responses_key, self.input_db_id_key]
        missing = [c for c in required if c not in dataframe.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def run(
        self,
        storage: DataFlowStorage,
        input_cot_responses_key: str = "cot_responses",
        input_db_id_key: str = "db_id",
        output_cot_key: str = "cot_reasoning",

    ):
        self.input_cot_responses_key = input_cot_responses_key
        self.input_db_id_key = input_db_id_key
        self.output_cot_key = output_cot_key

        self.logger.info("Starting CoT voting (execution-consistency)...")
        raw_df = storage.read("dataframe")
        self.check_column(raw_df)
        drop_cot_responses = True

        items = raw_df.to_dict("records")

        queries: List[Tuple[str, str]] = []
        mapping: List[Tuple[int, int, str, str]] = []

        for item_idx, item in enumerate(items):
            cot_responses = item.get(self.input_cot_responses_key, [])
            if not isinstance(cot_responses, list):
                continue
            db_id = item.get(self.input_db_id_key)
            if not db_id:
                continue
            for resp_idx, resp in enumerate(cot_responses):
                sql = _parse_response(resp)
                if sql:
                    queries.append((str(db_id).strip(), sql))
                    mapping.append((item_idx, resp_idx, sql, resp))

        if queries:
            query_results = self.database_manager.batch_execute_queries(queries)
        else:
            query_results = []
        item_candidates: Dict[int, List[Dict[str, Any]]] = defaultdict(list)

        for (item_idx, _resp_idx, sql, cot), result in zip(mapping, query_results):
            signature = _result_to_signature(result)
            is_valid = result.success
            item_candidates[item_idx].append({
                "cot": cot,
                "sql": sql,
                "signature": signature,
                "is_valid": is_valid,
            })

        output_items = []
        for item_idx, item in enumerate(items):
            candidates = item_candidates.get(item_idx, [])
            chosen = _vote_select(candidates, self.tie_breaker)

            if chosen is None:
                cot_responses = item.get(self.input_cot_responses_key, [])
                chosen_cot = cot_responses[0] if cot_responses else ""
            else:
                chosen_cot = chosen["cot"]

            out_item = {k: v for k, v in item.items() if k != self.input_cot_responses_key} if drop_cot_responses else dict(item)
            out_item[self.output_cot_key] = chosen_cot
            output_items.append(out_item)

        out_df = pd.DataFrame(output_items)
        output_file = storage.write(out_df)
        self.logger.info(f"CoT voting completed, saved to {output_file}, {len(output_items)} items")

        return [self.output_cot_key]
