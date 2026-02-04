from typing import Dict, Optional, Tuple, List, Union, Any
import pandas as pd
import re
import random
from collections import defaultdict
from dataflow.prompts.text2sql import Text2SQLCotGeneratorPrompt
from dataflow.core.prompt import DIYPromptABC, prompt_restrict
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.core import OperatorABC, LLMServingABC
from dataflow.utils.storage import DataFlowStorage
from dataflow.utils.text2sql.database_manager import DatabaseManager


@prompt_restrict(Text2SQLCotGeneratorPrompt)
@OPERATOR_REGISTRY.register()
class Text2SQLCoTGenerator(OperatorABC):
    def __init__(
        self,
        llm_serving: LLMServingABC,
        database_manager: DatabaseManager,
        prompt_template: Union[Text2SQLCotGeneratorPrompt, DIYPromptABC] = None,
        sampling_num: int = 3,
    ):
        self.llm_serving = llm_serving
        self.database_manager = database_manager
        self.prompt_template = prompt_template or Text2SQLCotGeneratorPrompt()
        self.logger = get_logger()

        self.sampling_num = int(sampling_num)
        if self.sampling_num < 1:
            raise ValueError("sampling_num must be >= 1")

    @staticmethod
    def get_desc(lang):
        if lang == "zh":
            return (
                "对于每个条目，生成从自然语言问题和数据库Schema到SQL的CoT长链路推理过程。生成sampling_num条推理轨迹，但是不做验证"
                "输入参数：\n"
                "- input_sql_key: 输入SQL列名\n"
                "- input_question_key: 输入问题列名\n"
                "- input_db_id_key: 输入数据库ID列名\n\n"
                "输出参数：\n"
                "- output_cot_key: 输出CoT列名"
            )
        elif lang == "en":
            return (
                "For each item, generate a CoT long chain of reasoning from natural language question and database Schema to SQL. Generate sampling_num reasoning trajectories, but do not verify."
                "Input parameters:\n"
                "- input_sql_key: The name of the input SQL column\n"
                "- input_question_key: The name of the input question column\n"
                "- input_db_id_key: The name of the input database ID column\n\n"
                "Output parameters:\n"
                "- output_cot_key: The name of the output CoT column"
            )
        else:
            return "CoT generator for Text2SQL tasks."

    def check_column(self, dataframe):
        required_columns = [self.input_sql_key, self.input_db_id_key, self.input_question_key]
        missing = [c for c in required_columns if c not in dataframe.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def extract_sql(self, response: str) -> str:
        if not isinstance(response, str):
            self.logger.warning(f"Invalid response type: {type(response)}, expected str. Response: {response}")
            return ""
        pattern = r"```sql\s*(.*?)\s*```"
        blocks = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)
        if blocks:
            return blocks[-1].strip()
        return ""

    def _build_prompt(self, item: Dict[str, Any]) -> str:
        db_id = item.get(self.input_db_id_key)
        question = item.get(self.input_question_key)
        gold_sql = item.get(self.input_sql_key)
        evidence = item.get(self.input_evidence_key, "")

        create_statements, _ = self.database_manager.get_create_statements_and_insert_statements(db_id)
        schema_str = "\n\n".join(create_statements)
        return self.prompt_template.build_prompt(schema_str, question, gold_sql, evidence)

    def _process_items(self, items_with_index: List[Tuple[int, Dict]]) -> List[Dict]:
        prompts = []
        mapping = []
        for orig_idx, item in items_with_index:
            prompt = self._build_prompt(item)
            for _ in range(self.sampling_num):
                prompts.append(prompt)
                mapping.append((orig_idx, item))
        responses = self.llm_serving.generate_from_input(prompts, "")
        grouped: Dict[int, List[str]] = defaultdict(list)
        for (orig_idx, item), resp in zip(mapping, responses):
            grouped[orig_idx].append(resp)

        results = []
        for orig_idx, item in items_with_index:
            results.append({
                **item,
                "__orig_index": orig_idx,
                "cot_responses": grouped.get(orig_idx, []),
            })
        return results

    def run(
        self,
        storage: DataFlowStorage,
        input_sql_key: str = "SQL",
        input_question_key: str = "question",
        input_db_id_key: str = "db_id",
        input_evidence_key: str = "evidence",
        output_cot_key: str = "cot_reasoning",
    ):
        self.input_question_key = input_question_key
        self.input_sql_key = input_sql_key
        self.input_db_id_key = input_db_id_key
        self.input_evidence_key = input_evidence_key
        self.output_cot_key = output_cot_key

        self.logger.info("Starting CoT generation (scheme C voting)...")
        raw_df = storage.read("dataframe")
        self.check_column(raw_df)
        
        items = raw_df.to_dict("records")
        items_with_index = list(enumerate(items))

        results = self._process_items(items_with_index)

        if not results:
            self.logger.warning("No CoT results generated.")
            return []

        out_df = pd.DataFrame(results).sort_values("__orig_index").drop(columns=["__orig_index"])
        output_file = storage.write(out_df)
        output_count = len(out_df)
        self.logger.info(f"CoT responses generation completed, saved to {output_file}")
        self.logger.info(f"Processed {output_count} items, original {len(items)} items")

        return ["cot_responses"]
