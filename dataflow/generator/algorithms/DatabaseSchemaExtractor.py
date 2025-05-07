import pandas as pd
import json
import os
import sqlite3
import logging
from tqdm import tqdm
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from dataflow.utils.registry import GENERATOR_REGISTRY
from dataflow.utils.utils import get_logger

@GENERATOR_REGISTRY.register()
class DatabaseSchemaExtractor:
    def __init__(self, config: Dict):
        self.config = config
        self.input_file = config['input_file']
        self.table_schema_path = config['table_schema_path']
        self.output_file = config['output_file']
        self.output_raw_schema_key = config['output_raw_schema_key']
        self.output_ddl_key = config['output_ddl_key']
        self.output_whole_format_schema_key = config['output_whole_format_schema_key']
        self.output_selected_format_schema_key = config['output_selected_format_schema_key']

        self.input_db_key = config['input_db_key']
        self.input_question_key = config['input_question_key']
        self.input_sql_key = config['input_sql_key']
        self.table_schema_file_db_key = config['table_schema_file_db_key']
        self.selected_schema_key = config['selected_schema_key']
        self.database_base_path = config['database_base_path']
        self.num_threads = config.get('num_threads', 5)
        self._schema_cache = {}

        self.logger = get_logger()


    def load_jsonl(self, file_path: str) -> List[Dict]:
        try:
            df = pd.read_json(file_path, lines=True)
            return df.to_dict('records')
        except Exception as e:
            self.logger.error(f"Error loading {file_path}: {e}")
            raise
    
    def save_jsonl(self, data: List[Dict], file_path: str) -> None:
        try:
            df = pd.DataFrame(data)
            df.to_json(file_path, orient='records', lines=True, force_ascii=False)
        except Exception as e:
            self.logger.error(f"Error saving to {file_path}: {e}")
            raise

    def collect_schema(self, db_id):
        if db_id in self._schema_cache:
            return self._schema_cache[db_id]

        # df = pd.read_json(self.table_schema_path, lines=True)
        table_schema = self.load_jsonl(self.table_schema_path)
        
        for schema in table_schema:
            if schema[self.table_schema_file_db_key] == db_id:
                self._schema_cache[db_id] = schema
                return schema
        
        return None

    def extract_schema(self, db_info: Dict, db_conn: sqlite3.Connection) -> Dict:
        schema = {
            'tables': {},
            'foreign_keys': [],
            'primary_keys': []
        }
        
        table_names = db_info["table_names_original"]
        cursor = db_conn.cursor()
        
        try:
            for i in range(len(db_info["column_names_original"])):
                if db_info["column_names_original"][i][0] == -1:
                    continue
                    
                table_idx = db_info["column_names_original"][i][0]
                table_name = table_names[table_idx]
                # col_name = db_info["column_names_original"][i][1].replace(" ", "_")
                col_name = db_info["column_names_original"][i][1]
                col_type = db_info["column_types"][i]
                
                if table_name not in schema['tables']:
                    schema['tables'][table_name] = {
                        'columns': {},
                        'primary_keys': []
                    }
                
                schema['tables'][table_name]['columns'][col_name] = {
                    'type': col_type,
                    'examples': []
                }
                
                try:
                    sql_query = f'SELECT "{col_name}" FROM "{table_name}" LIMIT 2'
                    cursor.execute(sql_query)
                    examples = [str(row[0]) for row in cursor.fetchall() if row[0] is not None]
                    schema['tables'][table_name]['columns'][col_name]['examples'] = examples
                except sqlite3.Error as e:
                    self.logger.warning(f"Unable to access examples for {table_name}.{col_name}: {e}")
            
            for pk in db_info["primary_keys"]:
                if isinstance(pk, list):  
                    table_idxs = {db_info["column_names_original"][col_idx][0] for col_idx in pk}
                    if len(table_idxs) != 1:
                        continue 
                    
                    table_idx = table_idxs.pop()
                    if table_idx == -1:
                        continue
                        
                    table_name = table_names[table_idx]
                    col_names = [
                        db_info["column_names_original"][col_idx][1]
                        for col_idx in pk
                    ]
                    
                    if table_name in schema['tables']:
                        schema['tables'][table_name]['primary_keys'].extend(col_names)
                        schema['primary_keys'].append({
                            'table': table_name,
                            'columns': col_names
                        })
                else:
                    table_idx = db_info["column_names_original"][pk][0]
                    if table_idx == -1:
                        continue
                        
                    table_name = table_names[table_idx]
                    col_name = db_info["column_names_original"][pk][1]
                    
                    if table_name in schema['tables']:
                        schema['tables'][table_name]['primary_keys'].append(col_name)
                        schema['primary_keys'].append({
                            'table': table_name,
                            'column': col_name
                        })
            
            for fk in db_info["foreign_keys"]:
                src_col_idx, ref_col_idx = fk
                
                src_table_idx = db_info["column_names_original"][src_col_idx][0]
                ref_table_idx = db_info["column_names_original"][ref_col_idx][0]
                
                if src_table_idx == -1 or ref_table_idx == -1:
                    continue
                    
                src_table = table_names[src_table_idx]
                src_col = db_info["column_names_original"][src_col_idx][1]
                ref_table = table_names[ref_table_idx]
                ref_col = db_info["column_names_original"][ref_col_idx][1]
                
                schema['foreign_keys'].append({
                    'source_table': src_table,
                    'source_column': src_col,
                    'referenced_table': ref_table,
                    'referenced_column': ref_col
                })
                
        finally:
            cursor.close()
            
        return schema

    def generate_ddl_from_schema(self, schema: Dict) -> str:
        ddl_statements = []
        
        for table_name, table_info in schema['tables'].items():
            columns_ddl = []
            
            for col_name, col_info in table_info['columns'].items():
                sql_type = {
                    "number": "INTEGER",
                    "text": "TEXT",
                    "date": "DATE",
                    "time": "TIME",
                    "datetime": "DATETIME"
                }.get(col_info['type'].lower(), "TEXT")
                
                columns_ddl.append(f"    {col_name} {sql_type}")
            
            if table_info['primary_keys']:
                pk_columns = ", ".join(table_info['primary_keys'])
                columns_ddl.append(f"    PRIMARY KEY ({pk_columns})")
            
            for fk in schema['foreign_keys']:
                if fk['source_table'] == table_name:
                    columns_ddl.append(
                        f"    FOREIGN KEY ({fk['source_column']}) "
                        f"REFERENCES {fk['referenced_table']}({fk['referenced_column']})"
                    )
            
            create_table_sql = (
                f"CREATE TABLE {table_name} (\n" +
                ",\n".join(columns_ddl) +
                "\n);"
            )
            ddl_statements.append(create_table_sql)
        
        return "\n\n".join(ddl_statements)

    def generate_formatted_schema(self, schema: Dict) -> str:
        formatted = []
        
        for table_name, table_info in schema['tables'].items():
            formatted.append(f"## Table: {table_name}")
            
            if table_info['primary_keys']:
                formatted.append(f"Primary Key: {', '.join(table_info['primary_keys'])}")
            
            formatted.append("Column Information:")
            for col_name, col_info in table_info['columns'].items():
                examples = ", ".join(col_info['examples']) if col_info['examples'] else ""
                formatted.append(
                    f"- {col_name} ({col_info['type']}) "
                    f"Example: {examples}"
                )
            
            table_fks = [
                fk for fk in schema['foreign_keys'] 
                if fk['source_table'] == table_name
            ]
            if table_fks:
                formatted.append("Foreign Key:")
                for fk in table_fks:
                    formatted.append(
                        f"- {fk['source_column']} → "
                        f"{fk['referenced_table']}.{fk['referenced_column']}"
                    )
            
            formatted.append("") 
        
        return "\n".join(formatted)
    
    def generate_selected_format_schema(self, selected_schema: Dict) -> str:
        selected_format_schema = ""
        for table in selected_schema:
            table_name = table["table_name"]
            columns = ",".join(table["columns"])
            selected_format_schema += f"Table {table_name}, columns = [{columns}]\n"

        return selected_format_schema
    
    def _process_item(self, item: Dict) -> Dict:
        db_id = item[self.input_db_key]
        # print(f"Processing database {db_id}")
        db_info = self.collect_schema(db_id)

        # logging.warning(db_info)

        if not db_info:
            self.logger.warning(f"No schema found for database {db_id}")
            return item
        
        db_path = os.path.join(self.database_base_path, db_id, f"{db_id}.sqlite")
        # logging.warning(f"Connecting to database at {db_path}")
        try:
            with sqlite3.connect(db_path) as db_conn:
                # logging.warning(f'start execute idx {item["question_id"]}')
                schema = self.extract_schema(db_info, db_conn)
                # logging.warning(1)
                item[self.output_raw_schema_key] = schema
                item[self.output_ddl_key] = self.generate_ddl_from_schema(schema)
                # logging.warning(2)
                item[self.output_whole_format_schema_key] = self.generate_formatted_schema(schema)
                item[self.output_selected_format_schema_key] = self.generate_selected_format_schema(item[self.selected_schema_key])
                # logging.warning(3)

        except sqlite3.Error as e:
            self.logger.error(f"Database error for {db_id}: {e}")

        # logging.warning(f"Schema extraction completed for {db_id}")
        
        return item

    def run(self) -> None:
        self.logger.info("Starting DatabaseSchemaExtractor")
        try:
            items = self.load_jsonl(self.input_file)
            self._schema_cache = {}
            
            question_id_to_index = {item['question_id']: idx for idx, item in enumerate(items)}
            
            with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
                futures = {
                    executor.submit(self._process_item, item): item['question_id']
                    for item in items
                }
                
                with tqdm(total=len(items), desc="Processing") as pbar:
                    for future in as_completed(futures):
                        question_id = futures[future]
                        try:
                            items[question_id_to_index[question_id]] = future.result()
                        except Exception as e:
                            self.logger.error(f"Error processing question_id={question_id}: {e}")
                        finally:
                            pbar.update(1)
            
            self.save_jsonl(items, self.output_file)
            
        except Exception as e:
            self.logger.error(f"Fatal error: {e}")
            raise