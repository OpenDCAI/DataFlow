from typing import Dict, Any, Optional, Tuple
from ..base import DatabaseConnectorABC, DatabaseInfo, QueryResult
import sqlite3
import glob
import os
import time
import re
import datetime
import decimal

# ============== SQLite Connector ==============
class SQLiteConnector(DatabaseConnectorABC):

    def connect(self, connection_info: Dict) -> sqlite3.Connection:
        db_path = connection_info['path']
        conn = sqlite3.connect(db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA query_only = ON")
        return conn
    
    def execute_query(self, connection: sqlite3.Connection, sql: str, params: Optional[Tuple] = None) -> QueryResult:
        start_time = time.time()
        cursor = None
        try:
            cursor = connection.cursor()
            if params:
                # Ensure params is a tuple and all values are properly formatted
                if not isinstance(params, (tuple, list)):
                    params = (params,)
                # Convert all params to strings and handle None values
                if params and not isinstance(params, (tuple, list)):
                    params = (params,)
                cursor.execute(sql, params or ())
            else:
                cursor.execute(sql)
            
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            data = [dict(zip(columns, row)) for row in rows] if columns else []
            
            return QueryResult(
                success=True,
                data=data,
                columns=columns,
                row_count=len(data)
            )
        except Exception as e:
            self.logger.debug(f"Query execution failed (expected during filtering): {e}")
            self.logger.debug(f"SQL: {sql}")
            self.logger.debug(f"Params: {params}")
            return QueryResult(
                success=False,
                error=str(e)
            )
        finally:
            if cursor:
                cursor.close()

    def explain_query(self, connection: sqlite3.Connection, sql: str, params: Optional[Tuple] = None) -> QueryResult:
        cursor = None
        try:
            cursor = connection.cursor()
            stripped = sql.lstrip()
            if not re.match(r'(?i)^explain(\s+query\s+plan)?\b', stripped):
                sql = f"EXPLAIN QUERY PLAN {sql}"
            if params:
                if not isinstance(params, (tuple, list)):
                    params = (params,)
                cursor.execute(sql, tuple(params))
            else:
                cursor.execute(sql)
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            data = [dict(zip(columns, row)) for row in rows] if columns else []
            return QueryResult(
                success=True,
                data=data,
                columns=columns,
                row_count=len(data)
            )
        except Exception as e:
            self.logger.debug(f"Query explain failed (expected during filtering): {e}")
            self.logger.debug(f"SQL: {sql}")
            self.logger.debug(f"Params: {params}")
            return QueryResult(
                success=False,
                error=str(e)
            )
        finally:
            if cursor:
                cursor.close()
    
    def get_schema_info(self, connection: sqlite3.Connection, db_id: Optional[str] = None) -> Dict[str, Any]:
        """Get complete schema information with formatted M-Schema"""
        schema = {'tables': {}}

        result = self.execute_query(
            connection, 
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
        
        if not result.success:
            return schema
        
        for row in result.data:
            table_name = row['name']
            table_info = self._get_table_info(connection, table_name)
            schema['tables'][table_name] = table_info

        if not db_id:
            db_id = self._get_db_id_from_connection(connection)
        schema['db_details'] = self._get_db_details(schema, db_id=db_id)
        return schema

    def _get_table_info(self, connection: sqlite3.Connection, table_name: str) -> Dict[str, Any]:
        """Get detailed table information"""
        table_info = {
            'columns': {},
            'primary_keys': [],
            'foreign_keys': [],
            'sample_data': [],
            'create_statement': None,
            'insert_statement': []
        }
        
        result = self.execute_query(connection, f"PRAGMA table_info({table_name})")
        if result.success:
            for col in result.data:
                col_name = col['name']
                default_value = col['dflt_value']
                if default_value is not None and isinstance(default_value, str):
                    if default_value.upper() in ('CURRENT_TIMESTAMP', 'NULL'):
                        default_value = default_value.upper()
                    else:
                        default_value = f"'{default_value}'"
                
                table_info['columns'][col_name] = {
                    'type': col['type'],
                    'nullable': not col['notnull'],
                    'default': default_value,
                    'primary_key': bool(col['pk']),
                    'comment': '',
                    'examples': []
                }
                if col['pk']:
                    table_info['primary_keys'].append(col_name)
        
        result = self.execute_query(connection, f"PRAGMA foreign_key_list({table_name})")
        if result.success:
            for fk in result.data:
                table_info['foreign_keys'].append({
                    'column': fk['from'],
                    'referenced_table': fk['table'],
                    'referenced_column': fk['to']
                })
        
        result = self.execute_query(connection, f"SELECT * FROM `{table_name}` LIMIT 2")
        if result.success and result.data:
            table_info['sample_data'] = result.data
            column_names = list(result.data[0].keys())
            
            for row in result.data:
                values = []
                for value in row.values():
                    if value is None:
                        values.append('NULL')
                    elif isinstance(value, (int, float)):
                        values.append(str(value))
                    else:
                        escaped = str(value).replace("'", "''")
                        values.append(f"'{escaped}'")
                
                table_info['insert_statement'].append(
                    f"INSERT INTO `{table_name}` ({', '.join(column_names)}) VALUES ({', '.join(values)});"
                )

        result = self.execute_query(connection, 
            "SELECT sql FROM sqlite_master WHERE type='table' AND name = ?", 
            (table_name,))
        if result.success and result.data:
            table_info['create_statement'] = result.data[0]['sql']

        for column_name in table_info['columns'].keys():
            examples = self._fetch_distinct_values(connection, table_name, column_name, max_num=5)
            table_info['columns'][column_name]['examples'] = self._examples_to_str(examples)

        return table_info

    def _get_db_details(self, schema: Dict[str, Any], db_id: str) -> str:
        """Generate formatted M-Schema statements from schema information"""
        output = []
        output.append(f"【DB_ID】 {db_id}")
        output.append("【Schema】")

        for table_name, table_info in schema['tables'].items():
            table_comment = table_info.get('comment', '')
            table_comment = '' if table_comment is None else table_comment.strip()

            if table_comment and table_comment != 'None':
                output.append(f"# Table: {table_name}, {table_comment}")
            else:
                output.append(f"# Table: {table_name}")

            field_lines = []
            for field_name, field_info in table_info['columns'].items():
                raw_type = self._get_field_type(field_info.get('type', ''), simple_mode=True)
                field_line = f"({field_name}:{raw_type.upper()}"

                field_comment = field_info.get('comment', '')
                if field_comment is not None and field_comment != '':
                    field_line += f", {field_comment.strip()}"

                if field_info.get('primary_key', False):
                    field_line += ", Primary Key"

                examples = field_info.get('examples', [])
                examples = [s for s in examples if s is not None]
                examples = self._examples_to_str(examples)
                example_num = 3

                if len(examples) > example_num:
                    examples = examples[:example_num]

                if raw_type in ['DATE', 'TIME', 'DATETIME', 'TIMESTAMP']:
                    examples = [examples[0]] if examples else []
                elif len(examples) > 0 and max(len(s) for s in examples) > 20:
                    if max(len(s) for s in examples) > 50:
                        examples = []
                    else:
                        examples = [examples[0]]

                if len(examples) > 0:
                    example_str = ', '.join([str(example) for example in examples])
                    field_line += f", Examples: [{example_str}]"

                field_line += ")"
                field_lines.append(field_line)

            output.append('[')
            output.append(',\n'.join(field_lines))
            output.append(']')

        foreign_keys = []
        for table_name, table_info in schema['tables'].items():
            for fk in table_info.get('foreign_keys', []):
                foreign_keys.append(
                    f"{table_name}.{fk['column']}={fk['referenced_table']}.{fk['referenced_column']}"
                )

        if foreign_keys:
            output.append("【Foreign keys】")
            output.extend(foreign_keys)

        return '\n'.join(output)

    def _get_db_id_from_connection(self, connection: sqlite3.Connection) -> str:
        cursor = None
        try:
            cursor = connection.execute("PRAGMA database_list")
            rows = cursor.fetchall()
        except Exception:
            rows = []
        finally:
            if cursor:
                cursor.close()

        for row in rows:
            try:
                name = row["name"]
                file_path = row["file"]
            except Exception:
                name = row[1] if len(row) > 1 else None
                file_path = row[2] if len(row) > 2 else None
            if name == "main" and file_path:
                return os.path.splitext(os.path.basename(file_path))[0]
        return "Anonymous"

    def _fetch_distinct_values(self, connection: sqlite3.Connection, table_name: str,
                               column_name: str, max_num: int = 5) -> list:
        quoted_table = self._quote_identifier(table_name)
        quoted_column = self._quote_identifier(column_name)
        sql = (
            f"SELECT DISTINCT {quoted_column} AS value "
            f"FROM {quoted_table} "
            f"WHERE {quoted_column} IS NOT NULL "
            f"LIMIT {int(max_num)}"
        )
        result = self.execute_query(connection, sql)
        if not result.success:
            return []
        return [row.get('value') for row in result.data if 'value' in row]

    def _quote_identifier(self, identifier: str) -> str:
        return '"' + identifier.replace('"', '""') + '"'

    def _get_field_type(self, field_type: str, simple_mode: bool = True) -> str:
        if not simple_mode:
            return field_type
        return field_type.split("(")[0]

    def _examples_to_str(self, examples: list) -> list:
        values = list(examples)
        for i in range(len(values)):
            if isinstance(values[i], datetime.date):
                values = [values[i]]
                break
            if isinstance(values[i], datetime.datetime):
                values = [values[i]]
                break
            if isinstance(values[i], decimal.Decimal):
                values[i] = str(float(values[i]))
            if self._is_email(str(values[i])):
                values = []
                break
            if 'http://' in str(values[i]) or 'https://' in str(values[i]):
                values = []
                break
            if values[i] is not None and not isinstance(values[i], str):
                pass
            elif values[i] is not None and '.com' in values[i]:
                pass

        return [str(v) for v in values if v is not None and len(str(v)) > 0]

    def _is_email(self, string: str) -> bool:
        pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
        match = re.match(pattern, string)
        return match is not None

    def discover_databases(self, config: Dict) -> Dict[str, DatabaseInfo]:
        """Discover SQLite database files"""
        databases = {}
        root_path = config.get('root_path', '.')
        patterns = config.get('patterns', ['*.sqlite', '*.sqlite3', '*.db'])

        if isinstance(root_path, (list, tuple, set)):
            root_paths = list(root_path)
        else:
            root_paths = [root_path]

        for root in root_paths:
            for pattern in patterns:
                for db_path in glob.glob(os.path.join(root, '**', pattern), recursive=True):
                    if os.path.isfile(db_path):
                        rel_path = os.path.relpath(db_path, root)
                        parts = rel_path.split(os.sep)
                        db_id = parts[0] if len(parts) > 1 else os.path.splitext(parts[0])[0]

                        original_id = db_id
                        counter = 1
                        while db_id in databases:
                            db_id = f"{original_id}_{counter}"
                            counter += 1

                        databases[db_id] = DatabaseInfo(
                            db_id=db_id,
                            db_type='sqlite',
                            connection_info={'path': db_path},
                            metadata={'file_path': db_path, 'file_size': os.path.getsize(db_path)}
                        )
                        self.logger.info(f"Discovered SQLite database: {db_id} at {db_path}")

        return databases

    def get_number_of_special_column(self, connection: sqlite3.Connection) -> int:
        """
        Get the number of columns in database
        """
        count = 0
        try:
            # Get the complete structure information of the database
            schema = self.get_schema_info(connection)
            
            # Get all the table information from the schema
            tables = schema.get('tables', {})
            
            # Traverse each table
            for table_name, table_info in tables.items():
                # Get all the column names from the table information
                columns = table_info.get('columns', {})
                
                # Traverse the dictionary of columns (key is column name)
                for column_name in columns:
                    # Check if the variable is a string
                    if isinstance(column_name, str):
                        count += 1
                        
        except Exception as e:
            print(f"error: Error counting embedding columns: {e}")

        if count == 0:
                print(f"error: No columns ending with '_embedding'.")
        return count   
