from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
import sqlite3
import pymysql
import threading
import os
import glob
from dataclasses import dataclass
from dataflow import get_logger


@dataclass
class DatabaseInfo:
    db_id: str
    db_type: str
    connection_info: Dict
    metadata: Optional[Dict] = None

class DatabaseConnector(ABC):
    
    @abstractmethod
    def connect(self, connection_info: Dict) -> Any:
        pass

    @abstractmethod
    def get_execution_plan(self, connection, sql: str) -> List[Any]:
        pass
    
    @abstractmethod
    def execute_query(self, connection, sql: str) -> List:
        pass
    
    @abstractmethod
    def get_tables(self, connection) -> List[str]:
        pass
    
    @abstractmethod
    def get_table_schema(self, connection, table_name: str) -> Dict:
        pass
    
    @abstractmethod
    def get_sample_data(self, connection, table_name: str, limit: int) -> Dict[str, Any]:
        pass

class SQLiteConnector(DatabaseConnector):
    
    def connect(self, connection_info: Dict) -> sqlite3.Connection:
        db_path = connection_info['path']
        return sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5)
    
    def get_execution_plan(self, connection: sqlite3.Connection, sql: str) -> List[Any]:
        cursor = connection.cursor()
        cursor.execute(f"EXPLAIN QUERY PLAN {sql}")
        return cursor.fetchall()

    def execute_query(self, connection: sqlite3.Connection, sql: str) -> List:
        cursor = connection.cursor()
        cursor.execute(sql)
        return cursor.fetchall()
    
    def get_tables(self, connection: sqlite3.Connection) -> List[str]:
        cursor = connection.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
        return [row[0] for row in cursor.fetchall()]
    
    def get_table_schema(self, connection: sqlite3.Connection, table_name: str) -> Dict:
        cursor = connection.cursor()
        
        cursor.execute(f"PRAGMA table_info([{table_name}])")
        columns = cursor.fetchall()
        
        schema = {
            'columns': {},
            'primary_keys': [],
            'foreign_keys': []
        }
        
        for col_info in columns:
            col_name = col_info[1]
            col_type = col_info[2]
            is_pk = col_info[5]
            
            schema['columns'][col_name] = {
                'type': self._normalize_type(col_type),
                'raw_type': col_type
            }
            
            if is_pk:
                schema['primary_keys'].append(col_name)
        
        cursor.execute(f"PRAGMA foreign_key_list([{table_name}])")
        foreign_keys = cursor.fetchall()
        
        for fk in foreign_keys:
            schema['foreign_keys'].append({
                'column': fk[3],
                'referenced_table': fk[2],
                'referenced_column': fk[4]
            })
        
        return schema
    
    def get_sample_data(self, connection: sqlite3.Connection, table_name: str, limit: int) -> Dict[str, Any]:
        cursor = connection.cursor()
        cursor.execute(f'SELECT * FROM "{table_name}" LIMIT {limit}')
        rows = cursor.fetchall()
        
        column_names = [description[0] for description in cursor.description]
        
        return {
            'columns': column_names,
            'rows': rows
        }
    
    def _normalize_type(self, db_type: str) -> str:
        if not db_type:
            return 'text'
        
        db_type = db_type.lower()
        
        if any(t in db_type for t in ['int', 'integer', 'number']):
            return 'integer'
        elif any(t in db_type for t in ['real', 'float', 'double']):
            return 'real'
        elif any(t in db_type for t in ['text', 'varchar', 'char']):
            return 'text'
        elif any(t in db_type for t in ['blob']):
            return 'blob'
        else:
            return 'text'

class MySQLConnector(DatabaseConnector):
    
    def connect(self, connection_info: Dict) -> pymysql.Connection:
        return pymysql.connect(**connection_info, connect_timeout=5)

    def get_execution_plan(self, connection: pymysql.Connection, sql: str) -> List[Any]:
        cursor = connection.cursor()
        cursor.execute(f"EXPLAIN {sql}")
        return cursor.fetchall()
    
    def execute_query(self, connection: pymysql.Connection, sql: str) -> List:
        cursor = connection.cursor()
        cursor.execute(sql)
        return cursor.fetchall()
    
    def get_tables(self, connection: pymysql.Connection) -> List[str]:
        cursor = connection.cursor()
        cursor.execute("SHOW TABLES")
        return [row[0] for row in cursor.fetchall()]
    
    def get_table_schema(self, connection: pymysql.Connection, table_name: str) -> Dict:
        cursor = connection.cursor()
        database = connection.db.decode('utf-8')
        
        cursor.execute("""
            SELECT column_name, data_type, column_key, is_nullable, column_default
            FROM information_schema.columns 
            WHERE table_schema = %s AND table_name = %s
            ORDER BY ordinal_position
        """, (database, table_name))
        columns = cursor.fetchall()
        
        schema = {
            'columns': {},
            'primary_keys': [],
            'foreign_keys': []
        }
        
        for col_name, col_type, col_key, is_nullable, col_default in columns:
            schema['columns'][col_name] = {
                'type': self._normalize_type(col_type),
                'raw_type': col_type,
                'nullable': is_nullable == 'YES',
                'default': col_default
            }
            
            if col_key == 'PRI':
                schema['primary_keys'].append(col_name)
        
        cursor.execute("""
            SELECT column_name, referenced_table_name, referenced_column_name
            FROM information_schema.key_column_usage 
            WHERE table_schema = %s AND table_name = %s
            AND referenced_table_name IS NOT NULL
        """, (database, table_name))
        foreign_keys = cursor.fetchall()
        
        for col_name, ref_table, ref_column in foreign_keys:
            schema['foreign_keys'].append({
                'column': col_name,
                'referenced_table': ref_table,
                'referenced_column': ref_column
            })
        
        return schema
    
    def get_sample_data(self, connection: pymysql.Connection, table_name: str, limit: int) -> Dict[str, Any]:
        cursor = connection.cursor()
        cursor.execute(f'SELECT * FROM `{table_name}` LIMIT {limit}')
        rows = cursor.fetchall()
        
        column_names = [description[0] for description in cursor.description]
        
        return {
            'columns': column_names,
            'rows': rows
        }
    
    def _normalize_type(self, db_type: str) -> str:
        if not db_type:
            return 'text'
        
        db_type = db_type.lower()
        
        type_mapping = {
            'int': 'integer', 'integer': 'integer', 'bigint': 'integer', 'smallint': 'integer',
            'tinyint': 'integer', 'mediumint': 'integer',
            'float': 'real', 'double': 'real', 'decimal': 'real', 'numeric': 'real',
            'varchar': 'text', 'char': 'text', 'text': 'text', 'longtext': 'text',
            'mediumtext': 'text', 'tinytext': 'text',
            'date': 'date', 'datetime': 'datetime', 'timestamp': 'datetime', 'time': 'time',
            'blob': 'blob', 'longblob': 'blob', 'mediumblob': 'blob', 'tinyblob': 'blob'
        }
        
        return type_mapping.get(db_type, 'text')

class DatabaseRegistry:
    
    def __init__(self, logger=None):
        self.logger = logger
        self._registry: Dict[str, DatabaseInfo] = {}
    
    def register_database(self, db_info: DatabaseInfo):
        self._registry[db_info.db_id] = db_info
    
    def get_database(self, db_id: str) -> Optional[DatabaseInfo]:
        return self._registry.get(db_id)
    
    def list_databases(self) -> List[str]:
        return list(self._registry.keys())
    
    def database_exists(self, db_id: str) -> bool:
        return db_id in self._registry
    
    def clear(self):
        self._registry.clear()

class SchemaCache:
    
    def __init__(self):
        self._cache: Dict[str, Dict] = {}
    
    def get(self, cache_key: str) -> Optional[Dict]:
        return self._cache.get(cache_key)
    
    def set(self, cache_key: str, schema: Dict):
        self._cache[cache_key] = schema
    
    def clear(self):
        self._cache.clear()
    
    def remove(self, cache_key: str):   
        self._cache.pop(cache_key, None)

class DatabaseManager:
    
    def __init__(self, db_type: str = "sqlite", config: Optional[Dict] = None, logger=None):
        self.db_type = db_type.lower()
        self.config = config or {}
        self.logger = get_logger()
        
        self.registry = DatabaseRegistry(logger)
        self.cache = SchemaCache()
        
        self.connectors = {
            'sqlite': SQLiteConnector(),
            'mysql': MySQLConnector()
        }
        
        self._validate_config()
        self._discover_databases()
    
    def _validate_config(self):
        if self.db_type not in self.connectors:
            raise ValueError(f"Unsupported database type: {self.db_type}")
        
        if self.db_type == 'sqlite':
            if 'root_path' not in self.config:
                raise ValueError("SQLite requires 'root_path' in config")
            if not os.path.exists(self.config['root_path']):
                raise ValueError(f"SQLite root path does not exist: {self.config['root_path']}")
        elif self.db_type in ['mysql']:
            required_params = ['host', 'user', 'password']
            missing = [p for p in required_params if p not in self.config]
            if missing:
                raise ValueError(f"Missing required {self.db_type} config: {missing}")
    
    def _discover_databases(self):
        try:
            if self.db_type == 'sqlite':
                self._discover_sqlite_databases()
            elif self.db_type == 'mysql':
                self._discover_mysql_databases()
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error discovering databases: {e}")
    
    def _discover_sqlite_databases(self):
        root_path = self.config['root_path']
        extensions = ['*.sqlite', '*.sqlite3', '*.db']
        
        for ext in extensions:
            for db_file in glob.glob(os.path.join(root_path, '**', ext), recursive=True):
                if os.path.isfile(db_file):
                    db_id = os.path.splitext(os.path.basename(db_file))[0]
                    
                    if not self.registry.database_exists(db_id):
                        db_info = DatabaseInfo(
                            db_id=db_id,
                            db_type='sqlite',
                            connection_info={'path': db_file},
                            metadata={'size': os.path.getsize(db_file)}
                        )
                        self.registry.register_database(db_info)
    
    def _discover_mysql_databases(self):
        connector = self.connectors['mysql']
        
        try:
            temp_config = {k: v for k, v in self.config.items() if k != 'database'}
            conn = connector.connect(temp_config)
            
            databases = connector.execute_query(conn, "SHOW DATABASES")
            system_dbs = {'information_schema', 'mysql', 'performance_schema', 'sys'}
            
            for (db_name,) in databases:
                if db_name not in system_dbs:
                    db_info = DatabaseInfo(
                        db_id=db_name,
                        db_type='mysql',
                        connection_info={**self.config, 'database': db_name}
                    )
                    self.registry.register_database(db_info)
            
            conn.close()
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error discovering MySQL databases: {e}")

    def execute_query_with_timeout(self, db_id: str, sql: str, timeout: float = 5.0) -> Dict[str, Any]:
        db_info = self.registry.get_database(db_id)
        if not db_info:
            raise ValueError(f"Database {db_id} not found")
        
        connector = self.connectors[db_info.db_type]
        result = None
        exception = None
        
        def execute_query():
            nonlocal result, exception
            conn = None
            try:
                conn = connector.connect(db_info.connection_info)
                rows = connector.execute_query(conn, sql)
                
                columns = []
                if hasattr(conn, 'cursor'):
                    cursor = conn.cursor()
                    if hasattr(cursor, 'description') and cursor.description:
                        columns = [desc[0] for desc in cursor.description]
                
                result = {
                    'success': True,
                    'data': rows,
                    'columns': columns,
                    'row_count': len(rows) if rows else 0
                }
            except Exception as e:
                exception = e
            finally:
                if conn:
                    try:
                        conn.close()
                    except:
                        pass
        
        thread = threading.Thread(target=execute_query)
        thread.daemon = True
        thread.start()
        
        thread.join(timeout)
        
        if thread.is_alive():   
            if self.logger:
                self.logger.warning(f"Query timeout after {timeout}s for database {db_id}")
            return {
                'success': False,
                'error': f'Query timeout after {timeout} seconds',
                'data': [],
                'columns': [],
                'row_count': 0
            }
        
        if exception:
            if self.logger:
                self.logger.error(f"Query error for database {db_id}: {exception}")
            return {
                'success': False,
                'error': str(exception),
                'data': [],
                'columns': [],
                'row_count': 0
            }
        
        return result or {
            'success': False,
            'error': 'Unknown error occurred',
            'data': [],
            'columns': [],
            'row_count': 0
        }
    
    def execute_query(self, db_id: str, sql: str, timeout: float = 5.0) -> Dict[str, Any]:
        return self.execute_query_with_timeout(db_id, sql, timeout)
    
    def analyze_sql_execution_plan(self, db_id: str, sql: str, timeout: float = 5.0) -> Dict[str, Any]:
        db_info = self.registry.get_database(db_id)
        if not db_info:
            self.logger.error(f"Database {db_id} not found")
            return {
                'success': False,
                'error': f'Database {db_id} not found'
            }
        
        connector = self.connectors[db_info.db_type]
        result = None
        exception = None
        
        def analyze_plan():
            nonlocal result, exception
            conn = None
            try:
                conn = connector.connect(db_info.connection_info)
                execution_plan = connector.get_execution_plan(conn, sql)
                result = {
                    'success': True,
                    'execution_plan': execution_plan
                }
            except Exception as e:
                exception = e
            finally:
                if conn:
                    try:
                        conn.close()
                    except:
                        pass
        
        thread = threading.Thread(target=analyze_plan)
        thread.daemon = True
        thread.start()
        thread.join(timeout)
        
        if thread.is_alive():
            if self.logger:
                self.logger.warning(f"Execution plan analysis timeout after {timeout}s for database {db_id}")
            return {
                'success': False,
                'error': f'Execution plan analysis timeout after {timeout} seconds'
            }
        
        if exception:
            if self.logger:
                self.logger.error(f"Execution plan analysis error for database {db_id}: {exception}")
            return {
                'success': False,
                'error': str(exception)
            }
        
        return result or {
            'success': False,
            'error': 'Unknown error occurred'
        }

    

    def compare_sql(self, db_id: str, sql1: str, sql2: str, timeout: float = 5.0) -> bool:
        try:
            result1 = self.execute_query_with_timeout(db_id, sql1, timeout)
            
            result2 = self.execute_query_with_timeout(db_id, sql2, timeout)
            
            if not result1['success'] or not result2['success']:
                if self.logger:
                    self.logger.warning(f"SQL comparison failed - one or both queries failed to execute")
                    if not result1['success']:
                        self.logger.warning(f"SQL1 error: {result1.get('error', 'Unknown error')}")
                    if not result2['success']:
                        self.logger.warning(f"SQL2 error: {result2.get('error', 'Unknown error')}")
                return False
            
            data1 = result1['data']
            data2 = result2['data']
            
            if len(data1) != len(data2):
                if self.logger:
                    self.logger.debug(f"Row count mismatch: {len(data1)} vs {len(data2)}")
                return False
            
            columns1 = result1['columns']
            columns2 = result2['columns']
            if len(columns1) != len(columns2):
                if self.logger:
                    self.logger.debug(f"Column count mismatch: {len(columns1)} vs {len(columns2)}")
                return False
            
            def normalize_row(row):
                if isinstance(row, (list, tuple)):
                    return tuple(str(item) if item is not None else None for item in row)
                return (str(row) if row is not None else None,)
            
            normalized_data1 = sorted([normalize_row(row) for row in data1])
            normalized_data2 = sorted([normalize_row(row) for row in data2])
            
            is_equal = normalized_data1 == normalized_data2
            
            if self.logger:
                if is_equal:
                    self.logger.debug(f"SQL comparison successful - results match ({len(data1)} rows)")
                else:
                    self.logger.debug(f"SQL comparison failed - results differ")
            
            return is_equal
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error during SQL comparison: {str(e)}")
            return False
    
    def get_database_schema(self, db_id: str, use_cache: bool = True) -> Dict:
        cache_key = f"{db_id}_{self.db_type}"
        if use_cache:
            cached = self.cache.get(cache_key)
            if cached:
                return cached
        
        db_info = self.registry.get_database(db_id)
        if not db_info:
            if self.logger:
                self.logger.error(f"Database {db_id} not found")
            return {}
        
        connector = self.connectors[db_info.db_type]
        schema = {'tables': {}, 'foreign_keys': [], 'primary_keys': []}
        
        try:
            with self._get_connection(db_info) as conn:
                tables = connector.get_tables(conn)
                
                for table_name in tables:
                    table_schema = connector.get_table_schema(conn, table_name)
                    
                    sample_data = connector.get_sample_data(conn, table_name, 2)
                    
                    for i, col_name in enumerate(table_schema['columns'].keys()):
                        examples = []
                        for row in sample_data['rows']:
                            if i < len(row) and row[i] is not None:
                                examples.append(str(row[i]))
                        table_schema['columns'][col_name]['examples'] = examples
                    
                    schema['tables'][table_name] = table_schema
                    
                    for pk in table_schema['primary_keys']:
                        schema['primary_keys'].append({
                            'table': table_name,
                            'column': pk
                        })

                    for fk in table_schema['foreign_keys']:
                        schema['foreign_keys'].append({
                            'source_table': table_name,
                            'source_column': fk['column'],
                            'referenced_table': fk['referenced_table'],
                            'referenced_column': fk['referenced_column']
                        })
            
            if use_cache:
                self.cache.set(cache_key, schema)
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error getting schema for {db_id}: {e}")
            return {}
        
        return schema
    
    def get_table_names_and_create_statements(self, db_id: str) -> tuple:
        schema = self.get_database_schema(db_id)
        if not schema:
            return [], []
        
        table_names = list(schema['tables'].keys())
        create_statements = self._generate_create_statements(schema)
        
        return table_names, create_statements
    
    def get_insert_statements(self, db_id: str, table_names: Optional[List[str]] = None, limit: int = 2) -> Dict[str, List[str]]:
        db_info = self.registry.get_database(db_id)
        if not db_info:
            return {}
        
        if table_names is None:
            schema = self.get_database_schema(db_id)
            table_names = list(schema.get('tables', {}).keys())
        
        connector = self.connectors[db_info.db_type]
        result = {}
        
        try:
            with self._get_connection(db_info) as conn:
                for table_name in table_names:
                    sample_data = connector.get_sample_data(conn, table_name, limit)
                    insert_statements = self._generate_insert_statements(
                        table_name, sample_data, db_info.db_type
                    )
                    result[table_name] = insert_statements
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error getting insert statements for {db_id}: {e}")
        
        return result
    
    def _get_connection(self, db_info: DatabaseInfo):
        connector = self.connectors[db_info.db_type]
        
        class ConnectionManager:
            def __init__(self, connector, connection_info):
                self.connector = connector
                self.connection_info = connection_info
                self.connection = None
            
            def __enter__(self):
                self.connection = self.connector.connect(self.connection_info)
                return self.connection
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                if self.connection:
                    self.connection.close()
        
        return ConnectionManager(connector, db_info.connection_info)
    
    def _generate_create_statements(self, schema: Dict) -> List[str]:
        statements = []
        
        for table_name, table_info in schema['tables'].items():
            columns = []
            
            for col_name, col_info in table_info['columns'].items():
                col_def = f'"{col_name}" {col_info["raw_type"]}'
                if not col_info.get('nullable', True):
                    col_def += ' NOT NULL'
                if col_info.get('default'):
                    col_def += f' DEFAULT {col_info["default"]}'
                columns.append(col_def)
            
            if table_info['primary_keys']:
                pk_cols = ', '.join([f'"{pk}"' for pk in table_info['primary_keys']])
                columns.append(f'PRIMARY KEY ({pk_cols})')
            
            create_sql = f'CREATE TABLE "{table_name}" (\n    ' + ',\n    '.join(columns) + '\n);'
            statements.append(create_sql)
        
        return statements
    
    def _generate_insert_statements(self, table_name: str, sample_data: Dict, db_type: str) -> List[str]:
        if not sample_data['rows']:
            return []
        
        statements = []
        columns = sample_data['columns']
        
        if db_type == 'mysql':
            table_quote = '`'
            col_quote = '`'
        else:
            table_quote = '"'
            col_quote = '"'
        
        for row in sample_data['rows']:
            values = []
            for value in row:
                if value is None:
                    values.append('NULL')
                elif isinstance(value, str):
                    escaped = value.replace("'", "''")
                    values.append(f"'{escaped}'")
                elif isinstance(value, (int, float)):
                    values.append(str(value))
                else:
                    values.append(f"'{str(value)}'")
            
            columns_str = ', '.join([f'{col_quote}{col}{col_quote}' for col in columns])
            values_str = ', '.join(values)
            
            statement = f'INSERT INTO {table_quote}{table_name}{table_quote} ({columns_str}) VALUES ({values_str});'
            statements.append(statement)
        
        return statements
    
    def generate_ddl_with_examples(self, db_id: str) -> str:
        schema = self.get_database_schema(db_id)
        return self._generate_ddl(schema, use_examples=True)
    
    def generate_ddl_without_examples(self, db_id: str) -> str:
        schema = self.get_database_schema(db_id)
        return self._generate_ddl(schema, use_examples=False)
    
    def generate_formatted_schema_with_examples(self, db_id: str) -> str:
        schema = self.get_database_schema(db_id)
        return self._generate_formatted_schema(schema, use_examples=True)
    
    def generate_formatted_schema_without_examples(self, db_id: str) -> str:
        schema = self.get_database_schema(db_id)    
        return self._generate_formatted_schema(schema, use_examples=False)
    
    def _generate_ddl(self, schema: Dict, use_examples: bool = False) -> str:
        if not schema or not schema.get('tables'):
            return ""
        
        ddl_statements = []
        
        for table_name, table_info in schema['tables'].items():
            columns_ddl = []
            
            for col_name, col_info in table_info['columns'].items():
                raw_type = col_info.get('raw_type', col_info.get('type', 'TEXT'))
                
                col_def = f'    "{col_name}" {raw_type}'
                
                if not col_info.get('nullable', True):
                    col_def += ' NOT NULL'
                
                if col_info.get('default'):
                    col_def += f' DEFAULT {col_info["default"]}'
                
                if use_examples and col_info.get('examples'):
                    examples = col_info['examples'][:3]
                    examples_str = ', '.join([str(ex) for ex in examples])
                    col_def += f'  -- Examples: {examples_str}'
                
                columns_ddl.append(col_def)
            
            if table_info.get('primary_keys'):
                pk_columns = ', '.join([f'"{pk}"' for pk in table_info['primary_keys']])
                columns_ddl.append(f'    PRIMARY KEY ({pk_columns})')
            
            for fk in table_info.get('foreign_keys', []):
                fk_def = (f'    FOREIGN KEY ("{fk["column"]}") '
                         f'REFERENCES "{fk["referenced_table"]}"("{fk["referenced_column"]}")')
                columns_ddl.append(fk_def)
            
            if columns_ddl:
                create_table_sql = (
                    f'CREATE TABLE "{table_name}" (\n' +
                    ',\n'.join(columns_ddl) +
                    '\n);'
                )
                ddl_statements.append(create_table_sql)
        
        global_fks = []
        for fk in schema.get('foreign_keys', []):
            table_fks = schema['tables'][fk['source_table']].get('foreign_keys', [])
            if not any(tfk['column'] == fk['source_column'] and 
                      tfk['referenced_table'] == fk['referenced_table'] and
                      tfk['referenced_column'] == fk['referenced_column'] 
                      for tfk in table_fks):
                fk_sql = (f'ALTER TABLE "{fk["source_table"]}" '
                         f'ADD FOREIGN KEY ("{fk["source_column"]}") '
                         f'REFERENCES "{fk["referenced_table"]}"("{fk["referenced_column"]}");')
                global_fks.append(fk_sql)
        
        if global_fks:
            ddl_statements.extend(global_fks)
        
        return '\n\n'.join(ddl_statements)
    
    def _generate_formatted_schema(self, schema: Dict, use_examples: bool = False) -> str:
        if not schema or not schema.get('tables'):
            return "No tables found in the database."
        
        formatted_parts = []
        
        table_count = len(schema['tables'])
        pk_count = len(schema.get('primary_keys', []))
        fk_count = len(schema.get('foreign_keys', []))
        
        formatted_parts.append("# Database Schema Overview")
        formatted_parts.append(f"- Total Tables: {table_count}")
        formatted_parts.append(f"- Primary Keys: {pk_count}")
        formatted_parts.append(f"- Foreign Keys: {fk_count}")
        formatted_parts.append("")
        
        for table_name, table_info in schema['tables'].items():
            formatted_parts.append(f"## Table: {table_name}")
            
            if table_info.get('primary_keys'):
                pk_list = ', '.join(table_info['primary_keys'])
                formatted_parts.append(f"**Primary Key:** {pk_list}")
            
            formatted_parts.append("**Columns:**")
            for col_name, col_info in table_info['columns'].items():
                raw_type = col_info.get('raw_type', col_info.get('type', 'TEXT'))
                
                col_desc = f"- `{col_name}` ({raw_type})"
                
                constraints = []
                if not col_info.get('nullable', True):
                    constraints.append("NOT NULL")
                if col_info.get('default'):
                    constraints.append(f"DEFAULT {col_info['default']}")
                
                if constraints:
                    col_desc += f" - {', '.join(constraints)}"
                
                if use_examples and col_info.get('examples'):
                    examples = col_info['examples'][:3]
                    examples_str = ', '.join([f"`{ex}`" for ex in examples])
                    col_desc += f" - Examples: {examples_str}"
                
                formatted_parts.append(col_desc)
            
            table_fks = [fk for fk in schema.get('foreign_keys', []) 
                        if fk['source_table'] == table_name]
            if table_fks:
                formatted_parts.append("**Foreign Keys:**")
                for fk in table_fks:
                    fk_desc = (f"- `{fk['source_column']}` → "
                              f"`{fk['referenced_table']}.{fk['referenced_column']}`")
                    formatted_parts.append(fk_desc)
            
            referenced_fks = [fk for fk in schema.get('foreign_keys', []) 
                             if fk['referenced_table'] == table_name]
            if referenced_fks:
                formatted_parts.append("**Referenced by:**")
                for fk in referenced_fks:
                    ref_desc = (f"- `{fk['source_table']}.{fk['source_column']}` → "
                               f"`{fk['referenced_column']}`")
                    formatted_parts.append(ref_desc)
            
            formatted_parts.append("")
        
        if schema.get('foreign_keys'):
            formatted_parts.append("## Table Relationships")
            for fk in schema['foreign_keys']:
                rel_desc = (f"- `{fk['source_table']}` → `{fk['referenced_table']}` "
                           f"({fk['source_column']} → {fk['referenced_column']})")
                formatted_parts.append(rel_desc)
        
        return '\n'.join(formatted_parts)
        
    
    def list_databases(self) -> List[str]:
        return self.registry.list_databases()
    
    def database_exists(self, db_id: str) -> bool:
        return self.registry.database_exists(db_id)
    
    def refresh_database_registry(self):
        self.registry.clear()
        self.cache.clear()
        self._discover_databases()
    
    def clear_cache(self):
        self.cache.clear()