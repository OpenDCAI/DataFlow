from typing import Dict, Any, Optional, Tuple
from ..base import DatabaseConnectorABC, DatabaseInfo, QueryResult
import sqlite3
import glob
import os
import re
import threading
import time
from filelock import FileLock
from dataflow import get_logger


# ============== SQLite Connector ==============
class SQLiteVecConnector(DatabaseConnectorABC):
    _class_lock = threading.Lock()
    _db_init_locks: Dict[str, threading.Lock] = {}
    _initialized_dbs = set()

    def __init__(self):
        self.logger = get_logger()
        self._thread_local = threading.local()
        self._extensions_loaded = False

    def _ensure_vec_available(self):
        if hasattr(self, '_sqlite_vec'):
            return
        try:
            import sqlite_vec
            self._sqlite_vec = sqlite_vec
        except ImportError:
            raise ImportError(
                "The 'vectorsql' optional dependencies are required but not installed.\n"
                "Please run: pip install 'open-dataflow[vectorsql]'"
            )

    def _ensure_lembed_available(self):
        if hasattr(self, '_sqlite_lembed'):
            return
        try:
            import sqlite_lembed
            self._sqlite_lembed = sqlite_lembed
        except ImportError:
            raise ImportError(
                "sqlite_lembed is required for vector SQL execution. "
                "Please run: pip install 'open-dataflow[vectorsql]'"
            )

    def connect(self, connection_info: Dict) -> sqlite3.Connection:
        """
        Connect to the database. Returns an independent connection for each calling thread,
        ensuring that dangerous initialization is performed only once, while safely loading
        the model into memory for each connection.
        """
        self._ensure_vec_available()
        
        db_path = connection_info.get('path')
        if not db_path:
            raise ValueError("Connection info must contain a 'path' key.")

        if not hasattr(self._thread_local, 'connections'):
            self._thread_local.connections = {}
        if db_path in self._thread_local.connections:
            return self._thread_local.connections[db_path]

        # Step 1: One-time initialization (ensure model has been written to DB file)
        # This code block is executed only once for each DB file in each process.
        # Use class-level lock to ensure the check and _initialize_database_disk_state call are thread-safe.
        with self._class_lock:
            if db_path not in self._initialized_dbs:
                self.logger.info(f"'{db_path}' is not initialized in this process. Performing one-time disk setup...")
                self._initialize_database_disk_state(connection_info)
                self._initialized_dbs.add(db_path)

        # Step 2: Create a connection for the current thread
        self.logger.debug(f"Thread {threading.get_ident()} creating new DB connection for '{db_path}'...")
        conn = sqlite3.connect(
            db_path,
            timeout=30.0,
            check_same_thread=False
        )
        conn.row_factory = sqlite3.Row
        
        # Step 3: Load extensions and activate model for each new connection
        model_name = connection_info.get('model_name')
        model_path = connection_info.get('model_path')
        enable_lembed = connection_info.get('enable_lembed', True)
        if model_path:
            model_path = os.path.abspath(model_path)

        lock_path = db_path + ".lock"
        file_lock = FileLock(lock_path, timeout=120)

        with file_lock:
            self.logger.debug(f"Process-safe lock acquired by thread {threading.get_ident()}. Configuring connection...")
            conn.enable_load_extension(True)
            self._sqlite_vec.load(conn)

            if enable_lembed:
                self._ensure_lembed_available()
                self._sqlite_lembed.load(conn)
                self._ensure_trusted_schema_enabled(conn)

            if enable_lembed and model_name and model_path:
                self.logger.debug(f"Activating model '{model_name}' for connection...")
                register_sql = """
                INSERT OR IGNORE INTO main.lembed_models (name, model)
                VALUES (?, lembed_model_from_file(?))
                """
                temp_register_sql = """
                INSERT OR IGNORE INTO temp.lembed_models (name, model)
                VALUES (?, lembed_model_from_file(?))
                """
                def _register(sql: str, target: str) -> bool:
                    for attempt in range(3):
                        try:
                            conn.execute(sql, (model_name, model_path))
                            conn.commit()
                            return True
                        except Exception as exc:
                            try:
                                conn.rollback()
                            except Exception:
                                pass
                            if attempt < 2:
                                wait = 0.5 * (attempt + 1)
                                self.logger.debug(
                                    f"Registering model '{model_name}' in {target} failed (attempt {attempt + 1}/3). "
                                    f"Retrying in {wait:.1f}s. Error: {exc}",
                                    extra={
                                        "db_path": db_path,
                                        "model_name": model_name,
                                        "target": target,
                                        "trusted_schema": self._get_trusted_schema_value(conn),
                                    },
                                    exc_info=True
                                )
                                time.sleep(wait)
                            else:
                                self.logger.debug(
                                    f"Registration attempt exhausted for '{model_name}' in {target}: {exc}",
                                    extra={
                                        "db_path": db_path,
                                        "model_name": model_name,
                                        "target": target,
                                        "trusted_schema": self._get_trusted_schema_value(conn),
                                    },
                                    exc_info=True
                                )
                    return False

                if not _register(register_sql, "main.lembed_models"):
                    if not _register(temp_register_sql, "temp.lembed_models"):
                        raise RuntimeError(
                            f"Failed to register model '{model_name}' for database '{db_path}' even in temp schema."
                        )
                    else:
                        self.logger.info(
                            f"Model '{model_name}' registered in temp.lembed_models for connection '{db_path}'."
                        )

        self._thread_local.connections[db_path] = conn
        self.logger.debug(f"Connection for thread {threading.get_ident()} created and configured successfully.")
        return conn

    def _get_trusted_schema_value(self, connection: sqlite3.Connection) -> Optional[int]:
        try:
            cursor = connection.execute("PRAGMA trusted_schema")
            result = cursor.fetchone()
            cursor.close()
        except Exception as exc:
            self.logger.debug(f"Unable to read PRAGMA trusted_schema: {exc}")
            return None

        if result:
            try:
                return int(result[0])
            except (TypeError, ValueError):
                return None
        return None

    def _ensure_trusted_schema_enabled(self, connection: sqlite3.Connection) -> None:
        """Enable trusted_schema when sqlite is built with it disabled by default."""
        value = self._get_trusted_schema_value(connection)

        if value == 1:
            return

        self.logger.info(
            "trusted_schema is disabled for current SQLite connection. Enabling it to allow sqlite_lembed to load models."
        )
        connection.execute("PRAGMA trusted_schema=ON;")
        self.logger.debug("trusted_schema set to 1 for current SQLite connection.")

    def _initialize_database_disk_state(self, connection_info: Dict):
        self._ensure_vec_available()
        
        db_path = connection_info['path']
        model_name = connection_info.get('model_name')
        model_path = connection_info.get('model_path')
        enable_lembed = connection_info.get('enable_lembed', True)
        if model_path:
            model_path = os.path.abspath(model_path)

        lock_path = db_path + ".lock"
        file_lock = FileLock(lock_path, timeout=120)

        with file_lock:
            init_conn = None
            try:
                init_conn = sqlite3.connect(db_path, timeout=30.0)
                init_conn.execute("PRAGMA journal_mode=WAL;")
                init_conn.enable_load_extension(True)
                self._sqlite_vec.load(init_conn)

                if enable_lembed:
                    self._ensure_lembed_available()
                    self._sqlite_lembed.load(init_conn)
                    self._ensure_trusted_schema_enabled(init_conn)
                    trusted_schema = self._get_trusted_schema_value(init_conn)
                else:
                    trusted_schema = None

                if enable_lembed and model_name and model_path:
                    if not os.path.exists(model_path):
                        raise FileNotFoundError(f"Embedding model file not found at path: {model_path}")
                    abs_model_path = os.path.abspath(model_path)

                    self.logger.info(
                        f"Lock acquired. Ensuring model '{model_name}' is persisted in '{db_path}' "
                        f"(source model path: '{abs_model_path}')..."
                    )
                    self.logger.debug(
                        f"trusted_schema value before model registration for '{db_path}': {trusted_schema}"
                    )
                    register_sql = """
                    INSERT OR IGNORE INTO main.lembed_models (name, model)
                    VALUES (?, lembed_model_from_file(?))
                    """
                    persisted = False
                    for attempt in range(3):
                        try:
                            init_conn.execute(register_sql, (model_name, abs_model_path))
                            init_conn.commit()
                            self.logger.info(f"Model '{model_name}' persistence check completed.")
                            persisted = True
                            break
                        except Exception as exc:
                            try:
                                init_conn.rollback()
                            except Exception:
                                pass
                            if attempt < 2:
                                wait_time = 0.5 * (attempt + 1)
                                self.logger.warning(
                                    "Persisting embedding model into database failed (attempt %d/3). "
                                    "Retrying in %.1fs. Error: %s",
                                    attempt + 1,
                                    wait_time,
                                    exc,
                                    extra={
                                        "db_path": db_path,
                                        "model_name": model_name,
                                        "model_path": abs_model_path,
                                        "trusted_schema": trusted_schema,
                                    },
                                    exc_info=True
                                )
                                time.sleep(wait_time)
                            else:
                                self.logger.warning(
                                    "Failed to persist embedding model into database after retries. "
                                    "Will rely on per-connection loading. Error: %s",
                                    exc,
                                    extra={
                                        "db_path": db_path,
                                        "model_name": model_name,
                                        "model_path": abs_model_path,
                                        "trusted_schema": trusted_schema,
                                    },
                                    exc_info=True
                                )
                    if not persisted:
                        self.logger.debug(
                            f"Model '{model_name}' not stored in '{db_path}'. Runtime registration will be used."
                        )
            finally:
                if init_conn:
                    init_conn.close()

    def close(self, connection: Optional[sqlite3.Connection] = None):
        if hasattr(self._thread_local, 'connections'):
            if connection:
                for db_path, conn in list(self._thread_local.connections.items()):
                    if conn == connection:
                        try:
                            conn.close()
                            self.logger.debug(f"Thread {threading.get_ident()} closed the specified connection to '{db_path}'.")
                            del self._thread_local.connections[db_path]
                        except Exception as e:
                            self.logger.error(f"Error closing specified connection to '{db_path}': {e}", exc_info=True)
                        break
            else:
                for db_path, conn in self._thread_local.connections.items():
                    try:
                        conn.close()
                        self.logger.debug(f"Thread {threading.get_ident()} closed connection to '{db_path}'.")
                    except Exception as e:
                        self.logger.error(f"Error closing connection to '{db_path}': {e}", exc_info=True)
                self._thread_local.connections = {}

    def execute_query(self, connection: sqlite3.Connection, sql: str, params: Optional[Tuple] = None) -> QueryResult:
        cursor = None
        try:
            cursor = connection.cursor()
            if params:
                cursor.execute(sql, params)
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
            self.logger.debug(f"Query execution error: {e}")
            return QueryResult(
                success=False,
                error=str(e)
            )
        finally:
            if cursor:
                try:
                    cursor.close()
                except:
                    pass

    def _get_db_details(self, schema: Dict[str, Any]) -> str:
        db_details = []
        
        for table_name, table_info in schema['tables'].items():
            original_create_statement = table_info.get('create_statement', '')
            is_virtual_table = "CREATE VIRTUAL TABLE" in original_create_statement.upper()

            if is_virtual_table:
                db_details.append(original_create_statement)
            else:
                column_defs = []
                for col_name, col_info in table_info.get('columns', {}).items():
                    col_def = f"    `{col_name}` {col_info['type']}"
                    if not col_info.get('nullable', True):
                        col_def += " NOT NULL"
                    if col_info.get('default') is not None:
                        col_def += f" DEFAULT {col_info['default']}"
                    column_defs.append(col_def)
                
                constraints = []
                primary_keys = table_info.get('primary_keys', [])
                if primary_keys:
                    pk_cols = ", ".join(f"`{col}`" for col in primary_keys)
                    constraints.append(f"    PRIMARY KEY ({pk_cols})")
                
                foreign_keys = table_info.get('foreign_keys', [])
                for i, fk in enumerate(foreign_keys):
                    constraints.append(
                        f"    CONSTRAINT fk_{table_name}_{fk['column']}_{i} "
                        f"FOREIGN KEY (`{fk['column']}`) "
                        f"REFERENCES `{fk['referenced_table']}` (`{fk['referenced_column']}`)"
                    )
                
                create_table = f"CREATE TABLE `{table_name}` (\n"
                all_parts = column_defs + constraints
                create_table += ",\n".join(all_parts)
                create_table += "\n);"
                
                insert_statements = table_info.get('insert_statement', [])
                if insert_statements:
                    create_table += "\n\n-- Sample data:\n"
                    if isinstance(insert_statements, list):
                        create_table += "\n".join(insert_statements)
                    else:
                        create_table += str(insert_statements)

                db_details.append(create_table)
        
        return "\n\n".join(db_details)

    def get_schema_info(self, connection: sqlite3.Connection) -> Dict[str, Any]:
        schema = {'tables': {}}
        
        result = self.execute_query(
            connection, 
            "SELECT name, sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
        
        if not result.success:
            return schema
        
        for row in result.data:
            table_name = row['name']
            create_statement = row['sql']
            table_info = self._get_table_info(connection, table_name, create_statement)
            schema['tables'][table_name] = table_info
        
        schema['db_details'] = self._get_db_details(schema)
        return schema

    def generate_create_statement(self, table_name: str, table_info: Dict[str, Any]) -> str:
        create_stmt = ""
        columns = []
        for col_name, col_info in table_info.get('columns', {}).items():
            col_def = f"`{col_name}` {col_info['type']}"
            if not col_info.get('nullable', True):
                col_def += " NOT NULL"
            if col_info.get('default') is not None:
                col_def += f" DEFAULT {col_info['default']}"
            columns.append(col_def)
            
        pk_cols = table_info.get('primary_keys', [])
        if pk_cols:
            columns.append(f"PRIMARY KEY ({', '.join(f'`{pk}`' for pk in pk_cols)})")
            
        for fk in table_info.get('foreign_keys', []):
            fk_def = (f"FOREIGN KEY (`{fk['column']}`) "
                        f"REFERENCES `{fk['referenced_table']}`(`{fk['referenced_column']}`)")
            columns.append(fk_def)
            
        if columns:
            create_stmt = f"CREATE TABLE `{table_name}` (\n  " + ",\n  ".join(columns) + "\n);"
        
        return create_stmt
    
    def _get_table_info(self, connection: sqlite3.Connection, table_name: str, create_statement: str) -> Dict[str, Any]:
        is_vec_table = "USING vec0" in create_statement.upper()

        table_info = {
            'columns': {}, 'primary_keys': [], 'foreign_keys': [], 'sample_data': [],
            'create_statement': create_statement, 'insert_statement': None
        }
        
        result = self.execute_query(connection, f"PRAGMA table_info([{table_name}])")
        if result.success:
            for col in result.data:
                col_name = col['name']
                table_info['columns'][col_name] = {
                    'type': col['type'], 'nullable': not col['notnull'],
                    'default': col['dflt_value'], 'primary_key': bool(col['pk'])
                }

        if is_vec_table:
            self.logger.info(f"Detected sqlite-vec virtual table: '{table_name}'. Skipping sample data and foreign key retrieval.")
        else:
            for col in result.data:
                if col['pk']:
                    table_info['primary_keys'].append(col['name'])
            
            fk_result = self.execute_query(connection, f"PRAGMA foreign_key_list([{table_name}])")
            if fk_result.success:
                for fk in fk_result.data:
                    table_info['foreign_keys'].append({
                        'column': fk['from'], 'referenced_table': fk['table'], 'referenced_column': fk['to']
                    })
            
            all_columns = list(table_info['columns'].keys())
            columns_to_select = [f"`{col}`" for col in all_columns if not col.endswith('_embedding')]
            
            if columns_to_select:
                select_clause = ", ".join(columns_to_select)
                sample_query = f"SELECT {select_clause} FROM [{table_name}] LIMIT 3"
                sample_result = self.execute_query(connection, sample_query)
                
                if sample_result.success and sample_result.data:
                    is_data_too_long = False
                    for row in sample_result.data:
                        row_char_length = sum(len(str(value)) for value in row.values())
                        if row_char_length > 10000:
                            self.logger.warning(f"A row of sample data for table '{table_name}' exceeds 10,000 characters and will be ignored.")
                            is_data_too_long = True
                            break
                    
                    if not is_data_too_long:
                        table_info['sample_data'] = sample_result.data

            table_info['create_statement'] = self.generate_create_statement(table_name, table_info)

            if table_info['sample_data']:
                insert_statement_list = []
                for row in table_info['sample_data']:  
                    columns = list(row.keys())
                    values = []
                    for col in columns:
                        val = row[col]
                        if val is None:
                            values.append('NULL')
                        elif isinstance(val, (int, float)):
                            values.append(str(val))
                        else:
                            escaped_val = str(val).replace("'", "''")
                            values.append(f"'{escaped_val}'")
                    
                    table_ref = f"[{table_name}]"
                    col_ref = [f"[{col}]" for col in columns]
                    insert_sql = f"INSERT INTO {table_ref} ({', '.join(col_ref)}) VALUES ({', '.join(values)})"
                    insert_statement_list.append(insert_sql)
                table_info['insert_statement'] = "\n\n".join(insert_statement_list)
        
        return table_info

    def discover_databases(self, config: Dict) -> Dict[str, DatabaseInfo]:
        databases = {}
        root_path = config.get('root_path', '.')
        patterns = config.get('patterns', ['*.sqlite', '*.sqlite3', '*.db'])
        
        for pattern in patterns:
            for db_path in glob.glob(os.path.join(root_path, '**', pattern), recursive=True):
                if os.path.isfile(db_path):
                    db_id = os.path.splitext(os.path.basename(db_path))[0]
                    
                    original_id = db_id
                    counter = 1
                    while db_id in databases:
                        db_id = f"{original_id}_{counter}"
                        counter += 1
                    
                    connection_info = {
                        'path': db_path,
                        'model_name': config.get('model_name'),
                        'model_path': config.get('model_path'),
                        'enable_lembed': config.get('enable_lembed', True)
                    }
                    
                    databases[db_id] = DatabaseInfo(
                        db_id=db_id,
                        db_type='sqlite-vec',
                        connection_info=connection_info,
                        metadata={'file_path': db_path, 'file_size': os.path.getsize(db_path)}
                    )
                    self.logger.info(f"Discovered SQLite database: {db_id} at {db_path}")
        
        return databases

    def get_number_of_special_column(self, connection: sqlite3.Connection) -> int:
        """
        get the number of vector columns in database
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
                columns = table_info.get('columns', [])
                
                # Directly traverse the list of column names
                for column_name in columns:
                    # Check if the variable is a string, and whether it ends with "_embedding"
                    if isinstance(column_name, str) and column_name.endswith("_embedding"):
                        count += 1
                        
        except Exception as e:
            print(f"error: Error counting embedding columns {e}")

        if count == 0:
                print(f"error: No columns ending with '_embedding'.")
        return count 
