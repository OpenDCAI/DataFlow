from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
import sqlite3
import pymysql
import threading
import copy
import os
import glob
from dataclasses import dataclass
from dataflow import get_logger
from tqdm import tqdm
from queue import Queue, Empty
from contextlib import contextmanager
import time
import concurrent.futures
import hashlib
from enum import Enum

class OperationType(Enum):
    EXECUTE_QUERY = "execute_query"
    ANALYZE_PLAN = "analyze_plan"
    COMPARE_SQL = "compare_sql"
    VALIDATE_SQL = "validate_sql"

@dataclass
class DatabaseInfo:
    db_id: str
    db_type: str
    connection_info: Dict
    metadata: Optional[Dict] = None

@dataclass
class BatchOperation:
    operation_type: OperationType
    db_id: str
    sql: str
    timeout: float = 2.0
    additional_params: Optional[Dict[str, Any]] = None
    operation_id: Optional[str] = None

@dataclass
class BatchResult:
    operation_id: str
    success: bool
    data: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0

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
        conn = sqlite3.connect(
            f"file:{db_path}?mode=ro", 
            uri=True, 
            timeout=2,
            check_same_thread=False 
        )
        conn.row_factory = sqlite3.Row
        return conn
    
    def set_query_timeout(self, connection: sqlite3.Connection, timeout_seconds: float):
        try:
            timeout_ms = int(timeout_seconds * 1000)
            connection.execute(f"PRAGMA busy_timeout = {timeout_ms}")
            
            try:
                connection.execute(f"PRAGMA statement_timeout = {timeout_ms}")
            except sqlite3.OperationalError:
                pass
                
        except Exception as e:
            pass
    
    def validate_connection(self, connection: sqlite3.Connection) -> bool:
        try:
            connection.execute("SELECT 1").fetchone()
            return True
        except (sqlite3.Error, AttributeError):
            return False
    
    def execute_query_with_timeout(self, connection: sqlite3.Connection, sql: str, timeout_seconds: float) -> List:
        import signal
        import threading
        
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Query execution exceeded {timeout_seconds} seconds")
        
        cursor = connection.cursor()
        result_container = {'result': None, 'error': None, 'completed': False}
        
        def execute_query():
            try:
                cursor.execute(sql)
                results = cursor.fetchall()
                
                if results and hasattr(results[0], 'keys'):
                    result_container['result'] = [self._process_row_nulls(dict(row)) for row in results]
                elif results:
                    columns = [description[0] for description in cursor.description]
                    result_container['result'] = [self._process_row_nulls(dict(zip(columns, row))) for row in results]
                else:
                    result_container['result'] = []
                    
                result_container['completed'] = True
                
            except Exception as e:
                result_container['error'] = str(e)
            finally:
                cursor.close()
        
        query_thread = threading.Thread(target=execute_query, daemon=True)
        query_thread.start()
        query_thread.join(timeout=timeout_seconds)
        
        if query_thread.is_alive():
            raise TimeoutError(f"SQLite query execution exceeded {timeout_seconds} seconds")
        
        if result_container['error']:
            raise Exception(result_container['error'])
        
        if not result_container['completed']:
            raise TimeoutError(f"SQLite query execution timed out after {timeout_seconds} seconds")
        
        return result_container['result']
    
    def execute_query(self, connection: sqlite3.Connection, sql: str) -> List:
        cursor = connection.cursor()
        try:
            cursor.execute(sql)
            results = cursor.fetchall()
            if results and hasattr(results[0], 'keys'):
                return [self._process_row_nulls(dict(row)) for row in results]
            elif results:
                columns = [description[0] for description in cursor.description]
                return [self._process_row_nulls(dict(zip(columns, row))) for row in results]
            return []
        finally:
            cursor.close()
    
    def get_execution_plan(self, connection: sqlite3.Connection, sql: str) -> List[Any]:
        cursor = connection.cursor()
        try:
            cursor.execute(f"EXPLAIN QUERY PLAN {sql}")
            results = cursor.fetchall()
            if results:
                columns = ['id', 'parent', 'notused', 'detail']
                return [dict(zip(columns, row)) for row in results]
            return []
        except Exception as e:
            raise Exception(f"SQLite execution plan error: {e}")
        finally:
            cursor.close()
    
    def _process_row_nulls(self, row_dict: dict) -> dict:
        processed = {}
        for key, value in row_dict.items():
            if value is None:
                processed[key] = None
            elif isinstance(value, str) and value.lower() in ('null', ''):
                processed[key] = None
            else:
                processed[key] = value
        return processed
    
    def get_tables(self, connection: sqlite3.Connection) -> List[str]:
        cursor = connection.cursor()
        try:
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
            return [row[0] for row in cursor.fetchall()]
        finally:
            cursor.close()
    
    def get_table_schema(self, connection: sqlite3.Connection, table_name: str) -> Dict:
        cursor = connection.cursor()
        try:
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
        finally:
            cursor.close()
    
    def get_sample_data(self, connection: sqlite3.Connection, table_name: str, limit: int) -> Dict[str, Any]:
        cursor = connection.cursor()
        try:
            cursor.execute(f'SELECT * FROM "{table_name}" LIMIT {limit}')
            rows = cursor.fetchall()
            
            column_names = [description[0] for description in cursor.description]
            
            if rows and hasattr(rows[0], 'keys'):
                dict_rows = [dict(row) for row in rows]
            else:
                dict_rows = [dict(zip(column_names, row)) for row in rows]
            
            return {
                'columns': column_names,
                'rows': dict_rows
            }
        finally:
            cursor.close()
    
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
        return pymysql.connect(**connection_info, connect_timeout=2)
    
    def set_query_timeout(self, connection: pymysql.Connection, timeout_seconds: float):
        try:
            cursor = connection.cursor()
            timeout_ms = int(timeout_seconds * 1000)
            cursor.execute(f"SET SESSION max_execution_time = {timeout_ms}")
            cursor.close()
        except Exception as e:
            pass
    
    def execute_query_with_timeout(self, connection: pymysql.Connection, sql: str, timeout_seconds: float) -> List:
        import threading
        
        result_container = {'result': None, 'error': None, 'completed': False}
        
        def execute_query():
            cursor = None
            try:
                cursor = connection.cursor(pymysql.cursors.DictCursor)
                cursor.execute(sql)
                result_container['result'] = cursor.fetchall()
                result_container['completed'] = True
            except Exception as e:
                result_container['error'] = str(e)
            finally:
                if cursor:
                    cursor.close()
        
        query_thread = threading.Thread(target=execute_query, daemon=True)
        query_thread.start()
        query_thread.join(timeout=timeout_seconds)
        
        if query_thread.is_alive():
            try:
                connection.cancel()
            except:
                pass
            raise TimeoutError(f"MySQL query execution exceeded {timeout_seconds} seconds")
        
        if result_container['error']:
            if 'max_execution_time' in result_container['error'].lower():
                raise TimeoutError(f"MySQL query execution exceeded {timeout_seconds} seconds")
            raise Exception(result_container['error'])
        
        if not result_container['completed']:
            raise TimeoutError(f"MySQL query execution timed out after {timeout_seconds} seconds")
        
        return result_container['result']

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

class CacheManager:
    
    def __init__(self, max_size: int = 100, ttl: int = 1800):
        self.max_size = max_size
        self.ttl = ttl
        
        self._schema_cache: Dict[str, Dict] = {}
        self._table_info_cache: Dict[str, Dict] = {}
        self._query_result_cache: Dict[str, Dict] = {}
        self._access_times: Dict[str, float] = {}
        self._lock = threading.RLock()
        
        self._hit_counts = {'schema': 0, 'table': 0, 'query': 0}
        self._total_requests = {'schema': 0, 'table': 0, 'query': 0}

    def _generate_cache_key(self, prefix: str, *args) -> str:
        key_parts = [prefix] + [str(arg).replace('|', '\\|') for arg in args]
        key_string = '|'.join(key_parts)
        return f"{prefix}:{hashlib.md5(key_string.encode()).hexdigest()[:16]}"
    
    def _get_from_cache(self, cache_dict: Dict, cache_key: str, cache_type: str) -> Optional[Any]:
        with self._lock:
            self._total_requests[cache_type] += 1
            
            if cache_key in cache_dict:
                if time.time() - self._access_times.get(cache_key, 0) > self.ttl:
                    self._remove_from_cache(cache_dict, cache_key)
                    return None
                
                self._access_times[cache_key] = time.time()
                self._hit_counts[cache_type] += 1
                return copy.deepcopy(cache_dict[cache_key])
            return None
    
    def _set_to_cache(self, cache_dict: Dict, cache_key: str, data: Any):
        with self._lock:
            if len(cache_dict) >= self.max_size:
                self._cleanup_lru_cache(cache_dict)
            
            cache_dict[cache_key] = copy.deepcopy(data)
            self._access_times[cache_key] = time.time()
    
    def _cleanup_lru_cache(self, cache_dict: Dict, cleanup_ratio: float = 0.25):
        if not cache_dict:
            return
        
        cleanup_count = max(1, int(len(cache_dict) * cleanup_ratio))
        
        cache_keys = list(cache_dict.keys())
        oldest_keys = sorted(
            [k for k in cache_keys if k in self._access_times],
            key=lambda k: self._access_times[k]
        )[:cleanup_count]
        
        for old_key in oldest_keys:
            cache_dict.pop(old_key, None)
            self._access_times.pop(old_key, None)
    
    def _remove_from_cache(self, cache_dict: Dict, cache_key: str):
        cache_dict.pop(cache_key, None)
        self._access_times.pop(cache_key, None)
    
    def get_schema(self, db_id: str, db_type: str) -> Optional[Dict]:
        cache_key = self._generate_cache_key("schema", db_id, db_type)
        return self._get_from_cache(self._schema_cache, cache_key, 'schema')
    
    def set_schema(self, db_id: str, db_type: str, schema: Dict):
        cache_key = self._generate_cache_key("schema", db_id, db_type)
        self._set_to_cache(self._schema_cache, cache_key, schema)
    
    def get_table_info(self, db_id: str, table_name: str) -> Optional[Dict]:
        cache_key = self._generate_cache_key("table", db_id, table_name)
        return self._get_from_cache(self._table_info_cache, cache_key, 'table')
    
    def set_table_info(self, db_id: str, table_name: str, info: Dict):
        cache_key = self._generate_cache_key("table", db_id, table_name)
        self._set_to_cache(self._table_info_cache, cache_key, info)
    
    def get_query_result(self, db_id: str, query_key: str) -> Optional[Dict]:
        cache_key = self._generate_cache_key("query", db_id, query_key)
        return self._get_from_cache(self._query_result_cache, cache_key, 'query')
    
    def set_query_result(self, db_id: str, query_key: str, result: Dict):
        cache_key = self._generate_cache_key("query", db_id, query_key)
        self._set_to_cache(self._query_result_cache, cache_key, result)
    
    def clear_all(self):
        with self._lock:
            self._schema_cache.clear()
            self._table_info_cache.clear()
            self._query_result_cache.clear()
            self._access_times.clear()
            
            for key in self._hit_counts:
                self._hit_counts[key] = 0
                self._total_requests[key] = 0
    
    def get_cache_stats(self) -> Dict[str, Any]:
        with self._lock:
            hit_rates = {}
            for cache_type in self._hit_counts:
                total_req = self._total_requests[cache_type]
                hit_rates[cache_type] = self._hit_counts[cache_type] / max(total_req, 1)
            
            return {
                'cache_sizes': {
                    'schema': len(self._schema_cache),
                    'table': len(self._table_info_cache),
                    'query': len(self._query_result_cache)
                },
                'max_size': self.max_size,
                'hit_rates': hit_rates,
                'total_requests': dict(self._total_requests)
            }


class ConnectionPool:
    
    def __init__(self, connector, connection_info, max_connections=10, max_idle_time=300):
        self.connector = connector
        self.connection_info = connection_info
        self.max_connections = max_connections
        self.max_idle_time = max_idle_time
        
        self._pool = Queue(maxsize=max_connections)
        self._active_connections = 0
        self._lock = threading.RLock()
        self._connection_times = {}
        self._closed = False
        
        self.logger = get_logger()
        
        self.is_sqlite = isinstance(connector, SQLiteConnector)
        if self.is_sqlite:
            self.max_connections = min(3, max_connections)
            self._sqlite_connections = {}
            self._sqlite_lock = threading.Lock()

    @contextmanager
    def get_connection(self):
        if self._closed:
            raise RuntimeError("Connection pool is closed")
        
        thread_id = threading.current_thread().ident
        conn = None
        conn_created = False
        conn_id = None
        
        try:
            if self.is_sqlite:
                with self._sqlite_lock:
                    if thread_id in self._sqlite_connections:
                        conn = self._sqlite_connections[thread_id]
                        if not self.connector.validate_connection(conn):
                            self._close_connection_safely(conn)
                            del self._sqlite_connections[thread_id]
                            conn = None
                    
                    if conn is None:
                        conn = self.connector.connect(self.connection_info)
                        self._sqlite_connections[thread_id] = conn
                        conn_created = True
                        conn_id = thread_id
                
                yield conn
                
            else:
                try:
                    conn_info = self._pool.get_nowait()
                    conn = conn_info['connection']
                    conn_id = conn_info.get('id')
                    
                    if not self.connector.validate_connection(conn):
                        self._close_connection_safely(conn)
                        conn = None
                        if conn_id:
                            self._connection_times.pop(conn_id, None)
                            
                except Empty:
                    pass
                
                if conn is None:
                    with self._lock:
                        if self._active_connections < self.max_connections and not self._closed:
                            conn = self.connector.connect(self.connection_info)
                            self._active_connections += 1
                            conn_created = True
                            conn_id = id(conn)
                            self._connection_times[conn_id] = time.time()
                        else:
                            if self._closed:
                                raise RuntimeError("Connection pool is closed")
                            conn_info = self._pool.get(timeout=10)
                            conn = conn_info['connection']
                            conn_id = conn_info.get('id')
                            
                            if not self.connector.validate_connection(conn):
                                self._close_connection_safely(conn)
                                if conn_id:
                                    self._connection_times.pop(conn_id, None)
                                raise Exception("No valid connections available")
                
                yield conn
                
        except Exception as e:
            self.logger.warning(f"Connection error for thread {thread_id}: {e}")
            if conn:
                if self.is_sqlite:
                    with self._sqlite_lock:
                        if thread_id in self._sqlite_connections:
                            self._close_connection_safely(self._sqlite_connections[thread_id])
                            del self._sqlite_connections[thread_id]
                elif not self.connector.validate_connection(conn):
                    self._close_connection_safely(conn)
                    if conn_created:
                        with self._lock:
                            self._active_connections -= 1
                    if conn_id:
                        self._connection_times.pop(conn_id, None)
                conn = None
            raise e
        finally:
            if conn and not self._closed and not self.is_sqlite:
                try:
                    self._pool.put_nowait({
                        'connection': conn,
                        'last_used': time.time(),
                        'id': conn_id
                    })
                except:
                    self._close_connection_safely(conn)
                    if conn_created:
                        with self._lock:
                            self._active_connections -= 1
                    if conn_id:
                        self._connection_times.pop(conn_id, None)

    def _close_connection_safely(self, conn):
        try:
            if conn and hasattr(conn, 'close'):
                conn.close()
        except Exception:
            pass
    
    def close_all(self):
        self._closed = True
        closed_count = 0
        
        if self.is_sqlite:
            with self._sqlite_lock:
                for thread_id, conn in self._sqlite_connections.items():
                    self._close_connection_safely(conn)
                    closed_count += 1
                self._sqlite_connections.clear()
        
        while True:
            try:
                conn_info = self._pool.get_nowait()
                self._close_connection_safely(conn_info['connection'])
                closed_count += 1
            except Empty:
                break
        
        with self._lock:
            self._active_connections = 0
            self._connection_times.clear()
        
        return closed_count

    def cleanup_idle_connections(self):
        if self._closed or self.is_sqlite:
            return 0
            
        current_time = time.time()
        active_connections = []
        cleaned_count = 0
        
        while True:
            try:
                conn_info = self._pool.get_nowait()
                if current_time - conn_info['last_used'] > self.max_idle_time:
                    self._close_connection_safely(conn_info['connection'])
                    with self._lock:
                        self._active_connections -= 1
                    conn_id = conn_info.get('id')
                    if conn_id:
                        self._connection_times.pop(conn_id, None)
                    cleaned_count += 1
                else:
                    active_connections.append(conn_info)
            except Empty:
                break
        
        for conn_info in active_connections:
            try:
                self._pool.put_nowait(conn_info)
            except:
                self._close_connection_safely(conn_info['connection'])
                with self._lock:
                    self._active_connections -= 1
                conn_id = conn_info.get('id')
                if conn_id:
                    self._connection_times.pop(conn_id, None)
        
        return cleaned_count
    
    def get_stats(self) -> dict:
        with self._lock:
            return {
                'active_connections': self._active_connections,
                'max_connections': self.max_connections,
                'pool_size': self._pool.qsize(),
                'pool_utilization': self._active_connections / self.max_connections if self.max_connections > 0 else 0,
                'closed': self._closed,
                'is_sqlite': self.is_sqlite
            }


class BatchOperationExecutor:
    
    def __init__(self, registry, cache, connectors, connection_pool_getter, logger, shared_executor=None):
        self.registry = registry
        self.cache = cache
        self.connectors = connectors
        self.get_connection_pool = connection_pool_getter
        self.logger = logger
        
        self.executor = shared_executor
        self._own_executor = shared_executor is None
        self._closed = False
        self._progress_bar = None
        self._progress_lock = threading.Lock()
        
        if self.executor is None:
            max_workers = min(8, (os.cpu_count() or 1) + 2)
            self.executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=max_workers,
                thread_name_prefix="BatchExecutor"
            )
    
    def execute_batch_operations(self, operations: List[BatchOperation], show_progress: bool = True) -> List[BatchResult]:
        if self._closed:
            raise RuntimeError("BatchOperationExecutor is closed")
            
        if not operations:
            return []

        operations_by_db = {}
        for op in operations:
            if op.db_id not in operations_by_db:
                operations_by_db[op.db_id] = []
            operations_by_db[op.db_id].append(op)
        
        if show_progress:
            self._progress_bar = tqdm(
                total=len(operations),
                desc="Executing SQL operations",
                unit="ops",
                ncols=100,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
            )
        
        try:
            future_to_operations = {}
            for db_id, db_operations in operations_by_db.items():
                future = self.executor.submit(self._execute_db_operations, db_id, db_operations)
                future_to_operations[future] = db_operations
            
            all_results = []
            completed_futures = set()
            
            total_ops = len(operations)
            timeout = min(300, max(60, total_ops * 0.5))
            
            # self.logger.info(f"Batch operation timeout set to {timeout}s for {total_ops} operations")
            
            start_time = time.time()
            
            try:
                for future in concurrent.futures.as_completed(future_to_operations.keys(), timeout=timeout):
                    completed_futures.add(future)
                    try:
                        results = future.result()
                        all_results.extend(results)
                        
                        if show_progress and self._progress_bar:
                            with self._progress_lock:
                                self._progress_bar.update(len(results))
                                
                                success_count = sum(1 for r in results if r.success)
                                timeout_count = sum(1 for r in results if not r.success and 'timeout' in r.error.lower())
                                self._progress_bar.set_postfix({
                                    'Success': success_count,
                                    'Timeout': timeout_count,
                                    'Failed': len(results) - success_count
                                })
                        
                        elapsed = time.time() - start_time
                        # self.logger.info(f"Completed batch for database, got {len(results)} results. Total elapsed: {elapsed:.2f}s")
                    except Exception as e:
                        # self.logger.warning(f"Error in batch operation: {e}")
                        operations = future_to_operations[future]
                        for op in operations:
                            all_results.append(BatchResult(
                                operation_id=op.operation_id or f"{op.operation_type.value}_{hash(op.sql)}",
                                success=False,
                                error=f"Batch execution error: {str(e)}"
                            ))
                        
                        if show_progress and self._progress_bar:
                            with self._progress_lock:
                                self._progress_bar.update(len(operations))
                                
            except concurrent.futures.TimeoutError:
                elapsed = time.time() - start_time
                # self.logger.warning(f"Batch operation timeout after {timeout}s (actual: {elapsed:.2f}s)")
            
            unfinished_futures = set(future_to_operations.keys()) - completed_futures
            for future in unfinished_futures:
                future.cancel()
                operations = future_to_operations[future]
                for op in operations:
                    all_results.append(BatchResult(
                        operation_id=op.operation_id or f"{op.operation_type.value}_{hash(op.sql)}",
                        success=False,
                        error="Operation timeout or cancelled"
                    ))
                
                if show_progress and self._progress_bar:
                    with self._progress_lock:
                        self._progress_bar.update(len(operations))
            
            success_count = sum(1 for r in all_results if r.success)
            # self.logger.info(f"Batch execution completed: {success_count}/{len(all_results)} operations succeeded")
            
            return all_results
            
        finally:
            if show_progress and self._progress_bar:
                self._progress_bar.close()
                self._progress_bar = None

    def _execute_db_operations(self, db_id: str, operations: List[BatchOperation]) -> List[BatchResult]:
        try:
            pool = self.get_connection_pool(db_id)
            db_info = self.registry.get_database(db_id)
            
            if not db_info:
                raise ValueError(f"Database {db_id} not found")
                
            connector = self.connectors.get(db_info.db_type)
            if not connector:
                raise ValueError(f"No connector found for database type: {db_info.db_type}")
            
            results = []
            connection_start = time.time()
            unified_timeout = operations[0].timeout if operations else 30.0
            
            with pool.get_connection() as conn:
                connection_time = time.time() - connection_start
                # self.logger.info(f"Got connection for {db_id} in {connection_time:.2f}s, executing {len(operations)} operations with timeout {unified_timeout}s")
                
                if hasattr(connector, 'set_query_timeout'):
                    try:
                        connector.set_query_timeout(conn, unified_timeout)
                        self.logger.debug(f"Set query timeout to {unified_timeout}s for database {db_id}")
                    except Exception as e:
                        self.logger.warning(f"Failed to set query timeout for {db_id}: {e}")
                
                for i, op in enumerate(operations):
                    if self._closed:
                        results.append(BatchResult(
                            operation_id=op.operation_id or f"{op.operation_type.value}_{hash(op.sql)}",
                            success=False,
                            error="Executor is closed"
                        ))
                        continue
                        
                    try:
                        start_time = time.time()
                        
                        result = self._execute_single_operation_with_db_timeout(conn, connector, op, unified_timeout)
                        result.execution_time = time.time() - start_time
                        results.append(result)
                        
                        # if result.execution_time > unified_timeout * 0.8:
                        #     self.logger.warning(f"Slow operation {op.operation_id} took {result.execution_time:.2f}s (timeout: {unified_timeout}s)")
                            
                    except Exception as e:
                        execution_time = time.time() - start_time
                        error_msg = str(e)
                        
                        if 'timeout' in error_msg.lower() or 'exceeded' in error_msg.lower():
                            error_msg = f"SQL execution timeout after {unified_timeout}s: {error_msg}"
                        
                        # self.logger.warning(f"Exception in operation {op.operation_id}: {error_msg}")
                        results.append(BatchResult(
                            operation_id=op.operation_id or f"{op.operation_type.value}_{hash(op.sql)}",
                            success=False,
                            error=error_msg,
                            execution_time=execution_time
                        ))
            
            success_count = sum(1 for r in results if r.success)
            timeout_count = sum(1 for r in results if not r.success and 'timeout' in r.error.lower())
            total_time = time.time() - connection_start
            
            # self.logger.info(f"Database {db_id} completed: {success_count}/{len(results)} operations succeeded, {timeout_count} timeouts, in {total_time:.2f}s")
            
            return results
            
        except Exception as e:
            # self.logger.warning(f"Error in _execute_db_operations for {db_id}: {e}")
            results = []
            for op in operations:
                results.append(BatchResult(
                    operation_id=op.operation_id or f"{op.operation_type.value}_{hash(op.sql)}",
                    success=False,
                    error=f"Connection error: {str(e)}"
                ))
            return results

    def _execute_single_operation_with_db_timeout(self, conn, connector, operation: BatchOperation, timeout_seconds: float) -> BatchResult:
        operation_id = operation.operation_id or f"{operation.operation_type.value}_{hash(operation.sql)}"
                
        try:
            if operation.operation_type == OperationType.EXECUTE_QUERY:
                return self._execute_query_operation_with_timeout(conn, connector, operation, operation_id, timeout_seconds)
            elif operation.operation_type == OperationType.ANALYZE_PLAN:
                return self._execute_plan_operation(conn, connector, operation, operation_id)
            elif operation.operation_type == OperationType.VALIDATE_SQL:
                return self._execute_validate_operation_with_timeout(conn, connector, operation, operation_id, timeout_seconds)
            elif operation.operation_type == OperationType.COMPARE_SQL:
                return self._execute_compare_operation_with_timeout(conn, connector, operation, operation_id, timeout_seconds)
            else:
                raise ValueError(f"Unsupported operation type: {operation.operation_type}")
                
        except Exception as e:
            error_msg = str(e)
            if 'timeout' in error_msg.lower() or 'exceeded' in error_msg.lower():
                error_msg = f"Operation timeout: {error_msg}"
            
            return BatchResult(
                operation_id=operation_id,
                success=False,
                error=error_msg
            )

    def _execute_query_operation_with_timeout(self, conn, connector, operation: BatchOperation, operation_id: str, timeout_seconds: float) -> BatchResult:
        try:
            if hasattr(connector, 'execute_query_with_timeout'):
                rows = connector.execute_query_with_timeout(conn, operation.sql, timeout_seconds)
            else:
                rows = connector.execute_query(conn, operation.sql)
            
            columns = []
            if rows and isinstance(rows[0], dict):
                columns = list(rows[0].keys())
            elif rows:
                columns = [f"col_{i}" for i in range(len(rows[0]))] if rows[0] else []
            
            result_data = {
                'success': True,
                'data': rows,
                'columns': columns,
                'row_count': len(rows) if rows else 0
            }
            
            return BatchResult(
                operation_id=operation_id,
                success=True,
                data=result_data
            )
            
        except TimeoutError as e:
            return BatchResult(
                operation_id=operation_id,
                success=False,
                error=f"Query execution timeout: {str(e)}"
            )
        except Exception as e:
            error_msg = str(e)
            if 'timeout' in error_msg.lower() or 'max_execution_time' in error_msg.lower():
                error_msg = f"Query execution timeout: {error_msg}"
            
            return BatchResult(
                operation_id=operation_id,
                success=False,
                error=error_msg
            )

    def _execute_plan_operation(self, conn, connector, operation: BatchOperation, operation_id: str) -> BatchResult:
        try:
            execution_plan = connector.get_execution_plan(conn, operation.sql)
            return BatchResult(
                operation_id=operation_id,
                success=True,
                data={'execution_plan': execution_plan}
            )
        except Exception as e:
            return BatchResult(
                operation_id=operation_id,
                success=False,
                error=str(e)
            )

    def _execute_validate_operation_with_timeout(self, conn, connector, operation: BatchOperation, operation_id: str, timeout_seconds: float) -> BatchResult:
        try:
            if hasattr(connector, 'execute_query_with_timeout'):
                connector.execute_query_with_timeout(conn, operation.sql, timeout_seconds)
            else:
                connector.execute_query(conn, operation.sql)
            
            return BatchResult(
                operation_id=operation_id,
                success=True,
                data={'valid': True}
            )
        except (TimeoutError, Exception) as e:
            error_msg = str(e)
            if 'timeout' in error_msg.lower():
                return BatchResult(
                    operation_id=operation_id,
                    success=True,
                    data={'valid': False, 'error': 'timeout'}
                )
            else:
                return BatchResult(
                    operation_id=operation_id,
                    success=True,
                    data={'valid': False, 'error': error_msg}
                )

    def _execute_compare_operation_with_timeout(self, conn, connector, operation: BatchOperation, operation_id: str, timeout_seconds: float) -> BatchResult:
        try:
            additional_params = operation.additional_params or {}
            sql2 = additional_params.get('sql2')
            if not sql2:
                raise ValueError("sql2 required for compare operation")
            
            result1 = self._execute_query_for_comparison_with_timeout(conn, connector, operation.sql, timeout_seconds)
            result2 = self._execute_query_for_comparison_with_timeout(conn, connector, sql2, timeout_seconds)
            
            is_equal = self._compare_query_results(result1, result2)
            
            return BatchResult(
                operation_id=operation_id,
                success=True,
                data={'equal': is_equal, 'result1': result1, 'result2': result2}
            )
        except Exception as e:
            error_msg = str(e)
            if 'timeout' in error_msg.lower():
                error_msg = f"Comparison operation timeout: {error_msg}"
            
            return BatchResult(
                operation_id=operation_id,
                success=False,
                error=error_msg
            )

    def _execute_query_for_comparison_with_timeout(self, conn, connector, sql: str, timeout_seconds: float) -> Dict:
        try:
            if hasattr(connector, 'execute_query_with_timeout'):
                rows = connector.execute_query_with_timeout(conn, sql, timeout_seconds)
            else:
                rows = connector.execute_query(conn, sql)
                
            columns = []
            if rows and isinstance(rows[0], dict):
                columns = list(rows[0].keys())
            return {
                'success': True,
                'data': rows,
                'columns': columns
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'data': [],
                'columns': []
            }
    
    def _compare_query_results(self, result1: Dict, result2: Dict) -> bool:
        if not result1['success'] or not result2['success']:
            return False
        
        data1 = result1['data']
        data2 = result2['data']
        
        if len(data1) != len(data2) or len(result1['columns']) != len(result2['columns']):
            return False
        
        def normalize_row(row):
            if isinstance(row, dict):
                return tuple(sorted([(k, str(v) if v is not None else None) for k, v in row.items()]))
            elif isinstance(row, (list, tuple)):
                return tuple(str(item) if item is not None else None for item in row)
            return (str(row) if row is not None else None,)
        
        normalized_data1 = sorted([normalize_row(row) for row in data1])
        normalized_data2 = sorted([normalize_row(row) for row in data2])
        
        return normalized_data1 == normalized_data2

    def close(self):
        self._closed = True
        if self._own_executor and hasattr(self, 'executor') and self.executor:
            try:
                self.executor.shutdown(wait=True, timeout=10)
            except Exception as e:
                if hasattr(self, 'logger'):
                    self.logger.warning(f"Error shutting down batch executor: {e}")

class DatabaseManager:
    
    def __init__(self, db_type: str = "sqlite", config: Optional[Dict] = None, 
                 logger=None, max_connections_per_db=3, max_workers=8):
        self.db_type = db_type.lower()
        self.config = config or {}
        self.logger = get_logger()
        self.max_connections_per_db = max_connections_per_db
        self.max_workers = max_workers
        
        self.registry = DatabaseRegistry(self.logger)
        self.cache = CacheManager(max_size=100, ttl=1800)
        
        self.connection_pools: Dict[str, ConnectionPool] = {}
        self.pool_lock = threading.RLock()
        
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="DatabaseManager"
        )
        
        self.connectors = {
            'sqlite': SQLiteConnector(),
            'mysql': MySQLConnector()
        }
        
        self.batch_executor = BatchOperationExecutor(
            registry=self.registry,
            cache=self.cache,
            connectors=self.connectors,
            connection_pool_getter=self._get_connection_pool,
            logger=self.logger,
            shared_executor=self.executor
        )
        
        self._cleanup_timer = None
        self._cleanup_lock = threading.RLock()
        self._shutdown_requested = False
        self._initialized = False
        
        try:
            self._validate_config()
            self._discover_databases()
            self._start_cleanup_timer()
            self._initialized = True
            
        except Exception as e:
            self.logger.warning(f"Failed to initialize DatabaseManager: {e}")
            self.close()
            raise

    def _start_cleanup_timer(self):
        with self._cleanup_lock:
            if self._cleanup_timer is None and not self._shutdown_requested:
                self._cleanup_timer = threading.Timer(300, self._periodic_cleanup)
                self._cleanup_timer.daemon = True
                self._cleanup_timer.start()
                self.logger.debug("Cleanup timer started")

    def _periodic_cleanup(self):
        if self._shutdown_requested:
            return
        
        try:
            total_cleaned = 0
            with self.pool_lock:
                for db_id, pool in self.connection_pools.items():
                    cleaned = pool.cleanup_idle_connections()
                    total_cleaned += cleaned
            
            if total_cleaned > 0:
                self.logger.info(f"Cleaned up {total_cleaned} idle connections")
            
            with self._cleanup_lock:
                if not self._shutdown_requested:
                    self._cleanup_timer = threading.Timer(300, self._periodic_cleanup)
                    self._cleanup_timer.daemon = True
                    self._cleanup_timer.start()
                    
        except Exception as e:
            self.logger.warning(f"Error in periodic cleanup: {e}")

    def _get_connection_pool(self, db_id: str) -> ConnectionPool:
        if not self._initialized:
            raise RuntimeError("DatabaseManager not properly initialized")
        
        if self._shutdown_requested:
            raise RuntimeError("DatabaseManager is shutting down")
            
        if db_id not in self.connection_pools:
            with self.pool_lock:
                if db_id not in self.connection_pools:
                    db_info = self.registry.get_database(db_id)
                    if not db_info:
                        raise ValueError(f"Database {db_id} not found")
                    
                    connector = self.connectors.get(db_info.db_type)
                    if not connector:
                        raise ValueError(f"No connector found for database type: {db_info.db_type}")
                    
                    self.connection_pools[db_id] = ConnectionPool(
                        connector=connector,
                        connection_info=db_info.connection_info,
                        max_connections=self.max_connections_per_db
                    )
                    self.logger.debug(f"Created connection pool for database: {db_id}")
        
        return self.connection_pools[db_id]

    def batch_execute_queries(self, queries: List[Dict[str, Any]]) -> List[BatchResult]:
        if not self._initialized:
            raise RuntimeError("DatabaseManager not properly initialized")
        
        if not queries:
            return []
        
        operations = []
        for i, query in enumerate(queries):
            if not isinstance(query, dict):
                raise ValueError(f"Query {i} must be a dictionary")
            
            if not all(key in query for key in ['db_id', 'sql']):
                raise ValueError(f"Query {i} missing required fields: db_id, sql")
            
            op = BatchOperation(
                operation_type=OperationType.EXECUTE_QUERY,
                db_id=query['db_id'],
                sql=query['sql'],
                timeout=query.get('timeout', 30.0),
                operation_id=query.get('operation_id', f"query_{i}")
            )
            operations.append(op)
        
        return self.batch_executor.execute_batch_operations(operations)
    
    def batch_analyze_execution_plans(self, queries: List[Dict[str, Any]]) -> List[BatchResult]:
        if not self._initialized:
            raise RuntimeError("DatabaseManager not properly initialized")
        
        if not queries:
            return []
        
        operations = []
        for i, query in enumerate(queries):
            if not isinstance(query, dict):
                raise ValueError(f"Query {i} must be a dictionary")
            
            if not all(key in query for key in ['db_id', 'sql']):
                raise ValueError(f"Query {i} missing required fields: db_id, sql")
            
            op = BatchOperation(
                operation_type=OperationType.ANALYZE_PLAN,
                db_id=query['db_id'],
                sql=query['sql'],
                timeout=query.get('timeout', 30.0),
                operation_id=query.get('operation_id', f"plan_{i}")
            )
            operations.append(op)
        
        return self.batch_executor.execute_batch_operations(operations)
    
    def batch_validate_sql(self, queries: List[Dict[str, Any]]) -> List[BatchResult]:
        if not self._initialized:
            raise RuntimeError("DatabaseManager not properly initialized")
        
        if not queries:
            return []
        
        operations = []
        for i, query in enumerate(queries):
            if not isinstance(query, dict):
                raise ValueError(f"Query {i} must be a dictionary")
            
            if not all(key in query for key in ['db_id', 'sql']):
                raise ValueError(f"Query {i} missing required fields: db_id, sql")
            
            op = BatchOperation(
                operation_type=OperationType.VALIDATE_SQL,
                db_id=query['db_id'],
                sql=query['sql'],
                timeout=query.get('timeout', 30.0),
                operation_id=query.get('operation_id', f"validate_{i}")
            )
            operations.append(op)
        
        return self.batch_executor.execute_batch_operations(operations)
    
    def batch_compare_sql(self, comparisons: List[Dict[str, Any]]) -> List[BatchResult]:
        if not self._initialized:
            raise RuntimeError("DatabaseManager not properly initialized")
        
        if not comparisons:
            return []
        
        operations = []
        for i, comp in enumerate(comparisons):
            if not isinstance(comp, dict):
                raise ValueError(f"Comparison {i} must be a dictionary")
            
            required_fields = ['db_id', 'sql1', 'sql2']
            if not all(key in comp for key in required_fields):
                raise ValueError(f"Comparison {i} missing required fields: {required_fields}")
            
            op = BatchOperation(
                operation_type=OperationType.COMPARE_SQL,
                db_id=comp['db_id'],
                sql=comp['sql1'],
                timeout=comp.get('timeout', 30.0),
                additional_params={'sql2': comp['sql2']},
                operation_id=comp.get('operation_id', f"compare_{i}")
            )
            operations.append(op)
        
        return self.batch_executor.execute_batch_operations(operations)
    
    # ==================== Single Operation Interface (Retained for Compatibility) ====================
    
    def execute_query(self, db_id: str, sql: str, use_cache: bool = False) -> Dict[str, Any]:
        if not self._initialized:
            raise RuntimeError("DatabaseManager not properly initialized")
        
        try:
            pool = self._get_connection_pool(db_id)
            db_info = self.registry.get_database(db_id)
            connector = self.connectors.get(db_info.db_type)
            
            with pool.get_connection() as conn:
                rows = connector.execute_query(conn, sql)
                
                columns = []
                if rows and isinstance(rows[0], dict):
                    columns = list(rows[0].keys())
                elif rows:
                    columns = [f"col_{i}" for i in range(len(rows[0]))] if rows[0] else []
                
                return {
                    'success': True,
                    'data': rows,
                    'columns': columns,
                    'row_count': len(rows) if rows else 0
                }
                
        except Exception as e:
            self.logger.warning(f"Query execution failed for {db_id}: {e}")
            return {
                'success': False,
                'error': str(e),
                'data': [],
                'columns': [],
                'row_count': 0
            }
    
    # ==================== Schema Related Methods ====================
    
    def get_database_schema(self, db_id: str, use_cache: bool = True) -> Dict[str, Any]:
        if not self._initialized:
            raise RuntimeError("DatabaseManager not properly initialized")
        
        try:
            db_info = self.registry.get_database(db_id)
            if not db_info:
                raise ValueError(f"Database {db_id} not found")
            
            if use_cache:
                cached_schema = self.cache.get_schema(db_id, db_info.db_type)
                if cached_schema:
                    self.logger.debug(f"Using cached schema for database: {db_id}")
                    return cached_schema
            
            pool = self._get_connection_pool(db_id)
            connector = self.connectors.get(db_info.db_type)
            
            schema = {'tables': {}, 'foreign_keys': [], 'primary_keys': []}
            
            with pool.get_connection() as conn:
                tables = connector.get_tables(conn)
                self.logger.debug(f"Found {len(tables)} tables in {db_id}: {tables}")
                
                if not tables:
                    self.logger.warning(f"No tables found in database {db_id}")
                    return schema
                
                for table_name in tables:
                    try:
                        if use_cache:
                            cached_table_info = self.cache.get_table_info(db_id, table_name)
                            if cached_table_info:
                                table_schema = cached_table_info
                            else:
                                table_schema = self._get_fresh_table_schema(conn, connector, table_name)
                                self.cache.set_table_info(db_id, table_name, table_schema)
                        else:
                            table_schema = self._get_fresh_table_schema(conn, connector, table_name)
                        
                        schema['tables'][table_name] = table_schema
                        
                        for pk in table_schema.get('primary_keys', []):
                            schema['primary_keys'].append({
                                'table': table_name,
                                'column': pk
                            })

                        for fk in table_schema.get('foreign_keys', []):
                            schema['foreign_keys'].append({
                                'source_table': table_name,
                                'source_column': fk['column'],
                                'referenced_table': fk['referenced_table'],
                                'referenced_column': fk['referenced_column']
                            })
                            
                    except Exception as table_error:
                        self.logger.warning(f"Error processing table {table_name}: {table_error}")
                        continue
            
            if use_cache and schema['tables']:
                self.cache.set_schema(db_id, db_info.db_type, schema)
                self.logger.debug(f"Cached schema for database: {db_id}")
            
            return schema
                
        except Exception as e:
            self.logger.warning(f"Failed to get schema for {db_id}: {e}")
            return {
                'tables': {},
                'foreign_keys': [],
                'primary_keys': []
            }
    
    def _get_fresh_table_schema(self, conn, connector, table_name: str) -> Dict:
        table_schema = connector.get_table_schema(conn, table_name)
        
        sample_data = connector.get_sample_data(conn, table_name, 2)
        
        for col_name in table_schema.get('columns', {}).keys():
            examples = []
            for row in sample_data.get('rows', []):
                if isinstance(row, dict) and col_name in row and row[col_name] is not None:
                    examples.append(str(row[col_name]))
                elif isinstance(row, (list, tuple)):
                    col_names = sample_data.get('columns', [])
                    if col_name in col_names:
                        col_index = col_names.index(col_name)
                        if col_index < len(row) and row[col_index] is not None:
                            examples.append(str(row[col_index]))
            
            table_schema['columns'][col_name]['examples'] = examples[:3]
        
        return table_schema
    
    def get_table_names_and_create_statements(self, db_id: str) -> Tuple[List[str], List[str]]:
        schema = self.get_database_schema(db_id, use_cache=True)
        if not schema or not schema.get('tables'):
            return [], []
        
        table_names = list(schema['tables'].keys())
        create_statements = self._generate_create_statements(schema)
        return table_names, create_statements
    
    def _generate_create_statements(self, schema: Dict) -> List[str]:
        create_statements = []
        
        for table_name, table_info in schema.get('tables', {}).items():
            columns_def = []
            
            for col_name, col_info in table_info.get('columns', {}).items():
                col_type = col_info.get('type', 'TEXT')
                nullable = col_info.get('nullable', True)
                
                col_def = f'"{col_name}" {col_type}'
                if not nullable:
                    col_def += ' NOT NULL'
                
                columns_def.append(col_def)
            
            primary_keys = table_info.get('primary_keys', [])
            if primary_keys:
                pk_cols = ', '.join([f'"{pk}"' for pk in primary_keys])
                columns_def.append(f'PRIMARY KEY ({pk_cols})')
            
            for fk in table_info.get('foreign_keys', []):
                fk_def = f'FOREIGN KEY ("{fk["column"]}") REFERENCES "{fk["referenced_table"]}"("{fk["referenced_column"]}")'
                columns_def.append(fk_def)
            
            if columns_def:
                create_statement = f'CREATE TABLE "{table_name}" (\n    ' + ',\n    '.join(columns_def) + '\n);'
                create_statements.append(create_statement)
        
        return create_statements

    def get_insert_statements(self, db_id: str, table_names: Optional[List[str]] = None, limit: int = 2) -> Dict[str, List[str]]:
        if not self._initialized:
            return {}
        
        if not db_id:
            return {}
        
        if limit <= 0:
            limit = 2
        
        table_names_key = tuple(sorted(table_names or []))
        cache_key = f"insert_statements_{db_id}_{limit}_{hash(table_names_key)}"
        
        cached_result = self.cache.get_query_result(db_id, cache_key)
        if cached_result:
            return cached_result.get('data', {})
        
        db_info = self.registry.get_database(db_id)
        if not db_info:
            return {}
        
        if table_names is None:
            schema = self.get_database_schema(db_id, use_cache=True)
            table_names = list(schema.get('tables', {}).keys())
        
        if not table_names:
            return {}
        
        connector = self.connectors.get(db_info.db_type)
        if not connector:
            return {}
        
        result = {}
        
        try:
            pool = self._get_connection_pool(db_id)
            with pool.get_connection() as conn:
                for table_name in table_names:
                    try:
                        table_cache_key = f"insert_{db_id}_{table_name}_{limit}"
                        cached_insert = self.cache.get_query_result(db_id, table_cache_key)
                        
                        if cached_insert:
                            result[table_name] = cached_insert.get('data', [])
                        else:
                            sample_data = connector.get_sample_data(conn, table_name, limit)
                            insert_statements = self._generate_insert_statements(
                                table_name, sample_data, db_info.db_type
                            )
                            result[table_name] = insert_statements
                            
                            self.cache.set_query_result(db_id, table_cache_key, {'data': insert_statements})
                    except Exception as e:
                        self.logger.warning(f"Error getting insert statements for table {table_name}: {e}")
                        result[table_name] = []
            
            self.cache.set_query_result(db_id, cache_key, {'data': result})
            
        except Exception as e:
            self.logger.warning(f"Error getting insert statements for {db_id}: {e}")
        
        return result
    
    def _generate_insert_statements(self, table_name: str, sample_data: Dict, db_type: str) -> List[str]:
        if not sample_data.get('rows'):
            return []
        
        statements = []
        columns = sample_data.get('columns', [])
        
        if not columns:
            return []
        
        if db_type == 'mysql':
            table_quote = '`'
            col_quote = '`'
        else:
            table_quote = '"'
            col_quote = '"'
        
        for row in sample_data['rows']:
            if not row:
                continue
                
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
            
            if values:
                columns_str = ', '.join([f'{col_quote}{col}{col_quote}' for col in columns])
                values_str = ', '.join(values)
                
                statement = f'INSERT INTO {table_quote}{table_name}{table_quote} ({columns_str}) VALUES ({values_str});'
                statements.append(statement)
        
        return statements

    # ==================== DDL Generation Methods ====================
    
    def generate_ddl_with_examples(self, db_id: str) -> str:
        schema = self.get_database_schema(db_id, use_cache=True)
        return self._generate_ddl(schema, use_examples=True)
    
    def generate_ddl_without_examples(self, db_id: str) -> str:
        schema = self.get_database_schema(db_id, use_cache=True)
        return self._generate_ddl(schema, use_examples=False)
    
    def generate_formatted_schema_with_examples(self, db_id: str) -> str:
        schema = self.get_database_schema(db_id, use_cache=True)
        return self._generate_formatted_schema(schema, use_examples=True)
    
    def generate_formatted_schema_without_examples(self, db_id: str) -> str:
        schema = self.get_database_schema(db_id, use_cache=True)    
        return self._generate_formatted_schema(schema, use_examples=False)
    
    def _generate_create_statements(self, schema: Dict) -> List[str]:
        statements = []
        
        for table_name, table_info in schema.get('tables', {}).items():
            columns = []
            
            for col_name, col_info in table_info.get('columns', {}).items():
                col_def = f'"{col_name}" {col_info.get("raw_type", "TEXT")}'
                if not col_info.get('nullable', True):
                    col_def += ' NOT NULL'
                if col_info.get('default'):
                    col_def += f' DEFAULT {col_info["default"]}'
                columns.append(col_def)
            
            if table_info.get('primary_keys'):
                pk_cols = ', '.join([f'"{pk}"' for pk in table_info['primary_keys']])
                columns.append(f'PRIMARY KEY ({pk_cols})')
            
            if columns:
                create_sql = f'CREATE TABLE "{table_name}" (\n    ' + ',\n    '.join(columns) + '\n);'
                statements.append(create_sql)
        
        return statements
    
    def _generate_ddl(self, schema: Dict, use_examples: bool = False) -> str:
        if not schema or not schema.get('tables'):
            return ""
        
        ddl_statements = []
        
        for table_name, table_info in schema['tables'].items():
            columns_ddl = []
            
            for col_name, col_info in table_info.get('columns', {}).items():
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
            source_table = schema['tables'].get(fk['source_table'], {})
            table_fks = source_table.get('foreign_keys', [])
            
            if not any(tfk.get('column') == fk['source_column'] and 
                      tfk.get('referenced_table') == fk['referenced_table'] and
                      tfk.get('referenced_column') == fk['referenced_column'] 
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
            for col_name, col_info in table_info.get('columns', {}).items():
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
    

    # ==================== Database Discovery and Management Methods ====================
    
    def _validate_config(self):
        if self.db_type not in self.connectors:
            raise ValueError(f"Unsupported database type: {self.db_type}")
        
        if self.db_type == 'sqlite':
            if 'root_path' not in self.config:
                raise ValueError("SQLite requires 'root_path' in config")
            root_path = self.config['root_path']
            if not os.path.exists(root_path):
                raise ValueError(f"SQLite root path does not exist: {root_path}")
            if not os.path.isdir(root_path):
                raise ValueError(f"SQLite root path is not a directory: {root_path}")
            if not os.access(root_path, os.R_OK):
                raise ValueError(f"SQLite root path is not readable: {root_path}")
                
        elif self.db_type == 'mysql':
            required_params = ['host', 'user', 'password']
            missing = [p for p in required_params if p not in self.config]
            if missing:
                raise ValueError(f"Missing required {self.db_type} config: {missing}")
            
            if 'port' in self.config:
                try:
                    port = int(self.config['port'])
                    if not (1 <= port <= 65535):
                        raise ValueError(f"Invalid port number: {port}")
                except (ValueError, TypeError):
                    raise ValueError(f"Port must be a valid integer: {self.config['port']}")
    
    def _discover_databases(self):
        self.logger.info(f"Discovering {self.db_type} databases")
        try:
            if self.db_type == 'sqlite':
                self._discover_sqlite_databases()
            elif self.db_type == 'mysql':
                self._discover_mysql_databases()
            else:
                self.logger.warning(f"Database discovery not implemented for type: {self.db_type}")
        except Exception as e:
            self.logger.warning(f"Error discovering databases: {e}")
            raise
    
    def _discover_sqlite_databases(self):
        root_path = self.config['root_path']
        extensions = ['*.sqlite', '*.sqlite3', '*.db']
        discovered_count = 0
        
        try:
            all_files = []
            for ext in extensions:
                pattern = os.path.join(root_path, '**', ext)
                files = glob.glob(pattern, recursive=True)
                all_files.extend(files)
            
            all_files = list(set(all_files))
            
            for db_file in tqdm(all_files, desc="Discovering SQLite databases"):
                if os.path.isfile(db_file):
                    try:
                        relative_path = os.path.relpath(db_file, root_path)
                        db_id = os.path.splitext(relative_path.replace(os.sep, '_'))[0]
                            
                        original_db_id = db_id
                        counter = 1
                        while self.registry.database_exists(db_id):
                            db_id = f"{original_db_id}_{counter}"
                            counter += 1
                            
                        if self._validate_sqlite_file(db_file):
                            db_info = DatabaseInfo(
                                db_id=db_id,
                                db_type='sqlite',
                                connection_info={'path': db_file},
                                metadata={
                                    'size': os.path.getsize(db_file),
                                    'modified_time': os.path.getmtime(db_file),
                                    'relative_path': relative_path
                                }
                            )
                            self.registry.register_database(db_info)
                            discovered_count += 1
                            self.logger.debug(f"Registered SQLite database: {db_id} -> {db_file}")
                        else:
                            self.logger.debug(f"Skipped invalid SQLite file: {db_file}")
                                
                    except Exception as e:
                        self.logger.warning(f"Failed to register database {db_file}: {e}")
            
            self.logger.info(f"Discovered {discovered_count} SQLite databases from {len(all_files)} files")
            
        except Exception as e:
            self.logger.warning(f"Error during SQLite database discovery: {e}")
            raise
    
    def _validate_sqlite_file(self, file_path: str) -> bool:
        try:
            if os.path.getsize(file_path) < 100:
                return False
            
            with open(file_path, 'rb') as f:
                header = f.read(16)
                return header.startswith(b'SQLite format 3\x00')
        except Exception:
            return False
    
    def _discover_mysql_databases(self):
        connector = self.connectors['mysql']
        discovered_count = 0
        
        try:
            temp_config = {k: v for k, v in self.config.items() if k != 'database'}
            
            self.logger.debug("Testing MySQL connection...")
            conn = connector.connect(temp_config)
            
            try:
                test_result = connector.execute_query(conn, "SELECT 1 as test")
                if not test_result or test_result[0][0] != 1:
                    raise Exception("MySQL connection test failed")
                
                databases = connector.execute_query(conn, "SHOW DATABASES")
                system_dbs = {'information_schema', 'mysql', 'performance_schema', 'sys'}
                
                for (db_name,) in databases:
                    if db_name not in system_dbs:
                        try:
                            if self._test_mysql_database_access(connector, temp_config, db_name):
                                db_info = DatabaseInfo(
                                    db_id=db_name,
                                    db_type='mysql',
                                    connection_info={**self.config, 'database': db_name},
                                    metadata={
                                        'host': self.config.get('host'),
                                        'port': self.config.get('port', 3306)
                                    }
                                )
                                self.registry.register_database(db_info)
                                discovered_count += 1
                                self.logger.debug(f"Registered MySQL database: {db_name}")
                            else:
                                self.logger.debug(f"Skipped inaccessible MySQL database: {db_name}")
                                
                        except Exception as e:
                            self.logger.warning(f"Failed to register MySQL database {db_name}: {e}")
                
                self.logger.info(f"Discovered {discovered_count} MySQL databases")
                
            finally:
                try:
                    conn.close()
                except Exception as e:
                    self.logger.warning(f"Error closing MySQL discovery connection: {e}")
                
        except Exception as e:
            self.logger.warning(f"Error discovering MySQL databases: {e}")
            raise
    
    def _test_mysql_database_access(self, connector, base_config: dict, db_name: str) -> bool:
        try:
            test_config = {**base_config, 'database': db_name}
            test_conn = connector.connect(test_config)
            try:
                connector.execute_query(test_conn, "SELECT 1")
                return True
            finally:
                test_conn.close()
        except Exception:
            return False
    
    # ==================== Single Operation Interface (Uniform Implementation) ====================
    
    def _execute_single_operation_template(self, 
                                         operation_type: OperationType,
                                         db_id: str, 
                                         sql: str, 
                                         timeout: float = 2.0,
                                         additional_params: Optional[Dict] = None) -> Dict[str, Any]:
        if not self._initialized:
            return self._create_error_response("DatabaseManager not properly initialized")
        
        if not db_id or not sql:
            return self._create_error_response("db_id and sql are required")
        
        operation = BatchOperation(
            operation_type=operation_type,
            db_id=db_id,
            sql=sql,
            timeout=timeout,
            additional_params=additional_params
        )
        
        try:
            results = self.batch_executor.execute_batch_operations([operation])
            
            if results and results[0].success:
                return results[0].data
            else:
                error_msg = results[0].error if results and results[0].error else 'Unknown error'
                return self._create_error_response(error_msg)
        except Exception as e:
            self.logger.warning(f"Error executing {operation_type.value} for {db_id}: {e}")
            return self._create_error_response(str(e))

    def execute_query(self, db_id: str, sql: str, timeout: float = 2.0) -> Dict[str, Any]:
        return self._execute_single_operation_template(OperationType.EXECUTE_QUERY, db_id, sql, timeout)
    
    def analyze_sql_execution_plan(self, db_id: str, sql: str, timeout: float = 2.0) -> Dict[str, Any]:
        result = self._execute_single_operation_template(OperationType.ANALYZE_PLAN, db_id, sql, timeout)
        if result.get('success'):
            return {'success': True, **result}
        return {'success': False, 'error': result.get('error', 'Unknown error')}
    
    def validate_sql(self, db_id: str, sql: str, timeout: float = 2.0) -> bool:
        result = self._execute_single_operation_template(OperationType.VALIDATE_SQL, db_id, sql, timeout)
        return result.get('valid', False) if result.get('success') else False
    
    def compare_sql(self, db_id: str, sql1: str, sql2: str, timeout: float = 2.0) -> bool:
        result = self._execute_single_operation_template(
            OperationType.COMPARE_SQL, db_id, sql1, timeout, {'sql2': sql2}
        )
        return result.get('equal', False) if result.get('success') else False
    
    # ==================== Public Interface Methods ====================
    
    def list_databases(self) -> List[str]:
        if not self._initialized:
            self.logger.warning("DatabaseManager not initialized")
            return []
        try:
            return self.registry.list_databases()
        except Exception as e:
            self.logger.warning(f"Error listing databases: {e}")
            return []
    
    def database_exists(self, db_id: str) -> bool:
        if not self._initialized:
            return False
        if not db_id:
            return False
        try:
            return self.registry.database_exists(db_id)
        except Exception as e:
            self.logger.warning(f"Error checking database existence for {db_id}: {e}")
            return False
    
    def get_database_info(self, db_id: str) -> Optional[Dict[str, Any]]:
        if not self._initialized:
            return None
        
        try:
            db_info = self.registry.get_database(db_id)
            if not db_info:
                return None
            
            return {
                'db_id': db_info.db_id,
                'db_type': db_info.db_type,
                'metadata': db_info.metadata,
                'connection_info': {k: v for k, v in db_info.connection_info.items() 
                                  if k not in ['password', 'passwd']}
            }
        except Exception as e:
            self.logger.warning(f"Error getting database info for {db_id}: {e}")
            return None
    
    def refresh_database_registry(self):
        if not self._initialized:
            raise RuntimeError("DatabaseManager not properly initialized")
        
        try:
            self.logger.info("Refreshing database registry...")
            
            self.registry.clear()
            self.cache.clear_all()
            
            with self.pool_lock:
                for db_id, pool in self.connection_pools.items():
                    try:
                        pool.close_all()
                    except Exception as e:
                        self.logger.warning(f"Error closing pool for {db_id}: {e}")
                self.connection_pools.clear()
            
            self._discover_databases()
            
            self.logger.info("Database registry refreshed successfully")
            
        except Exception as e:
            self.logger.warning(f"Error refreshing database registry: {e}")
            raise
    
    def clear_cache(self):
        try:
            self.cache.clear_all()
            self.logger.info("Cache cleared successfully")
        except Exception as e:
            self.logger.warning(f"Error clearing cache: {e}")
            raise

    def close(self):
        if self._shutdown_requested:
            self.logger.debug("DatabaseManager already shutting down")
            return
        
        self._shutdown_requested = True
        self.logger.info("Shutting down DatabaseManager...")
        
        try:
            with self._cleanup_lock:
                if self._cleanup_timer is not None:
                    try:
                        self._cleanup_timer.cancel()
                        self.logger.debug("Cleanup timer cancelled")
                    except Exception as e:
                        self.logger.warning(f"Error cancelling cleanup timer: {e}")
                    finally:
                        self._cleanup_timer = None
            
            if hasattr(self.batch_executor, 'close'):
                try:
                    self.batch_executor.close()
                    self.logger.debug("Batch executor closed")
                except Exception as e:
                    self.logger.warning(f"Error closing batch executor: {e}")
            
            with self.pool_lock:
                for db_id, pool in self.connection_pools.items():
                    try:
                        pool.close_all()
                        self.logger.debug(f"Closed connection pool for {db_id}")
                    except Exception as e:
                        self.logger.warning(f"Error closing connection pool for {db_id}: {e}")
                self.connection_pools.clear()
            
            if hasattr(self, 'executor') and self.executor:
                try:
                    self.executor.shutdown(wait=False)
                    
                    start_time = time.time()
                    while not self.executor._shutdown and time.time() - start_time < 10:
                        time.sleep(0.1)
                    
                    if not self.executor._shutdown:
                        self.logger.warning("Thread pool did not shut down gracefully, forcing shutdown")
                        self.executor.shutdown(wait=True)
                    
                    self.logger.debug("Thread pool shut down successfully")
                    
                except Exception as e:
                    self.logger.warning(f"Error shutting down thread pool: {e}")
            
            try:
                self.cache.clear_all()
                self.logger.debug("Cache cleared")
            except Exception as e:
                self.logger.warning(f"Error clearing cache: {e}")
            
            try:
                self.registry.clear()
                self.logger.debug("Registry cleared")
            except Exception as e:
                self.logger.warning(f"Error clearing registry: {e}")
            
            self._initialized = False
            self.logger.info("DatabaseManager shutdown completed successfully")
            
        except Exception as e:
            self.logger.warning(f"Error during shutdown: {e}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        if not self._initialized:
            return {'error': 'DatabaseManager not initialized', 'initialized': False}
        
        try:
            cache_stats = self.cache.get_cache_stats()
            
            pool_stats = {}
            total_connections = 0
            with self.pool_lock:
                for db_id, pool in self.connection_pools.items():
                    try:
                        stats = pool.get_stats()
                        pool_stats[db_id] = stats
                        total_connections += stats.get('active_connections', 0)
                    except Exception as e:
                        pool_stats[db_id] = {'error': str(e)}
            
            thread_pool_stats = {
                'max_workers': self.max_workers,
                'active_threads': getattr(self.executor, '_threads', 0) if hasattr(self.executor, '_threads') else 'unknown'
            }
            
            database_count = len(self.list_databases())
            
            return {
                'initialized': self._initialized,
                'shutdown_requested': self._shutdown_requested,
                'database_count': database_count,
                'db_type': self.db_type,
                'max_workers': self.max_workers,
                'max_connections_per_db': self.max_connections_per_db,
                'total_active_connections': total_connections,
                'cache_stats': cache_stats,
                'connection_pools': pool_stats,
                'active_pools': len(self.connection_pools),
                'thread_pool_stats': thread_pool_stats,
                'cleanup_timer_active': self._cleanup_timer is not None and self._cleanup_timer.is_alive()
            }
        except Exception as e:
            self.logger.warning(f"Error getting stats: {e}")
            return {'error': str(e), 'initialized': self._initialized}
    
    def health_check(self) -> Dict[str, Any]:
        health_status = {
            'status': 'healthy',
            'timestamp': time.time(),
            'checks': {}
        }
        
        try:
            health_status['checks']['initialization'] = {
                'status': 'pass' if self._initialized else 'fail',
                'message': 'System initialized' if self._initialized else 'System not initialized'
            }
            
            try:
                db_count = len(self.list_databases())
                health_status['checks']['database_registry'] = {
                    'status': 'pass',
                    'message': f'{db_count} databases registered',
                    'count': db_count
                }
            except Exception as e:
                health_status['checks']['database_registry'] = {
                    'status': 'fail',
                    'message': f'Registry error: {str(e)}'
                }
            
            try:
                if hasattr(self.executor, '_shutdown') and self.executor._shutdown:
                    health_status['checks']['thread_pool'] = {
                        'status': 'fail',
                        'message': 'Thread pool is shut down'
                    }
                else:
                    health_status['checks']['thread_pool'] = {
                        'status': 'pass',
                        'message': f'Thread pool operational (max workers: {self.max_workers})'
                    }
            except Exception as e:
                health_status['checks']['thread_pool'] = {
                    'status': 'fail',
                    'message': f'Thread pool error: {str(e)}'
                }
            
            try:
                with self.pool_lock:
                    pool_count = len(self.connection_pools)
                    health_status['checks']['connection_pools'] = {
                        'status': 'pass',
                        'message': f'{pool_count} connection pools active',
                        'count': pool_count
                    }
            except Exception as e:
                health_status['checks']['connection_pools'] = {
                    'status': 'fail',
                    'message': f'Connection pool error: {str(e)}'
                }
            
            try:
                cache_stats = self.cache.get_cache_stats()
                total_size = cache_stats.get('total_size', 0)
                health_status['checks']['cache'] = {
                    'status': 'pass',
                    'message': f'Cache operational (size: {total_size})',
                    'size': total_size
                }
            except Exception as e:
                health_status['checks']['cache'] = {
                    'status': 'fail',
                    'message': f'Cache error: {str(e)}'
                }
            
            timer_status = (self._cleanup_timer is not None and 
                          self._cleanup_timer.is_alive() and 
                          not self._shutdown_requested)
            
            health_status['checks']['cleanup_timer'] = {
                'status': 'pass' if timer_status else 'warn',
                'message': 'Cleanup timer running' if timer_status else 'Cleanup timer not running'
            }
            
            failed_checks = [name for name, check in health_status['checks'].items() 
                           if check['status'] == 'fail']
            warning_checks = [name for name, check in health_status['checks'].items() 
                            if check['status'] == 'warn']
            
            if failed_checks:
                health_status['status'] = 'unhealthy'
                health_status['failed_checks'] = failed_checks
            elif warning_checks:
                health_status['status'] = 'degraded'
                health_status['warning_checks'] = warning_checks
                
        except Exception as e:
            health_status['status'] = 'unhealthy'
            health_status['error'] = str(e)
            self.logger.warning(f"Health check failed: {e}")
        
        return health_status
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def __del__(self):
        try:
            if hasattr(self, '_initialized') and self._initialized and not self._shutdown_requested:
                self.close()
        except Exception:
            pass