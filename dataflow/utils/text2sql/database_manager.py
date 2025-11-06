from typing import Dict, List, Optional, Any, Tuple
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
import threading
import hashlib
import time
import os
from dataflow import get_logger
from .base import DatabaseInfo, QueryResult
from .database_connector.sqlite_connector import SQLiteConnector
from .database_connector.sqlite_vec_connector import SQLiteVecConnector
from .database_connector.mysql_connector import MySQLConnector


# ============== Cache Manager ==============
class CacheManager:
    
    def __init__(self, max_size: int = 100, ttl: int = 1800):
        self.max_size = max_size
        self.ttl = ttl
        self._cache = {}
        self._timestamps = {}
        self._lock = threading.RLock()
        self._last_cleanup = time.time() 
    
    def _make_key(self, *args) -> str:
        """Generate cache key from arguments"""
        key_str = '|'.join(str(arg) for arg in args)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, *args) -> Optional[Any]:
        """Get value from cache"""
        key = self._make_key(*args)
        with self._lock:
            current_time = time.time()
            if current_time - self._last_cleanup > 300:  
                self._cleanup_expired()
                self._last_cleanup = current_time
            
            if key in self._cache:
                if current_time - self._timestamps[key] < self.ttl:
                    return self._cache[key]
                else:
                    del self._cache[key]
                    del self._timestamps[key]
        return None
    
    def set(self, value: Any, *args):
        """Set value in cache"""
        key = self._make_key(*args)
        with self._lock:
            if len(self._cache) >= self.max_size:
                items_to_remove = len(self._cache) - self.max_size + 1
                oldest = sorted(self._timestamps.items(), key=lambda x: x[1])[:items_to_remove]
                for old_key, _ in oldest:
                    del self._cache[old_key]
                    del self._timestamps[old_key]
            
            self._cache[key] = value
            self._timestamps[key] = time.time()
    
    def clear(self):
        """Clear all cache entries"""
        with self._lock:
            self._cache.clear()
            self._timestamps.clear()
    
    def _cleanup_expired(self):
        """Clean up expired cache entries"""
        current_time = time.time()
        expired_keys = [
            key for key, timestamp in self._timestamps.items()
            if current_time - timestamp >= self.ttl
        ]
        for key in expired_keys:
            del self._cache[key]
            del self._timestamps[key]


# ============== Database Manager ==============
class DatabaseManager:
    
    # Registry of available connectors
    # You must add the database connector class here if you want to support a new database type
    CONNECTORS = {
        'sqlite': SQLiteConnector,
        'mysql': MySQLConnector,
        'sqlite-vec': SQLiteVecConnector,
        # Add new database types here
        # 'postgres': PostgresConnector
    }
    
    def __init__(self, db_type: str = "sqlite", config: Optional[Dict] = None):
        self.db_type = db_type.lower()
        self.config = config or {}

        self.logger = get_logger()
        self.max_connections_per_db = 100
        self.max_workers = min(64, max(32, os.cpu_count()))
        self.query_timeout = 5
        
        if self.db_type not in self.CONNECTORS:
            raise ValueError(f"Unsupported database type: {self.db_type}")
        
        self.connector = self.CONNECTORS[self.db_type]()
        self.databases = {}
        self.cache = CacheManager()
        
        self._discover_databases()


    # ============== Database Discovery ==============

    def _discover_databases(self):
        """Traverse to discover all available databases"""
        self.databases = self.connector.discover_databases(self.config)


    # ============== Connection Management ==============
    
    @contextmanager
    def get_connection(self, db_id: str):
        if db_id not in self.databases:
            raise ValueError(f"Database '{db_id}' not found")
        
        conn = None
        try:
            conn = self.connector.connect(self.databases[db_id].connection_info)
            yield conn
        except Exception as e:
            self.logger.error(f"Connection error for database '{db_id}': {e}")
            raise
        finally:
            if conn:
                try:
                    self.connector.close(conn)
                except Exception as e:
                    self.logger.error(f"Error closing connection: {e}")
    
    # ============== SQL Execution ==============

    def execute_query(self, db_id: str, sql: str, params: Optional[Tuple] = None) -> QueryResult:
        """Query execution with timeout control"""
        if not sql or not sql.strip():
            return QueryResult(success=False, error="Query cannot be empty")

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self._execute_query_sync, db_id, sql, params)
            try:
                return future.result(timeout=self.query_timeout)
            except TimeoutError:
                return QueryResult(success=False, error=f"Query timeout after {self.query_timeout}s")

    def _execute_query_sync(self, db_id: str, sql: str, params: Optional[Tuple] = None) -> QueryResult:
        try:
            with self.get_connection(db_id) as conn:
                return self.connector.execute_query(conn, sql, params)
        except Exception as e:
            return QueryResult(success=False, error=str(e))
    
    def batch_compare_queries(self, query_triples: List[Tuple[str, str, str]]) -> List[Dict[str, Any]]:
        """
        Compare multiple pairs of queries across different databases in parallel.
        """
        unique_db_ids = set(db_id for db_id, _, _ in query_triples)
        for db_id in unique_db_ids:
            if not self.database_exists(db_id):
                error_msg = f"Database '{db_id}' not found"
                return [self._create_error_result(error_msg) for _ in query_triples]
        
        # Flatten all queries for batch execution
        all_queries = []
        query_indices = []
        
        for idx, (db_id, gold_sql, pred_sql) in enumerate(query_triples):
            all_queries.extend([
                (db_id, gold_sql),
                (db_id, pred_sql)
            ])
            query_indices.extend([idx, idx])
        
        # Batch execute all queries
        all_results = self.batch_execute_queries(all_queries)
        
        # Group results by comparison pairs
        comparisons = []
        for i in range(0, len(query_triples)):
            result1 = None
            result2 = None
            
            for j in range(len(query_indices)):
                if query_indices[j] == i:
                    if result1 is None:
                        result1 = all_results[j]
                    else:
                        result2 = all_results[j]
                        break
            
            if result1 is not None and result2 is not None:
                comparison = self.compare_results(result1, result2)
                comparisons.append(comparison)
            else:
                comparisons.append(self._create_error_result("Internal error: failed to match query results"))
        
        return comparisons

    def batch_execute_queries(self, queries: List[Tuple[str, str]]) -> List[QueryResult]:
        results = [QueryResult(success=False, error="Not executed") for _ in queries]
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            semaphore = threading.Semaphore(self.max_connections_per_db)
            
            def execute_with_limit(idx, db_id, sql):
                with semaphore:
                    try:
                        with self.get_connection(db_id) as conn:
                            result = self.connector.execute_query(conn, sql)
                            results[idx] = result
                    except Exception as e:
                        results[idx] = QueryResult(
                            success=False, 
                            error=f"{type(e).__name__}: {str(e)}"
                        )
            
            for idx, (db_id, sql) in enumerate(queries):
                futures[executor.submit(execute_with_limit, idx, db_id, sql)] = idx
            
            try:
                for future in as_completed(futures, timeout=self.query_timeout*len(queries)):
                    future.result()
            except TimeoutError:
                self.logger.warning("Batch execution timed out")
        
        return results

    def compare_queries(self, db_id: str, sql1: str, sql2: str) -> Dict[str, Any]:
        """Compare results of two SQL queries"""
        results = self.batch_execute_queries([(db_id, sql1), (db_id, sql2)])
        return self.compare_results(results[0], results[1])


    def compare_results(self, result1: QueryResult, result2: QueryResult) -> Dict[str, Any]:
        """Compare two query results"""
        comparison = {
            'equal': False,
            'differences': [],
            'result1_success': result1.success,
            'result2_success': result2.success,
        }
        
        if not result1.success or not result2.success:
            if not result1.success:
                comparison['differences'].append(f'Query 1 failed: {result1.error}')
            if not result2.success:
                comparison['differences'].append(f'Query 2 failed: {result2.error}')
            return comparison
        
        # Compare row counts
        if result1.row_count != result2.row_count:
            comparison['differences'].append(
                f'Row count mismatch: {result1.row_count} vs {result2.row_count}'
            )
            return comparison
        
        # Compare columns
        if set(result1.columns) != set(result2.columns):
            comparison['differences'].append(
                f'Column mismatch: {result1.columns} vs {result2.columns}'
            )
            return comparison
        
        # Compare data - use a simpler approach that avoids None comparison issues
        def normalize_row(row):
            """Normalize row data, converting None to empty string to avoid comparison issues"""
            normalized_items = []
            for k, v in row.items():
                # Convert None to empty string to avoid comparison issues
                normalized_value = "" if v is None else str(v)
                normalized_items.append((k, normalized_value))
            return tuple(sorted(normalized_items))
        
        data1 = sorted([normalize_row(row) for row in result1.data])
        data2 = sorted([normalize_row(row) for row in result2.data])
        
        if data1 == data2:
            comparison['equal'] = True
        else:
            for i, (row1, row2) in enumerate(zip(data1, data2)):
                if row1 != row2:
                    comparison['differences'].append(f'Row {i} differs: {dict(row1)} vs {dict(row2)}')
                    if len(comparison['differences']) >= 10:
                        comparison['differences'].append('... and more differences')
                        break
        
        return comparison


    # ============== Schema Generation ==============
    # In our application case, only create_statements and insert_statements are needed
    # You can add more schema generation methods here if needed
    
    def _get_schema(self, db_id: str) -> Dict[str, Any]:
        schema_cache = self.cache.get('schema', db_id)
        if schema_cache:
            return schema_cache
        
        with self.get_connection(db_id) as conn:
            schema = self.connector.get_schema_info(conn)
            self.cache.set(schema, 'schema', db_id)
            return schema

    def _get_create_statements(self, schema: Dict[str, Any]) -> List[str]:
        create_statement_list = [table_info['create_statement'] for table_info in schema['tables'].values()]
        return create_statement_list

    def _get_insert_statements(self, schema: Dict[str, Any]) -> List[str]:
        insert_statement_list = []
        for table_info in schema['tables'].values():
            insert_statements_for_table = table_info.get('insert_statement')
            if insert_statements_for_table:
                insert_statement_list.extend(insert_statements_for_table)
                
        return insert_statement_list

    def get_create_statements_and_insert_statements(self, db_id: str) -> tuple:
        if not self.database_exists(db_id):
            raise ValueError(f"Database '{db_id}' not found")

        schema = self._get_schema(db_id)
        create_statements = self._get_create_statements(schema)
        insert_statements = self._get_insert_statements(schema)
        return (create_statements, insert_statements)

    def get_db_details(self, db_id: str) -> str:
        if not self.database_exists(db_id):
            raise ValueError(f"Database '{db_id}' not found")
        schema = self._get_schema(db_id)
        return schema['db_details']


    # ============== Utility Methods ==============
    
    def list_databases(self) -> List[str]:
        """List all available databases"""
        return list(self.databases.keys())
    
    def get_database_info(self, db_id: str) -> Optional[DatabaseInfo]:
        """Get database information"""
        return self.databases.get(db_id)
    
    def database_exists(self, db_id: str) -> bool:
        """Check if database exists"""
        return db_id in self.databases
    
    def get_table_names(self, db_id: str) -> List[str]:
        """Get list of table names in database"""
        schema = self.get_schema(db_id)
        return list(schema.get('tables', {}).keys())
    
    def get_table_info(self, db_id: str, table_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific table"""
        schema = self.get_schema(db_id)
        return schema.get('tables', {}).get(table_name)


    def _create_error_result(self, error_msg: str) -> Dict[str, Any]:
        """Create standardized error result"""
        return {
            'equal': False,
            'differences': [error_msg],
            'result1_success': False,
            'result2_success': False,
        }
    
    def get_number_of_special_column(self, db_id):
        """get the number of secial column"""
        with self.get_connection(db_id) as conn:
            return self.connector.get_number_of_special_column(conn)