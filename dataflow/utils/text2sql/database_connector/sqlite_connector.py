from typing import Dict, Any, Optional, Tuple
from ..base import DatabaseConnectorABC, DatabaseInfo, QueryResult
import sqlite3
import glob
import os
import time

# ============== SQLite Connector ==============
class SQLiteConnector(DatabaseConnectorABC):
    
    def connect(self, connection_info: Dict) -> sqlite3.Connection:
        db_path = connection_info['path']
        conn = sqlite3.connect(
            db_path,
            timeout=30,
            check_same_thread=False  
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA query_only = ON")
        return conn
    
    def execute_query(self, connection: sqlite3.Connection, sql: str, params: Optional[Tuple] = None) -> QueryResult:
        start_time = time.time()
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
            self.logger.error(f"Query execution error: {e}")
            return QueryResult(
                success=False,
                error=str(e)
            )
        finally:
            if cursor:
                cursor.close()
    
    def get_schema_info(self, connection: sqlite3.Connection) -> Dict[str, Any]:
        """Get complete schema information for SQLite database"""
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
        
        return schema

    def generate_create_statement(self, table_name: str, table_info: Dict[str, Any]) -> str:
        """Generate CREATE TABLE statements from schema"""
        create_stmt = ""

        columns = []
        for col_name, col_info in table_info.get('columns', {}).items():
            col_def = f"`{col_name}` {col_info['type']}"
            if not col_info.get('nullable', True):
                col_def += " NOT NULL"
            if col_info.get('default') is not None:
                col_def += f" DEFAULT {col_info['default']}"
            columns.append(col_def)
            
        # Primary keys
        pk_cols = table_info.get('primary_keys', [])
        if pk_cols:
            columns.append(f"PRIMARY KEY ({', '.join(f'`{pk}`' for pk in pk_cols)})")
            
        # Foreign keys
        for fk in table_info.get('foreign_keys', []):
            fk_def = (f"FOREIGN KEY (`{fk['column']}`) "
                        f"REFERENCES `{fk['referenced_table']}`(`{fk['referenced_column']}`)")
            columns.append(fk_def)
            
        if columns:
            create_stmt = f"CREATE TABLE `{table_name}` (\n  " + ",\n  ".join(columns) + "\n);"
        
        return create_stmt
    
    def _get_table_info(self, connection: sqlite3.Connection, table_name: str) -> Dict[str, Any]:
        """Get detailed table information"""
        # The info must include the following six fields: columns, primary_keys, foreign_keys, sample_data, create_statement, insert_statement
        table_info = {
            'columns': {},
            'primary_keys': [],
            'foreign_keys': [],
            'sample_data': [],
            'create_statement': None,
            'insert_statement': None
        }
        
        # Get columns information
        result = self.execute_query(connection, f"PRAGMA table_info([{table_name}])")
        if result.success:
            for col in result.data:
                col_name = col['name']
                table_info['columns'][col_name] = {
                    'type': col['type'],
                    'nullable': not col['notnull'],
                    'default': col['dflt_value'],
                    'primary_key': bool(col['pk'])
                }
                if col['pk']:
                    table_info['primary_keys'].append(col_name)
        
        # Get foreign keys
        result = self.execute_query(connection, f"PRAGMA foreign_key_list([{table_name}])")
        if result.success:
            for fk in result.data:
                table_info['foreign_keys'].append({
                    'column': fk['from'],
                    'referenced_table': fk['table'],
                    'referenced_column': fk['to']
                })
        
        # Get sample data from the table
        result = self.execute_query(connection, f"SELECT * FROM [{table_name}]")
        if result.success and result.data:
            table_info['sample_data'] = result.data

        # Get create statement from the table
        table_info['create_statement'] = self.generate_create_statement(table_name, table_info)

        # Generate insert statements from sample data from the table
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
        """Discover SQLite database files"""
        databases = {}
        root_path = config.get('root_path', '.')
        patterns = config.get('patterns', ['*.sqlite', '*.sqlite3', '*.db'])
        
        for pattern in patterns:
            for db_path in glob.glob(os.path.join(root_path, '**', pattern), recursive=True):
                if os.path.isfile(db_path):
                    rel_path = os.path.relpath(db_path, root_path)
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