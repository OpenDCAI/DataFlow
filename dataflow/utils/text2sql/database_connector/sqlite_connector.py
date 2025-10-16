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
                formatted_params = []
                for param in params:
                    if param is None:
                        formatted_params.append(None)
                    else:
                        formatted_params.append(str(param))
                cursor.execute(sql, tuple(formatted_params))
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
    
    def get_schema_info(self, connection: sqlite3.Connection) -> Dict[str, Any]:
        """Get complete schema information with formatted DDL"""
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

        schema['db_details'] = self._get_db_details(schema)
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
                    'primary_key': bool(col['pk'])
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
        
        return table_info

    def _get_db_details(self, schema: Dict[str, Any]) -> str:
        """Generate formatted DDL statements from schema information"""
        db_details = []
        
        for table_name, table_info in schema['tables'].items():
            column_defs = []
            for col_name, col_info in table_info['columns'].items():
                col_def = f"    `{col_name}` {col_info['type']}"
                
                if not col_info['nullable']:
                    col_def += " NOT NULL"
                    
                if col_info['default'] is not None:
                    col_def += f" DEFAULT {col_info['default']}"
                    
                column_defs.append(col_def)
            
            constraints = []
            
            if table_info['primary_keys']:
                pk_cols = ", ".join(f"`{col}`" for col in table_info['primary_keys'])
                constraints.append(f"    PRIMARY KEY ({pk_cols})")
            
            for i, fk in enumerate(table_info['foreign_keys']):
                constraints.append(
                    f"    CONSTRAINT fk_{table_name}_{fk['column']}_{i} "
                    f"FOREIGN KEY (`{fk['column']}`) "
                    f"REFERENCES `{fk['referenced_table']}` (`{fk['referenced_column']}`)"
                )
            
            create_table = f"CREATE TABLE `{table_name}` (\n"
            create_table += ",\n".join(column_defs + constraints)
            create_table += "\n);"
            
            if table_info['sample_data']:
                create_table += "\n\n-- Sample data:\n"
                create_table += "\n".join(table_info['insert_statement'])
            
            db_details.append(create_table)
        
        return "\n\n".join(db_details)

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
                columns = table_info.get('columns', [])
                
                # Directly traverse the list of column names
                for column_name in columns:
                    # Check if the variable is a string
                    if isinstance(column_name, str):
                        count += 1
                        
        except Exception as e:
            print(f"error: Error counting embedding columns: {e}")

        if count == 0:
                print(f"error: No columns ending with '_embedding'.")
        return count   
