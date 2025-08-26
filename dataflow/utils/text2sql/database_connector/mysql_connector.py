from typing import Dict, Any, Optional, Tuple
from ..base import DatabaseConnectorABC, DatabaseInfo, QueryResult
import pymysql
import pymysql.cursors
import time
import logging

# ============== MySQL Connector ==============
class MySQLConnector(DatabaseConnectorABC):
    """MySQL database connector implementation"""
    
    def __init__(self):
        """Initialize MySQL connector with logger"""
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def connect(self, connection_info: Dict) -> pymysql.Connection:
        config = {
            'connect_timeout': 10,
            'read_timeout': 30,
            'write_timeout': 30,
            'charset': 'utf8mb4',
            'autocommit': False,  
            'cursorclass': pymysql.cursors.DictCursor,
            **connection_info
        }
        
        conn = pymysql.connect(**config)
        with conn.cursor() as cursor:
            cursor.execute("SET SESSION wait_timeout = 300")
            cursor.execute("SET SESSION TRANSACTION READ ONLY") # read only transaction
        return conn
    
    def execute_query(self, connection: pymysql.Connection, sql: str, params: Optional[Tuple] = None) -> QueryResult:
        start_time = time.time()
        cursor = None
        try:
            cursor = connection.cursor()
            if params:
                cursor.execute(sql, params)
            else:
                cursor.execute(sql)
            
            sql_upper = sql.strip().upper()
            if sql_upper.startswith(('SELECT', 'SHOW', 'DESCRIBE', 'EXPLAIN')):
                rows = cursor.fetchall()
                columns = [desc[0] for desc in cursor.description] if cursor.description else []
                data = list(rows) if rows else []
            else:
                raise Exception("Write operations are not allowed in read-only mode")
            
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
    
    def get_schema_info(self, connection: pymysql.Connection) -> Dict[str, Any]:
        """Get complete schema information for MySQL database"""
        schema = {'tables': {}}
        
        # Get current database
        result = self.execute_query(connection, "SELECT DATABASE()")
        if not result.success or not result.data:
            return schema
        
        db_name = list(result.data[0].values())[0]
        if not db_name:
            self.logger.warning("No database selected")
            return schema
        
        # Get all tables from the database
        result = self.execute_query(connection, "SHOW TABLES")
        if not result.success:
            return schema
        
        for row in result.data:
            table_name = list(row.values())[0]  
            table_info = self._get_table_info(connection, db_name, table_name)
            schema['tables'][table_name] = table_info
        
        return schema

    def generate_create_statement(self, table_name: str, table_info: Dict[str, Any]) -> str:
        """Generate CREATE TABLE statements from schema"""
        if table_info.get('create_statement'):
            return table_info['create_statement']
        
        columns = []
        for col_name, col_info in table_info.get('columns', {}).items():
            col_def = f"`{col_name}` {col_info['type']}"
            if not col_info.get('nullable', True):
                col_def += " NOT NULL"
            if col_info.get('default') is not None:
                default_val = col_info['default']
                if isinstance(default_val, str) and default_val.upper() not in ('CURRENT_TIMESTAMP', 'NULL'):
                    col_def += f" DEFAULT '{default_val}'"
                else:
                    col_def += f" DEFAULT {default_val}"
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
        
        return ""
    
    def _get_table_info(self, connection: pymysql.Connection, db_name: str, table_name: str) -> Dict[str, Any]:
        """Get detailed table information"""
        table_info = {
            'columns': {},
            'primary_keys': [],
            'foreign_keys': [],
            'sample_data': [],
            'create_statement': None,
            'insert_statement': None
        }
        
        # Get columns information with more detailed type info
        query = """
            SELECT column_name, data_type, is_nullable, column_default, column_key,
                   character_maximum_length, numeric_precision, numeric_scale,
                   column_type
            FROM information_schema.columns
            WHERE table_schema = %s AND table_name = %s
            ORDER BY ordinal_position
        """
        result = self.execute_query(connection, query, (db_name, table_name))
        
        if result.success:
            for col in result.data:
                col_name = col['column_name']
                col_type = col['column_type'] if col['column_type'] else col['data_type']
                
                table_info['columns'][col_name] = {
                    'type': col_type,
                    'nullable': col['is_nullable'] == 'YES',
                    'default': col['column_default'],
                    'primary_key': col['column_key'] == 'PRI'
                }
                if col['column_key'] == 'PRI':
                    table_info['primary_keys'].append(col_name)
        
        # Get foreign keys from the table
        query = """
            SELECT column_name, referenced_table_name, referenced_column_name
            FROM information_schema.key_column_usage
            WHERE table_schema = %s AND table_name = %s
            AND referenced_table_name IS NOT NULL
        """
        result = self.execute_query(connection, query, (db_name, table_name))
        
        if result.success:
            for fk in result.data:
                table_info['foreign_keys'].append({
                    'column': fk['column_name'],
                    'referenced_table': fk['referenced_table_name'],
                    'referenced_column': fk['referenced_column_name']
                })

        # Get sample data from the table
        result = self.execute_query(connection, f"SELECT * FROM `{table_name}` LIMIT 100")
        if result.success and result.data:
            table_info['sample_data'] = result.data

        # Get create statement from the table
        result = self.execute_query(connection, f"SHOW CREATE TABLE `{table_name}`")
        if result.success and result.data:
            table_info['create_statement'] = result.data[0].get('Create Table', '')

        # Generate create statement from the table if not available
        if not table_info['create_statement']:
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
                elif isinstance(val, bool):
                    values.append('1' if val else '0')
                else:
                    escaped_val = str(val).replace("'", "''").replace("\\", "\\\\")
                    values.append(f"'{escaped_val}'")
            
            table_ref = f"`{table_name}`"
            col_ref = [f"`{col}`" for col in columns]
                
            insert_sql = f"INSERT INTO {table_ref} ({', '.join(col_ref)}) VALUES ({', '.join(values)})"
            insert_statement_list.append(insert_sql)

        table_info['insert_statement'] = "\n\n".join(insert_statement_list)
        
        return table_info

    def discover_databases(self, config: Dict) -> Dict[str, DatabaseInfo]:
        """Discover MySQL databases"""
        databases = {}
        try:
            temp_config = {k: v for k, v in config.items() if k != 'database'}
            conn = self.connect(temp_config)
            
            # Get list of databases from the server
            result = self.execute_query(conn, "SHOW DATABASES")
            if result.success:
                system_dbs = {'information_schema', 'mysql', 'performance_schema', 'sys'}
                for row in result.data:
                    db_name = list(row.values())[0]
                    if db_name not in system_dbs:
                        databases[db_name] = DatabaseInfo(
                            db_id=db_name,
                            db_type='mysql',
                            connection_info={**config, 'database': db_name},
                            metadata={'host': config.get('host', 'localhost'), 
                                    'port': config.get('port', 3306)}
                        )
                        self.logger.info(f"Discovered MySQL database: {db_name}")
            
            conn.close()
        except Exception as e:
            self.logger.error(f"Error discovering MySQL databases: {e}")
        
        return databases

    def close(self, connection: pymysql.Connection):
        """Close database connection"""
        try:
            if connection:
                connection.close()
        except Exception as e:
            self.logger.error(f"Error closing connection: {e}")