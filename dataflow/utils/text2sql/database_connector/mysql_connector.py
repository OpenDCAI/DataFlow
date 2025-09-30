from typing import Dict, Any, Optional, Tuple
from dataflow import get_logger
from ..base import DatabaseConnectorABC, DatabaseInfo, QueryResult
import pymysql
import pymysql.cursors
import time

class MySQLConnector(DatabaseConnectorABC):
    """MySQL database connector implementation with full schema support"""
    
    def __init__(self):
        self.logger = get_logger()
    
    def connect(self, connection_info: Dict) -> pymysql.Connection:
        """Connect to MySQL database with read-only settings"""
        config = {
            'connect_timeout': 10,
            'read_timeout': 30,
            'write_timeout': 30,
            'charset': 'utf8mb4',
            'cursorclass': pymysql.cursors.DictCursor,
            'autocommit': True
        }
        config.update(connection_info)
        
        try:
            conn = pymysql.connect(**config)
            with conn.cursor() as cursor:
                cursor.execute("SET SESSION TRANSACTION READ ONLY")
                cursor.execute("SET SESSION sql_mode = 'ANSI'")
            return conn
        except pymysql.Error as e:
            self.logger.error(f"MySQL connection failed: {e}")
            raise
    
    def execute_query(self, connection: pymysql.Connection, sql: str, params: Optional[Tuple] = None) -> QueryResult:
        """Execute query with enhanced error handling and result processing"""
        start_time = time.time()
        cursor = None
        
        try:
            cursor = connection.cursor()
            if params:
                cursor.execute(sql, params)
            else:
                cursor.execute(sql)
            
            # Handle different query types
            if sql.strip().upper().startswith(('SELECT', 'SHOW', 'DESC', 'EXPLAIN')):
                rows = cursor.fetchall()
                columns = [desc[0] for desc in cursor.description] if cursor.description else []
                # Since we're using DictCursor, rows are already dictionaries
                data = rows if rows else []
            else:
                raise Exception("Write operations are not allowed in read-only mode")
            
            return QueryResult(
                success=True,
                data=data,
                columns=columns,
                row_count=len(data)
            )
        except Exception as e:
            self.logger.error(f"Query failed: {e}\nSQL: {sql}")
            return QueryResult(
                success=False,
                error=str(e)
            )
        finally:
            if cursor:
                cursor.close()
    
    def get_schema_info(self, connection: pymysql.Connection) -> Dict[str, Any]:
        """Get complete schema information with formatted DDL"""
        schema = {'tables': {}}
        
        # Get current database
        db_result = self.execute_query(connection, "SELECT DATABASE() AS db")
        if not db_result.success or not db_result.data:
            return schema
        
        db_name = db_result.data[0]['db']
        if not db_name:
            self.logger.warning("No database selected")
            return schema
        
        # Get all tables
        tables_result = self.execute_query(connection, 
            "SELECT TABLE_NAME FROM information_schema.tables "
            "WHERE table_schema = %s AND table_type = 'BASE TABLE'", 
            (db_name,))
        
        if not tables_result.success:
            return schema
        
        for row in tables_result.data:
            table_name = row['TABLE_NAME']
            table_info = self._get_table_info(connection, db_name, table_name)
            schema['tables'][table_name] = table_info
        
        schema['db_details'] = self._get_db_details(schema)
        return schema
    
    def _get_table_info(self, connection: pymysql.Connection, db_name: str, table_name: str) -> Dict[str, Any]:
        """Get detailed table information"""
        table_info = {
            'columns': {},
            'primary_keys': [],
            'foreign_keys': [],
            'sample_data': [],
            'create_statement': None,
            'insert_statement': []
        }
        
        # Get columns information
        col_result = self.execute_query(connection, """
            SELECT COLUMN_NAME, COLUMN_TYPE, IS_NULLABLE, COLUMN_DEFAULT, COLUMN_KEY, 
                   EXTRA, CHARACTER_MAXIMUM_LENGTH, NUMERIC_PRECISION
            FROM information_schema.columns
            WHERE table_schema = %s AND table_name = %s
            ORDER BY ORDINAL_POSITION
        """, (db_name, table_name))
        
        if col_result.success:
            for col in col_result.data:
                col_name = col['COLUMN_NAME']
                default = col['COLUMN_DEFAULT']
                
                # Format default value
                if default is not None:
                    if isinstance(default, str) and default.upper() not in ('CURRENT_TIMESTAMP', 'NULL'):
                        default = f"'{default}'"
                    elif isinstance(default, bool):
                        default = str(int(default))
                
                table_info['columns'][col_name] = {
                    'type': col['COLUMN_TYPE'],
                    'nullable': col['IS_NULLABLE'] == 'YES',
                    'default': default,
                    'primary_key': col['COLUMN_KEY'] == 'PRI'
                }
                if col['COLUMN_KEY'] == 'PRI':
                    table_info['primary_keys'].append(col_name)
        
        # Get foreign keys
        fk_result = self.execute_query(connection, """
            SELECT 
                kcu.COLUMN_NAME,
                kcu.REFERENCED_TABLE_NAME,
                kcu.REFERENCED_COLUMN_NAME,
                rc.UPDATE_RULE,
                rc.DELETE_RULE
            FROM information_schema.key_column_usage kcu
            LEFT JOIN information_schema.referential_constraints rc
                ON rc.CONSTRAINT_SCHEMA = kcu.TABLE_SCHEMA
                AND rc.TABLE_NAME = kcu.TABLE_NAME
                AND rc.CONSTRAINT_NAME = kcu.CONSTRAINT_NAME
            WHERE kcu.TABLE_SCHEMA = %s 
                AND kcu.TABLE_NAME = %s
                AND kcu.REFERENCED_TABLE_NAME IS NOT NULL
        """, (db_name, table_name))
        
        if fk_result.success:
            for fk in fk_result.data:
                table_info['foreign_keys'].append({
                    'column': fk['COLUMN_NAME'],
                    'referenced_table': fk['REFERENCED_TABLE_NAME'],
                    'referenced_column': fk['REFERENCED_COLUMN_NAME'],
                    'update_rule': fk['UPDATE_RULE'],
                    'delete_rule': fk['DELETE_RULE']
                })
        
        # Get sample data (limited to 2 rows)
        sample_result = self.execute_query(connection, 
            f"SELECT * FROM `{table_name}` LIMIT 2")
        
        if sample_result.success and sample_result.data:
            table_info['sample_data'] = sample_result.data
            column_names = list(sample_result.data[0].keys())
            
            # Generate INSERT statements
            for row in sample_result.data:
                values = []
                for val in row.values():
                    if val is None:
                        values.append('NULL')
                    elif isinstance(val, (int, float)):
                        values.append(str(val))
                    elif isinstance(val, bool):
                        values.append('1' if val else '0')
                    else:
                        escaped = str(val).replace("'", "''").replace("\\", "\\\\")
                        values.append(f"'{escaped}'")
                
                insert_sql = (
                    f"INSERT INTO `{table_name}` ({', '.join(f'`{c}`' for c in column_names)}) "
                    f"VALUES ({', '.join(values)});"
                )
                table_info['insert_statement'].append(insert_sql)
        
        # Get create statement
        create_result = self.execute_query(connection, 
            f"SHOW CREATE TABLE `{table_name}`")
        
        if create_result.success and create_result.data:
            table_info['create_statement'] = create_result.data[0]['Create Table']
        
        return table_info
    
    def _get_db_details(self, schema: Dict[str, Any]) -> str:
        """Generate formatted DDL statements from schema information"""
        db_details = []
        
        for table_name, table_info in schema['tables'].items():
            # Use original create statement if available
            if table_info['create_statement']:
                create_table = table_info['create_statement']
            else:
                # Build create statement from metadata
                column_defs = []
                for col_name, col_info in table_info['columns'].items():
                    col_def = f"    `{col_name}` {col_info['type']}"
                    
                    if not col_info['nullable']:
                        col_def += " NOT NULL"
                    
                    if col_info['default'] is not None:
                        col_def += f" DEFAULT {col_info['default']}"
                    
                    if 'auto_increment' in col_info.get('extra', '').lower():
                        col_def += " AUTO_INCREMENT"
                    
                    column_defs.append(col_def)
                
                # Add constraints
                constraints = []
                
                # Primary keys
                if table_info['primary_keys']:
                    pk_cols = ", ".join(f"`{pk}`" for pk in table_info['primary_keys'])
                    constraints.append(f"    PRIMARY KEY ({pk_cols})")
                
                # Foreign keys
                for fk in table_info['foreign_keys']:
                    fk_def = (
                        f"    CONSTRAINT `fk_{table_name}_{fk['column']}` "
                        f"FOREIGN KEY (`{fk['column']}`) "
                        f"REFERENCES `{fk['referenced_table']}` (`{fk['referenced_column']}`)"
                    )
                    if fk.get('update_rule'):
                        fk_def += f" ON UPDATE {fk['update_rule']}"
                    if fk.get('delete_rule'):
                        fk_def += f" ON DELETE {fk['delete_rule']}"
                    constraints.append(fk_def)
                
                create_table = f"CREATE TABLE `{table_name}` (\n"
                create_table += ",\n".join(column_defs + constraints)
                create_table += "\n) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;"
            
            db_details.append(create_table)
            
            # Add sample data if available
            if table_info['sample_data']:
                db_details.append("\n-- Sample data:")
                db_details.extend(table_info['insert_statement'])
        
        return "\n\n".join(db_details)

    
    
    def discover_databases(self, config: Dict) -> Dict[str, DatabaseInfo]:
        """Discover MySQL databases on server"""
        databases = {}
        
        try:
            # Create temp connection without specific database
            temp_config = {k: v for k, v in config.items() if k != 'database'}
            conn = self.connect(temp_config)
            
            # Get database list
            result = self.execute_query(conn, 
                "SELECT schema_name as db_name FROM information_schema.schemata "
                "WHERE schema_name NOT IN ('information_schema', 'mysql', 'performance_schema', 'sys')")
            
            if result.success:
                for row in result.data:
                    db_name = row.get('db_name')
                    if not db_name:
                        continue
                    databases[db_name] = DatabaseInfo(
                        db_id=db_name,
                        db_type='mysql',
                        connection_info={**config, 'database': db_name},
                        metadata={
                            'host': config.get('host', 'localhost'),
                            'port': config.get('port', 3306),
                            'charset': config.get('charset', 'utf8mb4')
                        }
                    )
                    self.logger.info(f"Discovered MySQL database: {db_name}")
            
            conn.close()
        except Exception as e:
            self.logger.error(f"Database discovery failed: {e}")
        
        return databases

    def get_number_of_special_column(self, connection: pymysql.Connection) -> int:
        """
        Get the number of columns ending with '_embedding' in database
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
                
                # Traverse the column names
                for column_name in columns.keys():
                    # Check if the column name ends with '_embedding'
                    if isinstance(column_name, str) and column_name.endswith('_embedding'):
                        count += 1
                        
        except Exception as e:
            self.logger.error(f"Error counting embedding columns: {e}")

        if count == 0:
            self.logger.warning("No columns ending with '_embedding' found.")
        return count