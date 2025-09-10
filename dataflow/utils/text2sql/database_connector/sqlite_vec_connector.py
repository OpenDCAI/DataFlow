from typing import Dict, Any, Optional, Tuple
from ..base import DatabaseConnectorABC, DatabaseInfo, QueryResult
import sqlite3
import glob
import os
import time
import logging

# ============== SQLite Connector ==============
class SQLiteVecConnector(DatabaseConnectorABC):
    
    def connect(self, connection_info: Dict) -> sqlite3.Connection:
        db_path = connection_info['path']
        conn = sqlite3.connect(
            db_path,
            timeout=30,
            check_same_thread=False  
        )
        try:
            import sqlite_vec
            import sqlite_lembed
        except ImportError:
            logging.info("Fatal Error: 'sqlite_vec' library not installed. Please install with 'pip install sqlite-vec'")
            exit()
        
        # Enable and load the sqlite_vec extension.
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.row_factory = sqlite3.Row
        
        # PRAGMA query_only is a good security measure, but it prevents PRAGMA statements
        # needed for schema discovery. We will execute it only when we are done with schema part.
        # For now, we will comment it out from the connect method.
        # conn.execute("PRAGMA query_only = ON") 
        
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
    
    def _get_db_details(self, schema: Dict[str, Any]) -> str:
        """
        根据 schema 信息生成格式化的 DDL 语句，同时支持 sqlite-vec 虚拟表。
        """
        db_details = []
        
        for table_name, table_info in schema['tables'].items():
            # 从 schema 中获取原始的 CREATE 语句
            original_create_statement = table_info.get('create_statement', '')

            # 判断是否为虚拟表
            is_virtual_table = "CREATE VIRTUAL TABLE" in original_create_statement.upper()

            if is_virtual_table:
                # 对于虚拟表，直接使用其原始的 CREATE 语句
                # 因为它们不支持外键、主键约束，也不需要展示示例数据
                db_details.append(original_create_statement)
            else:
                # 对于普通表，使用原有的逻辑来构建详细信息
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
                
                # 拼接 CREATE TABLE 语句
                create_table = f"CREATE TABLE `{table_name}` (\n"
                all_parts = column_defs + constraints
                create_table += ",\n".join(all_parts)
                create_table += "\n);"
                
                # 添加示例数据
                insert_statements = table_info.get('insert_statement', [])
                if insert_statements:
                    create_table += "\n\n-- Sample data:\n"
                    # insert_statement 可能是一个列表，也可能是一个字符串，这里做兼容处理
                    if isinstance(insert_statements, list):
                        create_table += "\n".join(insert_statements)
                    else:
                        create_table += str(insert_statements)

                db_details.append(create_table)
        
        return "\n\n".join(db_details)

    def get_schema_info(self, connection: sqlite3.Connection) -> Dict[str, Any]:
        """Get complete schema information, now aware of sqlite-vec virtual tables."""
        schema = {'tables': {}}
        
        # Fetch the original 'CREATE' statement from sqlite_master.
        # This is the most reliable way to identify virtual tables created by sqlite-vec.
        result = self.execute_query(
            connection, 
            "SELECT name, sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
        
        if not result.success:
            return schema
        
        for row in result.data:
            table_name = row['name']
            create_statement = row['sql']
            
            # Pass the table name and its create statement to get detailed info.
            table_info = self._get_table_info(connection, table_name, create_statement)
            schema['tables'][table_name] = table_info
        
        schema['db_details'] = self._get_db_details(schema)
        return schema

    def generate_create_statement(self, table_name: str, table_info: Dict[str, Any]) -> str:
        """Generate CREATE TABLE statements for REGULAR tables from schema."""
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
    
    def _get_table_info(self, connection: sqlite3.Connection, table_name: str, create_statement: str) -> Dict[str, Any]:
        """
        获取详细的表信息。
        此版本会区分普通表和 sqlite-vec 虚拟表，
        并且在为普通表提取样本数据时，会排除以 '_embedding' 结尾的列，
        同时会检查单行样本数据的总长度是否超过10000个字符。
        """
        # 检查表是否为 sqlite-vec 虚拟表
        is_vec_table = "USING vec0" in create_statement.upper()

        table_info = {
            'columns': {},
            'primary_keys': [],
            'foreign_keys': [],
            'sample_data': [],
            'create_statement': create_statement,  # 始终先存储原始的 CREATE 语句
            'insert_statement': None
        }
        
        # 使用 PRAGMA 获取列信息，这对普通表和虚拟表都有效
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

        # 从这里开始，分别处理普通表和虚拟表
        if is_vec_table:
            # 对于 sqlite-vec 表，我们不需要主键、外键、样本数据或 INSERT 语句
            self.logger.info(f"检测到 sqlite-vec 虚拟表: '{table_name}'。跳过样本数据和外键获取。")
        else:
            # 对于普通表，继续执行原有逻辑以获取完整细节
            
            # 从 PRAGMA table_info 结果中获取主键
            for col in result.data:
                if col['pk']:
                    table_info['primary_keys'].append(col['name'])
            
            # 获取外键
            fk_result = self.execute_query(connection, f"PRAGMA foreign_key_list([{table_name}])")
            if fk_result.success:
                for fk in fk_result.data:
                    table_info['foreign_keys'].append({
                        'column': fk['from'],
                        'referenced_table': fk['table'],
                        'referenced_column': fk['to']
                    })
            
            # === 修改开始 ===
            # 获取表中所有列名
            all_columns = list(table_info['columns'].keys())
            
            # 筛选出不以 "_embedding" 结尾的列名
            columns_to_select = [f"`{col}`" for col in all_columns if not col.endswith('_embedding')]
            
            # 只有在筛选后还有列的情况下，才查询样本数据
            if columns_to_select:
                select_clause = ", ".join(columns_to_select)
                sample_query = f"SELECT {select_clause} FROM [{table_name}] LIMIT 3"
                
                sample_result = self.execute_query(connection, sample_query)
                
                # 增加对行长度的检查
                if sample_result.success and sample_result.data:
                    is_data_too_long = False
                    for row in sample_result.data:
                        # 计算单行所有值的总字符长度
                        row_char_length = sum(len(str(value)) for value in row.values())
                        if row_char_length > 10000:
                            self.logger.warning(f"表 '{table_name}' 的一行样本数据长度超过10000字符，将忽略该表的样本数据。")
                            is_data_too_long = True
                            break # 只要有一行超长，就停止检查并跳出循环
                    
                    # 只有在数据长度检查通过时，才添加样本数据
                    if not is_data_too_long:
                        table_info['sample_data'] = sample_result.data
            # === 修改结束 ===

            # 使用我们的 schema 信息重新生成一个干净的 CREATE 语句（可选，但有利于保持一致性）
            table_info['create_statement'] = self.generate_create_statement(table_name, table_info)

            # 只有在 sample_data 存在时才生成 INSERT 语句
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
        """Discover SQLite database files (No changes needed here)."""
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
