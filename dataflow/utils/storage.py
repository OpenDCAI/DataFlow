from abc import ABC, abstractmethod
from dataflow import get_logger
import pandas as pd
import json
from typing import Any, Literal
import os
import copy

class DataFlowStorage(ABC):
    """
    Abstract base class for data storage.
    """
    @abstractmethod
    def get_keys_from_dataframe(self) -> list[str]:
        """
        Get keys from the dataframe stored in the storage.
        This method should be implemented by subclasses to extract keys from the data.
        """
        pass
    @abstractmethod
    def read(self, output_type) -> Any:
        """
        Read data from file.
        type: type that you want to read to, such as "datatrame", List[dict], etc.
        """
        pass
    
    @abstractmethod
    def write(self, data: Any) -> Any:
        pass

class DummyStorage(DataFlowStorage):
    def __init__(
        self,
        cache_path:str=None,
        file_name_prefix:str=None,  
        cache_type: Literal["json", "jsonl", "csv", "parquet", "pickle", None] = None
    ):
        self._data = None
        self.cache_path = cache_path
        self.file_name_prefix = file_name_prefix
        self.cache_type = cache_type
        
    def set_data(self, data: Any):
        """
        Set data to be written later.
        """
        self._data = data
        
    def set_file_name_prefix(self, file_name_prefix: str):
        """
        Set the file name prefix for cache files.
        """
        self.file_name_prefix = file_name_prefix
            
    def read(self, output_type: Literal["dataframe", "dict"] = "dataframe") -> Any:
        return self._data

    def write(self, data):
        self._data = data
        if self.cache_type != None:
            cache_file_path = os.path.join(self.cache_path, f"{self.file_name_prefix}.{self.cache_type}")
            os.makedirs(os.path.dirname(cache_file_path), exist_ok=True)
            if self.cache_type == "json":
                data.to_json(cache_file_path, orient="records", force_ascii=False, indent=2)
            elif self.cache_type == "jsonl":
                data.to_json(cache_file_path, orient="records", lines=True, force_ascii=False)  
            elif self.cache_type == "csv":
                data.to_csv(cache_file_path, index=False)
            elif self.cache_type == "parquet":  
                data.to_parquet(cache_file_path)
            elif self.cache_type == "pickle":
                data.to_pickle(cache_file_path)
            else:
                raise ValueError(f"Unsupported file type: {self.cache_type}, output file should end with json, jsonl, csv, parquet, pickle")

class FileStorage(DataFlowStorage):
    """
    Storage for file system.
    """
    def __init__(
        self, 
        first_entry_file_name: str,
        cache_path:str="./cache",
        file_name_prefix:str="dataflow_cache_step",
        cache_type:Literal["json", "jsonl", "csv", "parquet", "pickle"] = "jsonl"
    ):
        self.first_entry_file_name = first_entry_file_name
        self.cache_path = cache_path
        self.file_name_prefix = file_name_prefix
        self.cache_type = cache_type
        self.operator_step = -1
        self.logger = get_logger()

    def _get_cache_file_path(self, step) -> str:
        if step == -1:
            self.logger.error("You must call storage.step() before reading or writing data. Please call storage.step() first for each operator step.")  
            raise ValueError("You must call storage.step() before reading or writing data. Please call storage.step() first for each operator step.")
        if step == 0:
            # If it's the first step, use the first entry file name
            return os.path.join(self.first_entry_file_name)
        else:
            return os.path.join(self.cache_path, f"{self.file_name_prefix}_step{step}.{self.cache_type}")

    def step(self):
        self.operator_step += 1
        return copy.copy(self) # TODO if future maintain an object in memory, we need to apply a deepcopy (except the dataframe object during copy to avoid OOM)
    
    def reset(self):
        self.operator_step = -1
        return self
    
    def get_keys_from_dataframe(self) -> list[str]:
        dataframe = self.read(output_type="dataframe")
        return dataframe.columns.tolist() if isinstance(dataframe, pd.DataFrame) else []
    
    def _load_local_file(self, file_path: str, file_type: str) -> pd.DataFrame:
        """Load data from local file based on file type."""
        # check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} does not exist. Please check the path.")
        # Load file based on type
        try:
            if file_type == "json":
                return pd.read_json(file_path)
            elif file_type == "jsonl":
                return pd.read_json(file_path, lines=True)
            elif file_type == "csv":
                return pd.read_csv(file_path)
            elif file_type == "parquet":
                return pd.read_parquet(file_path)
            elif file_type == "pickle":
                return pd.read_pickle(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
        except Exception as e:
            raise ValueError(f"Failed to load {file_type} file: {str(e)}")
    
    def _convert_output(self, dataframe: pd.DataFrame, output_type: str) -> Any:
        """Convert dataframe to requested output type."""
        if output_type == "dataframe":
            return dataframe
        elif output_type == "dict":
            return dataframe.to_dict(orient="records")
        raise ValueError(f"Unsupported output type: {output_type}")

    def read(self, output_type: Literal["dataframe", "dict"]="dataframe") -> Any:
        """
        Read data from current file managed by storage.
        
        Args:
            output_type: Type that you want to read to, either "dataframe" or "dict".
            Also supports remote datasets with prefix:
                - "hf:{dataset_name}{:config}{:split}"  => HuggingFace dataset eg. "hf:openai/gsm8k:main:train"
                - "ms:{dataset_name}{}:split}"          => ModelScope dataset eg. "ms:modelscope/gsm8k:train"
        
        Returns:
            Depending on output_type:
            - "dataframe": pandas DataFrame
            - "dict": List of dictionaries
        
        Raises:
            ValueError: For unsupported file types or output types
        """
        if self.operator_step == 0 and self.first_entry_file_name == "":
            self.logger.info("first_entry_file_name is empty, returning empty dataframe")
            empty_dataframe = pd.DataFrame()
            return self._convert_output(empty_dataframe, output_type)

        file_path = self._get_cache_file_path(self.operator_step)
        self.logger.info(f"Reading data from {file_path} with type {output_type}")

        if self.operator_step == 0:
            source = self.first_entry_file_name
            self.logger.info(f"Reading remote dataset from {source} with type {output_type}")
            if source.startswith("hf:"):
                from datasets import load_dataset
                _, dataset_name, *parts = source.split(":")

                if len(parts) == 1:
                    config, split = None, parts[0]
                elif len(parts) == 2:
                    config, split = parts
                else:
                    config, split = None, "train"

                dataset = (
                    load_dataset(dataset_name, config, split=split) 
                    if config 
                    else load_dataset(dataset_name, split=split)
                )
                dataframe = dataset.to_pandas()
                return self._convert_output(dataframe, output_type)
        
            elif source.startswith("ms:"):
                from modelscope import MsDataset
                _, dataset_name, *split_parts = source.split(":")
                split = split_parts[0] if split_parts else "train"

                dataset = MsDataset.load(dataset_name, split=split)
                dataframe = pd.DataFrame(dataset)
                return self._convert_output(dataframe, output_type)
                            
            else:
                local_cache = file_path.split(".")[-1]
        else:
            local_cache = self.cache_type

        dataframe = self._load_local_file(file_path, local_cache)
        return self._convert_output(dataframe, output_type)
        
    def write(self, data: Any) -> Any:
        """
        Write data to current file managed by storage.
        data: Any, the data to write, it should be a dataframe, List[dict], etc.
        """
        def clean_surrogates(obj):
            """递归清理数据中的无效Unicode代理对字符"""
            if isinstance(obj, str):
                # 替换无效的Unicode代理对字符（如\udc00）
                return obj.encode('utf-8', 'replace').decode('utf-8')
            elif isinstance(obj, dict):
                return {k: clean_surrogates(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [clean_surrogates(item) for item in obj]
            elif isinstance(obj, (int, float, bool)) or obj is None:
                # 数字、布尔值和None直接返回
                return obj
            else:
                # 其他类型（如自定义对象）尝试转为字符串处理
                try:
                    return clean_surrogates(str(obj))
                except:
                    # 如果转换失败，返回原对象或空字符串（根据需求选择）
                    return obj

        # 转换数据为DataFrame
        if isinstance(data, list):
            if len(data) > 0 and isinstance(data[0], dict):
                # 清洗列表中的每个字典
                cleaned_data = [clean_surrogates(item) for item in data]
                dataframe = pd.DataFrame(cleaned_data)
            else:
                raise ValueError(f"Unsupported data type: {type(data[0])}")
        elif isinstance(data, pd.DataFrame):
            # 对DataFrame的每个元素进行清洗
            dataframe = data.applymap(clean_surrogates)
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")

        # if type(data) == list:
        #     if type(data[0]) == dict:
        #         dataframe = pd.DataFrame(data)
        #     else:
        #         raise ValueError(f"Unsupported data type: {type(data[0])}")
        # elif type(data) == pd.DataFrame:
        #     dataframe = data
        # else:
        #     raise ValueError(f"Unsupported data type: {type(data)}")

        file_path = self._get_cache_file_path(self.operator_step + 1)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        self.logger.success(f"Writing data to {file_path} with type {self.cache_type}")
        if self.cache_type == "json":
            dataframe.to_json(file_path, orient="records", force_ascii=False, indent=2)
        elif self.cache_type == "jsonl":
            dataframe.to_json(file_path, orient="records", lines=True, force_ascii=False)
        elif self.cache_type == "csv":
            dataframe.to_csv(file_path, index=False)
        elif self.cache_type == "parquet":
            dataframe.to_parquet(file_path)
        elif self.cache_type == "pickle":
            dataframe.to_pickle(file_path)
        else:
            raise ValueError(f"Unsupported file type: {self.cache_type}, output file should end with json, jsonl, csv, parquet, pickle")
        
        return file_path    
    
from threading import Lock
import math
from dataflow.utils.db_pool.myscale_pool import ClickHouseConnectionPool

# 推荐在模块级别创建全局池
myscale_pool = None

def get_myscale_pool(db_config):
    global myscale_pool
    if myscale_pool is None:
        myscale_pool = ClickHouseConnectionPool(
            host=db_config['host'],
            port=db_config.get('port', 9000),
            user=db_config.get('user', 'default'),
            password=db_config.get('password', ''),
            database=db_config.get('database', 'default'),
            min_connections=5,
            max_connections=50,
        )
    return myscale_pool

# 预定义字段前缀
SYS_FIELD_PREFIX = 'sys:'
USER_FIELD_PREFIX = 'user:'

# 保留字段列表
RESERVED_SYS_FIELD_LIST=["raw_data_id"]
RESERVED_USER_FIELD_LIST=[]

# 安全加载 json 数据
def safe_json_loads(x):
        if isinstance(x, dict):
            return x
        if isinstance(x, str):
            try:
                return json.loads(x)
            except Exception:
                return x  # 保留原始字符串
        if pd.isna(x):
            return None
        return x  # 其它类型原样返回

# 安全合并列到 data 字段
def safe_merge(row, col):
    val = row[col]
    if isinstance(val, float) and math.isnan(val):
        return row['data']
    return {**row['data'], col: val}

# 预定义 min_hashes 计算方法，当前全部返回[0]
def _default_min_hashes(data_dict):
    return [0]

class MyScaleDBStorage(DataFlowStorage):
    """
    Storage for Myscale/ClickHouse database using clickhouse_driver.
    """
    def validate_required_params(self):
        """
        校验 MyScaleDBStorage 实例的关键参数有效性：
        - pipeline_id, input_task_id, output_task_id 必须非空，否则抛出异常。
        - page_size, page_num 若未设置则赋默认值（page_size=10000, page_num=0）。
        所有算子在使用 storage 前应调用本方法。
        """
        missing = []
        if not self.pipeline_id:
            missing.append('pipeline_id')
        if not self.input_task_id:
            missing.append('input_task_id')
        if not self.output_task_id:
            missing.append('output_task_id')
        if missing:
            raise ValueError(f"Missing required storage parameters: {', '.join(missing)}")
        if not hasattr(self, 'page_size') or self.page_size is None:
            self.page_size = 10000
        if not hasattr(self, 'page_num') or self.page_num is None:
            self.page_num = 0

    def __init__(
        self,
        db_config: dict,
        pipeline_id: str = None,
        input_task_id: str = None,
        output_task_id: str = None,
        page_size: int = 10000,
        page_num: int = 0
    ):
        """
        db_config: {
            'host': 'localhost',
            'port': 9000,
            'user': 'default',
            'password': '',
            'database': 'dataflow',
            'table': 'dataflow_table'
        }
        pipeline_id: str, 当前 pipeline 的标识（可选，默认 None）
        input_task_id: str, 输入任务的标识（可选，默认 None）
        output_task_id: str, 输出任务的标识（可选，默认 None）
        page_size: int, 分页时每页的记录数（默认 10000）
        page_num: int, 当前页码（默认 0）
        """
        self.db_config = db_config
        self.table = db_config.get('table', 'dataflow_table')
        self.logger = get_logger()
        self.pipeline_id: str = pipeline_id
        self.input_task_id: str = input_task_id
        self.output_task_id: str = output_task_id
        self.page_size: int = page_size
        self.page_num: int = page_num
        self.validate_required_params()

    def read(self, output_type: Literal["dataframe", "dict"]) -> Any:
        """
        Read data from Myscale/ClickHouse table.
        """
        pool = get_myscale_pool(self.db_config)
        with pool.get_connection() as client:
            where_clauses = []
            params = {}
            if self.pipeline_id:
                where_clauses.append("pipeline_id = %(pipeline_id)s")
                params['pipeline_id'] = self.pipeline_id
            if self.input_task_id:
                where_clauses.append("task_id = %(task_id)s")
                params['task_id'] = self.input_task_id
            where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
            limit_offset = f"LIMIT {self.page_size} OFFSET {(self.page_num-1)*self.page_size}" if self.page_size else ""
            sql = f"SELECT * FROM {self.table} {where_sql} {limit_offset}"
            self.logger.info(f"Reading from DB: {sql} with params {params}")
            result = client.execute(sql, params, with_column_types=True)
            rows, col_types = result
            columns = [col[0] for col in col_types]
            df = pd.DataFrame(rows, columns=columns)
            # 解析 data 字段为 dict
            if 'data' not in df.columns:
                raise ValueError("Result does not contain required 'data' field.")

            # 只保留 data 字段
            data_series = df['data'].apply(safe_json_loads)
            if output_type == "dataframe":
                # 提取 data 一列并自动展开为多列（如果 data 是 dict）
                result_df = pd.DataFrame({'data': data_series})
                if not data_series.empty and isinstance(data_series.iloc[0], dict):
                    expanded = data_series.apply(pd.Series)
                    # 合并展开列和原始 data 列
                    result_df = pd.concat([result_df, expanded], axis=1)
                return result_df
            elif output_type == "dict":
                # 返回 data 字段的 dict 列表
                return list(data_series)
            else:
                raise ValueError(f"Unsupported output type: {output_type}")

    def write(self, data: Any) -> Any:
        """
        Write data to Myscale/ClickHouse table.
        data: pd.DataFrame or List[dict]，每行是data字段内容（dict）。
        """
        pool = get_myscale_pool(self.db_config)
        with pool.get_connection() as client:
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, pd.DataFrame):
                df = data
            else:
                raise ValueError(f"Unsupported data type: {type(data)}")
            # data 字段本身就是每行的内容
            if 'data' not in df.columns:
                # 兼容直接传入 dict 列表的情况
                df['data'] = df.apply(lambda row: row.to_dict(), axis=1)
            # 统一处理 data 列
            df['data'] = df['data'].apply(lambda x: x if isinstance(x, dict) else (json.loads(x) if isinstance(x, str) else {}))
            # 合并所有非系统字段到 data 字段并删除原列
            system_cols = {'pipeline_id', 'task_id', 'raw_data_id', 'min_hashes', 'data'}
            for col in df.columns:
                if col not in system_cols:
                    df['data'] = df.apply(lambda row: safe_merge(row, col), axis=1)
                    df = df.drop(columns=[col])
            # 自动填充 pipeline_id, task_id, raw_data_id, min_hashes
            df['pipeline_id'] = self.pipeline_id
            df['task_id'] = self.output_task_id
            df['raw_data_id'] = df['data'].apply(lambda d: d.get(SYS_FIELD_PREFIX + 'raw_data_id', 0) if isinstance(d, dict) else 0)
            df['min_hashes'] = df['data'].apply(lambda d: _default_min_hashes(d) if isinstance(d, dict) else [0])
            # data 字段转为 JSON 字符串
            df['data'] = df['data'].apply(lambda x: json.dumps(x, ensure_ascii=False) if not isinstance(x, str) else x)
            # 只保留必需字段
            required_cols = ['pipeline_id', 'task_id', 'raw_data_id', 'min_hashes', 'data']
            df = df[required_cols]
            records = df.to_dict(orient="records")
            values = [
                (
                    rec['pipeline_id'],
                    rec['task_id'],
                    int(rec['raw_data_id']),
                    rec['min_hashes'],
                    rec['data']
                ) for rec in records
            ]
            insert_sql = f"""
            INSERT INTO {self.table} (pipeline_id, task_id, raw_data_id, min_hashes, data)
            VALUES
            """
            self.logger.info(f"Inserting {len(values)} rows into {self.table}")
            client.execute(insert_sql, values)
            return f"Inserted {len(values)} rows into {self.table}"

    def get_keys_from_dataframe(self) -> list[str]:
        """
        Get keys from the dataframe stored in MyScale/ClickHouse database.
        Returns column names from the dataframe after reading from database.
        """
        dataframe = self.read(output_type="dataframe")
        return dataframe.columns.tolist() if isinstance(dataframe, pd.DataFrame) else []