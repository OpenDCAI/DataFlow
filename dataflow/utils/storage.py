from abc import ABC, abstractmethod
from dataflow import get_logger
import pandas as pd
import json
from typing import Any, Literal
import os


class DataFlowStorage(ABC):
    """
    Abstract base class for data storage.
    """
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

class FileStorage(DataFlowStorage):
    """
    Storage for file system.
    """
    def __init__(self, 
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
            return os.path.join(self.cache_path, f"{self.file_name_prefix}_{step}.{self.cache_type}")

    def step(self):
        self.operator_step += 1
        return self
    
    def reset(self):
        self.operator_step = -1
        return self
    
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

    def read(self, output_type: Literal["dataframe", "dict"]) -> Any:
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
        if type(data) == list:
            if type(data[0]) == dict:
                dataframe = pd.DataFrame(data)
            else:
                raise ValueError(f"Unsupported data type: {type(data[0])}")
        elif type(data) == pd.DataFrame:
            dataframe = data
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")

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

_clickhouse_clients = {}
_clickhouse_clients_lock = Lock()

# 预定义字段前缀
SYS_FIELD_PREFIX = 'sys:'
USER_FIELD_PREFIX = 'user:'

# 获取ClickHouse Client单例
def get_clickhouse_client(db_config):
    key = (
        db_config['host'],
        db_config.get('port', 9000),
        db_config.get('user', 'default'),
        db_config.get('database', 'dataflow'),
    )
    with _clickhouse_clients_lock:
        if key not in _clickhouse_clients:
            try:
                from clickhouse_driver import Client
            except ImportError as e:
                raise ImportError("clickhouse_driver is required for MyScaleDBStorage but not installed. Please install it via 'pip install clickhouse-driver'.") from e
            _clickhouse_clients[key] = Client(
                host=db_config['host'],
                port=db_config.get('port', 9000),
                user=db_config.get('user', 'default'),
                password=db_config.get('password', ''),
                database=db_config.get('database', 'default'),
                settings={"use_numpy": True}
            )
        return _clickhouse_clients[key]

# 安全加载json数据
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

# 预定义min_hashes计算方法，当前全部返回[0]
def _default_min_hashes(self, data_dict):
    return [0]

class MyScaleDBStorage(DataFlowStorage):
    """
    Storage for Myscale/ClickHouse database using clickhouse_driver.
    """
    def validate_required_params(self):
        """
        校验MyScaleDBStorage实例的关键参数有效性：
        - pipeline_id, input_task_id, output_task_id 必须非空，否则抛出异常。
        - page_size, page_num 若未设置则赋默认值（page_size=10000, page_num=0）。
        所有算子在使用storage前应调用本方法。
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
        self.client = get_clickhouse_client(db_config)
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
        result = self.client.execute(sql, params, with_column_types=True)
        rows, col_types = result
        columns = [col[0] for col in col_types]
        df = pd.DataFrame(rows, columns=columns)
        # 解析 data 字段为 dict
        if 'data' not in df.columns:
            raise ValueError("Result does not contain required 'data' field.")

        # 只保留 data 字段
        data_series = df['data'].apply(safe_json_loads)

        if output_type == "dataframe":
            # 返回只有 data 一列的 DataFrame
            return pd.DataFrame({'data': data_series})
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
        if isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, pd.DataFrame):
            df = data
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
        # data字段本身就是每行的内容
        if 'data' not in df.columns:
            # 兼容直接传入dict列表的情况
            df['data'] = df.apply(lambda row: row.to_dict(), axis=1)
        # 统一处理data列
        df['data'] = df['data'].apply(lambda x: x if isinstance(x, dict) else (json.loads(x) if isinstance(x, str) else {}))
        # 自动填充pipeline_id, task_id, raw_data_id, min_hashes
        df['pipeline_id'] = self.pipeline_id
        df['task_id'] = self.output_task_id
        df['raw_data_id'] = df['data'].apply(lambda d: d.get(SYS_FIELD_PREFIX + 'raw_data_id', 0) if isinstance(d, dict) else 0)
        df['min_hashes'] = df['data'].apply(lambda d: _default_min_hashes(d) if isinstance(d, dict) else [0])
        # data字段转为JSON字符串
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
        self.client.execute(insert_sql, values)
        return f"Inserted {len(values)} rows into {self.table}"