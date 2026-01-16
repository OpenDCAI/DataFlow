from abc import ABC, abstractmethod
import atexit
import signal
import tempfile
import weakref

from dataflow import get_logger
import pandas as pd
import json
from typing import Any, Dict, Literal, Generator
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

# --- 新增的 __repr__ 方法 ---
    def __repr__(self):
        """
        返回一个表示该对象所有成员变量及其键值对的字符串。
        """
        # 获取实例的所有属性（成员变量）
        attrs = self.__dict__
        
        # 格式化键值对列表
        attr_strs = []
        for key, value in attrs.items():
            # 特殊处理 pandas DataFrame 或其他大型对象
            if isinstance(value, pd.DataFrame):
                # 如果是 DataFrame，只显示其类型和形状，避免输出过多内容
                value_repr = f"<DataFrame shape={value.shape}>"
            elif isinstance(value, set):
                # 简化集合的显示
                 value_repr = f"<{type(value).__name__} size={len(value)}>"
            elif isinstance(value, dict):
                # 简化字典的显示
                 value_repr = f"<{type(value).__name__} size={len(value)}>"
            else:
                # 使用标准的 repr() 获取值表示，并限制长度
                value_repr = repr(value)
                if len(value_repr) > 100:  # 限制长度以避免超长输出
                    value_repr = value_repr[:97] + "..."
                    
            attr_strs.append(f"  {key} = {value_repr}")
            
        # 构造最终的字符串
        body = "\n".join(attr_strs)
        return f"<{self.__class__.__name__} Object:\n{body}\n>"
    # ---------------------------
class LazyFileStorage(DataFlowStorage):
    """
    LazyFileStorage
    ----------------
    - 平时仅在内存中读写；只在进程正常结束/被中断或显式 flush 时落盘
    - 与传统 FileStorage 的接口保持一致（step/reset/read/write/…）
    - 支持首步数据源: 本地文件 / 'hf:' / 'ms:'
    - 原子落盘(os.replace)，多线程安全(RLock)
    """

    def __init__(
        self, 
        first_entry_file_name: str,
        cache_path:str="./cache",
        file_name_prefix:str="dataflow_cache_step",
        cache_type:Literal["json", "jsonl", "csv", "parquet", "pickle"] = "jsonl",
        save_on_exit: bool = True,        # 进程退出时自动 flush
        flush_all_steps: bool = False      # True: 所有缓冲步落盘；False: 仅最新一步
    ):
        """
        Initialize a LazyFileStorage.

        This storage keeps read/write operations in memory and only persists to disk
        when explicitly flushed or when the process exits / is interrupted (if
        save_on_exit is enabled). The class preserves an interface compatible with
        a traditional FileStorage (step / reset / read / write / ...). Step 0
        (the "first entry") may be loaded from a local file path or from remote
        dataset identifiers prefixed with "hf:" (HuggingFace datasets) or "ms:"
        (ModelScope datasets). Disk writes are performed atomically via os.replace,
        and internal buffers are protected with an RLock for thread safety.

        Args:
            first_entry_file_name (str):
                Path or source identifier used as the data source for step 0.
                - If it starts with "hf:" the remainder is interpreted as a
                HuggingFace dataset specification (dataset[:config][:split]).
                - If it starts with "ms:" the remainder is interpreted as a
                ModelScope dataset specification (dataset[:split]).
                - Otherwise it is treated as a local file path; the file extension
                (json, jsonl, csv, parquet, pickle) determines how it is loaded.
            cache_path (str, optional):
                Directory where per-step cache files are written. Defaults to "./cache".
            file_name_prefix (str, optional):
                Prefix used when composing cache filenames for non-zero steps.
                Defaults to "dataflow_cache_step" (files will be like
                "{file_name_prefix}_step{n}.{cache_type}").
            cache_type (Literal["json","jsonl","csv","parquet","pickle"], optional):
                Serialization format used when persisting buffers to disk.
                - "json": full JSON array
                - "jsonl": JSON Lines (one record per line)
                - "csv": comma-separated values
                - "parquet": Apache Parquet
                - "pickle": Python pickle
                Defaults to "jsonl".
            save_on_exit (bool, optional):
                If True (default), register handlers so that buffered data is flushed
                automatically when the process exits or when SIGINT/SIGTERM is received.
                In restricted environments registering signal handlers may fail silently.
            flush_all_steps (bool, optional):
                Controls flush_all() behavior:
                - True: flush_all() persists all buffered steps.
                - False (default): flush_all() persists only the most recently buffered step.

        Returns:
            None

        Notes:
            - The instance maintains an internal operator_step counter (initially -1).
            Calling step() increments this counter; write() buffers data for
            operator_step + 1.
            - write(...) buffers data in memory and returns the file path that will
            be used when the buffer is ultimately flushed to disk (useful for logging).
            - Disk writes are atomic: data is first written to a temporary file and
            then moved into place with os.replace to avoid partial files.
            - Thread-safety: an RLock protects buffer and dirty-step state.
            - Errors:
            - Reading a non-existent local file raises FileNotFoundError.
            - Unsupported file types raise ValueError.
        """
        self.first_entry_file_name = first_entry_file_name
        self.cache_path = cache_path
        self.file_name_prefix = file_name_prefix
        self.cache_type = cache_type
        self.operator_step = -1
        self.logger = get_logger()

        # 内存缓冲：step -> DataFrame
        self._buffers: Dict[int, pd.DataFrame] = {}
        self._dirty_steps: set[int] = set()
        self._lock = RLock()

        self._save_on_exit = save_on_exit
        self._flush_all_steps = flush_all_steps

        if self._save_on_exit:
            self._register_exit_hooks()

    # ---------- 生命周期 & 信号 ----------
    def _register_exit_hooks(self):
        _self_ref = weakref.ref(self)
        atexit.register(lambda: _self_ref() and _self_ref()._flush_if_dirty(reason="atexit"))
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, self._make_signal_handler(sig))
            except Exception:
                # 某些环境（子线程/受限容器）可能设置失败
                pass

    def _make_signal_handler(self, sig):
        def handler(signum, frame):
            try:
                self._flush_if_dirty(reason=f"signal:{signum}")
            finally:
                try:
                    signal.signal(signum, signal.SIG_DFL)
                except Exception:
                    pass
                try:
                    os.kill(os.getpid(), signum)
                except Exception:
                    # 某些解释器环境不允许再次发送信号，忽略
                    pass
        return handler

    def _flush_if_dirty(self, reason: str):
        with self._lock:
            dirty = bool(self._dirty_steps)
        if dirty:
            self.logger.info(f"[flush] Triggered by {reason}, flushing...")
            self.flush_all()
        else:
            self.logger.info(f"[flush] Triggered by {reason}, but no dirty buffers.")

    # ---------- step & 路径 ----------
    def _get_cache_file_path(self, step) -> str:
        if step == -1:
            self.logger.error("You must call storage.step() before reading or writing data. Please call storage.step() first for each operator step.")  
            raise ValueError("You must call storage.step() before reading or writing data. Please call storage.step() first for each operator step.")
        if step == 0:
            # 首步来源可能是远端或本地
            return os.path.join(self.first_entry_file_name)
        else:
            return os.path.join(self.cache_path, f"{self.file_name_prefix}_step{step}.{self.cache_type}")

    def step(self):
        self.operator_step += 1
        return copy.copy(self)  # 保持与原 FileStorage 兼容的使用方式
    
    def reset(self):
        self.operator_step = -1
        return self

    # ---------- 工具方法 ----------
    def get_keys_from_dataframe(self) -> list[str]:
        dataframe = self.read(output_type="dataframe")
        return dataframe.columns.tolist() if isinstance(dataframe, pd.DataFrame) else []

    def _load_local_file(self, file_path: str, file_type: str) -> pd.DataFrame:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} does not exist. Please check the path.")
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
        if output_type == "dataframe":
            return dataframe
        elif output_type == "dict":
            return dataframe.to_dict(orient="records")
        raise ValueError(f"Unsupported output type: {output_type}")

    def _read_first_entry_source(self) -> pd.DataFrame:
        source = self.first_entry_file_name
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
            return dataset.to_pandas()
        elif source.startswith("ms:"):
            from modelscope import MsDataset
            _, dataset_name, *split_parts = source.split(":")
            split = split_parts[0] if split_parts else "train"
            dataset = MsDataset.load(dataset_name, split=split)
            return pd.DataFrame(dataset)
        else:
            # 本地首文件
            local_cache = source.split(".")[-1]
            return self._load_local_file(source, local_cache)

    # ---------- 读 ----------
    def read(self, output_type: Literal["dataframe", "dict"]="dataframe") -> Any:
        if self.operator_step == 0 and self.first_entry_file_name == "":
            self.logger.info("first_entry_file_name is empty, returning empty dataframe")
            empty_dataframe = pd.DataFrame()
            return self._convert_output(empty_dataframe, output_type)

        step = self.operator_step
        with self._lock:
            if step in self._buffers:
                self.logger.info(f"[buffer] Reading step {step} from in-memory buffer.")
                return self._convert_output(self._buffers[step], output_type)

        file_path = self._get_cache_file_path(step)
        self.logger.info(f"[read] step={step}, path={file_path}, type={output_type}")

        if step == 0:
            df = self._read_first_entry_source()
            with self._lock:
                self._buffers[step] = df
            return self._convert_output(df, output_type)

        # 兜底：磁盘已有则载入（例如复跑时）
        local_cache = self.cache_type
        df = self._load_local_file(file_path, local_cache)
        with self._lock:
            self._buffers[step] = df
        return self._convert_output(df, output_type)

    # ---------- 写（仅缓冲，不落盘） ----------
    def write(self, data: Any) -> Any:
        """
        将数据写入内存缓冲（目标 step = operator_step + 1），不触盘。
        返回该 step 对应未来落盘的文件路径，便于日志对齐。
        """
        def clean_surrogates(obj):
            if isinstance(obj, str):
                return obj.encode('utf-8', 'replace').decode('utf-8')
            elif isinstance(obj, dict):
                return {k: clean_surrogates(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [clean_surrogates(item) for item in obj]
            elif isinstance(obj, (int, float, bool)) or obj is None:
                return obj
            else:
                try:
                    return clean_surrogates(str(obj))
                except:
                    return obj

        if isinstance(data, list):
            if len(data) > 0 and isinstance(data[0], dict):
                cleaned_data = [clean_surrogates(item) for item in data]
                dataframe = pd.DataFrame(cleaned_data)
            else:
                raise ValueError(f"Unsupported data type: {type(data[0]) if data else 'empty list'}")
        elif isinstance(data, pd.DataFrame):
            dataframe = data.applymap(clean_surrogates)
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")

        target_step = self.operator_step + 1
        with self._lock:
            self._buffers[target_step] = dataframe.reset_index(drop=True)
            self._dirty_steps.add(target_step)

        self.logger.success(f"[buffer] Buffered data for step {target_step} (type={self.cache_type}); not persisted yet.")
        return self._get_cache_file_path(target_step)

    # ---------- 落盘 ----------
    def flush_step(self, step: int):
        with self._lock:
            if step not in self._buffers:
                self.logger.info(f"No buffer for step {step}; nothing to flush.")
                return
            df = self._buffers[step]

        file_path = self._get_cache_file_path(step)
        os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)

        # 原子写：先写临时文件，再替换
        dir_name = os.path.dirname(file_path) or "."
        fd, tmp_path = tempfile.mkstemp(prefix=".tmp_", dir=dir_name)
        os.close(fd)
        try:
            if self.cache_type == "json":
                df.to_json(tmp_path, orient="records", force_ascii=False, indent=2)
            elif self.cache_type == "jsonl":
                df.to_json(tmp_path, orient="records", lines=True, force_ascii=False)
            elif self.cache_type == "csv":
                df.to_csv(tmp_path, index=False)
            elif self.cache_type == "parquet":
                df.to_parquet(tmp_path)
            elif self.cache_type == "pickle":
                df.to_pickle(tmp_path)
            else:
                raise ValueError(f"Unsupported file type: {self.cache_type}, output file should end with json, jsonl, csv, parquet, pickle")

            os.replace(tmp_path, file_path)
            with self._lock:
                self._dirty_steps.discard(step)
            self.logger.success(f"[flush] Persisted step {step} -> {file_path}")
        finally:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass

    def flush_all(self):
        with self._lock:
            if not self._buffers:
                self.logger.info("No buffers to flush.")
                return
            steps = sorted(self._buffers.keys()) if self._flush_all_steps else [max(self._buffers.keys())]
        for s in steps:
            self.flush_step(s)

class DummyStorage(DataFlowStorage):
    def __init__(
        self,
        cache_path:str=None,
        file_name_prefix:str=None,  
        cache_type: Literal["json", "jsonl", "csv", "parquet", "pickle", None] = None
    ):
        self._data = None
        self._store = {}  # Key-value store for get/set operations
        self.cache_path = cache_path
        self.file_name_prefix = file_name_prefix
        self.cache_type = cache_type
        
    def set_data(self, data: Any):
        """
        Set data to be written later.
        """
        self._data = data
    
    def get(self, key: str) -> Any:
        """
        Get data by key from the key-value store.
        """
        return self._store.get(key, self._data)
    
    def set(self, key: str, data: Any):
        """
        Set data by key in the key-value store.
        """
        self._store[key] = data
        self._data = data  # Also update _data for compatibility
        
    def set_file_name_prefix(self, file_name_prefix: str):
        """
        Set the file name prefix for cache files.
        """
        self.file_name_prefix = file_name_prefix
    
    def get_keys_from_dataframe(self) -> list[str]:
        """
        Get keys from the dataframe stored in the storage.
        """
        if isinstance(self._data, pd.DataFrame):
            return self._data.columns.tolist()
        return []
            
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
        """
        Initialize a FileStorage.

        FileStorage is a disk-backed storage implementation that reads from and
        writes to the local filesystem on every read/write operation. Unlike
        LazyFileStorage, data is persisted immediately and no in-memory buffering
        is maintained across steps.

        The storage follows a step-based access pattern compatible with
        DataFlowStorage. Step 0 (the "first entry") can be loaded from:
        - a local file path, or
        - a remote dataset specified with the "hf:" (HuggingFace) or "ms:"
        (ModelScope) prefix.

        Subsequent steps are materialized as files on disk under `cache_path`,
        with filenames derived from `file_name_prefix`, the step index, and
        `cache_type`.

        Args:
            first_entry_file_name (str):
                Path or source identifier used as the data source for step 0.
                - "hf:{dataset}[:config][:split]" loads a HuggingFace dataset.
                - "ms:{dataset}[:split]" loads a ModelScope dataset.
                - Otherwise, this is treated as a local file path, and the file
                extension determines how it is read.
            cache_path (str, optional):
                Directory where cache files for steps >= 1 are stored.
                Defaults to "./cache".
            file_name_prefix (str, optional):
                Prefix used when generating filenames for cached steps.
                Cache files are named as
                "{file_name_prefix}_step{n}.{cache_type}".
                Defaults to "dataflow_cache_step".
            cache_type (Literal["json","jsonl","csv","parquet","pickle"], optional):
                File format used when writing step outputs to disk.
                Defaults to "jsonl".

        Returns:
            None

        Notes:
            - The internal `operator_step` counter starts at -1 and is incremented
            by calling `step()`.
            - Calling `write(...)` writes data immediately to disk for
            `operator_step + 1`.
            - No atomic-write or buffering guarantees are provided beyond the
            underlying pandas file writers.
            - This class is simpler but less fault-tolerant than LazyFileStorage,
            as partially written files may exist if a process is interrupted.
        """
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
            elif self.cache_type == "xlsx":
                return pd.read_excel(file_path)
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
            dataframe = data.map(clean_surrogates)
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
        elif self.cache_type == "xlsx":
            dataframe.to_excel(file_path, index=False)
        else:
            raise ValueError(f"Unsupported file type: {self.cache_type}, output file should end with json, jsonl, csv, parquet, pickle or xlsx")
        
        return file_path    
    
from threading import Lock, RLock
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
        parent_pipeline_id: str = None,
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
        parent_pipeline_id: str, 父 pipeline 的标识（可选，默认 None）
        page_size: int, 分页时每页的记录数（默认 10000）
        page_num: int, 当前页码（默认 0）
        """
        self.db_config = db_config
        self.table = db_config.get('table', 'dataflow_table')
        self.logger = get_logger()
        self.pipeline_id: str = pipeline_id
        self.input_task_id: str = input_task_id
        self.output_task_id: str = output_task_id
        self.parent_pipeline_id: str = parent_pipeline_id
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
            if hasattr(self, 'parent_pipeline_id') and self.parent_pipeline_id:
                where_clauses.append("parent_pipeline_id = %(parent_pipeline_id)s")
                params['parent_pipeline_id'] = self.parent_pipeline_id
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
            system_cols = {'pipeline_id', 'task_id', 'raw_data_id', 'min_hashes', 'file_id', 'filename', 'parent_pipeline_id', 'data'}
            for col in df.columns:
                if col not in system_cols:
                    df['data'] = df.apply(lambda row: safe_merge(row, col), axis=1)
                    df = df.drop(columns=[col])
            # 自动填充 pipeline_id, task_id, raw_data_id, min_hashes, file_id, filename, parent_pipeline_id
            df['pipeline_id'] = self.pipeline_id
            df['task_id'] = self.output_task_id
            df['raw_data_id'] = df['data'].apply(lambda d: d.get(SYS_FIELD_PREFIX + 'raw_data_id', 0) if isinstance(d, dict) else 0)
            df['min_hashes'] = df['data'].apply(lambda d: _default_min_hashes(d) if isinstance(d, dict) else [0])
            # 从 data 中提取 file_id、filename、parent_pipeline_id 字段
            df['file_id'] = df['data'].apply(lambda d: d.get(SYS_FIELD_PREFIX + 'file_id', '') if isinstance(d, dict) else '')
            df['filename'] = df['data'].apply(lambda d: d.get(SYS_FIELD_PREFIX + 'filename', '') if isinstance(d, dict) else '')
            df['parent_pipeline_id'] = df['data'].apply(lambda d: d.get(SYS_FIELD_PREFIX + 'parent_pipeline_id', '') if isinstance(d, dict) else '')
            # 若 data 中未提供 parent_pipeline_id，使用实例属性回填
            if hasattr(self, 'parent_pipeline_id') and self.parent_pipeline_id:
                df['parent_pipeline_id'] = df['parent_pipeline_id'].apply(lambda v: v if v else self.parent_pipeline_id)
            # data 字段转为 JSON 字符串
            df['data'] = df['data'].apply(lambda x: json.dumps(x, ensure_ascii=False) if not isinstance(x, str) else x)
            # 只保留必需字段
            required_cols = ['pipeline_id', 'task_id', 'raw_data_id', 'min_hashes', 'file_id', 'filename', 'parent_pipeline_id', 'data']
            df = df[required_cols]
            records = df.to_dict(orient="records")
            values = [
                (
                    rec['pipeline_id'],
                    rec['task_id'],
                    int(rec['raw_data_id']),
                    rec['min_hashes'],
                    rec['file_id'],
                    rec['filename'],
                    rec['parent_pipeline_id'],
                    rec['data']
                ) for rec in records
            ]
            insert_sql = f"""
            INSERT INTO {self.table} (pipeline_id, task_id, raw_data_id, min_hashes, file_id, filename, parent_pipeline_id, data)
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
    
    
class BatchedFileStorage(FileStorage):
    """
    批量文件存储，支持按批次读写数据。
    """
    def __init__(
        self, 
        first_entry_file_name: str,
        cache_path:str="./cache",
        file_name_prefix:str="dataflow_cache_step",
        cache_type:Literal["jsonl", "csv"] = "jsonl",
    ):
        super().__init__(first_entry_file_name, cache_path, file_name_prefix, cache_type)
        self.batch_size = None
        self.batch_step = 0
        self._dataframe_buffer = {}
        if cache_type not in ["jsonl", "csv"]:
            raise ValueError(f"BatchedFileStorage only supports 'jsonl' and 'csv' cache types, got: {cache_type}")
        
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
        if self._dataframe_buffer.get(self.operator_step) is not None:
            dataframe = self._dataframe_buffer[self.operator_step].copy()
        else:
            dataframe = self._load_local_file(file_path, local_cache)
            self._dataframe_buffer[self.operator_step] = dataframe.copy()
        self.record_count = len(dataframe)
        # 读出当前批次数据
        if self.batch_size:
            dataframe = dataframe.iloc[
                self.batch_step * self.batch_size : (self.batch_step + 1) * self.batch_size
            ]
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
            dataframe = data.map(clean_surrogates)
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")

        file_path = self._get_cache_file_path(self.operator_step + 1)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        self.logger.success(f"Writing data to {file_path} with type {self.cache_type}")
        if self.cache_type == "jsonl":
            open_mode = 'w' if self.batch_step == 0 else 'a'
            with open(file_path, open_mode, encoding='utf-8') as f:
                dataframe.to_json(f, orient="records", lines=True, force_ascii=False)
        elif self.cache_type == "csv":
            if self.batch_step == 0:
                dataframe.to_csv(file_path, index=False)
            else:
                dataframe.to_csv(file_path, index=False, header=False, mode='a')
        else:
            raise ValueError(f"Unsupported file type: {self.cache_type}, output file should end with jsonl, csv")
        
        return file_path

class StreamBatchedFileStorage(BatchedFileStorage):
    """
    流式批量文件存储，支持按批次读写数据。
    """
    def __init__(
        self, 
        first_entry_file_name: str,
        cache_path:str="./cache",
        file_name_prefix:str="dataflow_cache_step",
        cache_type:Literal["jsonl", "csv"] = "jsonl",
    ):
        super().__init__(first_entry_file_name, cache_path, file_name_prefix, cache_type)
    
    def get_keys_from_dataframe(self) -> list[str]:
        if self._dataframe_buffer.get(self.operator_step) is not None:
            return self._dataframe_buffer[self.operator_step].columns.tolist()

        file_path = self._get_cache_file_path(self.operator_step)
        ext = self.cache_type if self.operator_step != 0 else file_path.split(".")[-1].lower()
        try:
            if ext == "csv":
                return pd.read_csv(file_path, nrows=0).columns.tolist()
            
            elif ext == "jsonl":
                with open(file_path, 'r', encoding='utf-8') as f:
                    line = f.readline()
                    if line:
                        return list(json.loads(line).keys())
                return []

            elif ext == "parquet":
                # Parquet 格式在 Footer 存有 Schema，读取开销几乎为零
                import pyarrow.parquet as pq
                return pq.ParquetFile(file_path).schema.names
                
        except Exception as e:
            self.logger.warning(f"Failed to read header from {file_path} directly: {e}. Falling back to read().")
        return []

    def _load_local_file(self, file_path: str, file_type: str,batchsize: int=None) -> pd.DataFrame:
        """Load data from local file based on file type."""
        # check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} does not exist. Please check the path.")

        try:
            if file_type == "jsonl":
                return pd.read_json(file_path, lines=True, chunksize=batchsize)
            elif file_type == "csv":
                return pd.read_csv(file_path, chunksize=batchsize)
            return super()._load_local_file(file_path, file_type)
        except Exception as e:
            raise ValueError(f"Failed to load {file_type} file: {str(e)}")

    def get_record_count(self) -> int:
        """显式获取记录总数，支持缓存以提高性能"""
        if hasattr(self, "_cached_record_count"):
            return self._cached_record_count

        file_path = self._get_cache_file_path(self.operator_step)
        local_cache = self.cache_type if self.operator_step != 0 else file_path.split(".")[-1]

        if local_cache == "parquet":
            import pyarrow.parquet as pq
            count = pq.ParquetFile(file_path).metadata.num_rows
        elif local_cache in ["jsonl", "csv"]:
            count = 0
            with open(file_path, 'rb') as f:
                for _ in f: count += 1
            if local_cache == "csv": count -= 1 # 扣除表头
        else:
            df = self._load_local_file(file_path, local_cache)
            count = len(df)

        self._cached_record_count = count
        return count

    def read(self, output_type: Literal["dataframe", "dict"]="dataframe") -> Any:

        if hasattr(self, "_current_streaming_chunk") and self._current_streaming_chunk is not None:
            return self._convert_output(self._current_streaming_chunk, output_type)
        
        if self.cache_type not in ["jsonl", "csv"]:
            self.logger.warning(
                f"Current cache_type '{self.cache_type}' does not support optimized streaming. "
                f"The storage will fall back to loading the full file into memory."
            )
        return super().read(output_type)
    
    def iter_chunks(self) -> Generator[pd.DataFrame, None, None]:
        """
        专门供 Pipeline 调用：产生原始的 DataFrame 流。
        """
        file_path = self._get_cache_file_path(self.operator_step)
        local_cache = self.cache_type if self.operator_step != 0 else file_path.split(".")[-1]
        
        # 这里的 _load_local_file 必须确保在 batch_size 不为 None 时返回迭代器
        reader = self._load_local_file(file_path, local_cache, self.batch_size)
        
        # 确保 reader 是可迭代的
        if hasattr(reader, "__iter__") or isinstance(reader, pd.io.json._json.JsonReader):
            for chunk in reader:
                yield chunk
        else:
            yield reader # 如果不支持流式，就产出整个 DF