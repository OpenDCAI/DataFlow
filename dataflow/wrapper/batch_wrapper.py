from typing import ParamSpec, TypeVar, Generic, Protocol, List
from functools import wraps
import inspect
from dataflow.logger import get_logger
from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage, DummyStorage, FileStorage
from tqdm import tqdm
import pandas as pd
P = ParamSpec("P")
R = TypeVar("R")

class HasRun(Protocol[P, R]):
    def run(self, *args: P.args, **kwargs: P.kwargs) -> R:
        """
        这里写一份通用的 run 文档也可以，
        不写的话会被下面动态拷贝原 operator.run 的 __doc__。
        """
        ...

class BatchWrapper(Generic[P, R]):
    """
    通用的批处理 Wrapper。

    在静态检查/IDE 里，BatchWrapper.run 的签名和 operator.run 完全一致。
    运行时，会把 operator.run 的 __doc__ 和 __signature__ 也拷过来，
    这样 help(bw.run) 时能看到原 operator 的文档。
    """
    def __init__(self, operator: HasRun[P, R], batch_size: int = 32, batch_cache: bool = False) -> None:
        self._operator = operator
        self._logger = get_logger()
        self._batch_size = batch_size
        self._batch_cache = batch_cache

        # 动态拷贝 operator.run 的 __doc__ 和 inspect.signature
        orig = operator.run
        sig = inspect.signature(orig)
        # wrapped = wraps(orig)(self.run)        # 先 wrap docstring, __name__…
        # wrapped.__signature__ = sig            # 再贴上准确的 signature
        # 把它绑到实例上覆盖掉 class method，这样 help(instance.run) 能看到原文档
        # object.__setattr__(self, "run", wrapped)

    def run(self, *args: P.args, **kwargs: P.kwargs) -> List[R]:
        # —— 1. 提取 storage —— 
        # only support FileStorage for now
        if args:
            storage: FileStorage = args[0]    # type: ignore[assignment]
            rest_args = args[1:]
            rest_kwargs = kwargs
        else:
            storage: FileStorage = kwargs.get("storage")      # type: ignore[assignment]
            if storage is None:
                raise ValueError(
                    f"A DataFlowStorage is required for {self._operator!r}.run()"
                )
            rest_kwargs = {k: v for k, v in kwargs.items() if k != "storage"}
            rest_args = ()
            
        # prepare a dummy storage，for batch processing
        if self._batch_cache: # if we need to cache each batch result
            self._dummy_storage = DummyStorage(
                cache_path=storage.cache_path,
                file_name_prefix=storage.file_name_prefix,
                cache_type=storage.cache_type
            )
        else:  # if we don't need to cache each batch result
            self._dummy_storage = DummyStorage()

        # —— 2. 读出全量数据并按 batch_size 切分 —— 
        whole_dataframe = storage.read()
        num_batches = (len(whole_dataframe) + self._batch_size - 1) // self._batch_size  # Calculate number of batches

        self._logger.info(f"Total {len(whole_dataframe)} items, will process in {num_batches} batches of size {self._batch_size}.")
        for batch_num in tqdm(range(num_batches)):
            start_index = batch_num * self._batch_size
            end_index = min((batch_num + 1) * self._batch_size, len(whole_dataframe))
            batch_df = whole_dataframe.iloc[start_index:end_index]
            # Clear and write the current batch
            self._dummy_storage.set_data(batch_df)
            self._dummy_storage.set_file_name_prefix(
                f"{storage.file_name_prefix}_step{storage.operator_step}_batch{batch_num}"
            )
            # Run the operator with the dummy storage
            self._logger.info(f"Running batch with {len(batch_df)} items...")
            self._operator.run(self._dummy_storage, *rest_args, **rest_kwargs)

            res: pd.DataFrame = self._dummy_storage.read()

            # Find columns in res that are not in whole_dataframe
            new_cols = [c for c in res.columns if c not in whole_dataframe.columns]

            # Create new columns in whole_dataframe with NaN values
            for c in new_cols:
                whole_dataframe[c] = pd.NA

            # Write the values from res back to whole_dataframe
            whole_dataframe.loc[res.index, res.columns] = res

        storage.write(whole_dataframe)