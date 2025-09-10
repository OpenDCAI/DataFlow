import hashlib
import struct
from tqdm import tqdm
from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage
from dataflow.utils.registry import OPERATOR_REGISTRY

def sha1_hash(data: bytes, d: int = 32) -> int:
    """
    Generate a d-bit hash value from the given data.

    Parameters
    ----------
    data : bytes
        The data to be hashed.
    d : int
        The number of bits of the hash value.

    Returns
    -------
    int
        The hash value.

    Examples
    --------
    >>> sha1_hash(b"hello world", 32)
    896314922
    >>> sha1_hash(b"hello world", 64)
    13028719972609469994
    >>> sha1_hash(b"hello world", 128)
    310522945683037930239412421226792791594
    """
    if d == 32:
        return struct.unpack("<I", hashlib.sha1(data, usedforsecurity=False).digest()[:4])[0]
    if d == 64:
        return struct.unpack("<Q", hashlib.sha1(data, usedforsecurity=False).digest()[:8])[0]
    # struct is faster but does not support arbitrary bit lengths
    return int.from_bytes(hashlib.sha1(data, usedforsecurity=False).digest()[: d // 8], byteorder="little")


@OPERATOR_REGISTRY.register()
class CCNetDeduplicateFilter(OperatorABC):
    
    def __init__(self, bit_length: int = 64):
        self.logger = get_logger()
        self.bit_length = bit_length
        self.logger.info(f"Initializing {self.__class__.__name__} with bit length = {bit_length}...")

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "CCNet去重方法，基于SHA-1哈希算法的前N位进行重复识别，实现精确去重。\n\n"
                "初始化参数：\n"
                "- bit_length: 哈希值的位数，默认为64位\n\n"
                "运行参数：\n"
                "- input_keys: 用于计算哈希的多个字段列表（与input_key二选一）\n"
                "- input_key: 用于计算哈希的单个字段名（与input_keys二选一）\n"
                "- output_key: 去重标记字段名，默认为'minhash_deduplicated_label'\n\n"
                "输出说明：标记为1的数据表示首次出现，标记为0的数据表示重复数据"
            )
        else:
            return (
                "CCNet deduplication method. Identify duplicates using first N bits of SHA-1 hash for exact deduplication.\n\n"
                "Initialization Parameters:\n"
                "- bit_length: Number of bits for hash value, default is 64\n\n"
                "Run Parameters:\n"
                "- input_keys: List of multiple fields for hash calculation (alternative to input_key)\n"
                "- input_key: Single field name for hash calculation (alternative to input_keys)\n"
                "- output_key: Deduplication label field name, default is 'minhash_deduplicated_label'\n\n"
                "Output Description: Data marked as 1 indicates first occurrence, 0 indicates duplicate"
            )

    def _compute_hash(self, text: str) -> str:
        return sha1_hash(text, self.bit_length)

    def run(self, storage: DataFlowStorage, input_keys: list = None, input_key: str = None, output_key: str = 'minhash_deduplicated_label'):
        if input_keys is None and input_key is None:
            self.logger.error(f"Need to specify either input_keys or input_key!")
            raise ValueError(f"Need to specify either input_keys or input_key!")
        if input_keys is not None and input_key is not None:
            self.logger.error(f"{self.__class__.__name__} only need one input args!")
            raise ValueError(f"{self.__class__.__name__} only need one input args!")
        if input_keys is not None:
            self.logger.info(f"Running {self.__class__.__name__} with input_keys = {input_keys} and output_key = {output_key}")
        else:
            self.logger.info(f"Running {self.__class__.__name__} with input_key = {input_key} and output_key = {output_key}")
        self.input_key = input_key
        self.input_keys = input_keys
        self.output_key = output_key
        seen_hashes = set()
        dataframe = storage.read("dataframe")
        labels = [0] * len(dataframe)
        for idx, sample in tqdm(enumerate(dataframe.to_dict(orient='records')), desc=f"Implementing {self.__class__.__name__}", total=len(dataframe)):
            if input_keys is not None and len(input_keys) > 1:
                text = '\n'.join([f"{k}:\n{sample[k]}" for k in input_keys])
            else:
                text = sample[self.input_key]
            text = text.encode('utf-8')
            hash_value = self._compute_hash(text)
            if hash_value not in seen_hashes:
                labels[idx] = 1
                seen_hashes.add(hash_value)
        dataframe[self.output_key] = labels
        filtered_dataframe = dataframe[(dataframe[self.output_key] > 0)]
        output_file = storage.write(filtered_dataframe)
        self.logger.info(f"Deduplication completed. Total unique items: {sum(labels)}")
        return [self.output_key,]
        
        

        
        

