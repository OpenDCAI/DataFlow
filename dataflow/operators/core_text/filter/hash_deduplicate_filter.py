from tqdm import tqdm
from hashlib import md5, sha256
from xxhash import xxh3_128
from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage
from dataflow.utils.registry import OPERATOR_REGISTRY

@OPERATOR_REGISTRY.register()
class HashDeduplicateFilter(OperatorABC):
    def __init__(self, hash_func: str = 'md5'):
        self.logger = get_logger()
        self.hash_func = hash_func
        self.hash_func_dict = {
            'md5': md5,
            'sha256': sha256,
            'xxh3': xxh3_128
        }
        
        if self.hash_func not in self.hash_func_dict:
            raise ValueError(f'Invalid hash function: {self.hash_func}')
        self.logger.info(f"Initializing {self.__class__.__name__} with hash_func = {self.hash_func}...")
    
    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "使用多种哈希函数对文本进行精确去重，支持md5、sha256或xxh3算法。通过计算文本的哈希值识别重复数据。\n\n"
                "初始化参数：\n"
                "- hash_func: 哈希函数名称，可选'md5'、'sha256'或'xxh3'，默认为'md5'\n\n"
                "运行参数：\n"
                "- input_keys: 用于计算哈希的多个字段列表（与input_key二选一）\n"
                "- input_key: 用于计算哈希的单个字段名（与input_keys二选一）\n"
                "- output_key: 去重标记字段名，默认为'minhash_deduplicated_label'\n\n"
                "输出说明：标记为1的数据表示首次出现，标记为0的数据表示重复数据\n"
                "算法特点：\n"
                "- md5: 128位哈希值，平衡速度和唯一性\n"
                "- sha256: 256位哈希值，更高安全性，速度较慢\n"
                "- xxh3: 128位哈希值，最快的哈希算法"
            )
        else:
            return (
                "Exact deduplication using multiple hash functions, chosen from md5, sha256 or xxh3. Identify duplicate data by calculating text hash values.\n\n"
                "Initialization Parameters:\n"
                "- hash_func: Hash function name, options are 'md5', 'sha256' or 'xxh3', default is 'md5'\n\n"
                "Run Parameters:\n"
                "- input_keys: List of multiple fields for hash calculation (alternative to input_key)\n"
                "- input_key: Single field name for hash calculation (alternative to input_keys)\n"
                "- output_key: Deduplication label field name, default is 'minhash_deduplicated_label'\n\n"
                "Output Description: Data marked as 1 indicates first occurrence, 0 indicates duplicate\n"
                "Algorithm Characteristics:\n"
                "- md5: 128-bit hash, balances speed and uniqueness\n"
                "- sha256: 256-bit hash, higher security, slower speed\n"
                "- xxh3: 128-bit hash, fastest hashing algorithm"
            )


    def _compute_hash(self, text: str) -> str:
        return self.hash_func_dict[self.hash_func](text.encode('utf-8')).hexdigest()

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
            hash_value = self._compute_hash(text)
            if hash_value not in seen_hashes:
                labels[idx] = 1
                seen_hashes.add(hash_value)
        dataframe[self.output_key] = labels
        filtered_dataframe = dataframe[(dataframe[self.output_key] > 0)]
        output_file = storage.write(filtered_dataframe)
        self.logger.info(f"Deduplication completed. Total unique items: {sum(labels)}")
        return [self.output_key,]
        
        

        
        

