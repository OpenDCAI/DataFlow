from tqdm import tqdm
from hashlib import md5, sha256
from xxhash import xxh3_128
from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage
from dataflow.utils.registry import OPERATOR_REGISTRY

@OPERATOR_REGISTRY.register()
class NgramHashDeduplicator(OperatorABC):
    def __init__(self, n_gram: int = 3, hash_func: str = 'md5', diff_size : int = 1):
        self.logger = get_logger()
        self.n_gram = n_gram
        self.hash_func = hash_func
        self.diff_size = diff_size
        self.hash_func_dict = {
            'md5': md5,
            'sha256': sha256,
            'xxh3': xxh3_128
        }
        
        if self.hash_func not in self.hash_func_dict:
            raise ValueError(f'Invalid hash function: {self.hash_func}')
        self.logger.info(f"Initializing {self.__class__.__name__} with n_gram = {self.n_gram}, hash_func = {self.hash_func}, diff_size = {self.diff_size}...")
        
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
        seen_hashes = []
        dataframe = storage.read("dataframe")
        labels = [0] * len(dataframe)
        for idx, sample in tqdm(enumerate(dataframe.to_dict(orient='records')), desc=f"Implementing {self.__class__.__name__}", total=len(dataframe)):
            if input_keys is not None and len(input_keys) > 1:
                text = '\n'.join([f"{k}:\n{sample[k]}" for k in input_keys])
            else:
                text = sample[self.input_key]
            gram_length = len(text) // self.n_gram
            ngrams = [text[i*gram_length:(i+1)*gram_length] for i in range(self.n_gram)]
            hash_value = set(self._compute_hash(ngram) for ngram in ngrams)
            if all(len(hash_value & hash) < self.diff_size for hash in seen_hashes):
                labels[idx] = 1
                seen_hashes.append(hash_value)
        dataframe[self.output_key] = labels
        filtered_dataframe = dataframe[(dataframe[self.output_key] > 0)]
        output_file = storage.write(filtered_dataframe)
        self.logger.info(f"Deduplication completed. Total unique items: {sum(labels)}")
        return [self.output_key,]
        
        

        
        

