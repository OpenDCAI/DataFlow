from tqdm import tqdm
from hashlib import md5, sha256
from xxhash import xxh3_128
from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage
from dataflow.utils.registry import OPERATOR_REGISTRY

@OPERATOR_REGISTRY.register()
class NgramHashDeduplicateFilter(OperatorABC):
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
        if self.n_gram < 1:
            raise ValueError(f"Invalid value for 'n_gram'. Must be >= 1, but got '{self.n_gram}'.")
        if self.diff_size < 1:
            raise ValueError(f"Invalid value for 'diff_size'. Must be >= 1, but got '{self.diff_size}'.")
        self.logger.info(f"Initializing {self.__class__.__name__} with n_gram = {self.n_gram}, hash_func = {self.hash_func}, diff_size = {self.diff_size}...")
    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "结合n-gram技术与哈希算法识别相似文本，实现近似去重。将文本分割为多个n-gram片段，计算每个片段的哈希值，通过比较哈希集合的相似度来判断文本相似性。\n"
                "输入参数：\n"
                "- n_gram：将文本分割的片段数量\n"
                "- hash_func：哈希函数类型，支持'md5'、'sha256'和'xxh3'\n"
                "- diff_size：哈希集合差异阈值，小于此值判定为相似文本\n"
                "- input_key / input_keys：输入文本字段名（单个字符串）或字段名列表（多个字段会拼接后比较），二者必须且只能指定一个\n"
                "- output_key：写入去重标签的输出字段名，默认为'minhash_deduplicated_label'\n"
                "输出参数：\n"
                "- 去重后的DataFrame，仅保留唯一文本；"
                "每行新增output_key字段，值为整数标签（1=首次出现被保留，0=判定为重复；由于只保留标签为1的行，输出中该值均为1），该字段保留在输出中\n"
                "- 返回包含output_key的列表"
            )
        else:
            return (
                "Detect similar text using n-gram technology and hashing algorithm for near deduplication. Splits text into multiple n-gram segments, computes hash values for each segment, and judges text similarity by comparing hash set similarity.\n"
                "Input Parameters:\n"
                "- n_gram: Number of segments to split text into\n"
                "- hash_func: Hash function type, supporting 'md5', 'sha256', and 'xxh3'\n"
                "- diff_size: Hash set difference threshold below which texts are considered similar\n"
                "- input_key / input_keys: Input text field name (string) or list of field names (multiple fields are concatenated before comparison); exactly one of the two must be given\n"
                "- output_key: Name of the output field that receives the deduplication label, default is 'minhash_deduplicated_label'\n\n"
                "Output Parameters:\n"
                "- Deduplicated DataFrame containing only unique texts; "
                "each row gets a new output_key field holding an integer label (1 = first occurrence kept, 0 = duplicate; since only label-1 rows are kept, it is 1 in the output), which is kept in the output\n"
                "- List containing output_key"
            )

    def _compute_hash(self, text: str) -> str:
        return self.hash_func_dict[self.hash_func](text.encode('utf-8')).hexdigest()

    def _split_chunks(self, text: str) -> list:
        """Split text into n_gram contiguous chunks that cover every character.

        The chunk count is fixed at n_gram, so the chunks cannot all have the same
        length unless len(text) is a multiple of n_gram. Boundaries are rounded
        instead of using a single integer chunk size, which would drop the
        len(text) % n_gram trailing characters and make texts that differ only
        in their tail look identical. Texts no longer than n_gram become a single
        chunk so that short texts still hash to something that distinguishes them.
        """
        if len(text) <= self.n_gram:
            return [text]
        step = len(text) / self.n_gram
        bounds = [round(i * step) for i in range(self.n_gram + 1)]
        return [text[bounds[i]:bounds[i + 1]] for i in range(self.n_gram)]

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
        # Inverted index: chunk hash -> rows kept so far that contain it. Comparing
        # a new row against every kept row is quadratic; walking the postings of the
        # row's own chunks is proportional to the rows that actually share a chunk.
        postings = {}
        dataframe = storage.read("dataframe")
        labels = [0] * len(dataframe)
        for idx, sample in tqdm(enumerate(dataframe.to_dict(orient='records')), desc=f"Implementing {self.__class__.__name__}", total=len(dataframe)):
            if input_keys is not None and len(input_keys) > 1:
                text = '\n'.join([f"{k}:\n{sample[k]}" for k in input_keys])
            else:
                text = sample[self.input_key if input_keys is None else input_keys[0]]
            hash_value = set(self._compute_hash(chunk) for chunk in self._split_chunks(text))
            overlaps = {}
            is_duplicate = False
            for chunk_hash in hash_value:
                for kept_idx in postings.get(chunk_hash, ()):
                    overlaps[kept_idx] = overlaps.get(kept_idx, 0) + 1
                    if overlaps[kept_idx] >= self.diff_size:
                        is_duplicate = True
                        break
                if is_duplicate:
                    break
            if not is_duplicate:
                labels[idx] = 1
                for chunk_hash in hash_value:
                    postings.setdefault(chunk_hash, set()).add(idx)
        dataframe[self.output_key] = labels
        filtered_dataframe = dataframe[(dataframe[self.output_key] > 0)]
        output_file = storage.write(filtered_dataframe)
        self.logger.info(f"Deduplication completed. Total unique items: {sum(labels)}")
        return [self.output_key,]
        
        

        
        

