from tqdm import tqdm
from datasketch import MinHash, MinHashLSH  # use datasketch-1.6.5
from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage
from dataflow.utils.registry import OPERATOR_REGISTRY

@OPERATOR_REGISTRY.register()
class MinHashDeduplicateFilter(OperatorABC):
    def __init__(self, num_perm=128, threshold=0.9, use_n_gram=True, ngram=5):
        self.logger = get_logger()
        self.num_perm = num_perm
        self.threshold = threshold
        self.use_n_gram = use_n_gram
        self.n_gram = ngram
        self.logger.info(f"Initializing {self.__class__.__name__} with num_perm = {self.num_perm}, threshold = {self.threshold}, use_n_gram = {self.use_n_gram}, ngram = {self.n_gram}...")

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "结合MinHash与LSH（局部敏感哈希）实现高效近似去重。将文本转换为MinHash签名，使用LSH快速查找相似文本，实现大规模数据集的近似去重。\n"
                "输入参数：\n"
                "- num_perm：生成MinHash签名的排列数\n"
                "- threshold：相似度阈值，超过此阈值判定为相似文本\n"
                "- use_n_gram：是否使用n-gram分词\n"
                "- ngram：n-gram的n值\n"
                "输出参数：\n"
                "- 去重后的DataFrame，仅保留唯一文本\n"
                "- 返回包含去重标签字段名的列表"
            )
        else:
            return (
                "Efficient near-duplicate detection using MinHash and LSH (Locality-Sensitive Hashing). Converts texts to MinHash signatures and uses LSH to quickly find similar texts, enabling near-deduplication for large-scale datasets.\n"
                "Input Parameters:\n"
                "- num_perm: Number of permutations for generating MinHash signatures\n"
                "- threshold: Similarity threshold above which texts are considered duplicates\n"
                "- use_n_gram: Whether to use n-gram tokenization\n"
                "- ngram: n value for n-gram\n\n"
                "Output Parameters:\n"
                "- Deduplicated DataFrame containing only unique texts\n"
                "- List containing deduplication label field name"
            )

    def create_minhash(self, data):
        minhash = MinHash(num_perm=self.num_perm)
        if self.use_n_gram:
            for i in range(len(data) - self.n_gram + 1):
                minhash.update(data[i:i + self.n_gram].encode('utf8'))
        else:
            for d in data:
                minhash.update(d.encode('utf8'))
        return minhash

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
        lsh = MinHashLSH(threshold=self.threshold, num_perm=self.num_perm)
        self.input_key = input_key
        self.input_keys = input_keys
        self.output_key = output_key
        dataframe = storage.read("dataframe")
        labels = [0] * len(dataframe)
        with lsh.insertion_session() as session:  
            for idx, sample in tqdm(enumerate(dataframe.to_dict(orient='records')), desc=f"Implementing {self.__class__.__name__}", total=len(dataframe)):
                if input_keys is not None and len(input_keys) > 1:
                    text = '\n'.join([f"{k}:\n{sample[k]}" for k in input_keys])
                else:
                    text = sample[self.input_key]
                minhash = self.create_minhash(text)
                result = lsh.query(minhash)
                
                if len(result) == 0:
                    labels[idx] = 1
                    session.insert(idx, minhash)
                    self.logger.debug(f"Inserted item {idx} into LSH with minhash.")
        dataframe[self.output_key] = labels
        filtered_dataframe = dataframe[(dataframe[self.output_key] > 0)]
        output_file = storage.write(filtered_dataframe)
        self.logger.info(f"Deduplication completed. Total unique items: {sum(labels)}")
        return [self.output_key,]
        
        

        
        

