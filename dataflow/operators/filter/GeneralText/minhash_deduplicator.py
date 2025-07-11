from tqdm import tqdm
from datasketch import MinHash, MinHashLSH  # use datasketch-1.6.5
from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.operators.eval import NgramScorer


@OPERATOR_REGISTRY.register()
class MinHashDeduplicator(OperatorABC):
    def __init__(self, num_perm=128, threshold=0.9, use_n_gram=True, ngram=5):
        self.logger = get_logger()
        self.num_perm = num_perm
        self.threshold = threshold
        self.use_n_gram = use_n_gram
        self.n_gram = ngram
        self.logger.info(f"Initializing {self.__class__.__name__} with num_perm = {self.num_perm}, threshold = {self.threshold}, use_n_gram = {self.use_n_gram}, ngram = {self.n_gram}...")

    @staticmethod
    def get_desc(lang: str = "zh"):
        return "使用MinHash算法进行文本去重" if lang == "zh" else "Deduplicate text using the MinHash algorithm."

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
        
        

        
        

