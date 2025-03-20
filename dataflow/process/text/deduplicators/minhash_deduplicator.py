from dataflow.core import TextDeduplicator
from dataflow.utils.registry import PROCESSOR_REGISTRY
from datasketch import MinHash, MinHashLSH  # use datasketch-1.6.5
from tqdm import tqdm
import json
from collections.abc import Sequence


@PROCESSOR_REGISTRY.register()
class MinHashDeduplicator(TextDeduplicator):
    def __init__(self, args_dict: dict):
        super().__init__(args_dict)
        self.dedupliactor_name = 'MinHashDeduplicator'
        self.num_perm = args_dict.get('num_perm', 128)
        self.threshold = args_dict.get('threshold', 0.9)
        self.use_n_gram = args_dict.get('use_n_gram', True) 
        self.n_gram = args_dict.get('n_gram', 5) 

    def create_minhash(self, data):
        minhash = MinHash(num_perm=self.num_perm)
        if self.use_n_gram:
            for i in range(len(data) - self.n_gram + 1):
                minhash.update(data[i:i + self.n_gram].encode('utf8'))
        else:
            for d in data:
                minhash.update(d.encode('utf8'))
        return minhash

    def dedup_func(self, dataset):
        lsh = MinHashLSH(threshold=self.threshold, num_perm=self.num_perm)
        hash_values = []
        with lsh.insertion_session() as session:  
            for idx, sample in tqdm(enumerate(dataset), desc=f"Implementing {self.dedupliactor_name}", total=len(dataset)):
                text = str(sample[dataset.keys])
                minhash = self.create_minhash(text)
                result = lsh.query(minhash)
                hash_values.append(result)
        print(json.dumps({"hash_values": hash_values}))
        return json.dumps({"hash_values": hash_values})

        

        
        

