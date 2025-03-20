from dataflow.core import TextDeduplicator
from dataflow.utils.registry import PROCESSOR_REGISTRY
from dataflow.utils.text_utils import md5, sha256, xxh3_128
from tqdm import tqdm
import json

@PROCESSOR_REGISTRY.register()
class NgramHashDeduplicator(TextDeduplicator):
    def __init__(self, args_dict: dict):
        super().__init__(args_dict)
        self.dedupliactor_name = 'NgramHashDeduplicator'
        self.n_gram = args_dict.get('n_gram', 3)
        self.hash_func = args_dict.get('hash_func', 'md5')
        self.diff_size = args_dict.get('diff_size', 1) # 有diff_size个hash值不同，则认为不同
        self.hash_func_dict = {
            'md5': md5,
            'sha256': sha256,
            'xxh3': xxh3_128
        }

    def _compute_hash(self, text: str) -> str:
        return self.hash_func_dict[self.hash_func](text.encode('utf-8')).hexdigest()

    def dedup_func(self, dataset):
        hash_values = []
        for idx, sample in tqdm(enumerate(dataset), desc=f"Implementing {self.dedupliactor_name}", total=len(dataset)):
            if isinstance(dataset.keys, list):
                text = " ".join([str(sample[key]) for key in dataset.keys])
            else:
                text = str(sample[dataset.keys])
            gram_length = len(text) // self.n_gram
            ngrams = [text[i*gram_length:(i+1)*gram_length] for i in range(self.n_gram)]
            hash_value = set(self._compute_hash(ngram) for ngram in ngrams)
            hash_values.append(hash_value)
        print(json.dumps({"hash_values": hash_values}))
        return json.dumps({"hash_values": hash_values})


                
