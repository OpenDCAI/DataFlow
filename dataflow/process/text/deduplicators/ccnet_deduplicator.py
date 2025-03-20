# 比较SHA-1数字前64位 CCNet
from dataflow.core import TextDeduplicator
from dataflow.utils.registry import PROCESSOR_REGISTRY
from dataflow.utils.text_utils import sha1_hash
from tqdm import tqdm
import json

@PROCESSOR_REGISTRY.register()
class CCNetDeduplicator(TextDeduplicator):
    def __init__(self, args_dict: dict):
        super().__init__(args_dict)
        self.dedupliactor_name = 'CCNetDeduplicator'
        self.bit_length = args_dict.get('bit_length', 64)
    
    def _compute_hash(self, text: str) -> str:
        return sha1_hash(text, self.bit_length)

    def dedup_func(self, dataset):
        hash_values = []
        for idx, sample in tqdm(enumerate(dataset), desc=f"Implementing {self.dedupliactor_name}", total=len(dataset)):
            if isinstance(dataset.keys, list):
                text = " ".join([str(sample[key]) for key in dataset.keys])
                text = text.encode('utf-8')
            else:
                text = str(sample[dataset.keys]).encode('utf-8')
            hash_value = self._compute_hash(text)
            hash_values.append(hash_value)
        print(json.dumps({"hash_values": hash_values}))
        return hash_values

    