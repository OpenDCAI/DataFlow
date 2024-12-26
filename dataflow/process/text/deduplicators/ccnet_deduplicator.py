# 比较SHA-1数字前64位 CCNet
from dataflow.core import TextDeduplicator
from dataflow.utils.registry import PROCESSOR_REGISTRY
from dataflow.utils.text_utils import sha1_hash
from tqdm import tqdm

@PROCESSOR_REGISTRY.register()
class CCNetDeduplicator(TextDeduplicator):
    def __init__(self, args_dict: dict):
        super().__init__(args_dict)
        self.dedupliactor_name = 'CCNetDeduplicator'
        self.bit_length = args_dict.get('bit_length', 64)
    
    def _compute_hash(self, text: str) -> str:
        return sha1_hash(text, self.bit_length)

    def dedup_func(self, dataset):
        seen_hashes = set()
        labels = [0] * len(dataset)
        for idx, sample in tqdm(enumerate(dataset), desc=f"Implementing {self.dedupliactor_name}", total=len(dataset)):
            if isinstance(dataset.keys, list):
                text = " ".join([str(sample[key]) for key in dataset.keys])
                text = text.encode('utf-8')
            else:
                text = str(sample[dataset.keys]).encode('utf-8')
            hash_value = self._compute_hash(text)
            if hash_value not in seen_hashes:
                labels[idx] = 1
                seen_hashes.add(hash_value)
        return labels
    