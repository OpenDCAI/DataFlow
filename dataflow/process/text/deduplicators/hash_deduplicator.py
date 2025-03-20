from dataflow.core import TextDeduplicator
import json
import hashlib
from dataflow.utils.registry import PROCESSOR_REGISTRY
from dataflow.utils.text_utils import md5, sha256, xxh3_128
from tqdm import tqdm

@PROCESSOR_REGISTRY.register()
class HashDeduplicator(TextDeduplicator):
    def __init__(self, args_dict: dict):
        super().__init__(args_dict)
        self.dedupliactor_name = 'HashDeduplicator'
        self.hash_func = args_dict.get('hash_func', 'md5')
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

            hash_value = self._compute_hash(text)
            hash_values.append(hash_value)
        print(json.dumps({"hash_values": hash_values}))
        return json.dumps({"hash_values": hash_values})
