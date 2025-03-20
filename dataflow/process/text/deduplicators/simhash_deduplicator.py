from dataflow.core import TextDeduplicator
from dataflow.utils.registry import PROCESSOR_REGISTRY
from collections import defaultdict
from typing import List
from dataflow.utils.text_utils import md5_digest, sha256_digest, xxh3_128_digest
from simhash import Simhash
from tqdm import tqdm
import json

@PROCESSOR_REGISTRY.register()
class SimHashDeduplicator(TextDeduplicator):
    def __init__(self, args_dict: dict):
        super().__init__(args_dict)
        self.deduplicator_name = 'SimHashDeduplicator'
        self.fingerprint_size = args_dict.get('fingerprint_size', 64)
        self.bound = args_dict.get('bound', 0.1)


    def dedup_func(self, dataset):
        simhashes = []
        labels = [0] * len(dataset)
        def get_similarity(simhash, another_simhash):
            max_hashbit = max(len(bin(simhash.value)), len(bin(another_simhash.value)))
            distince = simhash.distance(another_simhash)
            similar = 1 - distince / max_hashbit
            return similar
        for idx, sample in tqdm(enumerate(dataset), desc=f"Implementing {self.dedupliactor_name}", total=len(dataset)):
            if isinstance(dataset.keys, list):
                text = " ".join([str(sample[key]) for key in dataset.keys])
            else:
                text = str(sample[dataset.keys]) 
            simhash = Simhash(text, f=self.fingerprint_size)
            simhashes.append(simhash)
        print(json.dumps({"hash_values": [simhash.value for simhash in simhashes]}))
        return json.dumps({"hash_values": [simhash.value for simhash in simhashes]})
        

