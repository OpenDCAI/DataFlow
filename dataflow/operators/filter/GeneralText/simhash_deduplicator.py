from tqdm import tqdm
from simhash import Simhash
from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage
from dataflow.utils.registry import OPERATOR_REGISTRY

def get_similarity(simhash, another_simhash):
    max_hashbit = max(len(bin(simhash.value)), len(bin(another_simhash.value)))
    distince = simhash.distance(another_simhash)
    similar = 1 - distince / max_hashbit
    return similar

@OPERATOR_REGISTRY.register()
class SimHashDeduplicator(OperatorABC):
    def __init__(self, fingerprint_size: int = 64, bound: float = 0.1):
        self.logger = get_logger()
        self.fingerprint_size = fingerprint_size
        self.bound = bound
        self.logger.info(f"Initializing {self.__class__.__name__} with fingerprint_size = {self.fingerprint_size}, bound = {self.bound}...")

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
        dataframe = storage.read("dataframe")
        simhashes = []
        labels = [0] * len(dataframe)
        for idx, sample in tqdm(enumerate(dataframe.to_dict(orient='records')), desc=f"Implementing {self.__class__.__name__}", total=len(dataframe)):
            if input_keys is not None and len(input_keys) > 1:
                text = '\n'.join([f"{k}:\n{sample[k]}" for k in input_keys])
            else:
                text = sample[self.input_key]
            simhash = Simhash(text, f=self.fingerprint_size)
            if all(get_similarity(simhash, another_simhash) < 1 - self.bound for another_simhash in simhashes):
                labels[idx] = 1
                simhashes.append(simhash)
        dataframe[self.output_key] = labels
        filtered_dataframe = dataframe[(dataframe[self.output_key] > 0)]
        output_file = storage.write(filtered_dataframe)
        self.logger.info(f"Deduplication completed. Total unique items: {sum(labels)}")
        return [self.output_key,]
        
        

        
        

