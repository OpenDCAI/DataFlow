import numpy as np
from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.operators.eval import DebertaV3Scorer

@OPERATOR_REGISTRY.register()
class DebertaV3Filter(OperatorABC):

    def __init__(self, allowed_scores : list = ['Medium', 'High'], model_name='nvidia/quality-classifier-deberta', model_cache_dir='./dataflow_cache', device='cuda', batch_size=16):
        self.logger = get_logger()
        self.allowed_scores = allowed_scores
        self.scorer = DebertaV3Scorer(
            model_name=model_name,
            model_cache_dir=model_cache_dir,
            device=device,
            batch_size=batch_size,
        )
        self.logger.info(f"Initializing {self.__class__.__name__} with allowed_scores = {self.allowed_scores}...")
        
    def run(self, storage: DataFlowStorage, input_key: str, output_key: str = 'Debertav3Score'):
        self.input_key = input_key
        self.output_key = output_key
        dataframe = storage.read("dataframe")
        scores = self.scorer.eval(dataframe, self.input_key)
        dataframe[self.output_key] = scores
        labels = np.array([1 if score in self.allowed_scores else 0 for score in scores])
        filtered_dataframe = dataframe[labels == 1]
        output_file = storage.write(filtered_dataframe)
        self.logger.info(f"Filtering completed. Total records passing filter: {len(filtered_dataframe)}.")
        return [self.output_key]
        
        