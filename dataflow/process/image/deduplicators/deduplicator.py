from dataflow.core import ImageDeduplicator
from dataflow.data import DataFlowSubset
from dataflow.utils.registry import PROCESSOR_REGISTRY
import numpy as np

def dedup(dataset, hasher, threshold):
    encoding_map = {}
    for i in range(len(dataset)):
        encoding_map[str(i)] = hasher.encode_image(image_array=np.asarray(dataset[i]))
    dup = hasher.find_duplicates(encoding_map=encoding_map, max_distance_threshold=threshold)
    ban = [0] * len(dataset)
    unique_indices = []
    for i in range(len(dataset)):
        if ban[i]:
            continue
        unique_indices.append(i)
        for j in dup[str(i)]:
            ban[int(j)] = 1
    return unique_indices

@PROCESSOR_REGISTRY.register()
class ImagePHashDeduplicator(ImageDeduplicator):
    def __init__(self, args_dict: dict):
        super().__init__(args_dict)
        self.threshold = args_dict["threshold"]

    def dedup_func(self, dataset):
        from imagededup.methods import PHash    
        return DataFlowSubset(dataset, dedup(dataset, PHash(), self.threshold))

@PROCESSOR_REGISTRY.register()
class ImageDHashDeduplicator(ImageDeduplicator):
    def __init__(self, args_dict: dict):
        super().__init__(args_dict)
        self.threshold = args_dict["threshold"]

    def dedup_func(self, dataset):
        from imagededup.methods import DHash    
        return DataFlowSubset(dataset, dedup(dataset, DHash(), self.threshold))

@PROCESSOR_REGISTRY.register()
class ImageWHashDeduplicator(ImageDeduplicator):
    def __init__(self, args_dict: dict):
        super().__init__(args_dict)
        self.threshold = args_dict["threshold"]

    def dedup_func(self, dataset):
        from imagededup.methods import WHash    
        return DataFlowSubset(dataset, dedup(dataset, WHash(), self.threshold))

@PROCESSOR_REGISTRY.register()
class ImageAHashDeduplicator(ImageDeduplicator):
    def __init__(self, args_dict: dict):
        super().__init__(args_dict)
        self.threshold = args_dict["threshold"]

    def dedup_func(self, dataset):
        from imagededup.methods import AHash    
        return DataFlowSubset(dataset, dedup(dataset, AHash(), self.threshold))
