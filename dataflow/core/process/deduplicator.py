from datasets import Dataset

class Deduplicator:

    def __init__(self, args):
        pass

    def dedup_func(self, dataset):
        raise NotImplementedError

    def __call__(self, dataset):
        init_len = len(dataset)
        deduped_dataset = self.dedup_func(dataset)
        print(f'Implemented {self.__class__.__name__}. Data Number: {init_len} -> {len(deduped_dataset)}', flush=True)
        
        return deduped_dataset

class TextDeduplicator(Deduplicator):

    def __init__(self, args=None):
        self.data_type = "text"
        
    def __call__(self, dataset):
        init_len = len(dataset)
        labels = self.dedup_func(dataset)
        if isinstance(dataset.dataset, Dataset):
            def filter_by_labels(example, index):
                return labels[index] == 1
            dataset.dataset = dataset.dataset.filter(filter_by_labels, with_indices=True)
            deduped_dataset = dataset
        else:
            deduped_dataset = dataset.filter(labels)
        print(f'Implemented {self.dedupliactor_name}. Data Number: {init_len} -> {len(deduped_dataset)}')
        return deduped_dataset

class ImageDeduplicator(Deduplicator):

    def __init__(self, args=None):
        self.data_type = "image"
