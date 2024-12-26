# from torch.utils.data import Dataset
from .dataflow_dataset import DataFlowDataset, DataFlowSubset
from ..utils.json_utils import read_json_file
from PIL import Image
import os
import numpy as np

def void_preprocess(x):
    return x

class ImageDataset(DataFlowDataset):
    def __init__(self, dataset, image_key, image_folder_path, id_key=None):
        super().__init__()
        self.dataset = dataset
        self.image_key = image_key
        self.image_folder_path = image_folder_path
        # self.id_key = id_key
        self.image_preprocess = void_preprocess

    # def set_image_preprocess(self, preprocess):
    #     if preprocess is not None:
    #         self.image_preprocess = preprocess

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image_path = self.dataset[idx][self.image_key]
        image = Image.open(os.path.join(self.image_folder_path, image_path)).convert("RGB")
        # if self.id_key is None:
        #     id = idx
        # else:
        #     id = self.dataset[idx][self.id_key]
        return self.image_preprocess(image)
    
    # def filter(self, labels):
    #     print("***call ImageDataset")
    #     indices = np.where(labels == 1)[0]
    #     return ImageSubset(self, indices.tolist()) 
    
    def get_dump_data(self):
        return self.dataset

# class ImageSubset(ImageDataset, DataFlowSubset):
#     def __init__(self, dataset, indices):
#         DataFlowSubset.__init__(self, dataset, indices)

#     def set_image_preprocess(self, preprocess):
#         self.dataset.set_image_preprocess(preprocess)

#     def __getitem__(self, idx):
#         return DataFlowSubset.__getitem__(self, idx)
    
#     def __len__(self):
#         return DataFlowSubset.__len__(self)
    
#     def filter(self, labels):
#         print("***call ImageSubset")
#         return ImageDataset.filter(labels)

class ImageCaptionDataset(DataFlowDataset):
    def __init__(self, dataset, image_key, text_key, image_folder_path, id_key=None):
        super().__init__()
        self.dataset = dataset
        self.image_key = image_key
        self.text_key = text_key
        self.image_folder_path = image_folder_path
        # self.id_key = id_key
        self.image_preprocess = void_preprocess
        self.text_preprocess = void_preprocess

    # def set_image_preprocess(self, preprocess):
    #     self.image_preprocess = preprocess

    # def set_text_preprocess(self, preprocess):
    #     self.text_preprocess = preprocess

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image_path = self.dataset[idx][self.image_key]
        image = Image.open(os.path.join(self.image_folder_path, image_path)).convert("RGB")
        text = self.dataset[idx][self.text_key]
        # if self.id_key is None:
        #     id = idx
        # else:
        #     id = self.dataset[idx]['id']
        return self.image_preprocess(image), self.text_preprocess(text)
    
    def get_dump_data(self):
        return self.dataset

class jsonImageDataset(ImageDataset):
    def __init__(self, json_path, image_folder_path):
        self.json_file = read_json_file(json_path)
        self.image_folder_path = image_folder_path
        self.preprocess = void_preprocess

    def __len__(self):
        return len(self.json_file)

    def __getitem__(self, idx):
        image_path = self.json_file[idx]['image']
        image = Image.open(os.path.join(self.image_folder_path, image_path)).convert("RGB")
        id = self.json_file[idx]['id']
        return id, self.preprocess(image)


class jsonImageTextDataset(ImageCaptionDataset):
    def __init__(self, json_path, image_folder_path):
        self.json_file = read_json_file(json_path)
        self.image_folder_path = image_folder_path
        self.image_preprocess = void_preprocess
        self.text_preprocess = void_preprocess

    def __len__(self):
        return len(self.json_file)

    def __getitem__(self, idx):
        image_path = self.json_file[idx]['image']
        image = Image.open(os.path.join(self.image_folder_path, image_path)).convert("RGB")
        id = self.json_file[idx]['id']
        text = self.json_file[idx]['caption']
        return id, self.image_preprocess(image), self.text_preprocess(text)