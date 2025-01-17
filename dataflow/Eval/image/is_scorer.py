# modified from https://github.com/sbarratt/inception-score-pytorch

from dataflow.core.scorer import GenImageScorer
import json
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data
import json
import os
from PIL import Image
from torchvision.models.inception import inception_v3
import numpy as np
from scipy.stats import entropy
import torchvision.datasets as dset
import torchvision.transforms as transforms

from dataflow.utils.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register()
class ISScorer(GenImageScorer):
    def __init__(self, args_dict: dict):
        super().__init__(args_dict)
        self.batch_size = args_dict.get('batch_size')
        self.device = args_dict.get('device')
        self.resize = args_dict.get('resize')
        self.splits = args_dict.get('splits')
        self.image_preprocessor = self.get_image_preprocessor()
        self.data_type = "image"
        self.scorer_name = "ISScorer"

    def inception_score(self, sample, device, batch_size=32, resize=False, splits=1):
        """Computes the inception score of the generated images imgs

        imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
        cuda -- whether or not to run on GPU
        batch_size -- batch size for feeding into Inception v3
        splits -- number of splits
        """
        N = len(sample)

        assert batch_size > 0
        if N > batch_size:
            batch_size = N

        if device == 'cuda':
            dtype = torch.cuda.FloatTensor
        else:
            if torch.cuda.is_available():
                print("WARNING: You have a CUDA device, so you should probably set cuda=True")
            dtype = torch.FloatTensor

        dataloader = torch.utils.data.DataLoader(sample, batch_size=batch_size)

        # Load inception model
        inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
        inception_model.eval()
        up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)

        def get_pred(x):
            if resize:
                x = up(x)
            x = inception_model(x)
            return F.softmax(x).data.cpu().numpy()

        preds = np.zeros((N, 1000))

        for i, batch in enumerate(dataloader, 0):
            batch = batch.type(dtype)
            batchv = Variable(batch)
            batch_size_i = batch.size()[0]

            preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)

        # Now compute the mean kl-div
        split_scores = []

        for k in range(splits):
            part = preds[k * (N // splits): (k+1) * (N // splits), :]
            py = np.mean(part, axis=0)
            scores = []
            for i in range(part.shape[0]):
                pyx = part[i, :]
                scores.append(entropy(pyx, py))
            split_scores.append(np.exp(np.mean(scores)))

        return np.mean(split_scores), np.std(split_scores)
    
    def get_image_preprocessor(self):
        transform = transforms.Compose([
                                    transforms.Resize(32),  
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 标准化
                                ])
        return transform

    def evaluate_batch(self, sample, ref_sample=None):
        return self.inception_score(sample, device=self.device, batch_size=self.batch_size, resize=self.resize, splits=self.splits)
