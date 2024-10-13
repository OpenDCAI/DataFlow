# modified from https://github.com/mseitzer/pytorch-fid

import torchvision.transforms as TF
import numpy as np
import os
import torch
from tqdm import tqdm
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d
from .fid.inception import InceptionV3
from dataflow.core.scorer import GenImageScorer
from dataflow.utils.registry import MODEL_REGISTRY

@MODEL_REGISTRY.register()
class FIDScorer(GenImageScorer):
    def __init__(self, args_dict: dict, device = "cpu"):
        super().__init__(args_dict)
        self.batch_size = args_dict.get('batch_size')
        self.model = args_dict.get('model')
        self.num_workers = args_dict.get('num_workers')
        self.dims = args_dict.get('dims')
        self.device = args_dict.get('device')
        self.image_preprocessor = self.get_image_preprocessor()
        self.data_type = "image"
        self.scorer_name = "FIDScorer"

    def get_activations(self, sample, model, batch_size=50, dims=2048, device="cpu", num_workers=1):
        """Calculates the activations of the pool_3 layer for all images.
        Params:
        -- files       : List of image files paths
        -- model       : Instance of inception model
        -- batch_size  : Batch size of images for the model to process at once.
                        Make sure that the number of samples is a multiple of
                        the batch size, otherwise some samples are ignored. This
                        behavior is retained to match the original FID score
                        implementation.
        -- dims        : Dimensionality of features returned by Inception
        -- device      : Device to run calculations
        -- num_workers : Number of parallel dataloader workers

        Returns:
        -- A numpy array of dimension (num images, dims) that contains the
        activations of the given tensor when feeding inception with the
        query tensor.
        """
        model.eval()
        if batch_size > len(sample):
            print(
                (
                    "Warning: batch size is bigger than the data size. "
                    "Setting batch size to data size"
                )
            )
            batch_size = len(sample)

        dataloader = torch.utils.data.DataLoader(sample, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers)
        pred_arr = np.empty((len(sample), dims))
        start_idx = 0
        for batch in tqdm(dataloader):
            batch = batch.to(device)
            with torch.no_grad():
                pred = model(batch)[0]
            if pred.size(2) != 1 or pred.size(3) != 1:
                pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
            pred = pred.squeeze(3).squeeze(2).cpu().numpy()
            pred_arr[start_idx : start_idx + pred.shape[0]] = pred
            start_idx = start_idx + pred.shape[0]

        return pred_arr


    def calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
        """Numpy implementation of the Frechet Distance.
        The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
        and X_2 ~ N(mu_2, C_2) is
                d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

        Stable version by Dougal J. Sutherland.

        Params:
        -- mu1   : Numpy array containing the activations of a layer of the
                inception net (like returned by the function 'get_predictions')
                for generated samples.
        -- mu2   : The sample mean over activations, precalculated on an
                representative data set.
        -- sigma1: The covariance matrix over activations for generated samples.
        -- sigma2: The covariance matrix over activations, precalculated on an
                representative data set.

        Returns:
        --   : The Frechet Distance.
        """

        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert (mu1.shape == mu2.shape), "Training and test mean vectors have different lengths"
        assert (sigma1.shape == sigma2.shape), "Training and test covariances have different dimensions"

        diff = mu1 - mu2

        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = (
                "fid calculation produces singular product; "
                "adding %s to diagonal of cov estimates"
            ) % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError("Imaginary component {}".format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


    def calculate_activation_statistics(self, sample, model, batch_size=50, dims=2048, device="cpu", num_workers=1 ):
        """Calculation of the statistics used by the FID.
        Params:
        -- files       : List of image files paths
        -- model       : Instance of inception model
        -- batch_size  : The images numpy array is split into batches with
                        batch size batch_size. A reasonable batch size
                        depends on the hardware.
        -- dims        : Dimensionality of features returned by Inception
        -- device      : Device to run calculations
        -- num_workers : Number of parallel dataloader workers

        Returns:
        -- mu    : The mean over samples of the activations of the pool_3 layer of
                the inception model.
        -- sigma : The covariance matrix of the activations of the pool_3 layer of
                the inception model.
        """
        act = self.get_activations(sample, model, batch_size, dims, device, num_workers)
        mu = np.mean(act, axis=0)
        sigma = np.cov(act, rowvar=False)
        return mu, sigma

    def compute_statistics_of_path(self, sample, model, batch_size, dims, device, num_workers=1):
        m, s = self.calculate_activation_statistics(sample, model, batch_size, dims, device, num_workers)
        return m, s


    def calculate_fid_given_paths(self, sample, ref_sample, batch_size, device, dims, num_workers=1):
        """Calculates the FID of two paths"""
        print(dims)
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        model = InceptionV3([block_idx]).to(device)
        m1, s1 = self.compute_statistics_of_path(sample, model, batch_size, dims, device, num_workers)
        m2, s2 = self.compute_statistics_of_path(ref_sample, model, batch_size, dims, device, num_workers)
        fid_value = self.calculate_frechet_distance(m1, s1, m2, s2)

        return fid_value
    
    def get_image_preprocessor(self):
        preprocess = TF.Compose([
            TF.Resize((336, 336)),  
            TF.ToTensor()           
        ])
        return preprocess
    
    def evaluate_batch(self, sample, ref_sample=None):
        if self.device is None:
            device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
        else:
            device = torch.device(self.device)

        if self.num_workers is None:
            try:
                num_cpus = len(os.sched_getaffinity(0))
            except AttributeError:
                num_cpus = os.cpu_count()
            num_workers = min(num_cpus, 8) if num_cpus is not None else 0
        else:
            num_workers = self.num_workers

        fid_value = self.calculate_fid_given_paths(sample, ref_sample, self.batch_size, device, self.dims, num_workers)

        return fid_value
