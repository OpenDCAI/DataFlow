# modified from https://github.com/abdulfatir/gan-metrics-pytorch

import torchvision.transforms as TF
from dataflow.core.scorer import GenImageScorer
import json
import torch
from dataflow.utils.registry import MODEL_REGISTRY
import numpy as np
from sklearn.metrics.pairwise import polynomial_kernel
from scipy import linalg
from PIL import Image
from tqdm import tqdm
from torch.nn.functional import adaptive_avg_pool2d
import torchvision.transforms as transforms
from dataflow.Eval.image.kid.inception import InceptionV3
from dataflow.Eval.image..kid.lenet import LeNet5
from PIL import Image
import sys

@MODEL_REGISTRY.register()
class KIDScorer(GenImageScorer):
    def __init__(self, args_dict: dict, device = "cpu"):
        super().__init__(args_dict)
        self.batch_size = args_dict.get('batch_size')
        self.model = args_dict.get('model')
        self.num_workers = args_dict.get('num_workers')
        self.dims = args_dict.get('dims')
        self.device = args_dict.get('device')
        self.image_preprocessor = self.get_image_preprocessor()
        self.data_type = "image"
        self.scorer_name = "KIDScorer"

    def get_activations(self, sample, model, batch_size=50, dims=2048, cuda=False, verbose=False):
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
        -- cuda        : If set to True, use GPU
        -- verbose     : If set to True and parameter out_step is given, the number
                        of calculated batches is reported.
        Returns:
        -- A numpy array of dimension (num images, dims) that contains the
        activations of the given tensor when feeding inception with the
        query tensor.
        """
        model.eval()
        is_numpy = True if type(sample[0]) == np.ndarray else False
        if len(sample) % batch_size != 0:
            print(('Warning: number of images is not a multiple of the '
                'batch size. Some samples are going to be ignored.'))
        if batch_size > len(sample):
            print(('Warning: batch size is bigger than the data size. '
                'Setting batch size to data size'))
            batch_size = len(sample)

        n_batches = len(sample) // batch_size
        n_used_imgs = n_batches * batch_size

        pred_arr = np.empty((n_used_imgs, dims))

        for i in tqdm(range(n_batches)):
            if verbose:
                print('\rPropagating batch %d/%d' % (i + 1, n_batches), end='', flush=True)
            start = i * batch_size
            end = start + batch_size
            images = [np.array(f) for f in sample[start:end]]
            images = np.stack(images).astype(np.float32) / 255.
                # Reshape to (n_images, 3, height, width)
            images = images.transpose((0, 3, 1, 2))
            batch = torch.from_numpy(images).type(torch.FloatTensor)
            if cuda:
                batch = batch.cuda()

            pred = model(batch)[0]

            # If model output is not scalar, apply global spatial average pooling.
            # This happens if you choose a dimensionality not equal 2048.
            if pred.shape[2] != 1 or pred.shape[3] != 1:
                pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

            pred_arr[start:end] = pred.cpu().data.numpy().reshape(batch_size, -1)

        if verbose:
            print('done', np.min(images))

        return pred_arr


    def extract_lenet_features(self, imgs, net):
        net.eval()
        feats = []
        imgs = imgs.reshape([-1, 100] + list(imgs.shape[1:]))
        if imgs[0].min() < -0.001:
            imgs = (imgs + 1)/2.0
        print(imgs.shape, imgs.min(), imgs.max())
        imgs = torch.from_numpy(imgs)
        for i, images in enumerate(imgs):
            feats.append(net.extract_features(images).detach().cpu().numpy())
        feats = np.vstack(feats)
        return feats


    def _compute_activations(self, sample, model, batch_size, dims, cuda, model_type):
        if model_type == 'inception':
            act = self.get_activations(sample, model, batch_size, dims, cuda)
        elif model_type == 'lenet':
            act = self.extract_lenet_features(sample, model)
        return act


    def calculate_kid_given_paths(self, sample, ref_sample, batch_size, cuda, dims, model_type='inception'):
        """Calculates the KID of two paths"""
        if model_type == 'inception':
            block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
            model = InceptionV3([block_idx])
        elif model_type == 'lenet':
            model = LeNet5()
            model.load_state_dict(torch.load('./kid/lenet.pth'))
        if cuda:
            model.cuda()

        act_true = self._compute_activations(ref_sample, model, batch_size, dims, cuda, model_type)
        results = []
        actj = self._compute_activations(sample, model, batch_size, dims, cuda, model_type)
        kid_values = self.polynomial_mmd_averages(act_true, actj, n_subsets=100)
        results.append((kid_values[0].mean(), kid_values[0].std()))
        return kid_values[0].mean(), kid_values[0].std()

    def _sqn(self, arr):
        flat = np.ravel(arr)
        return flat.dot(flat)


    def polynomial_mmd_averages(self, codes_g, codes_r, n_subsets=50, subset_size=1000, ret_var=True, output=sys.stdout, **kernel_args):
        m = min(codes_g.shape[0], codes_r.shape[0])
        mmds = np.zeros(n_subsets)
        if ret_var:
            vars = np.zeros(n_subsets)
        choice = np.random.choice

        with tqdm(range(n_subsets), desc='MMD', file=output) as bar:
            for i in bar:
                g = codes_g[choice(len(codes_g), subset_size, replace=True)]
                r = codes_r[choice(len(codes_r), subset_size, replace=True)]
                o = self.polynomial_mmd(g, r, **kernel_args, var_at_m=m, ret_var=ret_var)
                if ret_var:
                    mmds[i], vars[i] = o
                else:
                    mmds[i] = o
                bar.set_postfix({'mean': mmds[:i+1].mean()})
        return mmds, vars if ret_var else mmds


    def polynomial_mmd(self, codes_g, codes_r, degree=3, gamma=None, coef0=1, var_at_m=None, ret_var=True):
        # use  k(x, y) = (gamma <x, y> + coef0)^degree
        # default gamma is 1 / dim
        X = codes_g
        Y = codes_r

        K_XX = polynomial_kernel(X, degree=degree, gamma=gamma, coef0=coef0)
        K_YY = polynomial_kernel(Y, degree=degree, gamma=gamma, coef0=coef0)
        K_XY = polynomial_kernel(X, Y, degree=degree, gamma=gamma, coef0=coef0)

        return self._mmd2_and_variance(K_XX, K_XY, K_YY, var_at_m=var_at_m, ret_var=ret_var)

    def _mmd2_and_variance(self, K_XX, K_XY, K_YY, unit_diagonal=False, mmd_est='unbiased', block_size=1024, var_at_m=None, ret_var=True):
        # based on
        # https://github.com/dougalsutherland/opt-mmd/blob/master/two_sample/mmd.py
        # but changed to not compute the full kernel matrix at once
        m = K_XX.shape[0]
        assert K_XX.shape == (m, m)
        assert K_XY.shape == (m, m)
        assert K_YY.shape == (m, m)
        if var_at_m is None:
            var_at_m = m

        # Get the various sums of kernels that we'll use
        # Kts drop the diagonal, but we don't need to compute them explicitly
        if unit_diagonal:
            diag_X = diag_Y = 1
            sum_diag_X = sum_diag_Y = m
            sum_diag2_X = sum_diag2_Y = m
        else:
            diag_X = np.diagonal(K_XX)
            diag_Y = np.diagonal(K_YY)

            sum_diag_X = diag_X.sum()
            sum_diag_Y = diag_Y.sum()

            sum_diag2_X = self._sqn(diag_X)
            sum_diag2_Y = self._sqn(diag_Y)

        Kt_XX_sums = K_XX.sum(axis=1) - diag_X
        Kt_YY_sums = K_YY.sum(axis=1) - diag_Y
        K_XY_sums_0 = K_XY.sum(axis=0)
        K_XY_sums_1 = K_XY.sum(axis=1)

        Kt_XX_sum = Kt_XX_sums.sum()
        Kt_YY_sum = Kt_YY_sums.sum()
        K_XY_sum = K_XY_sums_0.sum()

        if mmd_est == 'biased':
            mmd2 = ((Kt_XX_sum + sum_diag_X) / (m * m)
                    + (Kt_YY_sum + sum_diag_Y) / (m * m)
                    - 2 * K_XY_sum / (m * m))
        else:
            assert mmd_est in {'unbiased', 'u-statistic'}
            mmd2 = (Kt_XX_sum + Kt_YY_sum) / (m * (m-1))
            if mmd_est == 'unbiased':
                mmd2 -= 2 * K_XY_sum / (m * m)
            else:
                mmd2 -= 2 * (K_XY_sum - np.trace(K_XY)) / (m * (m-1))

        if not ret_var:
            return mmd2

        Kt_XX_2_sum = self._sqn(K_XX) - sum_diag2_X
        Kt_YY_2_sum = self._sqn(K_YY) - sum_diag2_Y
        K_XY_2_sum = self._sqn(K_XY)

        dot_XX_XY = Kt_XX_sums.dot(K_XY_sums_1)
        dot_YY_YX = Kt_YY_sums.dot(K_XY_sums_0)

        m1 = m - 1
        m2 = m - 2
        zeta1_est = (
            1 / (m * m1 * m2) * (
                self._sqn(Kt_XX_sums) - Kt_XX_2_sum + self._sqn(Kt_YY_sums) - Kt_YY_2_sum)
            - 1 / (m * m1)**2 * (Kt_XX_sum**2 + Kt_YY_sum**2)
            + 1 / (m * m * m1) * (
                self._sqn(K_XY_sums_1) + self._sqn(K_XY_sums_0) - 2 * K_XY_2_sum)
            - 2 / m**4 * K_XY_sum**2
            - 2 / (m * m * m1) * (dot_XX_XY + dot_YY_YX)
            + 2 / (m**3 * m1) * (Kt_XX_sum + Kt_YY_sum) * K_XY_sum
        )
        zeta2_est = (
            1 / (m * m1) * (Kt_XX_2_sum + Kt_YY_2_sum)
            - 1 / (m * m1)**2 * (Kt_XX_sum**2 + Kt_YY_sum**2)
            + 2 / (m * m) * K_XY_2_sum
            - 2 / m**4 * K_XY_sum**2
            - 4 / (m * m * m1) * (dot_XX_XY + dot_YY_YX)
            + 4 / (m**3 * m1) * (Kt_XX_sum + Kt_YY_sum) * K_XY_sum
        )
        var_est = (4 * (var_at_m - 2) / (var_at_m * (var_at_m - 1)) * zeta1_est
                + 2 / (var_at_m * (var_at_m - 1)) * zeta2_est)

        return mmd2, var_est
    
    def get_image_preprocessor(self):
        def preprocess(image):
            image = image.resize((336, 336))
            image = np.array(image)
            return image
        return preprocess
    
    def evaluate_batch(self, sample, ref_sample):
        results = self.calculate_kid_given_paths(sample, ref_sample, self.batch_size, self.device != 'cpu', self.dims, model_type=self.model)
        return results
