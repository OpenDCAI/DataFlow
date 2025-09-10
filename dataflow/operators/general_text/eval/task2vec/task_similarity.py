#!/usr/bin/env python3

# Copyright 2017-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.

import itertools
from typing import Tuple

import scipy.spatial.distance as distance
import numpy as np
import copy
import pickle

# import uutils

_DISTANCES = {}


# TODO: Remove methods that do not perform well

def _register_distance(distance_fn):
    _DISTANCES[distance_fn.__name__] = distance_fn
    return distance_fn


def is_excluded(k):
    exclude = ['fc', 'linear']
    return any([e in k for e in exclude])


def load_embedding(filename):
    with open(filename, 'rb') as f:
        e = pickle.load(f)
    return e


def get_trivial_embedding_from(e):
    trivial_embedding = copy.deepcopy(e)
    for l in trivial_embedding['layers']:
        a = np.array(l['filter_logvar'])
        a[:] = l['filter_lambda2']
        l['filter_logvar'] = list(a)
    return trivial_embedding


def binary_entropy(p):
    from scipy.special import xlogy
    return - (xlogy(p, p) + xlogy(1. - p, 1. - p))


def get_layerwise_variance(e, normalized=False):
    var = [np.exp(l['filter_logvar']) for l in e['layers']]
    if normalized:
        var = [v / np.linalg.norm(v) for v in var]
    return var


def get_variance(e, normalized=False):
    var = 1. / np.array(e.hessian)
    if normalized:
        lambda2 = 1. / np.array(e.scale)
        var = var / lambda2
    return var


def get_variances(*embeddings, normalized=False):
    return [get_variance(e, normalized=normalized) for e in embeddings]


def get_hessian(e, normalized=False):
    hess = np.array(e.hessian)
    if normalized:
        scale = np.array(e.scale)
        hess = hess / scale
    return hess


def get_hessians(*embeddings, normalized=False):
    return [get_hessian(e, normalized=normalized) for e in embeddings]


def get_scaled_hessian(e0, e1):
    h0, h1 = get_hessians(e0, e1, normalized=False)
    return h0 / (h0 + h1 + 1e-8), h1 / (h0 + h1 + 1e-8)


def get_full_kl(e0, e1):
    var0, var1 = get_variance(e0), get_variance(e1)
    kl0 = .5 * (var0 / var1 - 1 + np.log(var1) - np.log(var0))
    kl1 = .5 * (var1 / var0 - 1 + np.log(var0) - np.log(var1))
    return kl0, kl1


def layerwise_kl(e0, e1):
    layers0, layers1 = get_layerwise_variance(e0), get_layerwise_variance(e1)
    kl0 = []
    for var0, var1 in zip(layers0, layers1):
        kl0.append(np.sum(.5 * (var0 / var1 - 1 + np.log(var1) - np.log(var0))))
    return kl0


def layerwise_cosine(e0, e1):
    layers0, layers1 = get_layerwise_variance(e0, normalized=True), get_layerwise_variance(e1, normalized=True)
    res = []
    for var0, var1 in zip(layers0, layers1):
        res.append(distance.cosine(var0, var1))
    return res


@_register_distance
def kl(e0, e1):
    var0, var1 = get_variance(e0), get_variance(e1)
    kl0 = .5 * (var0 / var1 - 1 + np.log(var1) - np.log(var0))
    kl1 = .5 * (var1 / var0 - 1 + np.log(var0) - np.log(var1))
    return np.maximum(kl0, kl1).sum()


@_register_distance
def asymmetric_kl(e0, e1):
    var0, var1 = get_variance(e0), get_variance(e1)
    kl0 = .5 * (var0 / var1 - 1 + np.log(var1) - np.log(var0))
    kl1 = .5 * (var1 / var0 - 1 + np.log(var0) - np.log(var1))
    return kl0.sum()


@_register_distance
def jsd(e0, e1):
    var0, var1 = get_variance(e0), get_variance(e1)
    var = .5 * (var0 + var1)
    kl0 = .5 * (var0 / var - 1 + np.log(var) - np.log(var0))
    kl1 = .5 * (var1 / var - 1 + np.log(var) - np.log(var1))
    return (.5 * (kl0 + kl1)).mean()


@_register_distance
def cosine(e0, e1):
    h1, h2 = get_scaled_hessian(e0, e1)
    return distance.cosine(h1, h2)


@_register_distance
def normalized_cosine(e0, e1):
    h1, h2 = get_variances(e0, e1, normalized=True)
    return distance.cosine(h1, h2)


@_register_distance
def correlation(e0, e1):
    v1, v2 = get_variances(e0, e1, normalized=False)
    return distance.correlation(v1, v2)


@_register_distance
def entropy(e0, e1):
    h1, h2 = get_scaled_hessian(e0, e1)
    return np.log(2) - binary_entropy(h1).mean()


def get_normalized_embeddings(embeddings, normalization=None):
    F = [1. / get_variance(e, normalized=False) if e is not None else None for e in embeddings]
    zero_embedding = np.zeros_like([x for x in F if x is not None][0])
    F = np.array([x if x is not None else zero_embedding for x in F])
    # FIXME: compute variance using only valid embeddings
    if normalization is None:
        normalization = np.sqrt((F ** 2).mean(axis=0, keepdims=True))
    F /= normalization
    return F, normalization


def pdist(embeddings, distance='cosine') -> np.ndarray:
    distance_fn = _DISTANCES[distance]
    n = len(embeddings)
    distance_matrix = np.zeros([n, n])
    if distance != 'asymmetric_kl':
        for (i, e1), (j, e2) in itertools.combinations(enumerate(embeddings), 2):
            distance_matrix[i, j] = distance_fn(e1, e2)
            distance_matrix[j, i] = distance_matrix[i, j]
    else:
        for (i, e1) in enumerate(embeddings):
            for (j, e2) in enumerate(embeddings):
                distance_matrix[i, j] = distance_fn(e1, e2)
    return distance_matrix


def cross_pdist(embeddings1, embeddings2, distance='cosine') -> np.ndarray :
    """
    Compute pairwise distance between embeddings1 and embeddings2.

    ref: https://chat.openai.com/share/a5ca38dc-3393-4cfd-971c-4a29b0c56b63 
    """
    distance_fn = _DISTANCES[distance]
    n1 = len(embeddings1)
    n2 = len(embeddings2)
    distance_matrix = np.zeros([n1, n2])
    if distance != 'asymmetric_kl':
        for i, e1 in enumerate(embeddings1):
            for j, e2 in enumerate(embeddings2):
                distance_matrix[i, j] = distance_fn(e1, e2)
    else:
        for i, e1 in enumerate(embeddings1):
            for j, e2 in enumerate(embeddings2):
                distance_matrix[i, j] = distance_fn(e1, e2)
    return distance_matrix


def cdist(from_embeddings, to_embeddings, distance='cosine'):
    distance_fn = _DISTANCES[distance]
    distance_matrix = np.zeros([len(from_embeddings), len(to_embeddings)])
    for (i, e1) in enumerate(from_embeddings):
        for (j, e2) in enumerate(to_embeddings):
            if e1 is None or e2 is None:
                continue
            distance_matrix[i, j] = distance_fn(e1, e2)
    return distance_matrix


def plot_distance_matrix(embeddings, labels=None, distance='cosine', show_plot=True):
    import seaborn as sns
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import squareform
    import pandas as pd
    import matplotlib.pyplot as plt
    distance_matrix = pdist(embeddings, distance=distance)
    cond_distance_matrix = squareform(distance_matrix, checks=False)
    linkage_matrix = linkage(cond_distance_matrix, method='complete', optimal_ordering=True)
    if labels is not None:
        distance_matrix = pd.DataFrame(distance_matrix, index=labels, columns=labels)
    sns.clustermap(distance_matrix, row_linkage=linkage_matrix, col_linkage=linkage_matrix, cmap='viridis_r')
    if show_plot:
        plt.show()

## LLM DIV
def plot_distance_matrix_heatmap_only(embeddings, labels=None, distance='cosine', show_plot=True, title=None, save_file=None):
    import seaborn as sns
    import pandas as pd
    import matplotlib.pyplot as plt
    distance_matrix = pdist(embeddings, distance=distance)
    if labels is not None:
        distance_matrix = pd.DataFrame(distance_matrix, index=labels, columns=labels)
    sns.heatmap(distance_matrix, cmap='viridis_r')
    if title:
        plt.title(title)
    if save_file:
        _ = plt.savefig("plots/" + save_file + ".png", bbox_inches='tight')
    if show_plot:
        plt.show()

## LLM DIV
def plot_distance_matrix_from_distance_matrix(distance_matrix, labels=None, show_plot=True, title=None, save_file=None, cluster=False, plot_multi=False):
    import seaborn as sns
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import squareform
    import pandas as pd
    import matplotlib.pyplot as plt

    cond_distance_matrix = squareform(distance_matrix, checks=False)
    linkage_matrix = linkage(cond_distance_matrix, method='complete', optimal_ordering=True)
    if labels is not None:
        distance_matrix = pd.DataFrame(distance_matrix, index=labels, columns=labels)
   
    # plot multiple subplots in one figure
    # distance_matrix passed in is a list of distance_matrix (np.arrays)
    if plot_multi and not cluster:
        num_rows, num_cols = 3, 2
        f, ax = plt.subplots(num_rows, num_cols)#, figsize=(12, 15))
        i = 0
        for row_ind in range(len(num_rows)):
            for col_ind in range(len(num_cols)):
                sns.heatmap(distance_matrix[i], cmap='viridis_r', ax=ax[row_ind, col_ind])
                i += 1
    else:
        if cluster:
            sns.clustermap(distance_matrix, row_linkage=linkage_matrix, col_linkage=linkage_matrix, cmap='viridis_r')
        else:
            sns.heatmap(distance_matrix, cmap='viridis_r')
    
    if title:
        plt.title(title)
    if save_file:
        _ = plt.savefig("plots/" + save_file + ".png", bbox_inches='tight')
    if show_plot:
        plt.show()

## LLM DIV
# plot multiple subplots in one figure
# distance_matrix passed in is a list of distance_matrix np.arrays
def plot_multi_distance_matrix_from_distance_matrix_list(distance_matrix_lst, title_lst, labels, main_title=None, show_plot=True, title=None, save_file=None, vmin=None, vmax=None):
    import seaborn as sns
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import squareform
    import pandas as pd
    import matplotlib.pyplot as plt
    import math
    num_rows, num_cols = math.ceil(len(distance_matrix_lst)/2), 2
    if len(distance_matrix_lst) % 2 == 1:
        figsize = (12,10)
    else:
        figsize = (12,10)
    f, ax = plt.subplots(num_rows, num_cols, figsize=figsize)
    i = 0
    for row_ind in range(num_rows):
        for col_ind in range(num_cols):
            if i >= len(distance_matrix_lst):
                break
            distance_matrix = distance_matrix_lst[i]
            distance_matrix = pd.DataFrame(distance_matrix, index=labels[i], columns=labels[i])
            if len(distance_matrix_lst) > 2:
                ax[row_ind, col_ind].set_aspect('equal')
                if vmin is not None and vmax is not None:
                    sns.heatmap(distance_matrix, cmap='viridis_r', ax=ax[row_ind, col_ind], vmin=vmin, vmax=vmax)
                else:
                    sns.heatmap(distance_matrix, cmap='viridis_r', ax=ax[row_ind, col_ind])
                ax[row_ind, col_ind].set_title(title_lst[i])
            else:
                ax[col_ind].set_aspect('equal')
                sns.heatmap(distance_matrix, cmap='viridis_r', ax=ax[col_ind])
                ax[col_ind].set_title(title_lst[i])
            
            i += 1
    if len(distance_matrix_lst) % 2 == 1:
        f.delaxes(ax[num_rows-1,1])
    
    if main_title:
        f.suptitle(main_title)
        f.subplots_adjust(top=0.5)
    
    if len(distance_matrix_lst) % 2 == 1:
        plt.tight_layout(h_pad=2)
    else:
        plt.tight_layout(h_pad=2, w_pad=5)
    if save_file:
        _ = plt.savefig("plots/" + save_file + ".png", bbox_inches='tight')
    if show_plot:
        plt.show()

## LLM DIV       
def stats_of_distance_matrix(distance_matrix: np.ndarray,
                             remove_diagonal: bool = True,
                             variance_type: str = 'std',    # TODO: was ci_0.95. Changed to rid uutils call
                             get_total: bool = False,
                             ) -> Tuple[float, float]:
    if remove_diagonal:
        # - remove diagonal: ref https://stackoverflow.com/questions/46736258/deleting-diagonal-elements-of-a-numpy-array
        triu: np.ndarray = np.triu(distance_matrix)
        tril: np.ndarray = np.tril(distance_matrix)
        # distance_matrix = distance_matrix[~np.eye(distance_matrix.shape[0], dtype=bool)].reshape(distance_matrix.shape[0], -1)
        # remove diagonal and dummy zeros where the other triangular matrix was artificially placed.
        distance_matrix = triu[triu != 0.0]

    # - flatten
    distance_matrix: np.ndarray = distance_matrix.flatten()

    # - compute stats of distance matrix
    if variance_type == 'std':
        mu, var = distance_matrix.mean(), distance_matrix.std()
    # elif variance_type == 'ci_0.95':
    #     from uutils.torch_uu.metrics.confidence_intervals import mean_confidence_interval
    #     mu, var = mean_confidence_interval(distance_matrix, confidence=0.95)
    else:
        raise ValueError(f'Invalid variance type, got: {variance_type=}')

    # - double checks the mean was computed corrects. Since it's symmetric the mean after removing diagonal should be equal to just one side of the diagonals
    if remove_diagonal:
        # from uutils.torch_uu import approx_equal
        # assert approx_equal(triu.sum(), tril.sum(), tolerance=1e-4), f'Distance matrix is not symmetric, are you sure this is correct?'
        # assert approx_equal(distance_matrix.mean(), triu[triu != 0.0].mean(), tolerance=1e-4), f'Mean should be equal to triangular matrix'
        # assert approx_equal(mu, triu[triu != 0.0].mean(), tolerance=1e-4)

        print('Lower tri sum', tril.sum(), ' / Upper tri sum', triu.sum(), '| These should be approx equal!!')
        print('Total mean', distance_matrix.mean(), ' / Upper mean', triu[triu != 0.0].mean(), ' / Lower mean', tril[tril != 0.0].mean(), '| These should all be approx equal!!')
        print('mu (div coefficient)', mu, ' / Upper mean', triu[triu != 0.0].mean(), '| These should all be approx equal!!')
    if get_total:
        total = distance_matrix.sum()
        return mu, var, total
    else:
        return mu, var


def stats_cross_distance_matrix(distance_matrix: np.ndarray,
                                remove_diagonal: bool = False,
                                variance_type: str = 'std',     # TODO: was ci_0.95. Changed to rid uutils call
                                get_total: bool = False,
                                ) -> Tuple[float, float]:
    return stats_of_distance_matrix(distance_matrix, remove_diagonal=remove_diagonal, variance_type=variance_type, get_total=get_total)


def plot_histogram_of_distances(distance_matrix: np.ndarray, title, show_plot=True, save_file=None, bins_width=None, grid=True):
    import matplotlib.pyplot as plt
    triu = np.triu(distance_matrix)
    triu = triu[triu != 0.0]
    distance_values = triu.flatten()
    
    if grid:
        plt.grid(zorder=0)
    plt.axvline(np.mean(distance_values), color='k', linestyle='dashed', linewidth=1, zorder=4)
    if bins_width is not None:
        plt.hist(distance_values, edgecolor ="black", bins=np.arange(min(distance_values), max(distance_values) + bins_width, bins_width), zorder=3)
    else:
        plt.hist(distance_values, edgecolor ="black", zorder=3)
    plt.title(title)
    plt.xlabel("Cosine Distance between Task Pairs")
    plt.ylabel("Frequency")

    plt.tight_layout()
    if save_file:
        _ = plt.savefig("plots/" + save_file + ".png", bbox_inches='tight')

    if show_plot:
        plt.show()


## LLM DIV 
# plot multiple subplots in one figure
# distance_matrix passed in is a list of distance_matrix (np.arrays)
def plot_multi_histogram_of_distances(distance_matrix_lst, title_lst, main_title=None, show_plot=True, save_file=None, 
                                      xlabel="Cosine Distance between Task Pairs", grid=True, bins_width=None, 
                                      num_cols=2, figsize=(12,10)):
    import seaborn as sns
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import squareform
    import pandas as pd
    import matplotlib.pyplot as plt
    import math
    
    if num_cols == 2:
        num_rows = math.ceil(len(distance_matrix_lst)/2)
    else:
        num_rows = math.ceil(len(distance_matrix_lst)/num_cols)
    
    f, ax = plt.subplots(num_rows, num_cols, figsize=figsize)
    i = 0
    for row_ind in range(num_rows):
        for col_ind in range(num_cols):
            if i >= len(distance_matrix_lst):
                break
            triu = np.triu(distance_matrix_lst[i])
            triu = triu[triu != 0.0]
            distance_values = triu.flatten()
            
            if len(distance_matrix_lst) > 2:
                if grid:
                    ax[row_ind, col_ind].grid(zorder=0)
                if bins_width is not None:
                    ax[row_ind, col_ind].hist(distance_values, edgecolor ="black", zorder=3, bins=np.arange(min(distance_values), max(distance_values) + bins_width, bins_width))
                else:
                    ax[row_ind, col_ind].hist(distance_values, edgecolor ="black", zorder=3)
                ax[row_ind, col_ind].set_xlabel(xlabel)
                ax[row_ind, col_ind].set_ylabel("Frequency")
                ax[row_ind, col_ind].axvline(np.mean(distance_values), color='k', linestyle='dashed', linewidth=1, zorder=4)
                ax[row_ind, col_ind].set_title(title_lst[i])
            else:
                if grid:
                    ax[col_ind].grid(zorder=0)
                ax[col_ind].hist(distance_values, edgecolor ="black", zorder=3)
                if bins_width is not None:
                    ax[col_ind].hist(distance_values, edgecolor ="black", zorder=3, bins=np.arange(min(distance_values), max(distance_values) + bins_width, bins_width))
                else:
                    ax[col_ind].hist(distance_values, edgecolor ="black", zorder=3)
                ax[col_ind].set_xlabel(xlabel)
                ax[col_ind].set_ylabel("Frequency")
                ax[col_ind].set_title(title_lst[i])
            i += 1
    if len(distance_matrix_lst) % 2 == 1 and num_cols == 2:
        f.delaxes(ax[num_rows-1,1])
        
    if main_title:
        f.suptitle(main_title)
        f.subplots_adjust(top=1)

    plt.grid(True)
    plt.tight_layout()
    if save_file:
        _ = plt.savefig("plots/" + save_file + ".png", bbox_inches='tight')
    if show_plot:
        plt.show()