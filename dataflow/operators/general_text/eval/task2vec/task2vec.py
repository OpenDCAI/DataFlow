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
import math
import random
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
import numpy as np
from tqdm.auto import tqdm, trange
import logging
from torch.utils.data import DataLoader, Dataset

from .utils import AverageMeter, get_error, get_device

## LLM DIV
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

## LLM DIV
def get_loss(logits: torch.tensor, targets: torch.tensor, ignore_index=None) -> torch.tensor:
    """
    Computes the cross-entropy loss for either sequence classification or generation.
    """
    assert logits.dim() == 3 and ignore_index is not None
    loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
    logits = logits[:,:-1,:]
    logits = logits.transpose(1, 2) # batch_size, vocab_size (i.e. num_classes), sequence_length
    targets = targets[:,1:]
       
    return loss(logits, targets)

class Embedding:
    """
    task_embedding = diagonal of the FIM for the filters of size [F_total, 1] total filters for a network.

    Notes:
        - the diagonal of the Fisher Information Matrix for each layer.
        - embedding size should be the size of the total number of filters for the network.
    """

    def __init__(self, hessian, scale, meta=None):
        self.hessian = np.array(hessian)
        self.scale = np.array(scale)
        self.meta = meta

    def __repr__(self):
        return f'{self.hessian}'


class ProbeNetwork(ABC, nn.Module):
    """Abstract class that all probe networks should inherit from.

    This is a standard torch.nn.Module but needs to expose a classifier property that returns the final classicifation
    module (e.g., the last fully connected layer).
    """

    @property
    @abstractmethod
    def classifier(self):
        raise NotImplementedError("Override the classifier property to return the submodules of the network that"
                                  " should be interpreted as the classifier")

    @classifier.setter
    @abstractmethod
    def classifier(self, val):
        raise NotImplementedError("Override the classifier setter to set the submodules of the network that"
                                  " should be interpreted as the classifier")


class Task2Vec:

    def __init__(self, model: ProbeNetwork, skip_layers=0, max_samples=None, classifier_opts=None,
                 method='montecarlo', method_opts=None, loader_opts=None, bernoulli=False, mode='autoregressive'): ## LLM DIV
        if classifier_opts is None:
            classifier_opts = {}
        if method_opts is None:
            method_opts = {}
        if loader_opts is None:
            loader_opts = {}
        assert method in ('variational', 'montecarlo')
        assert skip_layers >= 0

        self.model = model
        # Fix batch norm running statistics (i.e., put batch_norm layers in eval mode)
        self.model.train()
        self.device = get_device(self.model)
        self.skip_layers = skip_layers
        self.max_samples = max_samples
        self.classifier_opts = classifier_opts
        self.method = method
        self.method_opts = method_opts
        self.loader_opts = loader_opts
        self.bernoulli = bernoulli
        self.mode = mode
        if self.mode == "autoregressive":
            self.loss_fn = get_loss
        else:
            self.loss_fn = nn.CrossEntropyLoss() if not self.bernoulli else nn.BCEWithLogitsLoss()
            self.loss_fn = self.loss_fn.to(self.device)

    def embed(self, dataset: Dataset, epochs: int = 5):
        ## LLM DIV
        # Cache the last layer features (needed to train the classifier) and (if needed) the intermediate layer features
        # so that we can skip the initial layers when computing the embedding
        # dataset.train()
        if self.mode == "autoregressive":
            loss = None
            print(f'{self.classifier_opts=}')
            if self.classifier_opts:  # is it something truthy? e.g., dict with something in it?
                if self.classifier_opts.get('finetune', False):  # finetune only if specified True, else no finetuning if not specified or False. 
                    epochs = 0
                    print(f'Warning: classifier_opts doesnt specify finetune or break early, thus no finetuning is being done. See: {self.classifier_opts=} {epochs=}')
                    loss = self._finetune_classifier(dataset, loader_opts=self.loader_opts, classifier_opts=self.classifier_opts, max_samples=self.max_samples, epochs=epochs)
                else:
                    loss = self._finetune_classifier(dataset, loader_opts=self.loader_opts, classifier_opts=self.classifier_opts, max_samples=self.max_samples, epochs=epochs)
            else:  # self.classifier_opts might be None or {}
                loss = self._finetune_classifier(dataset, loader_opts=self.loader_opts, classifier_opts=self.classifier_opts, max_samples=self.max_samples, epochs=epochs)
            print(f'{loss=} (after fine tune, if not done it will be None)')
            assert loss is not None, f'Err: {loss=}'
            self.compute_fisher(dataset)
            embedding = self.extract_embedding(self.model)
            return embedding, loss
        else:
            if self.skip_layers > 0:
                self._cache_features(dataset, indexes=(self.skip_layers, -1), loader_opts=self.loader_opts,
                                     max_samples=self.max_samples)
            else:
                self._cache_features(dataset, max_samples=self.max_samples)
            # Fits the last layer classifier using cached features
            self._fit_classifier(**self.classifier_opts)

            if self.skip_layers > 0:
                dataset = torch.utils.data.TensorDataset(self.model.layers[self.skip_layers].input_features,
                                                         self.model.layers[-1].targets)

            # dataset.eval()  # I added this so that the embedding is computed on the val set
            self.compute_fisher(dataset)
            embedding = self.extract_embedding(self.model)
            # dataset.train()  # returns to using the support set
            return embedding
        
    ### LLM DIV 
    def _finetune_classifier(self, dataset: Dataset, loader_opts: dict = None, classifier_opts: dict = None, max_samples=None, epochs = 5, learning_rate = 5e-5, adam_epsilon = 1e-8):
        """Fits the last layer of the HuggingFace transformer probe network."""
        logging.info("Finetune classifier...")
        if loader_opts is None:
            loader_opts = {}
        if classifier_opts is None:
            classifier_opts = {}
        data_loader = DataLoader(dataset, shuffle=False, batch_size=loader_opts.get('batch_size', 8),
                                 num_workers=loader_opts.get('num_workers', 0), drop_last=False)

        device = next(self.model.parameters()).device
        print("MODEL DEVICE: ", device)
        
        # num_examples = int(classifier_opts.get("task_batch_size", 256) / loader_opts.get('batch_size', 8))
        num_examples = len(list(data_loader))  # not ideal but it's quicker in dev time, usually we won't feed the entire data set to task2vec so this should be fine
        n_batches = num_examples
        
        optimizer_grouped_parameters = [
            {'params': [p for p in self.model.lm_head.parameters()],
             'weight_decay': classifier_opts.get("weight_decay",0.0001)},
        ]
        
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=classifier_opts.get("learning_rate",learning_rate), eps=classifier_opts.get("adam_epsilon",adam_epsilon))
        
        # Train!
        logging.info("***** Running training *****")
        # logging.info("  Num examples = %d", num_examples)
        logging.info("  Num Epochs = %d", epochs)
        logging.info("  Batch size = %d", loader_opts.get('batch_size', 8))
        
        train_iterator = trange(classifier_opts.get("epochs", epochs), desc="Epoch", leave=False)
        set_seed(classifier_opts.get("seed", 42))  # Added here for reproductibility (even between python 2 and 3)
        
        self.model.train()
        for epoch in train_iterator:
            metrics = AverageMeter()
            epoch_iterator = tqdm(data_loader, desc="Iteration", total=n_batches, leave=False)
            for step, batch in enumerate(epoch_iterator):
                optimizer.zero_grad()
                inputs = {'input_ids': batch['input_ids'].to(device),
                        'attention_mask': batch['attention_mask'].to(device)}
                logits = self.model(**inputs, labels=inputs["input_ids"]).logits
                loss = self.loss_fn(logits, inputs["input_ids"], ignore_index=50256)
                print(f'\nInitial loss {loss.item()} ({step=} {epoch=})') if step == 0 else None
                error = get_error(logits, inputs['input_ids'], ignore_index=50256)
                loss.backward()
                optimizer.step()
                
                metrics.update(n=batch['input_ids'].shape[0], loss=loss.item(), error=error)
                epoch_iterator.update(1)
                
                if classifier_opts.get("break_early", False):
                    print("----> breaking early")
                    break
            if classifier_opts.get("break_early", False):
                break
            logging.info(f"[epoch {epoch}]: " + "\t".join(f"{k}: {v}" for k, v in metrics.avg.items()))
        print(f'\nfinal loss {step=} {epoch=} of final layer loss {loss.item()} (note we are not recomputing loss after a step so this loss printed is larger than it should be/one off)')
        return loss.item()

    ### LLM DIV
    def montecarlo_fisher_autoregressive(self, dataset: Dataset, epochs: int = 1):
        logging.info("Using montecarlo Fisher")
        if self.loader_opts is None:
            loader_opts = {}
        else:
            loader_opts = self.loader_opts
            
        data_loader = DataLoader(dataset, shuffle=False, batch_size=loader_opts.get('batch_size', 8),
                                 num_workers=loader_opts.get('num_workers', 0), drop_last=False)
        device = get_device(self.model)

        # num_examples = int(classifier_opts.get("task_batch_size", 256) / loader_opts.get('batch_size', 8))
        num_examples = len(list(data_loader))  # not idea but it's quicker in dev time, usually we won't feed the entire data set to task2vec so this should be fine
        n_batches = num_examples

        logging.info("Computing Fisher...")
        for p in self.model.parameters():
            p.grad2_acc = torch.zeros_like(p.data)
            p.grad_counter = 0
            
        for k in range(epochs):
            logging.info(f"\tepoch {k + 1}/{epochs}")
            
            epoch_iterator = tqdm(data_loader, desc="Iteration", total=n_batches, leave=False)
            for step, batch in enumerate(epoch_iterator):
                inputs = {'input_ids': batch['input_ids'].to(device),
                        'attention_mask': batch['attention_mask'].to(device)}
                logits = self.model(**inputs, labels=inputs["input_ids"]).logits
                
                # The gradients used to compute the FIM needs to be for y sampled from
                # the model distribution y ~ p_w(y|x), not for y from the dataset
                if self.bernoulli:
                    target = torch.bernoulli(F.sigmoid(logits[:,:-1,:])).detach()
                else:
                    softmax_output = F.softmax(logits, dim=-1)
                    lst = [torch.multinomial(softmax_output[i,:,:], 1).detach().view(-1) for i in range(len(softmax_output))]
                    target = torch.stack(lst, dim=0)          
                
                loss = self.loss_fn(logits, target, ignore_index=50256)
                self.model.zero_grad()
                loss.backward()
                for p in self.model.parameters():
                    if p.grad is not None:
                        p.grad2_acc += p.grad.data ** 2
                        p.grad_counter += 1
                if self.classifier_opts.get("break_early", False):
                    break  # for debugging faster, otherwise FIM is really slow
            if self.classifier_opts.get("break_early", False):
                break  # for debugging faster, otherwise FIM is really slow
        for p in self.model.parameters():
            if p.grad_counter == 0:
                del p.grad2_acc
            else:
                p.grad2_acc /= p.grad_counter
        logging.info("done")
        
    def montecarlo_fisher(self, dataset: Dataset, epochs: int = 1):
        logging.info("Using montecarlo Fisher")
        if self.skip_layers > 0:
            dataset = torch.utils.data.TensorDataset(self.model.layers[self.skip_layers].input_features,
                                                     self.model.layers[-1].targets)
        data_loader = _get_loader(dataset, **self.loader_opts)
        device = get_device(self.model)
        logging.info("Computing Fisher...")

        for p in self.model.parameters():
            p.grad2_acc = torch.zeros_like(p.data)
            p.grad_counter = 0
        for k in range(epochs):
            logging.info(f"\tepoch {k + 1}/{epochs}")
            for i, (data, target) in enumerate(tqdm(data_loader, leave=False, desc="Computing Fisher")):
                data = data.to(device)
                output = self.model(data, start_from=self.skip_layers)
                # The gradients used to compute the FIM needs to be for y sampled from
                # the model distribution y ~ p_w(y|x), not for y from the dataset
                if self.bernoulli:
                    target = torch.bernoulli(F.sigmoid(output)).detach()
                else:
                    target = torch.multinomial(F.softmax(output, dim=-1), 1).detach().view(-1)
                loss = self.loss_fn(output, target)
                self.model.zero_grad()
                loss.backward()
                for p in self.model.parameters():
                    if p.grad is not None:
                        p.grad2_acc += p.grad.data ** 2
                        p.grad_counter += 1
        for p in self.model.parameters():
            if p.grad_counter == 0:
                del p.grad2_acc
            else:
                p.grad2_acc /= p.grad_counter
        logging.info("done")

    def _run_epoch(self, data_loader: DataLoader, model: ProbeNetwork, loss_fn,
                   optimizer: Optimizer, epoch: int, train: bool = True,
                   add_compression_loss: bool = False, skip_layers=0, beta=1.0e-7):
        metrics = AverageMeter()
        device = get_device(model)

        for i, (input, target) in enumerate(tqdm(data_loader, leave=False, desc="Computing Fisher")):
            input = input.to(device)
            target = target.to(device)
            output = model(input, start_from=skip_layers)

            loss = loss_fn(output, target)
            lz = beta * variational.get_compression_loss(model) if add_compression_loss else torch.zeros_like(loss)
            loss += lz

            error = get_error(output, target)

            metrics.update(n=input.size(0), loss=loss.item(), lz=lz.item(), error=error)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        # logging.info(
        print(
            "{}: [{epoch}] ".format('Epoch' if train else '', epoch=epoch) +
            "Data/Batch: {:.3f}/{:.3f} ".format(metrics.avg["data_time"], metrics.avg["batch_time"]) +
            "Loss {:.3f} Lz: {:.3f} ".format(metrics.avg["loss"], metrics.avg["lz"]) +
            "Error: {:.2f}".format(metrics.avg["error"])
        )
        return metrics.avg

    def variational_fisher(self, dataset: Dataset, epochs=1, beta=1e-7):
        logging.info("Training variational fisher...")
        parameters = []
        for layer in self.model.layers[self.skip_layers:-1]:
            if isinstance(layer, nn.Module):  # Skip lambda functions
                variational.make_variational(layer)
                parameters += variational.get_variational_vars(layer)
        bn_params = []
        # Allows batchnorm parameters to change
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                bn_params += list(m.parameters())
        # Avoids computing the gradients wrt to the weights to save time and memory
        for p in self.model.parameters():
            if p not in set(parameters) and p not in set(self.model.classifier.parameters()):
                p.old_requires_grad = p.requires_grad
                p.requires_grad = False

        optimizer = torch.optim.Adam([
            {'params': parameters},
            {'params': bn_params, 'lr': 5e-4},
            {'params': self.model.classifier.parameters(), 'lr': 5e-4}],
            lr=1e-2, betas=(.9, 0.999))
        if self.skip_layers > 0:
            dataset = torch.utils.data.TensorDataset(self.model.layers[self.skip_layers].input_features,
                                                     self.model.layers[-1].targets)
        train_loader = _get_loader(dataset, **self.loader_opts)

        for epoch in range(epochs):
            self._run_epoch(train_loader, self.model, self.loss_fn, optimizer, epoch, beta=beta,
                            add_compression_loss=True, train=True)

        # Resets original value of requires_grad
        for p in self.model.parameters():
            if hasattr(p, 'old_requires_grad'):
                p.requires_grad = p.old_requires_grad
                del p.old_requires_grad

    def compute_fisher(self, dataset: Dataset):
        """
        Computes the Fisher Information of the weights of the model wrt the model output on the dataset and stores it.

        The Fisher Information Matrix is defined as:
            F = E_{x ~ dataset} E_{y ~ p_w(y|x)} [\nabla_w log p_w(y|x) \nabla_w log p_w(y|x)^t]
        where p_w(y|x) is the output probability vector of the network and w are the weights of the network.
        Notice that the label y is sampled from the model output distribution and not from the dataset.

        This code only approximate the diagonal of F. The result is stored in the model layers and can be extracted
        using the `get_fisher` method. Different approximation methods of the Fisher information matrix are available,
        and can be selected in the __init__.

        :param dataset: dataset with the task to compute the Fisher on
        """
        if self.mode == 'autoregressive' and self.method == 'montecarlo':
            fisher_fn = self.montecarlo_fisher_autoregressive
        elif self.method == 'variational':
            fisher_fn = self.variational_fisher
        elif self.method == 'montecarlo':
            fisher_fn = self.montecarlo_fisher
        else:
            raise ValueError(f"Invalid Fisher method {self.method}")
        fisher_fn(dataset, **self.method_opts)

    def _cache_features(self, dataset: Dataset, indexes=(-1,), max_samples=None, loader_opts: dict = None):
        logging.info("Caching features...")
        if loader_opts is None:
            loader_opts = {}
        data_loader = DataLoader(dataset, shuffle=False, batch_size=loader_opts.get('batch_size', 64),
                                 num_workers=loader_opts.get('num_workers', 0), drop_last=False)

        device = next(self.model.parameters()).device

        def _hook(layer, inputs):
            if not hasattr(layer, 'input_features'):
                layer.input_features = []
            layer.input_features.append(inputs[0].data.cpu().clone())

        hooks = [self.model.layers[index].register_forward_pre_hook(_hook)
                 for index in indexes]
        if max_samples is not None:
            n_batches = min(
                math.floor(max_samples / data_loader.batch_size) - 1, len(data_loader))
        else:
            n_batches = len(data_loader)
        targets = []

        for i, (input, target) in tqdm(enumerate(itertools.islice(data_loader, 0, n_batches)), total=n_batches,
                                       leave=False,
                                       desc="Caching features"):
            targets.append(target.clone())
            self.model(input.to(device))
        for hook in hooks:
            hook.remove()
        for index in indexes:
            self.model.layers[index].input_features = torch.cat(self.model.layers[index].input_features)
        self.model.layers[-1].targets = torch.cat(targets)

    def _fit_classifier(self, optimizer='adam', learning_rate=0.0004, weight_decay=0.0001,
                        epochs=10):
        """Fits the last layer of the network using the cached features."""
        logging.info("Fitting final classifier...")
        if not hasattr(self.model.classifier, 'input_features'):
            raise ValueError("You need to run `cache_features` on model before running `fit_classifier`")
        targets = self.model.classifier.targets.to(self.device)
        features = self.model.classifier.input_features.to(self.device)

        dataset = torch.utils.data.TensorDataset(features, targets)
        data_loader = _get_loader(dataset, **self.loader_opts)

        if optimizer == 'adam':
            optimizer = torch.optim.Adam(self.model.fc.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer == 'sgd':
            optimizer = torch.optim.SGD(self.model.fc.parameters(), lr=learning_rate, weight_decay=weight_decay)
        else:
            raise ValueError(f'Unsupported optimizer {optimizer}')

        loss_fn = nn.CrossEntropyLoss()
        for epoch in tqdm(range(epochs), desc="Fitting classifier", leave=False):
            metrics = AverageMeter()
            for data, target in data_loader:
                optimizer.zero_grad()
                output = self.model.classifier(data)
                loss = loss_fn(self.model.classifier(data), target)
                error = get_error(output, target)
                loss.backward()
                optimizer.step()
                metrics.update(n=data.size(0), loss=loss.item(), error=error)
            logging.info(f"[epoch {epoch}]: " + "\t".join(f"{k}: {v}" for k, v in metrics.avg.items()))
        print(f'\nfinal loss after fitting final layer {loss=}')

    def extract_embedding(self, model: ProbeNetwork):
        """
        Reads the values stored by `compute_fisher` and returns them in a common format that describes the diagonal of the
        Fisher Information Matrix for each layer.

        :param model:
        :return:
        """
        if self.mode == 'autoregressive':
            hess, scale = [], []
            for name, module in model.named_modules():
                if module is model.lm_head:
                    continue
                # The other Fisher approximation methods directly approximate the hessian at the minimum
                if hasattr(module, 'weight') and hasattr(module.weight, 'grad2_acc'):
                    grad2 = module.weight.grad2_acc.cpu().detach().numpy()
                    filterwise_hess = grad2.reshape(grad2.shape[0], -1).mean(axis=1)
                    hess.append(filterwise_hess)
                    scale.append(np.ones_like(filterwise_hess))
        else:
            hess, scale = [], []
            for name, module in model.named_modules():
                if module is model.classifier:
                    continue
                # The variational Fisher approximation estimates the variance of noise that can be added to the weights
                # without increasing the error more than a threshold. The inverse of this is proportional to an
                # approximation of the hessian in the local minimum.
                if hasattr(module, 'logvar0') and hasattr(module, 'loglambda2'):
                    logvar = module.logvar0.view(-1).detach().cpu().numpy()
                    hess.append(np.exp(-logvar))
                    loglambda2 = module.loglambda2.detach().cpu().numpy()
                    scale.append(np.exp(-loglambda2).repeat(logvar.size))
                # The other Fisher approximation methods directly approximate the hessian at the minimum
                elif hasattr(module, 'weight') and hasattr(module.weight, 'grad2_acc'):
                    grad2 = module.weight.grad2_acc.cpu().detach().numpy()
                    filterwise_hess = grad2.reshape(grad2.shape[0], -1).mean(axis=1)
                    hess.append(filterwise_hess)
                    scale.append(np.ones_like(filterwise_hess))
        return Embedding(hessian=np.concatenate(hess), scale=np.concatenate(scale), meta=None)


def _get_loader(trainset, testset=None, batch_size=64, num_workers=0, num_samples=10000, drop_last=True):
    if getattr(trainset, 'is_multi_label', False):
        raise ValueError("Multi-label datasets not supported")
    # TODO: Find a way to standardize this
    if hasattr(trainset, 'labels'):
        labels = trainset.labels
    elif hasattr(trainset, 'targets'):
        labels = trainset.targets
    else:
        labels = list(trainset.tensors[1].cpu().numpy())
    num_classes = int(getattr(trainset, 'num_classes', max(labels) + 1))
    class_count = np.eye(num_classes)[labels].sum(axis=0)
    weights = 1. / class_count[labels] / num_classes
    weights /= weights.sum()

    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, num_samples=num_samples)
    # No need for mutli-threaded loading if everything is already in memory,
    # and would raise an error if TensorDataset is on CUDA
    num_workers = num_workers if not isinstance(trainset, torch.utils.data.TensorDataset) else 0
    trainloader = torch.utils.data.DataLoader(trainset, sampler=sampler, batch_size=batch_size,
                                              num_workers=num_workers, drop_last=drop_last)

    if testset is None:
        return trainloader
    else:
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, pin_memory=True, shuffle=False,
                                                 num_workers=num_workers)
        return trainloader, testloader
