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

from collections import defaultdict
import torch
import numpy as np


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = defaultdict(int)
        self.avg = defaultdict(float)
        self.sum = defaultdict(int)
        self.count = defaultdict(int)

    def update(self, n=1, **val):
        for k in val:
            self.val[k] = val[k]
            self.sum[k] += val[k] * n
            self.count[k] += n
            self.avg[k] = self.sum[k] / self.count[k]


def set_batchnorm_mode(model, train=True):
    """Allows to set batch_norm layer mode to train or eval, independendtly on the mode of the model."""
    def _set_batchnorm_mode(module):
        if isinstance(module, torch.nn.BatchNorm1d) or isinstance(module, torch.nn.BatchNorm2d):
            if train:
                module.train()
            else:
                module.eval()

    model.apply(_set_batchnorm_mode)

### LLM DIV
def get_error(output, target, mode='autoregressive', ignore_index=None):
    if mode == 'autoregressive': # output = logits here
        assert ignore_index is not None
        output = output[:,:-1,:] 
        logits_inds = torch.argmax(output, dim=-1)
        target = target[:,1:]
        if ignore_index is not None:
            acc = torch.eq(logits_inds, target.unsqueeze(0))[:, target != ignore_index]
        else:
            acc = torch.eq(logits_inds, target.unsqueeze(0))
        acc = acc.float().mean()
        return 1 - acc
    else:
        pred = output.argmax(dim=1)
        correct = pred.eq(target).float().sum()
        return float((1. - correct / output.size(0)) * 100.)


def adjust_learning_rate(optimizer, epoch, optimizer_cfg):
    lr = optimizer_cfg.lr * (0.1 ** np.less(optimizer_cfg.schedule, epoch).sum())
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_device(model: torch.nn.Module):
    return next(model.parameters()).device
