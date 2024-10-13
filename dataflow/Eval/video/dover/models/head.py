import math

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.ops import roi_align, roi_pool


class VQAHead(nn.Module):
    """MLP Regression Head for VQA.
    Args:
        in_channels: input channels for MLP
        hidden_channels: hidden channels for MLP
        dropout_ratio: the dropout ratio for features before the MLP (default 0.5)
        pre_pool: whether pre-pool the features or not (True for Aesthetic Attributes, False for Technical Attributes)
    """

    def __init__(
        self, in_channels=768, hidden_channels=64, dropout_ratio=0.5, pre_pool=False, **kwargs
    ):
        super().__init__()
        self.dropout_ratio = dropout_ratio
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.pre_pool = pre_pool
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        self.fc_hid = nn.Conv3d(self.in_channels, self.hidden_channels, (1, 1, 1))
        self.fc_last = nn.Conv3d(self.hidden_channels, 1, (1, 1, 1))
        self.gelu = nn.GELU()

        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

    def forward(self, x, rois=None):
        if self.pre_pool:
            x = self.avg_pool(x)
        x = self.dropout(x)
        qlt_score = self.fc_last(self.dropout(self.gelu(self.fc_hid(x))))
        return qlt_score
    
    



class VARHead(nn.Module):
    """MLP Regression Head for Video Action Recognition.
    Args:
        in_channels: input channels for MLP
        hidden_channels: hidden channels for MLP
        dropout_ratio: the dropout ratio for features before the MLP (default 0.5)
    """

    def __init__(self, in_channels=768, out_channels=400, dropout_ratio=0.5, **kwargs):
        super().__init__()
        self.dropout_ratio = dropout_ratio
        self.in_channels = in_channels
        self.out_channels = out_channels
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        self.fc = nn.Conv3d(self.in_channels, self.out_channels, (1, 1, 1))
        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

    def forward(self, x, rois=None):
        x = self.dropout(x)
        x = self.avg_pool(x)
        out = self.fc(x)
        return out


class IQAHead(nn.Module):
    """MLP Regression Head for IQA.
    Args:
        in_channels: input channels for MLP
        hidden_channels: hidden channels for MLP
        dropout_ratio: the dropout ratio for features before the MLP (default 0.5)
    """

    def __init__(
        self, in_channels=768, hidden_channels=64, dropout_ratio=0.5, **kwargs
    ):
        super().__init__()
        self.dropout_ratio = dropout_ratio
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        self.fc_hid = nn.Linear(self.in_channels, self.hidden_channels)
        self.fc_last = nn.Linear(self.hidden_channels, 1)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.dropout(x)
        qlt_score = self.fc_last(self.dropout(self.gelu(self.fc_hid(x))))
        return qlt_score
