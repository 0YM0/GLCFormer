# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from  mmengine.registry import MODELS 

@MODELS.register_module()
class MSELoss(nn.Module):
    def __init__(self, reduction='mean', loss_weight=1.0,
                loss_name = 'MSELoss'):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.criterion = nn.MSELoss(reduction=reduction)
        self.loss_name_ = loss_name 

    def forward(self, pred, target, weight=None, **kwargs):
        loss = self.criterion(pred, target)
        return self.loss_weight * loss

    @property
    def loss_name(self):
        return self.loss_name_
