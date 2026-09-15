# Copyright (c) OpenMMLab. All rights reserved.
import logging
from typing import List, Optional, Union, Dict
 
import torch  
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmengine.logging import print_log
from mmengine.optim import OptimWrapper 

from mmengine.registry import MODELS
from mmseg.utils import (ConfigType, OptConfigType, OptMultiConfig,
                         OptSampleList, SampleList, add_prefix)
from mmseg.models.segmentors import BaseSegmentor

import os
try:
    import rasterio
except Exception as e:
    print("fail to import rasterio", e)

norm_cfg = dict(requires_grad=True, type='SyncBN') 


@MODELS.register_module()
class PreMapper(nn.Module): 
    def __init__(self, 
                in_channels: int, 
                num_features: int=3, 
                layer_type: str ='conv'):
        super().__init__()
        #현재는 간단한 conv레이어만 있고 
        #추후에 Attention 타입 레이어 추가 
        if layer_type=='conv':
            self.layers = nn.ModuleList([
                nn.Conv2d(in_channels, num_features, 1,1,0),
                nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1)
            ])
        elif layer_type == 'conv_v2':
            self.layers = nn.ModuleList([
                nn.Conv2d(in_channels, num_features, kernel_size=3, stride=1, padding=1),  # 3x3 Conv로 공간 정보 반영
                nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1),
                nn.ReLU(inplace=True),  # 활성화 함수 추가 (ReLU 또는 Tanh)
                nn.Conv2d(num_features, num_features, kernel_size=1, stride=1, padding=0),  # 1x1 Conv로 채널 정리
                nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1)
            ])
        elif layer_type == 'landcover_mapper':
            # 논문 방식: 15채널 데이터를 3채널로 변환
            self.layers = nn.ModuleList([
                nn.Conv2d(in_channels, num_features, kernel_size=7, stride=1, padding=3),  # 커널 크기: 7x7
                nn.Tanh()  # Tanh 활성화 함수로 [-1, 1] 범위로 제한
            ])
            self._initialize_weights()
        else:
            raise ValueError("Pre mapper supports only conv layer or attention layer") 

    def forward(self, x):  
        for layer in self.layers:
            x = layer(x) 
        return x 

    def _initialize_weights(self):
        """Apply Xavier initialization to Conv2D layers."""
        for layer in self.layers:
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_uniform_(layer.weight)  # Xavier 초기화
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)  # 바이어스 초기화
    
