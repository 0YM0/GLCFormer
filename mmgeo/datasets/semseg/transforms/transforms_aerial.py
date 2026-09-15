# Copyright (c) OpenMMLab. All rights reserved.
import copy
import inspect
import warnings
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2
import mmcv
import mmengine
import numpy as np 
from mmcv.transforms import Pad as MMCV_Pad
from mmcv.transforms.base import BaseTransform
from mmcv.transforms.utils import cache_randomness 
from numpy import random 
from mmengine.registry import TRANSFORMS



@TRANSFORMS.register_module()
class PadAll(BaseTransform): 
    def __init__(self, pad_size: Sequence = (512,512) ):
        self.pad_size = pad_size

    def pad(self, image, ndim=2):
        h,w = image.shape[:2] 
        pad_height = max(0,self.pad_size[0] - h)   
        pad_width = max(0, self.pad_size[1] - w)    
        if ndim==2:
            pad_img = np.pad(image, 
                    ((0, pad_height), (0,pad_width)), 
                    mode='constant', 
                    constant_values=0) 
        else:
            pad_img = np.pad(image, 
            ((0, pad_height), (0,pad_width), (0,0)), 
            mode='constant', 
            constant_values=0) 
        return pad_img 

    def transform(self, results: dict) -> dict:
        results['img']= MMCV_Pad(results['img'], 
                                size=self.pad_size,
                                pad_val = dict(
                                    img=0,
                                    se255
                                )) 
        results['gt_seg_map'] = self.pad(results['gt_seg_map'], ndim=2) 
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__ 
        return repr_str 