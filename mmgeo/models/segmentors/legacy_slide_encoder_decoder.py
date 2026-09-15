# Copyright (c) OpenMMLab. All rights reserved.
import logging
from typing import List, Optional

import torch.nn as nn
import torch.nn.functional as F
from mmengine.logging import print_log
from torch import Tensor

from mmengine.registry import MODELS
from mmseg.utils import (ConfigType, OptConfigType, OptMultiConfig,
                         OptSampleList, SampleList, add_prefix)
from mmseg.models.segmentors.base import BaseSegmentor
from mmseg.models.segmentors import EncoderDecoder  
from mmseg.models.losses import accuracy
from mmseg.models.utils import resize 
import torch 
from mmengine.structures import PixelData
from mmseg.structures import SegDataSample
try:
    import segmentation_models_pytorch as smp
except:
    pass 

@MODELS.register_module()
class SlideEncoderDecoder(EncoderDecoder):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.

    1. The ``loss`` method is used to calculate the loss of model,
    which includes two steps: (1) Extracts features to obtain the feature maps
    (2) Call the decode head loss function to forward decode head model and
    calculate losses.

    .. code:: text

     loss(): extract_feat() -> _decode_head_forward_train() -> _auxiliary_head_forward_train (optional)
     _decode_head_forward_train(): decode_head.loss()
     _auxiliary_head_forward_train(): auxiliary_head.loss (optional)

    2. The ``predict`` method is used to predict segmentation results,
    which includes two steps: (1) Run inference function to obtain the list of
    seg_logits (2) Call post-processing function to obtain list of
    ``SegDataSample`` including ``pred_sem_seg`` and ``seg_logits``.

    .. code:: text

     predict(): inference() -> postprocess_result()
     infercen(): whole_inference()/slide_inference()
     whole_inference()/slide_inference(): encoder_decoder()
     encoder_decoder(): extract_feat() -> decode_head.predict()

    3. The ``_forward`` method is used to output the tensor by running the model,
    which includes two steps: (1) Extracts features to obtain the feature maps
    (2)Call the decode head forward function to forward decode head model.

    .. code:: text

     _forward(): extract_feat() -> _decode_head.forward()

    Args:

        backbone (ConfigType): The config for the backnone of segmentor.
        decode_head (ConfigType): The config for the decode head of segmentor.
        neck (OptConfigType): The config for the neck of segmentor.
            Defaults to None.
        auxiliary_head (OptConfigType): The config for the auxiliary head of
            segmentor. Defaults to None.
        train_cfg (OptConfigType): The config for training. Defaults to None.
        test_cfg (OptConfigType): The config for testing. Defaults to None.
        data_preprocessor (dict, optional): The pre-process config of
            :class:`BaseDataPreprocessor`.
        pretrained (str, optional): The path for pretrained model.
            Defaults to None.
        init_cfg (dict, optional): The weight initialized config for
            :class:`BaseModule`.
    """  # noqa: E501

    def __init__(self, 
                 backbone: ConfigType,
                 decode_head: ConfigType,
                 neck: OptConfigType = None,
                 auxiliary_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 pretrained: Optional[str] = None,
                 init_cfg: OptMultiConfig = None,
                 n_argumemt: int = None): 
        super().__init__(
            backbone = backbone, 
            decode_head = decode_head ,
            neck = neck, 
            auxiliary_head = auxiliary_head,
            train_cfg = train_cfg, 
            test_cfg = test_cfg, 
            pretrained = pretrained, 
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        
        self.n_argumemt = n_argumemt  

    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            inputs (Tensor): Input images.
            data_samples (list[:obj:`SegDataSample`]): The seg data samples.
                It usually includes information such as `metainfo` and
                `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """ 
        mode = self.train_cfg.get('mode', 'whole')
        if mode == 'slide':
            h_stride, w_stride = self.train_cfg.stride
            h_crop, w_crop = self.train_cfg.crop_size
            batch_size, _, h_img, w_img = inputs.size() 
            h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
            w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1 
            for h_idx in range(h_grids):
                for w_idx in range(w_grids):
                    y1 = h_idx * h_stride
                    x1 = w_idx * w_stride
                    y2 = min(y1 + h_crop, h_img)
                    x2 = min(x1 + w_crop, w_img)
                    y1 = max(y2 - h_crop, 0)
                    x1 = max(x2 - w_crop, 0)
                    crop_img = inputs[:, :, y1:y2, x1:x2] 

                    #=======
                    tmp = []
                    for i in range(len(data_samples)):   
                        data_sample = SegDataSample() 
                        img_meta = data_samples[i].metainfo
                        img_meta['img_shape'] = (h_crop, w_crop)
                        img_meta['pad_shape'] = (h_crop, w_crop)
                        gt_seg = PixelData(metainfo = img_meta )
                        gt_seg.data = data_samples[i].gt_sem_seg.data[:,y1:y2, x1:x2].cuda()
                        data_sample.gt_sem_seg = gt_seg 
                        tmp.append(data_sample)
                    data_samples = tmp   
                    #====
 

                    x = self.extract_feat(crop_img) 
                    losses = dict()
                    loss_decode = self._decode_head_forward_train(x, data_samples)
                    losses.update(loss_decode)
                    if self.with_auxiliary_head:
                        loss_aux = self._auxiliary_head_forward_train(x, data_samples)
                        losses.update(loss_aux) 
        else: 
            x = self.extract_feat(inputs) 

            losses = dict()

            loss_decode = self._decode_head_forward_train(x, data_samples)
    
            losses.update(loss_decode)

            if self.with_auxiliary_head:
                loss_aux = self._auxiliary_head_forward_train(x, data_samples)
                losses.update(loss_aux)

        return losses
 