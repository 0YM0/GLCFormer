# Copyright (c) OpenMMLab. All rights reserved.
import logging
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.logging import print_log
from torch import Tensor

from mmengine.registry import MODELS
from mmseg.utils import (ConfigType, OptConfigType, OptMultiConfig,
                         OptSampleList, SampleList, add_prefix)
from mmseg.models.segmentors.base import BaseSegmentor
from mmseg.models.losses import accuracy
from mmseg.models.utils import resize 
import segmentation_models_pytorch as smp

from mmengine.structures import PixelData
from mmseg.structures import SegDataSample

@MODELS.register_module()
class SMPSegmentor(BaseSegmentor):
    """Encoder Decoder model for smp module
    """
    def __init__(self, 
                 model_cfg: ConfigType, 
                 head_cfg: OptConfigType = None, 
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 pretrained: Optional[str] = None,
                 init_cfg: OptMultiConfig = None,
                 ):
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg) 
        ######     
        self.model_cfg = model_cfg
        smp_model = self._build_smp_model(model_cfg) 
        self.encoder = smp_model.encoder
        self.decoder = smp_model.decoder 
        self.segment_head = smp_model.segmentation_head 
  
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg


        self.num_classes = model_cfg['num_classes']
        self.align_corners = head_cfg.get('align_corners', False)
        loss_decode = head_cfg['loss_decode']
        
        if isinstance(loss_decode, dict):
            self.loss_decode = MODELS.build(loss_decode)
        elif isinstance(loss_decode, (list, tuple)):
            self.loss_decode = nn.ModuleList()
            for loss in loss_decode:
                self.loss_decode.append(MODELS.build(loss))
        else:
            raise TypeError(f'loss_decode must be a dict or sequence of dict,\
                but got {type(loss_decode)}') 
        
    @property
    def get_model(self):
        def smp_model(x):
            features = self.encoder(x)
            decoder_output = self.decoder(*features)
            masks = self.segment_head(decoder_output)
            return masks 
        return smp_model 

    def _build_smp_model(self, cfg: ConfigType) -> None:
        modelClass = getattr(smp, cfg['type']) 
        if 'decoder_channels' in cfg.keys():
            modelClass(
                decoder_channels = cfg.get('decoder_channels', (128, 64, 64)),
            )
        return modelClass(
            encoder_name = cfg['encoder_name'],
            classes = cfg['num_classes'],
            encoder_depth = cfg.get('encoder_depth', 3),
            encoder_weights = cfg.get('encoder_weights'), 
            in_channels = cfg.get('in_channels', 3),
            upsampling = cfg.get('upsampling', 4 )
        ) 
 
    def extract_feat(self, inputs: Tensor) -> List[Tensor]:
        """Extract features from images."""
        x = self.encoder(inputs)
        x = self.decoder(*x) 
        return x
    
    def _stack_batch_gt(self, batch_data_samples: SampleList) -> Tensor:
        gt_semantic_segs = [
            data_sample.gt_sem_seg.data for data_sample in batch_data_samples
        ]
        return torch.stack(gt_semantic_segs , dim=0)
    
    def _loss_by_feat(self, seg_logits: Tensor,
                     batch_data_samples: SampleList,
                     train_cfg: ConfigType, 
                     ignore_index=255,
                     sampler = None,
                     ) -> dict:
        seg_label = self._stack_batch_gt(batch_data_samples)
        loss = dict()  
        seg_logits = resize(
            input=seg_logits,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        
        if sampler is not None:
            seg_weight = self.sampler.sample(seg_logits, seg_label)
        else:
            seg_weight = None
        seg_label = seg_label.squeeze(1)

        if not isinstance(self.loss_decode, nn.ModuleList):
            losses_decode = [self.loss_decode]
        else:
            losses_decode = self.loss_decode
        for loss_decode in losses_decode:
            if loss_decode.loss_name not in loss:
                loss[loss_decode.loss_name] = loss_decode(
                    seg_logits,
                    seg_label,
                    weight=seg_weight,
                    ignore_index=ignore_index)
            else:
                loss[loss_decode.loss_name] += loss_decode(
                    seg_logits,
                    seg_label,
                    weight=seg_weight,
                    ignore_index=ignore_index)

        loss['acc_seg'] = accuracy(
            seg_logits, seg_label, ignore_index=ignore_index)
        return loss
    
    def _decode_head_forward_train(self, inputs: Tensor,
                                   data_samples: SampleList) -> dict:
        losses = dict()  
        seg_logits = self.segment_head(inputs) 
        loss_decode = self._loss_by_feat(seg_logits, data_samples,
                                         self.train_cfg) 
        losses.update(add_prefix(loss_decode,'decode'))
        return losses
        

    def loss(self, inputs: Tensor, data_samples: SampleList):
        x = self.extract_feat(inputs)
        losses = dict() 
        loss_decode = self._decode_head_forward_train(x, data_samples)
        losses.update(loss_decode) 
        return losses 
    
    def _forward(self, inputs:Tensor, 
                 data_samples: OptSampleList=None) -> Tensor: 
        x = self.extract_feat(inputs) 
        seg_logits = self.segment_head(x)  
        return seg_logits
    
    def _predict_by_feat(self, seg_logits: Tensor,
                        batch_img_metas: List[dict]) -> Tensor:
        if isinstance(batch_img_metas[0]['img_shape'], torch.Size):
            # slide inference
            size = batch_img_metas[0]['img_shape']
        elif 'pad_shape' in batch_img_metas[0]:
            size = batch_img_metas[0]['pad_shape'][:2]
        else:
            size = batch_img_metas[0]['img_shape']
        seg_logits = resize(
            input=seg_logits,
            size=size,
            mode='bilinear',
            align_corners=self.align_corners)
        return seg_logits
    
    def encode_decode(self, inputs:Tensor,
                      batch_img_metas: List[dict]) -> Tensor: 
        x = self.extract_feat(inputs)
        seg_logits = self.segment_head(x) 
        seg_logits = self._predict_by_feat(seg_logits, batch_img_metas)
        return seg_logits
    
    def predict(self, 
                inputs: Tensor,
                data_samples: OptSampleList = None) -> SampleList:
        if data_samples is not None:
            batch_img_metas = [
                data_sample.metainfo for data_sample in data_samples 
            ]
        else:
            batch_img_metas = [
                dict(
                    ori_shape=inputs.shape[2:],
                    img_shape=inputs.shape[2:],
                    pad_shape=inputs.shape[2:],
                    padding_size=[0, 0, 0, 0])
            ] * inputs.shape[0]
        seg_logits = self.inference(inputs, batch_img_metas)
        return self.postprocess_result(seg_logits, data_samples)
    
    def slide_inference(self, inputs: Tensor,
                        batch_img_metas: List[dict]) -> Tensor:
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.

        Args:
            inputs (tensor): the tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            batch_img_metas (List[dict]): List of image metainfo where each may
                also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', and 'pad_shape'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.

        Returns:
            Tensor: The segmentation results, seg_logits from model of each
                input image.
        """

        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = inputs.size() 
        out_channels = self.num_classes 
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = inputs.new_zeros((batch_size, out_channels, h_img, w_img))
        count_mat = inputs.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = inputs[:, :, y1:y2, x1:x2]
                # change the image shape to patch shape
                batch_img_metas[0]['img_shape'] = crop_img.shape[2:]
                # the output of encode_decode is seg logits tensor map
                # with shape [N, C, H, W]
                crop_seg_logit = self.encode_decode(crop_img, batch_img_metas)
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        seg_logits = preds / count_mat

        return seg_logits

    def whole_inference(self, inputs: Tensor,
                        batch_img_metas: List[dict]) -> Tensor:
        """Inference with full image.

        Args:
            inputs (Tensor): The tensor should have a shape NxCxHxW, which
                contains all images in the batch.
            batch_img_metas (List[dict]): List of image metainfo where each may
                also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', and 'pad_shape'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.

        Returns:
            Tensor: The segmentation results, seg_logits from model of each
                input image.
        """

        seg_logits = self.encode_decode(inputs, batch_img_metas)

        return seg_logits
    
    def inference(self,inputs:Tensor, batch_img_metas: List[dict])-> Tensor:
        assert self.test_cfg.get('mode', 'whole') in ['slide', 'whole'], \
            f'Only "slide" or "whole" test mode are supported, but got ' \
            f'{self.test_cfg["mode"]}.'
        ori_shape = batch_img_metas[0]['ori_shape']
        if not all(_['ori_shape'] == ori_shape for _ in batch_img_metas):
            print_log(
                'Image shapes are different in the batch.',
                logger='current',
                level=logging.WARN)
        if self.test_cfg.mode == 'slide':
            seg_logit = self.slide_inference(inputs, batch_img_metas)
        else:
            seg_logit = self.whole_inference(inputs, batch_img_metas)
        return seg_logit
    
    def aug_test(self, inputs, batch_img_metas, rescale=True):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale
        # to save memory, we get augmented seg logit inplace
        seg_logit = self.inference(inputs[0], batch_img_metas[0], rescale)
        for i in range(1, len(inputs)):
            cur_seg_logit = self.inference(inputs[i], batch_img_metas[i],
                                           rescale)
            seg_logit += cur_seg_logit
        seg_logit /= len(inputs)
        seg_pred = seg_logit.argmax(dim=1)
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred
