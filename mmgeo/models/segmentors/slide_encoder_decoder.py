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

@MODELS.register_module()
class SlideEncoderDecoder(BaseSegmentor): 
    def __init__(self, 
                 backbone: ConfigType,
                 decode_head: ConfigType,
                 neck: OptConfigType = None,
                 auxiliary_head: OptConfigType = None,
                 pre_mapper: OptConfigType = None, 
                 graph_fusion: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 pretrained: Optional[str] = None,
                 init_cfg: OptMultiConfig = None):
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        if pretrained is not None: 
            assert backbone.get('pretrained') is None, \
                'both backbone and segmentor set pretrained weight'
            backbone.pretrained = pretrained
        self.backbone = MODELS.build(backbone)
        if neck is not None:
            self.neck = MODELS.build(neck)
        self._init_decode_head(decode_head)
        self._init_auxiliary_head(auxiliary_head)
        self._init_pre_mapper(pre_mapper)
        self._init_graph_fusion(graph_fusion)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg 

        assert self.with_decode_head

    def _init_decode_head(self, decode_head: ConfigType) -> None:
        """Initialize ``decode_head``"""
        self.decode_head = MODELS.build(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes
        self.out_channels = self.decode_head.out_channels

    def _init_auxiliary_head(self, auxiliary_head: ConfigType) -> None:
        """Initialize ``auxiliary_head``"""
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                self.auxiliary_head = nn.ModuleList()
                for head_cfg in auxiliary_head:
                    self.auxiliary_head.append(MODELS.build(head_cfg))
            else:
                self.auxiliary_head = MODELS.build(auxiliary_head)

    def _init_pre_mapper(self, pre_mapper: ConfigType) -> None: 
        if pre_mapper is not None: 
            self.pre_mapper = MODELS.build(pre_mapper)
            print('pre mapper is ', self.pre_mapper)

    def _init_graph_fusion(self, graph_fusion: ConfigType) -> None:
        if graph_fusion is not None:
            self.graph_fusion = MODELS.build(graph_fusion)
            print('graph fusion is ', self.graph_fusion)

    def extract_feat(self, inputs: Tensor, data_samples: OptSampleList = None) -> List[Tensor]:
        """Extract features from images.""" 
        if hasattr(self, 'pre_mapper'):
            if isinstance(self.pre_mapper, nn.Module): 
                #inputs: 쉐이프가 B,C,H,W로 가정 (B, 3+C_landcover, H, W)
                LU = inputs[:,3:, :,:]   
                LU_embed = self.pre_mapper(LU) 
            x = inputs[:,:3,:,:]  + LU_embed
            x = self.backbone(x)   
        else:
           x = self.backbone(inputs)  
        if self.with_neck:
            x = self.neck(x)
        if hasattr(self, 'graph_fusion'):
            x = self.graph_fusion(x, data_samples)
        return x

    def encode_decode(self, inputs: Tensor,
                      batch_img_metas: List[dict],
                      data_samples: OptSampleList = None) -> Tensor:
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""  
        x = self.extract_feat(inputs, data_samples)  
        seg_logits = self.decode_head.predict(x, batch_img_metas,
                                              self.test_cfg) 

        return seg_logits

    def _decode_head_forward_train(self, inputs: List[Tensor],
                                   data_samples: SampleList) -> dict:
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        #print('data:', data_samples)
        loss_decode = self.decode_head.loss(inputs, data_samples,
                                            self.train_cfg)

        losses.update(add_prefix(loss_decode, 'decode'))
        return losses

    def _auxiliary_head_forward_train(self, inputs: List[Tensor],
                                      data_samples: SampleList) -> dict:
        """Run forward function and calculate loss for auxiliary head in
        training."""
        losses = dict()
        if isinstance(self.auxiliary_head, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head):
                loss_aux = aux_head.loss(inputs, data_samples, self.train_cfg)
                losses.update(add_prefix(loss_aux, f'aux_{idx}'))
        else:
            loss_aux = self.auxiliary_head.loss(inputs, data_samples,
                                                self.train_cfg)
            losses.update(add_prefix(loss_aux, 'aux'))

        return losses

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
        x = self.extract_feat(inputs, data_samples) 

        losses = dict()

        loss_decode = self._decode_head_forward_train(x, data_samples)
 
        losses.update(loss_decode)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(x, data_samples)
            losses.update(loss_aux)

        if hasattr(self, 'graph_fusion') and hasattr(self.graph_fusion, 'get_aux_losses'):
            graph_aux_losses = self.graph_fusion.get_aux_losses()
            if graph_aux_losses:
                losses.update(add_prefix(graph_aux_losses, 'graph'))

        return losses

    @staticmethod
    def draw_img():
        #image save 
        img_path = batch_img_metas[0]['img_path']
        img_name = os.path.basename(img_path)
        save_dir = self.test_cfg.get('save_dir', None) 
        if save_dir is None:
            save_dir = os.path.join(
                os.path.dirname(os.path.dirname(img_path)),
                                    preds)
        os.makedirs(save_dir, exist_ok = True )
        out_path = os.path.join(save_dir, img_name.replace(".tif", "_preds.tif"))
        with rasterio.open(img_path) as src:
            output_profile = src.profile.copy()  
            out_arr = torch.argmax(seg_logits[0].cpu(), dim=0) 
            print(torch.unique(out_arr))
            print(out_arr.shape) 
            output_profile.update(
                dict(
                    height = out_arr.shape[0], 
                    width = out_arr.shape[1], 
                )
            )
            print(output_profile)
            with rasterio.open(out_path, 'w', **output_profile) as dst:
                dst.write(out_arr) #device -> cpu -> numpy  
 

    #슬라이딩 가능하도록 수정 ? 
    def predict(self,
                inputs: Tensor,
                data_samples: OptSampleList = None) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W).
            data_samples (List[:obj:`SegDataSample`], optional): The seg data
                samples. It usually includes information such as `metainfo`
                and `gt_sem_seg`.

        Returns:
            list[:obj:`SegDataSample`]: Segmentation results of the
            input images. Each SegDataSample usually contain:

            - ``pred_sem_seg``(PixelData): Prediction of semantic segmentation.
            - ``seg_logits``(PixelData): Predicted logits of semantic
                segmentation before normalization.
        """
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

        seg_logits = self.inference(inputs, batch_img_metas, data_samples) 

        return self.postprocess_result(seg_logits, data_samples)

    def _forward(self,
                 inputs: Tensor,
                 data_samples: OptSampleList = None) -> Tensor:
        """Network forward process.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W).
            data_samples (List[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `metainfo` and `gt_sem_seg`.

        Returns:
            Tensor: Forward output of model without any post-processes.
        """ 
        x = self.extract_feat(inputs, data_samples) 
        return self.decode_head.forward(x) 

    def _resolve_slide_graph_path(self, meta: dict, x: int, y: int,
                                  chip_size: int) -> Optional[str]:
        graph_dir = meta.get('graph_dir', None)
        img_path = meta.get('img_path', None)
        if graph_dir is None or img_path is None:
            return None
        image_base = os.path.splitext(os.path.basename(img_path))[0]
        return os.path.join(graph_dir, f'{image_base}_x{int(x)}_y{int(y)}_s{int(chip_size)}.pt')

    def _load_slide_graph(self, graph_path: Optional[str]):
        if graph_path is None:
            return None
        if not hasattr(self, '_slide_graph_loader'):
            from mmgeo.datasets.semseg.transforms.loading_graph import LoadLandCoverGraph
            self._slide_graph_loader = LoadLandCoverGraph(required=False, allow_missing=True)
        return self._slide_graph_loader.transform({'graph_path': graph_path})['landcover_graph']

    def _build_slide_data_sample(self, meta: dict):
        from mmseg.structures import SegDataSample

        data_sample = SegDataSample()
        sample_meta = meta.copy()
        graph = self._load_slide_graph(sample_meta.get('graph_path', None))
        if graph is not None:
            sample_meta['landcover_graph'] = graph
        data_sample.set_metainfo(sample_meta)
        return data_sample

    def slide_inference(self, inputs: List,
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
        #그냥 무조간 val tile은 batch가 1개라고 가정함  
        tile_input = inputs[0]
        batch_meta_single = batch_img_metas[0]

        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        _, h_img, w_img = tile_input.size()
        out_channels = self.out_channels #num_classes 임 
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1

        preds = torch.zeros((1,out_channels, h_img, w_img))
        count_mat = torch.zeros((1, 1, h_img, w_img))
 
        preds = self.data_preprocessor.cast_data(preds) 
        count_mat = self.data_preprocessor.cast_data(count_mat) 

        #batch_count = 0
        patch_batch_size = self.test_cfg.patch_batch_size
        assert isinstance(patch_batch_size, int) and patch_batch_size>0, \
                "set the patch_batch_size for slide inference"
        
        batch_crop_img, crop_points, meta_batch = [], [], []
        data_sample_batch = []
        use_slide_graph = hasattr(self, 'graph_fusion')
        slide_graph_loaded = 0
        slide_graph_missing = 0

        #batch_metas = [batch_metas for _ in range(patch_batch_size)]

        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0) 
                crop_img = tile_input[:, y1:y2, x1:x2]
                ##if torch.sum(crop_img)<1:
                #    continue  
                c,h,w = crop_img.shape
                if (h,w) != (h_crop, w_crop): #마지막 부분이 남으면 패딩 
                    pad = (0, w_crop-w ,0, h_crop-h) #left, right, top, bottom
                    crop_img = F.pad(crop_img, pad, mode='constant', value=0) 
                crop_points.append((y1,y2,x1,x2)) #채택된 이미지만 담아야 함  
                meta = batch_meta_single.copy() 
                meta['img_shape'] = (h_crop, w_crop)
                meta['read_bbox'] = (x1, y1, w_crop, h_crop)
                if use_slide_graph:
                    graph_path = self._resolve_slide_graph_path(meta, x1, y1, h_crop)
                    meta['graph_path'] = graph_path
                    if graph_path is not None and os.path.exists(graph_path):
                        slide_graph_loaded += 1
                    elif graph_path is not None:
                        slide_graph_missing += 1
                    data_sample_batch.append(self._build_slide_data_sample(meta))
                meta_batch.append(meta) 
                batch_crop_img.append(crop_img)

                #batch_count +=1 
                #batch 개수만큼 쌓이거나 아니면 쌓이기 전에 마지막 배치이면  
                if len(batch_crop_img) == patch_batch_size:
                    #batch_count%patch_batch_size==0 or \
                    #(h_idx == h_grids-1):
                    batch_input = torch.stack(batch_crop_img, dim=0)   
                    batch_input = self.data_preprocessor.cast_data(batch_input)   
                    crop_data_samples = data_sample_batch if use_slide_graph else None
                    crop_seg_logits = self.encode_decode(batch_input, meta_batch, crop_data_samples) 
                    for idx, (y1,y2,x1,x2) in enumerate(crop_points):  
                        crop_pred = crop_seg_logits[idx][:, 0:y2-y1, 0:x2-x1]
                        preds[:,:, y1:y2, x1:x2] += crop_pred 
                        count_mat[:,:, y1:y2, x1:x2] += 1
                    batch_crop_img, crop_points, meta_batch, data_sample_batch = [], [], [], []

        # 루프 바깥에서 누락 타일 처리 
        if batch_crop_img:
            batch_input = torch.stack(batch_crop_img, dim=0)
            batch_input = self.data_preprocessor.cast_data(batch_input)
            crop_data_samples = data_sample_batch if use_slide_graph else None
            crop_seg_logits = self.encode_decode(batch_input, meta_batch, crop_data_samples)
            for idx, (y1, y2, x1, x2) in enumerate(crop_points):
                crop_pred = crop_seg_logits[idx][:, 0:y2 - y1, 0:x2 - x1]
                preds[:, :, y1:y2, x1:x2] += crop_pred
                count_mat[:, :, y1:y2, x1:x2] += 1

        if use_slide_graph and batch_meta_single.get('graph_dir', None) is not None:
            print_log(
                f"[LandCoverGraph] slide inference graphs: loaded={slide_graph_loaded}, "
                f"missing={slide_graph_missing}, graph_dir={batch_meta_single.get('graph_dir')}",
                logger='current',
                level=logging.INFO)
 
        preds[0,0][count_mat[0,0]==0] = 1e+7 #패스한 부분은 0번 클래스를 최대로 만들어줌
        count_mat[count_mat==0] = 1 #패스한 부분은 그냥 1로 채움  
        seg_logits = preds / count_mat  
        return seg_logits

    def whole_inference(self, inputs: Tensor,
                        batch_img_metas: List[dict],
                        data_samples: OptSampleList = None) -> Tensor:
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
        seg_logits = self.encode_decode(inputs, batch_img_metas, data_samples)

        return seg_logits

    def inference(self, inputs: Tensor, batch_img_metas: List[dict],
                  data_samples: OptSampleList = None) -> Tensor:
        """Inference with slide/whole style.

        Args:
            inputs (Tensor): The input image of shape (N, 3, H, W).
            batch_img_metas (List[dict]): List of image metainfo where each may
                also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', 'pad_shape', and 'padding_size'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.

        Returns:
            Tensor: The segmentation results, seg_logits from model of each
                input image.
        """
        if isinstance(data_samples, bool):
            data_samples = None

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
            seg_logit = self.whole_inference(inputs, batch_img_metas, data_samples)
 
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
 
    #overridding of BaseModel
    def val_step(self, data: Union[tuple, dict, list]) -> list:
        """Gets the predictions of given data.

        Calls ``self.data_preprocessor(data, False)`` and
        ``self(inputs, data_sample, mode='predict')`` in order. Return the
        predictions which will be passed to evaluator.

        Args:
            data (dict or tuple or list): Data sampled from dataset.

        Returns:
            list: The predictions of given data.
        """      
        if self.test_cfg.mode == 'slide':
            data = self.data_preprocessor(data, 
                    training=False,
                    stack_data=False)
        #여기서의 데이터는 tensor로 배치를 쌓지 않음 
        else: 
            data = self.data_preprocessor.cast_data(data)
            data = self.data_preprocessor(data, training=False, stack_data=True) 
        return self._run_forward(data, mode='predict')  # type: ignore

    #overridding of BaseModel
    def test_step(self, data: Union[dict, tuple, list]) -> list:
        """``BaseModel`` implements ``test_step`` the same as ``val_step``.

        Args:
            data (dict or tuple or list): Data sampled from dataset.

        Returns:
            list: The predictions of given data.
        """
        if self.test_cfg.mode == 'slide':
            data = self.data_preprocessor(data, 
                    training=False,
                    stack_data=False)
        #여기서의 데이터는 tensor로 배치를 쌓지 않음 
        else:
            data = self.data_preprocessor.cast_data(data)
            data = self.data_preprocessor(data, training=False, stack_data=True) 
        return self._run_forward(data, mode='predict')  # type: ignore
