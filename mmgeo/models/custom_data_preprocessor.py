# Copyright (c) OpenMMLab. All rights reserved.
from numbers import Number
from typing import Any, Dict, List, Optional, Sequence

import torch
from mmengine.model import BaseDataPreprocessor

from mmengine.registry import MODELS
from mmseg.utils import SampleList #  stack_batch 

# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
 
 
@MODELS.register_module()
class CustomSegDataPreProcessor(BaseDataPreprocessor):
    """
        무조건 디바이스가 CPU로 올라가도록 forward 됨 
    Image pre-processor for segmentation tasks.

    Comparing with the :class:`mmengine.ImgDataPreprocessor`,

    1. It won't do normalization if ``mean`` is not specified.
    2. It does normalization and color space conversion after stacking batch.
    3. It supports batch augmentations like mixup and cutmix.


    It provides the data pre-processing as follows

    - Collate and move data to the target device.
    - Pad inputs to the input size with defined ``pad_val``, and pad seg map
        with defined ``seg_pad_val``.
    - Stack inputs to batch_inputs.
    - Convert inputs from bgr to rgb if the shape of input is (3, H, W).
    - Normalize image with defined std and mean.
    - Do batch augmentations like Mixup and Cutmix during training.

    Args:
        mean (Sequence[Number], optional): The pixel mean of R, G, B channels.
            Defaults to None.
        std (Sequence[Number], optional): The pixel standard deviation of
            R, G, B channels. Defaults to None.
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_val (float, optional): Padding value. Default: 0.
        seg_pad_val (float, optional): Padding value of segmentation map.
            Default: 255.
        padding_mode (str): Type of padding. Default: constant.
            - constant: pads with a constant value, this value is specified
              with pad_val.
        bgr_to_rgb (bool): whether to convert image from BGR to RGB.
            Defaults to False.
        rgb_to_bgr (bool): whether to convert image from RGB to RGB.
            Defaults to False.
        batch_augments (list[dict], optional): Batch-level augmentations
        test_cfg (dict, optional): The padding size config in testing, if not
            specify, will use `size` and `size_divisor` params as default.
            Defaults to None, only supports keys `size` or `size_divisor`.
    """

    def __init__(
        self,
        mean: Sequence[Number] = None,
        std: Sequence[Number] = None,
        size: Optional[tuple] = None,
        size_divisor: Optional[int] = None,
        pad_val: Number = 0,
        seg_pad_val: Number = 255,
        bgr_to_rgb: bool = False,
        rgb_to_bgr: bool = False,
        batch_augments: Optional[List[dict]] = None,
        test_cfg: dict = None, 
    ):
        super().__init__()
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        self.seg_pad_val = seg_pad_val

        assert not (bgr_to_rgb and rgb_to_bgr), (
            '`bgr2rgb` and `rgb2bgr` cannot be set to True at the same time')
        self.channel_conversion = rgb_to_bgr or bgr_to_rgb

        if mean is not None:
            assert std is not None, 'To enable the normalization in ' \
                                    'preprocessing, please specify both ' \
                                    '`mean` and `std`.'
            # Enable the normalization in preprocessing.
            self._enable_normalize = True
            self.mean = torch.tensor(mean).view(-1,1,1)
            self.std = torch.tensor(std).view(-1,1,1)  
        else:
            self._enable_normalize = False

        # TODO: support batch augmentations.
        self.batch_augments = batch_augments

        # Support different padding methods in testing
        self.test_cfg = test_cfg 
    def _stack_batch(self, inputs: List[torch.Tensor],
                    data_samples: Optional[SampleList] = None,
                    size: Optional[tuple] = None,
                    size_divisor: Optional[int] = None,
                    pad_val: Union[int, float] = 0,
                    seg_pad_val: Union[int, float] = 255) -> torch.Tensor:
        """Stack multiple inputs to form a batch and pad the images and gt_sem_segs
        to the max shape use the right bottom padding mode.

        Args:
            inputs (List[Tensor]): The input multiple tensors. each is a
                CHW 3D-tensor.
            data_samples (list[:obj:`SegDataSample`]): The list of data samples.
                It usually includes information such as `gt_sem_seg`.
            size (tuple, optional): Fixed padding size.
            size_divisor (int, optional): The divisor of padded size.
            pad_val (int, float): The padding value. Defaults to 0
            seg_pad_val (int, float): The padding value. Defaults to 255

        Returns:
        Tensor: The 4D-tensor.
        List[:obj:`SegDataSample`]: After the padding of the gt_seg_map.
        """
        assert isinstance(inputs, list), \
            f'Expected input type to be list, but got {type(inputs)}'
        assert len({tensor.ndim for tensor in inputs}) == 1, \
            f'Expected the dimensions of all inputs must be the same, ' \
            f'but got {[tensor.ndim for tensor in inputs]}'
        assert inputs[0].ndim == 3, f'Expected tensor dimension to be 3, ' \
            f'but got {inputs[0].ndim}'
        assert len({tensor.shape[0] for tensor in inputs}) == 1, \
            f'Expected the channels of all inputs must be the same, ' \
            f'but got {[tensor.shape[0] for tensor in inputs]}'

        # only one of size and size_divisor should be valid
        assert (size is not None) ^ (size_divisor is not None), \
            'only one of size and size_divisor should be valid'

        padded_inputs = []
        padded_samples = []
        inputs_sizes = [(img.shape[-2], img.shape[-1]) for img in inputs]
        max_size = np.stack(inputs_sizes).max(0)
        if size_divisor is not None and size_divisor > 1:
            # the last two dims are H,W, both subject to divisibility requirement
            max_size = (max_size +
                        (size_divisor - 1)) // size_divisor * size_divisor

        for i in range(len(inputs)):
            tensor = inputs[i]
            if size is not None:
                width = max(size[-1] - tensor.shape[-1], 0)
                height = max(size[-2] - tensor.shape[-2], 0)
                # (padding_left, padding_right, padding_top, padding_bottom)
                padding_size = (0, width, 0, height)
            elif size_divisor is not None:
                width = max(max_size[-1] - tensor.shape[-1], 0)
                height = max(max_size[-2] - tensor.shape[-2], 0)
                padding_size = (0, width, 0, height)
            else:
                padding_size = [0, 0, 0, 0]

            # pad img
            pad_img = F.pad(tensor, padding_size, value=pad_val)
            padded_inputs.append(pad_img)
            # pad gt_sem_seg
            if data_samples is not None:
                data_sample = data_samples[i]
                pad_shape = None
                if 'gt_sem_seg' in data_sample:
                    gt_sem_seg = data_sample.gt_sem_seg.data
                    del data_sample.gt_sem_seg.data
                    data_sample.gt_sem_seg.data = F.pad(
                        gt_sem_seg, padding_size, value=seg_pad_val)
                    pad_shape = data_sample.gt_sem_seg.shape
                if 'gt_edge_map' in data_sample:
                    gt_edge_map = data_sample.gt_edge_map.data
                    del data_sample.gt_edge_map.data
                    data_sample.gt_edge_map.data = F.pad(
                        gt_edge_map, padding_size, value=seg_pad_val)
                    pad_shape = data_sample.gt_edge_map.shape
                if 'gt_depth_map' in data_sample:
                    gt_depth_map = data_sample.gt_depth_map.data
                    del data_sample.gt_depth_map.data
                    data_sample.gt_depth_map.data = F.pad(
                        gt_depth_map, padding_size, value=seg_pad_val)
                    pad_shape = data_sample.gt_depth_map.shape
                if 'gt_heat_map' in data_sample:
                    gt_heat_map = data_sample.gt_heat_map.data 
                    del data_sample.gt_heat_map.data
                    # 3차원인지 확인 후 pad 적용
                    if gt_heat_map.dim() == 2:  # [H, W]
                        gt_heat_map = F.pad(gt_heat_map, padding_size, value=seg_pad_val)
                        start = 0
                    elif gt_heat_map.dim() == 3:  # [1, H, W]
                        gt_heat_map = F.pad(gt_heat_map, padding_size, value=seg_pad_val)
                        start = 1
                    else:
                        raise ValueError(f"Unexpected shape for gt_heat_map: {gt_heat_map.shape}")
                    data_sample.gt_heat_map.data = gt_heat_map
                    pad_shape = data_sample.gt_heat_map.shape[start:]

                data_sample.set_metainfo({
                    'img_shape': tensor.shape[-2:],
                    'pad_shape': pad_shape,
                    'padding_size': padding_size
                })
                padded_samples.append(data_sample)
            else:
                padded_samples.append(
                    dict(
                        img_padding_size=padding_size,
                        pad_shape=pad_img.shape[-2:]))

        return torch.stack(padded_inputs, dim=0), padded_samples


    def forward(self, data: dict, 
        training: bool = False,
        stack_data: bool = True, ) -> Dict[str, Any]:
        """Perform normalization、padding and bgr2rgb conversion based on
        ``BaseDataPreprocessor``.

        Args:
            data (dict): data sampled from dataloader.
            training (bool): Whether to enable training time augmentation.

        Returns:
            Dict: Data in the same format as the model input.
        """  
        """
        cast_data 부분을 수정, 
        모델의 slide_inference 함수에서 crop_img를 GPU에 올릴 수 있도록 수정함 

        datapreprocessor에서는 device를 따로 올리지 않고 
        cpu 텐서로 올림 
        """
        if self.test_cfg is None: 
            data = self.cast_data(data)  # type: ignore  
        inputs = data['inputs']  
        data_samples = data.get('data_samples', None) 

        # TODO: whether normalize should be after stack_batch
        if self.channel_conversion and inputs[0].size(0) == 3:
            inputs = [_input[[2, 1, 0], ...] for _input in inputs]

        inputs = [_input.float() for _input in inputs] 
 
        
        if self._enable_normalize: #(c,h,w) - (3,1,1) : 채널마다 동작함 
            try:
                inputs = [(_input - self.mean) / self.std for _input in inputs]
            except: 
                inputs = [(_input - self.mean.to(self.device)) / self.std.to(self.device) for _input in inputs]

        if training:
            assert data_samples is not None, ('During training, ',
                                              '`data_samples` must be define.')   
            inputs, data_samples = self._stack_batch(
                inputs=inputs,
                data_samples=data_samples,
                size=self.size,
                size_divisor=self.size_divisor,
                pad_val=self.pad_val,
                seg_pad_val=self.seg_pad_val)

            if self.batch_augments is not None:
                inputs, data_samples = self.batch_augments(
                    inputs, data_samples)
        else:
            #List이고 batch개만큼의 요소가 있음. 
            #각 배치에는 3xhxw  
            """
            기존에 이미지를 바로 올리는 경우에는 당연히 이미지 사이즈가 같아야 하겠지만 
            sling inference의 경우에는 나중에 크기를 맞추어주기 때문에 여기서 배치를 쌓지 않음  
            """ 
            # pad images when testing
            if stack_data:
                img_size = inputs[0].shape[1:]
                assert all(input_.shape[1:] == img_size for input_ in inputs),  \
                    'The image size in a batch should be the same.'
                if self.test_cfg:
                    inputs, padded_samples = self._stack_batch(
                        inputs=inputs,
                        size=self.test_cfg.get('size', None),
                        size_divisor=self.test_cfg.get('size_divisor', None),
                        pad_val=self.pad_val,
                        seg_pad_val=self.seg_pad_val)
                    for data_sample, pad_info in zip(data_samples, padded_samples):
                        data_sample.set_metainfo({**pad_info})
                else:
                    inputs = torch.stack(inputs, dim=0) 
  
        return dict(inputs=inputs, data_samples=data_samples)
