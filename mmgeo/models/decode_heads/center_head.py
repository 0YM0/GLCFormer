import torch
import torch.nn as nn 
from torch import Tensor
from mmcv.cnn import ConvModule
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmengine.registry import MODELS
from mmseg.utils import ConfigType, SampleList
from mmseg.models.utils import resize

@MODELS.register_module()
class CenterHeatmapHead(BaseDecodeHead):
    """Center Heatmap Regression Head based on SegformerHead structure.

    This head is used for predicting center heatmaps for object detection/segmentation tasks,
    e.g., building center prediction.

    Args:
        interpolate_mode (str): Interpolation mode for upsampling. Default: 'bilinear'.
    """

    def __init__(self, interpolate_mode='bilinear', num_classes=1, **kwargs):
        super().__init__(input_transform='multiple_select', num_classes=num_classes,  **kwargs)

        self.interpolate_mode = interpolate_mode
        num_inputs = len(self.in_channels)

        assert num_inputs == len(self.in_index)

        self.convs = nn.ModuleList()
        for i in range(num_inputs):
            self.convs.append(
                ConvModule(
                    in_channels=self.in_channels[i],
                    out_channels=self.channels,
                    kernel_size=1,
                    stride=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))

        self.fusion_conv = ConvModule(
            in_channels=self.channels * num_inputs,
            out_channels=self.channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg)

        # NOTE: heatmap은 보통 채널 수 1
        self.heatmap_pred = nn.Conv2d(
            in_channels=self.channels,
            out_channels=1,
            kernel_size=1
        )

    def forward(self, inputs):
        inputs = self._transform_inputs(inputs)
        outs = []
        for idx in range(len(inputs)):
            x = inputs[idx]
            conv = self.convs[idx]
            outs.append(
                resize(
                    input=conv(x),
                    size=inputs[0].shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners))

        out = self.fusion_conv(torch.cat(outs, dim=1))

        heatmap = self.heatmap_pred(out)

        return heatmap


    def _stack_batch_gt(self, batch_data_samples: SampleList) -> Tensor:
        gt_heat_maps = [
            data_sample.gt_heat_map.data for data_sample in batch_data_samples
        ]
        return torch.stack(gt_heat_maps, dim=0)
