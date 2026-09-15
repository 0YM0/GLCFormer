# Copyright (c) OpenMMLab. All rights reserved.
"""Warm-start Swin GeoLink-style graph fusion.

This variant keeps the v2 spatial-bias two-way fusion block, but applies its
image-feature residual with a small learnable scale. The goal is to preserve the
pretrained Swin/UPerNet baseline behavior at the start of supervised training
and let the graph branch earn influence gradually.
"""

from typing import List, Sequence, Union

import torch
import torch.nn as nn
from mmengine.registry import MODELS
from torch import Tensor

from .swin_geolink_graph_fusion_v2 import SwinGeoLinkSpatialBiasLandCoverGraphFusion


@MODELS.register_module()
class SwinGeoLinkWarmStartLandCoverGraphFusion(SwinGeoLinkSpatialBiasLandCoverGraphFusion):
    """Spatial-bias GeoLink fusion with a warm-start graph residual scale.

    v2 directly adds the randomly initialized graph-fusion residual to the
    pretrained Swin feature map. This class keeps the same graph encoder and
    two-way attention, but changes the final residual to:

        fused = rgb_feat + sigmoid(graph_residual_logit) * graph_delta

    A small non-zero initial scale keeps gradients flowing through the graph
    branch while preventing early random graph features from dominating.
    """

    def __init__(
        self,
        *args,
        residual_scale_init: float = 0.05,
        residual_scale_trainable: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        scale = min(max(float(residual_scale_init), 1e-4), 1.0 - 1e-4)
        logit = torch.logit(torch.tensor(scale, dtype=torch.float32))
        if residual_scale_trainable:
            self.graph_residual_logit = nn.Parameter(logit)
        else:
            self.register_buffer('graph_residual_logit', logit)

    def _graph_residual_scale(self, dtype: torch.dtype, device: torch.device) -> Tensor:
        return torch.sigmoid(self.graph_residual_logit).to(device=device, dtype=dtype)

    def _forward_transformer(
        self,
        feats: Union[Sequence[Tensor], Tensor],
        data_samples=None,
    ) -> Union[List[Tensor], Tensor]:
        if data_samples is None:
            return feats

        is_tensor = torch.is_tensor(feats)
        feat_list = [feats] if is_tensor else list(feats)
        feat = feat_list[self.image_feature_index]
        batch_size, channels, height, width = feat.shape
        device = feat.device

        graph_key, graph_value, key_padding_mask, valid_graphs = self._encode_graph_batch(data_samples, device)
        if not valid_graphs.any():
            self.last_attn = None
            self.last_aux_losses = None
            return feats

        image_flat = feat.flatten(2).transpose(1, 2)
        image_content = self.image_proj(image_flat)
        image_pos = self._image_pos(batch_size, height, width, device, image_content.dtype)
        image_xy = self._image_xy(batch_size, height, width, device, image_content.dtype)
        graph_centroid, graph_bbox, graph_points = self._graph_spatial_batch(
            data_samples,
            device=device,
            max_nodes=graph_key.shape[1],
            dtype=image_content.dtype,
        )
        spatial_bias = self._make_spatial_bias(image_xy, graph_centroid, graph_bbox, graph_points)
        spatial_bias = spatial_bias.masked_fill(key_padding_mask[:, None, :], 0.0)
        spatial_bias = torch.where(valid_graphs.view(batch_size, 1, 1), spatial_bias, torch.zeros_like(spatial_bias))
        class_ids = self._graph_class_batch(data_samples, device, graph_key.shape[1])

        assert image_flat.shape == (batch_size, height * width, channels)
        assert image_content.shape == (batch_size, height * width, self.embed_dims)
        assert image_pos.shape == image_content.shape
        assert graph_key.shape == graph_value.shape
        assert graph_key.shape[:2] == key_padding_mask.shape
        assert spatial_bias.shape == (batch_size, height * width, graph_key.shape[1])

        x = image_content
        graph_token = graph_value
        graph_pos = graph_key - graph_value
        last_attn = None
        for block in self.fusion_blocks:
            x, graph_token, last_attn = block(
                image_token=x,
                image_pos=image_pos,
                graph_key=graph_token + graph_pos,
                graph_value=graph_token,
                key_padding_mask=key_padding_mask,
                spatial_bias=spatial_bias,
            )

        if self.return_attention and last_attn is not None:
            self.last_attn = last_attn.detach().cpu()
        else:
            self.last_attn = None

        if self.training and self.enable_landcover_aux_loss and self.landcover_aux_loss_weight > 0:
            self.last_aux_losses = self._compute_landcover_aux_loss(
                image_token=x,
                spatial_bias=spatial_bias,
                class_ids=class_ids,
                key_padding_mask=key_padding_mask,
                valid_graphs=valid_graphs,
            )
        else:
            self.last_aux_losses = None

        residual_scale = self._graph_residual_scale(dtype=feat.dtype, device=device)
        if self.fusion_residual_mode == 'delta':
            delta = x - image_content
            delta = torch.where(valid_graphs.view(batch_size, 1, 1), delta, torch.zeros_like(delta))
            delta_map = self.context_proj(delta).transpose(1, 2).reshape(batch_size, channels, height, width)
            delta_map = torch.where(valid_graphs.view(batch_size, 1, 1, 1), delta_map, torch.zeros_like(delta_map))
            fused = feat + residual_scale * delta_map
        else:
            out = self.context_proj(x).transpose(1, 2).reshape(batch_size, channels, height, width)
            blended = feat + residual_scale * (out - feat)
            fused = torch.where(valid_graphs.view(batch_size, 1, 1, 1), blended, feat)

        assert fused.shape == feat.shape
        feat_list[self.image_feature_index] = fused
        return self._restore_feats(feats, feat_list, is_tensor)
