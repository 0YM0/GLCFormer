# Copyright (c) OpenMMLab. All rights reserved.
"""GeoLink-style two-way image/graph fusion for Swin final-stage features."""

import math
from typing import Optional, Sequence, Tuple, Union, List

import torch
import torch.nn as nn
from mmengine.registry import MODELS
from torch import Tensor

from .v3_graph_fusion import LandCoverGraphImageFusionV3


class SinCosCoordinateProjection(nn.Module):
    """Project normalized coordinates with sinusoidal Fourier features."""

    def __init__(
        self,
        coord_dims: int,
        embed_dims: int,
        num_frequencies: int = 16,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.coord_dims = coord_dims
        self.num_frequencies = num_frequencies
        fourier_dims = coord_dims * (1 + 2 * num_frequencies)
        self.proj = nn.Sequential(
            nn.Linear(fourier_dims, embed_dims),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dims, embed_dims),
        )

    def forward(self, coords: Tensor) -> Tensor:
        coords = coords.clamp(0.0, 1.0)
        freq = torch.arange(
            self.num_frequencies,
            device=coords.device,
            dtype=coords.dtype,
        )
        freq = (2.0 ** freq) * math.pi
        scaled = coords.unsqueeze(-1) * freq
        sincos = torch.cat([scaled.sin(), scaled.cos()], dim=-1).flatten(-2)
        return self.proj(torch.cat([coords, sincos], dim=-1))


class GeoLinkTwoWayAttentionBlock(nn.Module):
    """Two-way image/graph attention block close to GeoLink fusion.

    The block updates graph object tokens using image patch tokens, then updates
    image patch tokens using the refined graph tokens. The returned image token
    is later projected back to the segmentation feature map.
    """

    def __init__(
        self,
        embed_dims: int,
        num_heads: int,
        dropout: float = 0.1,
        ffn_ratio: float = 4.0,
        return_attention: bool = False,
    ) -> None:
        super().__init__()
        self.return_attention = return_attention
        self.graph_self_attn = nn.MultiheadAttention(
            embed_dim=embed_dims,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.graph_to_image_attn = nn.MultiheadAttention(
            embed_dim=embed_dims,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.image_to_graph_attn = nn.MultiheadAttention(
            embed_dim=embed_dims,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm_graph_self = nn.LayerNorm(embed_dims)
        self.norm_graph_to_image_q = nn.LayerNorm(embed_dims)
        self.norm_image_k = nn.LayerNorm(embed_dims)
        self.norm_image_to_graph_q = nn.LayerNorm(embed_dims)
        self.norm_graph_k = nn.LayerNorm(embed_dims)
        self.norm_graph_ffn = nn.LayerNorm(embed_dims)
        self.norm_image_ffn = nn.LayerNorm(embed_dims)
        self.dropout = nn.Dropout(dropout)

        hidden_dims = int(embed_dims * ffn_ratio)
        self.graph_ffn = nn.Sequential(
            nn.Linear(embed_dims, hidden_dims),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims, embed_dims),
            nn.Dropout(dropout),
        )
        self.image_ffn = nn.Sequential(
            nn.Linear(embed_dims, hidden_dims),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims, embed_dims),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        image_token: Tensor,
        image_pos: Tensor,
        graph_key: Tensor,
        graph_value: Tensor,
        key_padding_mask: Tensor,
        valid_graphs: Tensor,
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        graph_pos = graph_key - graph_value
        graph_token = graph_value

        graph_query = self.norm_graph_self(graph_token + graph_pos)
        graph_update, _ = self.graph_self_attn(
            query=graph_query,
            key=graph_query,
            value=graph_token,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        graph_token = graph_token + self.dropout(graph_update)

        graph_query = self.norm_graph_to_image_q(graph_token + graph_pos)
        image_key = self.norm_image_k(image_token + image_pos)
        graph_update, _ = self.graph_to_image_attn(
            query=graph_query,
            key=image_key,
            value=image_token,
            need_weights=False,
        )
        graph_token = graph_token + self.dropout(graph_update)
        graph_token = graph_token + self.graph_ffn(self.norm_graph_ffn(graph_token))

        graph_key_refined = graph_token + graph_pos
        image_query = self.norm_image_to_graph_q(image_token + image_pos)
        graph_key_norm = self.norm_graph_k(graph_key_refined)
        image_update, image_to_graph_attn = self.image_to_graph_attn(
            query=image_query,
            key=graph_key_norm,
            value=graph_token,
            key_padding_mask=key_padding_mask,
            need_weights=self.return_attention,
        )
        image_token = image_token + self.dropout(image_update)
        image_token = image_token + self.image_ffn(self.norm_image_ffn(image_token))
        return image_token, graph_token, image_to_graph_attn if self.return_attention else None


@MODELS.register_module()
class SwinGeoLinkLandCoverGraphFusion(LandCoverGraphImageFusionV3):
    """GeoLink-style Swin final-stage graph fusion.

    Swin stages 0/1/2 remain RGB-only through UPerNet. The last Swin stage is
    flattened to patch tokens and fused with land-cover graph object tokens via
    a two-way attention block.
    """

    def __init__(
        self,
        image_in_channels: int,
        embed_dims: int = 512,
        image_feature_index: int = -1,
        num_heads: int = 8,
        num_graph_layers: int = 3,
        graph_num_heads: int = 8,
        dropout: float = 0.1,
        graph_gate_init: float = 0.1,
        num_class_ids: int = 32,
        num_large_ids: int = 16,
        num_middle_ids: int = 16,
        num_small_ids: int = 16,
        num_level_ids: int = 4,
        num_edge_types: int = 8,
        geom_feat_dim: int = 9,
        polygon_pos_dim: int = 8,
        edge_num_feat_dim: int = 6,
        graph_ffn_ratio: float = 4.0,
        graph_use_pos: bool = True,
        num_fusion_layers: int = 1,
        ffn_ratio: float = 4.0,
        fusion_residual_mode: str = 'delta',
        return_attention: bool = False,
        keep_scalar_gate: bool = False,
        pos_num_frequencies: int = 16,
    ) -> None:
        super().__init__(
            image_in_channels=image_in_channels,
            embed_dims=embed_dims,
            image_feature_index=image_feature_index,
            num_heads=num_heads,
            num_graph_layers=num_graph_layers,
            graph_num_heads=graph_num_heads,
            dropout=dropout,
            graph_gate_init=graph_gate_init,
            num_class_ids=num_class_ids,
            num_large_ids=num_large_ids,
            num_middle_ids=num_middle_ids,
            num_small_ids=num_small_ids,
            num_level_ids=num_level_ids,
            num_edge_types=num_edge_types,
            geom_feat_dim=geom_feat_dim,
            polygon_pos_dim=polygon_pos_dim,
            edge_num_feat_dim=edge_num_feat_dim,
            graph_ffn_ratio=graph_ffn_ratio,
            graph_use_pos=graph_use_pos,
            num_fusion_layers=num_fusion_layers,
            ffn_ratio=ffn_ratio,
            fusion_residual_mode=fusion_residual_mode,
            return_attention=return_attention,
            keep_scalar_gate=keep_scalar_gate,
        )
        self.polygon_pos_mlp = SinCosCoordinateProjection(
            coord_dims=polygon_pos_dim,
            embed_dims=embed_dims,
            num_frequencies=pos_num_frequencies,
            dropout=dropout,
        )
        self.image_pos_mlp = SinCosCoordinateProjection(
            coord_dims=2,
            embed_dims=embed_dims,
            num_frequencies=pos_num_frequencies,
            dropout=dropout,
        )
        self.fusion_blocks = nn.ModuleList([
            GeoLinkTwoWayAttentionBlock(
                embed_dims=embed_dims,
                num_heads=num_heads,
                dropout=dropout,
                ffn_ratio=ffn_ratio,
                return_attention=return_attention,
            )
            for _ in range(num_fusion_layers)
        ])

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
            return feats

        image_flat = feat.flatten(2).transpose(1, 2)
        image_content = self.image_proj(image_flat)
        image_pos = self._image_pos(batch_size, height, width, device, image_content.dtype)

        assert image_flat.shape == (batch_size, height * width, channels)
        assert image_content.shape == (batch_size, height * width, self.embed_dims)
        assert graph_key.shape == graph_value.shape
        assert graph_key.shape[:2] == key_padding_mask.shape

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
                valid_graphs=valid_graphs,
            )

        if self.return_attention and last_attn is not None:
            self.last_attn = last_attn.detach().cpu()
        else:
            self.last_attn = None

        if self.fusion_residual_mode == 'delta':
            delta = x - image_content
            delta = torch.where(valid_graphs.view(batch_size, 1, 1), delta, torch.zeros_like(delta))
            delta_map = self.context_proj(delta).transpose(1, 2).reshape(batch_size, channels, height, width)
            delta_map = torch.where(valid_graphs.view(batch_size, 1, 1, 1), delta_map, torch.zeros_like(delta_map))
            fused = feat + delta_map
        else:
            out = self.context_proj(x).transpose(1, 2).reshape(batch_size, channels, height, width)
            fused = torch.where(valid_graphs.view(batch_size, 1, 1, 1), out, feat)

        assert fused.shape == feat.shape
        feat_list[self.image_feature_index] = fused
        return self._restore_feats(feats, feat_list, is_tensor)
