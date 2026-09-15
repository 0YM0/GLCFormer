# Copyright (c) OpenMMLab. All rights reserved.
"""Swin GeoLink-style graph fusion v2 with explicit spatial attention bias."""

from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.registry import MODELS
from torch import Tensor

from .swin_geolink_graph_fusion import (
    SinCosCoordinateProjection,
    SwinGeoLinkLandCoverGraphFusion,
)


class SpatialBiasCrossAttention(nn.Module):
    """Multi-head cross-attention with an additive spatial logit bias."""

    def __init__(
        self,
        embed_dims: int,
        num_heads: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims={embed_dims} must be divisible by num_heads={num_heads}')
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.head_dims = embed_dims // num_heads
        self.scale = self.head_dims ** -0.5

        self.q_proj = nn.Linear(embed_dims, embed_dims)
        self.k_proj = nn.Linear(embed_dims, embed_dims)
        self.v_proj = nn.Linear(embed_dims, embed_dims)
        self.out_proj = nn.Linear(embed_dims, embed_dims)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attn_bias: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        batch_size, query_len, _ = query.shape
        key_len = key.shape[1]

        q = self.q_proj(query).view(batch_size, query_len, self.num_heads, self.head_dims).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, key_len, self.num_heads, self.head_dims).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, key_len, self.num_heads, self.head_dims).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if attn_bias is not None:
            if attn_bias.dim() == 3:
                scores = scores + attn_bias.unsqueeze(1).to(dtype=scores.dtype)
            elif attn_bias.dim() == 4:
                scores = scores + attn_bias.to(dtype=scores.dtype)
            else:
                raise ValueError(f'attn_bias must be [B,Lq,Lk] or [B,H,Lq,Lk], got {attn_bias.shape}')

        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask[:, None, None, :], torch.finfo(scores.dtype).min)

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(batch_size, query_len, self.embed_dims)
        out = self.out_proj(out)
        if need_weights:
            return out, attn.mean(dim=1)
        return out, None


class SpatialBiasGeoLinkTwoWayAttentionBlock(nn.Module):
    """Two-way fusion block with patch-polygon spatial bias."""

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
        self.graph_to_image_attn = SpatialBiasCrossAttention(embed_dims, num_heads, dropout)
        self.image_to_graph_attn = SpatialBiasCrossAttention(embed_dims, num_heads, dropout)

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
        spatial_bias: Tensor,
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
            attn_bias=spatial_bias.transpose(1, 2),
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
            attn_bias=spatial_bias,
            key_padding_mask=key_padding_mask,
            need_weights=self.return_attention,
        )
        image_token = image_token + self.dropout(image_update)
        image_token = image_token + self.image_ffn(self.norm_image_ffn(image_token))
        return image_token, graph_token, image_to_graph_attn if self.return_attention else None


@MODELS.register_module()
class SwinGeoLinkSpatialBiasLandCoverGraphFusion(SwinGeoLinkLandCoverGraphFusion):
    """GeoLink-style Swin graph fusion with explicit patch-polygon spatial bias.

    v1 uses image/polygon positional embeddings and lets attention infer spatial
    relevance. v2 additionally adds a distance/containment bias directly to the
    image-token to graph-node attention logits.
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
        spatial_distance_scale: float = 4.0,
        spatial_bbox_scale: float = 2.0,
        spatial_inside_bias: float = 1.0,
        learnable_spatial_bias: bool = True,
        enable_landcover_aux_loss: bool = True,
        landcover_aux_loss_weight: float = 0.05,
        landcover_aux_min_spatial_score: float = -2.0,
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
            pos_num_frequencies=pos_num_frequencies,
        )
        self.polygon_pos_mlp = SinCosCoordinateProjection(
            coord_dims=polygon_pos_dim,
            embed_dims=embed_dims,
            num_frequencies=pos_num_frequencies,
            dropout=dropout,
        )
        self.fusion_blocks = nn.ModuleList([
            SpatialBiasGeoLinkTwoWayAttentionBlock(
                embed_dims=embed_dims,
                num_heads=num_heads,
                dropout=dropout,
                ffn_ratio=ffn_ratio,
                return_attention=return_attention,
            )
            for _ in range(num_fusion_layers)
        ])

        self.learnable_spatial_bias = learnable_spatial_bias
        if learnable_spatial_bias:
            self.spatial_distance_scale = nn.Parameter(torch.tensor(float(spatial_distance_scale)))
            self.spatial_bbox_scale = nn.Parameter(torch.tensor(float(spatial_bbox_scale)))
            self.spatial_inside_bias = nn.Parameter(torch.tensor(float(spatial_inside_bias)))
        else:
            self.register_buffer('spatial_distance_scale', torch.tensor(float(spatial_distance_scale)))
            self.register_buffer('spatial_bbox_scale', torch.tensor(float(spatial_bbox_scale)))
            self.register_buffer('spatial_inside_bias', torch.tensor(float(spatial_inside_bias)))

        self.enable_landcover_aux_loss = enable_landcover_aux_loss
        self.landcover_aux_loss_weight = float(landcover_aux_loss_weight)
        self.landcover_aux_min_spatial_score = float(landcover_aux_min_spatial_score)
        self.landcover_aux_head = nn.Sequential(
            nn.LayerNorm(embed_dims),
            nn.Linear(embed_dims, self.class_embedding.num_embeddings),
        )
        self.last_aux_losses = None

    @staticmethod
    def _coords_from_polygon_pos(polygon_pos: Tensor) -> Tuple[Tensor, Tensor]:
        points = polygon_pos.view(polygon_pos.shape[0], -1, 2).clamp(0.0, 1.0)
        centroid = points[:, -1, :]
        bbox_min = points.min(dim=1).values
        bbox_max = points.max(dim=1).values
        bbox = torch.cat([bbox_min, bbox_max], dim=-1)
        return centroid, bbox

    def _graph_spatial_batch(
        self,
        data_samples,
        device: torch.device,
        max_nodes: int,
        dtype: torch.dtype,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        batch_size = len(data_samples)
        centroids = torch.zeros((batch_size, max_nodes, 2), device=device, dtype=dtype)
        bboxes = torch.zeros((batch_size, max_nodes, 4), device=device, dtype=dtype)
        point_count = max(1, self.polygon_pos_dim // 2)
        polygon_points = torch.zeros((batch_size, max_nodes, point_count, 2), device=device, dtype=dtype)

        for batch_idx, data_sample in enumerate(data_samples):
            graph = self._sample_graph(data_sample)
            if graph is None or bool(graph.get('is_empty_graph', False)):
                continue

            class_id_value = graph.get('class_id')
            if class_id_value is None:
                continue
            num_nodes = int(graph.get('num_nodes', 0) or len(class_id_value))
            if num_nodes <= 0:
                continue
            raw_num_nodes = num_nodes
            visible_idx = None
            visible_mask = self._tensor_from_graph(graph, 'node_visible_mask', device, torch.bool)
            if visible_mask is not None:
                visible_mask = visible_mask.view(-1)
                if visible_mask.numel() < raw_num_nodes:
                    pad = torch.ones((raw_num_nodes - visible_mask.numel(),), dtype=torch.bool, device=device)
                    visible_mask = torch.cat([visible_mask, pad], dim=0)
                visible_idx = torch.nonzero(visible_mask[:raw_num_nodes], as_tuple=False).view(-1)
                if visible_idx.numel() <= 0:
                    continue

            centroid = self._tensor_from_graph(graph, 'centroid', device, torch.float32)
            bbox = self._tensor_from_graph(graph, 'bbox', device, torch.float32)
            polygon_pos = self._tensor_from_graph(graph, 'polygon_pos', device, torch.float32)
            if polygon_pos is not None:
                polygon_pos = self._fit_feature_dim(polygon_pos.view(raw_num_nodes, -1), self.polygon_pos_dim)
            if centroid is None or bbox is None:
                if polygon_pos is None:
                    continue
                centroid, bbox = self._coords_from_polygon_pos(polygon_pos)
            else:
                centroid = centroid.view(raw_num_nodes, -1)[:, :2]
                bbox = bbox.view(raw_num_nodes, -1)[:, :4]
            if visible_idx is not None:
                centroid = centroid[visible_idx]
                bbox = bbox[visible_idx]
                if polygon_pos is not None:
                    polygon_pos = polygon_pos[visible_idx]
                num_nodes = int(visible_idx.numel())

            node_count = min(num_nodes, max_nodes, centroid.shape[0], bbox.shape[0])
            centroids[batch_idx, :node_count] = centroid[:node_count].to(dtype=dtype).clamp(0.0, 1.0)
            bboxes[batch_idx, :node_count] = bbox[:node_count].to(dtype=dtype).clamp(0.0, 1.0)
            if polygon_pos is not None:
                points = polygon_pos.view(num_nodes, point_count, 2)
                polygon_points[batch_idx, :node_count] = points[:node_count].to(dtype=dtype).clamp(0.0, 1.0)
            else:
                polygon_points[batch_idx, :node_count] = centroids[batch_idx, :node_count, None, :].expand(
                    -1, point_count, -1)

        return centroids, bboxes, polygon_points

    def _graph_class_batch(self, data_samples, device: torch.device, max_nodes: int) -> Tensor:
        batch_size = len(data_samples)
        class_ids = torch.zeros((batch_size, max_nodes), device=device, dtype=torch.long)
        for batch_idx, data_sample in enumerate(data_samples):
            graph = self._sample_graph(data_sample)
            if graph is None or bool(graph.get('is_empty_graph', False)):
                continue
            class_id = self._tensor_from_graph(graph, 'class_id', device, torch.long)
            if class_id is None:
                continue
            class_id = class_id.view(-1)
            num_nodes = int(graph.get('num_nodes', 0) or class_id.numel())
            if class_id.numel() < num_nodes:
                pad = torch.zeros((num_nodes - class_id.numel(),), dtype=torch.long, device=device)
                class_id = torch.cat([class_id, pad], dim=0)
            visible_mask = self._tensor_from_graph(graph, 'node_visible_mask', device, torch.bool)
            if visible_mask is not None:
                visible_mask = visible_mask.view(-1)
                if visible_mask.numel() < num_nodes:
                    pad = torch.ones((num_nodes - visible_mask.numel(),), dtype=torch.bool, device=device)
                    visible_mask = torch.cat([visible_mask, pad], dim=0)
                visible_idx = torch.nonzero(visible_mask[:num_nodes], as_tuple=False).view(-1)
                if visible_idx.numel() <= 0:
                    continue
                class_id = class_id[:num_nodes][visible_idx]
            node_count = min(class_id.numel(), max_nodes)
            class_ids[batch_idx, :node_count] = class_id[:node_count].clamp(
                0, self.class_embedding.num_embeddings - 1)
        return class_ids

    @staticmethod
    def _image_xy(
        batch_size: int,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        ys = torch.linspace(0.0, 1.0, height, device=device, dtype=dtype)
        xs = torch.linspace(0.0, 1.0, width, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')
        coords = torch.stack([xx, yy], dim=-1).view(1, height * width, 2)
        return coords.expand(batch_size, -1, -1)

    def _make_spatial_bias(
        self,
        image_xy: Tensor,
        graph_centroid: Tensor,
        graph_bbox: Tensor,
        graph_points: Tensor,
    ) -> Tensor:
        image = image_xy[:, :, None, :]
        centroid = graph_centroid[:, None, :, :]

        x = image[..., 0]
        y = image[..., 1]
        x0 = graph_bbox[:, None, :, 0]
        y0 = graph_bbox[:, None, :, 1]
        x1 = graph_bbox[:, None, :, 2]
        y1 = graph_bbox[:, None, :, 3]

        inside = (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
        dx = torch.maximum(torch.maximum(x0 - x, x - x1), torch.zeros_like(x))
        dy = torch.maximum(torch.maximum(y0 - y, y - y1), torch.zeros_like(y))
        bbox_width = (x1 - x0).clamp_min(1e-3)
        bbox_height = (y1 - y0).clamp_min(1e-3)
        bbox_diag = torch.sqrt(bbox_width * bbox_width + bbox_height * bbox_height).clamp_min(1e-3)
        bbox_distance = torch.sqrt(dx * dx + dy * dy) / bbox_diag

        centroid_distance = torch.linalg.norm(image - centroid, dim=-1) / bbox_diag
        point_distance = torch.linalg.norm(
            image[:, :, :, None, :] - graph_points[:, None, :, :, :],
            dim=-1,
        ).min(dim=-1).values / bbox_diag
        spatial_distance = torch.minimum(centroid_distance, point_distance)

        if self.learnable_spatial_bias:
            distance_scale = F.softplus(self.spatial_distance_scale)
            bbox_scale = F.softplus(self.spatial_bbox_scale)
            inside_scale = F.softplus(self.spatial_inside_bias)
        else:
            distance_scale = self.spatial_distance_scale
            bbox_scale = self.spatial_bbox_scale
            inside_scale = self.spatial_inside_bias
        return inside_scale * inside.to(dtype=image_xy.dtype) - distance_scale * spatial_distance - bbox_scale * bbox_distance

    def _compute_landcover_aux_loss(
        self,
        image_token: Tensor,
        spatial_bias: Tensor,
        class_ids: Tensor,
        key_padding_mask: Tensor,
        valid_graphs: Tensor,
    ) -> dict:
        logits = self.landcover_aux_head(image_token)
        masked_bias = spatial_bias.masked_fill(key_padding_mask[:, None, :], torch.finfo(spatial_bias.dtype).min)
        best_score, best_idx = masked_bias.max(dim=-1)
        target = class_ids.gather(1, best_idx.clamp(0, class_ids.shape[1] - 1))
        valid = (
            valid_graphs[:, None]
            & torch.isfinite(best_score)
            & (best_score > self.landcover_aux_min_spatial_score)
            & (target > 0)
        )
        target = target.masked_fill(~valid, -100)
        if not bool(valid.any()):
            loss = logits.sum() * 0.0
        else:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                target.reshape(-1),
                ignore_index=-100,
            )

        with torch.no_grad():
            pred = logits.argmax(dim=-1)
            acc = (pred[valid] == target[valid]).float().mean() if bool(valid.any()) else logits.new_tensor(0.0)

        return {
            'loss_landcover_aux': loss * self.landcover_aux_loss_weight,
            'landcover_aux_acc': acc,
        }

    def get_aux_losses(self):
        return self.last_aux_losses

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
