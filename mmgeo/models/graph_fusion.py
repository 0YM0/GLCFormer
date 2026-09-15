# Copyright (c) OpenMMLab. All rights reserved.
"""Land-cover graph encoder and image-to-graph cross-attention fusion."""

from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from mmengine.registry import MODELS
from torch import Tensor


def _mlp(in_dims: int, hidden_dims: int, out_dims: int, dropout: float = 0.0) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dims, hidden_dims),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dims, out_dims),
    )


class _GraphMessageLayer(nn.Module):
    """Lightweight edge-aware message passing without torch_geometric."""

    def __init__(self, embed_dims: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.message_proj = nn.Linear(embed_dims, embed_dims)
        self.norm1 = nn.LayerNorm(embed_dims)
        self.norm2 = nn.LayerNorm(embed_dims)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dims, embed_dims * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dims * 4, embed_dims),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, node_token: Tensor, edge_index: Tensor, edge_token: Tensor) -> Tensor:
        if edge_index.numel() == 0:
            return self.norm2(node_token + self.ffn(self.norm1(node_token)))

        src = edge_index[0].long().clamp(0, node_token.shape[0] - 1)
        dst = edge_index[1].long().clamp(0, node_token.shape[0] - 1)
        messages = self.message_proj(node_token[src] + edge_token)

        agg = node_token.new_zeros(node_token.shape)
        agg.index_add_(0, dst, messages)
        deg = node_token.new_zeros((node_token.shape[0], 1))
        deg.index_add_(0, dst, torch.ones((dst.shape[0], 1), device=node_token.device, dtype=node_token.dtype))
        agg = agg / deg.clamp_min(1.0)

        node_token = self.norm1(node_token + self.dropout(agg))
        node_token = self.norm2(node_token + self.ffn(node_token))
        return node_token


class ImageToGraphTransformerFusionBlock(nn.Module):
    """Transformer-style image-to-graph fusion block.

    x: [B, HW, D] image content token
    image_pos: [B, HW, D] image positional embedding
    graph_key: [B, N, D] graph node token + polygon PE
    graph_value: [B, N, D] graph node content token
    key_padding_mask: [B, N], True means padded node
    """

    def __init__(
        self,
        embed_dims: int,
        num_heads: int,
        dropout: float,
        ffn_ratio: float = 4.0,
        return_attention: bool = False,
    ) -> None:
        super().__init__()
        self.return_attention = return_attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dims,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_q = nn.LayerNorm(embed_dims)
        self.norm_k = nn.LayerNorm(embed_dims)
        self.norm_ffn = nn.LayerNorm(embed_dims)
        self.dropout = nn.Dropout(dropout)
        hidden_dims = int(embed_dims * ffn_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dims, hidden_dims),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims, embed_dims),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: Tensor,
        image_pos: Tensor,
        graph_key: Tensor,
        graph_value: Tensor,
        key_padding_mask: Tensor,
        valid_graphs: Tensor,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        q = self.norm_q(x + image_pos)
        k = self.norm_k(graph_key)
        v = graph_value

        attn_out, attn_weights = self.cross_attn(
            query=q,
            key=k,
            value=v,
            key_padding_mask=key_padding_mask,
            need_weights=self.return_attention,
        )
        x = x + self.dropout(attn_out)
        x = x + self.ffn(self.norm_ffn(x))
        return x, attn_weights if self.return_attention else None


@MODELS.register_module()
class LandCoverGraphImageFusion(nn.Module):
    """Fuse image feature tokens with land-cover polygon object tokens.

    The graph side follows the GeoLink idea: class/level/geometry become the
    semantic token, polygon_pos becomes a positional embedding, and image
    patch tokens query graph object tokens through Q/K/V cross-attention.
    """

    def __init__(
        self,
        image_in_channels: int,
        embed_dims: int = 512,
        image_feature_index: int = -1,
        num_heads: int = 8,
        num_graph_layers: int = 2,
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
        num_fusion_layers: int = 1,
        ffn_ratio: float = 4.0,
        fusion_residual_mode: str = "delta",
        return_attention: bool = False,
        keep_scalar_gate: bool = False,
    ) -> None:
        super().__init__()
        self.image_feature_index = image_feature_index
        self.embed_dims = embed_dims
        self.geom_feat_dim = geom_feat_dim
        self.polygon_pos_dim = polygon_pos_dim
        self.edge_num_feat_dim = edge_num_feat_dim
        self.fusion_residual_mode = fusion_residual_mode
        self.return_attention = return_attention
        self.keep_scalar_gate = keep_scalar_gate
        self.last_attn = None
        if self.fusion_residual_mode not in ("delta", "replace"):
            raise ValueError(
                "fusion_residual_mode must be one of {'delta', 'replace'}, "
                f"but got {fusion_residual_mode}")

        self.class_embedding = nn.Embedding(num_class_ids, embed_dims)
        self.large_embedding = nn.Embedding(num_large_ids, embed_dims)
        self.middle_embedding = nn.Embedding(num_middle_ids, embed_dims)
        self.small_embedding = nn.Embedding(num_small_ids, embed_dims)
        self.level_embedding = nn.Embedding(num_level_ids, embed_dims)
        self.geom_mlp = _mlp(geom_feat_dim, embed_dims, embed_dims, dropout)
        self.polygon_pos_mlp = _mlp(polygon_pos_dim, embed_dims, embed_dims, dropout)

        self.edge_type_embedding = nn.Embedding(num_edge_types, embed_dims)
        self.edge_num_mlp = _mlp(edge_num_feat_dim, embed_dims, embed_dims, dropout)
        self.graph_layers = nn.ModuleList([
            _GraphMessageLayer(embed_dims=embed_dims, dropout=dropout)
            for _ in range(num_graph_layers)
        ])
        self.graph_norm = nn.LayerNorm(embed_dims)

        self.image_proj = nn.Linear(image_in_channels, embed_dims)
        self.image_pos_mlp = _mlp(2, embed_dims, embed_dims, dropout)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dims,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(embed_dims)
        self.context_proj = nn.Linear(embed_dims, image_in_channels)
        self.fusion_blocks = nn.ModuleList([
            ImageToGraphTransformerFusionBlock(
                embed_dims=embed_dims,
                num_heads=num_heads,
                dropout=dropout,
                ffn_ratio=ffn_ratio,
                return_attention=return_attention,
            )
            for _ in range(num_fusion_layers)
        ])
        gate = float(graph_gate_init)
        gate = min(max(gate, 1e-4), 1 - 1e-4)
        self.graph_gate_logit = nn.Parameter(torch.tensor(gate / (1.0 - gate)).log())

    @staticmethod
    def _sample_graph(data_sample) -> Optional[Dict]:
        if data_sample is None:
            return None
        metainfo = getattr(data_sample, 'metainfo', {})
        return metainfo.get('landcover_graph', None)

    @staticmethod
    def _tensor_from_graph(graph: Dict, key: str, device: torch.device, dtype: Optional[torch.dtype] = None) -> Optional[Tensor]:
        value = graph.get(key)
        if value is None:
            return None
        if not torch.is_tensor(value):
            value = torch.as_tensor(value)
        value = value.to(device=device)
        if dtype is not None:
            value = value.to(dtype=dtype)
        return value

    def _ids(self, graph: Dict, key: str, device: torch.device, upper: int, n: int) -> Tensor:
        value = self._tensor_from_graph(graph, key, device, torch.long)
        if value is None:
            return torch.zeros((n,), dtype=torch.long, device=device)
        return value.view(-1)[:n].clamp(0, upper - 1)

    @staticmethod
    def _fit_feature_dim(value: Tensor, expected_dim: int) -> Tensor:
        if value.shape[-1] == expected_dim:
            return value
        if value.shape[-1] > expected_dim:
            return value[..., :expected_dim]
        pad = value.new_zeros(*value.shape[:-1], expected_dim - value.shape[-1])
        return torch.cat([value, pad], dim=-1)

    def _encode_one_graph(self, graph: Optional[Dict], device: torch.device) -> Tuple[Tensor, Tensor, bool]:
        if graph is None or bool(graph.get('is_empty_graph', False)):
            empty = torch.zeros((1, self.embed_dims), device=device)
            return empty, empty, False

        class_id_value = graph.get('class_id')
        if class_id_value is None:
            empty = torch.zeros((1, self.embed_dims), device=device)
            return empty, empty, False
        n = int(graph.get('num_nodes', 0) or len(class_id_value))
        if n <= 0:
            empty = torch.zeros((1, self.embed_dims), device=device)
            return empty, empty, False

        class_id = self._ids(graph, 'class_id', device, self.class_embedding.num_embeddings, n)
        large_id = self._ids(graph, 'large_id', device, self.large_embedding.num_embeddings, n)
        middle_id = self._ids(graph, 'middle_id', device, self.middle_embedding.num_embeddings, n)
        small_id = self._ids(graph, 'small_id', device, self.small_embedding.num_embeddings, n)
        level_id = self._ids(graph, 'level_id', device, self.level_embedding.num_embeddings, n)

        geom_feat = self._tensor_from_graph(graph, 'geom_feat', device, torch.float32)
        if geom_feat is None:
            geom_feat = torch.zeros((n, self.geom_feat_dim), dtype=torch.float32, device=device)
        geom_feat = geom_feat.view(n, -1)
        geom_feat = self._fit_feature_dim(geom_feat, self.geom_feat_dim)

        polygon_pos = self._tensor_from_graph(graph, 'polygon_pos', device, torch.float32)
        if polygon_pos is None:
            polygon_pos = torch.zeros((n, self.polygon_pos_dim), dtype=torch.float32, device=device)
        polygon_pos = polygon_pos.view(n, -1)
        polygon_pos = self._fit_feature_dim(polygon_pos, self.polygon_pos_dim)

        node_token = (
            self.class_embedding(class_id)
            + self.large_embedding(large_id)
            + self.middle_embedding(middle_id)
            + self.small_embedding(small_id)
            + self.level_embedding(level_id)
            + self.geom_mlp(geom_feat)
        )
        node_pos = self.polygon_pos_mlp(polygon_pos)

        edge_index = self._tensor_from_graph(graph, 'edge_index', device, torch.long)
        edge_type = self._tensor_from_graph(graph, 'edge_type', device, torch.long)
        edge_num_feat = self._tensor_from_graph(graph, 'edge_num_feat', device, torch.float32)
        if edge_index is None or edge_index.numel() == 0:
            edge_index = torch.arange(n, dtype=torch.long, device=device).repeat(2, 1)
        if edge_type is None:
            edge_type = torch.zeros((edge_index.shape[1],), dtype=torch.long, device=device)
        if edge_num_feat is None:
            edge_num_feat = torch.zeros((edge_index.shape[1], self.edge_num_feat_dim), dtype=torch.float32, device=device)

        edge_type = edge_type.view(-1)[:edge_index.shape[1]].clamp(0, self.edge_type_embedding.num_embeddings - 1)
        edge_num_feat = edge_num_feat.view(edge_index.shape[1], -1)
        edge_num_feat = self._fit_feature_dim(edge_num_feat, self.edge_num_feat_dim)
        edge_token = self.edge_type_embedding(edge_type) + self.edge_num_mlp(edge_num_feat)

        for layer in self.graph_layers:
            node_token = layer(node_token, edge_index, edge_token)
        node_token = self.graph_norm(node_token)
        key_token = node_token + node_pos
        value_token = node_token
        return key_token, value_token, True

    def _encode_graph_batch(self, data_samples, device: torch.device) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        graph_key_tokens: List[Tensor] = []
        graph_value_tokens: List[Tensor] = []
        valid_graphs: List[bool] = []
        max_nodes = 1
        for data_sample in data_samples:
            key_token, value_token, is_valid = self._encode_one_graph(self._sample_graph(data_sample), device)
            graph_key_tokens.append(key_token)
            graph_value_tokens.append(value_token)
            valid_graphs.append(is_valid)
            max_nodes = max(max_nodes, key_token.shape[0])

        batch = len(graph_key_tokens)
        padded_key = graph_key_tokens[0].new_zeros((batch, max_nodes, self.embed_dims))
        padded_value = graph_value_tokens[0].new_zeros((batch, max_nodes, self.embed_dims))
        key_padding_mask = torch.ones((batch, max_nodes), dtype=torch.bool, device=device)
        valid_tensor = torch.tensor(valid_graphs, dtype=torch.bool, device=device)
        for idx, (key_token, value_token) in enumerate(zip(graph_key_tokens, graph_value_tokens)):
            length = key_token.shape[0]
            padded_key[idx, :length] = key_token
            padded_value[idx, :length] = value_token
            key_padding_mask[idx, :length] = False
            if not valid_graphs[idx]:
                key_padding_mask[idx, 0] = False
        return padded_key, padded_value, key_padding_mask, valid_tensor

    def _image_pos(self, batch_size: int, height: int, width: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        ys = torch.linspace(0.0, 1.0, height, device=device, dtype=dtype)
        xs = torch.linspace(0.0, 1.0, width, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')
        coords = torch.stack([xx, yy], dim=-1).view(1, height * width, 2)
        coords = coords.expand(batch_size, -1, -1)
        return self.image_pos_mlp(coords)

    def _restore_feats(
        self,
        feats: Union[Sequence[Tensor], Tensor],
        feat_list: List[Tensor],
        is_tensor: bool,
    ) -> Union[List[Tensor], Tensor]:
        if is_tensor:
            return feat_list[0]
        if isinstance(feats, tuple):
            return tuple(feat_list)
        return feat_list

    def _forward_scalar_gate(
        self,
        feats: Union[Sequence[Tensor], Tensor],
        data_samples=None,
    ) -> Union[List[Tensor], Tensor]:
        if data_samples is None:
            return feats

        is_tensor = torch.is_tensor(feats)
        feat_list = [feats] if is_tensor else list(feats)
        feat = feat_list[self.image_feature_index]
        batch_size, channels, height, width = feat.shape  # [B, C, H, W]
        device = feat.device

        graph_key_tokens, graph_value_tokens, key_padding_mask, valid_graphs = self._encode_graph_batch(data_samples, device)
        if not valid_graphs.any():
            return feats

        image_flat = feat.flatten(2).transpose(1, 2)  # [B, HW, C]
        image_tokens = self.image_proj(image_flat)  # [B, HW, D]
        image_tokens = image_tokens + self._image_pos(batch_size, height, width, device, image_tokens.dtype)  # [B, HW, D]

        context, _ = self.cross_attn(
            query=image_tokens,
            key=graph_key_tokens,
            value=graph_value_tokens,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        context = self.attn_norm(context)
        context = self.context_proj(context).transpose(1, 2).reshape(batch_size, channels, height, width)
        context = torch.where(valid_graphs.view(batch_size, 1, 1, 1), context, torch.zeros_like(context))

        gate = torch.sigmoid(self.graph_gate_logit)
        feat_list[self.image_feature_index] = feat + gate * context
        self.last_attn = None
        return self._restore_feats(feats, feat_list, is_tensor)

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
        batch_size, channels, height, width = feat.shape  # [B, C, H, W]
        device = feat.device

        graph_key_tokens, graph_value_tokens, key_padding_mask, valid_graphs = self._encode_graph_batch(data_samples, device)
        if not valid_graphs.any():
            self.last_attn = None
            return feats

        assert graph_key_tokens.shape[:2] == key_padding_mask.shape
        assert graph_key_tokens.shape == graph_value_tokens.shape
        image_flat = feat.flatten(2).transpose(1, 2)  # [B, HW, C]
        image_content = self.image_proj(image_flat)  # [B, HW, D]
        image_pos = self._image_pos(batch_size, height, width, device, image_content.dtype)  # [B, HW, D]

        assert image_flat.shape == (batch_size, height * width, channels)
        assert image_content.shape == (batch_size, height * width, self.embed_dims)
        assert image_pos.shape == image_content.shape
        assert graph_key_tokens.shape[-1] == self.embed_dims
        assert graph_value_tokens.shape[-1] == self.embed_dims

        x = image_content
        last_attn = None
        for block in self.fusion_blocks:
            x, last_attn = block(
                x=x,
                image_pos=image_pos,
                graph_key=graph_key_tokens,
                graph_value=graph_value_tokens,
                key_padding_mask=key_padding_mask,
                valid_graphs=valid_graphs,
            )
        assert x.shape == image_content.shape

        if self.return_attention and last_attn is not None:
            self.last_attn = last_attn.detach().cpu()
        else:
            self.last_attn = None

        if self.fusion_residual_mode == "delta":
            delta = x - image_content  # [B, HW, D]
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

    def forward(
        self,
        feats: Union[Sequence[Tensor], Tensor],
        data_samples=None,
    ) -> Union[List[Tensor], Tensor]:
        if self.keep_scalar_gate:
            return self._forward_scalar_gate(feats, data_samples)
        return self._forward_transformer(feats, data_samples)
