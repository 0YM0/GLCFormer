# Copyright (c) OpenMMLab. All rights reserved.
"""v3 land-cover graph fusion with edge-aware graph attention encoder."""

from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from mmengine.registry import MODELS
from torch import Tensor

from .graph_fusion import ImageToGraphTransformerFusionBlock


def _mlp(in_dims: int, hidden_dims: int, out_dims: int, dropout: float = 0.0) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dims, hidden_dims),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dims, out_dims),
    )


class EdgeAwareGraphAttentionLayer(nn.Module):
    """Directed multi-head graph attention with edge-type/numeric conditioning."""

    def __init__(
        self,
        embed_dims: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        ffn_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims={embed_dims} must be divisible by num_heads={num_heads}')

        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.head_dims = embed_dims // num_heads
        self.scale = self.head_dims ** -0.5

        self.norm_attn = nn.LayerNorm(embed_dims)
        self.q_proj = nn.Linear(embed_dims, embed_dims)
        self.k_proj = nn.Linear(embed_dims, embed_dims)
        self.v_proj = nn.Linear(embed_dims, embed_dims)
        self.edge_k_proj = nn.Linear(embed_dims, embed_dims)
        self.edge_v_proj = nn.Linear(embed_dims, embed_dims)
        self.edge_bias_proj = nn.Linear(embed_dims, num_heads)
        self.out_proj = nn.Linear(embed_dims, embed_dims)
        self.dropout = nn.Dropout(dropout)

        self.norm_ffn = nn.LayerNorm(embed_dims)
        hidden_dims = int(embed_dims * ffn_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dims, hidden_dims),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims, embed_dims),
            nn.Dropout(dropout),
        )

    @staticmethod
    def _edge_softmax(scores: Tensor, dst: Tensor, num_nodes: int) -> Tensor:
        """Softmax over incoming edges per destination node and per head."""

        attn = torch.zeros_like(scores)
        unique_dst = torch.unique(dst)
        for node in unique_dst.tolist():
            if node < 0 or node >= num_nodes:
                continue
            mask = dst == node
            if bool(mask.any()):
                attn[mask] = torch.softmax(scores[mask], dim=0)
        return attn

    def forward(self, node_token: Tensor, edge_index: Tensor, edge_token: Tensor) -> Tensor:
        num_nodes = node_token.shape[0]
        if num_nodes == 0:
            return node_token
        if edge_index.numel() == 0:
            edge_index = torch.arange(num_nodes, dtype=torch.long, device=node_token.device).repeat(2, 1)
            edge_token = node_token.new_zeros((num_nodes, self.embed_dims))

        src = edge_index[0].long().clamp(0, num_nodes - 1)
        dst = edge_index[1].long().clamp(0, num_nodes - 1)
        num_edges = edge_index.shape[1]
        edge_token = edge_token[:num_edges]

        h = self.norm_attn(node_token)
        q = self.q_proj(h).view(num_nodes, self.num_heads, self.head_dims)
        k = self.k_proj(h).view(num_nodes, self.num_heads, self.head_dims)
        v = self.v_proj(h).view(num_nodes, self.num_heads, self.head_dims)
        edge_k = self.edge_k_proj(edge_token).view(num_edges, self.num_heads, self.head_dims)
        edge_v = self.edge_v_proj(edge_token).view(num_edges, self.num_heads, self.head_dims)
        edge_bias = self.edge_bias_proj(edge_token)

        scores = (q[dst] * (k[src] + edge_k)).sum(dim=-1) * self.scale
        scores = scores + edge_bias
        attn = self._edge_softmax(scores, dst, num_nodes)
        attn = self.dropout(attn)

        messages = attn.unsqueeze(-1) * (v[src] + edge_v)
        agg = node_token.new_zeros((num_nodes, self.num_heads, self.head_dims))
        agg.index_add_(0, dst, messages)
        agg = agg.reshape(num_nodes, self.embed_dims)

        node_token = node_token + self.dropout(self.out_proj(agg))
        node_token = node_token + self.ffn(self.norm_ffn(node_token))
        return node_token


class EdgeAwareGraphAttentionEncoder(nn.Module):
    """Stacked edge-aware GAT encoder for land-cover polygon graphs."""

    def __init__(
        self,
        embed_dims: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        ffn_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            EdgeAwareGraphAttentionLayer(
                embed_dims=embed_dims,
                num_heads=num_heads,
                dropout=dropout,
                ffn_ratio=ffn_ratio,
            )
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(embed_dims)

    def forward(self, node_token: Tensor, edge_index: Tensor, edge_token: Tensor) -> Tensor:
        for layer in self.layers:
            node_token = layer(node_token, edge_index, edge_token)
        return self.final_norm(node_token)


@MODELS.register_module()
class LandCoverGraphImageFusionV3(nn.Module):
    """Land-cover graph fusion v3.

    v3 upgrades the graph branch from mean aggregation to edge-aware multi-head
    graph attention. The image fusion side keeps the transformer-style
    image-to-graph residual encoder from v2.
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
    ) -> None:
        super().__init__()
        self.image_feature_index = image_feature_index
        self.embed_dims = embed_dims
        self.geom_feat_dim = geom_feat_dim
        self.polygon_pos_dim = polygon_pos_dim
        self.edge_num_feat_dim = edge_num_feat_dim
        self.graph_use_pos = graph_use_pos
        self.fusion_residual_mode = fusion_residual_mode
        self.return_attention = return_attention
        self.keep_scalar_gate = keep_scalar_gate
        self.last_attn = None
        self.last_graph_attn = None
        if self.fusion_residual_mode not in ('delta', 'replace'):
            raise ValueError(
                "fusion_residual_mode must be one of {'delta', 'replace'}, "
                f'but got {fusion_residual_mode}')

        self.class_embedding = nn.Embedding(num_class_ids, embed_dims)
        self.large_embedding = nn.Embedding(num_large_ids, embed_dims)
        self.middle_embedding = nn.Embedding(num_middle_ids, embed_dims)
        self.small_embedding = nn.Embedding(num_small_ids, embed_dims)
        self.level_embedding = nn.Embedding(num_level_ids, embed_dims)
        self.geom_mlp = _mlp(geom_feat_dim, embed_dims, embed_dims, dropout)
        self.polygon_pos_mlp = _mlp(polygon_pos_dim, embed_dims, embed_dims, dropout)

        self.edge_type_embedding = nn.Embedding(num_edge_types, embed_dims)
        self.edge_num_mlp = _mlp(edge_num_feat_dim, embed_dims, embed_dims, dropout)
        self.graph_encoder = EdgeAwareGraphAttentionEncoder(
            embed_dims=embed_dims,
            num_layers=num_graph_layers,
            num_heads=graph_num_heads,
            dropout=dropout,
            ffn_ratio=graph_ffn_ratio,
        )

        self.image_proj = nn.Linear(image_in_channels, embed_dims)
        self.image_pos_mlp = _mlp(2, embed_dims, embed_dims, dropout)
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

        self.scalar_cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dims,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.scalar_attn_norm = nn.LayerNorm(embed_dims)
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
        raw_n = n

        visible_idx = None
        visible_mask = self._tensor_from_graph(graph, 'node_visible_mask', device, torch.bool)
        if visible_mask is not None:
            visible_mask = visible_mask.view(-1)
            if visible_mask.numel() < raw_n:
                pad = torch.ones((raw_n - visible_mask.numel(),), dtype=torch.bool, device=device)
                visible_mask = torch.cat([visible_mask, pad], dim=0)
            visible_mask = visible_mask[:raw_n]
            visible_idx = torch.nonzero(visible_mask, as_tuple=False).view(-1)
            if visible_idx.numel() <= 0:
                empty = torch.zeros((1, self.embed_dims), device=device)
                return empty, empty, False
            n = int(visible_idx.numel())

        def selected_ids(key: str, upper: int) -> Tensor:
            value = self._tensor_from_graph(graph, key, device, torch.long)
            if value is None:
                value = torch.zeros((raw_n,), dtype=torch.long, device=device)
            value = value.view(-1)
            if value.numel() < raw_n:
                pad = torch.zeros((raw_n - value.numel(),), dtype=torch.long, device=device)
                value = torch.cat([value, pad], dim=0)
            value = value[:raw_n]
            if visible_idx is not None:
                value = value[visible_idx]
            return value.clamp(0, upper - 1)

        def selected_feature(key: str, dim: int) -> Tensor:
            value = self._tensor_from_graph(graph, key, device, torch.float32)
            if value is None:
                value = torch.zeros((raw_n, dim), dtype=torch.float32, device=device)
            value = value.view(value.shape[0], -1)
            if value.shape[0] < raw_n:
                pad = value.new_zeros((raw_n - value.shape[0], value.shape[1]))
                value = torch.cat([value, pad], dim=0)
            value = self._fit_feature_dim(value[:raw_n], dim)
            if visible_idx is not None:
                value = value[visible_idx]
            return value

        class_id = selected_ids('class_id', self.class_embedding.num_embeddings)
        large_id = selected_ids('large_id', self.large_embedding.num_embeddings)
        middle_id = selected_ids('middle_id', self.middle_embedding.num_embeddings)
        small_id = selected_ids('small_id', self.small_embedding.num_embeddings)
        level_id = selected_ids('level_id', self.level_embedding.num_embeddings)

        geom_feat = selected_feature('geom_feat', self.geom_feat_dim)
        polygon_pos = selected_feature('polygon_pos', self.polygon_pos_dim)

        node_semantic = (
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

        num_edges = edge_index.shape[1]
        edge_type = edge_type.view(-1)[:num_edges].clamp(0, self.edge_type_embedding.num_embeddings - 1)
        edge_num_feat = self._fit_feature_dim(edge_num_feat.view(num_edges, -1), self.edge_num_feat_dim)
        if visible_idx is not None:
            edge_index = edge_index.view(2, -1)
            src = edge_index[0]
            dst = edge_index[1]
            in_bounds = (src >= 0) & (src < raw_n) & (dst >= 0) & (dst < raw_n)
            src_safe = src.clamp(0, raw_n - 1)
            dst_safe = dst.clamp(0, raw_n - 1)
            keep = in_bounds & visible_mask[src_safe] & visible_mask[dst_safe]
            id_map = torch.full((raw_n,), -1, dtype=torch.long, device=device)
            id_map[visible_idx] = torch.arange(n, dtype=torch.long, device=device)
            edge_index = id_map[edge_index[:, keep]]
            edge_type = edge_type[keep]
            edge_num_feat = edge_num_feat[keep]
        else:
            edge_index = edge_index.view(2, -1)
            keep = (edge_index[0] >= 0) & (edge_index[0] < n) & (edge_index[1] >= 0) & (edge_index[1] < n)
            edge_index = edge_index[:, keep]
            edge_type = edge_type[keep]
            edge_num_feat = edge_num_feat[keep]

        if edge_index.numel() == 0:
            edge_index = torch.arange(n, dtype=torch.long, device=device).repeat(2, 1)
            edge_type = torch.zeros((n,), dtype=torch.long, device=device)
            edge_num_feat = torch.zeros((n, self.edge_num_feat_dim), dtype=torch.float32, device=device)
        edge_token = self.edge_type_embedding(edge_type) + self.edge_num_mlp(edge_num_feat)

        graph_input = node_semantic + node_pos if self.graph_use_pos else node_semantic
        graph_content = self.graph_encoder(graph_input, edge_index, edge_token)
        graph_key = graph_content + node_pos
        graph_value = graph_content
        return graph_key, graph_value, True

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

    @staticmethod
    def _restore_feats(
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
        batch_size, channels, height, width = feat.shape
        device = feat.device

        graph_key, graph_value, key_padding_mask, valid_graphs = self._encode_graph_batch(data_samples, device)
        if not valid_graphs.any():
            return feats

        image_flat = feat.flatten(2).transpose(1, 2)
        image_tokens = self.image_proj(image_flat)
        image_tokens = image_tokens + self._image_pos(batch_size, height, width, device, image_tokens.dtype)
        context, _ = self.scalar_cross_attn(
            query=image_tokens,
            key=graph_key,
            value=graph_value,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        context = self.scalar_attn_norm(context)
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
        assert image_pos.shape == image_content.shape
        assert graph_key.shape == graph_value.shape
        assert graph_key.shape[:2] == key_padding_mask.shape

        x = image_content
        last_attn = None
        for block in self.fusion_blocks:
            x, last_attn = block(
                x=x,
                image_pos=image_pos,
                graph_key=graph_key,
                graph_value=graph_value,
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

    def forward(
        self,
        feats: Union[Sequence[Tensor], Tensor],
        data_samples=None,
    ) -> Union[List[Tensor], Tensor]:
        if self.keep_scalar_gate:
            return self._forward_scalar_gate(feats, data_samples)
        return self._forward_transformer(feats, data_samples)
