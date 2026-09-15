# Copyright (c) OpenMMLab. All rights reserved.
"""Load precomputed land-cover graph files for chip segmentation."""

import os.path as osp
import sys
import types
from typing import Dict, Optional

import torch
from mmcv.transforms.base import BaseTransform
from mmengine.logging import print_log
from mmengine.registry import TRANSFORMS


class _GraphDataShim:
    """Pickle shim for graph Data objects saved outside this repository."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def keys(self):
        return [key for key in self.__dict__.keys() if not key.startswith('_')]

    def __getitem__(self, key):
        return getattr(self, key)


def _install_graph_pickle_shims() -> None:
    """Let torch.load read fallback/PyG Data objects without extra deps."""

    for module_name in ('graph_utils', 'preprocess.graph_utils'):
        module = sys.modules.get(module_name)
        if module is None:
            module = types.ModuleType(module_name)
            sys.modules[module_name] = module
        if not hasattr(module, 'Data'):
            module.Data = _GraphDataShim

    try:
        from torch_geometric.data import Data as _  # noqa: F401
    except Exception:
        tg_module = sys.modules.get('torch_geometric')
        if tg_module is None:
            tg_module = types.ModuleType('torch_geometric')
            sys.modules['torch_geometric'] = tg_module

        data_module = sys.modules.get('torch_geometric.data')
        if data_module is None:
            data_module = types.ModuleType('torch_geometric.data')
            sys.modules['torch_geometric.data'] = data_module
        data_module.Data = _GraphDataShim

        data_data_module = sys.modules.get('torch_geometric.data.data')
        if data_data_module is None:
            data_data_module = types.ModuleType('torch_geometric.data.data')
            sys.modules['torch_geometric.data.data'] = data_data_module
        data_data_module.Data = _GraphDataShim


def _torch_load(path: str):
    _install_graph_pickle_shims()
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(path, map_location='cpu')


def _as_dict(graph) -> Dict:
    if isinstance(graph, dict):
        return graph
    if hasattr(graph, 'to_dict'):
        try:
            return graph.to_dict()
        except Exception:
            pass
    return {
        key: value
        for key, value in vars(graph).items()
        if not key.startswith('_')
    }


def _to_tensor(value, dtype: Optional[torch.dtype] = None) -> Optional[torch.Tensor]:
    if value is None:
        return None
    if torch.is_tensor(value):
        tensor = value.detach().cpu()
    else:
        tensor = torch.as_tensor(value)
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    return tensor


def _empty_graph() -> Dict[str, torch.Tensor]:
    return dict(
        edge_index=torch.zeros((2, 1), dtype=torch.long),
        edge_type=torch.zeros((1,), dtype=torch.long),
        edge_num_feat=torch.tensor([[0., 0., 1., 0., 1., 1.]], dtype=torch.float32),
        class_code=torch.zeros((1,), dtype=torch.long),
        raw_class_code=torch.zeros((1,), dtype=torch.long),
        class_id=torch.zeros((1,), dtype=torch.long),
        large_id=torch.zeros((1,), dtype=torch.long),
        middle_id=torch.zeros((1,), dtype=torch.long),
        small_id=torch.zeros((1,), dtype=torch.long),
        level_id=torch.zeros((1,), dtype=torch.long),
        context_group_id=torch.zeros((1,), dtype=torch.long),
        node_is_prior=torch.zeros((1,), dtype=torch.bool),
        node_train_mask=torch.zeros((1,), dtype=torch.bool),
        node_infer_mask=torch.zeros((1,), dtype=torch.bool),
        node_visible_mask=torch.zeros((1,), dtype=torch.bool),
        prior_target=torch.zeros((1,), dtype=torch.float32),
        prior_loss_mask=torch.zeros((1,), dtype=torch.bool),
        geom_feat=torch.zeros((1, 9), dtype=torch.float32),
        polygon_pos=torch.zeros((1, 8), dtype=torch.float32),
        centroid=torch.zeros((1, 2), dtype=torch.float32),
        bbox=torch.zeros((1, 4), dtype=torch.float32),
        node_context_feat=torch.zeros((1, 0), dtype=torch.float32),
        num_nodes=1,
        is_empty_graph=True,
    )


@TRANSFORMS.register_module()
class LoadLandCoverGraph(BaseTransform):
    """Load a chip-level land-cover graph produced by GeoLink-style preprocessing."""

    def __init__(self, required: bool = False, allow_missing: bool = True) -> None:
        self.required = required
        self.allow_missing = allow_missing

    def _load_graph(self, graph_path: str) -> Dict:
        data = _as_dict(_torch_load(graph_path))
        graph = dict(
            edge_index=_to_tensor(data.get('edge_index'), torch.long),
            edge_type=_to_tensor(data.get('edge_type'), torch.long),
            edge_num_feat=_to_tensor(data.get('edge_num_feat'), torch.float32),
            class_code=_to_tensor(data.get('class_code'), torch.long),
            raw_class_code=_to_tensor(data.get('raw_class_code'), torch.long),
            class_id=_to_tensor(data.get('class_id'), torch.long),
            large_id=_to_tensor(data.get('large_id'), torch.long),
            middle_id=_to_tensor(data.get('middle_id'), torch.long),
            small_id=_to_tensor(data.get('small_id'), torch.long),
            level_id=_to_tensor(data.get('level_id'), torch.long),
            context_group_id=_to_tensor(data.get('context_group_id'), torch.long),
            node_is_prior=_to_tensor(data.get('node_is_prior'), torch.bool),
            node_train_mask=_to_tensor(data.get('node_train_mask'), torch.bool),
            node_infer_mask=_to_tensor(data.get('node_infer_mask'), torch.bool),
            node_visible_mask=_to_tensor(data.get('node_visible_mask'), torch.bool),
            prior_target=_to_tensor(data.get('prior_target'), torch.float32),
            prior_loss_mask=_to_tensor(data.get('prior_loss_mask'), torch.bool),
            geom_feat=_to_tensor(data.get('geom_feat'), torch.float32),
            polygon_pos=_to_tensor(data.get('polygon_pos'), torch.float32),
            centroid=_to_tensor(data.get('centroid'), torch.float32),
            bbox=_to_tensor(data.get('bbox'), torch.float32),
            node_context_feat=_to_tensor(data.get('node_context_feat'), torch.float32),
            num_nodes=int(data.get('num_nodes', 0) or 0),
            is_empty_graph=bool(data.get('is_empty_graph', False)),
        )

        if graph['class_id'] is None:
            return _empty_graph()
        if graph['num_nodes'] <= 0:
            graph['num_nodes'] = int(graph['class_id'].numel())
        if graph['edge_index'] is None:
            graph['edge_index'] = torch.arange(graph['num_nodes'], dtype=torch.long).repeat(2, 1)
        if graph['edge_type'] is None:
            graph['edge_type'] = torch.zeros((graph['edge_index'].shape[1],), dtype=torch.long)
        if graph['edge_num_feat'] is None:
            graph['edge_num_feat'] = torch.zeros((graph['edge_index'].shape[1], 6), dtype=torch.float32)
        if graph['class_code'] is None:
            graph['class_code'] = torch.zeros((graph['num_nodes'],), dtype=torch.long)
        if graph['raw_class_code'] is None:
            graph['raw_class_code'] = graph['class_code'].clone()
        if graph['context_group_id'] is None:
            graph['context_group_id'] = torch.zeros((graph['num_nodes'],), dtype=torch.long)
        if graph['node_is_prior'] is None:
            graph['node_is_prior'] = torch.zeros((graph['num_nodes'],), dtype=torch.bool)
        if graph['node_train_mask'] is None:
            graph['node_train_mask'] = torch.ones((graph['num_nodes'],), dtype=torch.bool)
        if graph['node_infer_mask'] is None:
            graph['node_infer_mask'] = torch.ones((graph['num_nodes'],), dtype=torch.bool)
        if graph['node_visible_mask'] is None:
            graph['node_visible_mask'] = torch.ones((graph['num_nodes'],), dtype=torch.bool)
        if graph['prior_target'] is None:
            graph['prior_target'] = torch.zeros((graph['num_nodes'],), dtype=torch.float32)
        if graph['prior_loss_mask'] is None:
            graph['prior_loss_mask'] = torch.zeros((graph['num_nodes'],), dtype=torch.bool)
        if graph['node_context_feat'] is None:
            graph['node_context_feat'] = torch.zeros((graph['num_nodes'], 0), dtype=torch.float32)
        if graph['geom_feat'] is None:
            graph['geom_feat'] = torch.zeros((graph['num_nodes'], 9), dtype=torch.float32)
        if graph['polygon_pos'] is None:
            graph['polygon_pos'] = torch.zeros((graph['num_nodes'], 8), dtype=torch.float32)
        if graph['centroid'] is None:
            graph['centroid'] = graph['polygon_pos'].view(graph['num_nodes'], -1, 2)[:, -1, :]
        if graph['bbox'] is None:
            points = graph['polygon_pos'].view(graph['num_nodes'], -1, 2)
            graph['bbox'] = torch.cat([points.min(dim=1).values, points.max(dim=1).values], dim=-1)

        if graph['num_nodes'] == 1 and graph['class_id'].numel() == 1:
            graph['is_empty_graph'] = graph['is_empty_graph'] or int(graph['class_id'][0]) == 0
        return graph

    def transform(self, results: dict) -> dict:
        graph_path = results.get('graph_path')
        required = self.required or bool(results.get('graph_required', False))

        if graph_path is None or not osp.exists(graph_path):
            if required or not self.allow_missing:
                raise FileNotFoundError(f'Land-cover graph not found: {graph_path}')
            results['landcover_graph'] = _empty_graph()
            return results

        try:
            results['landcover_graph'] = self._load_graph(graph_path)
        except Exception as exc:
            if required:
                raise
            print_log(
                f'[WARNING] Failed to load land-cover graph {graph_path}: {exc}',
                logger='current')
            results['landcover_graph'] = _empty_graph()
        return results

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(required={self.required}, '
                f'allow_missing={self.allow_missing})')
