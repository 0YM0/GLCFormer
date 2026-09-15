"""Sanity checks for LandCoverGraphImageFusionV3."""

import torch
from mmseg.structures import SegDataSample

from mmgeo.models.v3_graph_fusion import LandCoverGraphImageFusionV3


def make_sample(graph):
    sample = SegDataSample()
    sample.set_metainfo(dict(landcover_graph=graph))
    return sample


def make_valid_graph(num_nodes: int = 6):
    edge_index = torch.tensor([
        [0, 1, 2, 3, 4, 5, 0, 1, 2, 4],
        [1, 2, 3, 4, 5, 0, 0, 1, 2, 4],
    ], dtype=torch.long)
    return dict(
        edge_index=edge_index,
        edge_type=torch.tensor([1, 1, 2, 2, 5, 5, 0, 0, 6, 7], dtype=torch.long),
        edge_num_feat=torch.rand(edge_index.shape[1], 6),
        class_id=torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.long),
        large_id=torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.long),
        middle_id=torch.zeros(num_nodes, dtype=torch.long),
        small_id=torch.zeros(num_nodes, dtype=torch.long),
        level_id=torch.ones(num_nodes, dtype=torch.long),
        geom_feat=torch.rand(num_nodes, 9),
        polygon_pos=torch.rand(num_nodes, 8),
        num_nodes=num_nodes,
        is_empty_graph=False,
    )


def make_empty_graph():
    return dict(
        edge_index=torch.zeros((2, 1), dtype=torch.long),
        edge_type=torch.zeros((1,), dtype=torch.long),
        edge_num_feat=torch.zeros(1, 6),
        class_id=torch.zeros(1, dtype=torch.long),
        large_id=torch.zeros(1, dtype=torch.long),
        middle_id=torch.zeros(1, dtype=torch.long),
        small_id=torch.zeros(1, dtype=torch.long),
        level_id=torch.zeros(1, dtype=torch.long),
        geom_feat=torch.zeros(1, 9),
        polygon_pos=torch.zeros(1, 8),
        num_nodes=1,
        is_empty_graph=True,
    )


def main():
    torch.manual_seed(11)
    batch_size, channels, height, width = 2, 256, 24, 24
    feat = torch.randn(batch_size, channels, height, width)
    data_samples = [make_sample(make_valid_graph()), make_sample(make_empty_graph())]

    module = LandCoverGraphImageFusionV3(
        image_in_channels=channels,
        embed_dims=128,
        num_heads=4,
        num_graph_layers=2,
        graph_num_heads=4,
        num_fusion_layers=1,
        fusion_residual_mode='delta',
        keep_scalar_gate=False,
        dropout=0.0,
    )
    module.eval()

    with torch.no_grad():
        out_list = module([feat.clone()], data_samples)
        out_tuple = module((feat.clone(),), data_samples)

    assert isinstance(out_list, list)
    assert isinstance(out_tuple, tuple)
    assert out_list[0].shape == feat.shape
    assert out_tuple[0].shape == feat.shape
    assert torch.allclose(out_list[0][1], feat[1], atol=1e-6), 'empty graph sample changed'
    assert torch.isfinite(out_list[0][0]).all(), 'valid graph sample has non-finite values'
    print('v3 graph fusion shape sanity check passed')


if __name__ == '__main__':
    main()
