from __future__ import annotations

import math
from collections import Counter
from typing import Any

import geopandas as gpd
import torch

from class_mapping import (
    EDGE_TYPE_TO_ID,
    LEVEL_TO_ID,
    are_ancestor_descendant,
    is_ancestor_descendant,
)
from graph_utils import (
    Data,
    EPS,
    _edge_num_feat,
    _empty_graph,
    _geolink_polygon_relation,
    _safe_intersection_area,
    _self_edge_num_feat,
    records_from_gdf,
)


def _query_sindex(
    geometries: gpd.GeoSeries,
    bounds: tuple[float, float, float, float],
) -> list[int]:
    """Return geometries whose bounding boxes intersect bounds."""
    if len(geometries) == 0:
        return []
    try:
        return [int(idx) for idx in geometries.sindex.intersection(bounds)]
    except Exception:
        minx, miny, maxx, maxy = bounds
        candidates = []
        for idx, geom in enumerate(geometries):
            if geom is None or geom.is_empty:
                continue
            gx0, gy0, gx1, gy1 = geom.bounds
            if gx1 >= minx and gx0 <= maxx and gy1 >= miny and gy0 <= maxy:
                candidates.append(idx)
        return candidates


def _records_to_geoseries(records: list[Any]) -> gpd.GeoSeries:
    return gpd.GeoSeries([record.geometry for record in records])


def _build_data_from_records(
    records: list[Any],
    edge_src: list[int],
    edge_dst: list[int],
    edge_types: list[int],
    edge_feats: list[list[float]],
    hierarchy_pair_count: int,
) -> tuple[Data, dict[str, Any]]:
    data = Data(
        edge_index=torch.tensor([edge_src, edge_dst], dtype=torch.long),
        edge_type=torch.tensor(edge_types, dtype=torch.long),
        edge_num_feat=torch.tensor(edge_feats, dtype=torch.float32),
        class_code=torch.tensor([r.class_code for r in records], dtype=torch.long),
        large_code=torch.tensor([r.large_code for r in records], dtype=torch.long),
        middle_code=torch.tensor([r.middle_code for r in records], dtype=torch.long),
        small_code=torch.tensor([r.small_code for r in records], dtype=torch.long),
        class_id=torch.tensor([r.class_id for r in records], dtype=torch.long),
        large_id=torch.tensor([r.large_id for r in records], dtype=torch.long),
        middle_id=torch.tensor([r.middle_id for r in records], dtype=torch.long),
        small_id=torch.tensor([r.small_id for r in records], dtype=torch.long),
        level_id=torch.tensor([r.level_id for r in records], dtype=torch.long),
        geom_feat=torch.tensor([r.geom_feat for r in records], dtype=torch.float32),
        polygon_pos=torch.tensor([r.polygon_pos for r in records], dtype=torch.float32),
        centroid=torch.tensor([r.centroid for r in records], dtype=torch.float32),
        bbox=torch.tensor([r.bbox for r in records], dtype=torch.float32),
    )
    data.num_nodes = len(records)
    data.polygon_wkt = [record.geometry.wkt for record in records]

    edge_counts = Counter(edge_types)
    stats = {
        "empty": False,
        "large_nodes": sum(r.level_id == LEVEL_TO_ID["large"] for r in records),
        "middle_nodes": sum(r.level_id == LEVEL_TO_ID["middle"] for r in records),
        "small_nodes": sum(r.level_id == LEVEL_TO_ID["small"] for r in records),
        "total_nodes": len(records),
        "spatial_edges": sum(edge_counts.get(edge_type, 0) for edge_type in range(1, 6)),
        "hierarchy_edges": (
            edge_counts.get(EDGE_TYPE_TO_ID["hierarchy_parent_to_child"], 0)
            + edge_counts.get(EDGE_TYPE_TO_ID["hierarchy_child_to_parent"], 0)
        ),
        "total_edges": len(edge_types),
        "class_code_distribution": Counter(r.class_code for r in records),
        "level_id_distribution": Counter(r.level_id for r in records),
        "edge_type_distribution": edge_counts,
        "hierarchy_pair_count": hierarchy_pair_count,
    }
    return data, stats


def build_landcover_graph_fast(
    large_gdf: gpd.GeoDataFrame,
    middle_gdf: gpd.GeoDataFrame,
    small_gdf: gpd.GeoDataFrame,
    chip_bounds: tuple[float, float, float, float],
    near_threshold: float = 5.0,
    hierarchy_cover_threshold: float = 0.5,
    cross_level_spatial_edges: bool = False,
) -> tuple[Data, dict[str, Any]]:
    """Build the same graph schema after pruning pairs with spatial indexes."""
    del near_threshold  # Signature compatibility with the base graph builder.

    records = []
    records.extend(records_from_gdf(large_gdf, "large", chip_bounds))
    records.extend(records_from_gdf(middle_gdf, "middle", chip_bounds))
    records.extend(records_from_gdf(small_gdf, "small", chip_bounds))
    if not records:
        return _empty_graph()

    left, bottom, right, top = chip_bounds
    chip_diagonal = math.hypot(max(right - left, EPS), max(top - bottom, EPS))
    edge_src, edge_dst, edge_types, edge_feats = [], [], [], []
    hierarchy_pairs: set[frozenset[int]] = set()

    def add_edge(src: int, dst: int, edge_type: int, feat: list[float]) -> None:
        edge_src.append(src)
        edge_dst.append(dst)
        edge_types.append(edge_type)
        edge_feats.append(feat)

    for idx in range(len(records)):
        add_edge(idx, idx, EDGE_TYPE_TO_ID["self"], _self_edge_num_feat())

    all_geometries = _records_to_geoseries(records)
    for parent_idx, parent in enumerate(records):
        parent_codes = parent.code_dict()
        for child_idx in _query_sindex(all_geometries, parent.geometry.bounds):
            if parent_idx == child_idx:
                continue
            child = records[child_idx]
            if not is_ancestor_descendant(parent_codes, child.code_dict()):
                continue

            intersection_area = _safe_intersection_area(parent.geometry, child.geometry)
            if intersection_area <= 0.0:
                continue
            child_cover = intersection_area / max(float(child.geometry.area), EPS)
            parent_cover = intersection_area / max(float(parent.geometry.area), EPS)
            if child_cover < float(hierarchy_cover_threshold):
                continue

            feat = _edge_num_feat(
                parent.geometry,
                child.geometry,
                chip_diagonal,
                child_cover_ratio=child_cover,
                parent_cover_ratio=parent_cover,
            )
            add_edge(
                parent_idx,
                child_idx,
                EDGE_TYPE_TO_ID["hierarchy_parent_to_child"],
                feat,
            )
            add_edge(
                child_idx,
                parent_idx,
                EDGE_TYPE_TO_ID["hierarchy_child_to_parent"],
                feat,
            )
            hierarchy_pairs.add(frozenset((parent_idx, child_idx)))

    if cross_level_spatial_edges:
        level_groups = [list(range(len(records)))]
    else:
        by_level: dict[int, list[int]] = {}
        for idx, record in enumerate(records):
            by_level.setdefault(int(record.level_id), []).append(idx)
        level_groups = list(by_level.values())

    for indices in level_groups:
        if len(indices) < 2:
            continue
        geometries = gpd.GeoSeries([records[idx].geometry for idx in indices])
        for local_i, global_i in enumerate(indices):
            record_i = records[global_i]
            for local_j in _query_sindex(geometries, record_i.geometry.bounds):
                if local_j <= local_i:
                    continue
                global_j = indices[local_j]
                record_j = records[global_j]
                if cross_level_spatial_edges and are_ancestor_descendant(
                    record_i.code_dict(), record_j.code_dict()
                ):
                    continue
                relation = _geolink_polygon_relation(
                    record_i.geometry, record_j.geometry
                )
                if relation is None:
                    continue
                forward_type, backward_type = relation
                feat = _edge_num_feat(
                    record_i.geometry, record_j.geometry, chip_diagonal
                )
                add_edge(global_i, global_j, forward_type, feat)
                add_edge(global_j, global_i, backward_type, feat)

    return _build_data_from_records(
        records,
        edge_src,
        edge_dst,
        edge_types,
        edge_feats,
        len(hierarchy_pairs),
    )
