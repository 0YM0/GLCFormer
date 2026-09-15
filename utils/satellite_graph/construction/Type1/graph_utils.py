from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

import geopandas as gpd
import numpy as np
import torch
from shapely import relate
from shapely.geometry import Point

try:
    from torch_geometric.data import Data
except ModuleNotFoundError:
    class Data:
        """Small torch-saveable fallback used when torch_geometric is unavailable."""

        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

        def keys(self):
            return [key for key in self.__dict__.keys() if not key.startswith("_")]

        def __getitem__(self, key):
            return getattr(self, key)

        def __setitem__(self, key, value):
            setattr(self, key, value)

    Data.__module__ = "graph_utils"

from class_mapping import (
    EDGE_TYPE_TO_ID,
    LEVEL_TO_ID,
    are_ancestor_descendant,
    build_node_code_fields,
    build_node_id_fields,
    is_ancestor_descendant,
)


EPS = 1e-12


@dataclass
class NodeRecord:
    geometry: Any
    level: str
    class_code: int
    large_code: int
    middle_code: int
    small_code: int
    class_id: int
    large_id: int
    middle_id: int
    small_id: int
    level_id: int
    geom_feat: list[float]
    polygon_pos: list[float]
    centroid: list[float]
    bbox: list[float]

    def code_dict(self) -> dict[str, int]:
        return {
            "class_code": self.class_code,
            "large_code": self.large_code,
            "middle_code": self.middle_code,
            "small_code": self.small_code,
            "level_id": self.level_id,
        }


def _clamp01(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return min(1.0, max(0.0, float(value)))


def normalize_xy(x: float, y: float, chip_bounds: tuple[float, float, float, float]) -> tuple[float, float]:
    left, bottom, right, top = chip_bounds
    width = max(right - left, EPS)
    height = max(top - bottom, EPS)
    x_norm = (x - left) / width
    y_norm = 1.0 - ((y - bottom) / height)
    return _clamp01(x_norm), _clamp01(y_norm)


def normalize_bbox(bounds: tuple[float, float, float, float], chip_bounds: tuple[float, float, float, float]) -> list[float]:
    minx, miny, maxx, maxy = bounds
    x0, y_bottom = normalize_xy(minx, miny, chip_bounds)
    x1, y_top = normalize_xy(maxx, maxy, chip_bounds)
    y0 = min(y_top, y_bottom)
    y1 = max(y_top, y_bottom)
    return [_clamp01(x0), _clamp01(y0), _clamp01(x1), _clamp01(y1)]


def sample_polygon_points(geom, num_points: int = 3) -> list[Point]:
    if geom is None or geom.is_empty:
        return [Point(0.0, 0.0) for _ in range(num_points)]

    center = geom.representative_point()
    minx, miny, maxx, maxy = geom.bounds
    radius = max(maxx - minx, maxy - miny) * 0.35
    if radius <= 0:
        return [center for _ in range(num_points)]

    points: list[Point] = []
    angles = np.linspace(0.0, 2.0 * math.pi, num_points, endpoint=False)
    for theta in angles:
        chosen = None
        for scale in (1.0, 0.75, 0.5, 0.25, 0.1):
            candidate = Point(center.x + math.cos(theta) * radius * scale, center.y + math.sin(theta) * radius * scale)
            if geom.covers(candidate):
                chosen = candidate
                break
        points.append(chosen if chosen is not None else center)
    return points


def polygon_pos_feature(geom, chip_bounds: tuple[float, float, float, float]) -> list[float]:
    sampled = sample_polygon_points(geom, num_points=3)
    centroid = geom.centroid if geom is not None and not geom.is_empty else Point(0.0, 0.0)
    all_points = sampled + [centroid]

    out: list[float] = []
    for point in all_points:
        x, y = normalize_xy(point.x, point.y, chip_bounds)
        out.extend([x, y])
    return out


def geometry_features(
    geom,
    chip_bounds: tuple[float, float, float, float],
    clipped_area_ratio: float = 1.0,
) -> tuple[list[float], list[float], list[float], list[float]]:
    left, bottom, right, top = chip_bounds
    chip_width = max(right - left, EPS)
    chip_height = max(top - bottom, EPS)
    chip_area = max(chip_width * chip_height, EPS)
    chip_perimeter = max(2.0 * (chip_width + chip_height), EPS)

    area = max(float(geom.area), 0.0)
    perimeter = max(float(geom.length), 0.0)
    compactness = 0.0 if perimeter <= EPS else (4.0 * math.pi * area) / (perimeter * perimeter)

    minx, miny, maxx, maxy = geom.bounds
    bbox_width = max(maxx - minx, 0.0)
    bbox_height = max(maxy - miny, 0.0)
    aspect_ratio = bbox_width / max(bbox_height, EPS)

    centroid_geom = geom.centroid
    centroid = list(normalize_xy(centroid_geom.x, centroid_geom.y, chip_bounds))
    bbox = normalize_bbox(geom.bounds, chip_bounds)
    polygon_pos = polygon_pos_feature(geom, chip_bounds)

    geom_feat = [
        area / chip_area,
        perimeter / chip_perimeter,
        compactness,
        bbox_width / chip_width,
        bbox_height / chip_height,
        aspect_ratio,
        centroid[0],
        centroid[1],
        _clamp01(float(clipped_area_ratio)),
    ]
    return geom_feat, polygon_pos, centroid, bbox


def _safe_intersection_area(a, b) -> float:
    try:
        return max(float(a.intersection(b).area), 0.0)
    except Exception:
        return 0.0


def _safe_shared_boundary_length(a, b) -> float:
    try:
        return max(float(a.boundary.intersection(b.boundary).length), 0.0)
    except Exception:
        return 0.0


def _centroid_distance(a, b) -> float:
    try:
        return float(a.centroid.distance(b.centroid))
    except Exception:
        return 0.0


def _edge_num_feat(
    src_geom,
    dst_geom,
    chip_diagonal: float,
    child_cover_ratio: float = 0.0,
    parent_cover_ratio: float = 0.0,
) -> list[float]:
    distance = float(src_geom.distance(dst_geom))
    shared_boundary = _safe_shared_boundary_length(src_geom, dst_geom)
    intersection_area = _safe_intersection_area(src_geom, dst_geom)
    min_perimeter = max(min(float(src_geom.length), float(dst_geom.length)), EPS)
    min_area = max(min(float(src_geom.area), float(dst_geom.area)), EPS)
    centroid_distance = _centroid_distance(src_geom, dst_geom)

    return [
        distance / max(chip_diagonal, EPS),
        shared_boundary / min_perimeter,
        intersection_area / min_area,
        centroid_distance / max(chip_diagonal, EPS),
        float(child_cover_ratio),
        float(parent_cover_ratio),
    ]


def _self_edge_num_feat() -> list[float]:
    return [0.0, 0.0, 1.0, 0.0, 1.0, 1.0]


def records_from_gdf(
    gdf: gpd.GeoDataFrame,
    level: str,
    chip_bounds: tuple[float, float, float, float],
) -> list[NodeRecord]:
    records: list[NodeRecord] = []
    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        class_code = int(row["_lc_code"])
        clipped_area_ratio = float(row.get("_clipped_area_ratio", 1.0))
        code_fields = build_node_code_fields(level, class_code)
        id_fields = build_node_id_fields(code_fields)
        geom_feat, polygon_pos, centroid, bbox = geometry_features(
            geom=geom,
            chip_bounds=chip_bounds,
            clipped_area_ratio=clipped_area_ratio,
        )
        records.append(
            NodeRecord(
                geometry=geom,
                level=level,
                class_code=code_fields["class_code"],
                large_code=code_fields["large_code"],
                middle_code=code_fields["middle_code"],
                small_code=code_fields["small_code"],
                class_id=id_fields["class_id"],
                large_id=id_fields["large_id"],
                middle_id=id_fields["middle_id"],
                small_id=id_fields["small_id"],
                level_id=code_fields["level_id"],
                geom_feat=geom_feat,
                polygon_pos=polygon_pos,
                centroid=centroid,
                bbox=bbox,
            )
        )
    return records


def _geolink_polygon_relation(a, b) -> tuple[int, int] | None:
    """GeoLink same-dimension polygon relation, excluding disjoint/other."""
    try:
        relation = relate(b, a)

        # GeoLink get_spatial_relation_samedim:
        # self 0, disjoint 1, touches 2, overlaps 3, covers 4, covered_by 5, equals 6, other 7.
        if re.match("FF.{1,2}FF.{4,8}", relation) is not None:
            return None
        if (
            re.match("F[0,1,2,T].{7,14}", relation) is not None
            or re.match("F.{2,4}[0,1,2,T].{5,10}", relation) is not None
            or re.match("F.{3,6}[0,1,2,T].{4,8}", relation) is not None
        ):
            edge_type = EDGE_TYPE_TO_ID["spatial_touches"]
            return edge_type, edge_type
        if (
            re.match("[0,1,2,T].{5,10}FF.{1,2}", relation) is not None
            or re.match(".{1,2}[0,1,2,T].{4,8}FF.{1,2}", relation) is not None
            or re.match(".{3,6}[0,1,2,T].{2,4}FF.{1,2}", relation) is not None
            or re.match(".{4,8}[0,1,2,T].{1,2}FF.{1,2}", relation) is not None
        ):
            return EDGE_TYPE_TO_ID["spatial_covered_by"], EDGE_TYPE_TO_ID["spatial_covers"]
        if (
            re.match("[0,1,2,T].{1,2}F.{2,4}F.{3,6}", relation) is not None
            or re.match(".{1,2}[0,1,2,T]F.{2,4}F.{3,6}", relation) is not None
            or re.match(".{2,4}F[0,1,2,T].{1,2}F.{3,6}", relation) is not None
            or re.match(".{2,4}F.{1,2}[0,1,2,T]F.{3,6}", relation) is not None
        ):
            return EDGE_TYPE_TO_ID["spatial_covers"], EDGE_TYPE_TO_ID["spatial_covered_by"]
        if re.match("[0,1,2,T].{1,2}[0,1,2,T].{3,6}[0,1,2,T].{2,4}", relation) is not None:
            edge_type = EDGE_TYPE_TO_ID["spatial_overlaps"]
            return edge_type, edge_type
        if re.match("[0,1,2,T].{1,2}F.{2,4}FFF.{1,2}", relation) is not None:
            edge_type = EDGE_TYPE_TO_ID["spatial_equals"]
            return edge_type, edge_type
    except Exception:
        return None
    return None


def _empty_graph() -> tuple[Data, dict[str, Any]]:
    data = Data(
        edge_index=torch.tensor([[0], [0]], dtype=torch.long),
        edge_type=torch.tensor([EDGE_TYPE_TO_ID["self"]], dtype=torch.long),
        edge_num_feat=torch.tensor([_self_edge_num_feat()], dtype=torch.float32),
        class_code=torch.zeros(1, dtype=torch.long),
        large_code=torch.zeros(1, dtype=torch.long),
        middle_code=torch.zeros(1, dtype=torch.long),
        small_code=torch.zeros(1, dtype=torch.long),
        class_id=torch.zeros(1, dtype=torch.long),
        large_id=torch.zeros(1, dtype=torch.long),
        middle_id=torch.zeros(1, dtype=torch.long),
        small_id=torch.zeros(1, dtype=torch.long),
        level_id=torch.zeros(1, dtype=torch.long),
        geom_feat=torch.zeros((1, 9), dtype=torch.float32),
        polygon_pos=torch.zeros((1, 8), dtype=torch.float32),
        centroid=torch.zeros((1, 2), dtype=torch.float32),
        bbox=torch.zeros((1, 4), dtype=torch.float32),
    )
    data.num_nodes = 1
    data.polygon_wkt = [""]
    stats = {
        "empty": True,
        "large_nodes": 0,
        "middle_nodes": 0,
        "small_nodes": 0,
        "total_nodes": 1,
        "spatial_edges": 0,
        "hierarchy_edges": 0,
        "total_edges": 1,
        "class_code_distribution": Counter({0: 1}),
        "level_id_distribution": Counter({0: 1}),
        "edge_type_distribution": Counter({EDGE_TYPE_TO_ID["self"]: 1}),
    }
    return data, stats


def build_landcover_graph(
    large_gdf: gpd.GeoDataFrame,
    middle_gdf: gpd.GeoDataFrame,
    small_gdf: gpd.GeoDataFrame,
    chip_bounds: tuple[float, float, float, float],
    near_threshold: float = 5.0,
    hierarchy_cover_threshold: float = 0.5,
    cross_level_spatial_edges: bool = False,
) -> tuple[Data, dict[str, Any]]:
    records: list[NodeRecord] = []
    records.extend(records_from_gdf(large_gdf, "large", chip_bounds))
    records.extend(records_from_gdf(middle_gdf, "middle", chip_bounds))
    records.extend(records_from_gdf(small_gdf, "small", chip_bounds))

    if not records:
        return _empty_graph()

    left, bottom, right, top = chip_bounds
    chip_diagonal = math.hypot(max(right - left, EPS), max(top - bottom, EPS))

    edge_src: list[int] = []
    edge_dst: list[int] = []
    edge_types: list[int] = []
    edge_feats: list[list[float]] = []
    hierarchy_pairs: set[frozenset[int]] = set()
    ancestor_pairs: set[frozenset[int]] = set()

    def add_edge(src: int, dst: int, edge_type: int, feat: list[float]) -> None:
        edge_src.append(src)
        edge_dst.append(dst)
        edge_types.append(edge_type)
        edge_feats.append(feat)

    for idx in range(len(records)):
        add_edge(idx, idx, EDGE_TYPE_TO_ID["self"], _self_edge_num_feat())

    for parent_idx, parent in enumerate(records):
        parent_codes = parent.code_dict()
        for child_idx, child in enumerate(records):
            if parent_idx == child_idx:
                continue
            child_codes = child.code_dict()
            if not is_ancestor_descendant(parent_codes, child_codes):
                continue

            pair_key = frozenset((parent_idx, child_idx))
            ancestor_pairs.add(pair_key)
            intersection_area = _safe_intersection_area(parent.geometry, child.geometry)
            child_area = max(float(child.geometry.area), EPS)
            parent_area = max(float(parent.geometry.area), EPS)
            child_cover_ratio = intersection_area / child_area
            parent_cover_ratio = intersection_area / parent_area

            if child_cover_ratio >= float(hierarchy_cover_threshold):
                feat = _edge_num_feat(
                    parent.geometry,
                    child.geometry,
                    chip_diagonal,
                    child_cover_ratio=child_cover_ratio,
                    parent_cover_ratio=parent_cover_ratio,
                )
                add_edge(parent_idx, child_idx, EDGE_TYPE_TO_ID["hierarchy_parent_to_child"], feat)
                add_edge(child_idx, parent_idx, EDGE_TYPE_TO_ID["hierarchy_child_to_parent"], feat)
                hierarchy_pairs.add(pair_key)

    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            if not cross_level_spatial_edges and records[i].level_id != records[j].level_id:
                continue
            pair_key = frozenset((i, j))
            if pair_key in ancestor_pairs:
                continue
            if are_ancestor_descendant(records[i].code_dict(), records[j].code_dict()):
                continue
            relation = _geolink_polygon_relation(records[i].geometry, records[j].geometry)
            if relation is None:
                continue

            type_i_to_j, type_j_to_i = relation
            feat = _edge_num_feat(records[i].geometry, records[j].geometry, chip_diagonal)
            add_edge(i, j, type_i_to_j, feat)
            add_edge(j, i, type_j_to_i, feat)

    data = Data(
        edge_index=torch.tensor([edge_src, edge_dst], dtype=torch.long),
        edge_type=torch.tensor(edge_types, dtype=torch.long),
        edge_num_feat=torch.tensor(edge_feats, dtype=torch.float32),
        class_code=torch.tensor([record.class_code for record in records], dtype=torch.long),
        large_code=torch.tensor([record.large_code for record in records], dtype=torch.long),
        middle_code=torch.tensor([record.middle_code for record in records], dtype=torch.long),
        small_code=torch.tensor([record.small_code for record in records], dtype=torch.long),
        class_id=torch.tensor([record.class_id for record in records], dtype=torch.long),
        large_id=torch.tensor([record.large_id for record in records], dtype=torch.long),
        middle_id=torch.tensor([record.middle_id for record in records], dtype=torch.long),
        small_id=torch.tensor([record.small_id for record in records], dtype=torch.long),
        level_id=torch.tensor([record.level_id for record in records], dtype=torch.long),
        geom_feat=torch.tensor([record.geom_feat for record in records], dtype=torch.float32),
        polygon_pos=torch.tensor([record.polygon_pos for record in records], dtype=torch.float32),
        centroid=torch.tensor([record.centroid for record in records], dtype=torch.float32),
        bbox=torch.tensor([record.bbox for record in records], dtype=torch.float32),
    )
    data.num_nodes = len(records)
    data.polygon_wkt = [record.geometry.wkt for record in records]

    edge_type_distribution = Counter(edge_types)
    stats = {
        "empty": False,
        "large_nodes": sum(record.level_id == LEVEL_TO_ID["large"] for record in records),
        "middle_nodes": sum(record.level_id == LEVEL_TO_ID["middle"] for record in records),
        "small_nodes": sum(record.level_id == LEVEL_TO_ID["small"] for record in records),
        "total_nodes": len(records),
        "spatial_edges": sum(edge_type_distribution.get(edge_type, 0) for edge_type in range(1, 6)),
        "hierarchy_edges": edge_type_distribution.get(EDGE_TYPE_TO_ID["hierarchy_parent_to_child"], 0)
        + edge_type_distribution.get(EDGE_TYPE_TO_ID["hierarchy_child_to_parent"], 0),
        "total_edges": len(edge_types),
        "class_code_distribution": Counter(record.class_code for record in records),
        "level_id_distribution": Counter(record.level_id for record in records),
        "edge_type_distribution": edge_type_distribution,
        "hierarchy_pair_count": len(hierarchy_pairs),
    }
    return data, stats
