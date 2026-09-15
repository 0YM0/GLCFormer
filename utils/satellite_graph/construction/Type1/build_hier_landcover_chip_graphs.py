"""
cd /mnt/disk1/workspace_jym/lab_project/barn_detection/mmgeo_master
/home/jym/anaconda3/envs/mmseg_final/bin/python \
  utils/satellite_graph/construction/build_hier_landcover_chip_graphs.py \
  --csv_path data/satellite_graph/subset/data_list/spatial_split/80_10_10/sampling/train_sample_512_ocs10_rs10_subset50p.csv \
  --image_dir data/exp3_250818/images/recent \
  --landcover_root data/geospatial_unzip \
  --landcover_year 2024 \
  --min_landcover_year 2020 \
  --max_landcover_year 2024 \
  --out_dir data/satellite_graph/subset/graph_pt/type1_hier \
  --chip_size 512 \
  --image_col image_fn \
  --x_col xmin \
  --y_col ymin \
  --near_threshold 5.0 \
  --hierarchy_cover_threshold 0.5 \
  --num_workers 8
"""

from __future__ import annotations

import argparse
import logging
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from threading import Lock, RLock
from typing import Any

import geopandas as gpd
import pandas as pd
import rasterio
import torch
from rasterio.windows import Window, bounds as window_bounds
from tqdm import tqdm

from class_mapping import (
    EDGE_TYPE_TO_ID,
    GRAPH_SCHEMA_VERSION,
    selected_codes_for_level,
    write_vocab_json,
)
from graph_utils import build_landcover_graph
from landcover_io import (
    clip_landcover_to_bounds,
    discover_landcover_sources,
    explicit_sources,
    load_landcover_level,
    read_dbf_unique_values,
    reproject_if_needed,
    sanitize_polygon_gdf,
)


LOGGER = logging.getLogger(__name__)
LANDCOVER_LEVELS = ("large", "middle", "small")

IMAGE_COLUMN_CANDIDATES: tuple[str, ...] = (
    "image_fn",
    "img_fn",
    "image_file",
    "img_file",
    "img",
    "image",
    "img_path",
    "image_path",
    "filename",
    "file_name",
    "tif",
    "tif_path",
    "ori_img",
    "ori_image",
    "map_name",
    "mapid",
    "map_id",
)

X_COLUMN_CANDIDATES: tuple[str, ...] = (
    "x", "col", "left", "x_min", "xmin", "start_x", "offset_x"
)
Y_COLUMN_CANDIDATES: tuple[str, ...] = (
    "y", "row", "top", "y_min", "ymin", "start_y", "offset_y"
)
WIDTH_COLUMN_CANDIDATES: tuple[str, ...] = ("width", "w", "chip_w")
HEIGHT_COLUMN_CANDIDATES: tuple[str, ...] = ("height", "h", "chip_h")
SAMPLE_ID_CANDIDATES: tuple[str, ...] = ("sample_id", "id", "chip_id")
MAPID_COLUMN_CANDIDATES: tuple[str, ...] = (
    "mapid", "map_id", "map", "tile_id", "tileid", "sheet_id", "sheetid"
)
REGION_COLUMN_CANDIDATES: tuple[str, ...] = (
    "region", "city", "area", "sig", "sigungu", "sigungu_name"
)
REGION_NAME_ALIASES: dict[str, tuple[str, ...]] = {
    "Goesan": ("괴산군", "Goesan"),
    "Gyeongju": ("경주시", "Gyeongju"),
    "Gumi": ("구미시", "Gumi"),
    "Namwon": ("남원시", "Namwon"),
    "Nonsan": ("논산시", "Nonsan"),
    "Suncheon": ("순천시", "Suncheon"),
    "Yesan": ("예산군", "Yesan"),
    "Pocheon": ("포천시", "Pocheon"),
}


@dataclass
class ColumnConfig:
    image_col: str
    x_col: str
    y_col: str
    width_col: str | None = None
    height_col: str | None = None
    sample_id_col: str | None = None
    mapid_col: str | None = None
    region_col: str | None = None


@dataclass
class LandcoverBundle:
    large: gpd.GeoDataFrame
    middle: gpd.GeoDataFrame
    small: gpd.GeoDataFrame
    _cache: dict[str, dict[str, gpd.GeoDataFrame]] = field(default_factory=dict)
    _lock: Lock = field(default_factory=Lock)

    @staticmethod
    def _crs_key(crs: Any) -> str:
        if crs is None:
            return "none"
        if hasattr(crs, "to_string"):
            return str(crs.to_string())
        return str(crs)

    def for_crs(self, target_crs: Any) -> dict[str, gpd.GeoDataFrame]:
        key = self._crs_key(target_crs)
        with self._lock:
            if key not in self._cache:
                self._cache[key] = {
                    "large": reproject_if_needed(self.large, target_crs),
                    "middle": reproject_if_needed(self.middle, target_crs),
                    "small": reproject_if_needed(self.small, target_crs),
                }
            return self._cache[key]


class LandcoverProvider:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self._sources_cache: dict[int, dict[str, list[Any]]] = {}
        self._mapid_cache: dict[int, dict[str, set[str]]] = {}
        self._area_cache: dict[tuple[int, str], LandcoverBundle] = {}
        self._lock = RLock()

    def get_sources(self, year: int) -> dict[str, list[Any]]:
        with self._lock:
            if year not in self._sources_cache:
                LOGGER.info("Discovering land-cover sources for matched year %s", year)
                self._sources_cache[year] = discover_sources_for_year(self.args, year)
            return self._sources_cache[year]

    def source_ids_for_mapid(self, year: int, mapid: str | None) -> set[str] | None:
        if not mapid:
            return None
        mapid = str(mapid).strip()
        if not mapid:
            return None

        with self._lock:
            if year not in self._mapid_cache:
                sources = self.get_sources(year)
                unique_sources = {}
                for level_sources in sources.values():
                    for source in level_sources:
                        unique_sources[source.display_name] = source
                index: dict[str, set[str]] = {}
                for source_id, source in unique_sources.items():
                    vals = read_dbf_unique_values(source, "INX_NUM")
                    if vals:
                        LOGGER.info(
                            "Indexed %d INX_NUM values for %s",
                            len(vals),
                            source.display_name,
                        )
                    for value in vals:
                        index.setdefault(value, set()).add(source_id)
                self._mapid_cache[year] = index

            matches = self._mapid_cache[year].get(mapid)
            return matches if matches else None

    def source_ids_for_region(self, year: int, region: str | None) -> set[str] | None:
        if not region:
            return None
        region = str(region).strip()
        if not region:
            return None
        aliases = REGION_NAME_ALIASES.get(region, (region,))
        sources = self.get_sources(year)
        matches: set[str] = set()
        for level_sources in sources.values():
            for source in level_sources:
                source_text = source.display_name
                if any(alias in source_text for alias in aliases):
                    matches.add(source.display_name)
        return matches if matches else None

    def get_for_chip(
        self,
        year: int,
        bbox: tuple[float, float, float, float],
        mapid: str | None = None,
        region: str | None = None,
        cache_key: str | None = None,
    ) -> LandcoverBundle:
        if cache_key is not None:
            key = (int(year), cache_key)
            with self._lock:
                if key in self._area_cache:
                    return self._area_cache[key]

        sources = self.get_sources(year)
        allowed_source_ids = self.source_ids_for_region(year, region)
        if allowed_source_ids is None:
            allowed_source_ids = self.source_ids_for_mapid(year, mapid)
        if allowed_source_ids:
            sources = {
                level: [
                    source
                    for source in level_sources
                    if source.display_name in allowed_source_ids
                ]
                for level, level_sources in sources.items()
            }
        bundle = load_landcovers_for_year_bbox(self.args, sources, bbox)
        if cache_key is not None:
            with self._lock:
                self._area_cache[key] = bundle
        return bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build hierarchical land-cover chip graphs.")
    parser.add_argument("--csv_path", required=True)
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--out_dir", required=True)

    parser.add_argument("--landcover_large_path", default=None)
    parser.add_argument("--landcover_middle_path", default=None)
    parser.add_argument("--landcover_small_path", default=None)
    parser.add_argument("--landcover_root", default=None)
    parser.add_argument("--landcover_year", default=2024)
    parser.add_argument("--min_landcover_year", type=int, default=2020)
    parser.add_argument("--max_landcover_year", type=int, default=2024)
    parser.add_argument("--fixed_landcover_year", action="store_true")
    parser.add_argument("--allow_multilevel_fallback", action="store_true", default=True)
    parser.add_argument("--disable_multilevel_fallback", action="store_true")
    parser.add_argument("--region_name", default=None)

    parser.add_argument("--code_col", default=None)
    parser.add_argument("--chip_size", type=int, default=512)
    parser.add_argument("--image_col", default=None)
    parser.add_argument("--x_col", default=None)
    parser.add_argument("--y_col", default=None)
    parser.add_argument("--width_col", default=None)
    parser.add_argument("--height_col", default=None)
    parser.add_argument("--image_ext", default=".tif")
    parser.add_argument("--near_threshold", type=float, default=5.0)
    parser.add_argument("--hierarchy_cover_threshold", type=float, default=0.5)
    parser.add_argument("--min_area_abs", type=float, default=1.0)
    parser.add_argument("--min_area_ratio", type=float, default=0.0)
    parser.add_argument("--min_large_area_ratio", type=float, default=0.01)
    parser.add_argument("--min_middle_area_ratio", type=float, default=0.001)
    parser.add_argument("--min_small_area_ratio", type=float, default=0.001)
    parser.add_argument("--cross_level_spatial_edges", action="store_true")
    parser.add_argument("--no_merge_by_code", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--num_workers", type=int, default=0)
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def _find_column(
    columns: list[str],
    cli_value: str | None,
    candidates: tuple[str, ...],
    required: bool = True,
    label: str = "column",
) -> str | None:
    if cli_value:
        if cli_value not in columns:
            raise KeyError(f"Requested {label} '{cli_value}' not found. Columns: {columns}")
        return cli_value

    exact = {column: column for column in columns}
    lowered = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate in exact:
            return exact[candidate]
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]

    if required:
        raise KeyError(f"Could not auto-detect {label}. Candidates={candidates}. Columns={columns}")
    return None


def detect_columns(df: pd.DataFrame, args: argparse.Namespace) -> ColumnConfig:
    columns = [str(column) for column in df.columns]
    print("CSV columns:")
    for column in columns:
        print(f"  - {column}")

    return ColumnConfig(
        image_col=_find_column(
            columns, args.image_col, IMAGE_COLUMN_CANDIDATES, label="image column"
        ),
        x_col=_find_column(
            columns, args.x_col, X_COLUMN_CANDIDATES, label="x/col offset column"
        ),
        y_col=_find_column(
            columns, args.y_col, Y_COLUMN_CANDIDATES, label="y/row offset column"
        ),
        width_col=_find_column(
            columns, args.width_col, WIDTH_COLUMN_CANDIDATES,
            required=False, label="chip width column"
        ),
        height_col=_find_column(
            columns, args.height_col, HEIGHT_COLUMN_CANDIDATES,
            required=False, label="chip height column"
        ),
        sample_id_col=_find_column(
            columns, None, SAMPLE_ID_CANDIDATES,
            required=False, label="sample id column"
        ),
        mapid_col=_find_column(
            columns, None, MAPID_COLUMN_CANDIDATES,
            required=False, label="map id column"
        ),
        region_col=_find_column(
            columns, None, REGION_COLUMN_CANDIDATES,
            required=False, label="region column"
        ),
    )


def build_image_index(image_dir: Path, image_ext: str) -> dict[str, Path]:
    suffixes = {image_ext.lower()}
    if image_ext.lower() in {".tif", ".tiff"}:
        suffixes.update({".tif", ".tiff"})

    index: dict[str, Path] = {}
    prefix_hits: dict[str, list[Path]] = {}
    for path in sorted(image_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        index.setdefault(path.name, path)
        index.setdefault(path.stem, path)
        prefix = path.stem.split("_", 1)[0]
        if prefix:
            prefix_hits.setdefault(prefix, []).append(path)
    for prefix, paths in prefix_hits.items():
        if len(paths) == 1:
            index.setdefault(prefix, paths[0])
    LOGGER.info("Indexed %d image keys under %s", len(index), image_dir)
    return index


def resolve_image_path(value: Any, image_dir: Path, image_index: dict[str, Path], image_ext: str) -> Path:
    text = str(value).strip()
    if not text:
        raise FileNotFoundError("Empty image path/name value")

    raw = Path(text)
    if raw.is_absolute() and raw.exists():
        return raw

    candidates: list[Path] = []
    if not raw.is_absolute():
        candidates.append(image_dir / raw)
        candidates.append(image_dir / raw.name)
    if raw.suffix:
        candidates.append(image_dir / raw.name)
    else:
        candidates.append(image_dir / f"{raw.name}{image_ext}")

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    for key in (raw.name, raw.stem, text):
        if key in image_index:
            return image_index[key].resolve()

    prefix_matches = [
        path
        for key, path in image_index.items()
        if key.endswith(image_ext) and Path(key).stem.startswith(f"{text}_")
    ]
    unique_matches = sorted({path.resolve() for path in prefix_matches})
    if len(unique_matches) == 1:
        return unique_matches[0]
    if len(unique_matches) > 1:
        raise FileNotFoundError(
            f"Ambiguous image id '{text}' under {image_dir}; matched {len(unique_matches)} files. "
            "Use --image_col with a full filename column."
        )

    raise FileNotFoundError(f"Could not resolve image '{text}' under {image_dir}")


def safe_filename(value: str) -> str:
    value = value.strip()
    value = re.sub(r"[\\/:\s]+", "_", value)
    value = re.sub(r"[^0-9A-Za-z._가-힣-]+", "_", value)
    value = value.strip("._")
    return value or "sample"


def _row_int(row: pd.Series, column: str) -> int:
    return int(round(float(row[column])))


def row_sample_id(
    row: pd.Series,
    cfg: ColumnConfig,
    image_path: Path,
    x: int,
    y: int,
    chip_size: int,
) -> str:
    if cfg.sample_id_col is not None:
        value = str(row[cfg.sample_id_col]).strip()
        if value and value.lower() not in {"nan", "none", "null"}:
            return safe_filename(value)
    return safe_filename(f"{image_path.stem}_x{x}_y{y}_s{chip_size}")


def extract_year_from_image_name(
    image_name: str,
    fallback_year: int,
    min_year: int = 2020,
    max_year: int = 2024,
) -> int:
    matches = re.findall(r"(?<!\d)((?:19|20)\d{2})(?:\d{10})?(?!\d)", image_name)
    year = int(matches[-1]) if matches else int(fallback_year)
    if year < int(min_year):
        return int(min_year)
    if year > int(max_year):
        return int(max_year)
    return year


def discover_sources_for_year(
    args: argparse.Namespace, landcover_year: int | str
) -> dict[str, list[Any]]:
    explicit_mode = any(
        [args.landcover_large_path, args.landcover_middle_path, args.landcover_small_path]
    )
    if explicit_mode:
        return explicit_sources(
            args.landcover_large_path,
            args.landcover_middle_path,
            args.landcover_small_path,
        )
    if not args.landcover_root:
        raise ValueError("Provide either explicit landcover paths or --landcover_root.")
    return discover_landcover_sources(
        landcover_root=args.landcover_root,
        landcover_year=landcover_year,
        region_name=args.region_name,
        code_col=args.code_col,
        allow_multilevel_fallback=(
            bool(args.allow_multilevel_fallback)
            and not bool(args.disable_multilevel_fallback)
        ),
    )


def load_landcovers_for_year_bbox(
    args: argparse.Namespace,
    sources: dict[str, list[Any]],
    bbox: tuple[float, float, float, float],
) -> LandcoverBundle:

    bundle_frames: dict[str, gpd.GeoDataFrame] = {}
    for level in LANDCOVER_LEVELS:
        level_sources = sources.get(level, [])
        bundle_frames[level] = load_landcover_level(
            level_sources,
            level=level,
            code_col=args.code_col,
            target_codes=selected_codes_for_level(level),
            bbox=bbox,
        )

    return LandcoverBundle(
        large=bundle_frames["large"],
        middle=bundle_frames["middle"],
        small=bundle_frames["small"],
    )


def level_min_area_ratios(args: argparse.Namespace) -> dict[str, float]:
    base_ratio = float(args.min_area_ratio)
    return {
        level: max(base_ratio, float(getattr(args, f"min_{level}_area_ratio")))
        for level in LANDCOVER_LEVELS
    }


def clip_landcover_bundle(
    landcovers: dict[str, gpd.GeoDataFrame],
    chip_bounds: tuple[float, float, float, float],
    args: argparse.Namespace,
) -> tuple[dict[str, gpd.GeoDataFrame], dict[str, float]]:
    ratios = level_min_area_ratios(args)
    clipped = {
        level: clip_landcover_to_bounds(
            landcovers[level],
            chip_bounds,
            min_area_abs=args.min_area_abs,
            min_area_ratio=ratios[level],
        )
        for level in LANDCOVER_LEVELS
    }
    if not args.no_merge_by_code:
        clipped = {
            level: dissolve_connected_components_by_code(frame)
            for level, frame in clipped.items()
        }
    return clipped, ratios


def dissolve_connected_components_by_code(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Merge same-code fragments, then split disconnected components back into nodes."""
    if gdf.empty or "_lc_code" not in gdf.columns:
        return gdf

    rows = []
    crs = gdf.crs
    for code, group in gdf.groupby("_lc_code", sort=True):
        union_geom = group.geometry.union_all() if hasattr(group.geometry, "union_all") else group.geometry.unary_union
        if union_geom is None or union_geom.is_empty:
            continue
        rows.append(
            {
                "_lc_code": int(code),
                "_lc_level": group["_lc_level"].iloc[0] if "_lc_level" in group.columns else "",
                "_source": ";".join(sorted(set(str(value) for value in group.get("_source", []))))[:512],
                "_orig_area": float(group.get("_orig_area", group.geometry.area).sum()),
                "_clipped_area_ratio": float(group.get("_clipped_area_ratio", pd.Series([1.0])).mean()),
                "geometry": union_geom,
            }
        )

    merged = gpd.GeoDataFrame(rows, geometry="geometry", crs=crs)
    if merged.empty:
        return merged

    merged = sanitize_polygon_gdf(merged)
    if not merged.empty:
        merged["_lc_code"] = merged["_lc_code"].astype(int)
    return merged


def attach_metadata(
    data,
    sample_id: str,
    image_path: Path,
    x: int,
    y: int,
    chip_size: int,
    chip_width: int,
    chip_height: int,
    landcover_year: int,
    crs: Any,
    chip_bounds: tuple[float, float, float, float],
) -> None:
    data.sample_id = str(sample_id)
    data.image_path = str(image_path)
    data.image_name = image_path.name
    data.x = int(x)
    data.y = int(y)
    data.chip_size = int(chip_size)
    data.chip_width = int(chip_width)
    data.chip_height = int(chip_height)
    data.landcover_year = int(landcover_year)
    data.crs = "" if crs is None else str(crs)
    data.chip_bounds = [float(value) for value in chip_bounds]
    data.graph_schema_version = GRAPH_SCHEMA_VERSION
    data.edge_type_vocab = dict(EDGE_TYPE_TO_ID)
    data.spatial_relation_rule = "GeoLink prepare_data/utils.py:get_spatial_relation_samedim; disjoint/other omitted"


def process_row(
    row_idx: int,
    row: pd.Series,
    cfg: ColumnConfig,
    args: argparse.Namespace,
    image_dir: Path,
    image_index: dict[str, Path],
    landcover_provider: LandcoverProvider,
    out_dir: Path,
) -> dict[str, Any]:
    image_path = resolve_image_path(row[cfg.image_col], image_dir, image_index, args.image_ext)
    matched_year = int(args.landcover_year) if args.fixed_landcover_year else extract_year_from_image_name(
        image_path.name,
        fallback_year=int(args.landcover_year),
        min_year=int(args.min_landcover_year),
        max_year=int(args.max_landcover_year),
    )
    x = _row_int(row, cfg.x_col)
    y = _row_int(row, cfg.y_col)
    chip_width = _row_int(row, cfg.width_col) if cfg.width_col else int(args.chip_size)
    chip_height = _row_int(row, cfg.height_col) if cfg.height_col else int(args.chip_size)
    chip_size = chip_width if chip_width == chip_height else int(args.chip_size)
    sample_id = row_sample_id(row, cfg, image_path, x, y, chip_size)
    graph_path = out_dir / f"{sample_id}.pt"

    if graph_path.exists() and not args.overwrite:
        return {
            "status": "skipped",
            "row_idx": row_idx,
            "sample_id": sample_id,
            "image_name": image_path.name,
            "x": x,
            "y": y,
            "landcover_year": matched_year,
            "graph_path": str(graph_path),
        }

    with rasterio.open(image_path) as src:
        window = Window(col_off=x, row_off=y, width=chip_width, height=chip_height)
        chip_bounds = tuple(float(value) for value in window_bounds(window, src.transform))
        image_bounds = tuple(float(value) for value in src.bounds)
        raster_crs = src.crs

    mapid = str(row[cfg.mapid_col]).strip() if cfg.mapid_col is not None else None
    region = str(row[cfg.region_col]).strip() if cfg.region_col is not None else None
    landcovers = landcover_provider.get_for_chip(
        matched_year,
        image_bounds,
        mapid=mapid,
        region=region,
        cache_key=image_path.name,
    )
    landcovers_by_crs = landcovers.for_crs(raster_crs)
    clipped, min_area_ratios = clip_landcover_bundle(
        landcovers_by_crs, chip_bounds, args
    )

    data, stats = build_landcover_graph(
        large_gdf=clipped["large"],
        middle_gdf=clipped["middle"],
        small_gdf=clipped["small"],
        chip_bounds=chip_bounds,
        near_threshold=args.near_threshold,
        hierarchy_cover_threshold=args.hierarchy_cover_threshold,
        cross_level_spatial_edges=args.cross_level_spatial_edges,
    )
    attach_metadata(
        data=data,
        sample_id=sample_id,
        image_path=image_path,
        x=x,
        y=y,
        chip_size=chip_size,
        chip_width=chip_width,
        chip_height=chip_height,
        landcover_year=matched_year,
        crs=raster_crs,
        chip_bounds=chip_bounds,
    )
    data.cross_level_spatial_edges = bool(args.cross_level_spatial_edges)
    data.min_area_ratio_by_level = min_area_ratios

    torch.save(data, graph_path)

    result = {
        "status": "success",
        "row_idx": row_idx,
        "sample_id": sample_id,
        "image_name": image_path.name,
        "x": x,
        "y": y,
        "landcover_year": matched_year,
        "graph_path": str(graph_path),
        **stats,
    }
    return result


def failure_record(row_idx: int, row: pd.Series | None, error: Exception) -> dict[str, Any]:
    record = {"row_idx": row_idx, "error": repr(error)}
    if row is not None:
        for column in row.index:
            value = row[column]
            if pd.isna(value):
                continue
            text = str(value)
            if len(text) > 300:
                text = text[:300]
            record[str(column)] = text
    return record


def new_summary(total: int) -> dict[str, Any]:
    return {
        "total": total,
        "success": 0,
        "skipped": 0,
        "failed": 0,
        "empty_graphs": 0,
        "node_counts": [],
        "class_code_distribution": Counter(),
        "level_id_distribution": Counter(),
        "edge_type_distribution": Counter(),
    }


def update_summary(summary: dict[str, Any], result: dict[str, Any]) -> None:
    status = result["status"]
    if status == "success":
        summary["success"] += 1
        if result.get("empty", False):
            summary["empty_graphs"] += 1
        summary["node_counts"].append(int(result.get("total_nodes", 0)))
        summary["class_code_distribution"].update(result.get("class_code_distribution", {}))
        summary["level_id_distribution"].update(result.get("level_id_distribution", {}))
        summary["edge_type_distribution"].update(result.get("edge_type_distribution", {}))
    elif status == "skipped":
        summary["skipped"] += 1


def print_summary(summary: dict[str, Any]) -> None:
    node_counts = summary["node_counts"]
    if node_counts:
        node_series = pd.Series(node_counts)
        node_desc = {
            "min": int(node_series.min()),
            "p50": float(node_series.quantile(0.5)),
            "p90": float(node_series.quantile(0.9)),
            "max": int(node_series.max()),
        }
    else:
        node_desc = {}

    print("\nSummary")
    print(f"  total samples: {summary['total']}")
    print(f"  success: {summary['success']}")
    print(f"  skipped: {summary['skipped']}")
    print(f"  failed: {summary['failed']}")
    print(f"  empty graphs: {summary['empty_graphs']}")
    print(f"  total node distribution: {node_desc}")
    print(f"  class_code distribution: {dict(sorted(summary['class_code_distribution'].items()))}")
    print(f"  level_id distribution: {dict(sorted(summary['level_id_distribution'].items()))}")
    print(f"  edge_type distribution: {dict(sorted(summary['edge_type_distribution'].items()))}")


def progress_fields(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": result.get("sample_id", ""),
        "image_name": result.get("image_name", ""),
        "x": result.get("x", 0),
        "y": result.get("y", 0),
        "large": result.get("large_nodes", "-"),
        "middle": result.get("middle_nodes", "-"),
        "small": result.get("small_nodes", "-"),
        "nodes": result.get("total_nodes", "-"),
        "spatial": result.get("spatial_edges", "-"),
        "hier": result.get("hierarchy_edges", "-"),
        "edges": result.get("total_edges", "-"),
        "empty": result.get("empty", "-"),
    }


def consume_tasks(tasks, total: int):
    """Consume `(row index, row, result callback)` tasks in one common loop."""
    summary = new_summary(total)
    failures = []
    progress = tqdm(tasks, total=total, desc="Building graphs")
    for row_idx, row, get_result in progress:
        try:
            result = get_result()
            update_summary(summary, result)
            progress.set_postfix(**progress_fields(result))
        except Exception as exc:
            summary["failed"] += 1
            failures.append(failure_record(row_idx, row, exc))
            LOGGER.exception("Failed sample row_idx=%s: %s", row_idx, exc)
    return summary, failures


def run_graph_builds(
    df: pd.DataFrame,
    cfg: ColumnConfig,
    args: argparse.Namespace,
    image_dir: Path,
    image_index: dict[str, Path],
    landcover_provider: LandcoverProvider,
    out_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    process = partial(
        process_row,
        cfg=cfg,
        args=args,
        image_dir=image_dir,
        image_index=image_index,
        landcover_provider=landcover_provider,
        out_dir=out_dir,
    )

    if int(args.num_workers) <= 0:
        tasks = (
            (row_idx, row, partial(process, row_idx=row_idx, row=row))
            for row_idx, row in df.iterrows()
        )
        return consume_tasks(tasks, len(df))

    with ThreadPoolExecutor(max_workers=int(args.num_workers)) as executor:
        future_rows = {
            executor.submit(process, row_idx=row_idx, row=row): (row_idx, row)
            for row_idx, row in df.iterrows()
        }
        tasks = (
            (*future_rows[future], future.result)
            for future in as_completed(future_rows)
        )
        return consume_tasks(tasks, len(df))


def main() -> None:
    setup_logging()
    args = parse_args()

    csv_path = Path(args.csv_path).expanduser().resolve()
    image_dir = Path(args.image_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    write_vocab_json(out_dir / "vocab.json")

    df = pd.read_csv(csv_path)
    cfg = detect_columns(df, args)
    image_index = build_image_index(image_dir, args.image_ext)
    landcover_provider = LandcoverProvider(args)

    summary, failed = run_graph_builds(
        df, cfg, args, image_dir, image_index, landcover_provider, out_dir
    )

    failed_path = out_dir / "failed_samples.csv"
    pd.DataFrame(failed).to_csv(failed_path, index=False)
    print_summary(summary)
    print(f"  failed_samples.csv: {failed_path}")
    print(f"  vocab.json: {out_dir / 'vocab.json'}")


if __name__ == "__main__":
    main()
