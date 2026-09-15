"""
cd /mnt/disk1/workspace_jym/lab_project/barn_detection/mmgeo_master

/home/jym/anaconda3/envs/mmseg_final/bin/python \
  utils/satellite_graph/construction/build_hier_landcover_chip_graphs_fast.py \
  --csv_path data/satellite_graph/subset/data_list/spatial_split/80_10_10/sampling/train_sample_512_ocs10_rs10_subset50p.csv \
  --image_dir data/exp3_250818/images/recent \
  --landcover_root data/geospatial_unzip \
  --landcover_year 2024 \
  --min_landcover_year 2020 \
  --max_landcover_year 2024 \
  --out_dir data/satellite_graph/subset/graph_pt/type1_hier/train_512_ocs10_rs10_subset50p_landcover_wkt \
  --chip_size 512 \
  --image_col image_fn \
  --x_col xmin \
  --y_col ymin \
  --near_threshold 5.0 \
  --hierarchy_cover_threshold 0.5 \
  --num_workers 16
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterator

import pandas as pd
import torch
from tqdm import tqdm

import build_hier_landcover_chip_graphs as base
from graph_utils_fast import build_landcover_graph_fast


LOGGER = logging.getLogger(__name__)

# These objects are initialized before forking. Passing the provider through the
# executor would pickle several GB of GeoDataFrames and its locks are not picklable.
_WORKER_CFG: base.ColumnConfig | None = None
_WORKER_ARGS: Any = None
_WORKER_IMAGE_DIR: Path | None = None
_WORKER_IMAGE_INDEX: dict[str, Path] | None = None
_WORKER_LANDCOVER_PROVIDER: base.LandcoverProvider | None = None
_WORKER_OUT_DIR: Path | None = None
_RASTER_METADATA_CACHE: dict[str, tuple[Any, tuple[float, ...], Any]] = {}


class SourceCachedLandcoverProvider(base.LandcoverProvider):
    """Cache one full land-cover bundle for each year and selected source set."""

    def get_for_chip(
        self,
        year: int,
        bbox: tuple[float, float, float, float],
        mapid: str | None = None,
        region: str | None = None,
        cache_key: str | None = None,
    ) -> base.LandcoverBundle:
        del bbox, cache_key

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
            source_key: Any = tuple(sorted(allowed_source_ids))
        else:
            source_key = "all_sources"

        key = (int(year), repr(source_key))
        with self._lock:
            if key not in self._area_cache:
                LOGGER.info(
                    "Fast cache loading land-cover bundle for year=%s sources=%s",
                    year,
                    source_key,
                )
                self._area_cache[key] = base.load_landcovers_for_year_bbox(
                    self.args, sources, bbox=None
                )
            return self._area_cache[key]


def _row_optional_text(row: pd.Series, column: str | None) -> str | None:
    if column is None or pd.isna(row[column]):
        return None
    value = str(row[column]).strip()
    return value if value and value.lower() not in {"nan", "none", "null"} else None


def _resolve_row_image_and_year(
    row: pd.Series,
    cfg: base.ColumnConfig,
    args: Any,
    image_dir: Path,
    image_index: dict[str, Path],
) -> tuple[Path, int]:
    image_path = base.resolve_image_path(
        row[cfg.image_col], image_dir, image_index, args.image_ext
    )
    if args.fixed_landcover_year:
        year = int(args.landcover_year)
    else:
        year = base.extract_year_from_image_name(
            image_path.name,
            fallback_year=int(args.landcover_year),
            min_year=int(args.min_landcover_year),
            max_year=int(args.max_landcover_year),
        )
    return image_path, year


def _raster_metadata(image_path: Path) -> tuple[Any, tuple[float, ...], Any]:
    """Read immutable raster metadata once per image in each worker."""
    key = str(image_path)
    metadata = _RASTER_METADATA_CACHE.get(key)
    if metadata is None:
        with base.rasterio.open(image_path) as src:
            metadata = (
                src.transform,
                tuple(float(value) for value in src.bounds),
                src.crs,
            )
        _RASTER_METADATA_CACHE[key] = metadata
    return metadata


def process_row_fast(
    row_idx: int,
    row: pd.Series,
    cfg: base.ColumnConfig,
    args: Any,
    image_dir: Path,
    image_index: dict[str, Path],
    landcover_provider: base.LandcoverProvider,
    out_dir: Path,
) -> dict[str, Any]:
    """Build one graph while reusing cached raster and land-cover metadata."""
    image_path, matched_year = _resolve_row_image_and_year(
        row, cfg, args, image_dir, image_index
    )
    x = base._row_int(row, cfg.x_col)
    y = base._row_int(row, cfg.y_col)
    chip_width = base._row_int(row, cfg.width_col) if cfg.width_col else int(args.chip_size)
    chip_height = base._row_int(row, cfg.height_col) if cfg.height_col else int(args.chip_size)
    chip_size = chip_width if chip_width == chip_height else int(args.chip_size)
    sample_id = base.row_sample_id(row, cfg, image_path, x, y, chip_size)
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

    transform, image_bounds, raster_crs = _raster_metadata(image_path)
    window = base.Window(col_off=x, row_off=y, width=chip_width, height=chip_height)
    chip_bounds = tuple(
        float(value) for value in base.window_bounds(window, transform)
    )

    landcovers = landcover_provider.get_for_chip(
        matched_year,
        image_bounds,
        mapid=_row_optional_text(row, cfg.mapid_col),
        region=_row_optional_text(row, cfg.region_col),
        cache_key=image_path.name,
    )
    landcovers_by_crs = landcovers.for_crs(raster_crs)
    clipped, min_area_ratios = base.clip_landcover_bundle(
        landcovers_by_crs, chip_bounds, args
    )

    data, stats = build_landcover_graph_fast(
        large_gdf=clipped["large"],
        middle_gdf=clipped["middle"],
        small_gdf=clipped["small"],
        chip_bounds=chip_bounds,
        near_threshold=args.near_threshold,
        hierarchy_cover_threshold=args.hierarchy_cover_threshold,
        cross_level_spatial_edges=args.cross_level_spatial_edges,
    )
    base.attach_metadata(
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

    return {
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


def _preload_shared_data(
    df: pd.DataFrame,
    cfg: base.ColumnConfig,
    args: Any,
    image_dir: Path,
    image_index: dict[str, Path],
    landcover_provider: base.LandcoverProvider,
) -> None:
    """Load all required source bundles and CRS projections before ``fork``."""
    bundles: dict[int, base.LandcoverBundle] = {}
    image_paths: dict[str, Path] = {}
    progress = tqdm(df.iterrows(), total=len(df), desc="Preloading land cover")
    for _, row in progress:
        image_path, year = _resolve_row_image_and_year(
            row, cfg, args, image_dir, image_index
        )
        bundle = landcover_provider.get_for_chip(
            year,
            (0.0, 0.0, 0.0, 0.0),
            mapid=_row_optional_text(row, cfg.mapid_col),
            region=_row_optional_text(row, cfg.region_col),
            cache_key=image_path.name,
        )
        bundles[id(bundle)] = bundle
        image_paths[str(image_path)] = image_path

    crs_values: dict[str, Any] = {}
    for image_path in tqdm(
        image_paths.values(), desc="Reading raster metadata", total=len(image_paths)
    ):
        _, _, raster_crs = _raster_metadata(image_path)
        crs_values[str(raster_crs)] = raster_crs

    for bundle in tqdm(
        bundles.values(), desc="Preprojecting land cover", total=len(bundles)
    ):
        for raster_crs in crs_values.values():
            bundle.for_crs(raster_crs)

    LOGGER.info(
        "Preloaded %d land-cover bundles, %d raster files, and %d CRS values",
        len(bundles),
        len(image_paths),
        len(crs_values),
    )


def _set_worker_context(
    cfg: base.ColumnConfig,
    args: Any,
    image_dir: Path,
    image_index: dict[str, Path],
    landcover_provider: base.LandcoverProvider,
    out_dir: Path,
) -> None:
    global _WORKER_CFG
    global _WORKER_ARGS
    global _WORKER_IMAGE_DIR
    global _WORKER_IMAGE_INDEX
    global _WORKER_LANDCOVER_PROVIDER
    global _WORKER_OUT_DIR

    _WORKER_CFG = cfg
    _WORKER_ARGS = args
    _WORKER_IMAGE_DIR = image_dir
    _WORKER_IMAGE_INDEX = image_index
    _WORKER_LANDCOVER_PROVIDER = landcover_provider
    _WORKER_OUT_DIR = out_dir


def _worker_init() -> None:
    # Prevent native math libraries from creating nested thread pools in every process.
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def _process_chunk(
    chunk: list[tuple[int, dict[str, Any]]],
) -> list[tuple[dict[str, Any] | None, dict[str, Any] | None]]:
    if any(
        value is None
        for value in (
            _WORKER_CFG,
            _WORKER_ARGS,
            _WORKER_IMAGE_DIR,
            _WORKER_IMAGE_INDEX,
            _WORKER_LANDCOVER_PROVIDER,
            _WORKER_OUT_DIR,
        )
    ):
        raise RuntimeError("Multiprocessing worker context was not initialized")

    output = []
    for row_idx, row_data in chunk:
        row = pd.Series(row_data)
        try:
            result = process_row_fast(
                row_idx=row_idx,
                row=row,
                cfg=_WORKER_CFG,
                args=_WORKER_ARGS,
                image_dir=_WORKER_IMAGE_DIR,
                image_index=_WORKER_IMAGE_INDEX,
                landcover_provider=_WORKER_LANDCOVER_PROVIDER,
                out_dir=_WORKER_OUT_DIR,
            )
            output.append((result, None))
        except Exception as exc:
            output.append((None, base.failure_record(row_idx, row, exc)))
    return output


def _row_chunks(
    df: pd.DataFrame, image_col: str, chunk_size: int
) -> Iterator[list[tuple[int, dict[str, Any]]]]:
    # Grouping by source image improves each worker's raster metadata cache locality.
    ordered = df.sort_values(image_col, kind="stable")
    chunk: list[tuple[int, dict[str, Any]]] = []
    for row_idx, row in ordered.iterrows():
        chunk.append((int(row_idx), row.to_dict()))
        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def run_graph_builds_multiprocess(
    df: pd.DataFrame,
    cfg: base.ColumnConfig,
    args: Any,
    image_dir: Path,
    image_index: dict[str, Path],
    landcover_provider: base.LandcoverProvider,
    out_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    workers = int(args.num_workers)
    if workers <= 0:
        LOGGER.info("Running sequential fast graph builder")
        summary = base.new_summary(len(df))
        failures = []
        progress = tqdm(df.iterrows(), total=len(df), desc="Building graphs")
        for row_idx, row in progress:
            try:
                result = process_row_fast(
                    int(row_idx),
                    row,
                    cfg,
                    args,
                    image_dir,
                    image_index,
                    landcover_provider,
                    out_dir,
                )
                base.update_summary(summary, result)
                progress.set_postfix(**base.progress_fields(result))
            except Exception as exc:
                summary["failed"] += 1
                failures.append(base.failure_record(int(row_idx), row, exc))
                LOGGER.exception("Failed sample row_idx=%s: %s", row_idx, exc)
        return summary, failures

    if "fork" not in mp.get_all_start_methods():
        raise RuntimeError(
            "The fast multiprocessing builder requires Linux fork to share "
            "preloaded land-cover frames without pickling them."
        )

    workers = min(workers, len(df), os.cpu_count() or workers)
    _preload_shared_data(
        df, cfg, args, image_dir, image_index, landcover_provider
    )
    _set_worker_context(
        cfg, args, image_dir, image_index, landcover_provider, out_dir
    )

    # Small chunks retain load balancing because graph complexity varies per chip.
    chunk_size = max(1, min(8, len(df) // max(workers * 32, 1)))
    chunks = list(_row_chunks(df, cfg.image_col, chunk_size))
    LOGGER.info(
        "Starting %d fork workers with %d task chunks (chunk_size=%d)",
        workers,
        len(chunks),
        chunk_size,
    )

    summary = base.new_summary(len(df))
    failures: list[dict[str, Any]] = []
    context = mp.get_context("fork")
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=context,
        initializer=_worker_init,
    ) as executor:
        future_sizes = {
            executor.submit(_process_chunk, chunk): len(chunk) for chunk in chunks
        }
        progress = tqdm(total=len(df), desc="Building graphs")
        for future in as_completed(future_sizes):
            try:
                chunk_results = future.result()
            except Exception as exc:
                failed_count = future_sizes[future]
                summary["failed"] += failed_count
                failures.append(
                    {
                        "row_idx": "chunk",
                        "error": f"Worker chunk failed ({failed_count} rows): {exc!r}",
                    }
                )
                LOGGER.exception("Worker chunk failed: %s", exc)
                progress.update(failed_count)
                continue

            for result, failure in chunk_results:
                if failure is not None:
                    summary["failed"] += 1
                    failures.append(failure)
                elif result is not None:
                    base.update_summary(summary, result)
                    progress.set_postfix(**base.progress_fields(result))
                progress.update(1)
        progress.close()

    return summary, failures


base.build_landcover_graph = build_landcover_graph_fast
base.process_row = process_row_fast
base.LandcoverProvider = SourceCachedLandcoverProvider
base.run_graph_builds = run_graph_builds_multiprocess


if __name__ == "__main__":
    base.main()
