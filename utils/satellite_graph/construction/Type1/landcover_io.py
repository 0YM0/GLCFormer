from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import pandas as pd
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, box

from class_mapping import code_to_int, coarse_level_from_code


LOGGER = logging.getLogger(__name__)

CODE_COLUMN_CANDIDATES: tuple[str, ...] = (
    "code",
    "CODE",
    "ucode",
    "UCODE",
    "lclu_code",
    "LCLU_CODE",
    "L1_CODE",
    "L2_CODE",
    "L3_CODE",
    "l1_code",
    "l2_code",
    "l3_code",
    "MNUM",
    "mnum",
    "class",
    "class_code",
    "CLS_CD",
    "cls_cd",
    "DN",
    "Value",
    "value",
)

VECTOR_EXTENSIONS: tuple[str, ...] = (".shp", ".gpkg")


@dataclass(frozen=True)
class VectorSource:
    path: Path
    member: str | None = None
    level_hint: str | None = None

    @property
    def display_name(self) -> str:
        if self.member is None:
            return str(self.path)
        return f"{self.path}!{self.member}"

    @property
    def read_uri(self) -> str:
        if self.member is None:
            return str(self.path)
        return f"zip://{self.path}!{self.member}"

    @property
    def vsi_uri(self) -> str:
        if self.member is None:
            return str(self.path)
        return f"/vsizip/{self.path}/{self.member}"


def find_code_column(gdf: gpd.GeoDataFrame, code_col: str | None = None) -> str:
    if code_col:
        if code_col not in gdf.columns:
            raise KeyError(f"Requested code column '{code_col}' not found. Columns: {list(gdf.columns)}")
        return code_col

    for candidate in CODE_COLUMN_CANDIDATES:
        if candidate in gdf.columns:
            return candidate

    lower_map = {str(col).lower(): col for col in gdf.columns}
    for candidate in CODE_COLUMN_CANDIDATES:
        if candidate.lower() in lower_map:
            return str(lower_map[candidate.lower()])

    raise KeyError(f"Could not auto-detect code column. Columns: {list(gdf.columns)}")


def preferred_code_column_for_level(columns: Iterable[str], level: str, code_col: str | None = None) -> str | None:
    if code_col:
        return code_col
    columns_set = {str(column) for column in columns}
    lower_map = {str(column).lower(): str(column) for column in columns}
    preferred = {
        "large": ("L1_CODE", "l1_code"),
        "middle": ("L2_CODE", "l2_code"),
        "small": ("L3_CODE", "l3_code"),
    }.get(level, ())
    for candidate in preferred:
        if candidate in columns_set:
            return candidate
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    return None


def infer_level_from_name(name: str) -> str | None:
    lowered = name.lower()
    if any(token in lowered for token in ("대분류", "large", "l1", "level1", "30m", "30_m")):
        return "large"
    if any(token in lowered for token in ("중분류", "middle", "mid", "l2", "level2", "5m", "5_m")):
        return "middle"
    if any(token in lowered for token in ("세분류", "small", "detail", "l3", "level3", "1m", "1_m")):
        return "small"
    return None


def infer_level_from_code_distribution(gdf: gpd.GeoDataFrame, code_col: str | None = None) -> str | None:
    try:
        col = find_code_column(gdf, code_col)
    except KeyError:
        return None

    codes = [code_to_int(value) for value in gdf[col].dropna().head(10000).tolist()]
    codes = [code for code in codes if code is not None]
    if not codes:
        return None

    counts = {"large": 0, "middle": 0, "small": 0, "unknown": 0}
    for code in codes:
        counts[coarse_level_from_code(code)] += 1
    best_level = max(("large", "middle", "small"), key=lambda level: counts[level])
    if counts[best_level] == 0:
        return None
    return best_level


def zip_dbf_fields(source: VectorSource) -> list[str]:
    if source.member is None:
        return []
    dbf_member = str(Path(source.member).with_suffix(".dbf"))
    try:
        with zipfile.ZipFile(source.path) as zf:
            names = zf.namelist()
            if dbf_member not in names:
                target_name = Path(dbf_member).name
                matches = [name for name in names if Path(name).name == target_name]
                if not matches:
                    return []
                dbf_member = matches[0]
            data = zf.read(dbf_member)
    except Exception:
        return []

    if len(data) < 33:
        return []
    fields: list[str] = []
    pos = 32
    while pos + 32 <= len(data) and data[pos] != 0x0D:
        raw = data[pos : pos + 32]
        name = raw[:11].split(b"\x00", 1)[0].decode("ascii", errors="ignore").strip()
        if name:
            fields.append(name)
        pos += 32
    return fields


def has_multilevel_code_columns(source: VectorSource) -> bool:
    fields = {field.lower() for field in zip_dbf_fields(source)}
    return {"l1_code", "l2_code", "l3_code"}.issubset(fields)


def read_vector_preview(
    source: VectorSource,
    rows: int | None = None,
    bbox: tuple[float, float, float, float] | None = None,
) -> gpd.GeoDataFrame:
    read_kwargs_list: list[dict[str, int]] = []
    if rows is not None:
        read_kwargs_list.append({"rows": rows})
    read_kwargs_list.append({})

    candidates = [source.read_uri, source.vsi_uri]
    if source.member is not None:
        candidates.append(f"zip://{source.path}")

    errors: list[str] = []
    for uri in candidates:
        for read_kwargs in read_kwargs_list:
            try:
                gdf = gpd.read_file(uri, bbox=bbox, **read_kwargs)
                if rows is not None and "rows" not in read_kwargs:
                    return gdf.head(rows).copy()
                return gdf
            except Exception as exc:
                option = "rows" if "rows" in read_kwargs else "full"
                errors.append(f"{uri} ({option}): {exc}")

    if source.member is None:
        raise RuntimeError("; ".join(errors))

    with tempfile.TemporaryDirectory(prefix="landcover_zip_") as tmp_dir:
        extracted = extract_zip_layer_with_ascii_names(source, Path(tmp_dir))
        for read_kwargs in read_kwargs_list:
            try:
                gdf = gpd.read_file(extracted, bbox=bbox, **read_kwargs)
                if rows is not None and "rows" not in read_kwargs:
                    return gdf.head(rows).copy()
                return gdf
            except Exception as exc:
                option = "rows" if "rows" in read_kwargs else "full"
                errors.append(f"{extracted} ({option}): {exc}")
        raise RuntimeError("; ".join(errors))


def extract_zip_layer_with_ascii_names(source: VectorSource, out_dir: Path) -> Path:
    """Extract one shapefile/gpkg layer using ASCII filenames for GDAL compatibility."""
    if source.member is None:
        return source.path

    member_path = Path(source.member)
    suffix = member_path.suffix.lower()
    out_dir.mkdir(parents=True, exist_ok=True)

    extract_dir = out_dir / "raw"
    extract_dir.mkdir(parents=True, exist_ok=True)
    extract_errors: list[str] = []
    for extractor in (extract_all_with_7z, extract_all_with_unzip, extract_all_with_zipfile):
        try:
            extractor(source.path, extract_dir)
            break
        except Exception as exc:
            extract_errors.append(f"{extractor.__name__}: {exc}")
    else:
        raise RuntimeError("; ".join(extract_errors))

    if suffix == ".gpkg":
        extracted_gpkg = find_extracted_member(extract_dir, source.member)
        out_path = out_dir / "layer.gpkg"
        out_path.write_bytes(extracted_gpkg.read_bytes())
        return out_path

    extracted_shp = find_extracted_member(extract_dir, source.member)
    stem = extracted_shp.with_suffix("")
    sidecars = list(stem.parent.glob(f"{stem.name}.*"))
    if not sidecars:
        raise FileNotFoundError(f"No sidecar files found after extracting {source.path} for {source.member}")

    shp_out: Path | None = None
    for sidecar in sidecars:
        ext = sidecar.suffix.lower()
        if not ext:
            continue
        out_path = out_dir / f"layer{ext}"
        out_path.write_bytes(sidecar.read_bytes())
        if ext == ".shp":
            shp_out = out_path

    if shp_out is None:
        raise FileNotFoundError(f"No .shp member found in {source.path} for {source.member}")
    return shp_out


def extract_all_with_7z(zip_path: Path, out_dir: Path) -> None:
    seven_zip = shutil.which("7z") or shutil.which("7za")
    if seven_zip is None:
        raise RuntimeError("7z/7za not found")
    result = subprocess.run(
        [seven_zip, "x", "-y", f"-o{out_dir}", str(zip_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    combined = f"{result.stdout}\n{result.stderr}"
    if result.returncode != 0 or "Everything is Ok" not in combined:
        raise RuntimeError(combined.strip())


def extract_all_with_unzip(zip_path: Path, out_dir: Path) -> None:
    result = subprocess.run(
        ["unzip", "-q", "-o", str(zip_path), "-d", str(out_dir)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"unzip failed for {zip_path}: {result.stderr.strip()}")


def extract_all_with_zipfile(zip_path: Path, out_dir: Path) -> None:
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)


def find_extracted_member(extract_dir: Path, member: str) -> Path:
    exact = extract_dir / member
    if exact.exists():
        return exact

    member_name = Path(member).name
    matches = list(extract_dir.rglob(member_name))
    if matches:
        return matches[0]

    suffix = Path(member).suffix.lower()
    suffix_matches = [path for path in extract_dir.rglob(f"*{suffix}") if path.is_file()]
    if len(suffix_matches) == 1:
        return suffix_matches[0]
    raise FileNotFoundError(f"Could not find extracted member {member} under {extract_dir}")


def discover_landcover_sources(
    landcover_root: str | Path,
    landcover_year: int | str = 2024,
    region_name: str | None = None,
    code_col: str | None = None,
    allow_multilevel_fallback: bool = True,
) -> dict[str, list[VectorSource]]:
    year_dir = Path(landcover_root) / str(landcover_year)
    if not year_dir.exists():
        raise FileNotFoundError(f"Land-cover year directory not found: {year_dir}")

    zip_paths = sorted(year_dir.glob("*.zip"))
    if region_name:
        zip_paths = [path for path in zip_paths if region_name in path.name]
    vector_paths = [
        path
        for path in sorted(year_dir.rglob("*"))
        if path.is_file()
        and path.suffix.lower() in VECTOR_EXTENSIONS
        and not path.name.startswith(".")
        and (region_name is None or region_name in str(path))
    ]
    if not zip_paths and not vector_paths:
        raise FileNotFoundError(f"No zip/shp/gpkg files found under {year_dir} with region_name={region_name!r}")

    by_level: dict[str, list[VectorSource]] = {"large": [], "middle": [], "small": []}
    unknown_sources: list[VectorSource] = []

    for zip_path in zip_paths:
        with zipfile.ZipFile(zip_path) as zf:
            members = [
                member
                for member in zf.namelist()
                if Path(member).suffix.lower() in VECTOR_EXTENSIONS and not Path(member).name.startswith(".")
            ]
        for member in members:
            source = VectorSource(path=zip_path.resolve(), member=member)
            level = infer_level_from_name(f"{zip_path.name}/{member}")
            if level is None:
                unknown_sources.append(source)
            else:
                by_level[level].append(VectorSource(path=source.path, member=source.member, level_hint=level))

    for vector_path in vector_paths:
        source = VectorSource(path=vector_path.resolve())
        level = infer_level_from_name(str(vector_path))
        if level is None:
            unknown_sources.append(source)
        else:
            by_level[level].append(VectorSource(path=source.path, level_hint=level))

    for source in unknown_sources:
        try:
            gdf = read_vector_preview(source, rows=5000)
            if allow_multilevel_fallback and all(
                preferred_code_column_for_level(gdf.columns, level) for level in ("large", "middle", "small")
            ):
                LOGGER.info(
                    "Using multilevel code columns L1_CODE/L2_CODE/L3_CODE from %s for large/middle/small",
                    source.display_name,
                )
                for multilevel in ("large", "middle", "small"):
                    by_level[multilevel].append(
                        VectorSource(path=source.path, member=source.member, level_hint=multilevel)
                    )
                continue
            if all(preferred_code_column_for_level(gdf.columns, level) for level in ("large", "middle", "small")):
                LOGGER.warning(
                    "Skipping multilevel land-cover source %s for automatic level reuse. "
                    "Provide level-specific --landcover_*_path files or pass --allow_multilevel_fallback.",
                    source.display_name,
                )
                continue
            level = infer_level_from_code_distribution(gdf, code_col)
        except Exception as exc:
            if allow_multilevel_fallback and has_multilevel_code_columns(source):
                LOGGER.info(
                    "Using multilevel code columns L1_CODE/L2_CODE/L3_CODE from %s for large/middle/small",
                    source.display_name,
                )
                for multilevel in ("large", "middle", "small"):
                    by_level[multilevel].append(
                        VectorSource(path=source.path, member=source.member, level_hint=multilevel)
                    )
                continue
            if has_multilevel_code_columns(source):
                LOGGER.warning(
                    "Skipping multilevel land-cover source %s for automatic level reuse. "
                    "Provide level-specific --landcover_*_path files or pass --allow_multilevel_fallback.",
                    source.display_name,
                )
                continue
            LOGGER.warning("Could not inspect %s for level auto-detection: %s", source.display_name, exc)
            level = None
        if level in by_level:
            by_level[level].append(VectorSource(path=source.path, member=source.member, level_hint=level))

    missing = [level for level, sources in by_level.items() if not sources]
    if missing:
        LOGGER.warning("No land-cover source auto-detected for levels: %s", ", ".join(missing))

    return by_level


def explicit_sources(
    large_path: str | Path | None,
    middle_path: str | Path | None,
    small_path: str | Path | None,
) -> dict[str, list[VectorSource]]:
    paths = {"large": large_path, "middle": middle_path, "small": small_path}
    sources: dict[str, list[VectorSource]] = {"large": [], "middle": [], "small": []}
    for level, value in paths.items():
        if value is None:
            continue
        path = Path(value).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"{level} land-cover path not found: {path}")
        sources[level].append(VectorSource(path=path, level_hint=level))
    return sources


def _polygonal_part(geom):
    if geom is None or geom.is_empty:
        return None
    if isinstance(geom, (Polygon, MultiPolygon)):
        return geom
    if isinstance(geom, GeometryCollection):
        polygons = [part for part in geom.geoms if isinstance(part, Polygon) and not part.is_empty]
        multipolygons = [part for part in geom.geoms if isinstance(part, MultiPolygon) and not part.is_empty]
        all_polygons: list[Polygon] = polygons.copy()
        for multipolygon in multipolygons:
            all_polygons.extend([part for part in multipolygon.geoms if not part.is_empty])
        if not all_polygons:
            return None
        if len(all_polygons) == 1:
            return all_polygons[0]
        return MultiPolygon(all_polygons)
    return None


def repair_geometry(geom):
    if geom is None or geom.is_empty:
        return None
    fixed = geom
    if not fixed.is_valid:
        try:
            from shapely import make_valid

            fixed = make_valid(fixed)
        except Exception:
            try:
                fixed = fixed.buffer(0)
            except Exception:
                return None
    return _polygonal_part(fixed)


def sanitize_polygon_gdf(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.empty:
        return gdf.copy()
    out = gdf.copy()
    out = out[out.geometry.notna()].copy()
    if out.empty:
        return out
    out.geometry = out.geometry.map(repair_geometry)
    out = out[out.geometry.notna() & ~out.geometry.is_empty].copy()
    if out.empty:
        return out
    out = out[out.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
    if out.empty:
        return out
    out = out.explode(index_parts=False, ignore_index=True)
    out = out[out.geometry.notna() & ~out.geometry.is_empty].copy()
    out = out[out.geometry.geom_type == "Polygon"].copy()
    out.reset_index(drop=True, inplace=True)
    return out


def read_vector_source(
    source: VectorSource,
    code_col: str | None = None,
    level: str | None = None,
    bbox: tuple[float, float, float, float] | None = None,
) -> gpd.GeoDataFrame:
    gdf = read_vector_preview(source, bbox=bbox)
    preferred_col = preferred_code_column_for_level(gdf.columns, level or "", code_col)
    col = find_code_column(gdf, preferred_col)
    gdf = sanitize_polygon_gdf(gdf)
    if gdf.empty:
        gdf["_lc_code"] = pd.Series(dtype="int64")
        gdf["_source"] = source.display_name
        return gdf

    gdf["_lc_code"] = gdf[col].map(code_to_int)
    gdf = gdf[gdf["_lc_code"].notna()].copy()
    gdf["_lc_code"] = gdf["_lc_code"].astype(int)
    gdf["_source"] = source.display_name
    gdf.reset_index(drop=True, inplace=True)
    return gdf


def load_landcover_level(
    sources: Iterable[VectorSource],
    level: str,
    code_col: str | None = None,
    target_codes: Iterable[int] | None = None,
    bbox: tuple[float, float, float, float] | None = None,
) -> gpd.GeoDataFrame:
    frames: list[gpd.GeoDataFrame] = []
    for source in sources:
        LOGGER.debug("Loading %s land-cover source: %s", level, source.display_name)
        try:
            frame = read_vector_source(source, code_col=code_col, level=level, bbox=bbox)
        except Exception as exc:
            LOGGER.warning("Skipping %s land-cover source %s: %s", level, source.display_name, exc)
            continue
        if target_codes is not None and not frame.empty:
            frame = frame[frame["_lc_code"].isin(set(target_codes))].copy()
        frame["_lc_level"] = level
        frames.append(frame)

    if not frames:
        return gpd.GeoDataFrame({"_lc_code": [], "_lc_level": []}, geometry=[], crs=None)

    crs = next((frame.crs for frame in frames if frame.crs is not None), None)
    normalized_frames: list[gpd.GeoDataFrame] = []
    for frame in frames:
        if crs is not None and frame.crs is not None and frame.crs != crs:
            frame = frame.to_crs(crs)
        normalized_frames.append(frame)

    merged = gpd.GeoDataFrame(pd.concat(normalized_frames, ignore_index=True), geometry="geometry", crs=crs)
    merged = sanitize_polygon_gdf(merged)
    if not merged.empty:
        merged["_lc_code"] = merged["_lc_code"].astype(int)
        merged["_lc_level"] = level
    return merged


def dbf_path_for_source(source: VectorSource) -> Path | None:
    if source.member is not None:
        return None
    dbf_path = source.path.with_suffix(".dbf")
    return dbf_path if dbf_path.exists() else None


def read_dbf_unique_values(source: VectorSource, field_name: str, max_values: int | None = None) -> set[str]:
    dbf_path = dbf_path_for_source(source)
    if dbf_path is None:
        return set()

    values: set[str] = set()
    with dbf_path.open("rb") as f:
        header = f.read(32)
        if len(header) < 32:
            return values
        num_records = int.from_bytes(header[4:8], "little")
        header_len = int.from_bytes(header[8:10], "little")
        record_len = int.from_bytes(header[10:12], "little")

        fields: list[tuple[str, int, int]] = []
        offset = 1
        while True:
            raw = f.read(32)
            if not raw or raw[0] == 0x0D:
                break
            name = raw[:11].split(b"\x00", 1)[0].decode("ascii", errors="ignore").strip()
            length = int(raw[16])
            fields.append((name, offset, length))
            offset += length

        target = next((field for field in fields if field[0].lower() == field_name.lower()), None)
        if target is None:
            return values
        _, target_offset, target_length = target

        f.seek(header_len)
        for _ in range(num_records):
            record = f.read(record_len)
            if len(record) < record_len:
                break
            if record[:1] == b"*":
                continue
            value = record[target_offset : target_offset + target_length].decode("ascii", errors="ignore").strip()
            if value:
                values.add(value)
            if max_values is not None and len(values) >= max_values:
                break
    return values


def reproject_if_needed(gdf: gpd.GeoDataFrame, target_crs) -> gpd.GeoDataFrame:
    if gdf.empty or target_crs is None or gdf.crs is None or gdf.crs == target_crs:
        return gdf
    return gdf.to_crs(target_crs)


def _sindex_query(gdf: gpd.GeoDataFrame, geom) -> list[int]:
    if gdf.empty:
        return []
    try:
        return list(gdf.sindex.query(geom, predicate="intersects"))
    except TypeError:
        candidates = list(gdf.sindex.query(geom))
        return [idx for idx in candidates if gdf.geometry.iloc[idx].intersects(geom)]


def clip_landcover_to_bounds(
    gdf: gpd.GeoDataFrame,
    bounds: tuple[float, float, float, float],
    min_area_abs: float = 1.0,
    min_area_ratio: float = 0.0,
) -> gpd.GeoDataFrame:
    if gdf.empty:
        return gdf.copy()

    chip_geom = box(*bounds)
    chip_area = max(chip_geom.area, 1e-12)
    candidate_idx = _sindex_query(gdf, chip_geom)
    if not candidate_idx:
        return gpd.GeoDataFrame(gdf.iloc[[]].copy(), geometry="geometry", crs=gdf.crs)

    candidates = gdf.iloc[candidate_idx].copy()
    original_geometry = candidates.geometry.copy()
    original_area = original_geometry.area.clip(lower=1e-12)
    clipped_geometry = original_geometry.intersection(chip_geom)

    candidates["_orig_area"] = original_area.to_numpy()
    candidates.geometry = clipped_geometry
    candidates = sanitize_polygon_gdf(candidates)
    if candidates.empty:
        return candidates

    clipped_area = candidates.geometry.area
    candidates["_clipped_area_ratio"] = (clipped_area / candidates["_orig_area"].clip(lower=1e-12)).clip(0.0, 1.0)

    if min_area_abs > 0:
        candidates = candidates[clipped_area >= float(min_area_abs)].copy()
        clipped_area = candidates.geometry.area
    if min_area_ratio > 0:
        candidates = candidates[(clipped_area / chip_area) >= float(min_area_ratio)].copy()

    candidates.reset_index(drop=True, inplace=True)
    return candidates
