from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any


LARGE_CODES: list[int] = [100, 200, 300, 400, 500, 600, 700]
MIDDLE_CODES: list[int] = [120, 130, 140, 210, 220, 230, 240]
SMALL_CODES: list[int] = [111, 112, 151, 152, 153, 154, 155, 161, 162, 163, 252]
EXCLUDED_SMALL_CODES: set[int] = {251}

CLASS_VOCAB: list[int] = [0] + LARGE_CODES + MIDDLE_CODES + SMALL_CODES
LARGE_VOCAB: list[int] = [0] + LARGE_CODES
MIDDLE_VOCAB: list[int] = [0] + MIDDLE_CODES
SMALL_VOCAB: list[int] = [0] + SMALL_CODES

LEVEL_TO_ID: dict[str, int] = {
    "unknown": 0,
    "large": 1,
    "middle": 2,
    "small": 3,
}

EDGE_TYPE_TO_ID: dict[str, int] = {
    "self": 0,
    "spatial_touches": 1,
    "spatial_overlaps": 2,
    "spatial_covers": 3,
    "spatial_covered_by": 4,
    "spatial_equals": 5,
    "hierarchy_parent_to_child": 6,
    "hierarchy_child_to_parent": 7,
}

GRAPH_SCHEMA_VERSION = "hier_landcover_geolink_rel_v4_multilevel_columns"

CLASS_CODE_TO_ID: dict[int, int] = {code: idx for idx, code in enumerate(CLASS_VOCAB)}
LARGE_CODE_TO_ID: dict[int, int] = {code: idx for idx, code in enumerate(LARGE_VOCAB)}
MIDDLE_CODE_TO_ID: dict[int, int] = {code: idx for idx, code in enumerate(MIDDLE_VOCAB)}
SMALL_CODE_TO_ID: dict[int, int] = {code: idx for idx, code in enumerate(SMALL_VOCAB)}


def code_to_int(value: Any) -> int | None:
    """Parse a land-cover code from common numeric or string forms."""
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if math.isnan(value):
            return None
        return int(value)

    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None

    try:
        return int(float(text))
    except ValueError:
        pass

    match = re.search(r"\d+", text)
    if match is None:
        return None
    return int(match.group(0))


def infer_large_code(code: int) -> int:
    large_code = (int(code) // 100) * 100
    return large_code if large_code in LARGE_CODES else 0


def infer_middle_code_for_small(code: int) -> int:
    middle_code = (int(code) // 10) * 10
    return middle_code if middle_code in MIDDLE_CODES else 0


def coarse_level_from_code(code: int) -> str:
    code = int(code)
    if code in LARGE_CODES or (100 <= code <= 900 and code % 100 == 0):
        return "large"
    if 100 <= code <= 999 and code % 10 == 0:
        return "middle"
    if 100 <= code <= 999:
        return "small"
    return "unknown"


def is_selected_code(level: str, code: int) -> bool:
    code = int(code)
    if level == "large":
        return code in LARGE_CODES
    if level == "middle":
        return code in MIDDLE_CODES
    if level == "small":
        return code in SMALL_CODES and code not in EXCLUDED_SMALL_CODES
    return False


def selected_codes_for_level(level: str) -> list[int]:
    if level == "large":
        return LARGE_CODES.copy()
    if level == "middle":
        return MIDDLE_CODES.copy()
    if level == "small":
        return SMALL_CODES.copy()
    raise ValueError(f"Unknown land-cover level: {level}")


def build_node_code_fields(level: str, class_code: int) -> dict[str, int]:
    class_code = int(class_code)
    if level == "large":
        return {
            "class_code": class_code,
            "large_code": class_code,
            "middle_code": 0,
            "small_code": 0,
            "level_id": LEVEL_TO_ID["large"],
        }
    if level == "middle":
        return {
            "class_code": class_code,
            "large_code": infer_large_code(class_code),
            "middle_code": class_code,
            "small_code": 0,
            "level_id": LEVEL_TO_ID["middle"],
        }
    if level == "small":
        return {
            "class_code": class_code,
            "large_code": infer_large_code(class_code),
            "middle_code": infer_middle_code_for_small(class_code),
            "small_code": class_code,
            "level_id": LEVEL_TO_ID["small"],
        }
    return {
        "class_code": 0,
        "large_code": 0,
        "middle_code": 0,
        "small_code": 0,
        "level_id": LEVEL_TO_ID["unknown"],
    }


def id_from_vocab(code: int, vocab_name: str) -> int:
    code = int(code)
    if vocab_name == "class":
        return CLASS_CODE_TO_ID.get(code, 0)
    if vocab_name == "large":
        return LARGE_CODE_TO_ID.get(code, 0)
    if vocab_name == "middle":
        return MIDDLE_CODE_TO_ID.get(code, 0)
    if vocab_name == "small":
        return SMALL_CODE_TO_ID.get(code, 0)
    raise ValueError(f"Unknown vocab name: {vocab_name}")


def build_node_id_fields(code_fields: dict[str, int]) -> dict[str, int]:
    return {
        "class_id": id_from_vocab(code_fields["class_code"], "class"),
        "large_id": id_from_vocab(code_fields["large_code"], "large"),
        "middle_id": id_from_vocab(code_fields["middle_code"], "middle"),
        "small_id": id_from_vocab(code_fields["small_code"], "small"),
    }


def is_large_parent_of_child(parent_code: int, child_large_code: int) -> bool:
    return int(parent_code) != 0 and int(parent_code) == int(child_large_code)


def is_middle_parent_of_small(parent_code: int, child_middle_code: int) -> bool:
    return int(parent_code) != 0 and int(parent_code) == int(child_middle_code)


def is_ancestor_descendant(parent: dict[str, int], child: dict[str, int]) -> bool:
    parent_level = int(parent["level_id"])
    child_level = int(child["level_id"])
    if parent_level >= child_level:
        return False
    if parent_level == LEVEL_TO_ID["large"] and child_level in {LEVEL_TO_ID["middle"], LEVEL_TO_ID["small"]}:
        return is_large_parent_of_child(parent["class_code"], child["large_code"])
    if parent_level == LEVEL_TO_ID["middle"] and child_level == LEVEL_TO_ID["small"]:
        return is_middle_parent_of_small(parent["class_code"], child["middle_code"])
    return False


def are_ancestor_descendant(a: dict[str, int], b: dict[str, int]) -> bool:
    return is_ancestor_descendant(a, b) or is_ancestor_descendant(b, a)


def vocab_dict() -> dict[str, Any]:
    return {
        "graph_schema_version": GRAPH_SCHEMA_VERSION,
        "class_vocab": CLASS_VOCAB,
        "large_vocab": LARGE_VOCAB,
        "middle_vocab": MIDDLE_VOCAB,
        "small_vocab": SMALL_VOCAB,
        "level_vocab": LEVEL_TO_ID,
        "edge_type_vocab": EDGE_TYPE_TO_ID,
        "class_code_to_id": {str(k): v for k, v in CLASS_CODE_TO_ID.items()},
        "large_code_to_id": {str(k): v for k, v in LARGE_CODE_TO_ID.items()},
        "middle_code_to_id": {str(k): v for k, v in MIDDLE_CODE_TO_ID.items()},
        "small_code_to_id": {str(k): v for k, v in SMALL_CODE_TO_ID.items()},
    }


def write_vocab_json(out_path: str | Path) -> None:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(vocab_dict(), f, ensure_ascii=False, indent=2)
