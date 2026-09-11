from __future__ import annotations

import json
import re
from fractions import Fraction
from functools import lru_cache
from importlib.resources import files
from typing import Any

import numpy as np


ITC_OPERATION_DATA = "data/itc_operations.json"


def itc_coordinate_summaries(render_data: dict) -> dict[int, str]:
    """Return ITC table xyz strings matched to the current render operations.

    Matching is by exact (W, t) so it does not depend on operation ordering.
    """
    number = space_group_number_from_metadata(render_data)
    if number is None:
        return {}
    operation_table = itc_operation_table(number)
    if not operation_table:
        return {}

    summaries: dict[int, str] = {}
    for operation in render_data.get("operations", []):
        W = operation.get("matrix_frac")
        t = operation.get("translation_frac")
        if W is None or t is None:
            continue
        xyz = operation_table.get(operation_match_key(W, t))
        if xyz is not None:
            summaries[int(operation["index"])] = xyz
    return summaries


def space_group_number_from_metadata(render_data: dict) -> int | None:
    metadata = render_data.get("metadata", {})
    label = str(metadata.get("symmetry_label", ""))
    match = re.match(r"\s*(\d{1,3})\b", label)
    if match is None:
        return None
    number = int(match.group(1))
    return number if 1 <= number <= 230 else None


@lru_cache(maxsize=1)
def load_itc_operation_data() -> dict[str, Any]:
    path = files("crystal_viewer").joinpath(ITC_OPERATION_DATA)
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=230)
def itc_operation_table(space_group_number: int) -> dict[tuple, str]:
    data = load_itc_operation_data()
    group = data.get("space_groups", {}).get(str(space_group_number))
    if not group:
        return {}
    table: dict[tuple, str] = {}
    for item in group.get("operations", []):
        table[operation_match_key(item["W"], item["t"])] = str(item["xyz"])
    return table


def operation_match_key(W: np.ndarray | list, t: np.ndarray | list) -> tuple:
    matrix = tuple(np.rint(np.asarray(W, dtype=float)).astype(int).reshape(-1).tolist())
    translation = tuple(translation_match_component(value) for value in np.asarray(t, dtype=float).reshape(3))
    return matrix, translation


def translation_match_component(value: float) -> tuple[int, int]:
    normalized = float(value) % 1.0
    if abs(normalized) < 1e-8 or abs(normalized - 1.0) < 1e-8:
        normalized = 0.0
    fraction = Fraction(normalized).limit_denominator(24)
    if fraction == 1:
        fraction = Fraction(0)
    return fraction.numerator, fraction.denominator
