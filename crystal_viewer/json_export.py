from __future__ import annotations

import json
from dataclasses import is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .atom_mapping import AtomMappingSet
from .render_data import RenderData
from .source_kinds import normalize_source_kind


EXPORT_SCHEMA_VERSION = 7


def export_payload(
    render_data: RenderData,
    atom_mappings: AtomMappingSet,
    *,
    source_kind: str,
) -> dict[str, Any]:
    source_kind = normalize_source_kind(source_kind)
    return {
        "schema_version": EXPORT_SCHEMA_VERSION,
        "source_kind": source_kind,
        "render_data": render_data_to_dict(render_data),
        "atom_mappings": atom_mapping_set_to_dict(atom_mappings),
    }


def write_export_json(
    path: str | Path,
    render_data: RenderData,
    atom_mappings: AtomMappingSet,
    *,
    source_kind: str,
    indent: int = 2,
) -> None:
    source_kind = normalize_source_kind(source_kind)
    payload = export_payload(render_data, atom_mappings, source_kind=source_kind)
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=indent),
        encoding="utf-8",
    )


def render_data_to_dict(data: RenderData) -> dict[str, Any]:
    return {
        "metadata": to_jsonable(data.metadata),
        "atoms": [to_jsonable(atom) for atom in data.atoms],
        "asymmetric_atoms": [to_jsonable(atom) for atom in data.asymmetric_atoms],
        "operations": [to_jsonable(operation) for operation in data.operations],
        "axes": [to_jsonable(axis) for axis in data.axes],
        "planes": [to_jsonable(plane) for plane in data.planes],
        "centers": [to_jsonable(center) for center in data.centers],
        "unit_cell": to_jsonable(data.unit_cell),
        "bounds_min": to_jsonable(data.bounds_min),
        "bounds_max": to_jsonable(data.bounds_max),
    }


def atom_mapping_set_to_dict(mapping_set: AtomMappingSet) -> dict[str, Any]:
    return {
        "mode": mapping_set.mode,
        "complete": mapping_set.is_complete,
        "incomplete_operation_indices": list(mapping_set.incomplete_operation_indices),
        "mappings": [operation_mapping_to_dict(mapping) for mapping in mapping_set.mappings],
    }


def operation_mapping_to_dict(mapping) -> dict[str, Any]:
    return {
        "mode": mapping.mode,
        "operation_index": mapping.operation_index,
        "operation_kind": mapping.operation_kind,
        "complete": mapping.is_complete,
        "atom_to_atom": list(mapping.atom_to_atom),
        "max_distance": mapping.max_distance,
        "unmatched_atoms": list(mapping.unmatched_atoms),
        "entries": [to_jsonable(entry) for entry in mapping.entries],
    }


def to_jsonable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value):
        return {
            key: to_jsonable(item)
            for key, item in value.__dict__.items()
        }
    if isinstance(value, tuple | list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    return value
