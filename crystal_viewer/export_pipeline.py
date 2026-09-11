from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .atom_mapping import atom_mappings_from_analysis
from .json_export import export_payload, write_export_json
from .molecule_analysis import analyze_molecule_file
from .render_data import render_data_from_analysis
from .source_kinds import SOURCE_KIND_CRYSTAL, SOURCE_KIND_MOLECULE
from .structure_analysis import analyze_cif


DEFAULT_JSON_EXPORT_DIR = Path("exports/json")


def slug_from_path(path: str | Path) -> str:
    stem = Path(path).stem.lower()
    slug = re.sub(r"[^a-z0-9]+", "_", stem).strip("_")
    return slug or "structure"


def default_json_output_path(
    source_path: str | Path,
    output_dir: str | Path = DEFAULT_JSON_EXPORT_DIR,
) -> Path:
    return Path(output_dir) / f"{slug_from_path(source_path)}.json"


def analyze_for_export(path: str | Path, *, mode: str):
    if mode == SOURCE_KIND_CRYSTAL:
        return analyze_cif(path)
    if mode == SOURCE_KIND_MOLECULE:
        return analyze_molecule_file(path)
    raise ValueError(f"unsupported export mode: {mode}")


def build_export_payload(
    path: str | Path,
    *,
    mode: str,
    tolerance_cart: float = 1e-2,
) -> dict[str, Any]:
    analysis = analyze_for_export(path, mode=mode)
    render_data = render_data_from_analysis(analysis)
    atom_mappings = atom_mappings_from_analysis(analysis, tolerance_cart=tolerance_cart)
    return export_payload(render_data, atom_mappings, source_kind=mode)


def export_analysis_to_json(
    path: str | Path,
    *,
    mode: str,
    output_path: str | Path | None = None,
    output_dir: str | Path = DEFAULT_JSON_EXPORT_DIR,
    tolerance_cart: float = 1e-2,
    indent: int = 2,
) -> Path:
    analysis = analyze_for_export(path, mode=mode)
    render_data = render_data_from_analysis(analysis)
    atom_mappings = atom_mappings_from_analysis(analysis, tolerance_cart=tolerance_cart)
    resolved_output = (
        Path(output_path)
        if output_path is not None
        else default_json_output_path(path, output_dir)
    )
    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    write_export_json(
        resolved_output,
        render_data,
        atom_mappings,
        source_kind=mode,
        indent=indent,
    )
    return resolved_output
