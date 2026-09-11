from __future__ import annotations

from copy import deepcopy

import numpy as np
from pymatgen.core import Lattice, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from crystal_viewer.atom_mapping import atom_mappings_from_analysis
from crystal_viewer.json_export import export_payload
from crystal_viewer.render_data import render_data_from_analysis
from crystal_viewer.source_kinds import SOURCE_KIND_CRYSTAL, normalize_source_kind
from crystal_viewer.structure_analysis import analyze_structure


CELL_SETTING_NATIVE = "native"
CELL_SETTING_PRIMITIVE = "primitive"
CELL_SETTING_CONVENTIONAL = "conventional"
CELL_SETTING_REFINED = "refined"
CELL_SETTING_MODES = {
    CELL_SETTING_NATIVE,
    CELL_SETTING_PRIMITIVE,
    CELL_SETTING_CONVENTIONAL,
    CELL_SETTING_REFINED,
}


def standardized_payload(
    payload: dict,
    cell_setting: str,
    *,
    symprec: float = 1e-3,
    angle_tolerance: float = 5.0,
    tolerance_cart: float = 1e-2,
    require_distinct: bool = False,
) -> dict | None:
    """Return a full crystal payload reanalyzed in the requested cell setting."""
    normalized = _normalized_cell_setting(cell_setting)
    source_kind = normalize_source_kind(
        payload.get(
            "source_kind",
            (payload.get("render_data") or {}).get("metadata", {}).get("mode", SOURCE_KIND_CRYSTAL),
        )
    )
    if source_kind != SOURCE_KIND_CRYSTAL:
        return None

    if normalized == CELL_SETTING_NATIVE:
        converted = deepcopy(payload)
        render_data = converted.get("render_data") or {}
        metadata = dict(render_data.get("metadata") or {})
        metadata["display_cell_setting"] = CELL_SETTING_NATIVE
        metadata["display_atom_count"] = len(render_data.get("atoms") or [])
        lattice = (render_data.get("unit_cell") or {}).get("lattice")
        if lattice is not None:
            metadata["display_lattice_parameters"] = _lattice_parameters(np.asarray(lattice, dtype=float))
        converted["render_data"]["metadata"] = metadata
        return converted

    render_data = payload.get("render_data") or {}
    structure = standardized_structure_from_render_data(
        render_data,
        normalized,
        symprec=symprec,
        angle_tolerance=angle_tolerance,
    )
    if structure is None:
        return None
    if require_distinct and normalized == CELL_SETTING_PRIMITIVE and not _reduces_atom_count(render_data, structure):
        return None
    if require_distinct and normalized == CELL_SETTING_CONVENTIONAL and not _increases_atom_count(render_data, structure):
        return None
    if (
        require_distinct
        and normalized not in (CELL_SETTING_PRIMITIVE, CELL_SETTING_CONVENTIONAL)
        and not _is_distinct_cell(render_data, structure)
    ):
        return None

    source_file = (render_data.get("metadata") or {}).get("source_file") or "standardized_structure"
    warnings = tuple(str(item) for item in (render_data.get("metadata") or {}).get("warnings") or [])
    analysis = analyze_structure(
        structure,
        source_file=source_file,
        warnings=warnings,
        symprec=symprec,
        angle_tolerance=angle_tolerance,
    )
    converted_render_data = render_data_from_analysis(analysis)
    atom_mappings = atom_mappings_from_analysis(analysis, tolerance_cart=tolerance_cart)
    converted = export_payload(
        converted_render_data,
        atom_mappings,
        source_kind=SOURCE_KIND_CRYSTAL,
    )
    metadata = converted["render_data"]["metadata"]
    metadata["display_cell_setting"] = normalized
    metadata["display_lattice_parameters"] = _lattice_parameters(np.asarray(structure.lattice.matrix, dtype=float))
    metadata["display_atom_count"] = len(converted["render_data"]["atoms"])
    return converted


def standardized_structure_from_render_data(
    render_data: dict,
    cell_setting: str,
    *,
    symprec: float = 1e-3,
    angle_tolerance: float = 5.0,
) -> Structure | None:
    """Return a pymatgen Structure in a standardized cell setting."""
    normalized = _normalized_cell_setting(cell_setting)
    if normalized == CELL_SETTING_NATIVE:
        return _structure_from_render_data(render_data)

    unit_cell = render_data.get("unit_cell")
    atoms = render_data.get("atoms") or []
    if unit_cell is None or not atoms:
        return None

    try:
        structure = _structure_from_render_data(render_data)
        analyzer = SpacegroupAnalyzer(
            structure,
            symprec=symprec,
            angle_tolerance=angle_tolerance,
        )
        return _standardized_structure(analyzer, normalized)
    except Exception:
        return None


def standardized_cell_render_data(
    render_data: dict,
    cell_setting: str,
    *,
    symprec: float = 1e-3,
    angle_tolerance: float = 5.0,
) -> dict | None:
    """Return display-only render data in a pymatgen-standardized cell setting.

    This is intentionally independent from the viewer wiring. It transforms the
    lattice and displayed atom positions only; operation mappings and symmetry
    element annotations still need a separate mapping layer before this becomes
    an interactive UI mode.
    """
    normalized = _normalized_cell_setting(cell_setting)
    if normalized == CELL_SETTING_NATIVE:
        converted = deepcopy(render_data)
        metadata = dict(converted.get("metadata") or {})
        metadata["display_cell_setting"] = CELL_SETTING_NATIVE
        metadata["display_atom_count"] = len(converted.get("atoms") or [])
        lattice = (converted.get("unit_cell") or {}).get("lattice")
        if lattice is not None:
            metadata["display_lattice_parameters"] = _lattice_parameters(np.asarray(lattice, dtype=float))
        converted["metadata"] = metadata
        return converted

    standardized = standardized_structure_from_render_data(
        render_data,
        normalized,
        symprec=symprec,
        angle_tolerance=angle_tolerance,
    )
    if standardized is None:
        return None

    converted = _render_data_from_structure(
        render_data,
        standardized,
        display_cell_setting=normalized,
    )
    return converted


def hexagonal_conventional_render_data(render_data: dict, *, symprec: float = 1e-3) -> dict | None:
    """Return display-only hexagonal conventional render data for an R lattice.

    This intentionally does not transform operation mappings or symmetry elements.
    It is a small, isolated conversion layer that can be wired into the viewer after
    the visual behavior is validated.
    """
    converted = standardized_cell_render_data(
        render_data,
        CELL_SETTING_CONVENTIONAL,
        symprec=symprec,
    )
    if converted is None:
        return None

    if not _is_hexagonal_conventional_r_lattice(_structure_from_render_data(converted)):
        return None

    metadata = dict(converted.get("metadata") or {})
    metadata["display_cell_setting"] = "hexagonal_conventional"
    converted["metadata"] = metadata
    return converted


def _normalized_cell_setting(cell_setting: str) -> str:
    value = str(cell_setting or "").strip().lower().replace("-", "_")
    aliases = {
        "as_loaded": CELL_SETTING_NATIVE,
        "original": CELL_SETTING_NATIVE,
        "source": CELL_SETTING_NATIVE,
        "standard_primitive": CELL_SETTING_PRIMITIVE,
        "primitive_standard": CELL_SETTING_PRIMITIVE,
        "standard_conventional": CELL_SETTING_CONVENTIONAL,
        "conventional_standard": CELL_SETTING_CONVENTIONAL,
        "refined_structure": CELL_SETTING_REFINED,
    }
    value = aliases.get(value, value)
    if value not in CELL_SETTING_MODES:
        modes = ", ".join(sorted(CELL_SETTING_MODES))
        raise ValueError(f"Unknown cell setting {cell_setting!r}; expected one of: {modes}")
    return value


def _standardized_structure(analyzer: SpacegroupAnalyzer, cell_setting: str) -> Structure:
    if cell_setting == CELL_SETTING_PRIMITIVE:
        return analyzer.get_primitive_standard_structure(keep_site_properties=True)
    if cell_setting == CELL_SETTING_CONVENTIONAL:
        return analyzer.get_conventional_standard_structure(keep_site_properties=True)
    if cell_setting == CELL_SETTING_REFINED:
        return analyzer.get_refined_structure(keep_site_properties=True)
    raise ValueError(f"Unsupported standardized cell setting: {cell_setting}")


def _is_distinct_cell(render_data: dict, structure: Structure) -> bool:
    atoms = render_data.get("atoms") or []
    if len(structure) != len(atoms):
        return True
    unit_cell = render_data.get("unit_cell") or {}
    lattice = unit_cell.get("lattice")
    if lattice is None:
        return True
    old_lattice = np.asarray(lattice, dtype=float)
    new_lattice = np.asarray(structure.lattice.matrix, dtype=float)
    if old_lattice.shape != new_lattice.shape:
        return True
    return not np.allclose(old_lattice, new_lattice, atol=1e-6)


def _reduces_atom_count(render_data: dict, structure: Structure) -> bool:
    return len(structure) < len(render_data.get("atoms") or [])


def _increases_atom_count(render_data: dict, structure: Structure) -> bool:
    return len(structure) > len(render_data.get("atoms") or [])


def _render_data_from_structure(
    render_data: dict,
    structure: Structure,
    *,
    display_cell_setting: str,
) -> dict:
    converted = deepcopy(render_data)
    lattice = np.asarray(structure.lattice.matrix, dtype=float)
    converted["unit_cell"] = _unit_cell_from_lattice(lattice)
    converted["atoms"] = _atoms_from_structure(structure)
    converted["bounds_min"], converted["bounds_max"] = _bounds_from_atoms_and_cell(converted)
    metadata = dict(converted.get("metadata") or {})
    metadata["display_cell_setting"] = display_cell_setting
    metadata["display_lattice_parameters"] = _lattice_parameters(lattice)
    metadata["display_atom_count"] = len(converted["atoms"])
    converted["metadata"] = metadata
    return converted


def _structure_from_render_data(render_data: dict) -> Structure:
    unit_cell = render_data["unit_cell"]
    atoms = render_data["atoms"]
    lattice = Lattice(np.asarray(unit_cell["lattice"], dtype=float))
    species = [str(atom.get("element") or atom.get("label") or atom.get("index")) for atom in atoms]
    frac_coords = [atom.get("frac") for atom in atoms]
    source_indices = [int(atom.get("index", index)) for index, atom in enumerate(atoms)]
    asymmetric_indices = [atom.get("asymmetric_index") for atom in atoms]
    generation_indices = [atom.get("generation_operation_index") for atom in atoms]
    site_properties = {"source_index": source_indices}
    if all(index is not None for index in asymmetric_indices):
        site_properties["asymmetric_index"] = asymmetric_indices
    if all(index is not None for index in generation_indices):
        site_properties["generation_operation_index"] = generation_indices
    return Structure(
        lattice,
        species,
        frac_coords,
        coords_are_cartesian=False,
        to_unit_cell=True,
        site_properties=site_properties,
    )


def _is_hexagonal_conventional_r_lattice(structure: Structure) -> bool:
    symbol = str(SpacegroupAnalyzer(structure, symprec=1e-3).get_space_group_symbol())
    if not symbol.startswith("R"):
        return False
    alpha, beta, gamma = structure.lattice.angles
    return (
        abs(alpha - 90.0) < 1e-5
        and abs(beta - 90.0) < 1e-5
        and abs(gamma - 120.0) < 1e-5
    )


def _atoms_from_structure(structure: Structure) -> list[dict]:
    source_indices = structure.site_properties.get("source_index", [None] * len(structure))
    asymmetric_indices = structure.site_properties.get("asymmetric_index", [None] * len(structure))
    generation_indices = structure.site_properties.get("generation_operation_index", [None] * len(structure))
    atoms = []
    for index, site in enumerate(structure):
        element = str(site.specie.symbol)
        atom = {
            "index": index,
            "element": element,
            "atomic_number": int(site.specie.Z),
            "cart": np.asarray(site.coords, dtype=float).tolist(),
            "frac": np.asarray(site.frac_coords, dtype=float).tolist(),
            "asymmetric_index": asymmetric_indices[index],
            "generation_operation_index": generation_indices[index],
            "source_atom_index": source_indices[index],
        }
        atoms.append(atom)
    return atoms


def _unit_cell_from_lattice(lattice: np.ndarray) -> dict:
    lattice = np.asarray(lattice, dtype=float)
    a, b, c = lattice[0], lattice[1], lattice[2]
    vertices = np.asarray(
        [
            [0.0, 0.0, 0.0],
            a,
            b,
            c,
            a + b,
            a + c,
            b + c,
            a + b + c,
        ],
        dtype=float,
    )
    edges = (
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 4),
        (1, 5),
        (2, 4),
        (2, 6),
        (3, 5),
        (3, 6),
        (4, 7),
        (5, 7),
        (6, 7),
    )
    return {
        "lattice": lattice.tolist(),
        "vertices_cart": vertices.tolist(),
        "edges": [list(edge) for edge in edges],
    }


def _bounds_from_atoms_and_cell(render_data: dict) -> tuple[list[float], list[float]]:
    parts = []
    atoms = render_data.get("atoms") or []
    if atoms:
        parts.append(np.asarray([atom["cart"] for atom in atoms], dtype=float))
    unit_cell = render_data.get("unit_cell")
    if unit_cell is not None:
        parts.append(np.asarray(unit_cell["vertices_cart"], dtype=float))
    if not parts:
        zero = [0.0, 0.0, 0.0]
        return zero, zero
    points = np.vstack(parts)
    return points.min(axis=0).tolist(), points.max(axis=0).tolist()


def _lattice_parameters(lattice: np.ndarray) -> dict[str, float]:
    vectors = np.asarray(lattice, dtype=float)
    lengths = np.linalg.norm(vectors, axis=1)
    return {
        "a": float(lengths[0]),
        "b": float(lengths[1]),
        "c": float(lengths[2]),
        "alpha": _angle(vectors[1], vectors[2]),
        "beta": _angle(vectors[0], vectors[2]),
        "gamma": _angle(vectors[0], vectors[1]),
    }


def _angle(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom <= 1e-12:
        return 0.0
    value = float(np.clip(np.dot(a, b) / denom, -1.0, 1.0))
    return float(np.degrees(np.arccos(value)))
