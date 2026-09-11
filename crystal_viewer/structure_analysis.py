from __future__ import annotations

import importlib.util
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import numpy as np
import spglib
from pymatgen.core import Element, Lattice, Structure
from pymatgen.core.operations import SymmOp

from .analysis_models import (
    AsymmetricUnitSite,
    AtomSite,
    AxisElement,
    CenterElement,
    PlaneElement,
    SpaceGroupInfo,
    StructureAnalysisResult,
    StructureSummary,
    SymmetryOperationInfo,
)


LEGACY_CORE_ENV_VAR = "SYMMETRY_VIEW_LEGACY_CORE"
DEFAULT_LEGACY_CORE = Path(__file__).resolve().parent / "legacy" / "symmetry_core.py"

RHOMBOHEDRAL_SETTING_OPS: dict[int, tuple[str, ...]] = {
    146: ("x,y,z", "z,x,y", "y,z,x"),
    148: ("x,y,z", "z,x,y", "y,z,x", "-x,-y,-z", "-z,-x,-y", "-y,-z,-x"),
    155: ("x,y,z", "z,x,y", "y,z,x", "-y,-x,-z", "-x,-z,-y", "-z,-y,-x"),
    160: ("x,y,z", "z,x,y", "y,z,x", "y,x,z", "z,y,x", "x,z,y"),
    161: (
        "x,y,z",
        "z,x,y",
        "y,z,x",
        "y+1/2,x+1/2,z+1/2",
        "z+1/2,y+1/2,x+1/2",
        "x+1/2,z+1/2,y+1/2",
    ),
    166: (
        "x,y,z",
        "z,x,y",
        "y,z,x",
        "-y,-x,-z",
        "-x,-z,-y",
        "-z,-y,-x",
        "y,x,z",
        "x,z,y",
        "z,y,x",
        "-y,-z,-x",
        "-z,-x,-y",
        "-x,-y,-z",
    ),
    167: (
        "x,y,z",
        "z,x,y",
        "y,z,x",
        "y+1/2,x+1/2,z+1/2",
        "z+1/2,y+1/2,x+1/2",
        "x+1/2,z+1/2,y+1/2",
        "-x,-y,-z",
        "-z,-x,-y",
        "-y,-z,-x",
        "-y+1/2,-x+1/2,-z+1/2",
        "-z+1/2,-y+1/2,-x+1/2",
        "-x+1/2,-z+1/2,-y+1/2",
    ),
}


class AnalysisError(RuntimeError):
    pass


@dataclass(frozen=True)
class CifStructureLoad:
    structure: Structure
    asymmetric_atoms: tuple[AsymmetricUnitSite, ...] | None
    warnings: tuple[str, ...] = ()


def analyze_cif(
    cif_path: str | Path,
    *,
    symprec: float = 1e-3,
    angle_tolerance: float = 5.0,
    search_range: int = 2,
    legacy_core_path: str | Path | None = None,
) -> StructureAnalysisResult:
    cif_path = Path(cif_path)
    if not cif_path.exists():
        raise AnalysisError(f"CIF file not found: {cif_path}")

    loaded = load_structure_from_cif(cif_path)
    return analyze_structure(
        loaded.structure,
        source_file=cif_path,
        asymmetric_atoms=loaded.asymmetric_atoms or (),
        warnings=loaded.warnings,
        symprec=symprec,
        angle_tolerance=angle_tolerance,
        search_range=search_range,
        legacy_core_path=legacy_core_path,
    )


def analyze_structure(
    structure: Structure,
    *,
    source_file: str | Path,
    asymmetric_atoms: tuple[AsymmetricUnitSite, ...] = (),
    warnings: tuple[str, ...] = (),
    symprec: float = 1e-3,
    angle_tolerance: float = 5.0,
    search_range: int = 2,
    legacy_core_path: str | Path | None = None,
) -> StructureAnalysisResult:
    core = load_legacy_core(legacy_core_path)
    source_file = Path(source_file)
    cell = structure_to_spglib_cell(structure)
    dataset = spglib.get_symmetry_dataset(
        cell,
        symprec=symprec,
        angle_tolerance=angle_tolerance,
    )
    if dataset is None:
        raise AnalysisError("spglib could not determine symmetry for this structure.")

    rotations = np.asarray(dataset_value(dataset, "rotations"), dtype=int)
    translations = np.asarray(dataset_value(dataset, "translations"), dtype=float)
    equivalent_atoms = np.asarray(dataset_value(dataset, "equivalent_atoms"), dtype=int)
    merged, per_operation = core.collect_merged_elements(
        rotations,
        translations,
        search_range=search_range,
    )
    geometry_groups = core.group_elements_by_geometry(merged, rotations)

    operations = tuple(
        convert_operation(core, index, W, t, search_range)
        for index, (W, t) in enumerate(zip(rotations, translations))
    )

    return StructureAnalysisResult(
        structure=convert_structure(
            source_file,
            structure,
            asymmetric_atoms=asymmetric_atoms,
            rotations=rotations,
            translations=translations,
            equivalent_atoms=equivalent_atoms,
        ),
        space_group=convert_space_group(dataset),
        operations=operations,
        axes=tuple(convert_axis(axis) for axis in merged["axes"]),
        planes=tuple(convert_plane(plane) for plane in merged["planes"]),
        centers=tuple(convert_center(center) for center in merged["centers"]),
        geometry_groups=geometry_groups,
        raw_merged=merged,
        raw_per_operation=per_operation,
        warnings=warnings,
    )


def load_structure_from_cif(cif_path: Path) -> CifStructureLoad:
    try:
        return CifStructureLoad(
            structure=Structure.from_file(str(cif_path)),
            asymmetric_atoms=None,
        )
    except Exception as exc:
        text = cif_path.read_text(encoding="utf-8", errors="replace")
        if not has_empty_symmetry_equiv_loop(text):
            raise
        cell = parse_cif_cell_parameters(text)
        if not is_rhombohedral_primitive_cell(cell):
            raise
        space_group_number = parse_cif_space_group_number(text)
        if space_group_number not in RHOMBOHEDRAL_SETTING_OPS:
            raise AnalysisError(
                "CIF has an empty symmetry operation loop in a rhombohedral primitive cell, "
                f"but space-group number {space_group_number or 'unknown'} is not supported for fallback repair."
            ) from exc
        structure, asymmetric_atoms = structure_from_declared_rhombohedral_symmetry(cif_path, text, cell, space_group_number)
        warning = (
            "CIF contains an empty _symmetry_equiv_pos_as_xyz loop. "
            "Because the lattice is a rhombohedral primitive cell, the loader used the CIF-declared "
            f"space-group number {space_group_number} only to expand atom_site positions with R-setting operations. "
            "The final reported symmetry is still determined from the expanded atomic positions by spglib."
        )
        return CifStructureLoad(
            structure=structure,
            asymmetric_atoms=asymmetric_atoms,
            warnings=(warning, f"Original CIF parser error: {type(exc).__name__}: {exc}"),
        )


def has_empty_symmetry_equiv_loop(text: str) -> bool:
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line.strip() != "loop_":
            continue
        next_index = next_nonempty_line(lines, index + 1)
        if next_index is None or lines[next_index].strip() not in {
            "_symmetry_equiv_pos_as_xyz",
            "_symmetry_equiv_pos_as_xyz_",
            "_space_group_symop_operation_xyz",
            "_space_group_symop_operation_xyz_",
        }:
            continue
        data_index = next_nonempty_line(lines, next_index + 1)
        if data_index is None:
            return True
        token = lines[data_index].strip()
        if token == "loop_" or token.startswith("_"):
            return True
    return False


def next_nonempty_line(lines: list[str], start: int) -> int | None:
    for index in range(start, len(lines)):
        stripped = lines[index].strip()
        if stripped and not stripped.startswith("#"):
            return index
    return None


def structure_from_declared_rhombohedral_symmetry(
    cif_path: Path,
    text: str,
    cell: dict[str, float],
    space_group_number: int,
) -> tuple[Structure, tuple[AsymmetricUnitSite, ...]]:
    labels, symbols, frac_coords = parse_atom_site_loop(text)
    if not labels:
        raise AnalysisError(f"CIF atom_site loop not found or empty: {cif_path}")
    lattice = np.asarray(
        Lattice.from_parameters(
            cell["a"],
            cell["b"],
            cell["c"],
            cell["alpha"],
            cell["beta"],
            cell["gamma"],
        ).matrix,
        dtype=float,
    )
    elements = [normalize_element_symbol(symbol, label) for label, symbol in zip(labels, symbols)]
    asymmetric_atoms = tuple(
        AsymmetricUnitSite(
            index=index,
            label=str(label),
            element=element,
            atomic_number=int(Element(element).Z),
            frac=np.asarray(frac, dtype=float),
            cart=np.asarray(frac, dtype=float) @ lattice,
        )
        for index, (label, element, frac) in enumerate(zip(labels, elements, frac_coords))
    )

    operations = tuple(SymmOp.from_xyz_str(text) for text in RHOMBOHEDRAL_SETTING_OPS[space_group_number])
    expanded_elements: list[str] = []
    expanded_frac: list[np.ndarray] = []
    for element, frac in zip(elements, frac_coords):
        for operation in operations:
            candidate = wrap_fractional(operation.operate(frac))
            if not has_matching_periodic_site(expanded_elements, expanded_frac, element, candidate):
                expanded_elements.append(element)
                expanded_frac.append(candidate)

    structure = Structure(
        lattice,
        expanded_elements,
        np.asarray(expanded_frac, dtype=float),
        coords_are_cartesian=False,
        to_unit_cell=False,
    )
    return structure, asymmetric_atoms


def wrap_fractional(frac: np.ndarray) -> np.ndarray:
    wrapped = np.asarray(frac, dtype=float) - np.floor(np.asarray(frac, dtype=float))
    wrapped[np.isclose(wrapped, 1.0, atol=1e-8)] = 0.0
    return wrapped


def has_matching_periodic_site(
    elements: list[str],
    positions: list[np.ndarray],
    element: str,
    frac: np.ndarray,
    *,
    tolerance: float = 1e-6,
) -> bool:
    for existing_element, existing_frac in zip(elements, positions):
        if existing_element != element:
            continue
        delta = np.asarray(existing_frac, dtype=float) - np.asarray(frac, dtype=float)
        delta -= np.round(delta)
        if np.linalg.norm(delta) <= tolerance:
            return True
    return False


def is_rhombohedral_primitive_cell(cell: dict[str, float]) -> bool:
    lengths = np.asarray([cell["a"], cell["b"], cell["c"]], dtype=float)
    angles = np.asarray([cell["alpha"], cell["beta"], cell["gamma"]], dtype=float)
    length_scale = max(float(np.max(np.abs(lengths))), 1.0)
    return bool(
        np.max(np.abs(lengths - lengths[0])) <= length_scale * 1e-4
        and np.max(np.abs(angles - angles[0])) <= 1e-3
        and not np.isclose(angles[0], 90.0, atol=1e-4)
        and not np.isclose(angles[0], 120.0, atol=1e-4)
    )


def parse_cif_space_group_number(text: str) -> int | None:
    for key in ("_space_group_IT_number", "_symmetry_Int_Tables_number"):
        match = re.search(rf"^\s*{re.escape(key)}\s+([0-9]+)", text, flags=re.MULTILINE)
        if match:
            return int(match.group(1))
    return None


def parse_cif_cell_parameters(text: str) -> dict[str, float]:
    keys = {
        "_cell_length_a": "a",
        "_cell_length_b": "b",
        "_cell_length_c": "c",
        "_cell_angle_alpha": "alpha",
        "_cell_angle_beta": "beta",
        "_cell_angle_gamma": "gamma",
    }
    values: dict[str, float] = {}
    for line in text.splitlines():
        parts = line.strip().split(maxsplit=1)
        if len(parts) != 2 or parts[0] not in keys:
            continue
        values[keys[parts[0]]] = parse_cif_float(parts[1].strip().strip("'\""))
    missing = [value for value in keys.values() if value not in values]
    if missing:
        raise AnalysisError(f"CIF cell parameters are incomplete: missing {', '.join(missing)}")
    return values


def parse_atom_site_loop(text: str) -> tuple[list[str], list[str], np.ndarray]:
    lines = text.splitlines()
    index = 0
    while index < len(lines):
        if lines[index].strip() != "loop_":
            index += 1
            continue
        index += 1
        columns = []
        while index < len(lines):
            stripped = lines[index].strip()
            if not stripped or stripped.startswith("#"):
                index += 1
                continue
            if not stripped.startswith("_"):
                break
            columns.append(stripped.split()[0])
            index += 1
        required = {
            "_atom_site_fract_x",
            "_atom_site_fract_y",
            "_atom_site_fract_z",
        }
        if not required.issubset(columns):
            continue
        labels: list[str] = []
        symbols: list[str] = []
        coords: list[list[float]] = []
        while index < len(lines):
            stripped = lines[index].strip()
            if not stripped or stripped.startswith("#"):
                index += 1
                continue
            if stripped == "loop_" or stripped.startswith("_"):
                break
            parts = stripped.split()
            if len(parts) >= len(columns):
                row = dict(zip(columns, parts))
                label = row.get("_atom_site_label", f"site{len(labels)}")
                symbol = row.get("_atom_site_type_symbol", label)
                labels.append(label)
                symbols.append(symbol)
                coords.append(
                    [
                        parse_cif_float(row["_atom_site_fract_x"]),
                        parse_cif_float(row["_atom_site_fract_y"]),
                        parse_cif_float(row["_atom_site_fract_z"]),
                    ]
                )
            index += 1
        return labels, symbols, np.asarray(coords, dtype=float)
    return [], [], np.empty((0, 3), dtype=float)


def resolve_legacy_core_path(path: str | Path | None = None) -> Path:
    if path is not None:
        return Path(path)
    configured = os.environ.get(LEGACY_CORE_ENV_VAR)
    if configured:
        return Path(configured)
    return DEFAULT_LEGACY_CORE


def load_legacy_core(path: str | Path | None = None) -> ModuleType:
    path = resolve_legacy_core_path(path)
    if not path.exists():
        raise AnalysisError(
            f"legacy symmetry core not found: {path}. "
            f"Set {LEGACY_CORE_ENV_VAR} to a symmetry_core.py path if you need to override it."
        )

    module_name = "_kouzoukaiseki_symmetry_core"
    if module_name in sys.modules:
        return sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise AnalysisError(f"could not load legacy symmetry core: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def structure_to_spglib_cell(structure: Structure) -> tuple[np.ndarray, np.ndarray, list[int]]:
    lattice = np.asarray(structure.lattice.matrix, dtype=float)
    positions = np.asarray([site.frac_coords for site in structure], dtype=float)
    numbers = [primary_site_element(site)[1] for site in structure]
    return lattice, positions, numbers


def convert_structure(
    cif_path: Path,
    structure: Structure,
    *,
    asymmetric_atoms: tuple[AsymmetricUnitSite, ...] = (),
    rotations: np.ndarray | None = None,
    translations: np.ndarray | None = None,
    equivalent_atoms: np.ndarray | None = None,
) -> StructureSummary:
    derived_asymmetric_indices: list[int] | None = None
    if not asymmetric_atoms and equivalent_atoms is not None:
        asymmetric_atoms, derived_asymmetric_indices = asymmetric_unit_from_equivalent_atoms(structure, equivalent_atoms)

    atoms = []
    for index, site in enumerate(structure):
        element, atomic_number = primary_site_element(site)
        frac = np.asarray(site.frac_coords, dtype=float)
        if derived_asymmetric_indices is not None:
            asymmetric_index = derived_asymmetric_indices[index]
            source_frac = np.asarray(asymmetric_atoms[asymmetric_index].frac, dtype=float)
            generation_operation_index = identify_generation_operation(frac, source_frac, rotations, translations)
        else:
            asymmetric_index, generation_operation_index = identify_asymmetric_source(
                frac,
                element,
                asymmetric_atoms,
                rotations,
                translations,
            )
        atoms.append(
            AtomSite(
                index=index,
                element=element,
                atomic_number=atomic_number,
                frac=frac,
                cart=np.asarray(site.coords, dtype=float),
                asymmetric_index=asymmetric_index,
                generation_operation_index=generation_operation_index,
            )
        )

    return StructureSummary(
        source_file=cif_path,
        formula=structure.composition.reduced_formula,
        site_count=len(structure),
        lattice=np.asarray(structure.lattice.matrix, dtype=float),
        atoms=tuple(atoms),
        asymmetric_atoms=asymmetric_atoms,
    )


def asymmetric_unit_from_equivalent_atoms(
    structure: Structure,
    equivalent_atoms: np.ndarray,
) -> tuple[tuple[AsymmetricUnitSite, ...], list[int]]:
    lattice = np.asarray(structure.lattice.matrix, dtype=float)
    rep_to_asymmetric_index: dict[int, int] = {}
    asymmetric_atoms: list[AsymmetricUnitSite] = []
    atom_asymmetric_indices: list[int] = []
    for atom_index, representative_index in enumerate(np.asarray(equivalent_atoms, dtype=int)):
        representative_index = int(representative_index)
        if representative_index not in rep_to_asymmetric_index:
            representative_site = structure[representative_index]
            element, atomic_number = primary_site_element(representative_site)
            asymmetric_index = len(asymmetric_atoms)
            rep_to_asymmetric_index[representative_index] = asymmetric_index
            frac = np.asarray(representative_site.frac_coords, dtype=float)
            asymmetric_atoms.append(
                AsymmetricUnitSite(
                    index=asymmetric_index,
                    label=site_label(representative_site, element, asymmetric_index),
                    element=element,
                    atomic_number=atomic_number,
                    frac=frac,
                    cart=frac @ lattice,
                )
            )
        atom_asymmetric_indices.append(rep_to_asymmetric_index[representative_index])
    return tuple(asymmetric_atoms), atom_asymmetric_indices


def site_label(site, element: str, index: int) -> str:
    label = getattr(site, "label", None)
    if label:
        return str(label)
    return f"{element}{index + 1}"


def primary_site_element(site) -> tuple[str, int]:
    specie = getattr(site, "specie", None)
    if specie is not None:
        return str(specie.symbol), int(specie.Z)

    species = getattr(site, "species", None)
    if species is None:
        species = getattr(site, "species_and_occu", None)
    if species is None:
        raise AnalysisError(f"could not determine element for site: {site}")

    items = list(species.items())
    if not items:
        raise AnalysisError(f"empty species composition for site: {site}")

    selected_species, _ = max(
        items,
        key=lambda item: (float(item[1]), species_symbol(item[0])),
    )
    symbol = species_symbol(selected_species)
    atomic_number = species_atomic_number(selected_species, symbol)
    return symbol, atomic_number


def species_symbol(species) -> str:
    symbol = getattr(species, "symbol", None)
    if symbol is not None:
        return normalize_element_symbol(symbol)
    element = getattr(species, "element", None)
    if element is not None and getattr(element, "symbol", None) is not None:
        return normalize_element_symbol(element.symbol)
    return normalize_element_symbol(str(species))


def species_atomic_number(species, symbol: str) -> int:
    z_value = getattr(species, "Z", None)
    if z_value is not None:
        return int(z_value)
    element = getattr(species, "element", None)
    if element is not None and getattr(element, "Z", None) is not None:
        return int(element.Z)
    return int(Element(symbol).Z)


def normalize_element_symbol(value, label=None) -> str:
    text = str(value).strip()
    if "," in text:
        text = text.split(",", 1)[0].strip()
    text = re.sub(r"[^A-Za-z].*$", "", text)
    if text:
        return str(Element(text).symbol)

    if label is not None:
        label_text = str(label).strip()
        match = re.match(r"([A-Z][a-z]?)", label_text)
        if match:
            return str(Element(match.group(1)).symbol)

    raise AnalysisError(f"could not parse element symbol from {value!r}")


def parse_cif_float(value) -> float:
    text = str(value).strip()
    if "(" in text:
        text = text.split("(", 1)[0]
    return float(text)


def identify_asymmetric_source(
    frac: np.ndarray,
    element: str,
    asymmetric_atoms: tuple[AsymmetricUnitSite, ...],
    rotations: np.ndarray | None,
    translations: np.ndarray | None,
) -> tuple[int | None, int | None]:
    if not asymmetric_atoms or rotations is None or translations is None:
        return None, None

    # First match within tolerance wins, for the same reason as
    # identify_generation_operation: exact ties are common on special positions and
    # comparing distances lets floating point noise pick the winner.
    for asymmetric_atom in asymmetric_atoms:
        if asymmetric_atom.element != element:
            continue
        source_frac = np.asarray(asymmetric_atom.frac, dtype=float)
        for operation_index, (rotation, translation) in enumerate(zip(rotations, translations)):
            generated = wrap_frac(rotation @ source_frac + translation)
            delta = generated - frac
            delta = delta - np.round(delta)
            if float(np.linalg.norm(delta)) <= 1e-5:
                return asymmetric_atom.index, operation_index

    return None, None


def identify_generation_operation(
    frac: np.ndarray,
    source_frac: np.ndarray,
    rotations: np.ndarray | None,
    translations: np.ndarray | None,
) -> int | None:
    if rotations is None or translations is None:
        return None
    # An atom on a special position is generated by every operation in its site
    # symmetry group, so several operations match exactly. Picking the numerically
    # closest one lets ~1e-16 noise in the translation vectors decide the winner,
    # which makes the result flip between spglib releases. Take the first operation
    # within tolerance instead: any match is equally valid, and the lowest index is
    # stable.
    for operation_index, (rotation, translation) in enumerate(zip(rotations, translations)):
        generated = wrap_frac(rotation @ source_frac + translation)
        delta = generated - frac
        delta = delta - np.round(delta)
        if float(np.linalg.norm(delta)) <= 1e-5:
            return operation_index
    return None


def wrap_frac(frac: np.ndarray) -> np.ndarray:
    wrapped = np.mod(np.asarray(frac, dtype=float), 1.0)
    wrapped[np.abs(wrapped) < 1e-12] = 0.0
    wrapped[np.abs(wrapped - 1.0) < 1e-12] = 0.0
    return wrapped


def convert_space_group(dataset) -> SpaceGroupInfo:
    return SpaceGroupInfo(
        number=int(dataset_value(dataset, "number")),
        international=str(dataset_value(dataset, "international")),
        hall=str(dataset_value(dataset, "hall")),
        point_group=str(dataset_value(dataset, "pointgroup")),
        operation_count=len(dataset_value(dataset, "rotations")),
        wyckoffs=tuple(str(x) for x in dataset_value(dataset, "wyckoffs")),
        equivalent_atoms=tuple(int(x) for x in dataset_value(dataset, "equivalent_atoms")),
        site_symmetry_symbols=tuple(str(x) for x in dataset_value(dataset, "site_symmetry_symbols")),
    )


def convert_operation(core: ModuleType, index: int, W: np.ndarray, t: np.ndarray, search_range: int) -> SymmetryOperationInfo:
    kind, det, trace, order, angle = core.classify_operation(W, t, search_range=search_range)
    op = {
        "index": index,
        "type": kind,
        "order": order,
        "det": det,
        "trace": trace,
        "angle": angle,
    }
    return SymmetryOperationInfo(
        index=index,
        W=np.asarray(W, dtype=int),
        t=np.asarray(t, dtype=float),
        kind=str(kind),
        order=None if order is None else int(order),
        det=int(det),
        trace=int(trace),
        angle_deg=None if angle is None else float(angle),
        international_symbol=core.operation_international_symbol(op),
    )


def convert_axis(axis: dict) -> AxisElement:
    return AxisElement(
        point_frac=np.asarray(axis["point"], dtype=float),
        direction_frac=np.asarray(axis["direction"], dtype=float),
        operation_indices=operation_indices(axis),
        motion_coeffs=tuple(none_or_array(x) for x in axis.get("motion_coeffs", [])),
    )


def convert_plane(plane: dict) -> PlaneElement:
    return PlaneElement(
        point_frac=np.asarray(plane["point"], dtype=float),
        normal_frac=np.asarray(plane["normal"], dtype=float),
        basis_frac=np.asarray(plane["basis"], dtype=float),
        operation_indices=operation_indices(plane),
        motion_coeffs=tuple(none_or_array(x) for x in plane.get("motion_coeffs", [])),
    )


def convert_center(center: dict) -> CenterElement:
    return CenterElement(
        point_frac=np.asarray(center["point"], dtype=float),
        operation_indices=operation_indices(center),
    )


def operation_indices(element: dict) -> tuple[int, ...]:
    return tuple(int(op["index"]) for op in element.get("operations", []))


def none_or_array(value):
    if value is None:
        return None
    return np.asarray(value, dtype=float)


def dataset_value(dataset, key: str):
    if isinstance(dataset, dict):
        return dataset[key]
    return getattr(dataset, key)
