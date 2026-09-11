from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from .analysis_models import (
    AsymmetricUnitSite,
    AtomSite,
    MoleculeAnalysisResult,
    MolecularAxisElement,
    MolecularCenterElement,
    MolecularPlaneElement,
    StructureAnalysisResult,
)
from .geometry import normalize


RenderMode = Literal["crystal", "molecule"]
GENERATOR_ROTATION_SCALE = 1_000_000
GENERATOR_TRANSLATION_SCALE = 120


@dataclass(frozen=True)
class RenderAtomData:
    index: int
    element: str
    atomic_number: int
    cart: np.ndarray
    frac: np.ndarray | None = None
    asymmetric_index: int | None = None
    generation_operation_index: int | None = None


@dataclass(frozen=True)
class RenderAsymmetricAtomData:
    index: int
    label: str
    element: str
    atomic_number: int
    cart: np.ndarray
    frac: np.ndarray


@dataclass(frozen=True)
class RenderOperationData:
    index: int
    label: str
    kind: str
    order: int | None
    angle_deg: float | None
    symbol: str
    matrix_frac: np.ndarray | None = None
    translation_frac: np.ndarray | None = None
    matrix_cart: np.ndarray | None = None
    translation_cart: np.ndarray | None = None


@dataclass(frozen=True)
class RenderAxisData:
    point_cart: np.ndarray
    direction_cart: np.ndarray
    label: str
    operation_indices: tuple[int, ...]


@dataclass(frozen=True)
class RenderPlaneData:
    point_cart: np.ndarray
    basis1_cart: np.ndarray
    basis2_cart: np.ndarray
    normal_cart: np.ndarray
    label: str
    operation_indices: tuple[int, ...]


@dataclass(frozen=True)
class RenderCenterData:
    point_cart: np.ndarray
    label: str
    operation_indices: tuple[int, ...]


@dataclass(frozen=True)
class UnitCellRenderData:
    lattice: np.ndarray
    vertices_cart: np.ndarray
    edges: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class RenderMetadata:
    mode: RenderMode
    source_file: Path | None
    formula: str
    symmetry_label: str
    operation_count: int
    # The formula to show a learner.  For crystals the reduced formula is the
    # right one (NaCl); for molecules it is not (benzene reduces to HC), so the
    # molecule path supplies its own.
    display_formula: str = ""
    point_group_label: str | None = None
    lattice_parameters: dict[str, float] | None = None
    space_group_generators: tuple[str, ...] = ()
    point_group_generators: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class RenderData:
    metadata: RenderMetadata
    atoms: tuple[RenderAtomData, ...]
    asymmetric_atoms: tuple[RenderAsymmetricAtomData, ...]
    operations: tuple[RenderOperationData, ...]
    axes: tuple[RenderAxisData, ...]
    planes: tuple[RenderPlaneData, ...]
    centers: tuple[RenderCenterData, ...]
    unit_cell: UnitCellRenderData | None = None
    bounds_min: np.ndarray = field(default_factory=lambda: np.zeros(3))
    bounds_max: np.ndarray = field(default_factory=lambda: np.zeros(3))


def render_data_from_analysis(result: StructureAnalysisResult | MoleculeAnalysisResult) -> RenderData:
    if isinstance(result, StructureAnalysisResult):
        return render_data_from_crystal(result)
    if isinstance(result, MoleculeAnalysisResult):
        return render_data_from_molecule(result)
    raise TypeError(f"unsupported analysis result type: {type(result).__name__}")


def render_data_from_crystal(result: StructureAnalysisResult) -> RenderData:
    lattice = np.asarray(result.structure.lattice, dtype=float)
    atoms = tuple(render_atom_from_site(atom) for atom in result.structure.atoms)
    asymmetric_atoms = tuple(render_asymmetric_atom(site) for site in result.structure.asymmetric_atoms)
    unit_cell = unit_cell_from_lattice(lattice)
    operations = tuple(
        RenderOperationData(
            index=op.index,
            label=f"{op.index}: {op.international_symbol} {op.kind}",
            kind=op.kind,
            order=op.order,
            angle_deg=op.angle_deg,
            symbol=op.international_symbol,
            matrix_frac=np.asarray(op.W, dtype=float),
            translation_frac=np.asarray(op.t, dtype=float),
            matrix_cart=lattice.T @ np.asarray(op.W, dtype=float) @ np.linalg.inv(lattice.T),
            translation_cart=np.asarray(op.t, dtype=float) @ lattice,
        )
        for op in result.operations
    )

    symbols = {op.index: op.international_symbol for op in result.operations}

    axes = tuple(
        RenderAxisData(
            point_cart=axis.point_frac @ lattice,
            direction_cart=normalize(axis.direction_frac @ lattice),
            label=operation_group_label(axis.operation_indices, symbols),
            operation_indices=axis.operation_indices,
        )
        for axis in result.axes
    )

    planes = []
    for plane in result.planes:
        basis1_cart = plane.basis_frac[:, 0] @ lattice
        basis2_cart = plane.basis_frac[:, 1] @ lattice
        normal_cart = normalize(np.cross(basis1_cart, basis2_cart))
        planes.append(
            RenderPlaneData(
                point_cart=plane.point_frac @ lattice,
                basis1_cart=normalize(basis1_cart),
                basis2_cart=normalize(basis2_cart),
                normal_cart=normal_cart,
                label=operation_group_label(plane.operation_indices, symbols),
                operation_indices=plane.operation_indices,
            )
        )

    centers = tuple(
        RenderCenterData(
            point_cart=center.point_frac @ lattice,
            label=operation_group_label(center.operation_indices, symbols),
            operation_indices=center.operation_indices,
        )
        for center in result.centers
    )

    bpoints = bounds_points(atoms, unit_cell)
    bmin = np.min(bpoints, axis=0) if len(bpoints) else np.zeros(3)
    bmax = np.max(bpoints, axis=0) if len(bpoints) else np.zeros(3)

    return RenderData(
        metadata=RenderMetadata(
            mode="crystal",
            source_file=result.structure.source_file,
            formula=result.structure.formula,
            symmetry_label=f"{result.space_group.number} {result.space_group.international}",
            operation_count=result.space_group.operation_count,
            display_formula=result.structure.formula,
            point_group_label=result.space_group.point_group,
            lattice_parameters=lattice_parameters_from_matrix(lattice),
            space_group_generators=operation_generators(operations, mode="space"),
            point_group_generators=operation_generators(operations, mode="point"),
            warnings=result.warnings,
        ),
        atoms=atoms,
        asymmetric_atoms=asymmetric_atoms,
        operations=operations,
        axes=axes,
        planes=tuple(planes),
        centers=centers,
        unit_cell=unit_cell,
        bounds_min=bmin,
        bounds_max=bmax,
    )


def render_data_from_molecule(result: MoleculeAnalysisResult) -> RenderData:
    atoms = tuple(render_atom_from_site(atom) for atom in result.molecule.atoms)
    operations = tuple(
        RenderOperationData(
            index=op.index,
            label=f"{op.index}: {op.symbol} {op.kind}",
            kind=op.kind,
            order=op.order,
            angle_deg=op.angle_deg,
            symbol=op.symbol,
            matrix_cart=np.asarray(op.rotation, dtype=float),
            translation_cart=np.asarray(op.translation, dtype=float),
        )
        for op in result.operations
    )

    symbols = {op.index: op.symbol for op in result.operations}

    axes = tuple(render_axis_from_molecular_axis(axis, symbols) for axis in result.axes)
    planes = tuple(render_plane_from_molecular_plane(plane, symbols) for plane in result.planes)
    centers = tuple(render_center_from_molecular_center(center, symbols) for center in result.centers)

    bpoints = bounds_points(atoms, None)
    bmin = np.min(bpoints, axis=0) if len(bpoints) else np.zeros(3)
    bmax = np.max(bpoints, axis=0) if len(bpoints) else np.zeros(3)

    return RenderData(
        metadata=RenderMetadata(
            mode="molecule",
            source_file=result.molecule.source_file,
            formula=result.molecule.formula,
            symmetry_label=result.point_group.symbol,
            operation_count=result.point_group.operation_count,
            display_formula=result.molecule.display_formula or result.molecule.formula,
            point_group_label=result.point_group.symbol,
            point_group_generators=operation_generators(operations, mode="point"),
        ),
        atoms=atoms,
        asymmetric_atoms=(),
        operations=operations,
        axes=axes,
        planes=planes,
        centers=centers,
        unit_cell=None,
        bounds_min=bmin,
        bounds_max=bmax,
    )


def render_atom_from_site(atom: AtomSite) -> RenderAtomData:
    return RenderAtomData(
        index=atom.index,
        element=atom.element,
        atomic_number=atom.atomic_number,
        frac=None if atom.frac is None else np.asarray(atom.frac, dtype=float),
        cart=np.asarray(atom.cart, dtype=float),
        asymmetric_index=atom.asymmetric_index,
        generation_operation_index=atom.generation_operation_index,
    )


def render_asymmetric_atom(site: AsymmetricUnitSite) -> RenderAsymmetricAtomData:
    return RenderAsymmetricAtomData(
        index=site.index,
        label=site.label,
        element=site.element,
        atomic_number=site.atomic_number,
        frac=np.asarray(site.frac, dtype=float),
        cart=np.asarray(site.cart, dtype=float),
    )


def render_axis_from_molecular_axis(
    axis: MolecularAxisElement,
    symbols: dict[int, str],
) -> RenderAxisData:
    return RenderAxisData(
        point_cart=np.asarray(axis.point_cart, dtype=float),
        direction_cart=normalize(axis.direction_cart),
        label=operation_group_label(axis.operation_indices, symbols),
        operation_indices=axis.operation_indices,
    )


def render_plane_from_molecular_plane(
    plane: MolecularPlaneElement,
    symbols: dict[int, str],
) -> RenderPlaneData:
    basis = np.asarray(plane.basis_cart, dtype=float)
    basis1_cart = normalize(basis[:, 0])
    basis2_cart = normalize(basis[:, 1])
    normal_cart = normalize(plane.normal_cart)
    return RenderPlaneData(
        point_cart=np.asarray(plane.point_cart, dtype=float),
        basis1_cart=basis1_cart,
        basis2_cart=basis2_cart,
        normal_cart=normal_cart,
        label=operation_group_label(plane.operation_indices, symbols),
        operation_indices=plane.operation_indices,
    )


def render_center_from_molecular_center(
    center: MolecularCenterElement,
    symbols: dict[int, str],
) -> RenderCenterData:
    return RenderCenterData(
        point_cart=np.asarray(center.point_cart, dtype=float),
        label=operation_group_label(center.operation_indices, symbols),
        operation_indices=center.operation_indices,
    )


def unit_cell_from_lattice(lattice: np.ndarray) -> UnitCellRenderData:
    lattice = np.asarray(lattice, dtype=float)
    a, b, c = lattice[0], lattice[1], lattice[2]
    vertices = np.array(
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
    return UnitCellRenderData(lattice=lattice, vertices_cart=vertices, edges=edges)


def operation_group_label(operation_indices: tuple[int, ...], symbols_by_index: dict[int, str]) -> str:
    counts: dict[str, int] = {}
    for index in operation_indices:
        symbol = symbols_by_index.get(index, "?")
        counts[symbol] = counts.get(symbol, 0) + 1
    parts = [f"{symbol}x{count}" if count > 1 else symbol for symbol, count in sorted(counts.items())]
    return ", ".join(parts)


def lattice_parameters_from_matrix(lattice: np.ndarray) -> dict[str, float]:
    vectors = np.asarray(lattice, dtype=float)
    a_vec, b_vec, c_vec = vectors
    return {
        "a": float(np.linalg.norm(a_vec)),
        "b": float(np.linalg.norm(b_vec)),
        "c": float(np.linalg.norm(c_vec)),
        "alpha": vector_angle_deg(b_vec, c_vec),
        "beta": vector_angle_deg(a_vec, c_vec),
        "gamma": vector_angle_deg(a_vec, b_vec),
    }


def vector_angle_deg(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    denom = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denom <= 0:
        return 0.0
    cosine = float(np.dot(first, second) / denom)
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def operation_generators(
    operations: tuple[RenderOperationData, ...],
    *,
    mode: Literal["space", "point"],
) -> tuple[str, ...]:
    keyed_operations = generator_operation_keys(operations, mode=mode)
    if not keyed_operations:
        return ()
    target = {key for _, key in keyed_operations}
    identity = generator_identity_key(mode)
    if target == {identity}:
        return ("identity only",)

    candidates = [(operation, key) for operation, key in keyed_operations if key != identity]
    composition_table = generator_composition_table(target, mode=mode)
    selected: list[tuple[RenderOperationData, tuple]] = []
    generated = {identity}
    while generated != target and candidates:
        best_index = -1
        best_generated = generated
        for index, (_, key) in enumerate(candidates):
            candidate_generated = generator_closure(
                [selected_key for _, selected_key in selected] + [key],
                mode=mode,
                target=target,
                identity=identity,
                composition_table=composition_table,
            )
            if len(candidate_generated) > len(best_generated):
                best_index = index
                best_generated = candidate_generated
        if best_index < 0:
            break
        selected.append(candidates.pop(best_index))
        generated = best_generated

    if generated != target:
        return tuple(generator_label(operation, mode=mode) for operation, key in keyed_operations if key != identity)
    return tuple(generator_label(operation, mode=mode) for operation, _ in selected)


def generator_composition_table(target: set[tuple], *, mode: Literal["space", "point"]) -> dict[tuple[tuple, tuple], tuple]:
    components = {key: generator_key_components(key, mode=mode) for key in target}
    return {
        (first, second): compose_generator_key_components(
            components[first],
            components[second],
            mode=mode,
        )
        for first in target
        for second in target
    }


def generator_key_components(key: tuple, *, mode: Literal["space", "point"]) -> tuple[np.ndarray, np.ndarray | None]:
    if mode == "point":
        return np.asarray(key, dtype=np.int64), None
    rotation, translation = key
    return np.asarray(rotation, dtype=np.int64), np.asarray(translation, dtype=np.int64)


def compose_generator_key_components(
    first: tuple[np.ndarray, np.ndarray | None],
    second: tuple[np.ndarray, np.ndarray | None],
    *,
    mode: Literal["space", "point"],
) -> tuple:
    first_rotation, first_translation = first
    second_rotation, second_translation = second
    rotation = np.rint((first_rotation @ second_rotation) / GENERATOR_ROTATION_SCALE).astype(np.int64)
    rotation_tuple = tuple(tuple(int(value) for value in row) for row in rotation)
    if mode == "point":
        return rotation_tuple
    if first_translation is None or second_translation is None:
        raise ValueError("space-group generator keys require translation components")
    translation = np.rint((first_rotation @ second_translation) / GENERATOR_ROTATION_SCALE).astype(np.int64)
    translation = np.mod(translation + first_translation, GENERATOR_TRANSLATION_SCALE)
    return rotation_tuple, tuple(int(value) for value in translation)


def generator_operation_keys(
    operations: tuple[RenderOperationData, ...],
    *,
    mode: Literal["space", "point"],
) -> list[tuple[RenderOperationData, tuple]]:
    keyed: list[tuple[RenderOperationData, tuple]] = []
    seen: set[tuple] = set()
    for operation in operations:
        matrix = operation.matrix_frac if operation.matrix_frac is not None else operation.matrix_cart
        if matrix is None:
            continue
        rotation = rotation_key(matrix)
        if mode == "point":
            key = rotation
        else:
            translation = operation.translation_frac if operation.translation_frac is not None else operation.translation_cart
            key = (rotation, translation_key(translation))
        if key in seen:
            continue
        seen.add(key)
        keyed.append((operation, key))
    return keyed


def generator_identity_key(mode: Literal["space", "point"]) -> tuple:
    identity_rotation = rotation_key(np.eye(3))
    if mode == "point":
        return identity_rotation
    return (identity_rotation, translation_key(np.zeros(3)))


def generator_closure(
    generators: list[tuple],
    *,
    mode: Literal["space", "point"],
    target: set[tuple],
    identity: tuple | None = None,
    composition_table: dict[tuple[tuple, tuple], tuple] | None = None,
) -> set[tuple]:
    generated = {identity if identity is not None else generator_identity_key(mode)}
    frontier = list(generated)
    while frontier:
        existing = frontier.pop()
        for generator in generators:
            for first, second in ((existing, generator), (generator, existing)):
                if composition_table is None:
                    composed = compose_generator_keys(first, second, mode=mode)
                else:
                    composed = composition_table.get((first, second))
                    if composed is None:
                        continue
                if composed in target and composed not in generated:
                    generated.add(composed)
                    frontier.append(composed)
    return generated


def compose_generator_keys(first: tuple, second: tuple, *, mode: Literal["space", "point"]) -> tuple:
    if mode == "point":
        return rotation_key(rotation_from_key(first) @ rotation_from_key(second))
    first_rotation_key, first_translation_key = first
    second_rotation_key, second_translation_key = second
    first_rotation = rotation_from_key(first_rotation_key)
    second_rotation = rotation_from_key(second_rotation_key)
    first_translation = translation_from_key(first_translation_key)
    second_translation = translation_from_key(second_translation_key)
    return (
        rotation_key(first_rotation @ second_rotation),
        translation_key(first_rotation @ second_translation + first_translation),
    )


def rotation_key(matrix: np.ndarray) -> tuple[tuple[int, int, int], ...]:
    rounded = np.rint(np.asarray(matrix, dtype=float) * GENERATOR_ROTATION_SCALE).astype(int)
    return tuple(tuple(int(value) for value in row) for row in rounded)


def rotation_from_key(key: tuple[tuple[int, int, int], ...]) -> np.ndarray:
    return np.asarray(key, dtype=float) / GENERATOR_ROTATION_SCALE


def translation_key(translation: np.ndarray | None) -> tuple[int, int, int]:
    if translation is None:
        return (0, 0, 0)
    wrapped = np.mod(np.asarray(translation, dtype=float), 1.0)
    scaled = np.mod(np.rint(wrapped * GENERATOR_TRANSLATION_SCALE).astype(int), GENERATOR_TRANSLATION_SCALE)
    return tuple(int(value) for value in scaled)


def translation_from_key(key: tuple[int, int, int]) -> np.ndarray:
    return np.asarray(key, dtype=float) / GENERATOR_TRANSLATION_SCALE


def generator_label(operation: RenderOperationData, *, mode: Literal["space", "point"]) -> str:
    if mode == "point":
        return point_generator_label(operation)
    return space_generator_label(operation)


def point_generator_label(operation: RenderOperationData) -> str:
    kind = str(operation.kind)
    if kind == "identity":
        return "E"
    if kind == "mirror" or "glide" in kind:
        return "σ"
    if kind == "inversion":
        return "i"
    if "rotoinversion" in kind or "rotoreflection" in kind or "improper" in kind:
        if operation.order is not None and operation.order > 1:
            return f"S{operation.order}"
    if kind.startswith("rotation") or kind.startswith("screw"):
        if operation.order is not None and operation.order > 1:
            return f"C{operation.order}"
    return normalize_generator_symbol(operation.symbol or operation.kind)


def space_generator_label(operation: RenderOperationData) -> str:
    if is_translation_operation(operation):
        return f"t({format_fractional_vector(operation_translation_vector(operation))})"
    if "glide" in str(operation.kind):
        return f"g({format_fractional_vector(operation_translation_vector(operation))})"
    symbol = operation.symbol or operation.kind
    if "?" in str(symbol):
        return generator_fallback_symbol(operation)
    return normalize_generator_symbol(symbol)


def is_translation_operation(operation: RenderOperationData) -> bool:
    matrix = operation.matrix_frac if operation.matrix_frac is not None else operation.matrix_cart
    translation = operation_translation_vector(operation)
    if matrix is None or translation is None:
        return False
    return bool(
        np.linalg.norm(np.asarray(matrix, dtype=float) - np.eye(3)) <= 1e-8
        and np.linalg.norm(np.asarray(translation, dtype=float)) > 1e-8
    )


def operation_translation_vector(operation: RenderOperationData) -> np.ndarray | None:
    return operation.translation_frac if operation.translation_frac is not None else operation.translation_cart


def format_fractional_vector(vector: np.ndarray | None) -> str:
    if vector is None:
        return "0,0,0"
    return ",".join(format_fractional_component(value) for value in np.mod(np.asarray(vector, dtype=float), 1.0))


def format_fractional_component(value: float) -> str:
    value = float(value)
    if np.isclose(value, 0.0, atol=1e-8) or np.isclose(value, 1.0, atol=1e-8):
        return "0"
    for denominator in (2, 3, 4, 6, 8, 12):
        numerator = int(round(value * denominator))
        if numerator == denominator:
            numerator = 0
        if np.isclose(value, numerator / denominator, atol=1e-8):
            return f"{numerator}/{denominator}" if numerator != 0 else "0"
    return f"{value:.3f}".rstrip("0").rstrip(".")


def normalize_generator_symbol(symbol: str) -> str:
    text = str(symbol).strip()
    text = text.replace("sigma", "σ")
    text = text.replace("_v", "v").replace("_h", "h").replace("_d", "d")
    if text == "m":
        return "σ"
    if text.isdigit() and int(text) > 1:
        return f"C{text}"
    if text.startswith("-") and text[1:].isdigit():
        order = int(text[1:])
        if order == 1:
            return "i"
        return f"S{order if order % 2 == 0 else order * 2}"
    return text


def generator_fallback_symbol(operation: RenderOperationData) -> str:
    if operation.order is not None and operation.order > 1:
        if str(operation.kind).startswith("screw"):
            inferred = infer_compact_screw_symbol(operation)
            return inferred if inferred is not None else f"C{operation.order}"
        if "rotation" in str(operation.kind):
            return f"C{operation.order}"
    if str(operation.kind) == "mirror":
        return "σ"
    if str(operation.kind) == "inversion":
        return "i"
    return normalize_generator_symbol(operation.kind)


def infer_compact_screw_symbol(operation: RenderOperationData) -> str | None:
    if operation.order is None or operation.order <= 1 or operation.translation_frac is None:
        return None
    matrix = operation.matrix_frac if operation.matrix_frac is not None else operation.matrix_cart
    if matrix is None:
        return None
    values, vectors = np.linalg.eig(np.asarray(matrix, dtype=float))
    candidates = [index for index, value in enumerate(values) if abs(value - 1.0) <= 1e-6]
    if not candidates:
        return None
    axis = np.real(vectors[:, candidates[0]])
    norm = float(np.linalg.norm(axis))
    if norm <= 0:
        return None
    axis = axis / norm
    translation = np.asarray(operation.translation_frac, dtype=float)
    fraction = abs(float(np.dot(translation, axis)))
    fraction -= np.floor(fraction)
    screw = int(np.floor(fraction * operation.order + 0.5 + 1e-8))
    if screw == 0 and not np.isclose(fraction, 0.0, atol=1e-6):
        screw = 1
    if screw <= 0:
        return None
    if screw >= operation.order:
        screw = operation.order - 1
    return f"{operation.order}_{screw}"


def bounds_points(atoms: tuple[RenderAtomData, ...], unit_cell: UnitCellRenderData | None) -> np.ndarray:
    parts = []
    if atoms:
        parts.append(np.array([atom.cart for atom in atoms], dtype=float))
    if unit_cell is not None:
        parts.append(np.asarray(unit_cell.vertices_cart, dtype=float))
    if not parts:
        return np.empty((0, 3))
    return np.vstack(parts)
