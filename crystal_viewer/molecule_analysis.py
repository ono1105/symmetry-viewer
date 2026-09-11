from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from pymatgen.core import Composition, Molecule
from pymatgen.symmetry.analyzer import PointGroupAnalyzer

from .geometry import (
    plane_basis_from_normal_cart,
    rotation_angle_deg,
)
from .analysis_models import (
    AtomSite,
    MolecularAxisElement,
    MolecularCenterElement,
    MolecularPlaneElement,
    MolecularSymmetryOperationInfo,
    MoleculeAnalysisResult,
    MoleculePointGroupInfo,
    MoleculeSummary,
)
from .structure_analysis import AnalysisError


TOL = 1e-7


def analyze_molecule_file(
    molecule_path: str | Path,
    *,
    tolerance: float = 0.3,
    eigen_tolerance: float = 0.01,
    matrix_tolerance: float = 0.1,
) -> MoleculeAnalysisResult:
    molecule_path = Path(molecule_path)
    if not molecule_path.exists():
        raise AnalysisError(f"molecule file not found: {molecule_path}")

    molecule = Molecule.from_file(molecule_path)
    return analyze_molecule(
        molecule,
        source_file=molecule_path,
        tolerance=tolerance,
        eigen_tolerance=eigen_tolerance,
        matrix_tolerance=matrix_tolerance,
    )


def analyze_molecule(
    molecule: Molecule,
    *,
    source_file: str | Path | None = None,
    tolerance: float = 0.3,
    eigen_tolerance: float = 0.01,
    matrix_tolerance: float = 0.1,
) -> MoleculeAnalysisResult:
    analyzer = PointGroupAnalyzer(
        molecule,
        tolerance=tolerance,
        eigen_tolerance=eigen_tolerance,
        matrix_tolerance=matrix_tolerance,
    )
    operations = tuple(analyzer.get_symmetry_operations())
    center_cart = molecule_center_from_analyzer(molecule, analyzer)

    op_infos = tuple(
        convert_operation(index, op.rotation_matrix, op.translation_vector, center_cart)
        for index, op in enumerate(operations)
    )
    axes, planes, centers = collect_molecular_elements(op_infos, center_cart)
    extra_ops, extra_axes, extra_planes = linear_molecule_virtual_symmetry(
        molecule,
        analyzer,
        center_cart,
        start_index=len(op_infos),
    )
    op_infos = op_infos + extra_ops
    axes.extend(extra_axes)
    planes.extend(extra_planes)

    return MoleculeAnalysisResult(
        molecule=convert_molecule(source_file, molecule, center_cart),
        point_group=convert_point_group(analyzer, op_infos),
        operations=op_infos,
        axes=tuple(axes),
        planes=tuple(planes),
        centers=tuple(centers),
    )


def convert_molecule(source_file: str | Path | None, molecule: Molecule, center_cart: np.ndarray) -> MoleculeSummary:
    atoms = []
    for index, site in enumerate(molecule):
        atoms.append(
            AtomSite(
                index=index,
                element=site.specie.symbol,
                atomic_number=int(site.specie.Z),
                frac=None,
                cart=np.asarray(site.coords, dtype=float),
            )
        )

    return MoleculeSummary(
        source_file=None if source_file is None else Path(source_file),
        formula=molecule.composition.reduced_formula,
        site_count=len(molecule),
        atoms=tuple(atoms),
        center_cart=np.asarray(center_cart, dtype=float),
        display_formula=molecular_display_formula(molecule.composition),
    )


def molecular_display_formula(composition: Composition) -> str:
    """The molecular formula as a chemist writes it: C6H6, NH3, XeF4.

    The reduced formula collapses the molecule (benzene -> HC) and the Hill
    formula alphabetizes it (NH3 -> H3N), so neither can be shown to a learner.
    The IUPAC formula keeps both the true counts and the conventional element
    order; it only needs its spaces and its explicit "1" subscripts removed.
    """
    return re.sub(r"(?<=[A-Za-z])1(?![0-9])", "", composition.iupac_formula.replace(" ", ""))


def convert_point_group(analyzer: PointGroupAnalyzer, operations: tuple) -> MoleculePointGroupInfo:
    equivalent = analyzer.get_equivalent_atoms()
    eq_sets = equivalent.get("eq_sets", {})
    return MoleculePointGroupInfo(
        symbol=schoenflies_point_group_symbol(str(analyzer.get_pointgroup())),
        operation_count=len(operations),
        rotational_symmetry_number=int(analyzer.get_rotational_symmetry_number()),
        equivalent_atom_sets=tuple(tuple(sorted(indices)) for indices in eq_sets.values()),
    )


def schoenflies_point_group_symbol(symbol: str) -> str:
    return {"C*v": "C∞v", "D*h": "D∞h"}.get(symbol, symbol)


def convert_operation(
    index: int,
    rotation: np.ndarray,
    translation: np.ndarray,
    center_cart: np.ndarray,
) -> MolecularSymmetryOperationInfo:
    rotation = clean_matrix(rotation)
    translation = np.asarray(translation, dtype=float)
    det = int(round(float(np.linalg.det(rotation))))
    trace = float(np.trace(rotation))
    order = matrix_order(rotation)
    kind = classify_molecular_operation(rotation, det, trace, order)
    if kind == "mirror" and order is None:
        order = 2
    if kind == "mirror":
        rotation = canonical_reflection_matrix(rotation)
        det = -1
        trace = 1.0
        order = 2
    angle = rotation_angle_deg(rotation) if det == 1 and kind != "identity" else None

    # PointGroupAnalyzer operations act on centered coordinates. This is the
    # equivalent affine operation in the original molecular coordinates.
    original_translation = center_cart + translation - rotation @ center_cart

    return MolecularSymmetryOperationInfo(
        index=index,
        rotation=rotation,
        translation=clean_vector(original_translation),
        kind=kind,
        order=order,
        det=det,
        trace=trace,
        angle_deg=angle,
        symbol=operation_symbol(kind, order),
    )


def collect_molecular_elements(
    operations: tuple[MolecularSymmetryOperationInfo, ...],
    center_cart: np.ndarray,
) -> tuple[list[MolecularAxisElement], list[MolecularPlaneElement], list[MolecularCenterElement]]:
    axes: list[MolecularAxisElement] = []
    planes: list[MolecularPlaneElement] = []
    centers: list[MolecularCenterElement] = []

    for op in operations:
        if op.kind.startswith("rotation_"):
            direction = first_null_vector(op.rotation - np.eye(3))
            if direction is not None:
                merge_axis(
                    axes,
                    MolecularAxisElement(
                        point_cart=center_cart,
                        direction_cart=direction,
                        operation_indices=(op.index,),
                        kind=op.kind,
                        order=op.order,
                    ),
                )
        elif op.kind == "mirror":
            normal = first_null_vector(op.rotation + np.eye(3))
            if normal is not None:
                basis = plane_basis_from_normal(normal)
                merge_plane(
                    planes,
                    MolecularPlaneElement(
                        point_cart=center_cart,
                        normal_cart=normal,
                        basis_cart=basis,
                        operation_indices=(op.index,),
                        kind=op.kind,
                    ),
                )
        elif op.kind == "inversion":
            merge_center(
                centers,
                MolecularCenterElement(
                    point_cart=center_cart,
                    operation_indices=(op.index,),
                    kind=op.kind,
                ),
            )
        elif op.kind.startswith("improper_"):
            direction = first_null_vector(op.rotation + np.eye(3))
            if direction is not None:
                merge_axis(
                    axes,
                    MolecularAxisElement(
                        point_cart=center_cart,
                        direction_cart=direction,
                        operation_indices=(op.index,),
                        kind=op.kind,
                        order=op.order,
                    ),
                )

    return axes, planes, centers


def linear_molecule_virtual_symmetry(
    molecule: Molecule,
    analyzer: PointGroupAnalyzer,
    center_cart: np.ndarray,
    *,
    start_index: int,
) -> tuple[
    tuple[MolecularSymmetryOperationInfo, ...],
    list[MolecularAxisElement],
    list[MolecularPlaneElement],
]:
    point_group = str(analyzer.get_pointgroup())
    if "*" not in point_group:
        return (), [], []

    direction = molecule_line_direction(molecule, center_cart)
    if direction is None:
        return (), [], []
    normal, basis2 = plane_basis_from_normal(direction).T
    ops: list[MolecularSymmetryOperationInfo] = []
    axes: list[MolecularAxisElement] = []
    planes: list[MolecularPlaneElement] = []

    c_inf_index = start_index
    ops.append(
        MolecularSymmetryOperationInfo(
            index=c_inf_index,
            rotation=np.eye(3),
            translation=np.zeros(3),
            kind="rotation_infinite",
            order=None,
            det=1,
            trace=3.0,
            angle_deg=None,
            symbol="C∞",
        )
    )
    axes.append(
        MolecularAxisElement(
            point_cart=center_cart,
            direction_cart=direction,
            operation_indices=(c_inf_index,),
            kind="rotation_infinite",
            order=None,
        )
    )

    sigma_v_index = start_index + len(ops)
    sigma_v_rotation = reflection_matrix(normal)
    ops.append(
        MolecularSymmetryOperationInfo(
            index=sigma_v_index,
            rotation=sigma_v_rotation,
            translation=clean_vector(center_cart - sigma_v_rotation @ center_cart),
            kind="mirror",
            order=2,
            det=-1,
            trace=float(np.trace(sigma_v_rotation)),
            angle_deg=None,
            symbol="sigma_v",
        )
    )
    planes.append(
        MolecularPlaneElement(
            point_cart=center_cart,
            normal_cart=normal,
            basis_cart=np.column_stack((direction, basis2)),
            operation_indices=(sigma_v_index,),
            kind="mirror",
        )
    )

    if point_group.startswith("D"):
        sigma_h_index = start_index + len(ops)
        sigma_h_rotation = reflection_matrix(direction)
        ops.append(
            MolecularSymmetryOperationInfo(
                index=sigma_h_index,
                rotation=sigma_h_rotation,
                translation=clean_vector(center_cart - sigma_h_rotation @ center_cart),
                kind="mirror",
                order=2,
                det=-1,
                trace=float(np.trace(sigma_h_rotation)),
                angle_deg=None,
                symbol="sigma_h",
            )
        )
        planes.append(
            MolecularPlaneElement(
                point_cart=center_cart,
                normal_cart=direction,
                basis_cart=plane_basis_from_normal(direction),
                operation_indices=(sigma_h_index,),
                kind="mirror",
            )
        )

    return tuple(ops), axes, planes


def molecule_line_direction(molecule: Molecule, center_cart: np.ndarray) -> np.ndarray | None:
    coords = np.asarray(molecule.cart_coords, dtype=float) - np.asarray(center_cart, dtype=float)
    if len(coords) < 2 or np.linalg.norm(coords) < TOL:
        return None
    _, singular_values, vh = np.linalg.svd(coords, full_matrices=False)
    if len(singular_values) == 0 or singular_values[0] < TOL:
        return None
    if len(singular_values) > 1 and singular_values[1] > max(1e-5, singular_values[0] * 1e-5):
        return None
    return canonical_direction(vh[0])


def reflection_matrix(normal: np.ndarray) -> np.ndarray:
    normal = normalize(normal)
    return clean_matrix(np.eye(3) - 2.0 * np.outer(normal, normal))


def molecule_center_from_analyzer(molecule: Molecule, analyzer: PointGroupAnalyzer) -> np.ndarray:
    centered = getattr(analyzer, "centered_mol", None)
    if centered is not None and len(centered) == len(molecule):
        offsets = np.asarray(molecule.cart_coords, dtype=float) - np.asarray(centered.cart_coords, dtype=float)
        return clean_vector(np.mean(offsets, axis=0))
    return clean_vector(np.mean(np.asarray(molecule.cart_coords, dtype=float), axis=0))


def classify_molecular_operation(
    rotation: np.ndarray,
    det: int,
    trace: float,
    order: int | None,
) -> str:
    identity = np.eye(3)
    if np.linalg.norm(rotation - identity) < 1e-6:
        return "identity"
    if np.linalg.norm(rotation + identity) < 1e-6:
        return "inversion"
    if det == 1:
        return "rotation_unknown" if order is None else f"rotation_{order}"
    if det == -1 and abs(trace - 1.0) < 1e-3:
        return "mirror"
    if det == -1:
        return "improper_unknown" if order is None else f"improper_{order}"
    return "unknown"


def operation_symbol(kind: str, order: int | None) -> str:
    if kind == "identity":
        return "E"
    if kind == "inversion":
        return "i"
    if kind == "mirror":
        return "sigma"
    if kind.startswith("rotation_"):
        return f"C{order}" if order is not None else "C∞"
    if kind.startswith("improper_"):
        return f"S{order}" if order is not None else "S∞"
    return kind


def canonical_reflection_matrix(matrix: np.ndarray) -> np.ndarray:
    """Remove small rotational noise from an approximate mirror operation."""
    matrix = np.asarray(matrix, dtype=float)
    values, vectors = np.linalg.eigh(0.5 * (matrix + matrix.T))
    normal = canonical_direction(vectors[:, int(np.argmin(values))])
    reflection = np.eye(3) - 2.0 * np.outer(normal, normal)
    return clean_matrix(reflection)


def matrix_order(matrix: np.ndarray, max_order: int = 24) -> int | None:
    identity = np.eye(3)
    power = np.eye(3)
    for order in range(1, max_order + 1):
        power = power @ matrix
        if np.linalg.norm(power - identity) < 1e-5:
            return order
    return None


def nullspace(matrix: np.ndarray, tol: float = 1e-8) -> np.ndarray:
    _, singular_values, vh = np.linalg.svd(np.asarray(matrix, dtype=float))
    rank = int(np.sum(singular_values > tol))
    basis = vh[rank:].T.copy()
    basis[np.abs(basis) < 1e-12] = 0.0
    return basis


def first_null_vector(matrix: np.ndarray) -> np.ndarray | None:
    basis = nullspace(matrix)
    if basis.shape[1] == 0:
        return None
    return canonical_direction(basis[:, 0])


def canonical_direction(vector: np.ndarray) -> np.ndarray:
    vector = normalize(vector)
    for value in vector:
        if abs(value) > TOL:
            return clean_vector(vector if value > 0 else -vector)
    return clean_vector(vector)


def canonical_plane_basis(basis: np.ndarray) -> np.ndarray:
    v1 = normalize(basis[:, 0])
    v2 = basis[:, 1] - np.dot(basis[:, 1], v1) * v1
    v2 = normalize(v2)
    return np.column_stack((canonical_direction(v1), canonical_direction(v2)))


def plane_basis_from_normal(normal: np.ndarray) -> np.ndarray:
    v1, v2 = plane_basis_from_normal_cart(normal)
    return np.column_stack((canonical_direction(v1), canonical_direction(v2)))


def normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(vector)
    if norm < TOL:
        return vector
    return vector / norm


def merge_axis(axes: list[MolecularAxisElement], candidate: MolecularAxisElement) -> None:
    for i, axis in enumerate(axes):
        if axis.kind == candidate.kind and parallel(axis.direction_cart, candidate.direction_cart):
            axes[i] = MolecularAxisElement(
                point_cart=axis.point_cart,
                direction_cart=axis.direction_cart,
                operation_indices=axis.operation_indices + candidate.operation_indices,
                kind=axis.kind,
                order=axis.order,
            )
            return
    axes.append(candidate)


def merge_plane(planes: list[MolecularPlaneElement], candidate: MolecularPlaneElement) -> None:
    for i, plane in enumerate(planes):
        if plane.kind == candidate.kind and parallel(plane.normal_cart, candidate.normal_cart):
            planes[i] = MolecularPlaneElement(
                point_cart=plane.point_cart,
                normal_cart=plane.normal_cart,
                basis_cart=plane.basis_cart,
                operation_indices=plane.operation_indices + candidate.operation_indices,
                kind=plane.kind,
            )
            return
    planes.append(candidate)


def merge_center(centers: list[MolecularCenterElement], candidate: MolecularCenterElement) -> None:
    for i, center in enumerate(centers):
        if np.linalg.norm(center.point_cart - candidate.point_cart) < 1e-6 and center.kind == candidate.kind:
            centers[i] = MolecularCenterElement(
                point_cart=center.point_cart,
                operation_indices=center.operation_indices + candidate.operation_indices,
                kind=center.kind,
            )
            return
    centers.append(candidate)


def parallel(a: np.ndarray, b: np.ndarray) -> bool:
    return np.linalg.norm(np.cross(normalize(a), normalize(b))) < 1e-6


def clean_matrix(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    matrix[np.abs(matrix) < 1e-12] = 0.0
    return matrix


def clean_vector(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    vector[np.abs(vector) < 1e-12] = 0.0
    return vector
