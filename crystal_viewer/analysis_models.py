from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class AtomSite:
    index: int
    element: str
    atomic_number: int
    frac: np.ndarray | None
    cart: np.ndarray
    asymmetric_index: int | None = None
    generation_operation_index: int | None = None


@dataclass(frozen=True)
class AsymmetricUnitSite:
    index: int
    label: str
    element: str
    atomic_number: int
    frac: np.ndarray
    cart: np.ndarray


@dataclass(frozen=True)
class StructureSummary:
    source_file: Path
    formula: str
    site_count: int
    lattice: np.ndarray
    atoms: tuple[AtomSite, ...]
    asymmetric_atoms: tuple[AsymmetricUnitSite, ...] = ()


@dataclass(frozen=True)
class SymmetryOperationInfo:
    index: int
    W: np.ndarray
    t: np.ndarray
    kind: str
    order: int | None
    det: int
    trace: int
    angle_deg: float | None
    international_symbol: str


@dataclass(frozen=True)
class AxisElement:
    point_frac: np.ndarray
    direction_frac: np.ndarray
    operation_indices: tuple[int, ...]
    motion_coeffs: tuple[np.ndarray | None, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class PlaneElement:
    point_frac: np.ndarray
    normal_frac: np.ndarray
    basis_frac: np.ndarray
    operation_indices: tuple[int, ...]
    motion_coeffs: tuple[np.ndarray | None, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class CenterElement:
    point_frac: np.ndarray
    operation_indices: tuple[int, ...]


@dataclass(frozen=True)
class SpaceGroupInfo:
    number: int
    international: str
    hall: str
    point_group: str
    operation_count: int
    wyckoffs: tuple[str, ...]
    equivalent_atoms: tuple[int, ...]
    site_symmetry_symbols: tuple[str, ...]


@dataclass(frozen=True)
class StructureAnalysisResult:
    structure: StructureSummary
    space_group: SpaceGroupInfo
    operations: tuple[SymmetryOperationInfo, ...]
    axes: tuple[AxisElement, ...]
    planes: tuple[PlaneElement, ...]
    centers: tuple[CenterElement, ...]
    geometry_groups: dict
    raw_merged: dict
    raw_per_operation: list
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class MoleculeSummary:
    source_file: Path | None
    formula: str
    site_count: int
    atoms: tuple[AtomSite, ...]
    center_cart: np.ndarray
    # `formula` is the reduced formula, which is wrong for a molecule (benzene
    # reduces to HC).  `display_formula` is the formula as a chemist writes it.
    display_formula: str = ""


@dataclass(frozen=True)
class MolecularSymmetryOperationInfo:
    index: int
    rotation: np.ndarray
    translation: np.ndarray
    kind: str
    order: int | None
    det: int
    trace: float
    angle_deg: float | None
    symbol: str


@dataclass(frozen=True)
class MolecularAxisElement:
    point_cart: np.ndarray
    direction_cart: np.ndarray
    operation_indices: tuple[int, ...]
    kind: str
    order: int | None


@dataclass(frozen=True)
class MolecularPlaneElement:
    point_cart: np.ndarray
    normal_cart: np.ndarray
    basis_cart: np.ndarray
    operation_indices: tuple[int, ...]
    kind: str


@dataclass(frozen=True)
class MolecularCenterElement:
    point_cart: np.ndarray
    operation_indices: tuple[int, ...]
    kind: str


@dataclass(frozen=True)
class MoleculePointGroupInfo:
    symbol: str
    operation_count: int
    rotational_symmetry_number: int
    equivalent_atom_sets: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class MoleculeAnalysisResult:
    molecule: MoleculeSummary
    point_group: MoleculePointGroupInfo
    operations: tuple[MolecularSymmetryOperationInfo, ...]
    axes: tuple[MolecularAxisElement, ...]
    planes: tuple[MolecularPlaneElement, ...]
    centers: tuple[MolecularCenterElement, ...]
