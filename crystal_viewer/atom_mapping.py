from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .analysis_models import (
    MoleculeAnalysisResult,
    MolecularSymmetryOperationInfo,
    StructureAnalysisResult,
    SymmetryOperationInfo,
)


MappingMode = Literal["crystal", "molecule"]

# All 27 integer shifts in [-1, 0, 1]^3, precomputed once.
_PERIODIC_SHIFTS = np.array(
    [(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)],
    dtype=float,
)


@dataclass(frozen=True)
class AtomMappingEntry:
    source_atom: int
    target_atom: int | None
    distance: float
    # Crystal mode stores the nearest periodic image here so animation does not
    # jump across cell boundaries. Use transformed_frac for the raw W*x+t result.
    transformed_cart: np.ndarray
    transformed_frac: np.ndarray | None = None
    wrapped_frac: np.ndarray | None = None
    animation_frac: np.ndarray | None = None


@dataclass(frozen=True)
class OperationAtomMapping:
    mode: MappingMode
    operation_index: int
    operation_kind: str
    entries: tuple[AtomMappingEntry, ...]
    max_distance: float
    unmatched_atoms: tuple[int, ...]

    @property
    def is_complete(self) -> bool:
        return not self.unmatched_atoms

    @property
    def atom_to_atom(self) -> tuple[int | None, ...]:
        return tuple(entry.target_atom for entry in self.entries)


@dataclass(frozen=True)
class AtomMappingSet:
    mode: MappingMode
    mappings: tuple[OperationAtomMapping, ...]

    @property
    def is_complete(self) -> bool:
        return all(mapping.is_complete for mapping in self.mappings)

    @property
    def incomplete_operation_indices(self) -> tuple[int, ...]:
        return tuple(mapping.operation_index for mapping in self.mappings if not mapping.is_complete)


def atom_mappings_from_analysis(
    result: StructureAnalysisResult | MoleculeAnalysisResult,
    *,
    tolerance_cart: float = 1e-2,
) -> AtomMappingSet:
    if isinstance(result, StructureAnalysisResult):
        return atom_mappings_from_crystal(result, tolerance_cart=tolerance_cart)
    if isinstance(result, MoleculeAnalysisResult):
        return atom_mappings_from_molecule(result, tolerance_cart=tolerance_cart)
    raise TypeError(f"unsupported analysis result type: {type(result).__name__}")


def atom_mappings_from_crystal(
    result: StructureAnalysisResult,
    *,
    tolerance_cart: float = 1e-2,
) -> AtomMappingSet:
    mappings = tuple(
        crystal_operation_mapping(result, operation, tolerance_cart=tolerance_cart)
        for operation in result.operations
    )
    return AtomMappingSet(mode="crystal", mappings=mappings)


def atom_mappings_from_molecule(
    result: MoleculeAnalysisResult,
    *,
    tolerance_cart: float = 1e-2,
) -> AtomMappingSet:
    mappings = tuple(
        molecule_operation_mapping(result, operation, tolerance_cart=tolerance_cart)
        for operation in result.operations
    )
    return AtomMappingSet(mode="molecule", mappings=mappings)


def crystal_operation_mapping(
    result: StructureAnalysisResult,
    operation: SymmetryOperationInfo,
    *,
    tolerance_cart: float,
) -> OperationAtomMapping:
    lattice = np.asarray(result.structure.lattice, dtype=float)
    atoms = result.structure.atoms
    atoms_by_atomic_number = atoms_grouped_by_atomic_number(atoms)
    entries = []
    unmatched = []

    for atom in atoms:
        if atom.frac is None:
            raise ValueError("crystal atom is missing fractional coordinates")

        start_frac = np.asarray(atom.frac, dtype=float)
        transformed_frac = operation.W @ start_frac + operation.t
        wrapped_frac = wrap_frac(transformed_frac)
        target_atom, distance = find_matching_crystal_atom(
            wrapped_frac,
            atom.atomic_number,
            atoms_by_atomic_number,
            lattice,
        )
        if target_atom is None or distance > tolerance_cart:
            unmatched.append(atom.index)
            target_index = None
        else:
            target_index = target_atom.index

        animation_frac = choose_nearest_periodic_image(start_frac, transformed_frac, lattice)
        entries.append(
            AtomMappingEntry(
                source_atom=atom.index,
                target_atom=target_index,
                distance=float(distance),
                transformed_frac=np.asarray(transformed_frac, dtype=float),
                wrapped_frac=wrapped_frac,
                animation_frac=animation_frac,
                transformed_cart=animation_frac @ lattice,
            )
        )

    return OperationAtomMapping(
        mode="crystal",
        operation_index=operation.index,
        operation_kind=operation.kind,
        entries=tuple(entries),
        max_distance=max((entry.distance for entry in entries), default=0.0),
        unmatched_atoms=tuple(unmatched),
    )


def molecule_operation_mapping(
    result: MoleculeAnalysisResult,
    operation: MolecularSymmetryOperationInfo,
    *,
    tolerance_cart: float,
) -> OperationAtomMapping:
    atoms = result.molecule.atoms
    atoms_by_atomic_number = atoms_grouped_by_atomic_number(atoms)
    entries = []
    unmatched = []

    for atom in atoms:
        start_cart = np.asarray(atom.cart, dtype=float)
        transformed_cart = operation.rotation @ start_cart + operation.translation
        target_atom, distance = find_matching_molecule_atom(
            transformed_cart,
            atom.atomic_number,
            atoms_by_atomic_number,
        )
        if target_atom is None or distance > tolerance_cart:
            unmatched.append(atom.index)
            target_index = None
        else:
            target_index = target_atom.index

        entries.append(
            AtomMappingEntry(
                source_atom=atom.index,
                target_atom=target_index,
                distance=float(distance),
                transformed_cart=transformed_cart,
            )
        )

    return OperationAtomMapping(
        mode="molecule",
        operation_index=operation.index,
        operation_kind=operation.kind,
        entries=tuple(entries),
        max_distance=max((entry.distance for entry in entries), default=0.0),
        unmatched_atoms=tuple(unmatched),
    )


def find_matching_crystal_atom(
    wrapped_frac: np.ndarray,
    atomic_number: int,
    atoms_by_atomic_number,
    lattice: np.ndarray,
):
    best_atom = None
    best_distance = float("inf")
    for candidate in atoms_by_atomic_number.get(atomic_number, ()):
        if candidate.frac is None:
            continue
        delta_frac = wrapped_frac - np.asarray(candidate.frac, dtype=float)
        delta_frac = delta_frac - np.round(delta_frac)
        distance = np.linalg.norm(delta_frac @ lattice)
        if distance < best_distance:
            best_atom = candidate
            best_distance = float(distance)
    return best_atom, best_distance


def find_matching_molecule_atom(
    transformed_cart: np.ndarray,
    atomic_number: int,
    atoms_by_atomic_number,
):
    best_atom = None
    best_distance = float("inf")
    for candidate in atoms_by_atomic_number.get(atomic_number, ()):
        distance = np.linalg.norm(transformed_cart - np.asarray(candidate.cart, dtype=float))
        if distance < best_distance:
            best_atom = candidate
            best_distance = float(distance)
    return best_atom, best_distance


def atoms_grouped_by_atomic_number(atoms) -> dict[int, tuple]:
    grouped: dict[int, list] = {}
    for atom in atoms:
        grouped.setdefault(atom.atomic_number, []).append(atom)
    return {atomic_number: tuple(items) for atomic_number, items in grouped.items()}


def choose_nearest_periodic_image(
    start_frac: np.ndarray,
    target_frac: np.ndarray,
    lattice: np.ndarray,
) -> np.ndarray:
    candidates = target_frac + _PERIODIC_SHIFTS  # (27, 3)
    disp = (candidates - start_frac) @ lattice   # (27, 3)
    sq_distances = np.einsum("ij,ij->i", disp, disp)  # (27,) squared norms
    return np.asarray(candidates[np.argmin(sq_distances)], dtype=float)


def wrap_frac(frac: np.ndarray) -> np.ndarray:
    wrapped = np.mod(np.asarray(frac, dtype=float), 1.0)
    wrapped[np.abs(wrapped) < 1e-12] = 0.0
    wrapped[np.abs(wrapped - 1.0) < 1e-12] = 0.0
    return wrapped
