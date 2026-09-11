from __future__ import annotations

import numpy as np

from crystal_viewer.geometry import normalize, periodic_shifts


def glide_translation_frac(render_data: dict, operation: dict, plane: dict) -> np.ndarray | None:
    unit_cell = render_data.get("unit_cell")
    matrix = operation.get("matrix_cart")
    translation = operation.get("translation_cart")
    if unit_cell is None or matrix is None or translation is None:
        return None

    lattice = np.asarray(unit_cell["lattice"], dtype=float)
    matrix = np.asarray(matrix, dtype=float)
    translation = np.asarray(translation, dtype=float)
    point = np.asarray(plane["point_cart"], dtype=float)
    normal = normalize(np.asarray(plane["normal_cart"], dtype=float))

    best: tuple[float, float, np.ndarray] | None = None
    for shift in periodic_shifts(1):
        displacement = matrix @ point + translation + shift @ lattice - point
        normal_distance = abs(float(np.dot(displacement, normal)))
        displacement_norm = float(np.linalg.norm(displacement))
        score = (normal_distance, displacement_norm, displacement)
        if best is None or score[:2] < best[:2]:
            best = score

    if best is None or best[0] > 1e-5 or best[1] < 1e-10:
        return None
    return best[2] @ np.linalg.inv(lattice)


def centered_fractional_vector(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    centered = values - np.floor(values + 0.5)
    centered[np.isclose(centered, -0.5, atol=1e-8)] = 0.5
    return centered


def align_fractional_vector_to_reference(
    vector_frac: np.ndarray,
    reference_frac: np.ndarray,
    lattice: np.ndarray,
) -> np.ndarray:
    reference_cart = np.asarray(reference_frac, dtype=float) @ lattice
    reference_norm = float(np.linalg.norm(reference_cart))
    if reference_norm < 1e-10:
        return vector_frac

    best: tuple[float, float, np.ndarray] | None = None
    for shift in periodic_shifts(1):
        candidate = np.asarray(vector_frac, dtype=float) + shift
        candidate_cart = candidate @ lattice
        candidate_norm = float(np.linalg.norm(candidate_cart))
        if candidate_norm < 1e-10:
            continue
        cosine = float(np.dot(candidate_cart, reference_cart) / (candidate_norm * reference_norm))
        score = (-cosine, candidate_norm, candidate)
        if best is None or score[:2] < best[:2]:
            best = score
    return vector_frac if best is None else best[2]

def periodic_shift_vectors(radius: int) -> np.ndarray:
    return periodic_shifts(radius)
