from __future__ import annotations

from functools import lru_cache
from itertools import product

import numpy as np


def normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(vector)
    if norm < 1e-12:
        return vector
    return vector / norm


def interpolate(start: np.ndarray, target: np.ndarray, s: float) -> np.ndarray:
    return (1.0 - s) * start + s * target


def rotate_about_axis(
    point: np.ndarray,
    axis_point: np.ndarray,
    axis_direction: np.ndarray,
    angle_rad: float,
) -> np.ndarray:
    direction = normalize(axis_direction)
    relative = point - axis_point
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    rotated = (
        relative * cos_a
        + np.cross(direction, relative) * sin_a
        + direction * np.dot(direction, relative) * (1.0 - cos_a)
    )
    return axis_point + rotated


def signed_angle_to_target(
    start: np.ndarray,
    target: np.ndarray,
    axis_point: np.ndarray,
    axis_direction: np.ndarray,
    angle_deg: float,
) -> float:
    angle = np.deg2rad(float(angle_deg))
    plus = rotate_about_axis(start, axis_point, axis_direction, angle)
    minus = rotate_about_axis(start, axis_point, axis_direction, -angle)
    return angle if np.linalg.norm(target - plus) <= np.linalg.norm(target - minus) else -angle


def reflect_point(point: np.ndarray, plane_point: np.ndarray, plane_normal: np.ndarray) -> np.ndarray:
    normal = normalize(plane_normal)
    return point - 2.0 * np.dot(point - plane_point, normal) * normal


def rotation_angle_deg(rotation: np.ndarray) -> float:
    trace = float(np.trace(rotation))
    cos_angle = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    angle = float(np.degrees(np.arccos(cos_angle)))
    if abs(angle) < 1e-8:
        return 0.0
    return angle


def rotation_axis_sin_component(rotation: np.ndarray, axis_direction: np.ndarray) -> float:
    """``dot(axis, antisymmetric part of rotation)`` — proportional to sin(theta)
    for a proper rotation by theta about ``axis``.

    Also valid for an *improper* rotation (rotation + reflection through the
    plane perpendicular to the axis, e.g. Sn): the reflection's eigenvalue
    along the axis is purely diagonal, so it does not leak into the
    antisymmetric part this reads. Unlike ``signed_rotation_angle_from_matrix``
    (whose cos(theta), read from the trace, is off by the reflection's -1 and
    can round to the wrong branch), the *sign* of this value alone is a safe
    proper/improper-agnostic way to tell a rotation from its mirror-image twin
    (e.g. S3 from S3²) apart.
    """
    rotation = np.asarray(rotation, dtype=float)
    axis = normalize(axis_direction)
    return 0.5 * float(np.dot(
        axis,
        np.array(
            [
                rotation[2, 1] - rotation[1, 2],
                rotation[0, 2] - rotation[2, 0],
                rotation[1, 0] - rotation[0, 1],
            ]
        ),
    ))


def signed_rotation_angle_from_matrix(rotation: np.ndarray, axis_direction: np.ndarray) -> float:
    rotation = np.asarray(rotation, dtype=float)
    sin_angle = rotation_axis_sin_component(rotation, axis_direction)
    cos_angle = np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)
    if np.isclose(cos_angle, -1.0, atol=1e-8):
        return float(np.pi)
    return float(np.arctan2(sin_angle, cos_angle))


def plane_basis_from_normal_cart(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    normal = normalize(normal)
    helper = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(normal, helper))) > 0.9:
        helper = np.array([0.0, 1.0, 0.0])
    basis1 = normalize(np.cross(normal, helper))
    basis2 = normalize(np.cross(normal, basis1))
    return basis1, basis2


def integer_index_vector(values: np.ndarray, *, orient_positive: bool = True) -> np.ndarray | None:
    values = np.asarray(values, dtype=float)
    max_abs = float(np.max(np.abs(values)))
    if max_abs < 1e-10:
        return None
    scaled = values / max_abs
    best: np.ndarray | None = None
    best_error = float("inf")
    for limit in range(1, 13):
        candidate = np.rint(scaled * limit).astype(int)
        if not np.any(candidate):
            continue
        normalized = candidate / max(float(np.max(np.abs(candidate))), 1.0)
        error = float(np.linalg.norm(normalized - scaled))
        if error < best_error:
            best = candidate
            best_error = error
    if best is None or best_error > 1e-5:
        return None
    gcd = int(np.gcd.reduce(np.abs(best[np.nonzero(best)])))
    if gcd > 1:
        best = best // gcd
    if orient_positive:
        for value in best:
            if value < 0:
                best = -best
                break
            if value > 0:
                break
    return best


@lru_cache(maxsize=None)
def periodic_shifts(limit: int = 1) -> np.ndarray:
    limit = max(int(limit), 0)
    shifts = np.asarray(
        list(product(range(-limit, limit + 1), repeat=3)),
        dtype=float,
    )
    shifts.flags.writeable = False
    return shifts
