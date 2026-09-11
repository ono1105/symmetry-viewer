from __future__ import annotations

import numpy as np

from crystal_viewer.geometry import (
    interpolate,
    normalize,
    plane_basis_from_normal_cart,
    reflect_point,
    rotate_about_axis,
    rotation_angle_deg,
    signed_angle_to_target,
    signed_rotation_angle_from_matrix,
)
from crystal_viewer.viewer.display_atoms import display_atom_instances


NORMALIZED_TRAVEL_SPEED_PER_SECOND = 0.525
MINIMUM_ANIMATION_SECONDS = 1.0
MAXIMUM_ANIMATION_SECONDS = 16.0 / 3.0
STATIONARY_ANIMATION_SECONDS = 0.4


def effective_rotation_axis(operation: dict, axis: dict | None, center: dict | None) -> dict | None:
    if axis is not None:
        return axis
    kind = str(operation["kind"])
    if "rotoinversion" not in kind or center is None:
        return None
    matrix = operation.get("matrix_cart")
    if matrix is None:
        return None
    rotation = -np.asarray(matrix, dtype=float)
    direction = rotation_axis_from_matrix(rotation)
    if direction is None:
        return None
    return {
        "point_cart": center["point_cart"],
        "direction_cart": direction,
    }

def operation_angle_deg(operation: dict) -> float | None:
    angle = operation.get("angle_deg")
    if angle is not None:
        return float(angle)
    kind = str(operation["kind"])
    matrix = operation.get("matrix_cart")
    if matrix is None:
        return None
    rotation = operation_rotation_matrix(operation)
    if rotation is None:
        return None
    return rotation_angle_deg(rotation)

def operation_rotation_matrix(operation: dict) -> np.ndarray | None:
    kind = str(operation["kind"])
    matrix = operation.get("matrix_cart")
    if matrix is None:
        return None
    matrix = np.asarray(matrix, dtype=float)
    if "rotoinversion" in kind:
        return -matrix
    if kind.startswith(("rotation", "screw")):
        return matrix
    return None

def rotation_axis_from_matrix(rotation: np.ndarray) -> np.ndarray | None:
    values, vectors = np.linalg.eig(np.asarray(rotation, dtype=float))
    index = int(np.argmin(np.abs(values - 1.0)))
    if np.abs(values[index] - 1.0) > 1e-5:
        return None
    return normalize(np.real(vectors[:, index]))

def build_operation_path(
    start: np.ndarray,
    target: np.ndarray,
    operation: dict,
    *,
    axis: dict | None,
    plane: dict | None,
    center: dict | None,
    angle_override: float | None = None,
    translation_override: np.ndarray | None = None,
    improper_mode: str = "auto",
    source_kind: str = "",
) -> dict:
    kind = str(operation["kind"])
    axis = effective_rotation_axis(operation, axis, center)
    geometric_kind = (
        kind.startswith(("rotation", "screw"))
        or kind in ("mirror", "inversion")
        or "glide" in kind
        or "rotoinversion" in kind
        or "rotoreflection" in kind
        or "improper" in kind
    )
    if not geometric_kind and np.linalg.norm(target - start) < 1e-10:
        return translation_path(start, target)
    if kind.startswith("screw"):
        return screw_path(
            start,
            target,
            operation,
            axis,
            angle_override=angle_override,
            translation_override=translation_override,
        )
    if "glide" in kind:
        return glide_path(start, target, plane, translation_override=translation_override)
    if "rotoinversion" in kind or "rotoreflection" in kind or "improper" in kind:
        return improper_path(
            start,
            target,
            operation,
            axis,
            plane,
            center,
            angle_override=angle_override,
            improper_mode=improper_mode,
            source_kind=source_kind,
        )
    if kind.startswith("rotation"):
        return rotation_path(start, target, operation, axis, angle_override=angle_override)
    if kind == "mirror":
        return mirror_path(start, target, plane)
    if kind == "inversion":
        return inversion_path(start, target, center)
    return translation_path(start, target)

def rotation_path(
    start: np.ndarray,
    target: np.ndarray,
    operation: dict,
    axis: dict | None,
    *,
    angle_override: float | None = None,
) -> dict:
    angle_deg = operation_angle_deg(operation)
    if axis is None or (angle_deg is None and angle_override is None):
        return translation_path(start, target)
    axis_point = np.asarray(axis["point_cart"], dtype=float)
    axis_direction = normalize(np.asarray(axis["direction_cart"], dtype=float))
    angle = (
        angle_override
        if angle_override is not None
        else signed_angle_to_target(start, target, axis_point, axis_direction, angle_deg)
    )
    return {
        "type": "rotation",
        "start": start,
        "target": target,
        "axis_point": axis_point,
        "axis_direction": axis_direction,
        "angle": angle,
    }

def screw_path(
    start: np.ndarray,
    target: np.ndarray,
    operation: dict,
    axis: dict | None,
    *,
    angle_override: float | None = None,
    translation_override: np.ndarray | None = None,
) -> dict:
    rotated_path = rotation_path(start, target, operation, axis, angle_override=angle_override)
    if rotated_path["type"] != "rotation":
        return translation_path(start, target)
    rotated_end = rotate_about_axis(
        start,
        rotated_path["axis_point"],
        rotated_path["axis_direction"],
        rotated_path["angle"],
    )
    return {
        "type": "screw",
        "start": start,
        "target": target,
        "axis_point": rotated_path["axis_point"],
        "axis_direction": rotated_path["axis_direction"],
        "angle": rotated_path["angle"],
        "translation": target - rotated_end if translation_override is None else np.asarray(translation_override, dtype=float),
    }

def mirror_path(start: np.ndarray, target: np.ndarray, plane: dict | None) -> dict:
    if plane is None:
        return translation_path(start, target)
    point = np.asarray(plane["point_cart"], dtype=float)
    normal = normalize(np.asarray(plane["normal_cart"], dtype=float))
    return {
        "type": "mirror",
        "start": start,
        "target": target,
        "plane_point": point,
        "plane_normal": normal,
    }

def glide_path(
    start: np.ndarray,
    target: np.ndarray,
    plane: dict | None,
    *,
    translation_override: np.ndarray | None = None,
) -> dict:
    mirrored_path = mirror_path(start, target, plane)
    if mirrored_path["type"] != "mirror":
        return translation_path(start, target)
    mirrored_end = reflect_point(start, mirrored_path["plane_point"], mirrored_path["plane_normal"])
    return {
        "type": "glide",
        "start": start,
        "target": target,
        "plane_point": mirrored_path["plane_point"],
        "plane_normal": mirrored_path["plane_normal"],
        "translation": target - mirrored_end if translation_override is None else np.asarray(translation_override, dtype=float),
    }

def inversion_path(start: np.ndarray, target: np.ndarray, center: dict | None) -> dict:
    center_point = None if center is None else np.asarray(center["point_cart"], dtype=float)
    if center_point is None:
        return translation_path(start, target)
    return {
        "type": "inversion",
        "start": start,
        "target": target,
        "center": center_point,
    }

def improper_path(
    start: np.ndarray,
    target: np.ndarray,
    operation: dict,
    axis: dict | None,
    plane: dict | None,
    center: dict | None,
    *,
    angle_override: float | None = None,
    improper_mode: str = "auto",
    source_kind: str = "",
) -> dict:
    mode = preferred_improper_mode(operation, improper_mode, source_kind)
    plane = plane or improper_reflection_plane(axis)
    center = center or improper_inversion_center(axis)
    if mode == "rotoreflection" and (operation.get("order") is None or "infinite" in str(operation.get("kind", ""))):
        return mirror_after_hold_path(start, target, plane)
    if angle_override is None:
        angle_override = improper_rotation_angle(operation, axis, mode)
    rotated = rotation_path(start, target, operation, axis, angle_override=angle_override)
    if rotated["type"] != "rotation":
        if mode == "rotoinversion":
            return inversion_path(start, target, center)
        return mirror_path(start, target, plane)
    if mode == "rotoinversion":
        if center is None:
            return translation_path(start, target)
        return {
            "type": "rotoinversion",
            "start": start,
            "target": target,
            "axis_point": rotated["axis_point"],
            "axis_direction": rotated["axis_direction"],
            "angle": rotated["angle"],
            "center": np.asarray(center["point_cart"], dtype=float),
        }
    if plane is None:
        return translation_path(start, target)
    return {
        "type": "rotoreflection",
        "start": start,
        "target": target,
        "axis_point": rotated["axis_point"],
        "axis_direction": rotated["axis_direction"],
        "angle": rotated["angle"],
        "plane_point": np.asarray(plane["point_cart"], dtype=float),
        "plane_normal": normalize(np.asarray(plane["normal_cart"], dtype=float)),
    }

def mirror_after_hold_path(start: np.ndarray, target: np.ndarray, plane: dict | None) -> dict:
    mirrored = mirror_path(start, target, plane)
    if mirrored["type"] != "mirror":
        return translation_path(start, target)
    mirrored["type"] = "mirror_after_hold"
    mirrored["hold_fraction"] = 0.3
    return mirrored

def preferred_improper_mode(operation: dict, requested: str, source_kind: str) -> str:
    if requested in ("rotoreflection", "rotoinversion"):
        return requested
    kind = str(operation.get("kind", ""))
    if "rotoreflection" in kind or "improper" in kind:
        return "rotoreflection" if source_kind == "molecule" else "rotoinversion"
    return "rotoinversion"

def render_source_kind(render_data: dict) -> str:
    return str((render_data.get("metadata") or {}).get("mode") or "crystal")

def improper_reflection_plane(axis: dict | None) -> dict | None:
    if axis is None:
        return None
    normal = normalize(np.asarray(axis["direction_cart"], dtype=float))
    basis1, basis2 = plane_basis_from_normal_cart(normal)
    return {
        "point_cart": axis["point_cart"],
        "normal_cart": normal,
        "basis1_cart": basis1,
        "basis2_cart": basis2,
        "label": "rotoreflection mirror plane",
        "operation_indices": axis.get("operation_indices", ()),
    }

def improper_inversion_center(axis: dict | None) -> dict | None:
    if axis is None:
        return None
    return {"point_cart": axis["point_cart"]}

def improper_rotation_angle(operation: dict, axis: dict | None, mode: str) -> float | None:
    if axis is None or operation.get("matrix_cart") is None:
        return None
    matrix = np.asarray(operation["matrix_cart"], dtype=float)
    axis_direction = normalize(np.asarray(axis["direction_cart"], dtype=float))
    if mode == "rotoreflection":
        normal = axis_direction.reshape(3, 1)
        reflection = np.eye(3) - 2.0 * (normal @ normal.T)
        rotation = reflection @ matrix
    else:
        rotation = -matrix
    return signed_rotation_angle_from_matrix(rotation, axis_direction)

def translation_path(start: np.ndarray, target: np.ndarray) -> dict:
    return {"type": "linear", "start": start, "target": target}


def rotation_arc_length(path: dict, start: np.ndarray) -> float:
    axis_point = np.asarray(path["axis_point"], dtype=float)
    axis_direction = normalize(np.asarray(path["axis_direction"], dtype=float))
    relative = np.asarray(start, dtype=float) - axis_point
    radial = relative - np.dot(relative, axis_direction) * axis_direction
    return float(np.linalg.norm(radial) * abs(float(path["angle"])))


def two_phase_geometry(path: dict, start: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    path_type = str(path["type"])
    start = np.asarray(start, dtype=float)
    if path_type == "glide":
        midpoint = reflect_point(start, path["plane_point"], path["plane_normal"])
        endpoint = midpoint + np.asarray(path["translation"], dtype=float)
        first_length = float(np.linalg.norm(midpoint - start))
    else:
        midpoint = rotate_about_axis(
            start,
            path["axis_point"],
            path["axis_direction"],
            path["angle"],
        )
        first_length = rotation_arc_length(path, start)
        if path_type == "screw":
            endpoint = midpoint + np.asarray(path["translation"], dtype=float)
        elif path_type == "rotoinversion":
            endpoint = 2.0 * np.asarray(path["center"], dtype=float) - midpoint
        else:
            endpoint = reflect_point(midpoint, path["plane_point"], path["plane_normal"])
    second_length = float(np.linalg.norm(endpoint - midpoint))
    return midpoint, endpoint, first_length, second_length


def phase_fraction(first_length: float, second_length: float) -> float:
    total = first_length + second_length
    return first_length / total if total > 1e-12 else 0.5


def synchronize_compound_path_phases(paths: dict[int, dict]) -> None:
    """Give every atom in a compound operation the same phase boundary.

    Rotation/reflection travel depends on the atom's distance from the symmetry
    element.  Deriving the split independently therefore lets near atoms begin
    the second phase while farther atoms are still in the first phase.  Use the
    longest travel in each phase as the operation-wide timing reference.
    """
    sequential = [path for path in paths.values() if path.get("type") == "sequential"]
    if sequential:
        segment_count = min(len(path.get("segments", [])) for path in sequential)
        duration_lengths = []
        for index in range(segment_count):
            indexed = {key: path["segments"][index] for key, path in paths.items()
                       if path.get("type") == "sequential" and len(path.get("segments", [])) > index}
            synchronize_compound_path_phases(indexed)
            duration_lengths.append(max(
                animation_path_length(segment, start_override=np.asarray(segment["start"], dtype=float))
                for segment in indexed.values()
            ))
        total = sum(duration_lengths)
        weights = ([length / total for length in duration_lengths] if total > 1e-12
                   else [1.0 / segment_count] * segment_count) if segment_count else []
        for path in sequential:
            path["segment_weights"] = weights

    compound_types = {"screw", "glide", "rotoinversion", "rotoreflection"}
    phase_lengths = []
    compound_paths = []
    for path in paths.values():
        if path.get("type") not in compound_types:
            continue
        start = np.asarray(path["start"], dtype=float)
        _, _, first_length, second_length = two_phase_geometry(path, start)
        phase_lengths.append((first_length, second_length))
        compound_paths.append(path)
    if not compound_paths:
        return
    split = phase_fraction(
        max(lengths[0] for lengths in phase_lengths),
        max(lengths[1] for lengths in phase_lengths),
    )
    for path in compound_paths:
        path["phase_fraction"] = split


def evaluate_path(path: dict, s: float, start_override: np.ndarray | None = None) -> np.ndarray:
    s = float(np.clip(s, 0.0, 1.0))
    path_type = path["type"]
    if path_type == "sequential":
        segments = path["segments"]
        if not segments:
            raise ValueError("sequential path requires at least one segment")
        initial_start = segments[0]["start"] if start_override is None else start_override
        segment_start = np.asarray(initial_start, dtype=float)
        starts = []
        lengths = []
        for segment in segments:
            starts.append(segment_start)
            lengths.append(animation_path_length(segment, start_override=segment_start))
            segment_start = evaluate_path(segment, 1.0, start_override=segment_start)
        supplied_weights = path.get("segment_weights")
        weights = ([float(value) for value in supplied_weights]
                   if isinstance(supplied_weights, list) and len(supplied_weights) == len(segments)
                   else lengths)
        total = sum(weights)
        if total <= 1e-12:
            index = min(int(s * len(segments)), len(segments) - 1)
            local_s = (s * len(segments)) - index
        else:
            distance = s * total
            cumulative = 0.0
            index = len(segments) - 1
            local_s = 1.0
            for candidate, length in enumerate(weights):
                if distance <= cumulative + length or candidate == len(segments) - 1:
                    index = candidate
                    local_s = 1.0 if length <= 1e-12 else (distance - cumulative) / length
                    break
                cumulative += length
        return evaluate_path(segments[index], local_s, start_override=starts[index])
    start = np.asarray(path["start"], dtype=float) if start_override is None else np.asarray(start_override, dtype=float)
    if path_type == "affine_linear":
        if start_override is not None:
            W = np.asarray(path["matrix_cart"], dtype=float)
            t = np.asarray(path["translation_cart"], dtype=float)
            target = W @ start + t
        else:
            target = np.asarray(path["target"], dtype=float)
        return interpolate(start, target, s)
    if path_type == "rotation":
        rotated = rotate_about_axis(
            start,
            path["axis_point"],
            path["axis_direction"],
            path["angle"] * s,
        )
        return rotated
    if path_type in ("screw", "glide", "rotoinversion", "rotoreflection"):
        midpoint, endpoint, first_length, second_length = two_phase_geometry(path, start)
        split = float(path.get("phase_fraction", phase_fraction(first_length, second_length)))
        split = float(np.clip(split, 0.0, 1.0))
        if s <= split:
            local_s = s / split if split > 1e-12 else 1.0
            if path_type == "glide":
                return interpolate(start, midpoint, local_s)
            return rotate_about_axis(
                start,
                path["axis_point"],
                path["axis_direction"],
                path["angle"] * local_s,
            )
        local_s = (s - split) / (1.0 - split) if split < 1.0 - 1e-12 else 1.0
        return interpolate(midpoint, endpoint, local_s)
    if path_type == "mirror":
        mirrored = reflect_point(start, path["plane_point"], path["plane_normal"])
        return interpolate(start, mirrored, s)
    if path_type == "mirror_after_hold":
        hold_fraction = float(path.get("hold_fraction", 0.3))
        if s <= hold_fraction:
            return start.copy()
        local_s = (s - hold_fraction) / max(1.0 - hold_fraction, 1e-9)
        mirrored = reflect_point(start, path["plane_point"], path["plane_normal"])
        return interpolate(start, mirrored, local_s)
    if path_type == "inversion":
        inverted = 2.0 * path["center"] - start
        return interpolate(start, inverted, s)
    target = np.asarray(path["target"], dtype=float)
    if start_override is not None:
        target = target + (start - np.asarray(path["start"], dtype=float))
    return interpolate(start, target, s)


def animation_path_length(
    path: dict,
    *,
    start_override: np.ndarray | None = None,
) -> float:
    """Return the exact Cartesian arc length for a supported animation path."""
    path_type = str(path["type"])
    if path_type == "sequential":
        length = 0.0
        segments = path.get("segments", [])
        if not segments:
            return 0.0
        segment_start = np.asarray(
            segments[0]["start"] if start_override is None else start_override,
            dtype=float,
        )
        for segment in segments:
            length += animation_path_length(segment, start_override=segment_start)
            segment_start = evaluate_path(segment, 1.0, start_override=segment_start)
        return length
    start = np.asarray(
        path["start"] if start_override is None else start_override,
        dtype=float,
    )
    if path_type in ("rotation", "screw", "rotoinversion", "rotoreflection"):
        rotation_length = rotation_arc_length(path, start)
        if path_type == "rotation":
            return rotation_length
        _, _, _, second_length = two_phase_geometry(path, start)
        return rotation_length + second_length
    if path_type == "glide":
        _, _, first_length, second_length = two_phase_geometry(path, start)
        return first_length + second_length
    endpoint = np.asarray(evaluate_path(path, 1.0, start_override=start), dtype=float)
    return float(np.linalg.norm(endpoint - start))


def maximum_travel_distance(
    render_data: dict,
    paths: dict,
    *,
    display_mode: str,
    cell_origin_mode: str,
    include_boundary_images: bool,
    unit_cell_only: bool | None = None,
) -> float:
    """Longest Cartesian travel among the atom instances actually on screen.

    Playback time is derived from this, so it has to be measured over the
    *displayed* instances: a periodic clone starts somewhere else and therefore
    travels a different distance than the atom it was copied from.

    `unit_cell_only=None` reads the flag off each path; pass a bool when the
    caller has already decided it once for the whole response.
    """
    longest = 0.0
    for instance in display_atom_instances(
        render_data,
        display_mode=display_mode,
        cell_origin_mode=cell_origin_mode,
        include_boundary_images=include_boundary_images,
    ):
        path = paths.get(int(instance["atom"]["index"]))
        if path is None:
            continue
        primary_only = path.get("unit_cell_only") if unit_cell_only is None else unit_cell_only
        if primary_only and not instance["is_primary_image"]:
            continue
        longest = max(longest, animation_path_length(path, start_override=instance["cart"]))
    return longest


def normalized_animation_duration_seconds(maximum_travel_distance: float, scene_span: float) -> float:
    """Return a scale-independent base duration for the displayed structure."""
    distance = max(float(maximum_travel_distance), 0.0)
    if distance <= 1e-9:
        return STATIONARY_ANIMATION_SECONDS
    span = max(float(scene_span), 1e-9)
    duration = distance / span / NORMALIZED_TRAVEL_SPEED_PER_SECOND
    return float(np.clip(duration, MINIMUM_ANIMATION_SECONDS, MAXIMUM_ANIMATION_SECONDS))
