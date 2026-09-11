from __future__ import annotations

import numpy as np

from crystal_viewer.viewer.animation_path import (
    build_operation_path,
    effective_rotation_axis,
    evaluate_path,
    normalize,
    operation_angle_deg,
    operation_rotation_matrix,
    reflect_point,
    rotate_about_axis,
    signed_angle_to_target,
    signed_rotation_angle_from_matrix,
    synchronize_compound_path_phases,
)
from crystal_viewer.viewer.display_atoms import display_point_cart, periodic_shifts
from crystal_viewer.viewer.operation_lookup import filter_by_operation, selected_elements


def animation_paths(
    render_data: dict,
    operation: dict,
    mapping: dict,
    *,
    element_index: int | None = None,
    animation_scope: str = "all",
    representative_atom: int | None = None,
    selected_atoms: tuple[int, ...] = (),
    improper_mode: str = "auto",
    display_mode: str = "source",
    cell_origin_mode: str = "center",
) -> dict[int, dict]:
    atoms_by_index = {atom["index"]: atom for atom in render_data["atoms"]}
    axis, plane, center, reference_entry, shared_shift, shared_angle = select_animation_context(
        render_data,
        operation,
        mapping,
        atoms_by_index,
        element_index=element_index,
        representative_atom=representative_atom,
    )

    if reference_entry is None:
        return {}

    operation, axis, plane, center = display_equivalent_operation_context(
        render_data,
        operation,
        axis,
        plane,
        center,
        display_mode,
        cell_origin_mode,
    )

    translation_override = shared_step_translation(
        render_data,
        operation,
        atoms_by_index,
        reference_entry,
        axis,
        plane,
        shared_shift,
        shared_angle,
    )

    paths = {}
    entries = mapping["entries"]
    if animation_scope == "representative":
        entries = [reference_entry]
    elif animation_scope == "selected":
        selected_set = set(selected_atoms)
        entries = [entry for entry in entries if entry["source_atom"] in selected_set]

    for entry in entries:
        atom = atoms_by_index.get(entry["source_atom"])
        if atom is None:
            continue
        start = np.asarray(atom["cart"], dtype=float)
        target = animation_target(
            render_data,
            start,
            entry,
            operation,
            axis,
            plane,
            center,
            shared_shift=shared_shift,
        )
        paths[entry["source_atom"]] = build_operation_path(
            start,
            target,
            operation,
            axis=axis,
            plane=plane,
            center=center,
            angle_override=shared_angle,
            translation_override=translation_override,
            improper_mode=improper_mode,
            source_kind=str(render_data.get("metadata", {}).get("mode", "")),
        )
    synchronize_compound_path_phases(paths)
    return paths


def display_equivalent_operation_context(
    render_data: dict,
    operation: dict,
    axis: dict | None,
    plane: dict | None,
    center: dict | None,
    display_mode: str,
    cell_origin_mode: str = "center",
) -> tuple[dict, dict | None, dict | None, dict | None]:
    unit_cell = render_data.get("unit_cell")
    if unit_cell is None:
        return operation, axis, plane, center

    anchor = operation_anchor_point(operation, axis, plane, center)
    if anchor is None:
        return operation, axis, plane, center

    anchor = np.asarray(anchor, dtype=float)
    displayed_anchor = display_point_cart(render_data, anchor, display_mode, cell_origin_mode)
    shift = displayed_anchor - anchor
    if np.linalg.norm(shift) < 1e-10:
        return operation, axis, plane, center

    return (
        shifted_operation(operation, shift),
        shifted_point_element(axis, shift),
        shifted_point_element(plane, shift),
        shifted_point_element(center, shift),
    )


def operation_anchor_point(
    operation: dict,
    axis: dict | None,
    plane: dict | None,
    center: dict | None,
) -> np.ndarray | None:
    kind = str(operation.get("kind", ""))
    if (
        kind.startswith(("rotation", "screw"))
        or "rotoinversion" in kind
        or "rotoreflection" in kind
        or "improper" in kind
    ) and axis is not None:
        return np.asarray(axis["point_cart"], dtype=float)
    if ("glide" in kind or kind == "mirror") and plane is not None:
        return np.asarray(plane["point_cart"], dtype=float)
    if (kind == "inversion" or "rotoinversion" in kind) and center is not None:
        return np.asarray(center["point_cart"], dtype=float)
    return None


def shifted_operation(operation: dict, shift: np.ndarray) -> dict:
    matrix = operation.get("matrix_cart")
    translation = operation.get("translation_cart")
    if matrix is None or translation is None:
        return operation
    matrix = np.asarray(matrix, dtype=float)
    shift = np.asarray(shift, dtype=float)
    shifted = dict(operation)
    shifted["translation_cart"] = (
        np.asarray(translation, dtype=float) + shift - matrix @ shift
    ).tolist()
    return shifted


def shifted_point_element(element: dict | None, shift: np.ndarray) -> dict | None:
    if element is None or element.get("point_cart") is None:
        return element
    shifted = dict(element)
    shifted["point_cart"] = (np.asarray(element["point_cart"], dtype=float) + shift).tolist()
    return shifted

def select_animation_context(
    render_data: dict,
    operation: dict,
    mapping: dict,
    atoms_by_index: dict[int, dict],
    *,
    element_index: int | None,
    representative_atom: int | None,
) -> tuple[dict | None, dict | None, dict | None, dict | None, np.ndarray | None, float | None]:
    if element_index is not None:
        return build_animation_context(
            render_data,
            operation,
            mapping,
            atoms_by_index,
            element_index=element_index,
            representative_atom=representative_atom,
        )

    candidates = []
    max_count = max(
        len(filter_by_operation(render_data[key], operation["index"]))
        for key in ("axes", "planes", "centers")
    )
    if max_count == 0:
        candidates.append(
            build_animation_context(
                render_data,
                operation,
                mapping,
                atoms_by_index,
                element_index=None,
                representative_atom=representative_atom,
            )
        )
    else:
        for candidate_index in range(max_count):
            candidates.append(
                build_animation_context(
                    render_data,
                    operation,
                    mapping,
                    atoms_by_index,
                    element_index=candidate_index,
                    representative_atom=representative_atom,
                )
            )
    if len(candidates) == 1:
        return candidates[0]

    best_context = candidates[0]
    best_score = float("inf")
    for context in candidates:
        score = animation_context_score(
            render_data,
            operation,
            mapping,
            atoms_by_index,
            context,
            threshold=best_score,
        )
        if score < best_score:
            best_score = score
            best_context = context
        if best_score <= 1e-10:
            break
    return best_context

def build_animation_context(
    render_data: dict,
    operation: dict,
    mapping: dict,
    atoms_by_index: dict[int, dict],
    *,
    element_index: int | None,
    representative_atom: int | None,
) -> tuple[dict | None, dict | None, dict | None, dict | None, np.ndarray | None, float | None]:
    axes = selected_elements(render_data["axes"], operation["index"], element_index)
    planes = selected_elements(render_data["planes"], operation["index"], element_index)
    centers = selected_elements(render_data["centers"], operation["index"], element_index)
    axis = axes[0] if axes else None
    plane = planes[0] if planes else None
    center = centers[0] if centers else None
    reference_entry = representative_mapping_entry(
        render_data,
        operation,
        mapping,
        atoms_by_index,
        axis,
        plane,
        center,
        representative_atom,
    )
    shared_shift = shared_periodic_shift(render_data, reference_entry, operation, atoms_by_index, axis, plane, center)
    element_shift = symmetry_element_shared_shift(render_data, operation, axis, plane, center)
    if element_shift is not None:
        shared_shift = element_shift
    center = effective_operation_center(render_data, operation, center, shared_shift)
    axis = effective_rotation_axis(operation, axis, center)
    shared_angle = shared_rotation_angle(
        render_data,
        reference_entry,
        operation,
        atoms_by_index,
        axis,
        plane,
        center,
        shared_shift,
    )
    return axis, plane, center, reference_entry, shared_shift, shared_angle

def animation_context_score(
    render_data: dict,
    operation: dict,
    mapping: dict,
    atoms_by_index: dict[int, dict],
    context: tuple[dict | None, dict | None, dict | None, dict | None, np.ndarray | None, float | None],
    *,
    threshold: float = float("inf"),
) -> float:
    axis, plane, center, _, shared_shift, shared_angle = context
    reference_entry = context[3]
    translation_override = shared_step_translation(
        render_data,
        operation,
        atoms_by_index,
        reference_entry,
        axis,
        plane,
        shared_shift,
        shared_angle,
    )
    worst = 0.0
    for entry in mapping["entries"]:
        atom = atoms_by_index.get(entry["source_atom"])
        if atom is None:
            continue
        start = np.asarray(atom["cart"], dtype=float)
        target = animation_target(
            render_data,
            start,
            entry,
            operation,
            axis,
            plane,
            center,
            shared_shift=shared_shift,
        )
        path = build_operation_path(
            start,
            target,
            operation,
            axis=axis,
            plane=plane,
            center=center,
            angle_override=shared_angle,
            translation_override=translation_override,
            improper_mode="auto",
            source_kind=str(render_data.get("metadata", {}).get("mode", "")),
        )
        worst = max(worst, max_path_residual(path))
        if worst >= threshold:
            return worst
    return worst

def max_path_residual(path: dict) -> float:
    residual = 0.0
    stack = [path]
    while stack:
        current = stack.pop()
        if current["type"] == "sequential":
            stack.extend(current["segments"])
            continue
        target = current.get("target")
        if target is not None:
            endpoint = evaluate_path(current, 1.0)
            residual = max(residual, float(np.linalg.norm(np.asarray(target, dtype=float) - endpoint)))
    return residual

def representative_mapping_entry(
    render_data: dict,
    operation: dict,
    mapping: dict,
    atoms_by_index: dict[int, dict],
    axis: dict | None,
    plane: dict | None,
    center: dict | None,
    representative_atom: int | None,
) -> dict | None:
    entries = list(mapping["entries"])
    if representative_atom is not None:
        return next((entry for entry in entries if entry["source_atom"] == representative_atom), None)

    for entry in entries:
        atom = atoms_by_index.get(entry["source_atom"])
        if atom is None:
            continue
        start = np.asarray(atom["cart"], dtype=float)
        target = animation_target(render_data, start, entry, operation, axis, plane, center)
        if np.linalg.norm(target - start) > 1e-8:
            return entry
    return entries[0] if entries else None

def shared_periodic_shift(
    render_data: dict,
    reference_entry: dict | None,
    operation: dict,
    atoms_by_index: dict[int, dict],
    axis: dict | None,
    plane: dict | None,
    center: dict | None,
) -> np.ndarray | None:
    if reference_entry is None or reference_entry.get("transformed_frac") is None:
        return None
    unit_cell = render_data.get("unit_cell")
    if unit_cell is None:
        return None

    reference_atom = atoms_by_index.get(reference_entry["source_atom"])
    if reference_atom is None:
        return None

    start = np.asarray(reference_atom["cart"], dtype=float)
    target = animation_target(render_data, start, reference_entry, operation, axis, plane, center)
    lattice = np.asarray(unit_cell["lattice"], dtype=float)
    target_frac = target @ np.linalg.inv(lattice)
    raw_frac = np.asarray(reference_entry["transformed_frac"], dtype=float)
    return np.rint(target_frac - raw_frac)

def symmetry_element_shared_shift(
    render_data: dict,
    operation: dict,
    axis: dict | None,
    plane: dict | None,
    center: dict | None,
) -> np.ndarray | None:
    unit_cell = render_data.get("unit_cell")
    if unit_cell is None or operation.get("matrix_cart") is None or operation.get("translation_cart") is None:
        return None

    kind = str(operation["kind"])
    point = None
    if kind.startswith(("rotation", "screw")) and axis is not None:
        point = np.asarray(axis["point_cart"], dtype=float)
        shift = axis_preserving_periodic_shift(render_data, operation, axis)
        if shift is not None:
            return shift
    elif (kind == "inversion" or "rotoinversion" in kind) and center is not None:
        point = np.asarray(center["point_cart"], dtype=float)
    elif kind == "mirror" and plane is not None:
        point = np.asarray(plane["point_cart"], dtype=float)

    if point is None:
        return None

    matrix = np.asarray(operation["matrix_cart"], dtype=float)
    translation = np.asarray(operation["translation_cart"], dtype=float)
    lattice = np.asarray(unit_cell["lattice"], dtype=float)
    required = (point - (matrix @ point + translation)) @ np.linalg.inv(lattice)
    rounded = np.rint(required)
    if np.linalg.norm(required - rounded) > 1e-5:
        return None
    return rounded

def axis_preserving_periodic_shift(
    render_data: dict,
    operation: dict,
    axis: dict,
) -> np.ndarray | None:
    unit_cell = render_data.get("unit_cell")
    if unit_cell is None or operation.get("matrix_cart") is None or operation.get("translation_cart") is None:
        return None

    matrix = np.asarray(operation["matrix_cart"], dtype=float)
    translation = np.asarray(operation["translation_cart"], dtype=float)
    lattice = np.asarray(unit_cell["lattice"], dtype=float)
    point = np.asarray(axis["point_cart"], dtype=float)
    direction = normalize(np.asarray(axis["direction_cart"], dtype=float))
    kind = str(operation.get("kind", ""))

    best: tuple[float, float, np.ndarray] | None = None
    for shift in periodic_shifts(1):
        displacement = matrix @ point + translation + shift @ lattice - point
        parallel = float(np.dot(displacement, direction))
        perpendicular = displacement - parallel * direction
        perpendicular_norm = float(np.linalg.norm(perpendicular))
        parallel_penalty = 0.0 if kind.startswith("screw") else abs(parallel)
        total_norm = float(np.linalg.norm(displacement))
        score = (perpendicular_norm + parallel_penalty, total_norm, shift)
        if best is None or score[:2] < best[:2]:
            best = score

    if best is None or best[0] > 1e-5:
        return None
    return np.asarray(best[2], dtype=float)

def shared_rotation_angle(
    render_data: dict,
    reference_entry: dict | None,
    operation: dict,
    atoms_by_index: dict[int, dict],
    axis: dict | None,
    plane: dict | None,
    center: dict | None,
    shared_shift: np.ndarray | None,
) -> float | None:
    kind = str(operation["kind"])
    if not (
        kind.startswith(("rotation", "screw"))
        or "rotoinversion" in kind
        or "rotoreflection" in kind
        or "improper" in kind
    ):
        return None
    angle_deg = operation_angle_deg(operation)
    if reference_entry is None or axis is None or angle_deg is None:
        return None

    reference_atom = atoms_by_index.get(reference_entry["source_atom"])
    if reference_atom is None:
        return None

    start = np.asarray(reference_atom["cart"], dtype=float)
    axis_point = np.asarray(axis["point_cart"], dtype=float)
    axis_direction = normalize(np.asarray(axis["direction_cart"], dtype=float))
    rotation = operation_rotation_matrix(operation)
    if rotation is not None:
        return signed_rotation_angle_from_matrix(rotation, axis_direction)
    target = animation_target(
        render_data,
        start,
        reference_entry,
        operation,
        axis,
        plane,
        center,
        shared_shift=shared_shift,
    )
    return signed_angle_to_target(start, target, axis_point, axis_direction, angle_deg)

def shared_step_translation(
    render_data: dict,
    operation: dict,
    atoms_by_index: dict[int, dict],
    reference_entry: dict | None,
    axis: dict | None,
    plane: dict | None,
    shared_shift: np.ndarray | None,
    shared_angle: float | None,
) -> np.ndarray | None:
    if reference_entry is None:
        return None
    reference_atom = atoms_by_index.get(reference_entry["source_atom"])
    if reference_atom is None:
        return None

    kind = str(operation["kind"])
    start = np.asarray(reference_atom["cart"], dtype=float)
    target = animation_target(
        render_data,
        start,
        reference_entry,
        operation,
        axis,
        plane,
        None,
        shared_shift=shared_shift,
    )

    if kind.startswith("screw"):
        if axis is None or shared_angle is None:
            return None
        rotated_end = rotate_about_axis(
            start,
            np.asarray(axis["point_cart"], dtype=float),
            normalize(np.asarray(axis["direction_cart"], dtype=float)),
            shared_angle,
        )
        return target - rotated_end

    if "glide" in kind:
        if plane is None:
            return None
        mirrored_end = reflect_point(
            start,
            np.asarray(plane["point_cart"], dtype=float),
            normalize(np.asarray(plane["normal_cart"], dtype=float)),
        )
        return target - mirrored_end

    return None

def animation_target(
    render_data: dict,
    start: np.ndarray,
    entry: dict,
    operation: dict,
    axis: dict | None,
    plane: dict | None,
    center: dict | None,
    *,
    shared_shift: np.ndarray | None = None,
) -> np.ndarray:
    default_target = np.asarray(entry["transformed_cart"], dtype=float)
    if shared_shift is not None and entry.get("transformed_frac") is not None:
        affine_target = operation_affine_target(render_data, start, operation, shared_shift)
        if affine_target is not None:
            return affine_target
        unit_cell = render_data.get("unit_cell")
        if unit_cell is not None:
            lattice = np.asarray(unit_cell["lattice"], dtype=float)
            return (np.asarray(entry["transformed_frac"], dtype=float) + shared_shift) @ lattice

    candidates = periodic_target_candidates(render_data, entry)
    if len(candidates) == 1:
        return candidates[0]

    kind = str(operation["kind"])
    angle_deg = operation_angle_deg(operation)
    if kind.startswith(("rotation", "screw")) and axis is not None and angle_deg is not None:
        axis_point = np.asarray(axis["point_cart"], dtype=float)
        axis_direction = normalize(np.asarray(axis["direction_cart"], dtype=float))
        angle = signed_angle_to_target(start, default_target, axis_point, axis_direction, angle_deg)
        rotated = rotate_about_axis(start, axis_point, axis_direction, angle)
        return best_axis_consistent_target(candidates, rotated, axis_direction)
    if "glide" in kind or kind == "mirror":
        if plane is None:
            return default_target
        plane_point = np.asarray(plane["point_cart"], dtype=float)
        plane_normal = normalize(np.asarray(plane["normal_cart"], dtype=float))
        mirrored = reflect_point(start, plane_point, plane_normal)
        return best_plane_consistent_target(candidates, mirrored, plane_normal)
    if kind == "inversion" and center is not None:
        center_point = np.asarray(center["point_cart"], dtype=float)
        inverted = 2.0 * center_point - start
        return candidates[np.argmin(np.linalg.norm(candidates - inverted, axis=1))]
    return default_target

def operation_affine_target(
    render_data: dict,
    start: np.ndarray,
    operation: dict,
    shared_shift: np.ndarray | None,
) -> np.ndarray | None:
    affine = operation_affine_matrix_translation(render_data, operation, shared_shift)
    if affine is None:
        return None
    matrix, translation = affine
    return matrix @ start + translation

def operation_affine_matrix_translation(
    render_data: dict,
    operation: dict,
    shared_shift: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray] | None:
    matrix = operation.get("matrix_cart")
    translation = operation.get("translation_cart")
    if matrix is None or translation is None:
        return None
    affine_matrix = np.asarray(matrix, dtype=float)
    affine_translation = np.asarray(translation, dtype=float)
    if shared_shift is not None and operation.get("matrix_frac") is not None:
        unit_cell = render_data.get("unit_cell")
        if unit_cell is not None:
            lattice = np.asarray(unit_cell["lattice"], dtype=float)
            affine_translation = affine_translation + np.asarray(shared_shift, dtype=float) @ lattice
    return affine_matrix, affine_translation

def periodic_target_candidates(render_data: dict, entry: dict) -> np.ndarray:
    frac = entry.get("transformed_frac")
    unit_cell = render_data.get("unit_cell")
    if frac is None or unit_cell is None:
        return np.asarray([entry["transformed_cart"]], dtype=float)
    lattice = np.asarray(unit_cell["lattice"], dtype=float)
    frac = np.asarray(frac, dtype=float)
    return (frac + periodic_shifts(1)) @ lattice

def best_axis_consistent_target(candidates: np.ndarray, rotated: np.ndarray, axis_direction: np.ndarray) -> np.ndarray:
    residuals = candidates - rotated
    parallel = np.outer(residuals @ axis_direction, axis_direction)
    perpendicular = residuals - parallel
    scores = np.linalg.norm(perpendicular, axis=1) + 1e-6 * np.linalg.norm(residuals, axis=1)
    return candidates[np.argmin(scores)]

def best_plane_consistent_target(candidates: np.ndarray, mirrored: np.ndarray, plane_normal: np.ndarray) -> np.ndarray:
    residuals = candidates - mirrored
    normal_distance = np.abs(residuals @ plane_normal)
    scores = normal_distance + 1e-6 * np.linalg.norm(residuals, axis=1)
    return candidates[np.argmin(scores)]

def effective_operation_center(
    render_data: dict,
    operation: dict,
    center: dict | None,
    shared_shift: np.ndarray | None,
) -> dict | None:
    if render_data.get("unit_cell") is None:
        return center
    kind = str(operation["kind"])
    affine = operation_affine_matrix_translation(render_data, operation, shared_shift)
    if affine is None:
        return center
    matrix, translation = affine

    point = None
    if kind == "inversion":
        point = 0.5 * translation
    elif "rotoinversion" in kind:
        rotation = -matrix
        coeff = np.eye(3) + rotation
        point = np.linalg.pinv(coeff) @ translation

    if point is None:
        return center

    # Snap to the nearest periodic image of the visual center marker (Q4 fix).
    # The affine-computed fixed point can land on a different lattice image than
    # the visual element.  Only snap when shared_shift is absent or zero,
    # because the render_data visual center corresponds to the unshifted operation.
    # When shared_shift is nonzero, the correct pivot has already moved and
    # snapping to the original visual center would introduce a new error.
    shift_is_zero = shared_shift is None or not np.any(np.asarray(shared_shift, dtype=float))
    if center is not None and shift_is_zero:
        unit_cell = render_data.get("unit_cell")
        if unit_cell is not None:
            lattice = np.asarray(unit_cell["lattice"], dtype=float)
            visual_pt = np.asarray(center["point_cart"], dtype=float)
            diff_frac = (point - visual_pt) @ np.linalg.inv(lattice)
            snapped = point - np.round(diff_frac) @ lattice
            if np.linalg.norm(matrix @ snapped + translation - snapped) < 1e-8:
                point = snapped

    return {"point_cart": point}
