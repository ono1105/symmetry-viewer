from __future__ import annotations

import numpy as np

from crystal_viewer.viewer.animation_context import (
    animation_paths,
    select_animation_context,
    shared_step_translation,
)
from crystal_viewer.viewer.animation_path import (
    build_operation_path,
    effective_rotation_axis,
    evaluate_path,
    improper_inversion_center,
    improper_reflection_plane,
    normalize,
    preferred_improper_mode,
    render_source_kind,
)


def update_animated_atoms(
    animated_atoms: list[dict],
    paths: dict[int, dict],
    s: float,
    *,
    render_data: dict | None = None,
    boundary_mode: str = "continuous",
    cell_origin_mode: str = "center",
) -> None:
    boundary_context = boundary_wrap_context(render_data, boundary_mode, cell_origin_mode)
    for item in animated_atoms:
        atom = item["atom"]
        path = paths.get(atom["index"])
        display_shift = item["display_shift_cart"]
        center = np.asarray(atom["cart"], dtype=float) + display_shift
        if path is not None and path_applies_to_display_item(path, item):
            center = evaluate_path(path, s, start_override=np.asarray(atom["cart"], dtype=float) + display_shift)
            center = apply_boundary_context(center, boundary_context)
        previous = item.get("current_cart")
        item["position_dirty"] = previous is None or not np.allclose(previous, center, atol=1e-10, rtol=0.0)
        item["current_cart"] = center
        actor = item.get("actor")
        if actor is not None:
            actor.SetPosition(*center)

def path_applies_to_display_item(path: dict, item: dict) -> bool:
    if not path.get("unit_cell_only"):
        return True
    return bool(item.get("is_primary_image", False))


def apply_boundary_mode(
    center_cart: np.ndarray,
    render_data: dict | None,
    boundary_mode: str,
    cell_origin_mode: str,
) -> np.ndarray:
    return apply_boundary_context(center_cart, boundary_wrap_context(render_data, boundary_mode, cell_origin_mode))


def boundary_wrap_context(
    render_data: dict | None,
    boundary_mode: str,
    cell_origin_mode: str,
) -> tuple[np.ndarray, np.ndarray, str] | None:
    if boundary_mode != "wrap" or render_data is None:
        return None
    unit_cell = render_data.get("unit_cell")
    if unit_cell is None:
        return None
    lattice = np.asarray(unit_cell["lattice"], dtype=float)
    try:
        inverse_lattice = np.linalg.inv(lattice)
    except np.linalg.LinAlgError:
        return None
    return lattice, inverse_lattice, cell_origin_mode


def apply_boundary_context(
    center_cart: np.ndarray,
    boundary_context: tuple[np.ndarray, np.ndarray, str] | None,
) -> np.ndarray:
    if boundary_context is None:
        return center_cart
    lattice, inverse_lattice, cell_origin_mode = boundary_context
    frac = np.asarray(center_cart, dtype=float) @ inverse_lattice
    if cell_origin_mode == "corner":
        wrapped = frac - np.floor(frac)
        wrapped = np.where(wrapped >= 1.0 - 1e-9, wrapped - 1.0, wrapped)
    else:
        wrapped = frac - np.round(frac)
        wrapped = np.where(wrapped >= 0.5 - 1e-9, wrapped - 1.0, wrapped)
    return wrapped @ lattice


def operation_speed_multiplier(operation: dict) -> float:
    kind = str(operation.get("kind", ""))
    if kind == "mirror" or kind == "inversion" or "translation" in kind or "glide" in kind:
        return 2.0
    return 1.0


def custom_operation_speed_multiplier(op_type: str) -> float:
    if op_type in {"mirror", "inversion", "translation", "glide"}:
        return 2.0
    return 1.0
