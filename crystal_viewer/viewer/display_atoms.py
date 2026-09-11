from __future__ import annotations

from itertools import product

import numpy as np

from crystal_viewer.geometry import periodic_shifts as cached_periodic_shifts


def normalized_cell_origin_mode(cell_origin_mode: str = "center") -> str:
    return "corner" if cell_origin_mode == "corner" else "center"


def display_fractional_bounds(display_mode: str = "source", cell_origin_mode: str = "center") -> tuple[float, float]:
    margin = display_mode_margin(display_mode)
    if normalized_cell_origin_mode(cell_origin_mode) == "corner":
        return -margin, 1.0 + margin
    return -0.5 - margin, 0.5 + margin


def display_atom_instances(
    render_data: dict,
    *,
    display_mode: str = "expanded",
    cell_origin_mode: str = "center",
    include_boundary_images: bool = False,
) -> list[dict]:
    unit_cell = render_data.get("unit_cell")
    if unit_cell is None:
        return [
            {
                "atom": atom,
                "cart": np.asarray(atom["cart"], dtype=float),
                "display_shift_frac": np.zeros(3),
                "display_shift_cart": np.zeros(3),
                "is_primary_image": True,
            }
            for atom in render_data["atoms"]
        ]

    lattice = np.asarray(unit_cell["lattice"], dtype=float)
    instances = []
    for atom in render_data["atoms"]:
        frac = atom.get("frac")
        if frac is None:
            instances.append(
                {
                    "atom": atom,
                    "cart": np.asarray(atom["cart"], dtype=float),
                    "display_shift_frac": np.zeros(3),
                    "display_shift_cart": np.zeros(3),
                    "is_primary_image": True,
                }
            )
            continue

        frac = np.asarray(frac, dtype=float)
        for shift in display_fractional_shifts(
            frac,
            display_mode=display_mode,
            cell_origin_mode=cell_origin_mode,
            include_boundary_images=include_boundary_images,
        ):
            shift_cart = shift @ lattice
            is_primary = is_primary_display_image(frac, shift, cell_origin_mode=cell_origin_mode)
            instances.append(
                {
                    "atom": atom,
                    "cart": np.asarray(atom["cart"], dtype=float) + shift_cart,
                    "display_shift_frac": shift.copy(),
                    "display_shift_cart": shift_cart,
                    "is_primary_image": is_primary,
                }
            )
    return instances


def display_fractional_shifts(
    frac: np.ndarray,
    *,
    display_mode: str = "expanded",
    cell_origin_mode: str = "center",
    include_boundary_images: bool = False,
) -> list[np.ndarray]:
    margin = display_mode_margin(display_mode)
    shifts = []
    lower, upper = display_fractional_bounds(display_mode, cell_origin_mode)
    for shift in periodic_shifts(max(int(np.ceil(margin + 1.0)), 1)):
        image_frac = frac + shift
        within_upper = (
            np.all(image_frac <= upper + 1e-9)
            if include_boundary_images
            else np.all(image_frac < upper - 1e-9)
        )
        if np.all(image_frac >= lower - 1e-9) and within_upper:
            shifts.append(shift)
    return shifts


def is_primary_display_image(frac: np.ndarray, shift: np.ndarray, *, cell_origin_mode: str = "center") -> bool:
    image_frac = np.asarray(frac, dtype=float) + np.asarray(shift, dtype=float)
    lower, upper = display_fractional_bounds("source", cell_origin_mode)
    return bool(np.all(image_frac >= lower - 1e-9) and np.all(image_frac < upper - 1e-9))


def periodic_shifts(limit: int = 1) -> np.ndarray:
    return cached_periodic_shifts(limit)


def display_mode_margin(display_mode: str) -> float:
    if display_mode == "source":
        return 0.0
    if display_mode in ("expanded", "expanded_quarter"):
        return 0.25
    if display_mode == "expanded_half":
        return 0.5
    if display_mode.startswith("expanded_"):
        value = display_mode.removeprefix("expanded_").replace("_", ".")
        try:
            return max(float(value), 0.0)
        except ValueError:
            return 0.25
    return 0.25


def scene_span(render_data: dict) -> float:
    if render_data.get("bounds_min") is None or render_data.get("bounds_max") is None:
        points = np.asarray([atom["cart"] for atom in render_data.get("atoms", [])], dtype=float)
        if not len(points):
            return 1.0
        bounds_min = points.min(axis=0)
        bounds_max = points.max(axis=0)
    else:
        bounds_min = np.asarray(render_data["bounds_min"], dtype=float)
        bounds_max = np.asarray(render_data["bounds_max"], dtype=float)
    span = np.linalg.norm(bounds_max - bounds_min)
    return float(span if span > 1e-9 else 1.0)


def display_scene_span(render_data: dict, display_mode: str = "source", cell_origin_mode: str = "center") -> float:
    unit_cell = render_data.get("unit_cell")
    if unit_cell is None:
        return scene_span(render_data)
    lattice = np.asarray(unit_cell["lattice"], dtype=float)
    lower, upper = display_fractional_bounds(display_mode, cell_origin_mode)
    corners_frac = np.asarray(list(product((lower, upper), repeat=3)), dtype=float)
    corners_cart = corners_frac @ lattice
    span = np.linalg.norm(corners_cart.max(axis=0) - corners_cart.min(axis=0))
    return float(span if span > 1e-9 else scene_span(render_data))


def display_scene_center(render_data: dict, display_mode: str = "source", cell_origin_mode: str = "center") -> np.ndarray:
    unit_cell = render_data.get("unit_cell")
    if unit_cell is None:
        atoms = render_data.get("atoms", [])
        if atoms:
            points = np.asarray([atom["cart"] for atom in atoms], dtype=float)
            return np.mean(points, axis=0)
        return np.zeros(3)
    if normalized_cell_origin_mode(cell_origin_mode) == "corner":
        lattice = np.asarray(unit_cell["lattice"], dtype=float)
        return np.asarray([0.5, 0.5, 0.5], dtype=float) @ lattice
    return np.zeros(3)


def display_point_cart(
    render_data: dict,
    point_cart: list[float] | np.ndarray,
    display_mode: str,
    cell_origin_mode: str = "center",
) -> np.ndarray:
    point = np.asarray(point_cart, dtype=float)
    unit_cell = render_data.get("unit_cell")
    if unit_cell is None:
        return point
    lattice = np.asarray(unit_cell["lattice"], dtype=float)
    frac = point @ np.linalg.inv(lattice)
    if normalized_cell_origin_mode(cell_origin_mode) == "corner":
        wrapped = frac - np.floor(frac)
        wrapped = np.where(wrapped >= 1.0 - 1e-9, wrapped - 1.0, wrapped)
    else:
        wrapped = frac - np.round(frac)
        wrapped = np.where(wrapped >= 0.5 - 1e-9, wrapped - 1.0, wrapped)
    return wrapped @ lattice
