from __future__ import annotations

import numpy as np

from crystal_viewer.geometry import integer_index_vector, normalize, reflect_point
from crystal_viewer.viewer.animation import (
    improper_inversion_center,
    improper_reflection_plane,
    preferred_improper_mode,
    render_source_kind,
    select_animation_context,
    shared_step_translation,
)
from crystal_viewer.viewer.display_atoms import display_point_cart, display_scene_center, display_scene_span
from crystal_viewer.viewer.glide_geometry import (
    align_fractional_vector_to_reference,
    centered_fractional_vector,
    glide_translation_frac,
)
from crystal_viewer.viewer.operation_lookup import (
    operation_by_index,
    selected_elements,
    selected_mapping,
)


def add_symmetry_elements(
    plotter: pv.Plotter,
    render_data: dict,
    atom_mappings: dict | None,
    *,
    operation_index: int | None,
    element_index: int | None = None,
    display_mode: str = "source",
    cell_origin_mode: str = "center",
    improper_mode: str = "auto",
) -> list:
    operation = operation_by_index(render_data["operations"], operation_index)
    mapping = selected_mapping(atom_mappings, operation_index)
    axes, planes, centers = display_symmetry_elements(
        render_data,
        atom_mappings,
        operation_index,
        element_index,
        improper_mode=improper_mode,
    )
    return add_symmetry_element_actors(
        plotter,
        render_data,
        axes,
        planes,
        centers,
        operation=operation,
        mapping=mapping,
        display_mode=display_mode,
        cell_origin_mode=cell_origin_mode,
    )


def add_symmetry_element_actors(
    plotter: pv.Plotter,
    render_data: dict,
    axes: list[dict],
    planes: list[dict],
    centers: list[dict],
    *,
    operation: dict | None = None,
    mapping: dict | None = None,
    display_mode: str = "source",
    cell_origin_mode: str = "center",
) -> list:
    import pyvista as pv

    actors = []
    span = display_scene_span(render_data, display_mode, cell_origin_mode)
    axis_length = max(span * 0.75, 1.0)
    plane_scale = max(span * 0.45, 0.8)

    for axis in axes:
        point = display_point_cart(render_data, axis["point_cart"], display_mode, cell_origin_mode)
        direction = normalize(np.asarray(axis["direction_cart"], dtype=float))
        line = pv.Line(point - axis_length * direction, point + axis_length * direction)
        actors.append(plotter.add_mesh(line, color="#58d68d", line_width=6))

    for plane in planes:
        point = display_point_cart(render_data, plane["point_cart"], display_mode, cell_origin_mode)
        basis1 = normalize(np.asarray(plane["basis1_cart"], dtype=float)) * plane_scale
        basis2 = normalize(np.asarray(plane["basis2_cart"], dtype=float)) * plane_scale
        points = np.array(
            [
                point - basis1 - basis2,
                point + basis1 - basis2,
                point + basis1 + basis2,
                point - basis1 + basis2,
            ]
        )
        faces = np.array([4, 0, 1, 2, 3])
        mesh = pv.PolyData(points, faces)
        actors.append(
            plotter.add_mesh(
                mesh,
                color="#5dade2",
                opacity=0.35,
                show_edges=True,
                edge_color="#aed6f1",
            )
        )
        if operation is not None and "glide" in str(operation.get("kind", "")):
            actor = add_glide_direction_actor(
                plotter,
                render_data,
                operation,
                mapping,
                plane,
                point,
                axis_length,
            )
            if actor is not None:
                actors.append(actor)

    # Keep inversion centers compact and independent of Cell Range.  Using an
    # opaque octahedron avoids the transparency sorting problems seen on
    # WSL/Mesa while remaining distinguishable from spherical atoms.
    center_span = display_scene_span(render_data, "source", cell_origin_mode)
    center_radius = max(center_span * 0.04, 0.10)
    for center in centers:
        point = display_point_cart(render_data, center["point_cart"], display_mode, cell_origin_mode)
        marker = pv.Octahedron(radius=center_radius, center=point)
        actors.append(
            plotter.add_mesh(
                marker,
                color="#ff5f57",
                opacity=1.0,
                lighting=False,
            )
        )

    if operation is not None and _is_pure_translation(operation):
        actor = add_translation_direction_actor(
            plotter, render_data, operation, axis_length, display_mode, cell_origin_mode
        )
        if actor is not None:
            actors.append(actor)

    return actors


def _is_pure_translation(operation: dict) -> bool:
    kind = str(operation.get("kind", ""))
    return "translation" in kind and "glide" not in kind and "screw" not in kind


def add_translation_direction_actor(
    plotter: pv.Plotter,
    render_data: dict,
    operation: dict,
    line_length: float,
    display_mode: str,
    cell_origin_mode: str,
):
    import pyvista as pv

    unit_cell = render_data.get("unit_cell")
    translation_cart = operation.get("translation_cart")
    if translation_cart is None:
        return None

    translation = np.asarray(translation_cart, dtype=float)
    if unit_cell is not None:
        lattice = np.asarray(unit_cell["lattice"], dtype=float)
        t_frac = centered_fractional_vector(translation @ np.linalg.inv(lattice))
        translation = t_frac @ lattice

    norm = float(np.linalg.norm(translation))
    if norm < 1e-10:
        return None
    direction = translation / norm

    center = display_scene_center(render_data, display_mode, cell_origin_mode)
    line = pv.Line(center - 0.5 * line_length * direction, center + 0.5 * line_length * direction)
    return plotter.add_mesh(line, color="#f7dc6f", line_width=3, opacity=0.55)


def add_glide_direction_actor(
    plotter: pv.Plotter,
    render_data: dict,
    operation: dict,
    mapping: dict | None,
    plane: dict,
    displayed_point: np.ndarray,
    line_length: float,
):
    import pyvista as pv

    glide_cart = glide_translation_cart(render_data, operation, plane, mapping=mapping)
    if glide_cart is None:
        return None
    norm = float(np.linalg.norm(glide_cart))
    if norm < 1e-10:
        return None
    direction = glide_cart / norm
    center = np.asarray(displayed_point, dtype=float)
    line = pv.Line(center - 0.5 * line_length * direction, center + 0.5 * line_length * direction)
    return plotter.add_mesh(line, color="#f7dc6f", line_width=3, opacity=0.55)


def glide_translation_cart(
    render_data: dict,
    operation: dict,
    plane: dict,
    *,
    mapping: dict | None = None,
) -> np.ndarray | None:
    unit_cell = render_data.get("unit_cell")
    if unit_cell is None:
        return None

    lattice = np.asarray(unit_cell["lattice"], dtype=float)
    glide_frac = glide_translation_frac(render_data, operation, plane)
    if glide_frac is None:
        return None
    glide_frac = centered_fractional_vector(glide_frac)
    reference_frac = representative_animation_translation_frac(render_data, operation, plane, mapping)
    if reference_frac is not None:
        glide_frac = align_fractional_vector_to_reference(glide_frac, reference_frac, lattice)
    return glide_frac @ lattice


def representative_animation_translation_frac(
    render_data: dict,
    operation: dict,
    plane: dict,
    mapping: dict | None,
) -> np.ndarray | None:
    if mapping is None:
        return None
    unit_cell = render_data.get("unit_cell")
    if unit_cell is None:
        return None
    atoms_by_index = {atom["index"]: atom for atom in render_data.get("atoms", [])}
    axis, selected_plane, center, reference_entry, shared_shift, shared_angle = select_animation_context(
        render_data,
        operation,
        mapping,
        atoms_by_index,
        element_index=None,
        representative_atom=None,
    )
    if selected_plane is not None:
        if same_periodic_plane(render_data, selected_plane, plane):
            translation = shared_step_translation(
                render_data,
                operation,
                atoms_by_index,
                reference_entry,
                axis,
                selected_plane,
                shared_shift,
                shared_angle,
            )
            if translation is not None:
                return translation @ np.linalg.inv(np.asarray(unit_cell["lattice"], dtype=float))
    reference_entry = next(
        (entry for entry in mapping.get("entries", []) if atoms_by_index.get(entry.get("source_atom")) is not None),
        None,
    )
    if reference_entry is None:
        return None
    atom = atoms_by_index[reference_entry["source_atom"]]
    start = np.asarray(atom["cart"], dtype=float)
    mirrored = reflect_point(
        start,
        np.asarray(plane["point_cart"], dtype=float),
        normalize(np.asarray(plane["normal_cart"], dtype=float)),
    )
    target = np.asarray(reference_entry["transformed_cart"], dtype=float)
    translation = target - mirrored
    return translation @ np.linalg.inv(np.asarray(unit_cell["lattice"], dtype=float))


def same_periodic_plane(render_data: dict, first: dict, second: dict) -> bool:
    first_normal = normalize(np.asarray(first["normal_cart"], dtype=float))
    second_normal = normalize(np.asarray(second["normal_cart"], dtype=float))
    if np.linalg.norm(np.cross(first_normal, second_normal)) >= 1e-6:
        return False

    first_point = np.asarray(first["point_cart"], dtype=float)
    second_point = np.asarray(second["point_cart"], dtype=float)
    direct_distance = abs(float(np.dot(first_point - second_point, second_normal)))
    if direct_distance < 1e-5:
        return True

    unit_cell = render_data.get("unit_cell")
    if unit_cell is None:
        return False
    lattice = np.asarray(unit_cell["lattice"], dtype=float)
    hkl = lattice @ second_normal
    int_hkl = integer_index_vector(hkl)
    if int_hkl is None:
        return False
    delta_frac = (first_point - second_point) @ np.linalg.inv(lattice)
    offset_delta = float(np.dot(int_hkl, delta_frac))
    return abs(offset_delta - round(offset_delta)) < 1e-5


def display_symmetry_elements(
    render_data: dict,
    atom_mappings: dict | None,
    operation_index: int | None,
    element_index: int | None,
    *,
    improper_mode: str = "auto",
) -> tuple[list[dict], list[dict], list[dict]]:
    operation = operation_by_index(render_data["operations"], operation_index)
    if operation_index is not None and element_index is None and atom_mappings is not None:
        mapping = selected_mapping(atom_mappings, operation_index)
        if operation is not None and mapping is not None:
            atoms_by_index = {atom["index"]: atom for atom in render_data["atoms"]}
            axis, plane, center, _, _, _ = select_animation_context(
                render_data,
                operation,
                mapping,
                atoms_by_index,
                element_index=None,
                representative_atom=None,
            )
            return visual_improper_elements(
                render_data,
                operation,
                [axis] if axis is not None else [],
                [plane] if plane is not None else [],
                [center] if center is not None else [],
                improper_mode=improper_mode,
            )

    axes, planes, centers = (
        selected_elements(render_data["axes"], operation_index, element_index),
        selected_elements(render_data["planes"], operation_index, element_index),
        selected_elements(render_data["centers"], operation_index, element_index),
    )
    if operation is None:
        return axes, planes, centers
    return visual_improper_elements(
        render_data,
        operation,
        axes,
        planes,
        centers,
        improper_mode=improper_mode,
    )


def visual_improper_elements(
    render_data: dict,
    operation: dict,
    axes: list[dict],
    planes: list[dict],
    centers: list[dict],
    *,
    improper_mode: str,
) -> tuple[list[dict], list[dict], list[dict]]:
    kind = str(operation.get("kind", ""))
    if "rotoinversion" not in kind and "rotoreflection" not in kind and "improper" not in kind:
        return axes, planes, centers

    mode = preferred_improper_mode(operation, improper_mode, render_source_kind(render_data))
    axis = axes[0] if axes else None
    if mode == "rotoreflection" and axis is not None and not planes:
        plane = improper_reflection_plane(axis)
        if plane is not None:
            planes = [plane]
    elif mode == "rotoinversion" and axis is not None and not centers:
        center = improper_inversion_center(axis)
        if center is not None:
            centers = [center]
    return axes, planes, centers
