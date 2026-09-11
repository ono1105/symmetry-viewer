from __future__ import annotations

import numpy as np
import pyvista as pv

from crystal_viewer.viewer.atom_style import ATOM_MESH_STYLE, atom_color, display_atom_radius
from crystal_viewer.viewer.atom_instances import element_instance_batches
from crystal_viewer.viewer.display_atoms import display_atom_instances, display_fractional_bounds, scene_span
from crystal_viewer.viewer.glyph_atoms import build_element_glyph_preview
from crystal_viewer.viewer.operation_lookup import selected_mapping


VIEWER_LIGHT_BACKGROUND_COLOR = "#f4f6f3"
VIEWER_DARK_BACKGROUND_COLOR = "#101216"
VIEWER_LIGHT_TEXT_COLOR = "#111827"
VIEWER_DARK_TEXT_COLOR = "#eef2f7"
AXIS_A_COLOR = "#ff0000"
AXIS_B_COLOR = "#00b000"
AXIS_C_COLOR = "#2468ff"


def viewer_background_color(background_mode: str) -> str:
    return VIEWER_DARK_BACKGROUND_COLOR if background_mode == "dark" else VIEWER_LIGHT_BACKGROUND_COLOR


def viewer_text_color(background_mode: str) -> str:
    return VIEWER_DARK_TEXT_COLOR if background_mode == "dark" else VIEWER_LIGHT_TEXT_COLOR


def setup_viewer_lighting(plotter: pv.Plotter, *, background_mode: str = "dark") -> None:
    plotter.set_background(viewer_background_color(background_mode))
    try:
        plotter.remove_all_lights()
    except Exception:
        pass
    try:
        plotter.add_light(
            pv.Light(
                light_type="headlight",
                color="#ffffff",
                intensity=0.55,
            )
        )
        plotter.add_light(
            pv.Light(
                light_type="camera light",
                position=(-3.5, 4.0, 5.5),
                focal_point=(0.0, 0.0, 0.0),
                color="#ffffff",
                intensity=0.72,
                positional=True,
                cone_angle=80.0,
                exponent=1.0,
            )
        )
        plotter.add_light(
            pv.Light(
                light_type="camera light",
                position=(4.0, -3.5, 3.0),
                focal_point=(0.0, 0.0, 0.0),
                color="#ffffff",
                intensity=0.28,
                positional=True,
                cone_angle=100.0,
                exponent=1.0,
            )
        )
    except Exception:
        pass
    try:
        plotter.enable_shadows()
    except Exception:
        pass


def add_orientation_axes(plotter: pv.Plotter, *, unit_cell: "dict | bool | None" = False) -> None:
    if isinstance(unit_cell, dict):
        _add_crystal_orientation_widget(plotter, unit_cell)
    else:
        labels = ("a", "b", "c") if unit_cell else ("X", "Y", "Z")
        plotter.add_axes(
            x_color=AXIS_A_COLOR,
            y_color=AXIS_B_COLOR,
            z_color=AXIS_C_COLOR,
            xlabel=labels[0],
            ylabel=labels[1],
            zlabel=labels[2],
        )


def _add_crystal_orientation_widget(plotter: pv.Plotter, unit_cell: dict) -> None:
    """Add the corner orientation widget with axes aligned to the crystal lattice vectors.

    Uses plotter.add_axes() (vtkOrientationMarkerWidget) and then applies a user
    transform to the vtkAxesActor so that its X/Y/Z arrows point along the actual
    crystal a/b/c directions instead of world X/Y/Z.
    """
    import vtk

    lattice = np.asarray(unit_cell["lattice"], dtype=float)
    norms = np.linalg.norm(lattice, axis=1)
    safe_norms = np.where(norms < 1e-10, 1.0, norms)
    a_n, b_n, c_n = lattice / safe_norms[:, np.newaxis]

    actor = plotter.add_axes(
        x_color=AXIS_A_COLOR,
        y_color=AXIS_B_COLOR,
        z_color=AXIS_C_COLOR,
        xlabel="a",
        ylabel="b",
        zlabel="c",
    )

    # Build a 4×4 transform whose columns are the normalized lattice vectors.
    # vtkAxesActor local X→world a, local Y→world b, local Z→world c.
    vtk_matrix = vtk.vtkMatrix4x4()
    for row in range(3):
        vtk_matrix.SetElement(row, 0, a_n[row])
        vtk_matrix.SetElement(row, 1, b_n[row])
        vtk_matrix.SetElement(row, 2, c_n[row])
        vtk_matrix.SetElement(row, 3, 0.0)
    for col in range(4):
        vtk_matrix.SetElement(3, col, 1.0 if col == 3 else 0.0)

    transform = vtk.vtkTransform()
    transform.SetMatrix(vtk_matrix)
    actor.SetUserTransform(transform)


def atom_legend_entries(render_data: dict, *, element_colors: dict | None = None) -> list[list[str]]:
    by_element = {}
    for atom in render_data.get("atoms", []):
        element = str(atom.get("element", "")).strip()
        if element and element not in by_element:
            by_element[element] = atom
    return [
        [element, atom_color(atom, element_colors=element_colors)]
        for element, atom in sorted(by_element.items())
    ]


def add_atom_legend(
    plotter: pv.Plotter,
    render_data: dict,
    *,
    element_colors: dict | None = None,
    background_mode: str = "dark",
):
    labels = atom_legend_entries(render_data, element_colors=element_colors)
    if not labels:
        return None
    height = min(max(0.08, 0.045 * len(labels) + 0.04), 0.50)
    try:
        actor = plotter.add_legend(
            labels=labels,
            bcolor=viewer_background_color(background_mode),
            border=True,
            size=(0.22, height),
            loc="upper right",
            face="circle",
        )
        text_property = actor.GetEntryTextProperty()
        text_property.SetColor(
            (0.93, 0.95, 0.97) if background_mode == "dark" else (0.07, 0.09, 0.13)
        )
        text_property.SetFontSize(18)
        text_property.BoldOn()
        try:
            text_property.SetFontFamilyToArial()
            text_property.ShadowOff()
        except Exception:
            pass
        return actor
    except Exception:
        return None


def add_atoms(
    plotter: pv.Plotter,
    render_data: dict,
    *,
    display_mode: str = "expanded",
    cell_origin_mode: str = "center",
) -> None:
    for item in display_atom_instances(render_data, display_mode=display_mode, cell_origin_mode=cell_origin_mode):
        atom = item["atom"]
        center = item["cart"]
        radius = display_atom_radius(atom, render_data)
        color = atom_color(atom)
        sphere = pv.Sphere(
            radius=radius,
            center=center,
            theta_resolution=24,
            phi_resolution=16,
        )
        plotter.add_mesh(sphere, color=color, **ATOM_MESH_STYLE)


def add_glyph_atoms(
    plotter: pv.Plotter,
    render_data: dict,
    *,
    display_mode: str = "expanded",
    cell_origin_mode: str = "center",
) -> None:
    for batch in element_instance_batches(render_data, display_mode=display_mode, cell_origin_mode=cell_origin_mode):
        preview = build_element_glyph_preview(batch, render_data)
        color = atom_color({"element": batch.element, "atomic_number": batch.atomic_number})
        plotter.add_mesh(preview.glyph_mesh, color=color, **ATOM_MESH_STYLE)


def add_animated_atoms(
    plotter: pv.Plotter,
    render_data: dict,
    *,
    display_mode: str = "expanded",
    cell_origin_mode: str = "center",
    theta_resolution: int = 16,
    phi_resolution: int = 10,
    smooth_shading: bool = True,
) -> list[dict]:
    animated = []
    for item in display_atom_instances(render_data, display_mode=display_mode, cell_origin_mode=cell_origin_mode):
        atom = item["atom"]
        center = item["cart"]
        radius = display_atom_radius(atom, render_data)
        color = atom_color(atom)
        base = pv.Sphere(
            radius=radius,
            center=(0.0, 0.0, 0.0),
            theta_resolution=theta_resolution,
            phi_resolution=phi_resolution,
        )
        mesh = base.copy()
        mesh_style = dict(ATOM_MESH_STYLE)
        mesh_style["smooth_shading"] = smooth_shading
        actor = plotter.add_mesh(mesh, color=color, **mesh_style)
        actor.SetPosition(*center)
        animated.append(
            {
                "atom": atom,
                "display_shift_frac": item["display_shift_frac"],
                "display_shift_cart": item["display_shift_cart"],
                "is_primary_image": item.get("is_primary_image", True),
                "base_points": base.points.copy(),
                "mesh": mesh,
                "actor": actor,
            }
        )
    return animated


def add_unit_cell(plotter: pv.Plotter, unit_cell: dict, *, display_mode: str = "source", cell_origin_mode: str = "center") -> None:
    vertices = np.asarray(unit_cell["vertices_cart"], dtype=float)
    lattice = np.asarray(unit_cell["lattice"], dtype=float)
    lower, _upper = display_fractional_bounds(display_mode, cell_origin_mode)
    vertices = vertices + (np.asarray([lower, lower, lower], dtype=float) @ lattice)
    edge_colors = {
        (0, 1): AXIS_A_COLOR,
        (2, 4): AXIS_A_COLOR,
        (3, 5): AXIS_A_COLOR,
        (6, 7): AXIS_A_COLOR,
        (0, 2): AXIS_B_COLOR,
        (1, 4): AXIS_B_COLOR,
        (3, 6): AXIS_B_COLOR,
        (5, 7): AXIS_B_COLOR,
        (0, 3): AXIS_C_COLOR,
        (1, 5): AXIS_C_COLOR,
        (2, 6): AXIS_C_COLOR,
        (4, 7): AXIS_C_COLOR,
    }
    for start, end in unit_cell["edges"]:
        edge = (int(start), int(end))
        line = pv.Line(vertices[edge[0]], vertices[edge[1]])
        plotter.add_mesh(line, color=edge_colors.get(edge, "#d6dde6"), line_width=3)


def add_displacements(
    plotter: pv.Plotter,
    render_data: dict,
    atom_mappings: dict | None,
    *,
    operation_index: int | None,
) -> None:
    mapping = selected_mapping(atom_mappings, operation_index)
    if mapping is None:
        print("No atom mapping found. Use --operation with --show-displacements.")
        return

    atoms_by_index = {atom["index"]: atom for atom in render_data["atoms"]}
    span = scene_span(render_data)
    tube_radius = max(span * 0.004, 0.01)
    endpoint_radius = max(span * 0.015, 0.04)

    for entry in mapping["entries"]:
        source_atom = atoms_by_index.get(entry["source_atom"])
        if source_atom is None:
            continue
        start = np.asarray(source_atom["cart"], dtype=float)
        end = np.asarray(entry["transformed_cart"], dtype=float)
        if np.linalg.norm(end - start) < 1e-9:
            continue

        line = pv.Line(start, end)
        tube = line.tube(radius=tube_radius, n_sides=10)
        plotter.add_mesh(tube, color="#f4d03f", opacity=0.8)
        marker = pv.Sphere(radius=endpoint_radius, center=end, theta_resolution=16, phi_resolution=10)
        plotter.add_mesh(marker, color="#f7dc6f", opacity=0.85, smooth_shading=True)
