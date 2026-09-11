from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyvista as pv

from crystal_viewer.viewer.atom_instances import ElementInstanceBatch
from crystal_viewer.viewer.atom_style import display_atom_radius


@dataclass(frozen=True)
class ElementGlyphPreview:
    element: str
    atomic_number: int
    point_count: int
    glyph_point_count: int
    glyph_cell_count: int
    point_cloud: pv.PolyData
    glyph_mesh: pv.PolyData


@dataclass
class ElementGlyphMesh:
    element: str
    atomic_number: int
    template_points: np.ndarray
    mesh: pv.PolyData


def build_element_glyph_preview(
    batch: ElementInstanceBatch,
    render_data: dict,
    *,
    theta_resolution: int = 16,
    phi_resolution: int = 10,
) -> ElementGlyphPreview:
    atom = {
        "atomic_number": batch.atomic_number,
        "element": batch.element,
    }
    radius = display_atom_radius(atom, render_data)
    sphere = pv.Sphere(
        radius=radius,
        center=(0.0, 0.0, 0.0),
        theta_resolution=theta_resolution,
        phi_resolution=phi_resolution,
    )
    point_cloud = pv.PolyData(batch.positions.copy())
    glyph_mesh = point_cloud.glyph(geom=sphere, scale=False, orient=False)
    return ElementGlyphPreview(
        element=batch.element,
        atomic_number=batch.atomic_number,
        point_count=point_cloud.n_points,
        glyph_point_count=glyph_mesh.n_points,
        glyph_cell_count=glyph_mesh.n_cells,
        point_cloud=point_cloud,
        glyph_mesh=glyph_mesh,
    )


def build_element_glyph_mesh(
    batch: ElementInstanceBatch,
    render_data: dict,
    *,
    theta_resolution: int = 16,
    phi_resolution: int = 10,
) -> ElementGlyphMesh:
    atom = {
        "atomic_number": batch.atomic_number,
        "element": batch.element,
    }
    radius = display_atom_radius(atom, render_data)
    sphere = pv.Sphere(
        radius=radius,
        center=(0.0, 0.0, 0.0),
        theta_resolution=theta_resolution,
        phi_resolution=phi_resolution,
    )
    template_points = np.asarray(sphere.points, dtype=float).copy()
    points = batched_points(template_points, batch.positions)
    faces = offset_faces(np.asarray(sphere.faces, dtype=np.int64), len(template_points), len(batch.positions))
    return ElementGlyphMesh(
        element=batch.element,
        atomic_number=batch.atomic_number,
        template_points=template_points,
        mesh=pv.PolyData(points, faces),
    )


def update_element_glyph_mesh(mesh: ElementGlyphMesh, positions: np.ndarray) -> None:
    mesh.mesh.points = batched_points(mesh.template_points, np.asarray(positions, dtype=float))
    mesh.mesh.Modified()


def update_element_glyph_instance(mesh: ElementGlyphMesh, instance_index: int, position: np.ndarray) -> None:
    point_count = len(mesh.template_points)
    start = int(instance_index) * point_count
    stop = start + point_count
    mesh.mesh.points[start:stop] = mesh.template_points + np.asarray(position, dtype=float)


def batched_points(template_points: np.ndarray, positions: np.ndarray) -> np.ndarray:
    return (np.asarray(positions, dtype=float)[:, None, :] + template_points[None, :, :]).reshape(-1, 3)


def offset_faces(faces: np.ndarray, point_count: int, instance_count: int) -> np.ndarray:
    if instance_count <= 0:
        return np.asarray([], dtype=np.int64)
    faces = np.asarray(faces, dtype=np.int64)
    is_vertex_index = np.ones(len(faces), dtype=np.int64)
    cursor = 0
    while cursor < len(faces):
        is_vertex_index[cursor] = 0
        size = int(faces[cursor])
        cursor += size + 1
    tiled = np.tile(faces, instance_count)
    offsets = np.repeat(np.arange(instance_count, dtype=np.int64) * point_count, len(faces))
    masks = np.tile(is_vertex_index, instance_count)
    return tiled + offsets * masks
