from __future__ import annotations

import numpy as np

from crystal_viewer.geometry import normalize, signed_rotation_angle_from_matrix
from crystal_viewer.viewer.animation_context import display_equivalent_operation_context
from crystal_viewer.viewer.animation_path import build_operation_path, synchronize_compound_path_phases
from crystal_viewer.viewer.symmetry_elements import display_symmetry_elements


def build_custom_animation_paths(
    render_data: dict,
    atom_mappings: dict | None,
    request: dict,
    *,
    display_mode: str = "source",
    cell_origin_mode: str = "center",
    improper_mode: str = "auto",
) -> dict[int, dict]:
    unit_cell = render_data.get("unit_cell")
    if unit_cell is None:
        return {}
    lattice = np.asarray(unit_cell["lattice"], dtype=float)
    atom_indices = {int(index) for index in request.get("atom_indices", [])}
    unit_cell_only = bool(request.get("unit_cell_only", False))
    sequence_items = request.get("sequence_items") or []
    if sequence_items:
        contexts = custom_sequence_step_contexts(
            render_data, atom_mappings, sequence_items, lattice,
            display_mode=display_mode, cell_origin_mode=cell_origin_mode, improper_mode=improper_mode,
        )
        return build_sequence_paths(render_data, atom_indices, contexts, unit_cell_only, improper_mode)

    W_frac = np.asarray(request.get("W_frac"), dtype=float)
    t_frac = np.asarray(request.get("t_frac"), dtype=float)
    if W_frac.shape != (3, 3) or t_frac.shape != (3,):
        return {}
    operation, axis, plane, center = custom_operation_context(
        request.get("op_type", "matrix"), request.get("op_params") or {}, W_frac, t_frac, lattice,
    )
    operation, axis, plane, center = display_equivalent_operation_context(
        render_data, operation, axis, plane, center, display_mode, cell_origin_mode,
    )
    matrix = np.asarray(operation["matrix_cart"], dtype=float)
    translation = np.asarray(operation["translation_cart"], dtype=float)
    angle_override = operation_angle_override(operation, axis)
    paths = {}
    for atom in render_data.get("atoms", []):
        if int(atom["index"]) not in atom_indices:
            continue
        start = np.asarray(atom["cart"], dtype=float)
        target = matrix @ start + translation
        if operation["kind"] == "matrix":
            path = {"type": "affine_linear", "start": start, "target": target,
                    "matrix_cart": matrix, "translation_cart": translation}
        else:
            path = build_operation_path(
                start, target, operation, axis=axis, plane=plane, center=center,
                angle_override=angle_override, improper_mode=improper_mode,
                source_kind=str(render_data.get("metadata", {}).get("mode", "")),
            )
        if unit_cell_only:
            path["unit_cell_only"] = True
        paths[int(atom["index"])] = path
    synchronize_compound_path_phases(paths)
    return paths


def custom_operation_context(op_type, op_params, W_frac, t_frac, lattice):
    inverse_lattice = np.linalg.inv(lattice)
    W_cart = lattice.T @ W_frac @ np.linalg.inv(lattice.T)
    t_cart = t_frac @ lattice
    axis = plane = center = None
    if op_type in ("rotation", "screw"):
        axis = {"point_cart": (np.asarray(op_params.get("point", [0, 0, 0])) @ lattice).tolist(),
                "direction_cart": normalize(np.asarray(op_params.get("axis", [0, 0, 1])) @ lattice).tolist()}
    elif op_type in ("mirror", "glide"):
        plane = {"point_cart": (np.asarray(op_params.get("point", [0, 0, 0])) @ lattice).tolist(),
                 "normal_cart": normalize(inverse_lattice @ np.asarray(op_params.get("normal", [0, 0, 1]))).tolist()}
    elif op_type == "inversion":
        center = {"point_cart": (np.asarray(op_params.get("center", [0, 0, 0])) @ lattice).tolist()}
    elif op_type == "rotoinversion":
        point = (np.asarray(op_params.get("center", [0, 0, 0])) @ lattice).tolist()
        axis = {"point_cart": point, "direction_cart": normalize(np.asarray(op_params.get("axis", [0, 0, 1])) @ lattice).tolist()}
        center = {"point_cart": point}
    kind = {"rotation": "rotation_n", "screw": "screw_n", "mirror": "mirror", "glide": "glide",
            "inversion": "inversion", "rotoinversion": "rotoinversion_or_improper_n"}.get(str(op_type), "matrix")
    return ({"kind": kind, "angle_deg": float(op_params.get("angle", 90)), "order": None,
             "matrix_cart": W_cart.tolist(), "translation_cart": t_cart.tolist()}, axis, plane, center)


def operation_angle_override(operation, axis):
    if not str(operation.get("kind", "")).startswith(("rotation", "screw")) or axis is None:
        return None
    return signed_rotation_angle_from_matrix(
        np.asarray(operation["matrix_cart"], dtype=float), np.asarray(axis["direction_cart"], dtype=float)
    )


def custom_sequence_step_contexts(render_data, atom_mappings, items, lattice, *, display_mode, cell_origin_mode, improper_mode):
    operations = {int(operation["index"]): operation for operation in render_data.get("operations", [])}
    contexts = []
    for item in items:
        if str(item.get("type") or "operation") == "operation":
            operation = operations.get(int(item.get("index")))
            if operation is None:
                continue
            axes, planes, centers = display_symmetry_elements(
                render_data, atom_mappings, operation["index"], None, improper_mode=improper_mode
            )
            axis, plane, center = (axes[0] if axes else None, planes[0] if planes else None, centers[0] if centers else None)
            label = f"op {operation['index']}"
        else:
            W_frac, t_frac = np.asarray(item.get("W_frac")), np.asarray(item.get("t_frac"))
            if W_frac.shape != (3, 3) or t_frac.shape != (3,):
                continue
            operation, axis, plane, center = custom_operation_context(
                item.get("op_type", "matrix"), item.get("op_params") or {}, W_frac, t_frac, lattice
            )
            label = str(item.get("label") or "custom")
        operation, axis, plane, center = display_equivalent_operation_context(
            render_data, operation, axis, plane, center, display_mode, cell_origin_mode
        )
        contexts.append({"operation": operation, "axis": axis, "plane": plane, "center": center,
                         "matrix": np.asarray(operation["matrix_cart"]),
                         "translation": np.asarray(operation["translation_cart"]),
                         "angle_override": operation_angle_override(operation, axis), "label": label})
    return contexts


def build_sequence_paths(render_data, atom_indices, contexts, unit_cell_only, improper_mode):
    paths = {}
    for atom in render_data.get("atoms", []):
        if int(atom["index"]) not in atom_indices:
            continue
        current = np.asarray(atom["cart"], dtype=float)
        segments = []
        for context in contexts:
            target = context["matrix"] @ current + context["translation"]
            operation = context["operation"]
            if operation["kind"] == "matrix":
                segment = {"type": "affine_linear", "start": current, "target": target,
                           "matrix_cart": context["matrix"], "translation_cart": context["translation"]}
            else:
                segment = build_operation_path(
                    current, target, operation, axis=context["axis"], plane=context["plane"], center=context["center"],
                    angle_override=context["angle_override"], improper_mode=improper_mode,
                    source_kind=str(render_data.get("metadata", {}).get("mode", "")),
                )
            segments.append(segment)
            current = target
        path = {"type": "sequential", "segments": segments,
                "segment_elements": [{"axes": [c["axis"]] if c["axis"] else [],
                                      "planes": [c["plane"]] if c["plane"] else [],
                                      "centers": [c["center"]] if c["center"] else [],
                                      "glide_translation_cart": (
                                          segment.get("translation") if segment.get("type") == "glide" else None
                                      ),
                                      "label": c["label"]}
                                     for c, segment in zip(contexts, segments)]}
        if unit_cell_only:
            path["unit_cell_only"] = True
        paths[int(atom["index"])] = path
    synchronize_compound_path_phases(paths)
    return paths
