from __future__ import annotations

import argparse
import base64
import binascii
import io
import json
import os
import subprocess
import sys
import threading
import tempfile
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

try:
    import _bootstrap  # noqa: F401
except ModuleNotFoundError:
    from tools import _bootstrap  # noqa: F401

import logging
import numpy as np
from PIL import Image

from crystal_viewer.export_pipeline import (
    DEFAULT_JSON_EXPORT_DIR,
    default_json_output_path,
    export_analysis_to_json,
    slug_from_path,
)
from crystal_viewer.json_export import EXPORT_SCHEMA_VERSION, to_jsonable
from crystal_viewer.source_kinds import (
    SOURCE_KIND_CRYSTAL,
    SOURCE_KIND_EMPTY,
    SOURCE_KIND_MOLECULE,
    normalize_source_kind,
)
from crystal_viewer.symmetry_operations import (
    compose_operation_sequence,
    find_matching_operation_index,
    find_operation_sequence_bfs,
)
from crystal_viewer.viewer.atom_style import atom_color, display_atom_radius
from crystal_viewer.viewer.animation_context import animation_paths
from crystal_viewer.viewer.animation_path import evaluate_path, phase_fraction, two_phase_geometry
from crystal_viewer.viewer.display_atoms import display_atom_instances, display_fractional_bounds, display_scene_center
from crystal_viewer.viewer.animation_api import (
    animation_path_response,
    custom_animation_path_response,
    symmetry_elements_response,
)
from crystal_viewer.game.catalog import PUZZLE_COUNT_KEYS
from crystal_viewer.game.axis_orders import ROTATION_ORDER_OPTIONS
from crystal_viewer.game.axis_orders import axis_questions as axis_order_questions
from crystal_viewer.game.axis_orders import check_answer as check_axis_order_answer
from crystal_viewer.game.operation_identify import check_answer as check_operation_answer
from crystal_viewer.game.operation_identify import public_questions as public_operation_questions
from crystal_viewer.game.composition import check_answer as check_composition_answer
from crystal_viewer.game.composition import public_questions as public_composition_questions
from crystal_viewer.game.atom_mapping import check_answer as check_mapping_answer
from crystal_viewer.game.atom_mapping import public_questions as public_mapping_questions
from crystal_viewer.game.point_group import check_answer as check_point_group_answer
from crystal_viewer.game.point_group import public_questions as public_point_group_questions
from crystal_viewer.viewer.custom_operation import (
    build_custom_operation_frac,
    check_custom_operation,
    custom_operation_visuals,
)
from crystal_viewer.web.browser_ui import HTML
from crystal_viewer.viewer.operation_labels import atom_frac_label
from crystal_viewer.viewer.operation_lookup import operation_by_index, selected_mapping
from crystal_viewer.viewer.render_state import (
    apply_render_state_update,
    initial_render_state,
    preserved_render_state,
)
from crystal_viewer.viewer.session import ViewerSession

logging.getLogger("pyvista").setLevel(logging.ERROR)
logging.getLogger("vtkmodules").setLevel(logging.ERROR)


# True inside the packaged build (packaging/symmetry_view.spec). There every
# file sits under sys._MEIPASS rather than next to this script, and the program
# can be started from any working directory.
FROZEN = bool(getattr(sys, "frozen", False))
PROJECT_ROOT = Path(getattr(sys, "_MEIPASS", "")) if FROZEN else Path(__file__).resolve().parents[1]
UPLOAD_CIF_DIR = Path(tempfile.gettempdir()) / "symmetry_view_cif_uploads"
UPLOAD_MOLECULE_DIR = Path(tempfile.gettempdir()) / "symmetry_view_molecule_uploads"
# Bundled examples are exported here; anchored so the cwd does not matter.
EXAMPLE_JSON_DIR = PROJECT_ROOT / DEFAULT_JSON_EXPORT_DIR
# The packaged build keeps browser imports next to the uploads instead of
# writing into its own folder, so deleting that folder removes everything.
DEFAULT_BROWSER_IMPORT_JSON_DIR = (
    Path(tempfile.gettempdir()) / "symmetry_view_imported_json"
    if FROZEN
    else DEFAULT_JSON_EXPORT_DIR / "imported"
)
# The packaged executable cannot run tools/export_*.py with a Python it does not
# have, so it re-runs itself as `symmetry_view --worker <script> ...` instead.
WORKER_FLAG = "--worker"
DEFAULT_ANALYSIS_TIMEOUT_SEC = 120.0
MAX_REQUEST_BODY_BYTES = 128 * 1024 * 1024
CUSTOM_OPERATION_TOLERANCE_MIN = 0.001
CUSTOM_OPERATION_TOLERANCE_MAX = 2.0
EXAMPLE_DIRS = {
    SOURCE_KIND_CRYSTAL: PROJECT_ROOT / "examples/cif",
    SOURCE_KIND_MOLECULE: PROJECT_ROOT / "examples/molecules",
}
EXAMPLE_SUFFIXES = {
    SOURCE_KIND_CRYSTAL: ".cif",
    SOURCE_KIND_MOLECULE: ".xyz",
}
EXAMPLE_CATALOG_PATH = PROJECT_ROOT / "examples/example_catalog.json"
# Opened when the server is started without a file. Each must be an export of a
# bundled example; f2_pd and jacobsite used to sit here long after the examples
# they came from were removed.
DEFAULT_STARTER_PATHS = (
    Path("exports/json/halite.json"),
    Path("exports/json/benzene.json"),
    Path("exports/json/water.json"),
)
EMPTY_VIEWER_JSON_PATH = Path(tempfile.gettempdir()) / "symmetry_view_empty.json"
DEBUG_IMPORT_TIMING = os.environ.get("CRYSTAL_VIEWER_DEBUG_IMPORT") == "1"
_EXAMPLE_CATALOG_CACHE: dict[str, list[dict]] | None = None


def custom_operation_tolerance(value: object) -> tuple[float | None, str | None]:
    """Parse the browser's Custom Operation tolerance without accepting NaN/Inf."""
    try:
        tolerance = float(value)
    except (TypeError, ValueError) as exc:
        return None, f"Tolerance must be a number: {exc}"
    if not np.isfinite(tolerance):
        return None, "Tolerance must be a finite number"
    if not CUSTOM_OPERATION_TOLERANCE_MIN <= tolerance <= CUSTOM_OPERATION_TOLERANCE_MAX:
        return None, (
            "Tolerance must be between "
            f"{CUSTOM_OPERATION_TOLERANCE_MIN:g} and {CUSTOM_OPERATION_TOLERANCE_MAX:g} Å"
        )
    return tolerance, None


def custom_operation_matrix_error(W_frac: object, t_frac: object) -> str | None:
    """Validate externally supplied affine-operation data before it is composed."""
    try:
        W = np.asarray(W_frac, dtype=float)
        t = np.asarray(t_frac, dtype=float)
    except (TypeError, ValueError) as exc:
        return f"Operation matrix parameters must be numeric: {exc}"
    if W.shape != (3, 3) or not np.all(np.isfinite(W)):
        return "Operation matrix W must be a finite 3x3 matrix"
    if t.shape != (3,) or not np.all(np.isfinite(t)):
        return "Operation translation t must be a finite vector of length 3"
    return None


def debug_import_timing(label: str, start: float) -> None:
    if DEBUG_IMPORT_TIMING:
        print(f"[import] {label} {time.monotonic() - start:.3f}s", flush=True)


def replace_shared_state_for_load(shared_state: dict, next_state: dict, request_id: int) -> None:
    next_state["load_request_id"] = request_id
    shared_state.clear()
    shared_state.update(next_state)


def empty_viewer_payload() -> dict:
    return {
        "schema_version": EXPORT_SCHEMA_VERSION,
        "source_kind": SOURCE_KIND_EMPTY,
        "structure_loaded": False,
        "render_data": {
            "metadata": {
                "mode": SOURCE_KIND_EMPTY,
                "source_file": "",
                "formula": "No structure",
                "symmetry_label": "",
                "operation_count": 1,
            },
            "atoms": [],
            "asymmetric_atoms": [],
            "operations": [
                {
                    "index": 0,
                    "label": "0: load a structure",
                    "kind": "identity",
                    "order": 1,
                    "angle_deg": 0.0,
                    "symbol": "load",
                    "matrix_frac": None,
                    "translation_frac": None,
                    "matrix_cart": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    "translation_cart": [0.0, 0.0, 0.0],
                }
            ],
            "axes": [],
            "planes": [],
            "centers": [],
            "unit_cell": None,
            "bounds_min": [0.0, 0.0, 0.0],
            "bounds_max": [1.0, 1.0, 1.0],
        },
        "atom_mappings": {
            "mode": SOURCE_KIND_EMPTY,
            "complete": True,
            "incomplete_operation_indices": [],
            "mappings": [
                {
                    "mode": SOURCE_KIND_EMPTY,
                    "operation_index": 0,
                    "operation_kind": "identity",
                    "complete": True,
                    "atom_to_atom": [],
                    "max_distance": 0.0,
                    "unmatched_atoms": [],
                    "entries": [],
                }
            ],
        },
    }


def write_empty_viewer_json() -> Path:
    EMPTY_VIEWER_JSON_PATH.write_text(json.dumps(empty_viewer_payload(), indent=2), encoding="utf-8")
    return EMPTY_VIEWER_JSON_PATH


def atom_api_items(atoms: list[dict]) -> list[dict]:
    return [
        {
            "index": atom["index"],
            "element": atom["element"],
            "frac": atom.get("frac"),
            "frac_label": atom_frac_label(atom),
            "cart": atom.get("cart"),
            "asymmetric_index": atom.get("asymmetric_index"),
            "default_color": atom_color(atom),
        }
        for atom in atoms
    ]


def atom_render_style_items(
    render_data: dict,
    *,
    element_colors: dict | None = None,
    atom_colors: dict | None = None,
) -> list[dict]:
    return [
        {
            "index": int(atom["index"]),
            "color": atom_color(atom, element_colors=element_colors, atom_colors=atom_colors),
            "radius": display_atom_radius(atom, render_data),
        }
        for atom in render_data.get("atoms", [])
    ]


def display_atom_api_items(
    render_data: dict,
    *,
    display_mode: str,
    cell_origin_mode: str,
    include_boundary_images: bool = False,
    element_hidden: dict | None = None,
    atom_hidden: dict | None = None,
) -> list[dict]:
    items = []
    for instance_id, item in enumerate(
        display_atom_instances(
            render_data,
            display_mode=display_mode,
            cell_origin_mode=cell_origin_mode,
            include_boundary_images=include_boundary_images,
        )
    ):
        atom = item["atom"]
        if (element_hidden or {}).get(str(atom.get("element", ""))):
            continue
        if (atom_hidden or {}).get(str(atom.get("index", ""))):
            continue
        items.append(
            {
                "instance_id": instance_id,
                "source_atom": int(atom["index"]),
                "element": atom["element"],
                "atomic_number": int(atom["atomic_number"]),
                "cart": item["cart"],
                "display_shift_cart": item["display_shift_cart"],
                "is_primary_image": bool(item["is_primary_image"]),
            }
        )
    return items


def display_unit_cell_api_item(
    render_data: dict,
    *,
    display_mode: str,
    cell_origin_mode: str,
) -> dict | None:
    unit_cell = render_data.get("unit_cell")
    if unit_cell is None:
        return None
    lattice = np.asarray(unit_cell["lattice"], dtype=float)
    # Match the current PyVista browser view: Cell Range expands atom/symmetry
    # display context, but the drawn reference unit cell itself remains the
    # source cell.  Only the origin convention changes the cell placement.
    lower, _ = display_fractional_bounds("source", cell_origin_mode)
    shift = np.asarray([lower, lower, lower], dtype=float) @ lattice
    return {
        "vertices_cart": np.asarray(unit_cell["vertices_cart"], dtype=float) + shift,
        "edges": unit_cell["edges"],
    }


def atom_motion_api_items(
    render_data: dict,
    atom_mappings: dict | None,
    operation_index: int | None,
    *,
    paths: dict[int, dict] | None = None,
) -> list[dict]:
    mapping = selected_mapping(atom_mappings, operation_index)
    if mapping is None:
        return []
    atoms_by_index = {atom["index"]: atom for atom in render_data.get("atoms", [])}
    inverse_lattice = None
    unit_cell = render_data.get("unit_cell")
    if unit_cell is not None:
        try:
            inverse_lattice = np.linalg.inv(np.asarray(unit_cell["lattice"], dtype=float))
        except np.linalg.LinAlgError:
            inverse_lattice = None
    items = []
    for entry in mapping.get("entries", []):
        source_index = entry.get("source_atom")
        source_atom = atoms_by_index.get(source_index, {})
        item = {
            "source_atom": source_index,
            "target_atom": entry.get("target_atom"),
            "start_frac": source_atom.get("frac"),
            "start_cart": source_atom.get("cart"),
            "target_frac": entry.get("animation_frac") or entry.get("transformed_frac") or entry.get("wrapped_frac"),
            "target_cart": entry.get("transformed_cart"),
            "wrapped_frac": entry.get("wrapped_frac"),
            "animation_frac": entry.get("animation_frac"),
            "distance": entry.get("distance"),
        }
        path = (paths or {}).get(int(source_index)) if source_index is not None else None
        item["stages"] = compound_path_stage_items(path, inverse_lattice=inverse_lattice)
        items.append(item)
    return items


def compound_path_stage_items(
    path: dict | None,
    *,
    inverse_lattice: np.ndarray | None,
) -> list[dict]:
    if path is None or path.get("type") not in {"screw", "glide", "rotoinversion", "rotoreflection"}:
        return []
    start = np.asarray(path["start"], dtype=float)
    _, _, first_length, second_length = two_phase_geometry(path, start)
    progress = float(path.get("phase_fraction", phase_fraction(first_length, second_length)))
    position_cart = evaluate_path(path, progress)
    position_frac = None if inverse_lattice is None else position_cart @ inverse_lattice
    return [{
        "progress": progress,
        "cart": position_cart.tolist(),
        "frac": None if position_frac is None else position_frac.tolist(),
    }]


def compose_operation_indices(
    render_data: dict,
    operation_indices: list[int],
    tolerance_cart: float,
) -> dict:
    if not operation_indices:
        return {"error": "Operation sequence is empty"}
    unit_cell = render_data.get("unit_cell")
    if unit_cell is None:
        return {"error": "No unit cell in render_data"}
    operations = render_data.get("operations", [])
    selected_operations = []
    for index in operation_indices:
        operation = operation_by_index(operations, index)
        if operation is None:
            return {"error": f"Operation index not found: {index}"}
        if operation.get("matrix_frac") is None or operation.get("translation_frac") is None:
            return {"error": f"Operation {index} does not have fractional matrix data"}
        selected_operations.append(operation)

    composed = compose_operation_sequence(selected_operations)
    check_result = check_custom_operation(
        render_data,
        composed.W,
        composed.t,
        tolerance_cart,
    )
    check_result["W_frac"] = composed.W.tolist()
    check_result["t_frac"] = composed.t.tolist()
    check_result.update(cartesian_operation_items(unit_cell, composed.W, composed.t))
    check_result["operation_indices"] = operation_indices
    check_result["matching_operation_index"] = find_matching_operation_index(
        composed.W,
        composed.t,
        operations,
    )
    return check_result


def compose_operation_sequence_items(
    render_data: dict,
    sequence_items: list[dict],
    tolerance_cart: float,
) -> dict:
    if not sequence_items:
        return {"error": "Operation sequence is empty"}
    unit_cell = render_data.get("unit_cell")
    if unit_cell is None:
        return {"error": "No unit cell in render_data"}
    operations = render_data.get("operations", [])
    selected_operations = []
    labels = []
    for item in sequence_items:
        kind = str(item.get("type") or "operation")
        if kind == "operation":
            try:
                index = int(item["index"])
            except (KeyError, TypeError, ValueError) as exc:
                return {"error": f"Operation item requires an integer index: {exc}"}
            operation = operation_by_index(operations, index)
            if operation is None:
                return {"error": f"Operation index not found: {index}"}
            if operation.get("matrix_frac") is None or operation.get("translation_frac") is None:
                return {"error": f"Operation {index} does not have fractional matrix data"}
            selected_operations.append(operation)
            labels.append(f"op {index}")
        elif kind == "custom":
            if item.get("W_frac") is None or item.get("t_frac") is None:
                return {"error": "Custom operation item requires W_frac and t_frac"}
            matrix_error = custom_operation_matrix_error(item["W_frac"], item["t_frac"])
            if matrix_error is not None:
                return {"error": matrix_error}
            selected_operations.append({
                "matrix_frac": item["W_frac"],
                "translation_frac": item["t_frac"],
            })
            labels.append(str(item.get("label") or "custom"))
        else:
            return {"error": f"Unknown sequence item type: {kind}"}

    composed = compose_operation_sequence(selected_operations)
    check_result = check_custom_operation(
        render_data,
        composed.W,
        composed.t,
        tolerance_cart,
    )
    check_result["W_frac"] = composed.W.tolist()
    check_result["t_frac"] = composed.t.tolist()
    check_result.update(cartesian_operation_items(unit_cell, composed.W, composed.t))
    check_result["sequence_items"] = sequence_items
    check_result["sequence_labels"] = labels
    check_result["matching_operation_index"] = find_matching_operation_index(
        composed.W,
        composed.t,
        operations,
    )
    return check_result


def cartesian_operation_items(unit_cell: dict, W_frac, t_frac) -> dict:
    lattice = np.asarray(unit_cell["lattice"], dtype=float)
    W = np.asarray(W_frac, dtype=float)
    t = np.asarray(t_frac, dtype=float)
    W_cart = lattice.T @ W @ np.linalg.inv(lattice.T)
    t_cart = t @ lattice
    return {
        "matrix_cart": W_cart.tolist(),
        "translation_cart": t_cart.tolist(),
    }


def update_view_coordinate_state(render_data: dict, state: dict, update: dict) -> None:
    """Attach Python-computed Cartesian camera inputs to shared browser state."""
    unit_cell = render_data.get("unit_cell")
    lattice = None if unit_cell is None else np.asarray(unit_cell["lattice"], dtype=float)

    def vector(key: str) -> np.ndarray:
        value = np.asarray(update.get(key), dtype=float)
        if value.shape != (3,) or not np.all(np.isfinite(value)):
            raise ValueError(f"{key} must contain three finite numbers")
        return value

    if "reset_view_request_id" in update:
        state["reset_view_center_cart"] = display_scene_center(
            render_data,
            str(state.get("display_mode", "source")),
            str(state.get("cell_origin_mode", "center")),
        ).tolist()
    if "view_center_request_id" in update:
        center = vector("view_center_frac")
        state["view_center_cart"] = (center if lattice is None else center @ lattice).tolist()
    if "view_direction_request_id" in update:
        if lattice is None:
            raise ValueError("direction indices require a crystal unit cell")
        direction = vector("view_direction_frac") @ lattice
        if np.linalg.norm(direction) < 1e-10:
            raise ValueError("view direction must not be zero")
        state["view_direction_cart"] = (direction / np.linalg.norm(direction)).tolist()
        state["indexed_view_center_cart"] = display_scene_center(
            render_data,
            str(state.get("display_mode", "source")),
            str(state.get("cell_origin_mode", "center")),
        ).tolist()
    if "view_plane_request_id" in update:
        if lattice is None:
            raise ValueError("Miller indices require a crystal unit cell")
        normal = np.linalg.inv(lattice) @ vector("view_plane_hkl")
        if np.linalg.norm(normal) < 1e-10:
            raise ValueError("plane normal must not be zero")
        state["view_plane_normal_cart"] = (normal / np.linalg.norm(normal)).tolist()
        state["indexed_view_center_cart"] = display_scene_center(
            render_data,
            str(state.get("display_mode", "source")),
            str(state.get("cell_origin_mode", "center")),
        ).tolist()


def gif_bytes_from_data_urls(frames: list[str], frame_duration_ms: int) -> bytes:
    if not frames or len(frames) > 180:
        raise ValueError("GIF export requires between 1 and 180 frames")
    images = []
    try:
        for frame in frames:
            prefix = "data:image/png;base64,"
            if not isinstance(frame, str) or not frame.startswith(prefix):
                raise ValueError("GIF frames must be PNG data URLs")
            encoded = frame[len(prefix):]
            if len(encoded) > 8_000_000:
                raise ValueError("A GIF frame is too large")
            with Image.open(io.BytesIO(base64.b64decode(encoded, validate=True))) as source:
                image = source.convert("RGB")
            images.append(image)
        output = io.BytesIO()
        images[0].save(
            output,
            format="GIF",
            save_all=True,
            append_images=images[1:],
            duration=max(20, min(int(frame_duration_ms), 1000)),
            loop=0,
            optimize=False,
        )
        return output.getvalue()
    except (ValueError, TypeError, binascii.Error, OSError) as exc:
        raise ValueError(f"Invalid GIF frame data: {exc}") from exc
    finally:
        for image in images:
            image.close()


def find_operation_sequence_for_target(
    render_data: dict,
    target_operation_index: int,
    generator_indices: list[int],
    max_depth: int,
) -> dict:
    if not generator_indices:
        return {"error": "Generator sequence is empty"}
    operations = render_data.get("operations", [])
    target = operation_by_index(operations, target_operation_index)
    if target is None:
        return {"error": f"Target operation index not found: {target_operation_index}"}

    generators = []
    for index in generator_indices:
        operation = operation_by_index(operations, index)
        if operation is None:
            return {"error": f"Generator operation index not found: {index}"}
        generators.append(operation)

    sequence = find_operation_sequence_bfs(
        target,
        generators,
        operations,
        max_depth=max_depth,
    )
    return {
        "target_operation_index": target_operation_index,
        "generator_indices": generator_indices,
        "max_depth": max_depth,
        "sequence": None if sequence is None else list(sequence),
        "found": sequence is not None,
    }


def puzzle_public_questions(
    render_data: dict,
    atom_mappings: dict | None,
    puzzle_type: str,
    *,
    display_mode: str = "source",
    cell_origin_mode: str = "center",
    improper_mode: str = "auto",
) -> list[dict]:
    """Public puzzle questions (correct answer withheld).

    The highlighted axis geometry is taken from the SAME per-operation
    computation the analysis view uses (``symmetry_elements_response``), so the
    displayed axis coincides with the reveal animation. ``render_data["axes"]``
    can list several symmetry-equivalent parallel axes and the axis grouping may
    pick a different one than the operation actually acts on, which is what made
    the drawn axis and the animation disagree for crystals such as MgHPO3.
    """
    questions = axis_order_questions(render_data, puzzle_type)
    public: list[dict] = []
    for index, question in enumerate(questions):
        direction = question["direction_cart"]
        point = question["point_cart"]
        op_index = question["reveal_operations"].get(max(question["correct_orders"])) if question["correct_orders"] else None
        if op_index is not None:
            try:
                elements = symmetry_elements_response(
                    render_data,
                    atom_mappings,
                    op_index,
                    improper_mode=improper_mode,
                    display_mode=display_mode,
                    cell_origin_mode=cell_origin_mode,
                )
            except (ValueError, KeyError, TypeError):
                elements = None
            axes = (elements or {}).get("axes") or []
            if axes:
                direction = axes[0]["direction_cart"]
                point = axes[0]["point_cart"]
        public.append(
            {
                "id": index,
                "direction_cart": direction,
                "point_cart": point,
                "options": list(ROTATION_ORDER_OPTIONS),
                "equivalent_count": question["equivalent_count"],
                "infinite": question["infinite"],
            }
        )
    return public


def example_catalog() -> dict[str, list[dict]]:
    global _EXAMPLE_CATALOG_CACHE
    if _EXAMPLE_CATALOG_CACHE is not None:
        return _EXAMPLE_CATALOG_CACHE
    if EXAMPLE_CATALOG_PATH.exists():
        loaded = json.loads(EXAMPLE_CATALOG_PATH.read_text(encoding="utf-8"))
        _EXAMPLE_CATALOG_CACHE = normalize_example_catalog(loaded)
    else:
        _EXAMPLE_CATALOG_CACHE = filesystem_example_catalog()
    return _EXAMPLE_CATALOG_CACHE


def normalize_example_catalog(loaded: dict) -> dict[str, list[dict]]:
    catalog = {SOURCE_KIND_CRYSTAL: [], SOURCE_KIND_MOLECULE: []}
    for kind in (SOURCE_KIND_CRYSTAL, SOURCE_KIND_MOLECULE):
        for raw in loaded.get(kind, []):
            path = str(raw.get("path") or "")
            if not path:
                continue
            item = {
                "kind": kind,
                "name": str(raw.get("name") or Path(path).stem),
                "path": path,
                "formula": str(raw.get("formula") or ""),
                "display_formula": str(raw.get("display_formula") or raw.get("formula") or ""),
                "symmetry": str(raw.get("symmetry") or ""),
                "point_group": str(raw.get("point_group") or ""),
            }
            # Absent means playable, matching the puzzle_counts convention.
            item["beyond_quiz_vocabulary"] = bool(raw.get("beyond_quiz_vocabulary"))
            counts = raw.get("puzzle_counts")
            if isinstance(counts, dict):
                # Absent counts mean "unknown", and the picker treats unknown as
                # playable; only pass on counts we actually have.
                item["puzzle_counts"] = {
                    key: int(counts.get(key, 0)) for key in PUZZLE_COUNT_KEYS
                }
            if raw.get("error"):
                item["error"] = str(raw["error"])
            catalog[kind].append(item)
    return catalog


def filesystem_example_catalog() -> dict[str, list[dict]]:
    return {
        SOURCE_KIND_CRYSTAL: filesystem_example_items(SOURCE_KIND_CRYSTAL),
        SOURCE_KIND_MOLECULE: filesystem_example_items(SOURCE_KIND_MOLECULE),
    }


def filesystem_example_items(kind: str) -> list[dict]:
    directory = EXAMPLE_DIRS[kind]
    suffix = EXAMPLE_SUFFIXES[kind]
    items: list[dict] = []
    for path in sorted(directory.glob(f"*{suffix}")):
        items.append({
            "kind": kind,
            "name": path.stem,
            "path": str(path),
            "formula": "",
            "symmetry": "",
            "point_group": "",
        })
    return items


def resolve_example_path(kind: str, requested_path: str) -> Path:
    kind = normalize_source_kind(kind)
    if kind not in EXAMPLE_DIRS:
        raise ValueError(f"unsupported example kind: {kind}")
    base = EXAMPLE_DIRS[kind].resolve()
    path = Path(requested_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve()
    if path.suffix.lower() != EXAMPLE_SUFFIXES[kind]:
        raise ValueError(f"example must be a {EXAMPLE_SUFFIXES[kind]} file")
    if path.parent != base:
        raise ValueError("example path is outside the allowed examples directory")
    if not path.exists():
        raise FileNotFoundError(f"example not found: {requested_path}")
    return path


def write_uploaded_text(filename: str, content: str, *, upload_dir: Path, suffix: str, label: str) -> Path:
    if not content.strip():
        raise ValueError(f"{label} content is empty.")
    if len(content.encode("utf-8")) > 20 * 1024 * 1024:
        raise ValueError(f"{label} upload is too large.")
    slug = slug_from_path(filename or "uploaded_structure")
    path = upload_dir / f"{slug}{suffix}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def write_uploaded_cif(filename: str, content: str) -> Path:
    return write_uploaded_text(filename, content, upload_dir=UPLOAD_CIF_DIR, suffix=".cif", label="CIF")


def write_uploaded_molecule(filename: str, content: str) -> Path:
    return write_uploaded_text(filename, content, upload_dir=UPLOAD_MOLECULE_DIR, suffix=".xyz", label="Molecule")


def make_handler(
    session: ViewerSession,
    shared_state: dict,
    state_lock: threading.Lock,
    *,
    json_dir: Path,
    import_json_dir: Path,
    tolerance_cart: float,
    indent: int,
    analysis_timeout_sec: float,
    default_display_mode: str,
    initial_app_mode: str = "select",
) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed_url = urlparse(self.path)
            path = parsed_url.path
            if path == "/":
                page = HTML.replace("__INITIAL_APP_MODE__", json.dumps(initial_app_mode))
                self.send_bytes(page.encode("utf-8"), content_type="text/html; charset=utf-8")
                return
            if path == "/static/animation_path.js":
                module_path = PROJECT_ROOT / "crystal_viewer" / "web" / "animation_path.js"
                self.send_javascript_file(module_path)
                return
            if path == "/static/colors.js":
                module_path = PROJECT_ROOT / "crystal_viewer" / "web" / "colors.js"
                self.send_javascript_file(module_path)
                return
            if path == "/static/browser_ui.css":
                stylesheet_path = PROJECT_ROOT / "crystal_viewer" / "web" / "browser_ui.css"
                self.send_stylesheet_file(stylesheet_path)
                return
            if path == "/static/browser_ui.js":
                module_path = PROJECT_ROOT / "crystal_viewer" / "web" / "browser_ui.js"
                self.send_javascript_file(module_path)
                return
            if path == "/static/three_loader.js":
                module_path = PROJECT_ROOT / "crystal_viewer" / "web" / "three_loader.js"
                self.send_javascript_file(module_path)
                return
            if path == "/static/three_view.js":
                module_path = PROJECT_ROOT / "crystal_viewer" / "web" / "three_view.js"
                self.send_javascript_file(module_path)
                return
            if path == "/static/puzzle.js":
                module_path = PROJECT_ROOT / "crystal_viewer" / "web" / "puzzle.js"
                self.send_javascript_file(module_path)
                return
            if path == "/vendor/three/three.module.js":
                module_path = PROJECT_ROOT / "crystal_viewer" / "web" / "vendor" / "three" / "three.module.js"
                self.send_javascript_file(module_path, vendor=True)
                return
            if path == "/vendor/three/three.core.js":
                module_path = PROJECT_ROOT / "crystal_viewer" / "web" / "vendor" / "three" / "three.core.js"
                self.send_javascript_file(module_path, vendor=True)
                return
            if path == "/vendor/three/addons/controls/TrackballControls.js":
                # Its bare `from 'three'` import is already rewritten in the
                # vendored copy; see crystal_viewer/web/vendor/three/README.md.
                module_path = PROJECT_ROOT / "crystal_viewer" / "web" / "vendor" / "three" / "addons" / "controls" / "TrackballControls.js"
                self.send_javascript_file(module_path, vendor=True)
                return
            if path == "/api/operations":
                with state_lock:
                    ready = shared_state.get("summaries_ready", True)
                    operations = session.operation_summary_items
                body = {
                    "operations": operations,
                    "summaries_ready": ready,
                }
                self.send_json(body)
                return
            if path == "/api/atoms":
                with state_lock:
                    atoms = list(session.atoms)
                body = {"atoms": atom_api_items(atoms)}
                self.send_json(body)
                return
            if path == "/api/render_data":
                with state_lock:
                    render_data = session.render_data
                    schema_version = session.payload.get("schema_version")
                    source_kind = session.source_kind
                    display_mode = str(shared_state.get("display_mode", "source"))
                    cell_origin_mode = str(shared_state.get("cell_origin_mode", "center"))
                    include_boundary_images = bool(shared_state.get("include_boundary_images", False))
                    element_colors = dict(shared_state.get("element_colors", {}))
                    atom_colors = dict(shared_state.get("atom_colors", {}))
                    element_hidden = dict(shared_state.get("element_hidden", {}))
                    atom_hidden = dict(shared_state.get("atom_hidden", {}))
                # An explicit ?boundary_images= overrides the shared analysis state
                # (the puzzle shows the cell's equivalent boundary atoms without
                # toggling the analysis mode's own setting).
                boundary_override = parse_qs(parsed_url.query).get("boundary_images")
                if boundary_override:
                    include_boundary_images = boundary_override[0] in ("1", "true", "True")
                display_override = parse_qs(parsed_url.query).get("display_mode")
                if display_override:
                    display_mode = str(display_override[0])
                origin_override = parse_qs(parsed_url.query).get("cell_origin_mode")
                if origin_override:
                    cell_origin_mode = str(origin_override[0])
                body = {
                    "schema_version": schema_version,
                    "source_kind": source_kind,
                    "coordinate_space": "cartesian",
                    "render_data": render_data,
                    "display_mode": display_mode,
                    "cell_origin_mode": cell_origin_mode,
                    "include_boundary_images": include_boundary_images,
                    "display_atoms": display_atom_api_items(
                        render_data,
                        display_mode=display_mode,
                        cell_origin_mode=cell_origin_mode,
                        include_boundary_images=include_boundary_images,
                        element_hidden=element_hidden,
                        atom_hidden=atom_hidden,
                    ),
                    "display_unit_cell": display_unit_cell_api_item(
                        render_data,
                        display_mode=display_mode,
                        cell_origin_mode=cell_origin_mode,
                    ),
                    "atom_styles": atom_render_style_items(
                        render_data,
                        element_colors=element_colors,
                        atom_colors=atom_colors,
                    ),
                }
                self.send_json(body)
                return
            if path == "/api/animation_path":
                query = parse_qs(parsed_url.query)
                try:
                    with state_lock:
                        operation_index = int(
                            query.get("operation_index", [shared_state.get("operation_index", 0)])[0]
                        )
                        render_data = session.render_data
                        atom_mappings = session.atom_mappings
                        # An explicit ?scope= overrides the shared analysis state
                        # (the puzzle forces "displayed" so its reveal always moves
                        # every atom, regardless of any selection left in analysis).
                        scope = str(query.get("scope", [shared_state.get("scope", "displayed")])[0])
                        selected_atoms = tuple(
                            int(index) for index in shared_state.get("selected_atoms", [])
                        )
                        improper_mode = str(shared_state.get("improper_mode", "auto"))
                        display_mode = str(shared_state.get("display_mode", "source"))
                        cell_origin_mode = str(shared_state.get("cell_origin_mode", "center"))
                        include_boundary_images = bool(shared_state.get("include_boundary_images", False))
                        animation_boundary_mode = str(
                            shared_state.get("animation_boundary_mode", "continuous")
                        )
                    display_override = query.get("display_mode")
                    if display_override:
                        display_mode = str(display_override[0])
                    origin_override = query.get("cell_origin_mode")
                    if origin_override:
                        cell_origin_mode = str(origin_override[0])
                    boundary_override = query.get("boundary_images")
                    if boundary_override:
                        include_boundary_images = boundary_override[0] in ("1", "true", "True")
                    body = animation_path_response(
                        render_data,
                        atom_mappings,
                        operation_index,
                        scope=scope,
                        selected_atoms=selected_atoms,
                        improper_mode=improper_mode,
                        display_mode=display_mode,
                        cell_origin_mode=cell_origin_mode,
                        include_boundary_images=include_boundary_images,
                        animation_boundary_mode=animation_boundary_mode,
                    )
                except (TypeError, ValueError) as exc:
                    self.send_json_error(str(exc), status=400)
                    return
                self.send_json(body)
                return
            if path == "/api/custom_animation_path":
                with state_lock:
                    request = dict(shared_state.get("custom_op_animate") or {})
                    render_data = session.render_data
                    atom_mappings = session.atom_mappings
                    improper_mode = str(shared_state.get("improper_mode", "auto"))
                    display_mode = str(shared_state.get("display_mode", "source"))
                    cell_origin_mode = str(shared_state.get("cell_origin_mode", "center"))
                    include_boundary_images = bool(shared_state.get("include_boundary_images", False))
                    animation_boundary_mode = str(
                        shared_state.get("animation_boundary_mode", "continuous")
                    )
                if not request:
                    self.send_json_error("No custom animation is active", status=404)
                    return
                self.send_json(custom_animation_path_response(
                    render_data, atom_mappings, request, improper_mode=improper_mode,
                    display_mode=display_mode, cell_origin_mode=cell_origin_mode,
                    include_boundary_images=include_boundary_images,
                    animation_boundary_mode=animation_boundary_mode,
                ))
                return
            if path == "/api/symmetry_elements":
                query = parse_qs(parsed_url.query)
                try:
                    with state_lock:
                        operation_index = int(
                            query.get("operation_index", [shared_state.get("operation_index", 0)])[0]
                        )
                        render_data = session.render_data
                        atom_mappings = session.atom_mappings
                        improper_mode = str(shared_state.get("improper_mode", "auto"))
                        display_mode = str(shared_state.get("display_mode", "source"))
                        cell_origin_mode = str(shared_state.get("cell_origin_mode", "center"))
                    display_override = query.get("display_mode")
                    if display_override:
                        display_mode = str(display_override[0])
                    origin_override = query.get("cell_origin_mode")
                    if origin_override:
                        cell_origin_mode = str(origin_override[0])
                    body = symmetry_elements_response(
                        render_data,
                        atom_mappings,
                        operation_index,
                        improper_mode=improper_mode,
                        display_mode=display_mode,
                        cell_origin_mode=cell_origin_mode,
                    )
                except (TypeError, ValueError) as exc:
                    self.send_json_error(str(exc), status=400)
                    return
                self.send_json(body)
                return
            if path == "/api/atom_motion":
                with state_lock:
                    render_data = session.render_data
                    atom_mappings = session.atom_mappings
                    operation_index = int(shared_state.get("operation_index", 0))
                    improper_mode = str(shared_state.get("improper_mode", "auto"))
                    display_mode = str(shared_state.get("display_mode", "source"))
                    cell_origin_mode = str(shared_state.get("cell_origin_mode", "center"))
                operation = operation_by_index(render_data.get("operations", []), operation_index)
                mapping = selected_mapping(atom_mappings, operation_index)
                paths = {}
                kind = str((operation or {}).get("kind", ""))
                compound_operation = (
                    kind.startswith("screw")
                    or "glide" in kind
                    or "rotoinversion" in kind
                    or "rotoreflection" in kind
                    or "improper" in kind
                )
                if operation is not None and mapping is not None and compound_operation:
                    paths = animation_paths(
                        render_data,
                        operation,
                        mapping,
                        animation_scope="all",
                        improper_mode=improper_mode,
                        display_mode=display_mode,
                        cell_origin_mode=cell_origin_mode,
                    )
                body = {
                    "operation_index": operation_index,
                    "entries": atom_motion_api_items(
                        render_data,
                        atom_mappings,
                        operation_index,
                        paths=paths,
                    ),
                }
                self.send_json(body)
                return
            if path == "/api/puzzle/axis_orders":
                query = parse_qs(parsed_url.query)
                puzzle_type = query.get("type", ["rotation"])[0]
                if puzzle_type not in ("rotation", "improper"):
                    self.send_json_error("unknown puzzle type", status=400)
                    return
                with state_lock:
                    render_data = session.render_data
                    source_kind = session.source_kind
                    atom_mappings = session.atom_mappings
                    display_mode = str(shared_state.get("display_mode", "source"))
                    cell_origin_mode = str(shared_state.get("cell_origin_mode", "center"))
                    improper_mode = str(shared_state.get("improper_mode", "auto"))
                display_override = query.get("display_mode")
                if display_override:
                    display_mode = str(display_override[0])
                origin_override = query.get("cell_origin_mode")
                if origin_override:
                    cell_origin_mode = str(origin_override[0])
                # The correct answer stays server-side; it is revealed only on
                # /api/puzzle/axis_orders/check.
                questions = puzzle_public_questions(
                    render_data,
                    atom_mappings,
                    puzzle_type,
                    display_mode=display_mode,
                    cell_origin_mode=cell_origin_mode,
                    improper_mode=improper_mode,
                )
                self.send_json(
                    {
                        "source_kind": source_kind,
                        "puzzle_type": puzzle_type,
                        "questions": questions,
                    }
                )
                return
            if path == "/api/puzzle/operations":
                difficulty = parse_qs(parsed_url.query).get("difficulty", ["normal"])[0]
                with state_lock:
                    render_data = session.render_data
                    source_kind = session.source_kind
                # Each question carries only the operation index to animate; the
                # operation's name/fold is revealed on /api/puzzle/operations/check.
                self.send_json(
                    {
                        "source_kind": source_kind,
                        "difficulty": difficulty,
                        "questions": public_operation_questions(render_data, difficulty),
                    }
                )
                return
            if path == "/api/puzzle/composition":
                with state_lock:
                    render_data = session.render_data
                    source_kind = session.source_kind
                # Each question carries the two operation indices to animate; the
                # product's name/fold is revealed on /api/puzzle/composition/check.
                self.send_json(
                    {
                        "source_kind": source_kind,
                        "questions": public_composition_questions(render_data),
                    }
                )
                return
            if path == "/api/puzzle/mapping":
                with state_lock:
                    render_data = session.render_data
                    source_kind = session.source_kind
                # Each question carries the operation to animate and the atom to
                # highlight; the target atom is revealed on /api/puzzle/mapping/check.
                self.send_json(
                    {
                        "source_kind": source_kind,
                        "questions": public_mapping_questions(render_data),
                    }
                )
                return
            if path == "/api/puzzle/point_group":
                with state_lock:
                    render_data = session.render_data
                    source_kind = session.source_kind
                # The question carries only the answer choices; which one is
                # correct is revealed by /api/puzzle/point_group/check.
                self.send_json(
                    {
                        "source_kind": source_kind,
                        "questions": public_point_group_questions(render_data),
                    }
                )
                return
            if path == "/api/state":
                with state_lock:
                    body = dict(shared_state)
                self.send_json(body)
                return
            if path == "/api/examples":
                self.send_json(example_catalog())
                return
            self.send_error(404)

        def do_POST(self) -> None:
            path = urlparse(self.path).path
            length = int(self.headers.get("Content-Length", "0"))
            if length > MAX_REQUEST_BODY_BYTES:
                self.send_json_error("request body is too large", status=413)
                return
            try:
                payload = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError as exc:
                self.send_json_error(f"invalid JSON request body: {exc}", status=400)
                return

            if path == "/api/reset_analysis":
                empty_path = write_empty_viewer_json()
                empty_payload = empty_viewer_payload()
                empty_session = ViewerSession(empty_path, empty_payload)
                with state_lock:
                    session.replace_from(empty_session)
                    next_state = initial_render_state(
                        empty_payload,
                        initial_operation=None,
                        display_mode=default_display_mode,
                    )
                    next_state["reload_request_id"] = int(shared_state.get("reload_request_id") or 0) + 1
                    next_state["load_request_id"] = int(shared_state.get("load_request_id") or 0) + 1
                    next_state["json_path"] = str(empty_path)
                    shared_state.clear()
                    shared_state.update(next_state)
                    body = {
                        "ok": True,
                        "operations": session.operation_summary_items,
                        "atoms": atom_api_items(session.atoms),
                        "state": dict(shared_state),
                    }
                self.send_json(body)
                return

            if path == "/api/check_operation":
                with state_lock:
                    render_data = session.render_data
                unit_cell = render_data.get("unit_cell")
                if unit_cell is None:
                    self.send_json({"error": "No unit cell in render_data"})
                    return
                lattice = np.asarray(unit_cell["lattice"], dtype=float)
                op_type = str(payload.get("type", ""))
                params = payload.get("params", {})
                tolerance, tolerance_error = custom_operation_tolerance(payload.get("tolerance", 0.1))
                if tolerance_error is not None:
                    self.send_json({"error": tolerance_error})
                    return
                result = build_custom_operation_frac(op_type, params, lattice)
                if isinstance(result, str):
                    self.send_json({"error": result})
                    return
                W_frac, t_frac = result
                matrix_error = custom_operation_matrix_error(W_frac, t_frac)
                if matrix_error is not None:
                    self.send_json({"error": matrix_error})
                    return
                if payload.get("build_only"):
                    self.send_json({
                        "W_frac": W_frac.tolist(),
                        "t_frac": t_frac.tolist(),
                        **cartesian_operation_items(unit_cell, W_frac, t_frac),
                    })
                    return
                check_result = check_custom_operation(render_data, W_frac, t_frac, tolerance)
                visuals = custom_operation_visuals(op_type, params, lattice, W_frac, t_frac)
                # include the operation matrix so the browser can request animation
                check_result["W_frac"] = W_frac.tolist()
                check_result["t_frac"] = t_frac.tolist()
                check_result.update(cartesian_operation_items(unit_cell, W_frac, t_frac))
                check_result.update(visuals)
                # store result for PyVista highlight
                with state_lock:
                    shared_state["custom_op_result"] = check_result
                self.send_json(check_result)
                return

            if path == "/api/export_gif":
                try:
                    gif_data = gif_bytes_from_data_urls(
                        payload.get("frames", []),
                        int(payload.get("frame_duration_ms", 100)),
                    )
                except (TypeError, ValueError) as exc:
                    self.send_json_error(str(exc), status=400)
                    return
                self.send_bytes(gif_data, content_type="image/gif")
                return

            if path == "/api/puzzle/axis_orders/check":
                with state_lock:
                    render_data = session.render_data
                try:
                    question_id = int(payload.get("question_id"))
                    selected = payload.get("selected_order")
                    puzzle_type = str(payload.get("type", "rotation"))
                    if puzzle_type not in ("rotation", "improper"):
                        raise ValueError("unknown puzzle type")
                    result = check_axis_order_answer(render_data, question_id, selected, puzzle_type)
                except (TypeError, ValueError) as exc:
                    self.send_json_error(f"invalid puzzle answer: {exc}", status=400)
                    return
                if result is None:
                    self.send_json_error("puzzle question not found", status=404)
                    return
                self.send_json(result)
                return

            if path == "/api/puzzle/operations/check":
                with state_lock:
                    render_data = session.render_data
                try:
                    question_id = int(payload.get("question_id"))
                    kind = str(payload.get("kind", ""))
                    order = payload.get("order")
                    difficulty = str(payload.get("difficulty", "normal"))
                    shift = payload.get("shift")
                    result = check_operation_answer(
                        render_data,
                        question_id,
                        kind,
                        order,
                        difficulty,
                        selected_shift=shift,
                    )
                except (TypeError, ValueError) as exc:
                    self.send_json_error(f"invalid puzzle answer: {exc}", status=400)
                    return
                if result is None:
                    self.send_json_error("puzzle question not found", status=404)
                    return
                self.send_json(result)
                return

            if path == "/api/puzzle/composition/check":
                with state_lock:
                    render_data = session.render_data
                try:
                    question_id = int(payload.get("question_id"))
                    kind = str(payload.get("kind", ""))
                    order = payload.get("order")
                    result = check_composition_answer(render_data, question_id, kind, order)
                except (TypeError, ValueError) as exc:
                    self.send_json_error(f"invalid puzzle answer: {exc}", status=400)
                    return
                if result is None:
                    self.send_json_error("puzzle question not found", status=404)
                    return
                self.send_json(result)
                return

            if path == "/api/puzzle/mapping/check":
                with state_lock:
                    render_data = session.render_data
                try:
                    question_id = int(payload.get("question_id"))
                    selected_atom = payload.get("selected_atom_index")
                    result = check_mapping_answer(render_data, question_id, selected_atom)
                except (TypeError, ValueError) as exc:
                    self.send_json_error(f"invalid puzzle answer: {exc}", status=400)
                    return
                if result is None:
                    self.send_json_error("puzzle question not found", status=404)
                    return
                self.send_json(result)
                return

            if path == "/api/puzzle/point_group/check":
                with state_lock:
                    render_data = session.render_data
                try:
                    question_id = int(payload.get("question_id"))
                    selected = payload.get("selected")
                    result = check_point_group_answer(render_data, question_id, selected)
                except (TypeError, ValueError) as exc:
                    self.send_json_error(f"invalid puzzle answer: {exc}", status=400)
                    return
                if result is None:
                    self.send_json_error("puzzle question not found", status=404)
                    return
                self.send_json(result)
                return

            if path == "/api/compose_operations":
                with state_lock:
                    render_data = session.render_data
                try:
                    operation_indices = [
                        int(index)
                        for index in payload.get("operation_indices", [])
                    ]
                except (TypeError, ValueError) as exc:
                    self.send_json({"error": f"Parameter error: {exc}"})
                    return
                tolerance, tolerance_error = custom_operation_tolerance(payload.get("tolerance", 0.1))
                if tolerance_error is not None:
                    self.send_json({"error": tolerance_error})
                    return
                if "sequence_items" in payload:
                    check_result = compose_operation_sequence_items(
                        render_data,
                        payload.get("sequence_items", []),
                        tolerance,
                    )
                else:
                    check_result = compose_operation_indices(render_data, operation_indices, tolerance)
                if "error" not in check_result and payload.get("store_custom_result"):
                    with state_lock:
                        shared_state["custom_op_result"] = check_result
                self.send_json(check_result)
                return

            if path == "/api/find_operation_sequence":
                with state_lock:
                    render_data = session.render_data
                try:
                    target_operation_index = int(payload.get("target_operation_index"))
                    generator_indices = [
                        int(index)
                        for index in payload.get("generator_indices", [])
                    ]
                    max_depth = int(payload.get("max_depth", 4))
                except (TypeError, ValueError) as exc:
                    self.send_json({"error": f"Parameter error: {exc}"})
                    return
                self.send_json(
                    find_operation_sequence_for_target(
                        render_data,
                        target_operation_index,
                        generator_indices,
                        max_depth,
                    )
                )
                return

            if path == "/api/import_cif":
                self.handle_import_cif(payload)
                return

            if path == "/api/import_molecule":
                self.handle_import_molecule(payload)
                return

            if path == "/api/open_example":
                self.handle_open_example(payload)
                return

            if path == "/api/cell_setting":
                self.handle_cell_setting(payload)
                return

            if path != "/api/state":
                self.send_error(404)
                return

            with state_lock:
                try:
                    candidate_state = dict(shared_state)
                    apply_render_state_update(candidate_state, payload)
                    update_view_coordinate_state(session.render_data, candidate_state, payload)
                    shared_state.clear()
                    shared_state.update(candidate_state)
                    body = dict(candidate_state)
                except (ValueError, np.linalg.LinAlgError) as exc:
                    body = None
                    error = str(exc)
            if body is None:
                self.send_json_error(error, status=400)
                return
            self.send_json(body)

        def handle_open_example(self, payload: dict) -> None:
            started_at = time.monotonic()
            request_id = self.reserve_load_request(payload)
            kind = normalize_source_kind(str(payload.get("kind") or ""))
            requested_path = str(payload.get("path") or "")
            try:
                input_path = resolve_example_path(kind, requested_path)
                json_path = default_json_output_path(input_path, EXAMPLE_JSON_DIR)
                json_path, new_payload = export_analysis_to_json_worker_cached(
                    input_path,
                    mode=kind,
                    output_path=json_path,
                    tolerance_cart=tolerance_cart,
                    indent=indent,
                    timeout_sec=analysis_timeout_sec,
                )
                debug_import_timing("example export/cache", started_at)
                if new_payload is None:
                    new_payload = json.loads(json_path.read_text(encoding="utf-8"))
                    debug_import_timing("example read-json", started_at)
                else:
                    debug_import_timing("example cached-json", started_at)
                self.load_payload(json_path, new_payload, f"loaded example {input_path.name}", request_id=request_id)
                debug_import_timing("example response", started_at)
            except Exception as exc:
                with state_lock:
                    self.clear_import_if_current(request_id)
                    shared_state["import_status"] = f"example load failed: {exc}"
                    body = {"ok": False, "error": str(exc), "state": dict(shared_state)}
                self.send_json(body)

        def reserve_load_request(self, payload: dict) -> int:
            request_id = int(payload.get("request_id") or 0)
            with state_lock:
                shared_state["load_request_id"] = max(
                    int(shared_state.get("load_request_id") or 0),
                    request_id,
                )
                shared_state["import_in_progress"] = True
            return request_id

        def load_request_is_current(self, request_id: int) -> bool:
            return request_id == 0 or request_id == int(shared_state.get("load_request_id") or 0)

        def clear_import_if_current(self, request_id: int) -> None:
            if self.load_request_is_current(request_id):
                shared_state["import_in_progress"] = False

        def handle_import_cif(self, payload: dict) -> None:
            started_at = time.monotonic()
            request_id = self.reserve_load_request(payload)
            filename = str(payload.get("filename") or "uploaded_structure.cif")
            content = str(payload.get("content") or "")
            try:
                if not content.strip():
                    raise ValueError(f"empty CIF file: {filename}")
                cif_path = write_uploaded_cif(filename, content)
                debug_import_timing("cif write", started_at)
                json_path = default_json_output_path(cif_path, import_json_dir)
                json_path = export_analysis_to_json_worker(
                    cif_path,
                    mode=SOURCE_KIND_CRYSTAL,
                    output_path=json_path,
                    tolerance_cart=tolerance_cart,
                    indent=indent,
                    timeout_sec=analysis_timeout_sec,
                )
                debug_import_timing("cif export", started_at)
                new_payload = json.loads(json_path.read_text(encoding="utf-8"))
                debug_import_timing("cif read-json", started_at)
                self.load_payload(json_path, new_payload, f"loaded {filename} -> {json_path}", request_id=request_id)
                debug_import_timing("cif response", started_at)
            except Exception as exc:
                with state_lock:
                    self.clear_import_if_current(request_id)
                    shared_state["import_status"] = f"import failed: {exc}"
                    body = {"ok": False, "error": str(exc), "state": dict(shared_state)}
                self.send_json(body)

        def handle_import_molecule(self, payload: dict) -> None:
            started_at = time.monotonic()
            request_id = self.reserve_load_request(payload)
            filename = str(payload.get("filename") or "uploaded_molecule.xyz")
            content = str(payload.get("content") or "")
            try:
                if not content.strip():
                    raise ValueError(f"empty molecule file: {filename}")
                molecule_path = write_uploaded_molecule(filename, content)
                debug_import_timing("molecule write", started_at)
                json_path = default_json_output_path(molecule_path, import_json_dir)
                json_path = export_analysis_to_json_worker(
                    molecule_path,
                    mode=SOURCE_KIND_MOLECULE,
                    output_path=json_path,
                    tolerance_cart=tolerance_cart,
                    indent=indent,
                    timeout_sec=analysis_timeout_sec,
                )
                debug_import_timing("molecule export", started_at)
                new_payload = json.loads(json_path.read_text(encoding="utf-8"))
                debug_import_timing("molecule read-json", started_at)
                self.load_payload(json_path, new_payload, f"loaded {filename} -> {json_path}", request_id=request_id)
                debug_import_timing("molecule response", started_at)
            except Exception as exc:
                with state_lock:
                    self.clear_import_if_current(request_id)
                    shared_state["import_status"] = f"molecule import failed: {exc}"
                    body = {"ok": False, "error": str(exc), "state": dict(shared_state)}
                self.send_json(body)

        def handle_cell_setting(self, payload: dict) -> None:
            mode = str(payload.get("cell_setting_mode") or payload.get("mode") or "native")
            try:
                with state_lock:
                    if shared_state.get("import_in_progress"):
                        raise RuntimeError("A structure is still loading. Wait for it to finish before changing cell setting.")
                    base_payload = session.base_payload
                    current_payload = session.payload
                    current_json_path = session.json_path
                    preserved_status = shared_state.get("import_status", "")
                source_payload = base_payload if mode == "native" else current_payload
                converted_payload = export_cell_setting_json_worker(
                    source_payload,
                    cell_setting=mode,
                    tolerance_cart=tolerance_cart,
                    indent=indent,
                    timeout_sec=analysis_timeout_sec,
                    # A primitive P cell or a conventional P/Bravais cell can
                    # legitimately keep its atom count.  It is still a valid
                    # request to standardize and label that cell setting.
                    require_distinct=False,
                )
                new_session = ViewerSession(current_json_path, converted_payload, base_payload=base_payload)
                with state_lock:
                    if shared_state.get("import_in_progress"):
                        raise RuntimeError("A structure started loading before cell setting conversion finished.")
                    if session.json_path != current_json_path or session.payload is not current_payload:
                        raise RuntimeError("The loaded structure changed before cell setting conversion finished.")
                    preserved = {
                        **preserved_render_state(shared_state),
                        "cell_setting_mode": mode,
                        "reload_request_id": shared_state.get("reload_request_id"),
                        "import_status": preserved_status,
                        "json_path": shared_state.get("json_path", str(current_json_path)),
                        "pyvista_enabled": bool(shared_state.get("pyvista_enabled", False)),
                    }
                    session.replace_from(new_session)
                    next_state = initial_render_state(
                        converted_payload,
                        initial_operation=None,
                        display_mode=default_display_mode,
                        preserved=preserved,
                    )
                    next_state["reload_request_id"] = int(shared_state.get("reload_request_id") or 0) + 1
                    next_state["structure_loaded"] = True
                    next_state["import_status"] = preserved_status
                    next_state["json_path"] = str(current_json_path)
                    shared_state.clear()
                    shared_state.update(next_state)
                    body = {
                        "ok": True,
                        "operations": session.operation_summary_items,
                        "atoms": atom_api_items(session.atoms),
                        "state": dict(shared_state),
                    }
                self.send_json(body)
            except Exception as exc:
                self.send_json_error(str(exc), status=400)

        def load_payload(
            self,
            json_path: Path,
            new_payload: dict,
            import_status: str,
            *,
            request_id: int = 0,
        ) -> None:
            started_at = time.monotonic()
            new_session = ViewerSession(json_path, new_payload, summarize_operations=False)
            debug_import_timing("session minimal summaries", started_at)
            with state_lock:
                if not self.load_request_is_current(request_id):
                    body = {"ok": False, "stale": True, "state": dict(shared_state)}
                    self.send_json(body)
                    return
                preserved = {
                    "speed": shared_state.get("speed", 1.0),
                    "projection_mode": shared_state.get("projection_mode", "perspective"),
                    "background_mode": shared_state.get("background_mode", "dark"),
                    "legend_visible": shared_state.get("legend_visible", False),
                    "cell_origin_mode": shared_state.get("cell_origin_mode", "center"),
                    "cell_setting_mode": "native",
                    "improper_mode": shared_state.get("improper_mode", "auto"),
                    # A range suitable for the previous structure can create a
                    # very large atom scene after loading a denser structure.
                    "display_mode": "source",
                    "reload_request_id": shared_state.get("reload_request_id"),
                    "pyvista_enabled": bool(shared_state.get("pyvista_enabled", False)),
                }
                session.replace_from(new_session)
                next_state = initial_render_state(
                    new_payload,
                    initial_operation=None,
                    display_mode="source",
                    preserved=preserved,
                )
                next_state["reload_request_id"] = int(shared_state.get("reload_request_id") or 0) + 1
                next_state["structure_loaded"] = True
                next_state["import_in_progress"] = False
                next_state["import_status"] = import_status_with_warnings(import_status, new_payload)
                next_state["json_path"] = str(json_path)
                next_state["summaries_ready"] = False
                replace_shared_state_for_load(shared_state, next_state, request_id)
                body = {
                    "ok": True,
                    "json_path": str(json_path),
                    "operations": session.operation_summary_items,
                    "atoms": atom_api_items(session.atoms),
                    "state": dict(shared_state),
                }
            self.compute_operation_summaries_async(new_session, request_id)
            self.send_json(body)

        def compute_operation_summaries_async(self, loaded_session: ViewerSession, request_id: int) -> None:
            def worker() -> None:
                try:
                    summaries = loaded_session.compute_operation_summaries()
                except Exception as exc:
                    logging.exception("Failed to compute operation summaries for %s", loaded_session.json_path)
                    with state_lock:
                        if (
                            self.load_request_is_current(request_id)
                            and session.json_path == loaded_session.json_path
                            and session.payload is loaded_session.payload
                        ):
                            shared_state["summaries_error"] = str(exc)
                            shared_state["summaries_ready"] = True
                    return
                with state_lock:
                    if not self.load_request_is_current(request_id):
                        return
                    if session.json_path != loaded_session.json_path:
                        return
                    if session.payload is not loaded_session.payload:
                        return
                    session.operation_summary_items = summaries
                    shared_state["summaries_ready"] = True

            threading.Thread(target=worker, daemon=True).start()

        def send_json(self, body: dict) -> None:
            self.send_bytes(
                json.dumps(body, default=to_jsonable).encode("utf-8"),
                content_type="application/json",
            )

        def send_javascript_file(self, path: Path, *, vendor: bool = False, transform=None) -> None:
            try:
                text = path.read_text(encoding="utf-8")
            except FileNotFoundError:
                message = (
                    f"The vendored Three.js file is missing: crystal_viewer/web/vendor/three/.../{path.name}"
                    if vendor
                    else f"JavaScript asset not found: {path.name}"
                )
                self.send_json_error(message, status=503 if vendor else 404)
                return
            if transform is not None:
                text = transform(text)
            self.send_bytes(text.encode("utf-8"), content_type="text/javascript; charset=utf-8")

        def send_stylesheet_file(self, path: Path) -> None:
            try:
                text = path.read_text(encoding="utf-8")
            except FileNotFoundError:
                self.send_json_error(f"Stylesheet asset not found: {path.name}", status=404)
                return
            self.send_bytes(text.encode("utf-8"), content_type="text/css; charset=utf-8")

        def send_json_error(self, message: str, *, status: int = 500) -> None:
            body = json.dumps({"ok": False, "error": message}).encode("utf-8")
            try:
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                return

        def send_bytes(self, body: bytes, *, content_type: str) -> None:
            try:
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                return

        def log_message(self, *args) -> None:
            return

    return Handler


def open_url(url: str) -> None:
    """Open URL in the default browser.  On WSL, webbrowser.open() often fails
    because there is no Linux browser; use cmd.exe instead to open Windows Edge."""
    import platform
    import subprocess
    if "microsoft" in platform.uname().release.lower():
        try:
            subprocess.run(
                ["cmd.exe", "/c", "start", "", url],
                check=False,
                capture_output=True,
            )
            return
        except FileNotFoundError:
            pass
    webbrowser.open(url)


class ReusableThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True
    block_on_close = False


def start_server(host: str, port: int, handler: type[BaseHTTPRequestHandler]) -> ThreadingHTTPServer:
    server = ReusableThreadingHTTPServer((host, port), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def worker_command(script_name: str) -> list[str]:
    """Start of the command that runs tools/<script_name> in a child process."""
    if FROZEN:
        # packaging/launcher.py dispatches this back to the script's main().
        return [sys.executable, WORKER_FLAG, script_name]
    return [sys.executable, str(Path(__file__).with_name(script_name))]


def export_analysis_to_json_worker(
    input_path: Path,
    *,
    mode: str,
    output_path: Path,
    tolerance_cart: float,
    indent: int,
    timeout_sec: float | None = None,
) -> Path:
    command = [
        *worker_command("export_analysis_json.py"),
        str(input_path),
        "--mode",
        mode,
        "--output",
        str(output_path),
        "--tolerance-cart",
        str(tolerance_cart),
        "--indent",
        str(indent),
    ]
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"analysis timed out after {timeout_sec:g} seconds for {input_path}"
        ) from exc
    if result.returncode != 0:
        message = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(message or f"export worker failed with exit code {result.returncode}")
    return output_path


def cached_export_json_path(
    input_path: Path,
    output_path: Path,
    *,
    mode: str,
) -> tuple[Path, dict] | None:
    if not output_path.exists():
        return None
    try:
        # The packaged build ships each example with its export and nothing edits
        # either afterwards, while the copies made on the way (build, archive,
        # unpacking) do not promise to keep the export the newer file.
        if not FROZEN and output_path.stat().st_mtime < input_path.stat().st_mtime:
            return None
        payload = json.loads(output_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if int(payload.get("schema_version") or 0) < EXPORT_SCHEMA_VERSION:
        return None

    render_data = payload.get("render_data") or {}
    metadata = render_data.get("metadata") or {}
    source_kind = normalize_source_kind(
        payload.get("source_kind", metadata.get("mode", ""))
    )
    if source_kind != mode:
        return None
    if not cached_source_matches(metadata.get("source_file"), input_path):
        return None
    return output_path, payload


def cached_source_matches(cached_source: object, input_path: Path) -> bool:
    if not cached_source:
        return False
    cached_path = Path(str(cached_source))
    if not cached_path.is_absolute():
        cached_path = PROJECT_ROOT / cached_path
    try:
        return cached_path.resolve() == input_path.resolve()
    except OSError:
        return str(cached_source) == str(input_path)


def export_analysis_to_json_worker_cached(
    input_path: Path,
    *,
    mode: str,
    output_path: Path,
    tolerance_cart: float,
    indent: int,
    timeout_sec: float | None = None,
) -> tuple[Path, dict | None]:
    cached = cached_export_json_path(input_path, output_path, mode=mode)
    if cached is not None:
        return cached
    exported_path = export_analysis_to_json_worker(
        input_path,
        mode=mode,
        output_path=output_path,
        tolerance_cart=tolerance_cart,
        indent=indent,
        timeout_sec=timeout_sec,
    )
    return exported_path, None


def export_cell_setting_json_worker(
    base_payload: dict,
    *,
    cell_setting: str,
    tolerance_cart: float,
    indent: int,
    timeout_sec: float | None = None,
    require_distinct: bool = False,
) -> dict:
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json",
        prefix="symmetry_view_cell_base_",
        dir=tempfile.gettempdir(),
        encoding="utf-8",
        delete=False,
    ) as base_file:
        json.dump(base_payload, base_file, ensure_ascii=False)
        base_path = Path(base_file.name)
    output_path = Path(tempfile.gettempdir()) / f"symmetry_view_cell_setting_{time.monotonic_ns()}.json"
    command = [
        *worker_command("export_cell_setting_json.py"),
        str(base_path),
        "--cell-setting",
        cell_setting,
        "--output",
        str(output_path),
        "--tolerance-cart",
        str(tolerance_cart),
        "--indent",
        str(indent),
    ]
    if require_distinct:
        command.append("--require-distinct")
    try:
        try:
            result = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"cell setting conversion timed out after {timeout_sec:g} seconds"
            ) from exc
        if result.returncode != 0:
            message = (result.stderr or result.stdout or "").strip()
            raise RuntimeError(message or f"cell setting worker failed with exit code {result.returncode}")
        return json.loads(output_path.read_text(encoding="utf-8"))
    finally:
        for path in (base_path, output_path):
            try:
                path.unlink()
            except FileNotFoundError:
                pass


def import_status_with_warnings(import_status: str, payload: dict) -> str:
    metadata = (payload.get("render_data") or {}).get("metadata") or {}
    warnings = [str(item) for item in metadata.get("warnings") or [] if str(item)]
    if not warnings:
        return import_status
    return f"{import_status}; warning: {' | '.join(warnings)}"


def resolve_viewer_json_path(
    input_path: Path,
    *,
    json_output: Path | None,
    json_dir: Path,
    tolerance_cart: float,
    indent: int,
) -> Path:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    suffix = input_path.suffix.lower()
    if suffix == ".json":
        if json_output is not None:
            raise ValueError("--json-output can only be used when the input is a CIF or XYZ file.")
        return input_path

    if suffix == ".cif":
        output_path = (
            json_output
            if json_output is not None
            else default_json_output_path(input_path, json_dir)
        )
        if json_output is None:
            cached = cached_export_json_path(input_path, output_path, mode=SOURCE_KIND_CRYSTAL)
            if cached is not None:
                cached_path, _ = cached
                print(f"Using cached JSON: {cached_path}", flush=True)
                return cached_path
        print(f"Analyzing CIF: {input_path}", flush=True)
        output_path = export_analysis_to_json(
            input_path,
            mode=SOURCE_KIND_CRYSTAL,
            output_path=output_path,
            tolerance_cart=tolerance_cart,
            indent=indent,
        )
        print(f"Wrote JSON: {output_path}", flush=True)
        return output_path

    if suffix == ".xyz":
        output_path = (
            json_output
            if json_output is not None
            else default_json_output_path(input_path, json_dir)
        )
        if json_output is None:
            cached = cached_export_json_path(input_path, output_path, mode=SOURCE_KIND_MOLECULE)
            if cached is not None:
                cached_path, _ = cached
                print(f"Using cached JSON: {cached_path}", flush=True)
                return cached_path
        print(f"Analyzing molecule: {input_path}", flush=True)
        output_path = export_analysis_to_json(
            input_path,
            mode=SOURCE_KIND_MOLECULE,
            output_path=output_path,
            tolerance_cart=tolerance_cart,
            indent=indent,
        )
        print(f"Wrote JSON: {output_path}", flush=True)
        return output_path

    raise ValueError(f"Unsupported input file type: {input_path.suffix}. Use .json, .cif, or .xyz.")


def default_starter_path() -> Path:
    for path in DEFAULT_STARTER_PATHS:
        if path.exists():
            return path
    raise FileNotFoundError(
        "No input_path was provided and no starter JSON was found. "
        "Provide a JSON/CIF path, or create exports/json/halite.json."
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Web symmetry viewer for exported JSON, CIF, or XYZ structures."
    )
    parser.add_argument(
        "input_path",
        type=Path,
        nargs="?",
        help="Exported JSON, or a CIF/XYZ file to analyze and export first.",
    )
    parser.add_argument("--operation", type=int, default=None, help="Initial operation index.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--port",
        type=int,
        default=0,
        help="HTTP port. Defaults to an automatically selected free port.",
    )
    parser.add_argument("--no-browser", action="store_true", help="Do not open the browser automatically.")
    parser.add_argument(
        "--mode",
        choices=("analysis", "puzzle"),
        default=None,
        help="Open directly in this mode. Default: show the Analysis/Puzzle selection screen.",
    )
    view_backend = parser.add_mutually_exclusive_group()
    view_backend.add_argument(
        "--with-pyvista",
        dest="pyvista_enabled",
        action="store_true",
        help="Also open the legacy PyVista comparison window and enable legacy GIF export.",
    )
    view_backend.add_argument(
        "--web-only",
        dest="pyvista_enabled",
        action="store_false",
        help="Run only the Web viewer (default; retained for command compatibility).",
    )
    parser.set_defaults(pyvista_enabled=False)
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Where to write the generated JSON when input_path is a CIF or XYZ file.",
    )
    parser.add_argument(
        "--json-dir",
        type=Path,
        default=DEFAULT_JSON_EXPORT_DIR,
        help="Directory for generated JSON when input_path is a CIF or XYZ file and --json-output is not set.",
    )
    parser.add_argument(
        "--import-json-dir",
        type=Path,
        default=DEFAULT_BROWSER_IMPORT_JSON_DIR,
        help="Directory for JSON generated by browser Open CIF/Open XYZ imports.",
    )
    parser.add_argument(
        "--tolerance-cart",
        type=float,
        default=1e-2,
        help="Atom mapping tolerance in Angstrom for generated CIF/XYZ exports.",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indentation for generated CIF/XYZ exports.",
    )
    parser.add_argument(
        "--analysis-timeout-sec",
        type=float,
        default=DEFAULT_ANALYSIS_TIMEOUT_SEC,
        help="Maximum seconds to wait for browser CIF/XYZ import analysis.",
    )
    parser.add_argument(
        "--expanded",
        action="store_true",
        help="Show quarter-cell periodic display clones. Slower, but useful for boundary checks.",
    )
    return parser


def main() -> int:
    args = build_argument_parser().parse_args()

    input_path = args.input_path if args.input_path is not None else write_empty_viewer_json()
    json_path = resolve_viewer_json_path(
        input_path,
        json_output=args.json_output,
        json_dir=args.json_dir,
        tolerance_cart=args.tolerance_cart,
        indent=args.indent,
    )

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    display_mode = "expanded_quarter" if args.expanded else "source"
    session = ViewerSession(json_path, payload)

    state_lock = threading.Lock()
    shared_state = initial_render_state(
        payload,
        initial_operation=args.operation,
        display_mode=display_mode,
    )
    shared_state["reset"] = False
    shared_state["reload_request_id"] = 0
    shared_state["json_path"] = str(json_path)
    shared_state["pyvista_enabled"] = args.pyvista_enabled

    handler = make_handler(
        session,
        shared_state,
        state_lock,
        json_dir=args.json_dir,
        import_json_dir=args.import_json_dir,
        tolerance_cart=args.tolerance_cart,
        indent=args.indent,
        analysis_timeout_sec=args.analysis_timeout_sec,
        default_display_mode=display_mode,
        initial_app_mode=args.mode or "select",
    )
    example_catalog()
    server = start_server(args.host, args.port, handler)
    url = f"http://{args.host}:{server.server_port}/"
    # Flushed so a launcher or test reading this through a pipe sees it now.
    print(f"Control panel: {url}", flush=True)
    if not args.no_browser:
        open_url(url)

    try:
        if args.pyvista_enabled:
            try:
                from crystal_viewer.viewer.pyvista_controller import BrowserControlledViewer
            except ImportError as exc:
                # PyVista is an optional extra, so --with-pyvista is the one flag
                # that can fail on an otherwise complete install.
                print(
                    f"--with-pyvista needs the optional PyVista renderer, which is not installed ({exc}).\n"
                    "Install it with:\n"
                    "  .venv/bin/python -m pip install -r requirements-pyvista.txt\n"
                    "The Web viewer at the URL above works without it.",
                    file=sys.stderr,
                )
                return 1

            app = BrowserControlledViewer(
                json_path,
                display_mode=display_mode,
                initial_operation=shared_state["operation_index"],
                scope=shared_state["scope"],
                shared_state=shared_state,
                state_lock=state_lock,
                viewer_session=session,
            )
            app.show()
        else:
            while True:
                time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
