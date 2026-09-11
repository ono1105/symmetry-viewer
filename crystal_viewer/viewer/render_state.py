from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from crystal_viewer.source_kinds import SOURCE_KIND_CRYSTAL, normalize_source_kind


STATE_UPDATE_KEYS = frozenset(
    {
        "operation_index",
        "playing",
        "reset",
        "selected_atoms",
        "element_colors",
        "atom_colors",
        "element_hidden",
        "atom_hidden",
        "speed",
        "projection_mode",
        "background_mode",
        "legend_visible",
        "cell_origin_mode",
        "cell_setting_mode",
        "improper_mode",
        "display_mode",
        "include_boundary_images",
        "animation_boundary_mode",
        "pause_at_breakpoints",
        "show_trajectories",
        "active_mode",
        "scope",
        "view_request_id",
        "reset_view_request_id",
        "view_center_request_id",
        "view_center_frac",
        "view_direction_request_id",
        "view_direction_frac",
        "view_plane_request_id",
        "view_plane_hkl",
        "camera_request_id",
        "camera_direction",
        "camera_angle",
        "custom_op_check_id",
        "clear_custom_check",
        "custom_op_animate",
        "reload_request_id",
        "import_status",
    }
)

# Preferences retained when the same structure is rebuilt in another cell
# basis. Keep this list as the single source of truth for that transition.
PRESERVED_STATE_KEYS = (
    "speed",
    "projection_mode",
    "background_mode",
    "legend_visible",
    "cell_origin_mode",
    "cell_setting_mode",
    "improper_mode",
    "display_mode",
    "include_boundary_images",
    "animation_boundary_mode",
    "pause_at_breakpoints",
    "show_trajectories",
)


def preserved_render_state(state: dict) -> dict:
    return {key: state[key] for key in PRESERVED_STATE_KEYS if key in state}


@dataclass(frozen=True)
class RenderStateSnapshot:
    operation_index: int
    playing: bool
    scope: str
    selected_atoms: tuple[int, ...]
    element_colors: dict[str, str]
    atom_colors: dict[str, str]
    element_hidden: dict[str, bool]
    atom_hidden: dict[str, bool]
    speed: float
    projection_mode: str
    background_mode: str
    legend_visible: bool
    cell_origin_mode: str
    cell_setting_mode: str
    improper_mode: str
    display_mode: str
    include_boundary_images: bool
    animation_boundary_mode: str
    pause_at_breakpoints: bool
    active_mode: str
    view_request_id: Any
    reset_view_request_id: Any
    view_center_request_id: Any
    view_center_frac: Any
    view_direction_request_id: Any
    view_direction_frac: Any
    view_plane_request_id: Any
    view_plane_hkl: Any
    camera_request_id: Any
    camera_direction: str
    camera_angle: float
    reload_request_id: Any
    custom_op_check_id: Any
    clear_custom_check: bool
    custom_op_result: Any
    custom_op_animate: Any
    reset: bool


def initial_render_state(
    payload: dict,
    *,
    initial_operation: int | None,
    display_mode: str,
    preserved: dict | None = None,
) -> dict:
    preserved = preserved or {}
    render_data = payload["render_data"]
    operations = render_data["operations"]
    first_operation = operations[0]["index"] if operations else 0
    selected_operation = (
        initial_operation
        if initial_operation is not None and operation_exists(operations, initial_operation)
        else first_operation
    )
    selected_atoms = [atom["index"] for atom in render_data["atoms"]]
    return {
        "operation_index": selected_operation,
        "playing": False,
        "reset": True,
        "source_kind": normalize_source_kind(
            payload.get(
                "source_kind",
                render_data.get("metadata", {}).get("mode", SOURCE_KIND_CRYSTAL),
            )
        ),
        "structure_loaded": bool(payload.get("structure_loaded", True)),
        "metadata": render_data.get("metadata", {}),
        "scope": "displayed",
        "selected_atoms": selected_atoms,
        "element_colors": {},
        "atom_colors": {},
        "element_hidden": {},
        "atom_hidden": {},
        "speed": float(preserved.get("speed", 1.0)),
        "projection_mode": preserved.get("projection_mode", "perspective"),
        # Analysis is a field-work surface, not a terminal: start with the
        # same soft light ground as the browser workspace.  A preserved
        # setting is still honoured for old sessions and the optional native
        # viewer.
        "background_mode": preserved.get("background_mode", "light"),
        "legend_visible": bool(preserved.get("legend_visible", True)),
        "cell_origin_mode": preserved.get("cell_origin_mode", "center"),
        "cell_setting_mode": preserved.get("cell_setting_mode", "native"),
        "improper_mode": preserved.get("improper_mode", "auto"),
        "display_mode": preserved.get("display_mode", display_mode),
        "include_boundary_images": bool(preserved.get("include_boundary_images", True)),
        "animation_boundary_mode": preserved.get("animation_boundary_mode", "continuous"),
        "pause_at_breakpoints": bool(preserved.get("pause_at_breakpoints", False)),
        "show_trajectories": bool(preserved.get("show_trajectories", False)),
        "active_mode": "standard",
        "status_message": "",
        "view_request_id": None,
        "reset_view_request_id": None,
        "view_center_request_id": None,
        "view_center_frac": None,
        "view_direction_request_id": None,
        "view_direction_frac": None,
        "view_plane_request_id": None,
        "view_plane_hkl": None,
        "camera_request_id": None,
        "camera_direction": "",
        "camera_angle": 90.0,
        "custom_op_check_id": None,
        "custom_op_result": None,
        "custom_op_animate": None,
        "reload_request_id": preserved.get("reload_request_id"),
        "import_status": preserved.get("import_status", ""),
        "import_in_progress": False,
        "json_path": str(preserved.get("json_path", "")),
        "summaries_ready": True,
        "pyvista_enabled": bool(preserved.get("pyvista_enabled", False)),
    }


def operation_exists(operations: list[dict], operation_index: int) -> bool:
    return any(operation["index"] == operation_index for operation in operations)


def apply_render_state_update(state: dict, update: dict) -> None:
    for key, value in update.items():
        if key in STATE_UPDATE_KEYS:
            state[key] = value


def pop_render_state_snapshot(state: dict) -> RenderStateSnapshot:
    return RenderStateSnapshot(
        operation_index=int(state["operation_index"]),
        playing=bool(state["playing"]),
        scope=str(state.get("scope", "representative")),
        selected_atoms=tuple(int(index) for index in state.get("selected_atoms", [])),
        element_colors=dict(state.get("element_colors", {})),
        atom_colors=dict(state.get("atom_colors", {})),
        element_hidden=dict(state.get("element_hidden", {})),
        atom_hidden=dict(state.get("atom_hidden", {})),
        speed=float(state.get("speed", 1.0)),
        projection_mode=str(state.get("projection_mode", "perspective")),
        background_mode=str(state.get("background_mode", "dark")),
        legend_visible=bool(state.get("legend_visible", True)),
        cell_origin_mode=str(state.get("cell_origin_mode", "center")),
        cell_setting_mode=str(state.get("cell_setting_mode", "native")),
        improper_mode=str(state.get("improper_mode", "auto")),
        display_mode=str(state.get("display_mode", "")),
        include_boundary_images=bool(state.get("include_boundary_images", False)),
        animation_boundary_mode=str(state.get("animation_boundary_mode", "continuous")),
        pause_at_breakpoints=bool(state.get("pause_at_breakpoints", False)),
        active_mode=str(state.get("active_mode", "standard")),
        view_request_id=state.get("view_request_id"),
        reset_view_request_id=state.get("reset_view_request_id"),
        view_center_request_id=state.get("view_center_request_id"),
        view_center_frac=state.get("view_center_frac"),
        view_direction_request_id=state.get("view_direction_request_id"),
        view_direction_frac=state.get("view_direction_frac"),
        view_plane_request_id=state.get("view_plane_request_id"),
        view_plane_hkl=state.get("view_plane_hkl"),
        camera_request_id=state.get("camera_request_id"),
        camera_direction=str(state.get("camera_direction", "")),
        camera_angle=float(state.get("camera_angle", 90.0)),
        reload_request_id=state.get("reload_request_id"),
        custom_op_check_id=state.get("custom_op_check_id"),
        clear_custom_check=bool(state.pop("clear_custom_check", False)),
        custom_op_result=state.get("custom_op_result"),
        custom_op_animate=state.get("custom_op_animate"),
        reset=bool(state.pop("reset", False)),
    )
