from __future__ import annotations

import datetime as dt
import os
import threading
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pyvista as pv

from crystal_viewer.geometry import normalize
from crystal_viewer.viewer.native_gui import NativePyVistaViewer
from crystal_viewer.viewer.animation_path import (
    maximum_travel_distance,
    normalized_animation_duration_seconds,
)
from crystal_viewer.viewer.atom_style import HIGHLIGHT_RADIUS_SCALE, atom_color, color_to_rgb
from crystal_viewer.viewer.display_atoms import (
    display_atom_instances,
    display_fractional_bounds,
    display_scene_center,
    display_scene_span,
)
from crystal_viewer.viewer.custom_animation import build_custom_animation_paths as build_shared_custom_paths
from crystal_viewer.viewer.operation_labels import (
    camera_up_vector,
    custom_focus_point_cart,
    is_pure_translation_operation,
    operation_focus_point_cart,
    operation_view_direction_cart,
    rotate_vector,
    visual_translation_direction_cart,
)
from crystal_viewer.viewer.render_state import pop_render_state_snapshot
from crystal_viewer.viewer.scene_rendering import (
    add_atom_legend,
    add_orientation_axes,
    add_unit_cell,
    setup_viewer_lighting,
    viewer_background_color,
)
from crystal_viewer.viewer.symmetry_elements import (
    add_symmetry_element_actors,
    add_symmetry_elements,
    display_symmetry_elements,
)


class BrowserControlledViewer(NativePyVistaViewer):
    def __init__(
        self,
        *args,
        shared_state: dict,
        state_lock: threading.Lock,
        viewer_session=None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.shared_state = shared_state
        self.state_lock = state_lock
        self.viewer_session = viewer_session
        self.timer_interval_ms = 100
        # Disable VTK depth peeling to prevent transparency-related warning loops
        # on WSL/Mesa drivers. Translucent objects render without correct sorting
        # but without recursive PyVista→logging→VTK→PyVista error floods.
        try:
            for renderer in self.plotter.renderers:
                renderer.UseDepthPeelingOff()
        except Exception:
            pass
        self.last_operation_index: int | None = None
        self.last_scope: str | None = None
        self.last_selected_atoms: tuple[int, ...] | None = None
        self.last_element_colors: dict[str, str] | None = None
        self.last_atom_colors: dict[str, str] | None = None
        self.last_element_hidden: dict[str, bool] | None = None
        self.last_atom_hidden: dict[str, bool] | None = None
        self.last_display_mode: str | None = self.display_mode
        self.include_boundary_images = bool(shared_state.get("include_boundary_images", False))
        self.last_include_boundary_images: bool | None = self.include_boundary_images
        self.last_cell_origin_mode: str | None = self.cell_origin_mode
        self.last_projection_mode: str | None = None
        self.last_background_mode: str | None = None
        self.last_legend_visible: bool | None = None
        self.last_improper_mode: str | None = None
        self.last_animation_boundary_mode: str | None = None
        self.animation_boundary_mode = str(shared_state.get("animation_boundary_mode", "continuous"))
        self.improper_mode = str(shared_state.get("improper_mode", "auto"))
        self.last_reload_request_id: int | None = shared_state.get("reload_request_id")
        self.last_active_mode: str | None = None
        self.last_view_request_id: int | None = None
        self.last_reset_view_request_id: int | None = None
        self.last_view_center_request_id: int | None = None
        self.last_view_direction_request_id: int | None = None
        self.last_view_plane_request_id: int | None = None
        self.last_camera_request_id: int | None = None
        self.last_custom_op_check_id: object = None
        self.custom_check_actors: list = []
        self.custom_sequence_element_actors: list = []
        self.last_custom_sequence_segment_index: int | None = None
        self.legend_actor = None
        self.custom_view_direction_cart: np.ndarray | None = None
        self.custom_focus_cart: np.ndarray | None = None
        self._path_length_cache_paths: dict[int, dict] | None = None
        self._maximum_path_length: float = 0.0
        self.last_custom_op_animate_id: object = None
        self.using_custom_paths: bool = False
        self.debug_timer = os.environ.get("CRYSTAL_VIEWER_DEBUG_TIMER") == "1"
        self.debug_tick_count = 0
        self.debug_render_count = 0
        self.debug_last_report = time.monotonic()

    def add_controls(self) -> None:
        self.plotter.add_key_event("space", self.toggle_play_from_keyboard)
        self.plotter.add_key_event("r", self.reset_from_keyboard)

    def on_timer(self, step: int) -> None:
        self._on_timer_inner(step)

    def _on_timer_inner(self, step: int) -> None:
        del step
        should_render = False
        should_update_status = False
        with self.state_lock:
            if self.shared_state.get("import_in_progress"):
                return
            snapshot = pop_render_state_snapshot(self.shared_state)

        reset = snapshot.reset
        self.speed = max(snapshot.speed, 0.1)
        operation_index = snapshot.operation_index
        requested_playing = snapshot.playing
        scope = snapshot.scope
        selected_atoms = snapshot.selected_atoms
        element_colors = snapshot.element_colors
        atom_colors = snapshot.atom_colors
        element_hidden = snapshot.element_hidden
        atom_hidden = snapshot.atom_hidden
        display_mode = snapshot.display_mode
        include_boundary_images = snapshot.include_boundary_images
        cell_origin_mode = "corner" if snapshot.cell_origin_mode == "corner" else "center"
        active_mode = snapshot.active_mode
        animation_boundary_mode = "wrap" if snapshot.animation_boundary_mode == "wrap" else "continuous"
        background_mode = "light" if snapshot.background_mode == "light" else "dark"
        legend_visible = bool(snapshot.legend_visible)

        if snapshot.reload_request_id is not None and snapshot.reload_request_id != self.last_reload_request_id:
            self.reload_from_session(
                display_mode=snapshot.display_mode,
                operation_index=snapshot.operation_index,
                scope=snapshot.scope,
                selected_atoms=snapshot.selected_atoms,
            )
            self.last_reload_request_id = snapshot.reload_request_id
            self.last_operation_index = snapshot.operation_index
            self.last_scope = snapshot.scope
            self.last_selected_atoms = snapshot.selected_atoms
            self.last_display_mode = snapshot.display_mode
            self.last_cell_origin_mode = cell_origin_mode
            self.last_active_mode = snapshot.active_mode
            self.last_background_mode = background_mode
            self.last_legend_visible = legend_visible
            self.last_custom_op_check_id = None
            should_render = True

        if cell_origin_mode != self.last_cell_origin_mode:
            self.reload_from_session(
                display_mode=snapshot.display_mode,
                operation_index=snapshot.operation_index,
                scope=snapshot.scope,
                selected_atoms=snapshot.selected_atoms,
            )
            self.last_cell_origin_mode = cell_origin_mode
            self.last_operation_index = snapshot.operation_index
            self.last_scope = snapshot.scope
            self.last_selected_atoms = snapshot.selected_atoms
            self.last_display_mode = snapshot.display_mode
            should_render = True

        if background_mode != self.last_background_mode:
            self.background_mode = background_mode
            self.plotter.set_background(viewer_background_color(background_mode))
            self.update_atom_legend(visible=legend_visible)
            self.last_background_mode = background_mode
            should_update_status = True
            should_render = True

        if legend_visible != self.last_legend_visible:
            self.update_atom_legend(visible=legend_visible)
            self.last_legend_visible = legend_visible
            should_render = True

        projection_mode = "orthographic" if snapshot.projection_mode == "orthographic" else "perspective"
        if projection_mode != self.last_projection_mode:
            self.set_projection_mode(projection_mode)
            self.last_projection_mode = projection_mode
            should_render = True

        if snapshot.improper_mode != self.last_improper_mode:
            self.improper_mode = snapshot.improper_mode
            self.last_improper_mode = snapshot.improper_mode
            self.clear_element_actor_cache()
            self.build_paths()
            reset = True
            should_render = True

        if animation_boundary_mode != self.last_animation_boundary_mode:
            self.animation_boundary_mode = animation_boundary_mode
            self.last_animation_boundary_mode = animation_boundary_mode
            reset = True
            should_render = True

        if element_colors != self.last_element_colors or atom_colors != self.last_atom_colors:
            self.element_colors = element_colors
            self.atom_colors = atom_colors
            if self.use_glyph_atom_rendering() != bool(self.atom_glyph_groups):
                current_display_mode = self.display_mode
                self.display_mode = ""
                self.rebuild_display_atoms(current_display_mode)
            else:
                self.apply_atom_colors()
            self.last_element_colors = dict(element_colors)
            self.last_atom_colors = dict(atom_colors)
            self.update_atom_legend(visible=legend_visible)
            should_render = True

        if element_hidden != self.last_element_hidden or atom_hidden != self.last_atom_hidden:
            self.element_hidden = element_hidden
            self.atom_hidden = atom_hidden
            self.clear_glyph_atom_cache()
            current_display_mode = self.display_mode
            self.display_mode = ""
            self.rebuild_display_atoms(current_display_mode)
            self.build_paths()
            self.update_start_markers()
            self.last_element_hidden = dict(element_hidden)
            self.last_atom_hidden = dict(atom_hidden)
            reset = True
            should_render = True
            should_update_status = True

        if active_mode != self.last_active_mode:
            if active_mode == "custom":
                self.hide_element_actors()
                self.hide_start_markers()
            else:
                self.clear_custom_check_actors()
                self.clear_custom_sequence_element_actors()
                if self.using_custom_paths:
                    self.using_custom_paths = False
                    self.build_paths()
                    reset = True
                self.show_element_actors(self.current_operation()["index"])
                self.update_start_markers()
            self.last_active_mode = active_mode
            should_render = True

        if operation_index != self.last_operation_index:
            self.set_operation_index(operation_index)
            self.last_operation_index = operation_index
            self.using_custom_paths = False
            if active_mode == "custom":
                self.hide_element_actors()
            reset = True
            should_render = True
            should_update_status = True

        if scope != self.last_scope or selected_atoms != self.last_selected_atoms:
            self.using_custom_paths = False
            self.scope = scope
            self.selected_atoms = selected_atoms
            self.build_paths()
            self.update_start_markers()
            self.last_scope = scope
            self.last_selected_atoms = selected_atoms
            reset = True
            should_render = True
            should_update_status = True

        if display_mode != self.last_display_mode:
            self.playing = False
            self.hide_element_actors()
            self.clear_element_actor_cache()
            old_display_mode = self.display_mode
            self.rebuild_display_atoms(display_mode)
            self._path_length_cache_paths = None
            if active_mode == "standard":
                self.build_paths()
            self.last_display_mode = display_mode
            self.recenter_camera_for_display_mode(old_display_mode, display_mode)
            if active_mode == "standard":
                self.show_element_actors(self.current_operation()["index"])
            reset = True
            should_render = True
            should_update_status = True

        if include_boundary_images != self.last_include_boundary_images:
            self.playing = False
            self.include_boundary_images = include_boundary_images
            self.clear_glyph_atom_cache()
            current_display_mode = self.display_mode
            self.display_mode = ""
            self.rebuild_display_atoms(current_display_mode)
            self._path_length_cache_paths = None
            if active_mode == "standard":
                self.build_paths()
            self.last_include_boundary_images = include_boundary_images
            reset = True
            should_render = True
            should_update_status = True

        if reset:
            self.frame_position = 0.0
            self.update_atoms(0.0)
            self.update_start_markers()
            if self.using_custom_paths:
                self.update_custom_sequence_element_actors(0.0)
            should_render = True

        if snapshot.view_request_id is not None and snapshot.view_request_id != self.last_view_request_id:
            self.view_along_current_operation()
            self.last_view_request_id = snapshot.view_request_id
            should_render = True

        if snapshot.reset_view_request_id is not None and snapshot.reset_view_request_id != self.last_reset_view_request_id:
            self.reset_view_center()
            self.last_reset_view_request_id = snapshot.reset_view_request_id
            should_render = True

        if snapshot.view_center_request_id is not None and snapshot.view_center_request_id != self.last_view_center_request_id:
            try:
                self.set_manual_view_center(snapshot.view_center_frac)
            except Exception as exc:
                self.set_status_message(f"view center failed: {exc}")
                should_update_status = True
            self.last_view_center_request_id = snapshot.view_center_request_id
            should_render = True

        if (
            snapshot.view_direction_request_id is not None
            and snapshot.view_direction_request_id != self.last_view_direction_request_id
        ):
            try:
                self.view_along_fractional_direction(snapshot.view_direction_frac)
            except Exception as exc:
                self.set_status_message(f"view direction failed: {exc}")
                should_update_status = True
            self.last_view_direction_request_id = snapshot.view_direction_request_id
            should_render = True

        if snapshot.view_plane_request_id is not None and snapshot.view_plane_request_id != self.last_view_plane_request_id:
            try:
                self.view_along_plane_normal(snapshot.view_plane_hkl)
            except Exception as exc:
                self.set_status_message(f"view plane failed: {exc}")
                should_update_status = True
            self.last_view_plane_request_id = snapshot.view_plane_request_id
            should_render = True

        if snapshot.camera_request_id is not None and snapshot.camera_request_id != self.last_camera_request_id:
            self.rotate_current_camera(snapshot.camera_direction, snapshot.camera_angle)
            self.last_camera_request_id = snapshot.camera_request_id
            should_render = True

        if snapshot.clear_custom_check:
            self.clear_custom_check_actors()
            self.clear_custom_sequence_element_actors()
            self.using_custom_paths = False
            self.build_paths()
            with self.state_lock:
                self.shared_state["custom_op_result"] = None
                self.shared_state["custom_op_animate"] = None
            reset = True
            should_render = True
        elif snapshot.custom_op_check_id is not None and snapshot.custom_op_check_id != self.last_custom_op_check_id:
            self.last_custom_op_check_id = snapshot.custom_op_check_id  # update first to prevent exception loops
            try:
                self.apply_custom_check(snapshot.custom_op_result)
            except Exception as exc:
                self.clear_custom_check_actors()
                self.set_status_message(f"check highlight failed: {exc}")
                should_update_status = True
            # New check discards any previous custom animation paths
            if self.using_custom_paths:
                self.using_custom_paths = False
                self.clear_custom_sequence_element_actors()
                self.build_paths()
                reset = True
            should_render = True

        if snapshot.custom_op_animate is not None:
            animate_id = snapshot.custom_op_animate.get("animate_id")
            if animate_id != self.last_custom_op_animate_id:
                self.paths = build_shared_custom_paths(
                    self.render_data,
                    self.atom_mappings,
                    snapshot.custom_op_animate,
                    display_mode=self.display_mode,
                    cell_origin_mode=self.cell_origin_mode,
                    improper_mode=self.improper_mode,
                )
                self.using_custom_paths = True
                self.last_custom_op_animate_id = animate_id
                self.last_custom_sequence_segment_index = None
                self.update_custom_sequence_element_actors(0.0)
                self.update_start_markers()
                reset = True
                should_render = True

        playing_before_update = self.playing
        if requested_playing and not self.playing and self.frame_position >= self.frame_count - 1:
            self.frame_position = 0.0
        if requested_playing != self.playing:
            should_update_status = True
        self.playing = requested_playing
        if self.playing and self.paths:
            duration = self.animation_duration_seconds()
            frame_step = (self.timer_interval_ms / 1000.0) / duration * max(self.frame_count - 1, 1)
            self.frame_position = min(self.frame_position + frame_step, self.frame_count - 1)
            s = self.frame_position / max(self.frame_count - 1, 1)
            self.update_atoms(s)
            if self.using_custom_paths:
                self.update_custom_sequence_element_actors(s)
            should_render = True
            if self.frame_position >= self.frame_count - 1:
                self.playing = False
                should_update_status = True

        if self.playing != playing_before_update:
            self.update_start_markers()
            should_render = True

        with self.state_lock:
            self.shared_state["playing"] = self.playing

        if should_update_status:
            self.update_status()
            should_render = True
        if should_render:
            self.plotter.render()
        if self.debug_timer:
            self.debug_tick_count += 1
            if should_render:
                self.debug_render_count += 1
            now = time.monotonic()
            if now - self.debug_last_report >= 5.0:
                print(
                    "viewer timer:",
                    f"ticks={self.debug_tick_count}",
                    f"renders={self.debug_render_count}",
                    f"playing={self.playing}",
                    flush=True,
                )
                self.debug_tick_count = 0
                self.debug_render_count = 0
                self.debug_last_report = now

    def set_operation_index(self, operation_index: int) -> None:
        for position, operation in enumerate(self.operations):
            if operation["index"] == operation_index:
                self.set_operation_position(position)
                return

    def show_element_actors(self, operation_index: int) -> None:
        cache_key = (operation_index, self.display_mode, self.improper_mode)
        if cache_key not in self.element_actor_cache:
            actors = add_symmetry_elements(
                self.plotter,
                self.render_data,
                self.atom_mappings,
                operation_index=operation_index,
                element_index=None,
                display_mode=self.display_mode,
                cell_origin_mode=self.cell_origin_mode,
                improper_mode=self.improper_mode,
            )
            self.element_actor_cache[cache_key] = actors
        self.element_actors = self.element_actor_cache[cache_key]
        for actor in self.element_actors:
            try:
                actor.SetVisibility(True)
            except Exception:
                pass

    def clear_element_actor_cache(self) -> None:
        for actors in self.element_actor_cache.values():
            for actor in actors:
                try:
                    self.plotter.remove_actor(actor)
                except Exception:
                    pass
        self.element_actor_cache = {}
        self.element_actors = []

    def reload_from_session(
        self,
        *,
        display_mode: str,
        operation_index: int,
        scope: str,
        selected_atoms: tuple[int, ...],
    ) -> None:
        if self.viewer_session is None:
            return
        debug_start = time.monotonic() if self.debug_timer else None
        self.playing = False
        self.frame_position = 0.0
        self.custom_check_actors = []
        self.custom_view_direction_cart = None
        self.custom_focus_cart = None
        self.last_custom_op_animate_id = None
        self.using_custom_paths = False

        self.json_path = self.viewer_session.json_path
        self.payload = self.viewer_session.payload
        self.render_data = self.viewer_session.render_data
        self.atom_mappings = self.payload.get("atom_mappings")
        self.operations = self.render_data["operations"]
        self.operation_position = 0
        self.scope = scope
        self.selected_atoms = selected_atoms
        self.display_mode = display_mode
        self.cell_origin_mode = "corner" if self.shared_state.get("cell_origin_mode") == "corner" else "center"

        self.element_actors = []
        self.element_actor_cache = {}
        self.start_marker_actors = []
        self.animated_atoms = []
        self.atom_actor_cache = {}
        self.atom_glyph_cache = {}
        self.atom_glyph_groups = []
        self.sphere_mesh_cache = {}
        self.paths = {}
        self.status_actor = None
        self.legend_actor = None

        self.plotter.clear()
        if self.debug_timer:
            print(f"[viewer] reload clear {time.monotonic() - debug_start:.3f}s", flush=True)
        self.background_mode = "light" if self.shared_state.get("background_mode") == "light" else "dark"
        setup_viewer_lighting(self.plotter, background_mode=self.background_mode)
        self.display_mode = ""
        self.rebuild_display_atoms(display_mode)
        if self.debug_timer:
            print(f"[viewer] reload atoms {time.monotonic() - debug_start:.3f}s", flush=True)
        if self.render_data.get("unit_cell"):
            add_unit_cell(self.plotter, self.render_data["unit_cell"], cell_origin_mode=self.cell_origin_mode)
        add_orientation_axes(self.plotter, unit_cell=self.render_data.get("unit_cell"))
        self.update_atom_legend(visible=bool(self.shared_state.get("legend_visible", False)))
        if self.debug_timer:
            print(f"[viewer] reload axes {time.monotonic() - debug_start:.3f}s", flush=True)
        self.set_operation_index(operation_index)
        if self.debug_timer:
            print(f"[viewer] reload operation {time.monotonic() - debug_start:.3f}s", flush=True)
        self.plotter.reset_camera()
        self.set_projection_mode(str(self.shared_state.get("projection_mode", "perspective")))
        self.update_status()
        if self.debug_timer:
            print(f"[viewer] reload done {time.monotonic() - debug_start:.3f}s", flush=True)

    def atom_color(self, atom: dict) -> str:
        return atom_color(
            atom,
            element_colors=getattr(self, "element_colors", {}),
            atom_colors=getattr(self, "atom_colors", {}),
        )

    def display_atom_visible(self, display_item: dict) -> bool:
        atom = display_item["atom"]
        if getattr(self, "element_hidden", {}).get(str(atom.get("element", ""))):
            return False
        if getattr(self, "atom_hidden", {}).get(str(atom.get("index", ""))):
            return False
        return True

    def apply_atom_colors(self) -> None:
        if self.use_glyph_atom_rendering():
            for group in self.atom_glyph_groups:
                actor = group.get("actor")
                if actor is None:
                    continue
                color = self.atom_color({"element": group.get("element", ""), "atomic_number": 0})
                try:
                    actor.GetProperty().SetColor(*color_to_rgb(color))
                except Exception:
                    pass
            return
        for item in self.animated_atoms:
            actor = item.get("actor")
            if actor is None:
                continue
            color = self.atom_color(item["atom"])
            try:
                actor.GetProperty().SetColor(*color_to_rgb(color))
            except Exception:
                pass

    def update_atom_legend(self, *, visible: bool | None = None) -> None:
        if self.legend_actor is not None:
            try:
                self.plotter.remove_actor(self.legend_actor)
            except Exception:
                pass
        if visible is None:
            visible = bool(getattr(self, "last_legend_visible", False))
        if not visible:
            self.legend_actor = None
            return
        self.legend_actor = add_atom_legend(
            self.plotter,
            self.render_data,
            element_colors=getattr(self, "element_colors", {}),
            background_mode=getattr(self, "background_mode", "dark"),
        )

    def use_glyph_atom_rendering(self) -> bool:
        return not bool(getattr(self, "atom_colors", {}))

    def set_projection_mode(self, projection_mode: str) -> None:
        if projection_mode == "orthographic":
            self.plotter.camera.ParallelProjectionOn()
        else:
            self.plotter.camera.ParallelProjectionOff()
        self.plotter.reset_camera_clipping_range()

    def toggle_play_from_keyboard(self) -> None:
        with self.state_lock:
            self.shared_state["playing"] = not bool(self.shared_state["playing"])

    def reset_from_keyboard(self) -> None:
        with self.state_lock:
            self.shared_state["playing"] = False
            self.shared_state["reset"] = True

    def view_along_current_operation(self) -> None:
        basis = self.operation_camera_basis()
        if basis is None:
            return
        center, direction, up, distance = basis
        self.set_camera_view(center, direction, up, distance)

    def view_along_fractional_direction(self, direction_frac: list[float] | tuple[float, ...] | None) -> None:
        if direction_frac is None or len(direction_frac) != 3:
            raise ValueError("enter three direction indices")
        unit_cell = self.render_data.get("unit_cell")
        if unit_cell is None:
            raise ValueError("direction indices require a crystal unit cell")
        direction_frac_array = np.asarray(direction_frac, dtype=float)
        if not np.all(np.isfinite(direction_frac_array)):
            raise ValueError("direction indices must contain finite numbers")
        lattice = np.asarray(unit_cell["lattice"], dtype=float)
        direction = direction_frac_array @ lattice
        if np.linalg.norm(direction) < 1e-10:
            raise ValueError("direction must not be zero")
        direction = normalize(direction)
        center = self.display_center()
        distance = self.camera_distance_for_view(center, direction)
        up = camera_up_vector(direction)
        self.set_camera_view(center, direction, up, distance)

    def view_along_plane_normal(self, hkl: list[float] | tuple[float, ...] | None) -> None:
        if hkl is None or len(hkl) != 3:
            raise ValueError("enter three Miller indices")
        unit_cell = self.render_data.get("unit_cell")
        if unit_cell is None:
            raise ValueError("Miller indices require a crystal unit cell")
        hkl_array = np.asarray(hkl, dtype=float)
        if not np.all(np.isfinite(hkl_array)):
            raise ValueError("Miller indices must contain finite numbers")
        lattice = np.asarray(unit_cell["lattice"], dtype=float)
        normal = np.linalg.inv(lattice) @ hkl_array
        if np.linalg.norm(normal) < 1e-10:
            raise ValueError("plane normal must not be zero")
        direction = normalize(normal)
        center = self.display_center()
        distance = self.camera_distance_for_view(center, direction)
        up = camera_up_vector(direction)
        self.set_camera_view(center, direction, up, distance)

    def operation_camera_basis(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, float] | None:
        if self.custom_view_direction_cart is not None:
            direction = self.custom_view_direction_cart
            if np.linalg.norm(direction) < 1e-10:
                return None
            direction = normalize(direction)
            center = self.custom_focus_cart if self.custom_focus_cart is not None else self.display_center()
            distance = self.camera_distance_for_view(center, direction)
            up = camera_up_vector(direction)
            return center, direction, up, distance

        operation = self.current_operation()
        axes, planes, centers = display_symmetry_elements(
            self.render_data,
            self.atom_mappings,
            operation["index"],
            element_index=None,
            improper_mode=self.improper_mode,
        )
        direction = None
        if is_pure_translation_operation(operation):
            direction = visual_translation_direction_cart(self.render_data, operation, self.atom_mappings)
        if direction is None:
            direction = operation_view_direction_cart(self.render_data, operation, axes, planes, centers)
        if direction is None:
            return None
        direction = normalize(direction)
        center = operation_focus_point_cart(
            self.render_data,
            operation,
            axes,
            planes,
            centers,
            self.display_mode,
            self.cell_origin_mode,
        )
        distance = self.camera_distance_for_view(center, direction)
        up = camera_up_vector(direction)
        return center, direction, up, distance

    def set_camera_view(self, center: np.ndarray, direction: np.ndarray, up: np.ndarray, distance: float) -> None:
        scale = self.camera_parallel_scale_for_view(center, direction)
        self.plotter.camera_position = [
            tuple(center + direction * distance),
            tuple(center),
            tuple(up),
        ]
        if self.plotter.camera.GetParallelProjection():
            self.plotter.camera.SetParallelScale(scale)
        self.plotter.reset_camera_clipping_range()

    def camera_distance_for_view(self, center: np.ndarray, direction: np.ndarray) -> float:
        radius, depth = self.camera_view_extent(center, direction)
        angle = np.deg2rad(float(self.plotter.camera.GetViewAngle()))
        angle = float(np.clip(angle, np.deg2rad(12.0), np.deg2rad(70.0)))
        perspective_distance = radius / max(np.tan(angle * 0.5), 1e-6)
        depth_margin = min(depth * 0.2, radius * 1.5)
        return max(perspective_distance * 1.15 + depth_margin, radius * 2.2, 2.0)

    def camera_parallel_scale_for_view(self, center: np.ndarray, direction: np.ndarray) -> float:
        radius, _ = self.camera_view_extent(center, direction)
        return max(radius * 1.15, 0.8)

    def camera_view_extent(self, center: np.ndarray, direction: np.ndarray) -> tuple[float, float]:
        points = self.camera_reference_points()
        if points.size == 0:
            span = display_scene_span(self.render_data, self.display_mode, self.cell_origin_mode)
            radius = max(span * 0.35, 1.0)
            return radius, span

        center = np.asarray(center, dtype=float)
        direction = normalize(np.asarray(direction, dtype=float))
        offsets = points - center
        depths = offsets @ direction
        projected = offsets - np.outer(depths, direction)
        projected_radii = np.linalg.norm(projected, axis=1)
        full_radius = float(np.max(projected_radii)) if len(projected_radii) else 1.0
        robust_radius = float(np.percentile(projected_radii, 85)) if len(projected_radii) > 4 else full_radius
        scene_span = display_scene_span(self.render_data, self.display_mode, self.cell_origin_mode)

        source_kind = str(self.render_data.get("metadata", {}).get("mode", "crystal"))
        if source_kind == "molecule":
            radius = full_radius
        else:
            local_cap = max(scene_span * 0.22, robust_radius * 1.35, 1.0)
            radius = min(full_radius, local_cap)

        radius = max(radius, min(scene_span * 0.18, 2.0), 0.8)
        depth = float(np.max(depths) - np.min(depths)) if len(depths) else scene_span
        return radius, max(depth, 1.0)

    def camera_reference_points(self) -> np.ndarray:
        points = []
        try:
            points.extend(
                item["cart"]
                for item in display_atom_instances(
                    self.render_data,
                    display_mode=self.display_mode,
                    cell_origin_mode=self.cell_origin_mode,
                    include_boundary_images=self.include_boundary_images,
                )
            )
        except Exception:
            points.extend(np.asarray(atom["cart"], dtype=float) for atom in self.render_data.get("atoms", []))

        unit_cell = self.render_data.get("unit_cell")
        if unit_cell is not None:
            lattice = np.asarray(unit_cell["lattice"], dtype=float)
            lower, upper = display_fractional_bounds(self.display_mode, self.cell_origin_mode)
            corners = np.asarray(
                [
                    [x, y, z]
                    for x in (lower, upper)
                    for y in (lower, upper)
                    for z in (lower, upper)
                ],
                dtype=float,
            )
            points.extend(corners @ lattice)

        if not points:
            return np.empty((0, 3), dtype=float)
        return np.asarray(points, dtype=float)

    def rotate_current_camera(self, direction: str, angle_deg: float) -> None:
        angle = float(np.clip(angle_deg, 0.0, 180.0))
        if angle <= 1e-8:
            return
        position = np.asarray(self.plotter.camera.GetPosition(), dtype=float)
        focal_point = np.asarray(self.plotter.camera.GetFocalPoint(), dtype=float)
        up = normalize(np.asarray(self.plotter.camera.GetViewUp(), dtype=float))
        radius_vector = position - focal_point
        if np.linalg.norm(radius_vector) < 1e-10:
            return
        view_direction = normalize(focal_point - position)
        screen_right = normalize(np.cross(view_direction, up))
        if direction == "right":
            axis = up
            signed_angle = angle
        elif direction == "left":
            axis = up
            signed_angle = -angle
        elif direction == "up":
            axis = screen_right
            signed_angle = -angle
        elif direction == "down":
            axis = screen_right
            signed_angle = angle
        elif direction == "roll-left":
            axis = view_direction
            signed_angle = -angle
        elif direction == "roll-right":
            axis = view_direction
            signed_angle = angle
        else:
            return

        rotated_up = rotate_vector(up, axis, np.deg2rad(signed_angle))
        if direction.startswith("roll-"):
            rotated_radius = radius_vector
        else:
            rotated_radius = rotate_vector(radius_vector, axis, np.deg2rad(signed_angle))
        self.plotter.camera_position = [
            tuple(focal_point + rotated_radius),
            tuple(focal_point),
            tuple(normalize(rotated_up)),
        ]
        self.plotter.reset_camera_clipping_range()

    def display_center(self) -> np.ndarray:
        return display_scene_center(self.render_data, self.display_mode, self.cell_origin_mode)

    def reset_view_center(self) -> None:
        self.set_camera_center(self.display_center())

    def set_manual_view_center(self, center_frac: list[float] | tuple[float, ...] | None) -> None:
        if center_frac is None or len(center_frac) != 3:
            coordinate_label = "fractional" if self.render_data.get("unit_cell") else "Cartesian Å"
            raise ValueError(f"enter three {coordinate_label} coordinates")
        center_frac_array = np.asarray(center_frac, dtype=float)
        if not np.all(np.isfinite(center_frac_array)):
            raise ValueError("view center must contain finite numbers")
        unit_cell = self.render_data.get("unit_cell")
        if unit_cell is None:
            new_center = center_frac_array
        else:
            lattice = np.asarray(unit_cell["lattice"], dtype=float)
            new_center = center_frac_array @ lattice
        self.set_camera_center(new_center)

    def set_camera_center(self, new_center: np.ndarray) -> None:
        position = np.asarray(self.plotter.camera.GetPosition(), dtype=float)
        focal_point = np.asarray(self.plotter.camera.GetFocalPoint(), dtype=float)
        up = normalize(np.asarray(self.plotter.camera.GetViewUp(), dtype=float))
        shift = new_center - focal_point
        self.plotter.camera_position = [
            tuple(position + shift),
            tuple(new_center),
            tuple(up),
        ]
        self.plotter.reset_camera_clipping_range()

    def recenter_camera_for_display_mode(self, old_display_mode: str, new_display_mode: str) -> None:
        old_center = display_scene_center(self.render_data, old_display_mode, self.cell_origin_mode)
        new_center = display_scene_center(self.render_data, new_display_mode, self.cell_origin_mode)
        shift = new_center - old_center
        if np.linalg.norm(shift) < 1e-10:
            return
        position = np.asarray(self.plotter.camera.GetPosition(), dtype=float)
        focal_point = np.asarray(self.plotter.camera.GetFocalPoint(), dtype=float)
        up = normalize(np.asarray(self.plotter.camera.GetViewUp(), dtype=float))
        self.plotter.camera_position = [
            tuple(position + shift),
            tuple(focal_point + shift),
            tuple(up),
        ]
        self.plotter.reset_camera_clipping_range()

    def set_status_message(self, status: str) -> None:
        """Report a PyVista-side failure in the browser's status panel.

        Named gif_status until 2026-08-16, which was wrong in both directions:
        nothing about GIF writing sets it, and the camera/highlight errors it
        does carry looked like GIF noise.
        """
        with self.state_lock:
            self.shared_state["status_message"] = status

    def animation_duration_seconds(self) -> float:
        # Cached on path identity: this runs on every frame draw, and the paths
        # only change when the operation or the display options do.
        if self.paths is not self._path_length_cache_paths:
            self._maximum_path_length = maximum_travel_distance(
                self.render_data,
                self.paths,
                display_mode=self.display_mode,
                cell_origin_mode=self.cell_origin_mode,
                include_boundary_images=self.include_boundary_images,
            )
            self._path_length_cache_paths = self.paths
        speed_multiplier = max(float(self.speed), 0.1)
        base_duration = normalized_animation_duration_seconds(
            self._maximum_path_length,
            display_scene_span(self.render_data, self.display_mode, self.cell_origin_mode),
        )
        return base_duration / speed_multiplier

    def update_custom_sequence_element_actors(self, s: float) -> None:
        metadata = self.current_sequence_segment_elements(s)
        if metadata is None:
            self.clear_custom_sequence_element_actors()
            return
        segment_index, elements = metadata
        if segment_index == self.last_custom_sequence_segment_index:
            return
        self.clear_custom_sequence_element_actors()
        self.custom_sequence_element_actors = add_symmetry_element_actors(
            self.plotter,
            self.render_data,
            elements.get("axes", []),
            elements.get("planes", []),
            elements.get("centers", []),
            display_mode=self.display_mode,
            cell_origin_mode=self.cell_origin_mode,
        )
        self.last_custom_sequence_segment_index = segment_index

    def current_sequence_segment_elements(self, s: float) -> tuple[int, dict] | None:
        for path in self.paths.values():
            if path.get("type") != "sequential":
                continue
            elements = path.get("segment_elements") or []
            if not elements:
                return None
            segment_count = len(elements)
            weights = path.get("segment_weights")
            if not isinstance(weights, list) or len(weights) != segment_count:
                weights = [1.0 / segment_count] * segment_count
            total = float(sum(max(float(weight), 0.0) for weight in weights))
            progress = float(np.clip(s, 0.0, 1.0))
            cumulative = 0.0
            segment_index = segment_count - 1
            if total > 1e-12:
                for index, weight in enumerate(weights):
                    cumulative += max(float(weight), 0.0) / total
                    if progress <= cumulative + 1e-12:
                        segment_index = index
                        break
            else:
                segment_index = min(int(progress * segment_count), segment_count - 1)
            return segment_index, elements[segment_index]
        return None

    def clear_custom_sequence_element_actors(self) -> None:
        for actor in self.custom_sequence_element_actors:
            try:
                self.plotter.remove_actor(actor)
            except Exception:
                pass
        self.custom_sequence_element_actors = []
        self.last_custom_sequence_segment_index = None

    def apply_custom_check(self, result: dict | None) -> None:
        self.clear_custom_check_actors()
        if not result or result.get("error"):
            return

        elements = result.get("elements") or {}
        axes = elements.get("axes") or []
        planes = elements.get("planes") or []
        centers = elements.get("centers") or []
        if axes or planes or centers:
            self.custom_check_actors.extend(
                add_symmetry_element_actors(
                    self.plotter,
                    self.render_data,
                    axes,
                    planes,
                    centers,
                    display_mode=self.display_mode,
                    cell_origin_mode=self.cell_origin_mode,
                )
            )
        direction = result.get("view_direction_cart")
        if direction is not None:
            self.custom_view_direction_cart = np.asarray(direction, dtype=float)
        self.custom_focus_cart = custom_focus_point_cart(
            result,
            self.render_data,
            self.display_mode,
            self.cell_origin_mode,
        )

        if not result.get("unmapped"):
            return
        unmapped_set = {item["source"] for item in result["unmapped"]}
        for item in self.animated_atoms:
            atom = item["atom"]
            if atom["index"] not in unmapped_set:
                continue
            center = np.asarray(atom["cart"], dtype=float) + np.asarray(item["display_shift_cart"], dtype=float)
            radius = self.atom_radius(atom["atomic_number"]) * HIGHLIGHT_RADIUS_SCALE
            sphere = pv.Sphere(radius=radius, center=center, theta_resolution=24, phi_resolution=14)
            actor = self.plotter.add_mesh(
                sphere,
                color="#ff8a80",
                style="wireframe",
                line_width=1,
                opacity=0.55,
                lighting=False,
            )
            self.custom_check_actors.append(actor)

    def clear_custom_check_actors(self) -> None:
        for actor in self.custom_check_actors:
            try:
                self.plotter.remove_actor(actor)
            except Exception:
                pass
        self.custom_check_actors = []
        self.custom_view_direction_cart = None
        self.custom_focus_cart = None
