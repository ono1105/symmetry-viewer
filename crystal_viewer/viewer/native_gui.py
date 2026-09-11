from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import logging
import numpy as np
import pyvista as pv

# PyVista/VTK can emit repetitive warnings through the Python logger.
# Suppress them here so the console stays readable.
logging.getLogger("pyvista").setLevel(logging.ERROR)
logging.getLogger("vtkmodules").setLevel(logging.ERROR)

if os.environ.get("CRYSTAL_VIEWER_VTK_WARNINGS") != "1":
    try:
        pv.set_error_output_file(os.devnull)
    except Exception:
        pass
    try:
        import vtk

        vtk.vtkObject.GlobalWarningDisplayOff()
    except Exception:
        pass

from crystal_viewer.viewer.animation import (
    animation_paths,
    operation_speed_multiplier,
    path_applies_to_display_item,
    update_animated_atoms,
)
from crystal_viewer.viewer.atom_instances import display_instance_key, element_instance_batches
from crystal_viewer.viewer.atom_style import (
    ATOM_MESH_STYLE,
    HIGHLIGHT_RADIUS_SCALE,
    atom_color,
    color_to_rgb,
    display_atom_radius,
)
from crystal_viewer.viewer.cli_helpers import (
    parse_selected_atoms,
    print_operations,
)
from crystal_viewer.viewer.display_atoms import display_atom_instances
from crystal_viewer.viewer.glyph_atoms import build_element_glyph_mesh, update_element_glyph_instance
from crystal_viewer.viewer.operation_lookup import selected_mapping
from crystal_viewer.viewer.scene_rendering import (
    add_atom_legend,
    add_orientation_axes,
    add_unit_cell,
    setup_viewer_lighting,
    viewer_text_color,
)
from crystal_viewer.viewer.symmetry_elements import add_symmetry_elements


class NativePyVistaViewer:
    def __init__(
        self,
        json_path: Path,
        *,
        scope: str = "all",
        selected_atoms: tuple[int, ...] = (),
        display_mode: str = "source",
        initial_operation: int | None = None,
    ) -> None:
        self.json_path = json_path
        self.payload = json.loads(json_path.read_text(encoding="utf-8"))
        self.render_data = self.payload["render_data"]
        self.atom_mappings = self.payload.get("atom_mappings")
        self.operations = self.render_data["operations"]
        self.operation_position = 0
        self.scope = scope
        self.selected_atoms = selected_atoms
        self.display_mode = display_mode
        self.initial_operation = initial_operation
        self.element_actors: list = []
        self.element_actor_cache: dict[int, list] = {}
        self.start_marker_actors: list = []
        self.animated_atoms: list[dict] = []
        self.atom_actor_cache: dict[tuple[int, tuple[int, int, int]], dict] = {}
        self.atom_glyph_cache: dict[str, list[dict]] = {}
        self.atom_glyph_groups: list[dict] = []
        self.sphere_mesh_cache: dict[tuple[int, float], pv.PolyData] = {}
        self.paths: dict[int, dict] = {}
        self.playing = False
        self.frame_position = 0.0
        self.frame_count = 96
        self.speed = 1.0
        self.timer_interval_ms = 33
        self.status_actor = None
        self.legend_actor = None
        self.background_mode = "dark"
        self.cell_origin_mode = "center"
        self.include_boundary_images = False

        self.plotter = pv.Plotter()
        setup_viewer_lighting(self.plotter, background_mode=self.background_mode)

    def show(self) -> None:
        self.rebuild_display_atoms(self.display_mode)
        if self.render_data.get("unit_cell"):
            add_unit_cell(self.plotter, self.render_data["unit_cell"], cell_origin_mode=self.cell_origin_mode)
        add_orientation_axes(self.plotter, unit_cell=self.render_data.get("unit_cell"))
        self.legend_actor = add_atom_legend(
            self.plotter,
            self.render_data,
            background_mode=self.background_mode,
        )
        self.plotter.reset_camera()

        self.add_controls()
        self.set_operation_position(self.initial_operation_position())
        self.plotter.add_timer_event(
            max_steps=1000000,
            duration=self.timer_interval_ms,
            callback=self.on_timer,
        )
        self.plotter.show()

    def initial_operation_position(self) -> int:
        if self.initial_operation is None:
            return 0
        for position, operation in enumerate(self.operations):
            if operation["index"] == self.initial_operation:
                return position
        return 0

    def add_controls(self) -> None:
        max_index = max(len(self.operations) - 1, 0)
        self.plotter.add_slider_widget(
            self.on_operation_slider,
            rng=[0, max_index],
            value=0,
            title="operation",
            pointa=(0.03, 0.08),
            pointb=(0.45, 0.08),
            style="modern",
        )
        self.plotter.add_slider_widget(
            self.on_speed_slider,
            rng=[0.1, 4.0],
            value=1.0,
            title="speed",
            pointa=(0.55, 0.08),
            pointb=(0.95, 0.08),
            style="modern",
        )
        self.plotter.add_key_event("space", self.toggle_play)
        self.plotter.add_key_event("n", self.next_operation)
        self.plotter.add_key_event("p", self.previous_operation)
        self.plotter.add_key_event("r", self.reset_animation)
        self.plotter.add_key_event("1", lambda: self.set_scope("all"))
        self.plotter.add_key_event("2", lambda: self.set_scope("representative"))
        self.plotter.add_key_event("3", lambda: self.set_scope("selected"))

    def on_operation_slider(self, value: float) -> None:
        self.set_operation_position(int(round(value)))

    def on_speed_slider(self, value: float) -> None:
        self.speed = max(float(value), 0.1)
        self.update_status()

    def set_operation_position(self, position: int) -> None:
        position = max(0, min(int(position), len(self.operations) - 1))
        if position == self.operation_position and self.paths:
            return
        self.operation_position = position
        self.frame_position = 0.0
        self.hide_element_actors()
        self.hide_start_markers()
        operation = self.current_operation()
        self.show_element_actors(operation["index"])
        self.build_paths()
        self.update_start_markers()
        self.update_atoms(0.0)
        self.update_status()
        self.plotter.render()

    def current_operation(self) -> dict:
        return self.operations[self.operation_position]

    def hide_element_actors(self) -> None:
        for actor in self.element_actors:
            try:
                actor.SetVisibility(False)
            except Exception:
                pass
        self.element_actors = []

    def show_element_actors(self, operation_index: int) -> None:
        if operation_index not in self.element_actor_cache:
            self.element_actor_cache[operation_index] = add_symmetry_elements(
                self.plotter,
                self.render_data,
                self.atom_mappings,
                operation_index=operation_index,
                element_index=None,
                display_mode=self.display_mode,
                cell_origin_mode=self.cell_origin_mode,
            )
        self.element_actors = self.element_actor_cache[operation_index]
        for actor in self.element_actors:
            try:
                actor.SetVisibility(True)
            except Exception:
                pass

    def hide_start_markers(self) -> None:
        for actor in self.start_marker_actors:
            try:
                actor.SetVisibility(False)
            except Exception:
                pass

    def create_start_markers(self) -> None:
        for item in self.animated_atoms:
            if item.get("marker_actor") is not None:
                continue
            if self.use_glyph_atom_rendering() and not item.get("is_primary_image", True):
                continue
            atom = item["atom"]
            center = np.asarray(atom["cart"], dtype=float) + item["display_shift_cart"]
            radius = self.atom_radius(atom["atomic_number"]) * HIGHLIGHT_RADIUS_SCALE
            marker = self.sphere_mesh(atom["atomic_number"], radius)
            actor = self.plotter.add_mesh(
                marker,
                color="#f7dc6f",
                opacity=0.28,
                lighting=False,
                smooth_shading=True,
            )
            actor.SetPosition(*center)
            actor.SetVisibility(False)
            self.start_marker_actors.append(actor)
            item["marker_actor"] = actor

    def update_start_markers(self) -> None:
        self.start_marker_actors = []
        for item in self.animated_atoms:
            actor = item.get("marker_actor")
            if actor is None:
                continue
            self.start_marker_actors.append(actor)
            path = self.paths.get(item["atom"]["index"])
            animation_started = self.playing or self.frame_position > 0.0
            visible = animation_started and path is not None and path_applies_to_display_item(path, item)
            try:
                actor.SetVisibility(visible)
            except Exception:
                pass

    def rebuild_display_atoms(self, display_mode: str) -> None:
        if display_mode == self.display_mode and self.animated_atoms:
            return
        self.display_mode = display_mode
        self.hide_start_markers()
        self.hide_glyph_atom_groups()
        for cached in self.atom_actor_cache.values():
            for actor_name in ("actor", "marker_actor"):
                actor = cached.get(actor_name)
                if actor is not None:
                    try:
                        actor.SetVisibility(False)
                    except Exception:
                        pass

        if self.use_glyph_atom_rendering():
            visible_items = self.ensure_glyph_display_atoms(self.display_mode)
        else:
            visible_items = []
            for display_item in display_atom_instances(
                self.render_data,
                display_mode=self.display_mode,
                cell_origin_mode=self.cell_origin_mode,
                include_boundary_images=self.include_boundary_images,
            ):
                if not self.display_atom_visible(display_item):
                    continue
                item = self.ensure_display_atom(display_item)
                actor = item.get("actor")
                if actor is not None:
                    try:
                        actor.SetVisibility(True)
                    except Exception:
                        pass
                visible_items.append(item)

        self.animated_atoms = visible_items
        self.create_start_markers()
        self.update_start_markers()
        self.update_atoms(self.frame_position / max(self.frame_count - 1, 1))

    def ensure_display_atom(self, display_item: dict) -> dict:
        atom = display_item["atom"]
        shift_frac = display_item.get("display_shift_frac", np.zeros(3))
        key = display_instance_key(display_item)
        cached = self.atom_actor_cache.get(key)
        center = np.asarray(display_item["cart"], dtype=float)
        if cached is not None:
            cached["display_shift_frac"] = np.asarray(shift_frac, dtype=float)
            cached["display_shift_cart"] = np.asarray(display_item["display_shift_cart"], dtype=float)
            cached["is_primary_image"] = bool(display_item.get("is_primary_image", True))
            actor = cached.get("actor")
            if actor is not None:
                actor.SetPosition(*center)
            return cached

        radius = self.atom_radius(atom["atomic_number"])
        color = self.atom_color(atom)
        mesh = self.sphere_mesh(atom["atomic_number"], radius)
        actor = self.plotter.add_mesh(mesh, color=color, **ATOM_MESH_STYLE)
        actor.SetPosition(*center)
        cached = {
            "atom": atom,
            "display_shift_frac": np.asarray(shift_frac, dtype=float),
            "display_shift_cart": np.asarray(display_item["display_shift_cart"], dtype=float),
            "is_primary_image": bool(display_item.get("is_primary_image", True)),
            "base_points": mesh.points.copy(),
            "mesh": mesh,
            "actor": actor,
            "marker_actor": None,
        }
        self.atom_actor_cache[key] = cached
        return cached

    def use_glyph_atom_rendering(self) -> bool:
        return False

    def hide_glyph_atom_groups(self) -> None:
        for group in self.atom_glyph_groups:
            actor = group.get("actor")
            if actor is not None:
                try:
                    actor.SetVisibility(False)
                except Exception:
                    pass
        self.atom_glyph_groups = []

    def clear_glyph_atom_cache(self) -> None:
        seen = set()
        for groups in self.atom_glyph_cache.values():
            for group in groups:
                actor = group.get("actor")
                if actor is None or id(actor) in seen:
                    continue
                seen.add(id(actor))
                try:
                    self.plotter.remove_actor(actor)
                except Exception:
                    pass
        self.atom_glyph_cache = {}
        self.atom_glyph_groups = []

    def ensure_glyph_display_atoms(self, display_mode: str) -> list[dict]:
        groups = self.atom_glyph_cache.get(display_mode)
        if groups is None:
            groups = []
            for batch in element_instance_batches(
                self.render_data,
                display_mode=display_mode,
                cell_origin_mode=self.cell_origin_mode,
                include_boundary_images=self.include_boundary_images,
                item_filter=self.display_atom_visible,
            ):
                glyph = build_element_glyph_mesh(batch, self.render_data)
                color = self.atom_color({"element": batch.element, "atomic_number": batch.atomic_number})
                actor = self.plotter.add_mesh(glyph.mesh, color=color, **ATOM_MESH_STYLE)
                items = []
                for position, display_item in enumerate(batch.items):
                    item = {
                        "atom": display_item["atom"],
                        "display_shift_frac": np.asarray(display_item["display_shift_frac"], dtype=float),
                        "display_shift_cart": np.asarray(display_item["display_shift_cart"], dtype=float),
                        "is_primary_image": bool(display_item.get("is_primary_image", True)),
                        "current_cart": np.asarray(display_item["cart"], dtype=float),
                        "position_dirty": False,
                        "actor": None,
                        "marker_actor": None,
                        "glyph_element": batch.element,
                        "glyph_position": position,
                    }
                    items.append(item)
                groups.append(
                    {
                        "element": batch.element,
                        "glyph": glyph,
                        "actor": actor,
                        "items": items,
                    }
                )
            self.atom_glyph_cache[display_mode] = groups
        self.atom_glyph_groups = groups
        visible_items = []
        for group in groups:
            for item in group["items"]:
                item["current_cart"] = np.asarray(item["atom"]["cart"], dtype=float) + item["display_shift_cart"]
                item["position_dirty"] = True
            actor = group.get("actor")
            if actor is not None:
                try:
                    color = self.atom_color({"element": group["element"], "atomic_number": 0})
                    actor.GetProperty().SetColor(*color_to_rgb(color))
                    actor.SetVisibility(True)
                except Exception:
                    pass
            visible_items.extend(group["items"])
        return visible_items

    def display_atom_visible(self, display_item: dict) -> bool:
        return True

    def sphere_mesh(self, atomic_number: int, radius: float) -> pv.PolyData:
        key = (int(atomic_number), round(float(radius), 6))
        mesh = self.sphere_mesh_cache.get(key)
        if mesh is None:
            mesh = pv.Sphere(
                radius=radius,
                center=(0.0, 0.0, 0.0),
                theta_resolution=16,
                phi_resolution=10,
            )
            self.sphere_mesh_cache[key] = mesh
        return mesh

    def atom_radius(self, atomic_number: int) -> float:
        return display_atom_radius(
            {"atomic_number": atomic_number},
            self.render_data,
        )

    def atom_color(self, atom: dict) -> str:
        return atom_color(atom)

    def build_paths(self) -> None:
        operation = self.current_operation()
        mapping = selected_mapping(self.atom_mappings, operation["index"])
        if mapping is None:
            self.paths = {}
            return
        selected_atoms = self.selected_atoms if self.scope in ("selected", "selected_displayed") else ()
        animation_scope = self.scope
        unit_cell_only = self.scope in ("selected", "unit_cell", "representative")
        if self.scope == "displayed":
            animation_scope = "selected"
            selected_atoms = tuple(atom["index"] for atom in self.render_data["atoms"])
        elif self.scope == "unit_cell":
            animation_scope = "selected"
            selected_atoms = tuple(atom["index"] for atom in self.render_data["atoms"])
        elif self.scope == "selected_displayed":
            animation_scope = "selected"
        representative_atom = selected_atoms[0] if self.scope in ("selected", "selected_displayed") and selected_atoms else None
        self.paths = animation_paths(
            self.render_data,
            operation,
            mapping,
            element_index=None,
            animation_scope=animation_scope,
            representative_atom=representative_atom,
            selected_atoms=selected_atoms,
            improper_mode=getattr(self, "improper_mode", "auto"),
            display_mode=self.display_mode,
            cell_origin_mode=self.cell_origin_mode,
        )
        if unit_cell_only:
            for path in self.paths.values():
                path["unit_cell_only"] = True

    def toggle_play(self) -> None:
        self.playing = not self.playing
        self.update_status()

    def next_operation(self) -> None:
        self.set_operation_position(self.operation_position + 1)

    def previous_operation(self) -> None:
        self.set_operation_position(self.operation_position - 1)

    def reset_animation(self) -> None:
        self.playing = False
        self.frame_position = 0.0
        self.update_atoms(0.0)
        self.update_status()
        self.plotter.render()

    def set_scope(self, scope: str) -> None:
        if scope == "selected" and not self.selected_atoms:
            self.scope = "representative"
        else:
            self.scope = scope
        self.frame_position = 0.0
        self.build_paths()
        self.update_start_markers()
        self.update_atoms(0.0)
        self.update_status()
        self.plotter.render()

    def on_timer(self, step: int) -> None:
        del step
        if not self.playing or not self.paths:
            return
        multiplier = operation_speed_multiplier(self.current_operation())
        self.frame_position = min(self.frame_position + self.speed * multiplier, self.frame_count - 1)
        self.update_atoms(self.frame_position / max(self.frame_count - 1, 1))
        if self.frame_position >= self.frame_count - 1:
            self.playing = False
            self.update_status()
        self.plotter.render()

    def update_atoms(self, s: float) -> None:
        update_animated_atoms(
            self.animated_atoms,
            self.paths,
            s,
            render_data=self.render_data,
            boundary_mode=getattr(self, "animation_boundary_mode", "continuous"),
            cell_origin_mode=self.cell_origin_mode,
        )
        self.update_glyph_atom_groups()

    def update_glyph_atom_groups(self) -> None:
        for group in self.atom_glyph_groups:
            changed = False
            for item in group["items"]:
                if not item.get("position_dirty"):
                    continue
                update_element_glyph_instance(group["glyph"], item["glyph_position"], item["current_cart"])
                item["position_dirty"] = False
                changed = True
            if changed:
                group["glyph"].mesh.Modified()

    def update_status(self) -> None:
        operation = self.current_operation()
        metadata = self.render_data["metadata"]
        selected = ",".join(str(index) for index in self.selected_atoms) if self.selected_atoms else "-"
        text = (
            f"{metadata['formula']} | {metadata['symmetry_label']}\n"
            f"op {operation['index']}: {operation['label']} | {operation['kind']}\n"
            f"state={'play' if self.playing else 'pause'} speed={self.speed:.1f} "
            f"scope={self.scope} display={self.display_mode} selected={selected}\n"
            "keys: space play/stop, n/p operation, r reset, 1 all, 2 representative, 3 selected"
        )
        if self.status_actor is not None:
            try:
                self.plotter.remove_actor(self.status_actor)
            except Exception:
                pass
        self.status_actor = self.plotter.add_text(
            text,
            position="upper_left",
            font_size=9,
            color=viewer_text_color(getattr(self, "background_mode", "dark")),
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Native PyVista GUI for exported symmetry JSON.")
    parser.add_argument("json_path", type=Path)
    parser.add_argument("--operation", type=int, default=None, help="Initial operation index.")
    parser.add_argument("--list-operations", action="store_true", help="Print operations and exit.")
    parser.add_argument("--list-atoms", action="store_true", help="Print atoms and exit.")
    parser.add_argument(
        "--scope",
        choices=("all", "representative", "selected", "unit_cell", "displayed"),
        default="all",
        help="Initial animation scope.",
    )
    parser.add_argument(
        "--selected-atoms",
        nargs="+",
        default=None,
        help="Source atom indices for selected scope. Accepts spaces or commas.",
    )
    parser.add_argument(
        "--expanded",
        action="store_true",
        help="Show quarter-cell periodic display clones. Slower, but useful for boundary checks.",
    )
    args = parser.parse_args()

    payload = json.loads(args.json_path.read_text(encoding="utf-8"))
    render_data = payload["render_data"]
    if args.list_operations:
        print_operations(render_data, payload.get("atom_mappings"))
        return 0
    if args.list_atoms:
        print_atoms(render_data)
        return 0

    selected_atoms = parse_selected_atoms(args.selected_atoms)
    scope = args.scope
    if scope == "selected" and not selected_atoms:
        scope = "representative"

    display_mode = "expanded_quarter" if args.expanded else "source"
    app = NativePyVistaViewer(
        args.json_path,
        scope=scope,
        selected_atoms=selected_atoms,
        display_mode=display_mode,
        initial_operation=args.operation,
    )
    app.show()
    return 0


def print_atoms(render_data: dict) -> None:
    print("=== Atoms ===")
    for atom in render_data["atoms"]:
        frac = atom.get("frac")
        frac_text = f" frac={frac}" if frac is not None else ""
        asym = atom.get("asymmetric_index")
        asym_text = f" asym={asym}" if asym is not None else ""
        print(
            f"{atom['index']:4d}: {atom['element']} "
            f"cart={atom['cart']}{frac_text}{asym_text}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
