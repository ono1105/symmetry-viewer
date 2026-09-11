from __future__ import annotations

import time
from pathlib import Path

import pyvista as pv

from crystal_viewer.viewer.animation import (
    animation_paths,
    update_animated_atoms,
)
from crystal_viewer.viewer.operation_lookup import (
    operation_by_index,
    selected_mapping,
)


def run_animation(
    plotter: pv.Plotter,
    render_data: dict,
    atom_mappings: dict | None,
    *,
    operation_index: int | None,
    animated_atoms: list[dict] | None,
    frame_count: int,
    fps: float,
    speed: float,
    output_path: Path | None,
    element_index: int | None,
    animation_scope: str,
    representative_atom: int | None,
    selected_atoms: tuple[int, ...],
    display_mode: str = "source",
) -> None:
    if animated_atoms is None:
        print("Animation skipped because atoms are hidden.")
        return
    operation = operation_by_index(render_data["operations"], operation_index)
    mapping = selected_mapping(atom_mappings, operation_index)
    if operation is None or mapping is None:
        print("Animation requires a valid --operation with atom mapping.")
        return
    available_atoms = {entry["source_atom"] for entry in mapping["entries"]}
    if representative_atom is not None and representative_atom not in available_atoms:
        print(f"Animation skipped because --representative-atom {representative_atom} is not in this mapping.")
        return
    missing_atoms = sorted(set(selected_atoms) - available_atoms)
    if missing_atoms:
        print(f"Animation skipped because selected atoms are not in this mapping: {missing_atoms}")
        return

    frames = max(frame_count, 2)
    playback_fps = effective_animation_fps(fps, speed)
    paths = animation_paths(
        render_data,
        operation,
        mapping,
        element_index=element_index,
        animation_scope=animation_scope,
        representative_atom=representative_atom,
        selected_atoms=selected_atoms,
        display_mode=display_mode,
    )
    if not paths:
        print("Animation skipped because no atom path could be built. Check --representative-atom.")
        return

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plotter.open_gif(str(output_path), fps=playback_fps)
        plotter.show(auto_close=False, interactive_update=True)
        for frame in range(frames):
            update_animated_atoms(animated_atoms, paths, frame / (frames - 1))
            plotter.write_frame()
        plotter.close()
        print(f"Wrote {output_path}")
        return

    plotter.show(auto_close=False, interactive_update=True)
    delay = 1.0 / playback_fps
    for frame in range(frames):
        update_animated_atoms(animated_atoms, paths, frame / (frames - 1))
        plotter.update()
        time.sleep(delay)
    plotter.show()


def effective_animation_fps(fps: float, speed: float) -> float:
    return max(float(fps), 1.0) * max(float(speed), 0.05)
