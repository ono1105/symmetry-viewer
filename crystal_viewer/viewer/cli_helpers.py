from __future__ import annotations

from pathlib import Path

import pyvista as pv

from crystal_viewer.viewer.operation_lookup import (
    filter_by_operation,
    operation_by_index,
    selected_mapping,
)


def parse_selected_atoms(values: list[str] | None) -> tuple[int, ...]:
    if not values:
        return ()
    selected = []
    for value in values:
        for item in str(value).split(","):
            item = item.strip()
            if item:
                selected.append(int(item))
    return tuple(dict.fromkeys(selected))


def print_operations(render_data: dict, atom_mappings: dict | None) -> None:
    mapping_by_index = {}
    if atom_mappings is not None:
        mapping_by_index = {
            mapping["operation_index"]: mapping
            for mapping in atom_mappings.get("mappings", [])
        }

    print("=== Operations ===")
    for operation in render_data["operations"]:
        mapping = mapping_by_index.get(operation["index"])
        if mapping is None:
            status = "mapping=missing"
        else:
            status = f"mapping={'ok' if mapping['complete'] else 'incomplete'} max_dist={mapping['max_distance']:.3e}"
        print(
            f"{operation['index']:3d}: {operation['label']} "
            f"kind={operation['kind']} order={operation['order']} "
            f"angle={operation.get('angle_deg')} {status}"
        )


def add_title(
    plotter: pv.Plotter,
    render_data: dict,
    json_path: Path,
    operation_index: int | None,
) -> None:
    metadata = render_data["metadata"]
    text = f"{metadata['formula']}  |  {metadata['symmetry_label']}  |  {metadata['mode']}"
    if operation_index is not None:
        operation = operation_by_index(render_data["operations"], operation_index)
        suffix = operation["label"] if operation else f"operation {operation_index}"
        text = f"{text}  |  {suffix}"
    text = f"{text}\n{json_path}"
    plotter.add_text(text, position="upper_left", font_size=10, color="#111827")


def print_elements(render_data: dict, operation_index: int | None) -> None:
    if operation_index is None:
        print("Use --operation with --list-elements.")
        return

    print(f"=== Symmetry Elements for operation {operation_index} ===")
    for _kind, key in (("axis", "axes"), ("plane", "planes"), ("center", "centers")):
        elements = filter_by_operation(render_data[key], operation_index)
        if not elements:
            continue
        print(f"{key}:")
        for index, element in enumerate(elements):
            label = element.get("label", "?")
            point = element.get("point_cart")
            direction = element.get("direction_cart") or element.get("normal_cart")
            print(f"  {index}: label={label} point={point} direction/normal={direction}")


def print_mapping(atom_mappings: dict | None, operation_index: int | None) -> None:
    mapping = selected_mapping(atom_mappings, operation_index)
    if mapping is None:
        print("No atom mapping found. Use --operation with --show-mapping.")
        return

    print("=== Atom Mapping ===")
    print(
        f"Operation {mapping['operation_index']}: {mapping['operation_kind']}, "
        f"complete={mapping['complete']}, max_distance={mapping['max_distance']:.3e}"
    )
    print("atom_to_atom:", mapping["atom_to_atom"])
    for entry in mapping["entries"]:
        print(
            f"  {entry['source_atom']} -> {entry['target_atom']} "
            f"dist={entry['distance']:.3e} target={entry['transformed_cart']}"
        )
