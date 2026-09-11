from __future__ import annotations


def selected_mapping(atom_mappings: dict | None, operation_index: int | None) -> dict | None:
    if atom_mappings is None or operation_index is None:
        return None
    mappings = atom_mappings.get("mappings", [])
    return {
        mapping["operation_index"]: mapping
        for mapping in mappings
    }.get(operation_index)


def filter_by_operation(elements: list[dict], operation_index: int | None) -> list[dict]:
    if operation_index is None:
        return elements
    return [
        element
        for element in elements
        if operation_index in element.get("operation_indices", [])
    ]


def selected_elements(elements: list[dict], operation_index: int | None, element_index: int | None) -> list[dict]:
    filtered = filter_by_operation(elements, operation_index)
    if element_index is None:
        return filtered
    if element_index < 0 or element_index >= len(filtered):
        return []
    return [filtered[element_index]]


def has_element_index(render_data: dict, operation_index: int, element_index: int) -> bool:
    return any(
        0 <= element_index < len(filter_by_operation(render_data[key], operation_index))
        for key in ("axes", "planes", "centers")
    )


def operation_by_index(operations: list[dict], operation_index: int) -> dict | None:
    return {
        operation["index"]: operation
        for operation in operations
    }.get(operation_index)
