"""Judging for the axis-order puzzles.

Two problem types share this logic (docs/PUZZLE_SPEC.md):
  * ``rotation``  — "how many-fold is this rotation axis?" (Cn)
  * ``improper``  — "how many-fold is this rotoreflection axis?" (Sn)

Renderer-independent: consumes ``render_data`` only. For a given axis the
correct answer is the set of orders present on it (read straight from the
operations that live on that axis), restricted to the offered options; an
infinite rotation axis (C-inf on a linear molecule) accepts every option.
Collinear axis elements are merged, and symmetry-equivalent axes (e.g. the
x/y/z C4 axes of an octahedron) collapse into one representative via their
orbit under the group operations.
"""

from __future__ import annotations

import numpy as np

from crystal_viewer.game.operation_identify import _schoenflies_index

ROTATION_ORDER_OPTIONS: tuple[int, ...] = (2, 3, 4, 6)

_DIRECTION_TOL = 1e-4

def _rotation_order(operation: dict) -> int | None:
    order = operation.get("order")
    return int(order) if order is not None else None


def _improper_order(operation: dict) -> int | None:
    """Schoenflies n of an S_n operation, from its rotation angle.

    The stored matrix order (and symbol) uses the matrix period, which for odd
    n is 2n (an S3 matrix has period 6). The Schoenflies index is 360/theta,
    where trace(S) = 2*cos(theta) - 1 for a rotoreflection.
    """
    matrix = operation.get("matrix_cart")
    if matrix is None:
        return _rotation_order(operation)
    m = np.asarray(matrix, dtype=float)
    if np.linalg.det(m) > 0:  # a proper rotation slipped in
        return None
    cos_theta = np.clip((np.trace(m) + 1.0) / 2.0, -1.0, 1.0)
    theta = float(np.degrees(np.arccos(cos_theta)))
    if theta < 1e-6:  # pure reflection (S1 = sigma)
        return None
    # Rounding alone names every angle something; S10^3 (108 deg) came back as
    # 3. Only keep an index that reproduces the angle. See _schoenflies_index.
    return _schoenflies_index(theta)


# Which operation kinds count for each problem type, and how to read the order.
_AXIS_KINDS = {
    "rotation": {"prefixes": ("rotation_",), "infinite": "rotation_infinite", "order_fn": _rotation_order},
    "improper": {"prefixes": ("improper_", "rotoinversion"), "infinite": None, "order_fn": _improper_order},
}


def canonical_direction(vector) -> np.ndarray | None:
    """Unit direction with a canonical sign (first significant component > 0)."""
    vec = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(vec))
    if norm < 1e-9:
        return None
    vec = vec / norm
    for component in vec:
        if abs(component) > 1e-6:
            if component < 0:
                vec = -vec
            break
    return vec


def _geometric_axes(render_data: dict, spec: dict) -> list[dict]:
    """Merge collinear axis elements carrying the requested operation kinds."""
    operations = {int(op["index"]): op for op in render_data.get("operations", [])}
    groups: dict[tuple, dict] = {}
    for axis in render_data.get("axes", []):
        direction = canonical_direction(axis.get("direction_cart"))
        if direction is None:
            continue
        orders: set[int] = set()
        # order -> a representative operation index that realises exactly that
        # fold (a rotation by 360/order about this axis). The puzzle reveal plays
        # this operation's animation, reusing the analysis-mode animation path.
        order_operations: dict[int, int] = {}
        infinite = False
        for index in axis.get("operation_indices", []):
            operation = operations.get(int(index))
            if operation is None:
                continue
            kind = str(operation.get("kind", ""))
            if spec["infinite"] and kind == spec["infinite"]:
                infinite = True
            elif any(kind.startswith(prefix) for prefix in spec["prefixes"]):
                order = spec["order_fn"](operation)
                if order is not None:
                    orders.add(int(order))
                    order_operations.setdefault(int(order), int(index))
        if not orders and not infinite:
            continue
        # Molecules share the centre, so the canonical direction identifies the
        # line. (Crystals with parallel axes at different points would need the
        # point in the key too; out of scope for the molecule-first puzzle.)
        key = tuple(np.round(direction, 4))
        group = groups.get(key)
        if group is None:
            group = {
                "direction": direction,
                "point": np.asarray(axis.get("point_cart", [0.0, 0.0, 0.0]), dtype=float),
                "orders": set(),
                "order_operations": {},
                "infinite": False,
            }
            groups[key] = group
        group["orders"] |= orders
        for order, index in order_operations.items():
            group["order_operations"].setdefault(order, index)
        group["infinite"] = group["infinite"] or infinite
    return list(groups.values())


def _equivalence_classes(axes: list[dict], render_data: dict) -> list[list[int]]:
    """Group geometric axes by their orbit under all group operations."""
    matrices = [
        np.asarray(op["matrix_cart"], dtype=float)
        for op in render_data.get("operations", [])
        if op.get("matrix_cart") is not None
    ]
    directions = [axis["direction"] for axis in axes]
    parent = list(range(len(axes)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        parent[find(a)] = find(b)

    def match(direction: np.ndarray) -> int | None:
        canonical = canonical_direction(direction)
        if canonical is None:
            return None
        for index, other in enumerate(directions):
            if np.linalg.norm(canonical - other) < _DIRECTION_TOL:
                return index
        return None

    for i, axis in enumerate(axes):
        for matrix in matrices:
            target = match(matrix @ axis["direction"])
            if target is not None:
                union(i, target)

    classes: dict[int, list[int]] = {}
    for i in range(len(axes)):
        classes.setdefault(find(i), []).append(i)
    return list(classes.values())


def axis_questions(
    render_data: dict,
    kind: str = "rotation",
    *,
    options: tuple[int, ...] = ROTATION_ORDER_OPTIONS,
) -> list[dict]:
    """One question per symmetry-inequivalent axis of the given problem type."""
    spec = _AXIS_KINDS[kind]
    axes = _geometric_axes(render_data, spec)
    questions = []
    for members in _equivalence_classes(axes, render_data):
        axis = axes[members[0]]
        if axis["infinite"]:
            correct = list(options)
        else:
            if any(order not in options for order in axis["orders"]):
                continue
            correct = sorted(order for order in options if order in axis["orders"])
            if not correct:
                continue
        reveal_operations = {
            order: axis["order_operations"][order]
            for order in correct
            if order in axis["order_operations"]
        }
        questions.append(
            {
                "direction_cart": axis["direction"].tolist(),
                "point_cart": axis["point"].tolist(),
                "correct_orders": correct,
                "reveal_operations": reveal_operations,
                "equivalent_count": len(members),
                "infinite": bool(axis["infinite"]),
            }
        )
    questions.sort(key=lambda q: (-max(q["correct_orders"]), q["correct_orders"]))
    return questions


def rotation_axis_questions(
    render_data: dict,
    *,
    options: tuple[int, ...] = ROTATION_ORDER_OPTIONS,
) -> list[dict]:
    return axis_questions(render_data, "rotation", options=options)


def public_questions(
    render_data: dict,
    kind: str = "rotation",
    *,
    options: tuple[int, ...] = ROTATION_ORDER_OPTIONS,
) -> list[dict]:
    """Questions for the client with the correct answer withheld.

    ``group`` is an opaque id shared by axis-questions with the same answer
    (same highest fold, or both infinite) -- it does not reveal what that
    answer is, only that two questions share one. The client uses it to pick
    an answer-type bucket before an axis within it, the same balancing
    operation_identify/composition/atom_mapping already do, so a structure
    with several inequivalent low-fold axes and only one high-fold axis (e.g.
    benzene: two independent C2 axis classes but a single C2/C3/C6 principal
    axis) does not flood the picker with the common answer.
    """
    questions = axis_questions(render_data, kind, options=options)
    group_ids: dict[tuple, int] = {}
    result = []
    for index, question in enumerate(questions):
        key = ("inf",) if question["infinite"] else tuple(question["correct_orders"])
        group = group_ids.setdefault(key, len(group_ids))
        result.append(
            {
                "id": index,
                "direction_cart": question["direction_cart"],
                "point_cart": question["point_cart"],
                "options": list(options),
                "equivalent_count": question["equivalent_count"],
                "infinite": question["infinite"],
                "group": group,
            }
        )
    return result


_INFINITE_ANSWER = "inf"


def _axis_answer(question: dict):
    """The single expected answer: the highest fold, or ``inf`` for a C-inf axis."""
    if question["infinite"]:
        return _INFINITE_ANSWER
    return max(question["correct_orders"])


def _normalize_selection(selected):
    """Client answer to an int order or the infinite sentinel."""
    if isinstance(selected, str):
        text = selected.strip().lower()
        if text in {"inf", "infinite", "∞"}:
            return _INFINITE_ANSWER
        return int(text)
    return int(selected)


def check_answer(
    render_data: dict,
    question_id: int,
    selected_order,
    kind: str = "rotation",
    *,
    options: tuple[int, ...] = ROTATION_ORDER_OPTIONS,
) -> dict | None:
    """Judge one answer (the axis' highest fold) and reveal it. ``None`` if unknown."""
    questions = axis_questions(render_data, kind, options=options)
    if not 0 <= question_id < len(questions):
        return None
    question = questions[question_id]
    answer = _axis_answer(question)
    selected = _normalize_selection(selected_order)
    # Operation whose animation demonstrates the answered fold (none for C-inf);
    # handed out only once the answer has been submitted.
    reveal_operation = None if answer == _INFINITE_ANSWER else question["reveal_operations"].get(answer)
    return {
        "correct": selected == answer,
        "answer": answer,
        "selected": selected,
        "reveal_operation": reveal_operation,
    }
