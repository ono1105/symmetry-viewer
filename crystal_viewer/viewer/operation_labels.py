from __future__ import annotations

import re
from fractions import Fraction
from functools import lru_cache
from math import gcd

import numpy as np

from crystal_viewer.geometry import integer_index_vector, normalize, signed_rotation_angle_from_matrix
from crystal_viewer.itc_tables import itc_coordinate_summaries
from crystal_viewer.viewer.animation import animation_paths
from crystal_viewer.viewer.animation_path import effective_rotation_axis
from crystal_viewer.viewer.display_atoms import display_point_cart, display_scene_center
from crystal_viewer.viewer.glide_geometry import centered_fractional_vector, glide_translation_frac
from crystal_viewer.viewer.operation_lookup import selected_elements, selected_mapping
from crystal_viewer.viewer.symmetry_elements import visual_improper_elements


def operation_summaries(
    render_data: dict,
    atom_mappings: dict | None,
) -> list[dict]:
    summaries = []
    itc_coordinates = itc_coordinate_summaries(render_data)
    for operation in render_data["operations"]:
        summary_operation = dict(operation)
        visual_translation = visual_translation_direction_cart(render_data, operation, atom_mappings)
        if visual_translation is not None:
            summary_operation["_display_translation_cart"] = visual_translation.tolist()
        axes, planes, centers = operation_summary_elements(
            render_data,
            summary_operation,
        )
        display_symbol = display_operation_symbol(render_data, summary_operation, axes, planes)
        summaries.append(
            {
                "index": operation["index"],
                "label": operation["label"],
                "symbol": operation.get("symbol") or operation["label"],
                "display_symbol": display_symbol,
                "kind": operation["kind"],
                "order": operation.get("order"),
                "angle_deg": operation.get("angle_deg"),
                "notation_order": operation_notation_order(operation),
                "element_summary": operation_element_summary(
                    render_data,
                    summary_operation,
                    axes,
                    planes,
                    centers,
                    display_symbol=display_symbol,
                ),
                "itc_like_summary": operation_itc_like_summary(
                    render_data,
                    summary_operation,
                    axes,
                    planes,
                    centers,
                    display_symbol=display_symbol,
                ),
                "itc_coordinate_summary": itc_coordinates.get(operation["index"], ""),
                "element_sort_key": operation_element_sort_key(render_data, summary_operation, axes, planes, centers),
                "direction_sort_key": operation_direction_sort_key(render_data, summary_operation, axes, planes, centers),
                "direction_label": operation_direction_label(render_data, summary_operation, axes, planes, centers),
                "direction_filter_label": operation_direction_filter_label(render_data, summary_operation, axes, planes, centers),
                "matrix_frac": operation.get("matrix_frac"),
                "translation_frac": operation.get("translation_frac"),
                "matrix_cart": operation.get("matrix_cart"),
                "translation_cart": operation.get("translation_cart"),
                "fixed_atom_indices": operation_fixed_atom_indices(render_data, operation),
            }
        )
    return summaries


def minimal_operation_summaries(render_data: dict) -> list[dict]:
    summaries = []
    for operation in render_data["operations"]:
        summaries.append(
            {
                "index": operation["index"],
                "label": operation["label"],
                "symbol": operation.get("symbol") or operation["label"],
                "display_symbol": operation.get("symbol") or operation["label"],
                "kind": operation["kind"],
                "order": operation.get("order"),
                "angle_deg": operation.get("angle_deg"),
                "notation_order": operation_notation_order(operation),
                "element_summary": "",
                "itc_like_summary": "",
                "itc_coordinate_summary": "",
                "element_sort_key": "",
                "direction_sort_key": "",
                "direction_label": "",
                "direction_filter_label": "",
                "matrix_frac": operation.get("matrix_frac"),
                "translation_frac": operation.get("translation_frac"),
                "matrix_cart": operation.get("matrix_cart"),
                "translation_cart": operation.get("translation_cart"),
                "fixed_atom_indices": operation_fixed_atom_indices(render_data, operation),
            }
        )
    return summaries


def operation_fixed_atom_indices(
    render_data: dict,
    operation: dict,
    *,
    tolerance_cart: float = 1e-6,
    molecule_tolerance_cart: float = 1e-2,
) -> list[int]:
    """Return atoms whose positions are unchanged by an operation.

    Crystal positions are compared modulo lattice translations with a tight
    tolerance (spglib operations are numerically exact). Molecule positions are
    compared directly in Cartesian coordinates with a looser tolerance, since
    point-group operations from the analyzer carry ~1e-4 A imprecision that
    would otherwise reject genuine on-axis / in-plane fixed atoms.
    """
    atoms = render_data.get("atoms", [])
    unit_cell = render_data.get("unit_cell")
    if unit_cell is not None:
        matrix = np.asarray(operation.get("matrix_frac"), dtype=float)
        translation = np.asarray(operation.get("translation_frac"), dtype=float)
        lattice = np.asarray(unit_cell.get("lattice"), dtype=float)
        if matrix.shape != (3, 3) or translation.shape != (3,) or lattice.shape != (3, 3):
            return []
        indices, positions = _atom_positions(atoms, "frac")
        if not len(indices):
            return []
        # One operation over every atom at once: this runs for all operations of
        # a structure, which is 192 x 56 matrix products on the largest example.
        displacement = positions @ matrix.T + translation - positions
        displacement -= np.rint(displacement)
        within = np.linalg.norm(displacement @ lattice, axis=1) <= tolerance_cart
    else:
        matrix = np.asarray(operation.get("matrix_cart"), dtype=float)
        translation = np.asarray(operation.get("translation_cart"), dtype=float)
        if matrix.shape != (3, 3) or translation.shape != (3,):
            return []
        indices, positions = _atom_positions(atoms, "cart")
        if not len(indices):
            return []
        displacement = positions @ matrix.T + translation - positions
        within = np.linalg.norm(displacement, axis=1) <= molecule_tolerance_cart
    return [int(index) for index in indices[within]]


def _atom_positions(atoms: list, key: str) -> tuple[np.ndarray, np.ndarray]:
    """Indices and positions of the atoms that carry a usable `key` coordinate."""
    indices = []
    positions = []
    for atom in atoms:
        position = atom.get(key)
        if position is None or len(position) != 3:
            continue
        indices.append(int(atom["index"]))
        positions.append(position)
    if not indices:
        return np.empty(0, dtype=int), np.empty((0, 3), dtype=float)
    return np.asarray(indices, dtype=int), np.asarray(positions, dtype=float)


def operation_notation_order(operation: dict) -> int | None:
    """Return the order written in the symbol rather than the matrix order."""
    symbol = str(operation.get("symbol") or operation.get("display_symbol") or "")
    match = re.search(r"[0-9]+", symbol)
    if match is not None:
        return int(match.group(0))
    order = operation.get("order")
    return int(order) if order is not None else None


def operation_summary_elements(
    render_data: dict,
    operation: dict,
) -> tuple[list[dict], list[dict], list[dict]]:
    operation_index = operation["index"]
    axes, planes, centers = (
        selected_elements(render_data["axes"], operation_index, element_index=None),
        selected_elements(render_data["planes"], operation_index, element_index=None),
        selected_elements(render_data["centers"], operation_index, element_index=None),
    )
    return visual_improper_elements(
        render_data,
        operation,
        axes,
        planes,
        centers,
        improper_mode="auto",
    )


def display_operation_symbol(render_data: dict, operation: dict, axes: list[dict], planes: list[dict]) -> str:
    symbol = operation.get("symbol") or operation["label"]
    if is_pure_translation_operation(operation):
        return "t"
    if str(operation["kind"]).find("glide") >= 0 and str(symbol) == "g":
        # The letter of the operation itself, as in the ITC-like notation.
        matrix = operation.get("matrix_frac")
        translation = operation.get("translation_frac")
        if matrix is not None and translation is not None:
            W = np.asarray(matrix, dtype=float)
            return _itc_glide_letter(W, _itc_t_intrinsic(W, np.asarray(translation, dtype=float), 2))
    if str(operation["kind"]).startswith("screw"):
        # The name of the operation itself, as in the ITC-like notation: a screw
        # advancing one whole lattice period (bcc's 3+(1/2,1/2,1/2)) is a plain 3.
        return itc_operation_symbol(render_data, operation, str(symbol)).rstrip("+-")
    return str(symbol)


def plane_hkl_vector(render_data: dict, plane: dict) -> np.ndarray | None:
    unit_cell = render_data.get("unit_cell")
    if unit_cell is None:
        return None
    lattice = np.asarray(unit_cell["lattice"], dtype=float)
    normal = np.asarray(plane["normal_cart"], dtype=float)
    return lattice @ normal


def glide_intrinsic_translation_frac(operation: dict) -> np.ndarray | None:
    """Intrinsic translation of a glide/reflection: (I + W_frac) / 2 @ t_frac.

    This is the canonical ITC glide vector, which equals the projection of t_frac
    onto the eigenspace of W_frac with eigenvalue +1 (the mirror plane).
    """
    W_frac = operation.get("matrix_frac")
    t_frac = operation.get("translation_frac")
    if W_frac is None or t_frac is None:
        return None
    W = np.asarray(W_frac, dtype=float)
    t = np.asarray(t_frac, dtype=float)
    return (np.eye(3) + W) @ t / 2


# Axis directions with the sign ITC Vol. A writes them in: the body diagonals keep
# a positive product of components, the face diagonals follow the cyclic order
# x -> y -> z ([1-10], [01-1], [-101]).  Other directions put the first
# non-zero component positive.
_ITC_DIRECTIONS = tuple(
    np.array(direction, dtype=float)
    for direction in (
        (1, 0, 0), (0, 1, 0), (0, 0, 1),
        (1, 1, 0), (1, -1, 0), (0, 1, 1), (0, 1, -1), (1, 0, 1), (-1, 0, 1),
        (1, 2, 0), (2, 1, 0),
        (1, 1, 1), (-1, 1, -1), (1, -1, -1), (-1, -1, 1),
    )
)


def _itc_direction(vector: np.ndarray) -> np.ndarray | None:
    ints = integer_index_vector(np.asarray(vector, dtype=float))
    if ints is None:
        return None
    ints = np.asarray(ints, dtype=float)
    for direction in _ITC_DIRECTIONS:
        if np.allclose(ints, direction) or np.allclose(ints, -direction):
            return direction.copy()
    first = ints[np.flatnonzero(np.abs(ints) > 1e-8)[0]]
    return ints if first > 0 else -ints


def _itc_parameter_name(direction: np.ndarray) -> str:
    return "xyz"[int(np.flatnonzero(np.abs(direction) > 1e-8)[0])]


def _itc_line(point: np.ndarray, direction: np.ndarray) -> str:
    """Spell the line point + s*direction as ITC does: the constant is zero in z,
    otherwise in x, otherwise in y ('x, -x+1/2, 1/4', '-x+1/3, x+1/3, -x')."""
    k = next(index for index in (2, 0, 1) if abs(direction[index]) > 1e-8)
    const = point - (point[k] / direction[k]) * direction
    name = _itc_parameter_name(direction)
    return ", ".join(_itc_coord_str(float(const[i]), [(float(direction[i]), name)]) for i in range(3))


def _itc_sense(render_data: dict, matrix_cart: np.ndarray, direction_frac: np.ndarray) -> str:
    lattice = np.asarray((render_data.get("unit_cell") or {}).get("lattice"), dtype=float)
    if lattice.shape != (3, 3):
        return ""
    direction_cart = normalize(np.asarray(direction_frac, dtype=float) @ lattice)
    angle = signed_rotation_angle_from_matrix(np.asarray(matrix_cart, dtype=float), direction_cart)
    if abs(angle) < 1e-8 or abs(abs(angle) - np.pi) < 1e-8:
        return ""
    return "+" if angle > 0 else "-"


def _itc_rotoinversion_notation(render_data: dict, operation: dict) -> str | None:
    """ITC notation of a 3-, 4- or 6-fold rotoinversion: '-4+ 0, 0, z; 0, 0, 0'.

    ITC gives the axis with the rotation sense of the rotation part -W, and the
    inversion point on it.
    """
    W_frac = operation.get("matrix_frac")
    t_frac = operation.get("translation_frac")
    matrix_cart = operation.get("matrix_cart")
    if W_frac is None or t_frac is None or matrix_cart is None:
        return None
    W = np.asarray(W_frac, dtype=float)
    if round(float(np.linalg.det(W))) != -1:
        return None
    fold = {0: 3, 1: 4, 2: 6}.get(round(float(np.trace(-W))))
    if fold is None:
        return None
    axes = _itc_null_space(-W - np.eye(3))
    if len(axes) != 1:
        return None
    direction = _itc_direction(axes[0])
    if direction is None:
        return None
    point = np.linalg.solve(W - np.eye(3), -np.asarray(t_frac, dtype=float))
    sense = _itc_sense(render_data, -np.asarray(matrix_cart, dtype=float), direction)
    point_text = ", ".join(_itc_coord_str(float(value), []) for value in point)
    return f"-{fold}{sense} {_itc_line(point, direction)}; {point_text}"


def _itc_reflection_notation(operation: dict) -> str | None:
    """ITC symbol of a reflection or glide, e.g. 'c' or 'n(1/2,1/2,1/2)'.

    The letter follows the operation's own glide vector, as ITC Vol. A does, not
    the shortest glide among lattice-equivalent planes (display_operation_symbol
    keeps that for the quiz). a, b, c and m print no vector; n, d and g do.
    """
    W_frac = operation.get("matrix_frac")
    t_frac = operation.get("translation_frac")
    if W_frac is None or t_frac is None:
        return None
    W = np.asarray(W_frac, dtype=float)
    if round(float(np.linalg.det(W))) != -1 or round(float(np.trace(W))) != 1:
        return None
    glide = _itc_t_intrinsic(W, np.asarray(t_frac, dtype=float), 2)
    letter = _itc_glide_letter(W, glide)
    if letter in ("n", "d", "g"):
        return f"{letter}{fractional_vector_label(glide)}"
    return letter


def _itc_glide_letter(W: np.ndarray, glide: np.ndarray) -> str:
    """a/b/c: half a basis vector; n: half of two (axial plane) or three (diagonal
    plane) basis vectors; d: the same with quarters; anything else is g."""
    normals = _itc_null_space(np.asarray(W, dtype=float).T + np.eye(3))
    axial = len(normals) == 1 and int(np.count_nonzero(np.abs(normals[0]) > 1e-8)) == 1
    magnitudes = np.abs(np.asarray(glide, dtype=float))
    nonzero = [index for index, value in enumerate(magnitudes) if value > 1e-6]
    if not nonzero:
        return "m"
    halves = all(abs(magnitudes[index] - 0.5) < 1e-6 for index in nonzero)
    quarters = all(min(abs(magnitudes[index] - 0.25), abs(magnitudes[index] - 0.75)) < 1e-6 for index in nonzero)
    if halves and len(nonzero) == 1:
        return "abc"[nonzero[0]]
    if len(nonzero) == (2 if axial else 3):
        if halves:
            return "n"
        if quarters:
            return "d"
    return "g"


def _itc_t_intrinsic(W: np.ndarray, t: np.ndarray, order: int) -> np.ndarray:
    """Intrinsic translation: (1/n) * sum_{k=0}^{n-1} W^k @ t."""
    acc = np.zeros(3)
    Wk = np.eye(3)
    for _ in range(order):
        acc += Wk @ t
        Wk = Wk @ W
    return acc / order


def _itc_null_space(A: np.ndarray, tol: float = 1e-7) -> list[np.ndarray]:
    """Rational null space basis vectors of integer matrix A via SVD."""
    _, s, vh = np.linalg.svd(A)
    rank = int(np.sum(s > tol))
    result = []
    for row in vh[rank:]:
        v = _itc_rationalize(row)
        if np.linalg.norm(v) > 1e-8:
            result.append(v)
    return result


def _itc_rationalize(v: np.ndarray) -> np.ndarray:
    """Round a unit float vector to primitive integer form."""
    v = np.asarray(v, dtype=float)
    max_abs = float(np.max(np.abs(v)))
    if max_abs < 1e-10:
        return np.zeros(3)
    v = v / max_abs
    fracs = [crystallographic_fraction(float(x)) or Fraction(round(float(x))) for x in v]
    lcm_d = 1
    for f in fracs:
        lcm_d = lcm_d * f.denominator // gcd(lcm_d, f.denominator)
    ints = [round(float(f) * lcm_d) for f in fracs]
    g = 1
    for x in ints:
        if x != 0:
            g = gcd(g, abs(x))
    return np.array([x / g for x in ints], dtype=float)


def _itc_coord_str(const: float, terms: list[tuple[float, str]]) -> str:
    """Format one coordinate: 'x+1/2', '-x', '1/4', '0', etc."""
    parts: list[str] = []
    for coeff, name in terms:
        c = crystallographic_fraction(float(coeff)) or Fraction(round(float(coeff)))
        if abs(float(c)) < 1e-8:
            continue
        if c == 1:
            parts.append(name)
        elif c == -1:
            parts.append(f"-{name}")
        elif c.denominator == 1:
            parts.append(f"{c.numerator}{name}")
        else:
            parts.append(f"({c}){name}")

    c_val = float(const)
    c_frac = crystallographic_fraction(c_val) or Fraction(0)
    has_const = abs(float(c_frac)) > 1e-8

    # str(Fraction), not format_fraction: that one wraps 1 to 0 for cell coordinates,
    # while an element location keeps constants such as x+1 or -x+3/2.
    if not parts:
        return str(c_frac) if has_const else "0"

    result = parts[0]
    for p in parts[1:]:
        result += p if p.startswith("-") else "+" + p
    if has_const:
        result += f"+{c_frac}" if c_frac > 0 else str(c_frac)
    return result


def _itc_plane_basis(null_vecs: list[np.ndarray]) -> list[np.ndarray]:
    """Two in-plane directions from _ITC_DIRECTIONS with different parameter
    names, shortest first ('x, x, z' uses [110] and [001], '-x, y, x' [-101] and [010])."""
    span = np.vstack(null_vecs)
    candidates = sorted(
        (direction for direction in _ITC_DIRECTIONS if np.linalg.matrix_rank(np.vstack([span, direction]), tol=1e-6) == 2),
        key=lambda direction: float(np.abs(direction).sum()),
    )
    for index, first in enumerate(candidates):
        for second in candidates[index + 1:]:
            if _itc_parameter_name(first) != _itc_parameter_name(second):
                return sorted((first, second), key=_itc_parameter_name)
    directions = [_itc_direction(vector) for vector in null_vecs]
    return sorted((d for d in directions if d is not None), key=_itc_parameter_name)


def _itc_plane(point: np.ndarray, basis: list[np.ndarray]) -> str:
    """Spell the plane through point as ITC does: the constants are zero in y and z
    when possible, otherwise in x and z, otherwise in x and y ('x+1/2, -x, z')."""
    const = point
    for i, j in ((1, 2), (0, 2), (0, 1)):
        minor = np.array([[basis[0][i], basis[1][i]], [basis[0][j], basis[1][j]]])
        if abs(float(np.linalg.det(minor))) > 1e-8:
            shift = np.linalg.solve(minor, -np.array([point[i], point[j]]))
            const = point + shift[0] * basis[0] + shift[1] * basis[1]
            break
    names = [_itc_parameter_name(direction) for direction in basis]
    return ", ".join(
        _itc_coord_str(float(const[i]), [(float(direction[i]), name) for direction, name in zip(basis, names)])
        for i in range(3)
    )


def operation_itc_position(operation: dict) -> str | None:
    """Location of the symmetry element in ITC Vol. A spelling, e.g. 'x, -x+1/2, 1/4'.

    Solves (W - I) x = -t_loc where t_loc = t - t_int, for the operation itself
    (no lattice translation is added or removed, so 3/4, 0, z stays 3/4, 0, z).
    """
    W_frac = operation.get("matrix_frac")
    t_frac = operation.get("translation_frac")
    order = operation.get("order")
    kind = str(operation.get("kind", ""))
    if W_frac is None or t_frac is None or order is None or order < 1:
        return None
    if "identity" in kind or is_pure_translation_operation(operation):
        return None

    W = np.asarray(W_frac, dtype=float)
    t = np.asarray(t_frac, dtype=float)

    t_int = _itc_t_intrinsic(W, t, order)
    t_loc = t - t_int

    A = W - np.eye(3)
    b = -t_loc

    null_vecs = _itc_null_space(A)
    x0, *_ = np.linalg.lstsq(A, b, rcond=None)
    if len(null_vecs) == 1:
        direction = _itc_direction(null_vecs[0])
        if direction is not None:
            return _itc_line(x0, direction)
    elif len(null_vecs) == 2:
        basis = _itc_plane_basis(null_vecs)
        if len(basis) == 2:
            return _itc_plane(x0, basis)
    elif not null_vecs:
        return ", ".join(_itc_coord_str(float(value), []) for value in x0)
    return None


def operation_element_summary(
    render_data: dict,
    operation: dict,
    axes: list[dict],
    planes: list[dict],
    centers: list[dict],
    *,
    display_symbol: str | None = None,
) -> str:
    parts = []
    effective_axis = axes[0] if axes else effective_axis_from_operation(operation, centers)
    if effective_axis is not None:
        axis = effective_axis
        parts.append(
            f"{axis_direction_label(render_data, axis)} "
            f"@ {point_label(render_data, axis['point_cart'])}"
        )
    if planes:
        plane = planes[0]
        summary = (
            f"{plane_normal_label(render_data, plane)} "
            f"@ {point_label(render_data, plane['point_cart'])}"
        )
        if "glide" in str(operation["kind"]) and display_symbol == "g":
            glide_frac = glide_intrinsic_translation_frac(operation)
            if glide_frac is None:
                glide_frac_geo = glide_translation_frac(render_data, operation, plane)
                if glide_frac_geo is not None:
                    glide_frac = centered_fractional_vector(glide_frac_geo)
            if glide_frac is not None:
                summary += f"; glide {fractional_vector_label(glide_frac)}"
        parts.append(summary)
    if centers and effective_axis is None:
        center = centers[0]
        parts.append(f"@ {point_label(render_data, center['point_cart'])}")
    if is_pure_translation_operation(operation) and not parts:
        t_frac = operation.get("translation_frac")
        if t_frac is not None:
            parts.append(fractional_vector_label(np.asarray(t_frac, dtype=float)))
        else:
            direction = translation_direction_label(render_data, operation)
            if direction is not None:
                parts.append(direction)
    return "; ".join(parts)


def operation_itc_like_summary(
    render_data: dict,
    operation: dict,
    axes: list[dict],
    planes: list[dict],
    centers: list[dict],
    *,
    display_symbol: str | None = None,
) -> str:
    # Identity: ITC notation is just "1"
    if "identity" in str(operation.get("kind", "")):
        return str(display_symbol or operation.get("symbol") or operation.get("label", "1"))

    # Pure translations: t|(p/q,r/s,u/v)
    if is_pure_translation_operation(operation):
        t_label = translation_frac_label(operation)
        if t_label is not None:
            return t_label

    rotoinversion = _itc_rotoinversion_notation(render_data, operation)
    if rotoinversion is not None:
        return rotoinversion

    # All other operations: symbol(t_int) position_expression
    position = operation_itc_position(operation)
    reflection = _itc_reflection_notation(operation)
    if position is not None and reflection is not None:
        return f"{reflection} {position}"
    if position is not None:
        symbol = itc_operation_symbol(
            render_data,
            operation,
            display_symbol or str(operation.get("symbol") or operation.get("label", "?")),
        )
        order = operation.get("order")
        W_frac = operation.get("matrix_frac")
        t_frac = operation.get("translation_frac")
        if order and W_frac is not None and t_frac is not None:
            t_int = _itc_t_intrinsic(
                np.asarray(W_frac, dtype=float),
                np.asarray(t_frac, dtype=float),
                order,
            )
            if np.linalg.norm(t_int) > 1e-8:
                return f"{symbol}{fractional_vector_label(t_int)} {position}"
        return f"{symbol} {position}"

    # Fallback to element summary
    return operation_element_summary(
        render_data,
        operation,
        axes,
        planes,
        centers,
        display_symbol=display_symbol,
    )


def itc_operation_symbol(render_data: dict, operation: dict, symbol: str) -> str:
    """Return an ITC operation symbol, including the sense of proper rotations.

    ITC distinguishes the two inverse 3-, 4-, and 6-fold operations with + and
    -. The sign is measured around a canonically oriented crystallographic axis,
    rather than depending on whichever equivalent axis element was selected.
    """
    kind = str(operation.get("kind", ""))
    order = operation.get("order")
    if order not in (2, 3, 4, 6) or not kind.startswith(("rotation_", "screw_")):
        return symbol

    matrix_frac = operation.get("matrix_frac")
    translation_frac = operation.get("translation_frac")
    matrix_cart = operation.get("matrix_cart")
    if matrix_frac is None or translation_frac is None or matrix_cart is None:
        return symbol
    W = np.asarray(matrix_frac, dtype=float)
    null_vecs = _itc_null_space(W - np.eye(3))
    if len(null_vecs) != 1:
        return symbol
    # The sense is measured about the axis direction written in the position
    # (ITC's 3+ -x, x, -x turns about [-11-1], not [1-11]).
    direction_frac = _itc_direction(null_vecs[0])
    if direction_frac is None:
        return symbol
    sense = _itc_sense(render_data, np.asarray(matrix_cart, dtype=float), direction_frac) if order != 2 else ""
    t_int = _itc_t_intrinsic(W, np.asarray(translation_frac, dtype=float), int(order))
    return f"{_itc_rotation_name(render_data, int(order), t_int, direction_frac, sense)}{sense}"


def _itc_rotation_name(render_data: dict, order: int, t_int: np.ndarray, direction: np.ndarray, sense: str) -> str:
    """'4' or '4_1' for the operation itself, not for its lattice-equivalent class.

    The subscript counts how many 1/n periods the operation's own intrinsic
    translation advances along the axis; the period is the shortest lattice vector
    along it, centring vectors included, so bcc's 3+(1/2,1/2,1/2) is a plain 3.
    A clockwise turn advancing m/n lies on an n_(n-m) axis.
    """
    if np.allclose(t_int, 0.0, atol=1e-6):
        return str(order)
    centrings = [np.zeros(3)] + [
        np.asarray(other["translation_frac"], dtype=float)
        for other in render_data.get("operations", [])
        if is_pure_translation_operation(other) and other.get("translation_frac") is not None
    ]
    period = np.asarray(direction, dtype=float)
    for divisor in range(12, 1, -1):
        candidate = period / divisor
        if any(np.allclose((candidate - c + 0.5) % 1.0 - 0.5, 0.0, atol=1e-6) for c in centrings):
            period = candidate
            break
    fraction = (float(t_int @ period) / float(period @ period)) % 1.0
    if sense == "-":
        fraction = (1.0 - fraction) % 1.0
    steps = int(round(fraction * order)) % order
    return f"{order}_{steps}" if steps else str(order)


def operation_element_sort_key(
    render_data: dict,
    operation: dict,
    axes: list[dict],
    planes: list[dict],
    centers: list[dict],
) -> str:
    direction = operation_direction_sort_key(render_data, operation, axes, planes, centers)
    point = operation_point_sort_key(render_data, operation, axes, planes, centers)
    return f"{direction}|{point}|{operation.get('symbol', '')}"


def operation_direction_sort_key(
    render_data: dict,
    operation: dict,
    axes: list[dict],
    planes: list[dict],
    centers: list[dict],
) -> str:
    effective_axis = axes[0] if axes else effective_axis_from_operation(operation, centers)
    if effective_axis is not None:
        return "axis:" + plain_index_label(axis_direction_label(render_data, effective_axis))
    if planes:
        return "plane:" + plain_index_label(plane_normal_label(render_data, planes[0]))
    translation_direction = translation_direction_label(render_data, operation)
    if translation_direction is not None:
        return "translation:" + plain_index_label(translation_direction)
    if centers:
        return "center"
    return "none"


def operation_direction_label(
    render_data: dict,
    operation: dict,
    axes: list[dict],
    planes: list[dict],
    centers: list[dict],
) -> str:
    effective_axis = axes[0] if axes else effective_axis_from_operation(operation, centers)
    if effective_axis is not None:
        return axis_direction_label(render_data, effective_axis)
    if planes:
        return plane_normal_label(render_data, planes[0])
    translation_direction = translation_direction_label(render_data, operation)
    if translation_direction is not None:
        return translation_direction
    if centers:
        return "center"
    return "none"


def operation_direction_filter_label(
    render_data: dict,
    operation: dict,
    axes: list[dict],
    planes: list[dict],
    centers: list[dict],
) -> str:
    effective_axis = axes[0] if axes else effective_axis_from_operation(operation, centers)
    if effective_axis is not None:
        return axis_direction_label_text(render_data, effective_axis)
    if planes:
        return plane_normal_label_text(render_data, planes[0])
    translation_direction = translation_direction_label_text(render_data, operation)
    if translation_direction is not None:
        return translation_direction
    if centers:
        return "center"
    return "none"


def operation_point_sort_key(
    render_data: dict,
    operation: dict,
    axes: list[dict],
    planes: list[dict],
    centers: list[dict],
) -> str:
    effective_axis = axes[0] if axes else effective_axis_from_operation(operation, centers)
    if effective_axis is not None:
        return point_sort_key(render_data, effective_axis["point_cart"])
    if planes:
        return point_sort_key(render_data, planes[0]["point_cart"])
    if centers:
        return point_sort_key(render_data, centers[0]["point_cart"])
    return ""


def operation_view_direction_cart(
    render_data: dict,
    operation: dict,
    axes: list[dict],
    planes: list[dict],
    centers: list[dict],
) -> np.ndarray | None:
    effective_axis = axes[0] if axes else effective_axis_from_operation(operation, centers)
    if effective_axis is not None:
        return np.asarray(effective_axis["direction_cart"], dtype=float)
    if planes:
        return np.asarray(planes[0]["normal_cart"], dtype=float)
    return translation_direction_cart(operation)


def operation_focus_point_cart(
    render_data: dict,
    operation: dict,
    axes: list[dict],
    planes: list[dict],
    centers: list[dict],
    display_mode: str,
    cell_origin_mode: str = "center",
) -> np.ndarray:
    effective_axis = axes[0] if axes else effective_axis_from_operation(operation, centers)
    if effective_axis is not None:
        return display_point_cart(render_data, effective_axis["point_cart"], display_mode, cell_origin_mode)
    if planes:
        return display_point_cart(render_data, planes[0]["point_cart"], display_mode, cell_origin_mode)
    if centers:
        return display_point_cart(render_data, centers[0]["point_cart"], display_mode, cell_origin_mode)
    return display_scene_center(render_data, display_mode, cell_origin_mode)


def custom_focus_point_cart(
    result: dict,
    render_data: dict,
    display_mode: str,
    cell_origin_mode: str = "center",
) -> np.ndarray | None:
    elements = result.get("elements") or {}
    for key in ("axes", "planes", "centers"):
        items = elements.get(key) or []
        if items:
            return display_point_cart(render_data, items[0]["point_cart"], display_mode, cell_origin_mode)
    return None


def visual_translation_direction_cart(
    render_data: dict,
    operation: dict,
    atom_mappings: dict | None,
) -> np.ndarray | None:
    if not is_pure_translation_operation(operation):
        return None
    mapping = selected_mapping(atom_mappings, operation["index"])
    if mapping is None:
        return None
    paths = animation_paths(
        render_data,
        operation,
        mapping,
        animation_scope="representative",
    )
    if not paths:
        return None
    path = next(iter(paths.values()))
    start = np.asarray(path["start"], dtype=float)
    target = np.asarray(path["target"], dtype=float)
    displacement = target - start
    if np.linalg.norm(displacement) < 1e-10:
        return None
    return displacement


def translation_frac_label(operation: dict) -> str | None:
    t_frac = operation.get("translation_frac")
    if t_frac is None:
        return None
    return "t|" + fractional_vector_label(np.asarray(t_frac, dtype=float))


def translation_direction_label(render_data: dict, operation: dict) -> str | None:
    direction = translation_direction_cart(operation)
    if direction is None:
        return None
    unit_cell = render_data.get("unit_cell")
    if unit_cell is None:
        return vector_label(direction, bracket=("[", "]"))
    frac_direction = direction @ lattice_inverse(unit_cell)
    return integer_index_label(frac_direction, bracket=("[", "]"), orient_positive=False)


def translation_direction_label_text(render_data: dict, operation: dict) -> str | None:
    direction = translation_direction_cart(operation)
    if direction is None:
        return None
    unit_cell = render_data.get("unit_cell")
    if unit_cell is None:
        return vector_label(direction, bracket=("[", "]"))
    frac_direction = direction @ lattice_inverse(unit_cell)
    return integer_index_label_text(frac_direction, bracket=("[", "]"), orient_positive=False)


def translation_direction_cart(operation: dict) -> np.ndarray | None:
    if not is_pure_translation_operation(operation):
        return None
    display_translation = operation.get("_display_translation_cart")
    if display_translation is not None:
        direction = np.asarray(display_translation, dtype=float)
        if np.linalg.norm(direction) >= 1e-10:
            return direction
    translation = operation.get("translation_cart")
    if translation is None:
        return None
    direction = np.asarray(translation, dtype=float)
    if np.linalg.norm(direction) < 1e-10:
        return None
    return direction


def is_pure_translation_operation(operation: dict) -> bool:
    kind = str(operation.get("kind", ""))
    if "translation" not in kind:
        return False
    matrix = operation.get("matrix_cart")
    if matrix is None:
        return True
    return bool(np.allclose(np.asarray(matrix, dtype=float), np.eye(3), atol=1e-8))


def point_sort_key(render_data: dict, point_cart: list[float]) -> str:
    point = np.asarray(point_cart, dtype=float)
    unit_cell = render_data.get("unit_cell")
    if unit_cell is None:
        values = point
    else:
        values = point @ lattice_inverse(unit_cell)
        values = values - np.floor(values + 1e-9)
    return ",".join(f"{float(value):.6f}" for value in values)


def plain_index_label(label: str) -> str:
    return (
        label.replace("<span class=\"overline\">", "-")
        .replace("</span>", "")
        .replace("[", "")
        .replace("]", "")
        .replace("(", "")
        .replace(")", "")
    )


def effective_axis_from_operation(operation: dict, centers: list[dict]) -> dict | None:
    kind = str(operation["kind"])
    if "rotoinversion" not in kind and "rotoreflection" not in kind and "improper" not in kind:
        return None
    if not centers:
        return None
    center = centers[0]
    axis = effective_rotation_axis(operation, None, center)
    return axis


def axis_direction_label(render_data: dict, axis: dict) -> str:
    vector = np.asarray(axis["direction_cart"], dtype=float)
    unit_cell = render_data.get("unit_cell")
    if unit_cell is None:
        return vector_label(vector, bracket=("[", "]"))
    frac_direction = vector @ lattice_inverse(unit_cell)
    return integer_index_label(frac_direction, bracket=("[", "]"))


def axis_direction_label_text(render_data: dict, axis: dict) -> str:
    vector = np.asarray(axis["direction_cart"], dtype=float)
    unit_cell = render_data.get("unit_cell")
    if unit_cell is None:
        return vector_label(vector, bracket=("[", "]"))
    frac_direction = vector @ lattice_inverse(unit_cell)
    return integer_index_label_text(frac_direction, bracket=("[", "]"))


def plane_normal_label(render_data: dict, plane: dict) -> str:
    hkl = plane_hkl_vector(render_data, plane)
    if hkl is None:
        normal = np.asarray(plane["normal_cart"], dtype=float)
        return vector_label(normal, bracket=("(", ")"))
    return integer_index_label(hkl, bracket=("(", ")"))


def plane_normal_label_text(render_data: dict, plane: dict) -> str:
    hkl = plane_hkl_vector(render_data, plane)
    if hkl is None:
        normal = np.asarray(plane["normal_cart"], dtype=float)
        return vector_label(normal, bracket=("(", ")"))
    return integer_index_label_text(hkl, bracket=("(", ")"))


def point_label(render_data: dict, point_cart: list[float]) -> str:
    point = np.asarray(point_cart, dtype=float)
    unit_cell = render_data.get("unit_cell")
    if unit_cell is None:
        return vector_label(point, bracket=("(", ")"))
    frac = point @ lattice_inverse(unit_cell)
    wrapped = frac - np.floor(frac + 1e-9)
    return "(" + ", ".join(format_fraction(value) for value in wrapped) + ")"


def fractional_vector_label(values: np.ndarray) -> str:
    return "(" + ",".join(format_fraction(float(value)) for value in values) + ")"


def lattice_inverse(unit_cell: dict) -> np.ndarray:
    lattice = np.asarray(unit_cell["lattice"], dtype=float)
    return cached_lattice_inverse(tuple(float(value) for value in lattice.ravel()))


@lru_cache(maxsize=64)
def cached_lattice_inverse(flat_lattice: tuple[float, ...]) -> np.ndarray:
    lattice = np.asarray(flat_lattice, dtype=float).reshape((3, 3))
    inverse = np.linalg.inv(lattice)
    inverse.flags.writeable = False
    return inverse


def atom_frac_label(atom: dict) -> str | None:
    frac = atom.get("frac")
    if frac is None:
        return None
    return "(" + ", ".join(format_fraction(float(value)) for value in frac) + ")"


def integer_index_label(values: np.ndarray, *, bracket: tuple[str, str], orient_positive: bool = True) -> str:
    ints = integer_index_vector(values, orient_positive=orient_positive)
    if ints is None:
        return f"{bracket[0]}0 0 0{bracket[1]}"
    return bracket[0] + " ".join(format_index(int(value)) for value in ints) + bracket[1]


def integer_index_label_text(values: np.ndarray, *, bracket: tuple[str, str], orient_positive: bool = True) -> str:
    ints = integer_index_vector(values, orient_positive=orient_positive)
    if ints is None:
        return f"{bracket[0]}0 0 0{bracket[1]}"
    return bracket[0] + " ".join(format_index_text(int(value)) for value in ints) + bracket[1]


def format_index(value: int) -> str:
    return f"<span class=\"overline\">{abs(value)}</span>" if value < 0 else str(value)


def format_index_text(value: int) -> str:
    return f"{abs(value)}\u0305" if value < 0 else str(value)


# Denominators that occur in the 230 space groups' symmetry-element coordinates
# and intrinsic translations: 1/2,1/3,1/4,1/6 (most groups) and 1/8 (Fd-3m etc.).
# Listed smallest-first so the simplest valid fraction wins.
CRYSTALLOGRAPHIC_DENOMINATORS = (1, 2, 3, 4, 6, 8, 12)


def crystallographic_fraction(value: float, *, tol: float = 1e-3) -> Fraction | None:
    """Snap a value to the nearest fraction with a crystallographically valid
    denominator (a divisor of 24).  Returns None when no valid fraction is
    within tolerance, so genuinely non-crystallographic values surface as
    decimals instead of invented fractions like 13/20.
    """
    value = float(value)
    for denominator in CRYSTALLOGRAPHIC_DENOMINATORS:
        numerator = round(value * denominator)
        candidate = Fraction(numerator, denominator)
        if abs(value - float(candidate)) < tol:
            return candidate
    return None


def format_fraction(value: float) -> str:
    value = float(value)
    if abs(value) < 1e-8 or abs(value - 1.0) < 1e-8:
        return "0"
    fraction = crystallographic_fraction(value)
    if fraction is not None:
        if fraction.denominator == 1:
            return str(fraction.numerator)
        return f"{fraction.numerator}/{fraction.denominator}"
    return f"{value:.3f}"


def vector_label(values: np.ndarray, *, bracket: tuple[str, str]) -> str:
    return bracket[0] + ", ".join(f"{float(value):.3f}" for value in values) + bracket[1]


def camera_up_vector(direction: np.ndarray) -> np.ndarray:
    direction = normalize(direction)
    candidates = [
        np.asarray([0.0, 0.0, 1.0]),
        np.asarray([0.0, 1.0, 0.0]),
        np.asarray([1.0, 0.0, 0.0]),
    ]
    up = min(candidates, key=lambda candidate: abs(float(np.dot(candidate, direction))))
    up = up - np.dot(up, direction) * direction
    return normalize(up)


def rotate_vector(vector: np.ndarray, axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = normalize(axis)
    vector = np.asarray(vector, dtype=float)
    return (
        vector * np.cos(angle_rad)
        + np.cross(axis, vector) * np.sin(angle_rad)
        + axis * np.dot(axis, vector) * (1.0 - np.cos(angle_rad))
    )
