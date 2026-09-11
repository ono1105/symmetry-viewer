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
    if str(operation["kind"]).find("glide") >= 0 and str(symbol) == "g" and planes:
        inferred = infer_standard_glide_symbol_for_planes(render_data, operation, planes)
        if inferred is not None:
            return inferred
    if "?" not in str(symbol):
        return str(symbol)
    if not str(operation["kind"]).startswith("screw") or not axes:
        return str(symbol)
    inferred = infer_screw_symbol(render_data, operation, axes[0])
    return inferred or str(symbol)


def infer_screw_symbol(render_data: dict, operation: dict, axis: dict) -> str | None:
    order = operation.get("order")
    matrix = operation.get("matrix_cart")
    translation = operation.get("translation_cart")
    unit_cell = render_data.get("unit_cell")
    if order is None or matrix is None or translation is None or unit_cell is None:
        return None

    point = np.asarray(axis["point_cart"], dtype=float)
    direction = normalize(np.asarray(axis["direction_cart"], dtype=float))
    moved = np.asarray(matrix, dtype=float) @ point + np.asarray(translation, dtype=float)
    displacement = moved - point
    projected = float(np.dot(displacement, direction))

    lattice = np.asarray(unit_cell["lattice"], dtype=float)
    frac_direction = direction @ lattice_inverse(unit_cell)
    primitive_frac = integer_index_vector(frac_direction)
    if primitive_frac is None:
        return None
    period = float(np.linalg.norm(primitive_frac @ lattice))
    if period < 1e-10:
        return None

    fraction = (projected / period) % 1.0
    order_int = int(order)
    if order_int == 2 and not np.isclose(fraction, 0.0, atol=1e-6):
        screw = 1
    else:
        raw_screw = fraction * order_int
        screw = int(np.floor(raw_screw + 0.5 + 1e-8))
        if screw == 0 and not np.isclose(fraction, 0.0, atol=1e-6):
            screw = 1
        elif screw >= order_int:
            screw = order_int - 1
    if screw == 0:
        return None
    return f"{order_int}_{screw}"


def plane_hkl_vector(render_data: dict, plane: dict) -> np.ndarray | None:
    unit_cell = render_data.get("unit_cell")
    if unit_cell is None:
        return None
    lattice = np.asarray(unit_cell["lattice"], dtype=float)
    normal = np.asarray(plane["normal_cart"], dtype=float)
    return lattice @ normal


def classify_standard_glide_vector(glide_frac: np.ndarray) -> str | None:
    magnitudes = np.abs(centered_fractional_vector(glide_frac))
    half_axes = [index for index, value in enumerate(magnitudes) if abs(float(value) - 0.5) < 1e-5]
    quarter_axes = [
        index
        for index, value in enumerate(magnitudes)
        if min(abs(float(value) - 0.25), abs(float(value) - 0.75)) < 1e-5
    ]
    other_axes = [
        index
        for index, value in enumerate(magnitudes)
        if value > 1e-5 and index not in half_axes and index not in quarter_axes
    ]
    if other_axes:
        return None
    if len(quarter_axes) >= 2 and not half_axes:
        return "d"
    if len(half_axes) >= 2 and not quarter_axes:
        return "n"
    if len(half_axes) == 1 and not quarter_axes:
        return ("a", "b", "c")[half_axes[0]]
    return None


def infer_standard_glide_symbol_for_planes(render_data: dict, operation: dict, planes: list[dict]) -> str | None:
    candidates = []
    for plane in planes:
        glide_frac = glide_translation_frac(render_data, operation, plane)
        if glide_frac is None:
            continue
        symbol = classify_standard_glide_vector(glide_frac)
        if symbol is None:
            continue
        centered = centered_fractional_vector(glide_frac)
        candidates.append((glide_vector_cart_norm(render_data, centered), symbol))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    best_norm = candidates[0][0]
    symbols = {symbol for norm, symbol in candidates if abs(norm - best_norm) < 1e-5}
    if len(symbols) != 1:
        return None
    return next(iter(symbols))


def glide_vector_cart_norm(render_data: dict, glide_frac: np.ndarray) -> float:
    unit_cell = render_data.get("unit_cell")
    if unit_cell is None:
        return float(np.linalg.norm(glide_frac))
    lattice = np.asarray(unit_cell["lattice"], dtype=float)
    return float(np.linalg.norm(np.asarray(glide_frac, dtype=float) @ lattice))


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


def _itc_param_names(null_vecs: list[np.ndarray]) -> list[str]:
    """Assign 'x', 'y', 'z' to null space vectors by dominant component."""
    avail = ["x", "y", "z"]
    used: set[str] = set()
    names = []
    for v in null_vecs:
        order = list(np.argsort(-np.abs(v)))
        name = next((avail[i] for i in order if avail[i] not in used), None)
        if name is None:
            name = next(n for n in avail if n not in used)
        used.add(name)
        names.append(name)
    return names


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

    if not parts:
        return format_fraction(float(c_frac)) if has_const else "0"

    result = parts[0]
    for p in parts[1:]:
        result += p if p.startswith("-") else "+" + p
    if has_const:
        c_str = format_fraction(float(c_frac))
        result += "+" + c_str if float(c_frac) > 0 else c_str
    return result


def _itc_normalize(
    x0: np.ndarray, null_vecs: list[np.ndarray]
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Normalize parametric position toward ITC canonical form.

    Rules applied in order for each null vector:
    1. Flip sign so the first non-zero component is positive.
    2a. (negative coeff) Shift to zero the constant at the first coordinate
        with a rounded negative integer coefficient ('-x' style).
    2b. (all-positive coeff) Shift to zero the constant at the LAST non-zero
        coordinate, placing any remainder on earlier coordinates.
        Example: [1,1,0] → zeroes coord 1, leaves constant on coord 0.
    3. Center each x0 component in (-1/2, 1/2] to match ITC convention.

    Known limitations — see REVIEW_NOTES for details:
    - Null vecs with multiple negative integer components (e.g. [2,-1,-1]):
      only the first negative coordinate's constant is zeroed.
    - Ambiguous tie-breaking when equally valid shifts exist.
    """
    x0 = x0.copy()
    out_vecs = []
    for v in null_vecs:
        v = v.copy()
        # Rule 1: flip sign so first non-zero component is positive
        for comp in v:
            if abs(comp) > 1e-8:
                if comp < 0:
                    v = -v
                break
        # Rule 2a: shift to zero constant at first coord with rounded coeff < 0
        zeroed = False
        for j in range(3):
            r = round(float(v[j]))
            if r < 0 and abs(float(v[j]) - r) < 1e-8:
                c = -float(x0[j]) / float(v[j])
                x0 = x0 + c * v
                zeroed = True
                break
        # Rule 2b: for all-positive vecs, zero the last non-zero component
        if not zeroed:
            for j in range(2, -1, -1):
                r = round(float(v[j]))
                if abs(r) > 0 and abs(float(v[j]) - r) < 1e-8 and abs(float(x0[j])) > 1e-8:
                    c = -float(x0[j]) / float(v[j])
                    x0 = x0 + c * v
                    break
        out_vecs.append(v)
    # Rule 3: center each x0 component in (-1/2, 1/2] (ITC convention)
    for i in range(3):
        xi = float(x0[i]) % 1.0
        if xi > 0.5 + 1e-8:
            xi -= 1.0
        x0[i] = 0.0 if abs(xi) < 1e-8 else xi
    return x0, out_vecs


def operation_itc_position(operation: dict) -> str | None:
    """Parametric position of the symmetry element, e.g. 'x+1/4, -x+1/4, z'.

    Solves (W - I) x = -t_loc where t_loc = t - t_int.
    Free variables in the null space become parameters (x, y, z) assigned
    by dominant component.  Parameter normalization to canonical ITC form
    is NOT applied here; this can be added as a post-processing step later.
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
    x0, null_vecs = _itc_normalize(x0, null_vecs)

    param_names = _itc_param_names(null_vecs)

    coords = []
    for i in range(3):
        terms = [(float(v[i]), p) for v, p in zip(null_vecs, param_names)]
        coords.append(_itc_coord_str(float(x0[i]), terms))
    return ", ".join(coords)


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

    # All other operations: symbol(t_int) position_expression
    position = operation_itc_position(operation)
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
    if operation_notation_order(operation) not in (3, 4, 6) or not kind.startswith(("rotation_", "screw_")):
        return symbol
    if symbol.endswith(("+", "-")):
        return symbol

    unit_cell = render_data.get("unit_cell") or {}
    lattice = np.asarray(unit_cell.get("lattice"), dtype=float)
    matrix_frac = operation.get("matrix_frac")
    matrix_cart = operation.get("matrix_cart")
    if lattice.shape != (3, 3) or matrix_frac is None or matrix_cart is None:
        return symbol

    null_vecs = _itc_null_space(np.asarray(matrix_frac, dtype=float) - np.eye(3))
    if len(null_vecs) != 1:
        return symbol
    direction_frac = integer_index_vector(null_vecs[0])
    if direction_frac is None:
        return symbol
    direction_cart = normalize(direction_frac @ lattice)
    if np.linalg.norm(direction_cart) < 1e-8:
        return symbol

    angle = signed_rotation_angle_from_matrix(np.asarray(matrix_cart, dtype=float), direction_cart)
    if abs(angle) < 1e-8:
        return symbol
    return f"{symbol}{'+' if angle > 0 else '-'}"


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
