from __future__ import annotations

import numpy as np

from crystal_viewer.geometry import normalize, plane_basis_from_normal_cart


def rotation_matrix_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues rotation formula. axis must be a unit vector."""
    c, s = np.cos(angle), np.sin(angle)
    t = 1.0 - c
    x, y, z = axis
    return np.array([
        [t*x*x + c,   t*x*y - s*z, t*x*z + s*y],
        [t*x*y + s*z, t*y*y + c,   t*y*z - s*x],
        [t*x*z - s*y, t*y*z + s*x, t*z*z + c  ],
    ])


def build_custom_operation_frac(
    op_type: str,
    params: dict,
    lattice: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | str:
    """
    Convert human-friendly parameters to (W_frac 3x3, t_frac 3D).
    lattice: 3x3 with rows = a,b,c vectors (pymatgen convention).
    Convention: x'_frac = W_frac @ x_frac + t_frac  (column vector, spglib-style).
    Returns error string on bad input.
    """
    try:
        inv_lattice = np.linalg.inv(lattice)
        inv_lt = np.linalg.inv(lattice.T)
        lt = lattice.T

        def w_from_cart(W_cart: np.ndarray) -> np.ndarray:
            return inv_lt @ W_cart @ lt

        def finish(W_frac: np.ndarray, t_frac: np.ndarray) -> tuple[np.ndarray, np.ndarray] | str:
            W = np.asarray(W_frac, dtype=float)
            t = np.asarray(t_frac, dtype=float)
            if W.shape != (3, 3) or not np.all(np.isfinite(W)):
                return "Operation matrix W must be a finite 3x3 matrix"
            if t.shape != (3,) or not np.all(np.isfinite(t)):
                return "Operation translation t must be a finite vector of length 3"
            return W, t

        if op_type == "identity":
            return finish(np.eye(3), np.zeros(3))

        if op_type == "translation":
            t = np.asarray(params["vector"], dtype=float)
            return finish(np.eye(3), t)

        if op_type == "rotation":
            uvw = np.asarray(params["axis"], dtype=float)
            d_cart = uvw @ lattice
            if np.linalg.norm(d_cart) < 1e-10:
                return "Axis direction is zero vector"
            d_hat = d_cart / np.linalg.norm(d_cart)
            angle = np.deg2rad(float(params["angle"]))
            W_cart = rotation_matrix_from_axis_angle(d_hat, angle)
            W_frac = w_from_cart(W_cart)
            p = np.asarray(params.get("point", [0, 0, 0]), dtype=float)
            return finish(W_frac, (np.eye(3) - W_frac) @ p)

        if op_type == "mirror":
            hkl = np.asarray(params["normal"], dtype=float)
            # hkl = L @ n_cart  →  n_cart = inv(L) @ hkl
            n_cart = inv_lattice @ hkl
            if np.linalg.norm(n_cart) < 1e-10:
                return "Plane normal is zero vector"
            n_hat = n_cart / np.linalg.norm(n_cart)
            W_cart = np.eye(3) - 2.0 * np.outer(n_hat, n_hat)
            W_frac = w_from_cart(W_cart)
            p = np.asarray(params.get("point", [0, 0, 0]), dtype=float)
            return finish(W_frac, (np.eye(3) - W_frac) @ p)

        if op_type == "inversion":
            c = np.asarray(params.get("center", [0, 0, 0]), dtype=float)
            W_frac = -np.eye(3)
            return finish(W_frac, 2.0 * c)

        if op_type == "screw":
            uvw = np.asarray(params["axis"], dtype=float)
            d_cart = uvw @ lattice
            if np.linalg.norm(d_cart) < 1e-10:
                return "Axis direction is zero vector"
            d_hat = d_cart / np.linalg.norm(d_cart)
            angle = np.deg2rad(float(params["angle"]))
            W_cart = rotation_matrix_from_axis_angle(d_hat, angle)
            W_frac = w_from_cart(W_cart)
            p = np.asarray(params.get("point", [0, 0, 0]), dtype=float)
            screw = np.asarray(params.get("screw", [0, 0, 0]), dtype=float)
            return finish(W_frac, (np.eye(3) - W_frac) @ p + screw)

        if op_type == "glide":
            hkl = np.asarray(params["normal"], dtype=float)
            n_cart = inv_lattice @ hkl
            if np.linalg.norm(n_cart) < 1e-10:
                return "Plane normal is zero vector"
            n_hat = n_cart / np.linalg.norm(n_cart)
            W_cart = np.eye(3) - 2.0 * np.outer(n_hat, n_hat)
            W_frac = w_from_cart(W_cart)
            p = np.asarray(params.get("point", [0, 0, 0]), dtype=float)
            glide = np.asarray(params.get("glide", [0, 0, 0]), dtype=float)
            return finish(W_frac, (np.eye(3) - W_frac) @ p + glide)

        if op_type == "rotoinversion":
            uvw = np.asarray(params["axis"], dtype=float)
            d_cart = uvw @ lattice
            if np.linalg.norm(d_cart) < 1e-10:
                return "Axis direction is zero vector"
            d_hat = d_cart / np.linalg.norm(d_cart)
            angle = np.deg2rad(float(params["angle"]))
            W_rot_cart = rotation_matrix_from_axis_angle(d_hat, angle)
            W_cart = -W_rot_cart  # rotoinversion = rotation then inversion
            W_frac = w_from_cart(W_cart)
            c = np.asarray(params.get("center", [0, 0, 0]), dtype=float)
            return finish(W_frac, (np.eye(3) - W_frac) @ c)

        if op_type == "matrix":
            W_frac = np.asarray(params["W"], dtype=float).reshape(3, 3)
            t_frac = np.asarray(params["t"], dtype=float)
            return finish(W_frac, t_frac)

        return f"Unknown operation type: {op_type!r}"

    except (KeyError, ValueError, TypeError, np.linalg.LinAlgError) as exc:
        return f"Parameter error: {exc}"


def custom_operation_visuals(
    op_type: str,
    params: dict,
    lattice: np.ndarray,
    W_frac: np.ndarray,
    t_frac: np.ndarray,
) -> dict:
    axes: list[dict] = []
    planes: list[dict] = []
    centers: list[dict] = []
    view_direction: np.ndarray | None = None

    try:
        inv_lattice = np.linalg.inv(lattice)
        if op_type in ("rotation", "screw"):
            uvw = np.asarray(params.get("axis", [0, 0, 1]), dtype=float)
            direction = uvw @ lattice
            if np.linalg.norm(direction) >= 1e-10:
                point = np.asarray(params.get("point", [0, 0, 0]), dtype=float) @ lattice
                direction = normalize(direction)
                axes.append(
                    {
                        "label": "custom axis",
                        "point_cart": point.tolist(),
                        "direction_cart": direction.tolist(),
                    }
                )
                view_direction = direction

        elif op_type in ("mirror", "glide"):
            hkl = np.asarray(params.get("normal", [0, 0, 1]), dtype=float)
            normal = inv_lattice @ hkl
            if np.linalg.norm(normal) >= 1e-10:
                point = np.asarray(params.get("point", [0, 0, 0]), dtype=float) @ lattice
                normal = normalize(normal)
                basis1, basis2 = plane_basis_from_normal(normal)
                planes.append(
                    {
                        "label": "custom plane",
                        "point_cart": point.tolist(),
                        "normal_cart": normal.tolist(),
                        "basis1_cart": basis1.tolist(),
                        "basis2_cart": basis2.tolist(),
                    }
                )
                view_direction = normal

        elif op_type == "inversion":
            center = np.asarray(params.get("center", [0, 0, 0]), dtype=float) @ lattice
            centers.append({"label": "custom center", "point_cart": center.tolist()})

        elif op_type == "rotoinversion":
            uvw = np.asarray(params.get("axis", [0, 0, 1]), dtype=float)
            direction = uvw @ lattice
            center = np.asarray(params.get("center", [0, 0, 0]), dtype=float) @ lattice
            centers.append({"label": "custom center", "point_cart": center.tolist()})
            if np.linalg.norm(direction) >= 1e-10:
                direction = normalize(direction)
                axes.append(
                    {
                        "label": "custom axis",
                        "point_cart": center.tolist(),
                        "direction_cart": direction.tolist(),
                    }
                )
                view_direction = direction

        elif op_type == "translation":
            direction = np.asarray(t_frac, dtype=float) @ lattice
            if np.linalg.norm(direction) >= 1e-10:
                view_direction = direction

        elif op_type == "matrix":
            direction = np.asarray(t_frac, dtype=float) @ lattice
            if np.linalg.norm(direction) >= 1e-10:
                view_direction = direction

    except (KeyError, ValueError, TypeError, np.linalg.LinAlgError):
        pass

    return {
        "elements": {
            "axes": axes,
            "planes": planes,
            "centers": centers,
        },
        "view_direction_cart": None if view_direction is None else np.asarray(view_direction, dtype=float).tolist(),
    }


def plane_basis_from_normal(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return plane_basis_from_normal_cart(normal)


def check_custom_operation(
    render_data: dict,
    W_frac: np.ndarray,
    t_frac: np.ndarray,
    tolerance_cart: float = 0.1,
) -> dict:
    """
    Apply (W_frac, t_frac) to atoms in the exported unit cell only.
    For each transformed position, check if a same-element atom is within tolerance_cart.
    """
    atoms = [
        atom for atom in render_data.get("atoms", [])
        if atom.get("frac") is not None
    ]
    unit_cell = render_data.get("unit_cell")
    if unit_cell is None:
        return {"error": "No unit cell — molecule mode not supported for custom operation check"}

    lattice = np.asarray(unit_cell["lattice"], dtype=float)
    validity_error = custom_matrix_validity_error(W_frac, t_frac, lattice)
    if validity_error is not None:
        return {"error": validity_error}
    matrix_issue = custom_matrix_symmetry_issue(W_frac, lattice)

    fracs = {}
    for atom in atoms:
        frac = atom.get("frac")
        if frac is not None:
            fracs[atom["index"]] = np.asarray(frac, dtype=float)

    mapped = []
    unmapped = []
    for atom in atoms:
        frac = fracs.get(atom["index"])
        if frac is None:
            continue

        x_prime = W_frac @ frac + t_frac
        # wrap to [0,1)
        x_prime_w = x_prime - np.floor(x_prime + 1e-9)

        best_dist = float("inf")
        best_idx = None
        for other in atoms:
            if other["element"] != atom["element"]:
                continue
            other_frac = fracs.get(other["index"])
            if other_frac is None:
                continue
            delta = x_prime_w - other_frac
            delta -= np.round(delta)
            dist = float(np.linalg.norm(delta @ lattice))
            if dist < best_dist:
                best_dist = dist
                best_idx = other["index"]

        if best_dist <= tolerance_cart:
            mapped.append({
                "source": atom["index"],
                "target": best_idx,
                "element": atom["element"],
                "source_frac": [round(float(v), 4) for v in frac],
                "transformed_frac": [round(float(v), 4) for v in x_prime_w],
                "target_frac": [round(float(v), 4) for v in fracs[best_idx]],
                "distance": round(best_dist, 6),
            })
        else:
            unmapped.append({
                "source": atom["index"],
                "element": atom["element"],
                "source_frac": [round(float(v), 4) for v in frac],
                "transformed_frac": [round(float(v), 4) for v in x_prime_w],
                "frac": [round(float(v), 4) for v in x_prime_w],
                "nearest_atom": best_idx,
                "nearest_frac": (
                    [round(float(v), 4) for v in fracs[best_idx]]
                    if best_idx is not None else None
                ),
                "distance": round(best_dist, 6),
            })

    return {
        "is_symmetry": len(unmapped) == 0 and matrix_issue is None,
        "total": len(atoms),
        "mapped_count": len(mapped),
        "unmapped_count": len(unmapped),
        "mapped": mapped,
        "unmapped": unmapped,
        "tolerance_cart": tolerance_cart,
        "matrix_issue": matrix_issue,
    }


def custom_matrix_validity_error(
    W_frac: np.ndarray,
    t_frac: np.ndarray,
    lattice: np.ndarray,
) -> str | None:
    W_frac = np.asarray(W_frac, dtype=float)
    if W_frac.shape != (3, 3) or not np.all(np.isfinite(W_frac)):
        return "Operation matrix W must be a finite 3x3 matrix"
    t_frac = np.asarray(t_frac, dtype=float)
    if t_frac.shape != (3,) or not np.all(np.isfinite(t_frac)):
        return "Operation translation t must be a finite vector of length 3"
    try:
        np.linalg.inv(lattice.T)
    except np.linalg.LinAlgError:
        return "Unit-cell lattice is singular"
    return None


def custom_matrix_symmetry_issue(W_frac: np.ndarray, lattice: np.ndarray) -> str | None:
    """Explain why a finite affine matrix cannot be a symmetry isometry."""
    W_cart = lattice.T @ np.asarray(W_frac, dtype=float) @ np.linalg.inv(lattice.T)
    determinant = float(np.linalg.det(W_cart))
    if abs(abs(determinant) - 1.0) > 1e-5:
        return "The matrix does not preserve volume (determinant must be ±1)."
    metric = W_cart.T @ W_cart
    if not np.allclose(metric, np.eye(3), atol=1e-5):
        return "The matrix does not preserve distances; scaling/shear is not a symmetry operation."
    return None
