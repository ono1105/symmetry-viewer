import itertools
import numpy as np
import spglib
from pymatgen.core import Structure
from fractions import Fraction
from math import gcd
from functools import reduce
from pathlib import Path


TOL = 1e-8
_INTEGER_OFFSETS_CACHE = {}
_MATRIX_ORDER_CACHE = {}
_NULLSPACE_CACHE = {}
_PINV_CACHE = {}


def array_cache_key(A):
    """
    数値配列を完全一致でキャッシュキー化する。
    丸めずbytesを使うので、近い別行列を誤って同一扱いしない。
    """
    A = np.ascontiguousarray(np.asarray(A))
    return A.shape, A.dtype.str, A.tobytes()


def cached_pinv(A):
    """
    同じ行列に対する擬似逆行列を使い回す。
    """
    A = np.asarray(A, dtype=float)
    key = array_cache_key(A)
    if key not in _PINV_CACHE:
        _PINV_CACHE[key] = np.linalg.pinv(A)
    return _PINV_CACHE[key]


def default_output_path_from_cif(cif_path):
    cif = Path(cif_path)
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    return str(output_dir / cif.with_suffix(".md").name)

def structure_to_spglib_cell(structure: Structure):
    lattice = structure.lattice.matrix
    positions = structure.frac_coords
    numbers = [site.specie.Z for site in structure]
    return lattice, positions, numbers

def wrap_frac(v, tol=1e-8):
    """
    分率座標を [0, 1) に入れる。
    ただし、数値誤差で 0.999999999 になったものは 0 に戻す。
    """
    v = np.asarray(v, dtype=float)
    w = np.mod(v, 1.0)

    # 1に非常に近い値は0とみなす
    w[np.abs(w - 1.0) < tol] = 0.0

    # 0に非常に近い値も0にする
    w[np.abs(w) < tol] = 0.0

    return w

def lcm(a, b):
    return abs(a * b) // gcd(a, b) if a and b else 0

def lcmm(numbers):
    numbers = [abs(int(n)) for n in numbers if int(n) != 0]
    if not numbers:
        return 1
    return reduce(lcm, numbers, 1)

def format_fraction_value(x, max_denominator=24, tol=1e-7):
    """
    floatを見やすい分数に変換する。
    ただし、近い分数が見つからない場合は小数表示にする。
    """
    x = float(x)

    if abs(x) < tol:
        return "0"

    if abs(x - round(x)) < tol:
        return str(int(round(x)))

    frac = Fraction(x).limit_denominator(max_denominator)

    if abs(float(frac) - x) < tol:
        if frac.denominator == 1:
            return str(frac.numerator)
        return f"{frac.numerator}/{frac.denominator}"

    # 分数に無理にしない
    return f"{x:.6f}".rstrip("0").rstrip(".")

def format_frac_vector(v, max_denominator=24, tol=1e-7, wrap=True):
    """
    分率座標ベクトルを [1/3 5/6 0] のように表示する。
    wrap=True のときは [0,1) に入れて表示する。
    """
    if v is None:
        return "None"

    v = np.asarray(v, dtype=float)

    if wrap:
        v = wrap_frac(v, tol=tol)

    parts = [
        format_fraction_value(x, max_denominator=max_denominator, tol=tol)
        for x in v
    ]

    return "[" + " ".join(parts) + "]"

def format_coeff_vector(v, max_denominator=24, tol=1e-7):
    """
    motion_coeff用。これは [0,1) にwrapしない。
    """
    if v is None:
        return "None"

    v = np.asarray(v, dtype=float)

    parts = [
        format_fraction_value(x, max_denominator=max_denominator, tol=tol)
        for x in v
    ]

    return "[" + " ".join(parts) + "]"

def format_direction_vector(v, max_denominator=24, tol=1e-6):
    """
    軸方向・面法線を比として見やすく表示する。

    例:
      [0.707106 0.707106 0] → [1 1 0]
      [0.447214 0.894427 0] → [1 2 0]

    注意:
      計算には使わず、表示専用。
    """
    if v is None:
        return "None"

    v = np.asarray(v, dtype=float)

    if np.linalg.norm(v) < tol:
        return "[0 0 0]"

    # まず小さい成分を0にする
    v = v.copy()
    v[np.abs(v) < tol] = 0.0

    fracs = []
    for x in v:
        if abs(x) < tol:
            fracs.append(Fraction(0))
        else:
            fracs.append(Fraction(float(x)).limit_denominator(max_denominator))

    den_lcm = lcmm([f.denominator for f in fracs])
    ints = [int(f * den_lcm) for f in fracs]

    # 最大公約数で割る
    nonzero = [abs(i) for i in ints if i != 0]
    if nonzero:
        g = reduce(gcd, nonzero)
        ints = [i // g for i in ints]

    # 最初の非ゼロ成分を正にする
    for i in ints:
        if i != 0:
            if i < 0:
                ints = [-j for j in ints]
            break

    return "[" + " ".join(str(i) for i in ints) + "]"

def format_vector(v, precision=6):
    if v is None:
        return "None"

    v = np.asarray(v, dtype=float)
    v[np.abs(v) < 1e-10] = 0.0
    return np.array2string(v, precision=precision, suppress_small=True)

def is_zero_mod1(v, tol=TOL):
    """
    分率座標ベクトルが格子並進を除いて0か判定する。
    """
    v = np.asarray(v, dtype=float)
    wrapped = v - np.round(v)
    return np.linalg.norm(wrapped) < tol

def matrix_order(W, max_order=12):
    """
    W^n = I となる最小の n を返す。
    """
    W = np.asarray(W, dtype=int)
    key = tuple(W.flatten().tolist()), max_order
    if key in _MATRIX_ORDER_CACHE:
        return _MATRIX_ORDER_CACHE[key]

    I = np.eye(3, dtype=int)

    power = np.eye(3, dtype=int)
    for n in range(1, max_order + 1):
        power = power @ W
        if np.array_equal(power, I):
            _MATRIX_ORDER_CACHE[key] = n
            return n

    _MATRIX_ORDER_CACHE[key] = None
    return None

def rotation_angle_deg(W):
    """
    det(W)=+1 の回転角を trace から求める。
    trace(W) = 1 + 2 cos(theta)
    """
    W = np.asarray(W, dtype=float)
    tr = np.trace(W)
    cos_theta = (tr - 1.0) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))

def nullspace(A, tol=1e-10):
    """
    A x = 0 の解空間の基底を列ベクトルとして返す。
    返り値は shape = (3, k)
    """
    A = np.asarray(A, dtype=float)
    key = array_cache_key(A), tol
    if key in _NULLSPACE_CACHE:
        return _NULLSPACE_CACHE[key].copy()

    u, s, vh = np.linalg.svd(A)

    rank = np.sum(s > tol)
    ns = vh[rank:].T
    ns[np.abs(ns) < 1e-12] = 0.0
    _NULLSPACE_CACHE[key] = ns
    return ns.copy()

def normalize_vector(v):
    """
    ベクトルを長さ1に正規化する。
    """
    v = np.asarray(v, dtype=float)
    norm = np.linalg.norm(v)

    if norm < TOL:
        return v

    v = v / norm

    # 表示を安定させるため、最初の非ゼロ成分を正にする
    for x in v:
        if abs(x) > 1e-8:
            if x < 0:
                v = -v
            break

    v[np.abs(v) < 1e-12] = 0.0
    return v

def integer_offsets(search_range):
    """
    周期境界の整数シフト候補をまとめて返す。
    search_rangeは変えず、同じ候補配列を使い回して計算量を抑える。
    """
    if search_range not in _INTEGER_OFFSETS_CACHE:
        _INTEGER_OFFSETS_CACHE[search_range] = np.array(
            list(itertools.product(
                range(-search_range, search_range + 1),
                repeat=3,
            )),
            dtype=float,
        )

    return _INTEGER_OFFSETS_CACHE[search_range]

def find_fixed_solutions(W, t, search_range=2, tol=1e-7):
    """
    固定点条件を満たす点を全列挙する。

        x = W x + t  mod 1

    つまり、

        (I - W)x = t + n

    を整数ベクトル n を変えながら解く。
    """
    W = np.asarray(W, dtype=float)
    t = np.asarray(t, dtype=float)

    A = np.eye(3) - W
    offsets = integer_offsets(search_range)
    rhs = offsets + t

    # Aは同じなので、125回lstsqする代わりに擬似逆行列を1回だけ作る。
    pinv = cached_pinv(A)
    solutions = rhs @ pinv.T
    residuals = solutions @ A.T - rhs
    valid = np.linalg.norm(residuals, axis=1) < tol
    points = wrap_frac(solutions[valid])

    return deduplicate_points(points, tol=tol)

def find_one_fixed_solution(W, t, search_range=2, tol=1e-7):
    """
    固定点が存在するかだけを調べる。
    """
    W = np.asarray(W, dtype=float)
    t = np.asarray(t, dtype=float)

    A = np.eye(3) - W
    rhs = integer_offsets(search_range) + t

    pinv = cached_pinv(A)
    solutions = rhs @ pinv.T
    residuals = solutions @ A.T - rhs
    valid_indices = np.flatnonzero(np.linalg.norm(residuals, axis=1) < tol)

    if len(valid_indices) == 0:
        return False, None

    return True, wrap_frac(solutions[valid_indices[0]])

def find_invariant_element_points(W, t, invariant_basis, search_range=2, tol=1e-7):
    """
    軸・面などの不変集合上の代表点を全列挙する。

    条件:

        W x + t = x + V a + n

    整理して、

        (I - W)x + V a = t + n

    を解く。

    V は、
      軸の場合: 軸方向ベクトル 1本
      面の場合: 面内方向ベクトル 2本
    """
    W = np.asarray(W, dtype=float)
    t = np.asarray(t, dtype=float)
    V = np.asarray(invariant_basis, dtype=float)

    if V.ndim == 1:
        V = V.reshape(3, 1)

    A = np.eye(3) - W

    # 未知数は [x1, x2, x3, a1, a2, ...]
    M = np.hstack([A, V])
    offsets = integer_offsets(search_range)
    rhs = offsets + t

    # Mは候補nに依らないため、まとめて最小二乗解を評価する。
    pinv = cached_pinv(M)
    solutions = rhs @ pinv.T
    residuals = solutions @ M.T - rhs
    valid_indices = np.flatnonzero(np.linalg.norm(residuals, axis=1) < tol)

    candidates = []
    for idx in valid_indices:
        sol = solutions[idx]
        candidates.append((wrap_frac(sol[:3]), sol[3:]))

    return candidates

def deduplicate_points(points, tol=1e-7):
    """
    分率座標の点を mod 1 で重複除去する。
    """
    unique = []

    for p in points:
        p = wrap_frac(p)

        duplicate = False
        for q in unique:
            diff = p - q
            diff = diff - np.round(diff)
            if np.linalg.norm(diff) < tol:
                duplicate = True
                break

        if not duplicate:
            unique.append(p)

    return unique

def same_affine_subspace_mod_lattice(p1, basis1, p2, basis2, search_range=2, tol=1e-7):
    """
    2つの軸または面が、周期境界込みで同じか判定する。

    軸:
        p1 + span(V)
        p2 + span(V)

    面:
        p1 + span(V1, V2)
        p2 + span(V1, V2)

    同じである条件は、

        p2 - p1 = V a + n

    を満たす a と整数ベクトル n が存在すること。
    """
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)

    V = np.asarray(basis1, dtype=float)
    if V.ndim == 1:
        V = V.reshape(3, 1)

    rhs = integer_offsets(search_range) + (p2 - p1)

    pinv = cached_pinv(V)
    coeffs = rhs @ pinv.T
    residuals = coeffs @ V.T - rhs

    return np.any(np.linalg.norm(residuals, axis=1) < tol)

def deduplicate_axes(axes, search_range=2, tol=1e-7):
    """
    軸の重複除去。

    axes の要素:
        {
          "direction": v,
          "point": p,
          "motion_coeff": coeff
        }
    """
    unique = []

    for axis in axes:
        p = axis["point"]
        v = axis["direction"]

        duplicate = False

        for other in unique:
            p2 = other["point"]
            v2 = other["direction"]

            # 方向が平行か確認
            cross = np.cross(v, v2)
            if np.linalg.norm(cross) > tol:
                continue

            # 同じ直線か確認
            if same_affine_subspace_mod_lattice(
                p,
                v.reshape(3, 1),
                p2,
                v2.reshape(3, 1),
                search_range=search_range,
                tol=tol,
            ):
                duplicate = True
                break

        if not duplicate:
            unique.append(axis)

    return unique

def deduplicate_planes(planes, search_range=2, tol=1e-7):
    """
    面の重複除去。

    planes の要素:
        {
          "normal": n,
          "basis": V,
          "point": p,
          "motion_coeff": coeff
        }
    """
    unique = []

    for plane in planes:
        p = plane["point"]
        normal = plane["normal"]
        basis = plane["basis"]

        duplicate = False

        for other in unique:
            p2 = other["point"]
            normal2 = other["normal"]
            basis2 = other["basis"]

            # 法線が平行か確認
            cross = np.cross(normal, normal2)
            if np.linalg.norm(cross) > tol:
                continue

            # 同じ平面か確認
            if same_affine_subspace_mod_lattice(
                p,
                basis,
                p2,
                basis2,
                search_range=search_range,
                tol=tol,
            ):
                duplicate = True
                break

        if not duplicate:
            unique.append(plane)

    return unique

def classify_operation(W, t, search_range=2):
    """
    W, t から対称操作の種類を分類する。
    """
    W = np.asarray(W, dtype=int)
    t = np.asarray(t, dtype=float)

    I = np.eye(3, dtype=int)
    minus_I = -I

    det = round(np.linalg.det(W))
    tr = int(np.trace(W))
    order = matrix_order(W)

    if np.array_equal(W, I):
        if is_zero_mod1(t):
            return "identity", det, tr, order, 0.0
        else:
            return "pure_translation_or_centering_translation", det, tr, order, 0.0

    if np.array_equal(W, minus_I):
        return "inversion", det, tr, order, None

    if det == 1:
        angle = rotation_angle_deg(W)
        fixed_exists, _ = find_one_fixed_solution(W, t, search_range=search_range)

        if fixed_exists:
            return f"rotation_{order}", det, tr, order, angle
        else:
            return f"screw_{order}", det, tr, order, angle

    if det == -1:
        # 鏡映・映進: 固有値が 1, 1, -1 のタイプ
        if order == 2 and tr == 1:
            fixed_exists, _ = find_one_fixed_solution(W, t, search_range=search_range)

            if fixed_exists:
                return "mirror", det, tr, order, None
            else:
                return "glide", det, tr, order, None

        return f"rotoinversion_or_improper_{order}", det, tr, order, None

    return "unknown", det, tr, order, None

def analyze_all_geometry(W, t, op_type, search_range=2, tol=1e-7):
    """
    1つの対称操作について、単位格子内に現れる
    軸・面・中心の候補を全列挙する。
    """
    W = np.asarray(W, dtype=int)
    t = np.asarray(t, dtype=float)

    result = {
        "centers": [],
        "axes": [],
        "planes": [],
    }

    # 反転中心
    if op_type == "inversion":
        centers = find_fixed_solutions(W, t, search_range=search_range, tol=tol)
        result["centers"] = centers
        return result

    # 回転軸・らせん軸
    if op_type.startswith("rotation_") or op_type.startswith("screw_"):
        axis_basis = nullspace(W - np.eye(3))

        if axis_basis.shape[1] >= 1:
            axis_direction = normalize_vector(axis_basis[:, 0])

            candidates = find_invariant_element_points(
                W,
                t,
                axis_direction.reshape(3, 1),
                search_range=search_range,
                tol=tol,
            )

            axes = []
            for point, coeff in candidates:
                axes.append({
                    "direction": axis_direction,
                    "point": point,
                    "motion_coeff": coeff,
                })

            result["axes"] = deduplicate_axes(
                axes,
                search_range=search_range,
                tol=tol,
            )

        return result

    # 鏡映面・映進面
    if op_type in ["mirror", "glide"]:
        plane_basis = nullspace(W - np.eye(3))
        normal_basis = nullspace(W + np.eye(3))

        if plane_basis.shape[1] >= 2 and normal_basis.shape[1] >= 1:
            plane_normal = normalize_vector(normal_basis[:, 0])

            candidates = find_invariant_element_points(
                W,
                t,
                plane_basis,
                search_range=search_range,
                tol=tol,
            )

            planes = []
            for point, coeff in candidates:
                planes.append({
                    "normal": plane_normal,
                    "basis": plane_basis,
                    "point": point,
                    "motion_coeff": coeff,
                })

            result["planes"] = deduplicate_planes(
                planes,
                search_range=search_range,
                tol=tol,
            )

        return result

    # 回反・不正回転系：今回は固定点を中心として列挙する
    if op_type.startswith("rotoinversion_or_improper"):
        centers = find_fixed_solutions(W, t, search_range=search_range, tol=tol)
        result["centers"] = centers
        return result

    return result

def same_point_mod_lattice(p1, p2, tol=1e-7):
    """
    2点が周期境界込みで同じか判定する。
    """
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)

    diff = p1 - p2
    diff = diff - np.round(diff)

    return np.linalg.norm(diff) < tol

def add_or_merge_center(merged_centers, center, op_info, tol=1e-7):
    """
    反転中心・回反中心などを全操作で統合する。
    """
    center = wrap_frac(center)

    for item in merged_centers:
        if same_point_mod_lattice(center, item["point"], tol=tol):
            item["operations"].append(op_info)
            return

    merged_centers.append({
        "point": center,
        "operations": [op_info],
    })

def add_or_merge_axis(merged_axes, axis, op_info, search_range=2, tol=1e-7):
    """
    軸を全操作で統合する。
    """
    p = axis["point"]
    v = axis["direction"]

    for item in merged_axes:
        p2 = item["point"]
        v2 = item["direction"]

        # 方向が平行か
        if np.linalg.norm(np.cross(v, v2)) > tol:
            continue

        # 同じ直線か
        if same_affine_subspace_mod_lattice(
            p,
            v.reshape(3, 1),
            p2,
            v2.reshape(3, 1),
            search_range=search_range,
            tol=tol,
        ):
            item["operations"].append(op_info)
            item["motion_coeffs"].append(axis.get("motion_coeff"))
            return

    merged_axes.append({
        "point": p,
        "direction": v,
        "operations": [op_info],
        "motion_coeffs": [axis.get("motion_coeff")],
    })

def add_or_merge_plane(merged_planes, plane, op_info, search_range=2, tol=1e-7):
    """
    面を全操作で統合する。
    """
    p = plane["point"]
    normal = plane["normal"]
    basis = plane["basis"]

    for item in merged_planes:
        p2 = item["point"]
        normal2 = item["normal"]
        basis2 = item["basis"]

        # 法線が平行か
        if np.linalg.norm(np.cross(normal, normal2)) > tol:
            continue

        # 同じ平面か
        if same_affine_subspace_mod_lattice(
            p,
            basis,
            p2,
            basis2,
            search_range=search_range,
            tol=tol,
        ):
            item["operations"].append(op_info)
            item["motion_coeffs"].append(plane.get("motion_coeff"))
            return

    merged_planes.append({
        "point": p,
        "normal": normal,
        "basis": basis,
        "operations": [op_info],
        "motion_coeffs": [plane.get("motion_coeff")],
    })

def collect_merged_elements(rotations, translations, search_range=2, tol=1e-7):
    """
    全対称操作から得られる軸・面・中心を統合して返す。
    """
    merged = {
        "centers": [],
        "axes": [],
        "planes": [],
    }

    per_operation = []

    for i, (W, t) in enumerate(zip(rotations, translations)):
        op_type, det, tr, order, angle = classify_operation(
            W,
            t,
            search_range=search_range,
        )

        geometry = analyze_all_geometry(
            W,
            t,
            op_type,
            search_range=search_range,
            tol=tol,
        )

        op_info = {
            "index": i,
            "type": op_type,
            "order": order,
            "det": det,
            "trace": tr,
            "angle": angle,
        }

        per_operation.append({
            "operation": op_info,
            "geometry": geometry,
        })

        for center in geometry["centers"]:
            add_or_merge_center(
                merged["centers"],
                center,
                op_info,
                tol=tol,
            )

        for axis in geometry["axes"]:
            add_or_merge_axis(
                merged["axes"],
                axis,
                op_info,
                search_range=search_range,
                tol=tol,
            )

        for plane in geometry["planes"]:
            add_or_merge_plane(
                merged["planes"],
                plane,
                op_info,
                search_range=search_range,
                tol=tol,
            )

    return merged, per_operation

def summarize_operation_types(operations):
    """
    統合された軸・面・中心に対応する操作タイプを簡単にまとめる。
    """
    counts = {}

    for op in operations:
        typ = op["type"]
        counts[typ] = counts.get(typ, 0) + 1

    parts = []
    for typ, count in sorted(counts.items()):
        parts.append(f"{typ}×{count}")

    return ", ".join(parts)

def improper_symbol_from_order_trace(order, trace):
    """
    det(W)=-1 の不正操作を国際記号に寄せて表す。
    -3 は行列としては6回で単位元へ戻るため、orderだけでは区別しない。
    """
    if order == 4:
        return "-4"
    if order == 6:
        if trace == 0:
            return "-3"
        if trace == -2:
            return "-6"
        return "-?"
    return f"-{order}" if order is not None else "improper"

def integer_direction_from_vector(v, max_denominator=24, tol=1e-6):
    """
    正規化された軸方向・面法線を、表示用の整数比ベクトルに戻す。
    """
    v = np.asarray(v, dtype=float)

    if np.linalg.norm(v) < tol:
        return np.zeros(3, dtype=int)

    v = v.copy()
    v[np.abs(v) < tol] = 0.0

    fracs = []
    for x in v:
        if abs(x) < tol:
            fracs.append(Fraction(0))
        else:
            fracs.append(Fraction(float(x)).limit_denominator(max_denominator))

    den_lcm = lcmm([f.denominator for f in fracs])
    ints = np.array([int(f * den_lcm) for f in fracs], dtype=int)

    nonzero = [abs(i) for i in ints if i != 0]
    if nonzero:
        g = reduce(gcd, nonzero)
        ints = ints // g

    for i in ints:
        if i != 0:
            if i < 0:
                ints = -ints
            break

    return ints

def screw_symbol_from_axis(axis, op, tol=1e-6):
    """
    らせん軸の国際記号を motion_coeff から推定する。
    例: screw_4 + 1/4並進 -> 4_1
    """
    order = op.get("order")
    if order is None:
        return None

    coeffs = axis.get("motion_coeffs", [])
    operations = axis.get("operations", [])
    coeff = None

    for axis_op, axis_coeff in zip(operations, coeffs):
        if axis_op["index"] == op["index"]:
            coeff = axis_coeff
            break

    if coeff is None or len(coeff) == 0:
        return None

    direction_int = integer_direction_from_vector(axis["direction"])
    period = np.linalg.norm(direction_int.astype(float))
    if period < tol:
        return None

    fraction = float(coeff[0]) / period
    fraction = fraction - np.floor(fraction)
    screw_index = int(round(fraction * order)) % order

    if abs(fraction - screw_index / order) > tol:
        return None

    if screw_index == 0:
        return str(order)

    return f"{order}_{screw_index}"

def operation_international_symbol(op, element=None):
    """
    内部typeに対応する国際記号風の短い表示を返す。
    """
    typ = op["type"]
    order = op.get("order")

    if typ == "identity":
        return "1"
    if typ == "pure_translation_or_centering_translation":
        return "translation"
    if typ == "inversion":
        return "-1"
    if typ == "mirror":
        return "m"
    if typ == "glide":
        return "g"
    if typ.startswith("rotation_"):
        return str(order)
    if typ.startswith("screw_"):
        if element is not None:
            symbol = screw_symbol_from_axis(element, op)
            if symbol is not None:
                return symbol
        return f"{order}_?"
    if typ.startswith("rotoinversion_or_improper"):
        return improper_symbol_from_order_trace(order, op.get("trace"))

    return typ


def describe_international_symbol(symbol):
    """
    国際表記の短い意味を返す。
    """
    if symbol == "1":
        return "identity"
    if symbol == "-1":
        return "inversion center"
    if symbol == "m":
        return "mirror plane"
    if symbol == "g":
        return "glide plane"
    if symbol == "translation":
        return "translation"
    if symbol in ["2", "3", "4", "6"]:
        return "rotation axis"
    if symbol.startswith(("2_", "3_", "4_", "6_")):
        return "screw axis"
    if symbol in ["-3", "-4", "-6"] or symbol.startswith("-?"):
        return "rotoinversion"
    if symbol.endswith("_?"):
        return "screw axis"

    return "operation"


def operation_international_label(op, element=None):
    """
    operationに対応する国際表記と意味をまとめて返す。
    """
    symbol = operation_international_symbol(op, element=element)
    return f"{symbol} {describe_international_symbol(symbol)}"


def operation_international_label_for_group(op, axes, planes, centers):
    """
    W group内の幾何要素を使って、operationの国際表記ラベルを返す。
    """
    symbol = operation_symbol_for_group(op, axes, planes, centers)
    return f"{symbol} {describe_international_symbol(symbol)}"

def point_group_symbol_from_W(W):
    """
    並進tを無視して、Wが点群でどの操作に対応するかを返す。
    """
    W = np.asarray(W, dtype=int)
    I = np.eye(3, dtype=int)

    det = round(np.linalg.det(W))
    tr = int(np.trace(W))
    order = matrix_order(W)

    if np.array_equal(W, I):
        return "1"
    if np.array_equal(W, -I):
        return "-1"

    if det == 1:
        return str(order)

    if det == -1:
        if order == 2 and tr == 1:
            return "m"
        return improper_symbol_from_order_trace(order, tr)

    return "?"

def point_group_direction_from_W(W, symbol):
    """
    点群操作の軸方向または鏡映面法線をWから求める。
    """
    W = np.asarray(W, dtype=float)

    if symbol in ["1", "-1", "?"]:
        return None

    if symbol == "m" or symbol.startswith("-"):
        basis = nullspace(W + np.eye(3))
    else:
        basis = nullspace(W - np.eye(3))

    if basis.shape[1] == 0:
        return None

    return normalize_vector(basis[:, 0])

def point_group_operation_label(W):
    """
    Wに対応する点群操作を、人間が読みやすい短い説明にする。
    """
    symbol = point_group_symbol_from_W(W)
    direction = point_group_direction_from_W(W, symbol)
    direction_text = format_direction_vector(direction) if direction is not None else None

    if symbol == "1":
        return "1 identity"
    if symbol == "-1":
        return "-1 inversion"
    if symbol == "m":
        return f"m mirror, normal {direction_text}"
    if symbol in ["2", "3", "4", "6"]:
        return f"{symbol} rotation, axis {direction_text}"
    if symbol in ["-3", "-4", "-6"]:
        return f"{symbol} rotoinversion, axis {direction_text}"

    return f"{symbol} unknown"

def summarize_operation_symbols(operations, element=None):
    counts = {}

    for op in operations:
        symbol = operation_international_symbol(op, element=element)
        counts[symbol] = counts.get(symbol, 0) + 1

    parts = []
    for symbol, count in sorted(counts.items()):
        parts.append(f"{symbol}×{count}")

    return ", ".join(parts)

def operation_indices(operations, max_show=12):
    """
    操作番号を表示する。
    長すぎるときは途中で省略。
    """
    indices = [op["index"] for op in operations]

    if len(indices) <= max_show:
        return str(indices)

    return str(indices[:max_show])[:-1] + ", ...]"

def format_matrix_for_markdown(M):
    """
    行列をMarkdown内のコードブロック用文字列にする。
    """
    M = np.asarray(M)
    lines = []
    for row in M:
        lines.append("[" + " ".join(f"{int(x):2d}" for x in row) + "]")
    return "\n".join(lines)

def format_translation_for_markdown(t):
    """
    並進ベクトルを分数表記で表示する。
    """
    return format_frac_vector(t)

def matrix_key(W):
    """
    Wを辞書キーにするため、tuple化する。
    """
    W = np.asarray(W, dtype=int)
    return tuple(W.flatten().tolist())

def add_unique_operation(operations, op):
    """
    operationをindexで重複除去しながら追加する。
    """
    if op["index"] not in [item["index"] for item in operations]:
        operations.append(op)

def add_unique_element(elements, element):
    """
    軸・面・中心の辞書を同一オブジェクトとして重複除去しながら追加する。
    """
    if id(element) not in [id(item) for item in elements]:
        elements.append(element)

def ensure_W_bucket(parent, op, rotations):
    """
    ある中心・軸方向・面法線の下にWごとの入れ物を作る。
    """
    idx = op["index"]
    W = rotations[idx]
    key = matrix_key(W)

    if key not in parent["W_groups"]:
        parent["W_groups"][key] = {
            "key": key,
            "W": W,
            "operations": [],
            "centers": [],
            "axes": [],
            "planes": [],
        }

    bucket = parent["W_groups"][key]
    add_unique_operation(bucket["operations"], op)
    return bucket

def sorted_W_buckets(parent):
    buckets = list(parent["W_groups"].values())
    for bucket in buckets:
        bucket["operations"].sort(key=lambda op: op["index"])
    buckets.sort(key=lambda bucket: min(op["index"] for op in bucket["operations"]))
    return buckets

def group_elements_by_geometry(merged, rotations):
    """
    Wではなく、幾何要素を主語にしてまとめる。

    centers: 中心座標 -> W
    axes:    軸方向 -> W -> 通過点
    planes:  法線方向 -> W -> 代表点
    """
    groups = {
        "centers": [],
        "axes": [],
        "planes": [],
    }
    center_by_key = {}
    axis_by_key = {}
    plane_by_key = {}

    for center in merged["centers"]:
        key = format_frac_vector(center["point"])
        if key not in center_by_key:
            center_by_key[key] = {
                "key": key,
                "point": center["point"],
                "W_groups": {},
            }
            groups["centers"].append(center_by_key[key])

        for op in center["operations"]:
            bucket = ensure_W_bucket(center_by_key[key], op, rotations)
            add_unique_element(bucket["centers"], center)

    for axis in merged["axes"]:
        key = format_direction_vector(axis["direction"])
        if key not in axis_by_key:
            axis_by_key[key] = {
                "key": key,
                "direction": axis["direction"],
                "W_groups": {},
            }
            groups["axes"].append(axis_by_key[key])

        for op in axis["operations"]:
            bucket = ensure_W_bucket(axis_by_key[key], op, rotations)
            add_unique_element(bucket["axes"], axis)

    for plane in merged["planes"]:
        key = format_direction_vector(plane["normal"])
        if key not in plane_by_key:
            plane_by_key[key] = {
                "key": key,
                "normal": plane["normal"],
                "W_groups": {},
            }
            groups["planes"].append(plane_by_key[key])

        for op in plane["operations"]:
            bucket = ensure_W_bucket(plane_by_key[key], op, rotations)
            add_unique_element(bucket["planes"], plane)

    groups["centers"].sort(key=lambda item: item["key"])
    groups["axes"].sort(key=lambda item: item["key"])
    groups["planes"].sort(key=lambda item: item["key"])
    return groups

def motion_coeff_text_for_operations(element, operations):
    """
    軸・面に保存されたmotion_coeffsから、表示中のoperation分だけ抜き出す。
    """
    coeffs = element.get("motion_coeffs", [])
    if not coeffs:
        return ""

    operation_indices_set = {op["index"] for op in operations}
    shown = []
    for op, coeff in zip(element.get("operations", []), coeffs):
        if op["index"] not in operation_indices_set or coeff is None:
            continue
        text = format_coeff_vector(coeff)
        if text not in shown:
            shown.append(text)

    return ", ".join(shown)

def element_for_operation(op, elements):
    """
    W group内で、operationに対応する軸・面・中心を1つ返す。
    """
    for element in elements:
        for element_op in element.get("operations", []):
            if element_op["index"] == op["index"]:
                return element

    return None

def operation_symbol_for_group(op, axes, planes, centers):
    """
    W group内の幾何要素を使って、operationの表示記号を返す。
    """
    element = None

    if op["type"].startswith("screw_") or op["type"].startswith("rotation_"):
        element = element_for_operation(op, axes)
    elif op["type"] in ["mirror", "glide"]:
        element = element_for_operation(op, planes)
    elif op["type"] == "inversion" or op["type"].startswith("rotoinversion_or_improper"):
        element = element_for_operation(op, centers)

    return operation_international_symbol(op, element=element)

def summarize_operation_symbols_for_group(operations, axes, planes, centers):
    counts = {}

    for op in operations:
        symbol = operation_symbol_for_group(op, axes, planes, centers)
        counts[symbol] = counts.get(symbol, 0) + 1

    parts = []
    for symbol, count in sorted(counts.items()):
        parts.append(f"{symbol}×{count}")

    return ", ".join(parts)

def group_elements_by_W(merged, rotations):
    """
    統合済みの centers / axes / planes を W ごとにまとめる。
    同じ要素が複数のW groupに出ることは許容する。
    """
    groups = {}

    def ensure_group(op):
        idx = op["index"]
        W = rotations[idx]
        key = matrix_key(W)

        if key not in groups:
            groups[key] = {
                "key": key,
                "W": W,
                "operations": [],
                "centers": [],
                "axes": [],
                "planes": [],
            }

        if idx not in [x["index"] for x in groups[key]["operations"]]:
            groups[key]["operations"].append(op)

        return groups[key]

    for center in merged["centers"]:
        for op in center["operations"]:
            add_unique_element(ensure_group(op)["centers"], center)

    for axis in merged["axes"]:
        for op in axis["operations"]:
            add_unique_element(ensure_group(op)["axes"], axis)

    for plane in merged["planes"]:
        for op in plane["operations"]:
            add_unique_element(ensure_group(op)["planes"], plane)

    result = list(groups.values())

    for group in result:
        group["operations"].sort(key=lambda op: op["index"])

    result.sort(key=lambda group: min(op["index"] for op in group["operations"]))
    return result
