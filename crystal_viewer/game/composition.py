"""Judging for the "compose two operations" puzzle (docs/PUZZLE_SPEC.md §7).

Two symmetry operations A and B are animated in turn; the player names the single
operation equal to "A then B" (the product B∘A). The classic insight — two mirror
planes compose to a rotation, a rotation and a mirror compose to another mirror —
is the point of the quiz.

Renderer-independent: consumes ``render_data`` only. The composition maths is the
same the analysis mode uses (``symmetry_operations``), and the product is named
with the same vocabulary as the operation-identify quiz (``operation_identify``),
so judging stays single-source.

Scope: A, B and the product are all *nameable moving point operations* (rotation /
mirror / inversion / rotoreflection / rotoinversion). Screw/glide products carry a
translation and are out of scope for this quiz, so pairs whose product is not a
clean point operation are dropped. The product must also differ from both A and B
(otherwise the "combination" is trivial).
"""

from __future__ import annotations

import numpy as np

from crystal_viewer.symmetry_operations import (
    compose_operation_sequence,
    find_matching_operation_index,
)
from crystal_viewer.viewer.operation_lookup import operation_by_index

# Reuse the operation-identify judging so the product is named identically and the
# motion/answer filters stay in one place.
from crystal_viewer.game.operation_identify import (
    _answer_matches,
    _moves_any_atom,
    _operation_answer,
    _revealed_answers,
    _visual_signature,
)


# Keep the public list small and varied on highly symmetric crystals.  Pair
# enumeration must see every visually distinct operand: truncating each operand
# kind before composing silently drops valid product orders (for example C3 from
# two mirrors in Oh).  The cap is therefore applied only after products and
# operand strata have been identified.
_MAX_PER_PRODUCT = 8

# Molecular point-group matrices exported by PointGroupAnalyzer are not always
# exact group representations (water is off by about 1.6e-3).  Match the nearest
# Cartesian group element within a conservative tolerance and require the runner
# up to be clearly separated so relaxing the tolerance cannot select by order.
_MOLECULE_MATCH_TOL = 1e-2
_MOLECULE_MATCH_SEPARATION = 0.1


def _answer_key(answer: dict) -> tuple:
    return (answer["kind"], -1 if answer["order"] is None else int(answer["order"]))


def _basis_operations(render_data: dict, atoms: list[dict], starts: np.ndarray) -> list[dict]:
    """Visually distinct nameable moving point operations.

    Deduping by visual signature keeps coincident animations together while still
    keeping the two σ planes of e.g. water (whose animations differ) apart.  Do
    not cap here: a representative discarded before multiplication can be the
    only route to a textbook product order.
    """
    representatives: list[dict] = []
    seen: set = set()
    for operation in render_data.get("operations", []):
        answer = _operation_answer(render_data, operation)
        if answer is None:
            continue
        if not _moves_any_atom(operation, atoms, starts):
            continue
        signature = _visual_signature(operation, atoms)
        if signature is None or signature in seen:
            continue
        seen.add(signature)
        representatives.append(operation)
    return representatives


def _frac_key(matrix, translation) -> tuple | None:
    """A fast hash key for conventional fractional crystallographic operations.

    Standard space-group settings use integer rotation parts and translations on a
    1/24 grid, and this returns those exact integers (translations counted in
    24ths).  An arbitrary origin shift need not preserve the grid, so anything
    off it returns ``None`` rather than being rounded onto a neighbour; callers
    then fall back to an exact group scan.
    """
    if matrix is None or translation is None:
        return None
    w = np.asarray(matrix, dtype=float)
    t = np.asarray(translation, dtype=float)
    w_int = np.rint(w)
    if np.max(np.abs(w - w_int)) > 1e-6:
        return None
    t_mod = t % 1.0
    t24 = np.rint(t_mod * 24.0) % 24
    snap_error = (t_mod - t24 / 24.0 + 0.5) % 1.0 - 0.5
    if np.max(np.abs(snap_error)) > 1e-6:
        return None
    # Plain ints, not numpy scalars: the key is composed with Python arithmetic.
    return (tuple(w_int.flatten().astype(int).tolist()), tuple(t24.astype(int).tolist()))


def _crystal_keys(operations: list[dict]) -> tuple[dict[tuple, int], dict[int, tuple]]:
    """``(key -> index, index -> key)`` for operations on the conventional grid."""
    lookup: dict[tuple, int] = {}
    keys: dict[int, tuple] = {}
    for operation in operations:
        key = _frac_key(operation.get("matrix_frac"), operation.get("translation_frac"))
        if key is None:
            continue
        index = int(operation["index"])
        keys[index] = key
        lookup.setdefault(key, index)
    return lookup, keys


def _compose_frac_keys(first_key: tuple, second_key: tuple) -> tuple:
    """Key of "first then second" (the product B∘A), in exact integer arithmetic.

    Both keys hold an integer rotation and a translation counted in 24ths, and the
    product ``W_B·W_A`` / ``W_B·t_A + t_B`` of such values stays on the same grid.
    Composing the integers directly therefore needs no rounding, so — unlike a
    float compose followed by a snap to the grid — it cannot land a near-miss on a
    neighbouring group element.
    """
    (a, a_t), (b, b_t) = first_key, second_key
    w = (
        b[0] * a[0] + b[1] * a[3] + b[2] * a[6],
        b[0] * a[1] + b[1] * a[4] + b[2] * a[7],
        b[0] * a[2] + b[1] * a[5] + b[2] * a[8],
        b[3] * a[0] + b[4] * a[3] + b[5] * a[6],
        b[3] * a[1] + b[4] * a[4] + b[5] * a[7],
        b[3] * a[2] + b[4] * a[5] + b[5] * a[8],
        b[6] * a[0] + b[7] * a[3] + b[8] * a[6],
        b[6] * a[1] + b[7] * a[4] + b[8] * a[7],
        b[6] * a[2] + b[7] * a[5] + b[8] * a[8],
    )
    t = (
        (b[0] * a_t[0] + b[1] * a_t[1] + b[2] * a_t[2] + b_t[0]) % 24,
        (b[3] * a_t[0] + b[4] * a_t[1] + b[5] * a_t[2] + b_t[1]) % 24,
        (b[6] * a_t[0] + b[7] * a_t[1] + b[8] * a_t[2] + b_t[2]) % 24,
    )
    return (w, t)


def _cartesian_operations(operations: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stacked ``(matrices, translations, indices)`` for Cartesian group matching."""
    usable = [operation for operation in operations if operation.get("matrix_cart") is not None]
    matrices = np.array([operation["matrix_cart"] for operation in usable], dtype=float).reshape(
        len(usable), 3, 3
    )
    translations = np.array(
        [operation.get("translation_cart") or [0.0, 0.0, 0.0] for operation in usable], dtype=float
    ).reshape(len(usable), 3)
    indices = np.array([int(operation["index"]) for operation in usable], dtype=int)
    return matrices, translations, indices


def _product_index(
    first: dict,
    second: dict,
    operations: list[dict],
    *,
    is_crystal: bool,
    crystal_lookup: dict[tuple, int] | None = None,
    crystal_keys: dict[int, tuple] | None = None,
    cartesian_operations: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> int | None:
    """Index of the group operation equal to "first then second", or None.

    Crystals compose fractionally (matching is modulo lattice translations);
    molecules have no lattice — and no fractional matrices — so they compose in
    Cartesian, where point operations sit at the origin (translation ~ 0).
    """
    if is_crystal:
        # Standard Hall-setting operations have integer fractional rotations and
        # 1/24-grid translations, and both operands' keys were computed (and
        # verified to be on the grid) once per structure.  Composing those keys is
        # an O(1) integer step, which keeps a high-symmetry group's ~19k ordered
        # pairs responsive; the generic normalisation below costs ~70x more.
        first_key = (crystal_keys or {}).get(int(first["index"]))
        second_key = (crystal_keys or {}).get(int(second["index"]))
        if crystal_lookup is not None and first_key is not None and second_key is not None:
            key = _compose_frac_keys(first_key, second_key)
            candidate_index = crystal_lookup.get(key)
            # A dict hit means the stored operation's own verified key equals the
            # product's, i.e. equal rotations and translations equal modulo the
            # lattice.  There is nothing further to check, and no tolerance to get
            # wrong, because both sides are exact integers.
            if candidate_index is not None:
                return candidate_index

        # Nonstandard origin choices need not lie on the 1/24 grid, and a missing
        # or colliding fast key must never create a false match.  Fall back to the
        # shared generic composer and exact modulo-lattice group scan.
        try:
            # compose_operation_sequence applies operations[0] then [1].
            product = compose_operation_sequence([first, second])
        except (KeyError, ValueError, np.linalg.LinAlgError):
            return None
        return find_matching_operation_index(product.W, product.t, operations)

    first_matrix = first.get("matrix_cart")
    second_matrix = second.get("matrix_cart")
    if first_matrix is None or second_matrix is None:
        return None
    w_first = np.asarray(first_matrix, dtype=float)
    w_second = np.asarray(second_matrix, dtype=float)
    t_first = np.asarray(first.get("translation_cart") or [0.0, 0.0, 0.0], dtype=float)
    t_second = np.asarray(second.get("translation_cart") or [0.0, 0.0, 0.0], dtype=float)
    product_w = w_second @ w_first
    product_t = w_second @ t_first + t_second
    matrices, translations, indices = (
        cartesian_operations
        if cartesian_operations is not None
        else _cartesian_operations(operations)
    )
    if indices.size == 0:
        return None
    # Error per group element, evaluated on the whole group at once: a molecule
    # pairs every operand with every other, so a Python loop over the group here
    # would dominate the quiz's response time.
    matrix_errors = np.linalg.norm((matrices - product_w).reshape(indices.size, -1), axis=1)
    translation_errors = np.linalg.norm(translations - product_t, axis=1)
    errors = np.maximum(matrix_errors, translation_errors)
    ranked = np.argsort(errors, kind="stable")
    best_error = float(errors[ranked[0]])
    if best_error > _MOLECULE_MATCH_TOL:
        return None
    if ranked.size > 1 and float(errors[ranked[1]]) < best_error + _MOLECULE_MATCH_SEPARATION:
        return None
    return int(indices[ranked[0]])


# One player action asks for the questions twice: the public list on GET and the
# private list again on /check.  A high-symmetry crystal evaluates ~19k ordered
# pairs, so keep the last structure's result.  The viewer replaces render_data
# wholesale when another structure loads and never mutates it in place, so object
# identity is a sound key; holding the dict keeps its id from being reused.
_questions_cache: tuple[dict, list[dict]] | None = None


def composition_questions(render_data: dict) -> list[dict]:
    """One question per ordered pair (A, B) whose product is a nameable point op.

    Each question carries the two operation indices to animate and the product's
    ``answers`` (the name is withheld from the public list until ``check_answer``).

    The returned list is shared with the cache above; callers must treat it, and
    the questions in it, as read-only.
    """
    global _questions_cache
    cached = _questions_cache
    if cached is not None and cached[0] is render_data:
        return cached[1]
    questions = _build_questions(render_data)
    _questions_cache = (render_data, questions)
    return questions


def _build_questions(render_data: dict) -> list[dict]:
    atoms = [atom for atom in render_data.get("atoms", []) if atom.get("cart") is not None]
    if not atoms:
        return []
    starts = np.array([atom["cart"] for atom in atoms], dtype=float)

    operations = render_data.get("operations", [])
    operations_by_index = {int(operation["index"]): operation for operation in operations}
    is_crystal = render_data.get("unit_cell") is not None
    crystal_lookup, crystal_keys = _crystal_keys(operations) if is_crystal else (None, None)
    cartesian_ops = None if is_crystal else _cartesian_operations(operations)
    basis = _basis_operations(render_data, atoms, starts)
    # Answer classification and motion checks involve matrix analysis.  Cache
    # them once per group element instead of repeating them for every ordered
    # pair; this keeps high-symmetry crystals responsive after removing the
    # lossy pre-composition representative cap.
    answer_by_index = {
        index: _operation_answer(render_data, operation)
        for index, operation in operations_by_index.items()
    }
    moves_by_index = {
        index: _moves_any_atom(operation, atoms, starts)
        for index, operation in operations_by_index.items()
    }

    candidates_by_product: dict[tuple, list[tuple[tuple, tuple, dict]]] = {}
    for first in basis:
        first_index = int(first["index"])
        first_answer = answer_by_index.get(first_index)
        if first_answer is None:
            continue
        first_key = _answer_key(first_answer)
        for second in basis:
            second_index = int(second["index"])
            if first_index == second_index:
                continue
            second_answer = answer_by_index.get(second_index)
            if second_answer is None:
                continue
            second_key = _answer_key(second_answer)
            match = _product_index(
                first,
                second,
                operations,
                is_crystal=is_crystal,
                crystal_lookup=crystal_lookup,
                crystal_keys=crystal_keys,
                cartesian_operations=cartesian_ops,
            )
            if match is None or match in (first_index, second_index):
                continue
            product_op = operations_by_index.get(match)
            if product_op is None:
                continue
            answer = answer_by_index.get(match)
            if answer is None or not moves_by_index.get(match, False):
                continue
            key = _answer_key(answer)
            candidates_by_product.setdefault(key, []).append(
                (
                    first_key,
                    second_key,
                    {
                        "operation_index_a": first_index,
                        "operation_index_b": second_index,
                        "product_index": int(match),
                        "answers": [{**answer, "_operation_index": int(match)}],
                    },
                )
            )

    questions: list[dict] = []
    for candidates in candidates_by_product.values():
        questions.extend(_select_product_candidates(candidates))
    return questions


def _select_product_candidates(
    candidates: list[tuple[tuple, tuple, dict]],
) -> list[dict]:
    """Select a cap-sized, operand-diverse sample for one product answer.

    Taking the first N pairs makes high-symmetry structures heavily favour the
    first basis operation (often inversion).  Round-robin selection across
    operand-type pairs keeps textbook combinations such as mirror × mirror in the
    public pool.  Mirror × mirror is placed first so it survives even when a
    product has more operand strata than the cap.
    """
    strata: dict[tuple[tuple, tuple], list[dict]] = {}
    for first_key, second_key, question in candidates:
        strata.setdefault((first_key, second_key), []).append(question)

    def stratum_order(item: tuple[tuple, tuple]) -> tuple:
        mirror_pair = item[0][0] == "mirror" and item[1][0] == "mirror"
        return (0 if mirror_pair else 1, item)

    ordered = [strata[key] for key in sorted(strata, key=stratum_order)]
    selected: list[dict] = []
    offset = 0
    while len(selected) < _MAX_PER_PRODUCT:
        added = False
        for group in ordered:
            if offset < len(group):
                selected.append(group[offset])
                added = True
                if len(selected) >= _MAX_PER_PRODUCT:
                    break
        if not added:
            break
        offset += 1
    return selected


def public_questions(render_data: dict) -> list[dict]:
    """Questions for the client (named operands to animate; product withheld).

    Each question carries an opaque ``group`` id shared by questions with the same
    product name, so the client can sample product types evenly without learning
    the answer.
    """
    group_ids: dict[tuple, int] = {}
    operations = render_data.get("operations", [])
    public = []
    for index, question in enumerate(composition_questions(render_data)):
        signature = tuple(
            sorted(
                (a["kind"], -1 if a["order"] is None else a["order"])
                for a in question["answers"]
            )
        )
        group = group_ids.setdefault(signature, len(group_ids))
        operands = []
        for operation_index in (
            question["operation_index_a"],
            question["operation_index_b"],
        ):
            operation = operation_by_index(operations, operation_index)
            answer = _operation_answer(render_data, operation) if operation else None
            operands.append(
                _revealed_answers(
                    render_data,
                    [{**answer, "_operation_index": operation_index}],
                )[0]
                if answer is not None
                else None
            )
        public.append(
            {
                "id": index,
                "operation_index_a": question["operation_index_a"],
                "operation_index_b": question["operation_index_b"],
                "operation_a": operands[0],
                "operation_b": operands[1],
                "group": group,
            }
        )
    return public


def check_answer(
    render_data: dict,
    question_id: int,
    selected_kind,
    selected_order=None,
) -> dict | None:
    """Judge one answer and reveal the product. ``None`` if the id is unknown."""
    questions = composition_questions(render_data)
    if not 0 <= question_id < len(questions):
        return None
    question = questions[question_id]
    correct = any(
        _answer_matches(answer, selected_kind, selected_order)
        for answer in question["answers"]
    )
    return {
        "correct": correct,
        "answers": _revealed_answers(render_data, question["answers"]),
        "operation_index_a": question["operation_index_a"],
        "operation_index_b": question["operation_index_b"],
        "product_index": question["product_index"],
    }
