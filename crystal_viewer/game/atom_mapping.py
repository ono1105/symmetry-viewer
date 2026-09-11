"""Judging for the "where does this atom go?" puzzle (docs/PUZZLE_SPEC.md §7).

One atom is highlighted and a symmetry operation is indicated; the player predicts
which atom it maps onto and clicks it. This builds the most concrete intuition —
that an operation acts on points — and scaffolds the composition quiz.

Renderer-independent: consumes ``render_data`` only. Molecule-focused: every atom
is displayed and the image of an atom is another displayed atom, so the target is
unambiguous. Crystals are out of scope here (an atom's image can be a periodic
copy outside the drawn cell); such render_data simply yields no clean questions.

The operation is one of the nameable moving point operations (same basis as the
operation-identify / composition quizzes), so judging stays single-source.
"""

from __future__ import annotations

import numpy as np

from crystal_viewer.game.operation_identify import (
    _operation_answer,
    _revealed_answers,
    _transform,
)

# The image of an atom under a symmetry operation lands on another atom; require it
# within this tolerance to count as "landing", and require a clear separation from
# the second-nearest atom so the target is unambiguous.
_LAND_TOL = 1e-2
_SEPARATION_TOL = 0.1

# Keep the list small and varied: a few source atoms per operation type is enough.
_MAX_PER_GROUP = 6


def _answer_key(answer: dict) -> tuple:
    return (answer["kind"], -1 if answer["order"] is None else int(answer["order"]))


def mapping_questions(render_data: dict) -> list[dict]:
    """One question per (operation, source atom) whose image is a distinct atom.

    Each question carries the operation to animate, the atom to highlight and the
    target atom (withheld from the public list until ``check_answer``).
    """
    # A crystal image can be a periodic copy outside the displayed cell.  Until
    # the quiz represents such copies explicitly, do not let a caller bypass the
    # molecule-only picker and receive a geometrically misleading question.
    if render_data.get("unit_cell") is not None:
        return []

    atoms = [atom for atom in render_data.get("atoms", []) if atom.get("cart") is not None]
    if len(atoms) < 2:
        return []
    coords = np.array([atom["cart"] for atom in atoms], dtype=float)
    indices = [int(atom["index"]) for atom in atoms]
    elements = [atom.get("element") for atom in atoms]

    candidates_by_group: dict[tuple, list[tuple[tuple, dict]]] = {}
    for operation in render_data.get("operations", []):
        answer = _operation_answer(render_data, operation)
        if answer is None:
            continue
        images = _transform(operation, atoms)
        if images is None or images.size == 0:
            continue
        key = _answer_key(answer)
        for source in range(len(atoms)):
            distances = np.linalg.norm(coords - images[source], axis=1)
            nearest = int(np.argmin(distances))
            if distances[nearest] > _LAND_TOL:
                continue  # image does not land on any atom
            if indices[nearest] == indices[source]:
                continue  # the operation fixes this atom — no mapping to ask
            if elements[nearest] != elements[source]:
                continue  # a symmetry operation maps like onto like
            second = float(np.partition(distances, 1)[1])
            if second < distances[nearest] + _SEPARATION_TOL:
                continue  # ambiguous — two atoms are about equally close
            stratum = (int(operation["index"]), str(elements[source]))
            candidates_by_group.setdefault(key, []).append(
                (
                    stratum,
                    {
                        "operation_index": int(operation["index"]),
                        "source_atom_index": indices[source],
                        "target_atom_index": indices[nearest],
                    },
                )
            )

    questions: list[dict] = []
    for candidates in candidates_by_group.values():
        questions.extend(_select_group_candidates(candidates))
    return questions


def _select_group_candidates(candidates: list[tuple[tuple, dict]]) -> list[dict]:
    """Sample one answer group across operations and chemical elements."""
    strata: dict[tuple, list[dict]] = {}
    for stratum, question in candidates:
        strata.setdefault(stratum, []).append(question)
    ordered = [strata[key] for key in sorted(strata)]

    selected: list[dict] = []
    offset = 0
    while len(selected) < _MAX_PER_GROUP:
        added = False
        for group in ordered:
            if offset < len(group):
                selected.append(group[offset])
                added = True
                if len(selected) >= _MAX_PER_GROUP:
                    break
        if not added:
            break
        offset += 1
    return selected


def public_questions(render_data: dict) -> list[dict]:
    """Questions for the client (named operation + source atom; target withheld).

    Each question carries an opaque ``group`` id shared by questions with the same
    operation type, so the client can sample operation types evenly.
    """
    group_ids: dict[tuple, int] = {}
    public = []
    for index, question in enumerate(mapping_questions(render_data)):
        # mapping_questions only keeps operations that classify to a named answer,
        # so both the lookup and the classification resolve here.  Do not paper
        # over a None with a placeholder group: the operation is published as part
        # of the question, and a question naming no operation is unanswerable.
        operation = _operation_by_index(render_data, question["operation_index"])
        answer = _operation_answer(render_data, operation)
        group = group_ids.setdefault(_answer_key(answer), len(group_ids))
        public.append(
            {
                "id": index,
                "operation_index": question["operation_index"],
                # The operation is part of the question, not the answer.  A short
                # partial trajectory is not sufficient to distinguish Cn from Sn
                # (or equivalent axes), so publish the same canonical label used
                # by the operation-identify and composition quizzes.
                "operation": _revealed_answers(
                    render_data,
                    [{**answer, "_operation_index": question["operation_index"]}],
                )[0],
                "source_atom_index": question["source_atom_index"],
                "group": group,
            }
        )
    return public


def _operation_by_index(render_data: dict, index: int) -> dict | None:
    for operation in render_data.get("operations", []):
        if int(operation.get("index", -1)) == int(index):
            return operation
    return None


def check_answer(render_data: dict, question_id: int, selected_atom_index) -> dict | None:
    """Judge one answer and reveal the target. ``None`` if the id is unknown."""
    questions = mapping_questions(render_data)
    if not 0 <= question_id < len(questions):
        return None
    question = questions[question_id]
    try:
        selected = int(selected_atom_index)
    except (TypeError, ValueError):
        selected = None
    return {
        "correct": selected == question["target_atom_index"],
        "target_atom_index": question["target_atom_index"],
        "operation_index": question["operation_index"],
        "source_atom_index": question["source_atom_index"],
    }
