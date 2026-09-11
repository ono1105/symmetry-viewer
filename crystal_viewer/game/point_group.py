"""Judging for the "name this point group" puzzle (docs/PUZZLE_SPEC.md).

The structure itself is shown, bare -- no highlighted axis or animation, the same
way the structure picker renders it -- and the player names its overall point
group from a small multiple-choice list. Unlike the other four quizzes, which
each ask about one axis, operation, product or atom, this is a single
capstone question per structure: the whole structure's own symmetry class,
already known from ``render_data["metadata"]["point_group_label"]``. No new
geometry is computed here.

Renderer-independent: consumes ``render_data`` only.

Molecules and crystals use different label vocabularies -- Schoenflies (``Oh``,
``D3h``) for molecules, Hermann-Mauguin/ITC (``m-3m``, ``-6m2``) for crystals --
so a crystal question must not offer a Schoenflies distractor and vice versa.
Both vocabularies are described by the same small table of standard
group-theory facts (order, highest proper rotation, and a notational "shape"
family such as "has a horizontal mirror" vs "has diagonal mirrors"), and
distractors are the closest entries by those facts. This is deliberately not a
hand-curated table of confusable pairs (docs/CLAUDE_HANDOFF.md warns against
hand-written exclusion/pairing lists): classic confusions like D3h vs D3d or
C2v vs C2h fall out on their own because those pairs differ in exactly one of
these properties.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random


@dataclass(frozen=True)
class _PointGroupProps:
    order: int
    principal_order: int  # highest-order proper rotation present; 0 for the axial infinity groups
    family: str  # notational shape, independent of the fold number


_INFINITE_ORDER = 1_000_000

# Schoenflies point groups. Standard group-theory facts (order, highest proper
# rotation, notational family), not project-specific data -- covers every label
# `molecule_analysis.py` can currently produce for the bundled molecules, plus a
# handful of close relatives kept only to give small families (e.g. "cubic") a
# richer distractor pool.
_SCHOENFLIES: dict[str, _PointGroupProps] = {
    "C1": _PointGroupProps(1, 1, "C"),
    "Cs": _PointGroupProps(2, 1, "Cs"),
    "Ci": _PointGroupProps(2, 1, "Ci"),
    "C2": _PointGroupProps(2, 2, "C"),
    "C3": _PointGroupProps(3, 3, "C"),
    "C4": _PointGroupProps(4, 4, "C"),
    "C6": _PointGroupProps(6, 6, "C"),
    "C2v": _PointGroupProps(4, 2, "Cv"),
    "C3v": _PointGroupProps(6, 3, "Cv"),
    "C4v": _PointGroupProps(8, 4, "Cv"),
    "C6v": _PointGroupProps(12, 6, "Cv"),
    "C2h": _PointGroupProps(4, 2, "Ch"),
    "C3h": _PointGroupProps(6, 3, "Ch"),
    "C4h": _PointGroupProps(8, 4, "Ch"),
    "C6h": _PointGroupProps(12, 6, "Ch"),
    "D2": _PointGroupProps(4, 2, "D"),
    "D3": _PointGroupProps(6, 3, "D"),
    "D4": _PointGroupProps(8, 4, "D"),
    "D6": _PointGroupProps(12, 6, "D"),
    "D2h": _PointGroupProps(8, 2, "Dh"),
    "D3h": _PointGroupProps(12, 3, "Dh"),
    "D4h": _PointGroupProps(16, 4, "Dh"),
    "D6h": _PointGroupProps(24, 6, "Dh"),
    "D2d": _PointGroupProps(8, 2, "Dd"),
    "D3d": _PointGroupProps(12, 3, "Dd"),
    "S4": _PointGroupProps(4, 2, "S"),
    "S6": _PointGroupProps(6, 3, "S"),
    "T": _PointGroupProps(12, 3, "cubic"),
    "Td": _PointGroupProps(24, 3, "cubic"),
    "Th": _PointGroupProps(24, 3, "cubic"),
    "O": _PointGroupProps(24, 4, "cubic"),
    "Oh": _PointGroupProps(48, 4, "cubic"),
    "I": _PointGroupProps(60, 5, "icosahedral"),
    "Ih": _PointGroupProps(120, 5, "icosahedral"),
    "C∞v": _PointGroupProps(_INFINITE_ORDER, 0, "Cv"),
    "D∞h": _PointGroupProps(_INFINITE_ORDER, 0, "Dh"),
}

# Crystallographic (Hermann-Mauguin) labels -- the 32 crystal classes -- mapped to
# the Schoenflies entry with the same abstract structure, so a crystal question's
# distractors are scored with the same table but stay in HM notation. Covers every
# label `structure_analysis.py`'s spglib path can produce; a symbol missing here
# (it should not happen -- there are only 32) falls back to no question rather
# than crashing, same as an unrecognised Schoenflies label.
_HM_TO_SCHOENFLIES: dict[str, str] = {
    "1": "C1", "-1": "Ci",
    "2": "C2", "m": "Cs", "2/m": "C2h",
    "222": "D2", "mm2": "C2v", "mmm": "D2h",
    "4": "C4", "-4": "S4", "4/m": "C4h",
    "422": "D4", "4mm": "C4v", "-42m": "D2d", "4/mmm": "D4h",
    "3": "C3", "-3": "S6",
    "32": "D3", "3m": "C3v", "-3m": "D3d",
    "6": "C6", "-6": "C3h", "6/m": "C6h",
    "622": "D6", "6mm": "C6v", "-6m2": "D3h", "6/mmm": "D6h",
    "23": "T", "m-3": "Th",
    "432": "O", "-43m": "Td", "m-3m": "Oh",
}

_CRYSTAL_PROPS: dict[str, _PointGroupProps] = {
    hm: _SCHOENFLIES[schoenflies] for hm, schoenflies in _HM_TO_SCHOENFLIES.items()
}

_DISTRACTOR_COUNT = 3

# Human-readable gloss for each notational family, reusing exactly the facts
# already in _PointGroupProps -- no new geometry, just naming what "family"
# already encodes (see the module docstring). Used to build a short post-answer
# explanation (docs/PUZZLE_SPEC.md).
_FAMILY_DESCRIPTIONS: dict[str, str] = {
    "C": "回転軸だけを持つ",
    "Cs": "鏡映面を1枚持つだけ",
    "Ci": "反転中心だけを持つ",
    "Cv": "主軸を含む鏡映面（縦の鏡映面）を持つ",
    "Ch": "主軸に垂直な鏡映面（水平の鏡映面）を持つ",
    "D": "主軸に垂直な2回軸を持つ",
    "Dh": "主軸に垂直な2回軸と、水平の鏡映面を持つ",
    "Dd": "主軸に垂直な2回軸と、対角の鏡映面を持つ",
    "S": "回映軸だけを持つ",
    "cubic": "複数の高次回転軸を持つ立方対称",
    "icosahedral": "5回軸を持つ正20面体対称",
}


def _explanation(symbol: str, table: dict[str, _PointGroupProps]) -> str:
    """A one-line gloss of why this point group looks the way it does."""
    props = table[symbol]
    family_text = _FAMILY_DESCRIPTIONS.get(props.family, "")
    if props.order >= _INFINITE_ORDER:
        parts = ["位数は無限（直線分子）", "主軸まわりに無限回対称"]
    else:
        parts = [f"位数{props.order}", f"最高固有回転は{props.principal_order}回"]
    if family_text:
        parts.append(family_text)
    return "、".join(parts) + "。"


def _order_scale(order: int) -> float:
    return math.log2(max(order, 1))


def _similarity_score(a: _PointGroupProps, b: _PointGroupProps) -> float:
    """Lower is more confusable: close order, close principal fold, same family shape.

    The principal-fold gap is weighted above the family-shape gap on purpose:
    same order and principal fold but a different family (D3h vs D3d, C2v vs
    C2h -- textbook mirror-placement confusions) should rank as more
    confusable than the same family at a different fold (D3h vs D4h).
    """
    family_gap = 0.0 if a.family == b.family else 1.0
    order_gap = abs(_order_scale(a.order) - _order_scale(b.order))
    principal_gap = abs(a.principal_order - b.principal_order)
    return family_gap + order_gap + 2.0 * principal_gap


def _distractors(correct: str, table: dict[str, _PointGroupProps]) -> list[str]:
    correct_props = table[correct]
    ranked = sorted(
        (symbol for symbol in table if symbol != correct),
        key=lambda symbol: (_similarity_score(correct_props, table[symbol]), symbol),
    )
    return ranked[:_DISTRACTOR_COUNT]


def point_group_question(render_data: dict) -> dict | None:
    """The structure's own point group as a multiple-choice question.

    ``None`` when there is no label, or the label is not one this table
    recognises -- the structure then has no point-group question, the same
    "no questions for this structure" outcome the catalog already handles for
    the other quizzes, rather than a crash on unexpected data.
    """
    label = (render_data.get("metadata") or {}).get("point_group_label")
    if not label:
        return None
    symbol = str(label)
    table = _CRYSTAL_PROPS if render_data.get("unit_cell") is not None else _SCHOENFLIES
    if symbol not in table:
        return None
    distractors = _distractors(symbol, table)
    if not distractors:
        return None
    return {"correct": symbol, "options": [symbol, *distractors]}


def public_questions(render_data: dict) -> list[dict]:
    """The question for the client: answer choices only, correct one withheld.

    Shuffled so the correct answer's position carries no signal across rounds.
    """
    question = point_group_question(render_data)
    if question is None:
        return []
    options = list(question["options"])
    random.shuffle(options)
    return [{"id": 0, "options": options}]


def check_answer(render_data: dict, question_id: int, selected) -> dict | None:
    """Judge the answer. ``None`` if ``question_id`` does not refer to a question."""
    if question_id != 0:
        return None
    question = point_group_question(render_data)
    if question is None:
        return None
    correct = question["correct"]
    table = _CRYSTAL_PROPS if render_data.get("unit_cell") is not None else _SCHOENFLIES
    return {
        "correct": str(selected) == correct,
        "answer": correct,
        "explanation": _explanation(correct, table),
    }
