"""Per-structure quiz question counts, recorded in the example catalog.

The structure picker has to know which structures can actually be played for the
quiz the player chose: water has no composition questions, HCl has no mapping
questions, and a symmorphic crystal has no screw/glide operations to identify.
Offering them and then showing "no questions" is a dead end.

Counting is done once, offline, by ``tools/regenerate_example_assets.py`` and
stored in ``examples/example_catalog.json``: the counts are a property of the
exported structure, so recomputing them per request (composition alone pairs
every ordered operation pair) would repeat work that never changes.
"""

from __future__ import annotations

from crystal_viewer.game import atom_mapping, axis_orders, composition, operation_identify, point_group


PUZZLE_COUNT_KEYS: tuple[str, ...] = (
    "axis",
    "operation_normal",
    "operation_hard",
    "composition",
    "mapping",
    "point_group",
)


def beyond_quiz_vocabulary(render_data: dict) -> bool:
    """True when the structure carries symmetry no quiz can name.

    The answer vocabulary is closed at folds 2/3/4/6 (plus ∞ for linear
    molecules), so a 5-fold axis or an S10 has no answer to pick. Counting
    questions is not enough to catch this: an icosahedral cluster still has
    plenty of C2/C3/mirror questions, and offering only those would teach that
    its 5-fold axes do not exist. Such a structure is analysis-mode only.

    Note this is all-or-nothing — one unnameable operation withdraws the
    structure from every quiz, including ones that could have asked about it.
    That is deliberate here, but if a future structure trips this for a
    marginal reason, fix the answer path rather than letting the flag hide it.

    The point-group quiz is the one exception the client makes to this flag
    (see ``isEligible()`` in ``puzzle.js``): naming the whole structure's point
    group (e.g. "Ih") never asks the player to pick a fold from the closed
    2/3/4/6 vocabulary, so a 5-fold axis does not make that question a lie the
    way it would for the other four.
    """
    return bool(operation_identify.unnameable_operations(render_data))


def puzzle_counts(render_data: dict) -> dict[str, int]:
    """How many questions each quiz can pose for this structure.

    Note the axis count covers the rotation-axis quiz only; the improper-axis
    variant exists in the API but no client asks for it.
    """
    return {
        "axis": len(axis_orders.public_questions(render_data)),
        "operation_normal": len(
            operation_identify.public_questions(render_data, operation_identify.NORMAL)
        ),
        "operation_hard": len(
            operation_identify.public_questions(render_data, operation_identify.HARD)
        ),
        "composition": len(composition.public_questions(render_data)),
        "mapping": len(atom_mapping.public_questions(render_data)),
        "point_group": len(point_group.public_questions(render_data)),
    }
