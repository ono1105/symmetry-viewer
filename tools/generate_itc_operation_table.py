from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

import numpy as np
import spglib
from pymatgen.core.operations import SymmOp


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate local ITC/ITA general-position operation table data.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("crystal_viewer/data/itc_operations.json"),
    )
    parser.add_argument("--indent", type=int, default=None)
    args = parser.parse_args()

    # Every ITA setting spglib knows (530 Hall symbols: unique axes, cell and
    # origin choices, hexagonal/rhombohedral axes), merged per space-group
    # number. A coordinate triplet depends only on (W, t), so merging settings
    # cannot give one operation two different strings.
    data = {
        "schema_version": 2,
        "source": "spglib Hall-symbol database, all 530 settings, merged per space-group number",
        "space_groups": {},
    }
    for hall_number in range(1, 531):
        space_group_type = spglib.get_spacegroup_type(hall_number)
        number = int(space_group_type.number)
        group = data["space_groups"].setdefault(
            str(number),
            {
                "number": number,
                "symbol": space_group_type.international_short,
                "settings": [],
                "operations": {},
            },
        )
        group["settings"].append(
            {
                "hall_number": hall_number,
                "hall_symbol": space_group_type.hall_symbol,
                "choice": space_group_type.choice,
            }
        )
        symmetry = spglib.get_symmetry_from_database(hall_number)
        for rotation, translation in zip(symmetry["rotations"], symmetry["translations"]):
            translation = np.mod(np.asarray(translation, dtype=float), 1.0)
            translation[np.isclose(translation, 1.0)] = 0.0
            operation = SymmOp.from_rotation_and_translation(rotation, translation)
            xyz = operation.as_xyz_str()
            group["operations"].setdefault(
                xyz,
                {
                    "xyz": xyz,
                    "W": np.asarray(rotation, dtype=int).tolist(),
                    "t": [float(value) for value in translation],
                },
            )

    for group in data["space_groups"].values():
        operations = sorted(group["operations"].values(), key=lambda item: item["xyz"])
        group["operations"] = operations
        group["operation_count"] = len(operations)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(data, ensure_ascii=False, indent=args.indent, separators=(",", ":") if args.indent is None else None)
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
