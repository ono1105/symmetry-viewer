from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from pymatgen.symmetry.groups import SpaceGroup


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate local ITC/ITA general-position operation table data.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("crystal_viewer/data/itc_operations.json"),
    )
    parser.add_argument("--indent", type=int, default=None)
    args = parser.parse_args()

    data = {
        "schema_version": 1,
        "source": "pymatgen.symmetry.groups.SpaceGroup general-position symmetry operations",
        "space_groups": {},
    }
    for number in range(1, 231):
        space_group = SpaceGroup.from_int_number(number)
        operations = []
        for operation in sorted(space_group.symmetry_ops, key=lambda op: op.as_xyz_str()):
            operations.append(
                {
                    "xyz": operation.as_xyz_str(),
                    "W": operation.rotation_matrix.astype(int).tolist(),
                    "t": [float(value) for value in operation.translation_vector],
                }
            )
        data["space_groups"][str(number)] = {
            "number": number,
            "symbol": space_group.symbol,
            "operation_count": len(operations),
            "operations": operations,
        }

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
