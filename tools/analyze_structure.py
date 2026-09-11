from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from crystal_viewer.structure_analysis import analyze_cif


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze crystal structure symmetry from a CIF file.")
    parser.add_argument("cif", type=Path)
    parser.add_argument("--search-range", type=int, default=2)
    parser.add_argument("--symprec", type=float, default=1e-3)
    parser.add_argument("--angle-tolerance", type=float, default=5.0)
    args = parser.parse_args()

    result = analyze_cif(
        args.cif,
        search_range=args.search_range,
        symprec=args.symprec,
        angle_tolerance=args.angle_tolerance,
    )

    print("=== Structure ===")
    print(f"file: {result.structure.source_file}")
    print(f"formula: {result.structure.formula}")
    print(f"sites: {result.structure.site_count}")

    print("\n=== Space group ===")
    print(f"number: {result.space_group.number}")
    print(f"international: {result.space_group.international}")
    print(f"hall: {result.space_group.hall}")
    print(f"point group: {result.space_group.point_group}")
    print(f"operations: {result.space_group.operation_count}")

    print("\n=== Elements ===")
    print(f"centers: {len(result.centers)}")
    print(f"axes: {len(result.axes)}")
    print(f"planes: {len(result.planes)}")

    print("\n=== First operations ===")
    for op in result.operations[:12]:
        print(f"{op.index:3d}: {op.international_symbol:>11s}  {op.kind:36s}  t={op.t}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
