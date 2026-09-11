from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from crystal_viewer.molecule_analysis import analyze_molecule_file


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze molecular point-group symmetry.")
    parser.add_argument("molecule", type=Path)
    parser.add_argument("--tolerance", type=float, default=0.3)
    parser.add_argument("--eigen-tolerance", type=float, default=0.01)
    parser.add_argument("--matrix-tolerance", type=float, default=0.1)
    args = parser.parse_args()

    result = analyze_molecule_file(
        args.molecule,
        tolerance=args.tolerance,
        eigen_tolerance=args.eigen_tolerance,
        matrix_tolerance=args.matrix_tolerance,
    )

    print("=== Molecule ===")
    print(f"file: {result.molecule.source_file}")
    print(f"formula: {result.molecule.formula}")
    print(f"sites: {result.molecule.site_count}")
    print(f"center cart: {result.molecule.center_cart}")

    print("\n=== Point group ===")
    print(f"symbol: {result.point_group.symbol}")
    print(f"operations: {result.point_group.operation_count}")
    print(f"rotational symmetry number: {result.point_group.rotational_symmetry_number}")
    print(f"equivalent atom sets: {result.point_group.equivalent_atom_sets}")

    print("\n=== Elements ===")
    print(f"centers: {len(result.centers)}")
    print(f"axes: {len(result.axes)}")
    print(f"planes: {len(result.planes)}")

    print("\n=== Operations ===")
    for op in result.operations:
        print(f"{op.index:3d}: {op.symbol:>6s}  {op.kind:16s}  det={op.det:2d}  order={op.order}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
