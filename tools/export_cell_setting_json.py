from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from crystal_viewer.viewer.cell_settings import standardized_payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Reanalyze an exported crystal JSON in a standardized cell setting.")
    parser.add_argument("json_path", type=Path)
    parser.add_argument("--cell-setting", choices=["native", "primitive", "conventional"], required=True)
    parser.add_argument("--output", "-o", type=Path, required=True)
    parser.add_argument("--tolerance-cart", type=float, default=1e-2)
    parser.add_argument("--indent", type=int, default=2)
    parser.add_argument("--require-distinct", action="store_true")
    args = parser.parse_args()

    payload = json.loads(args.json_path.read_text(encoding="utf-8"))
    converted = standardized_payload(
        payload,
        args.cell_setting,
        tolerance_cart=args.tolerance_cart,
        require_distinct=args.require_distinct,
    )
    if converted is None:
        if args.require_distinct and args.cell_setting == "primitive":
            raise RuntimeError("This structure is already represented as a primitive cell.")
        if args.require_distinct and args.cell_setting == "conventional":
            raise RuntimeError("This structure is already represented as a Bravais cell.")
        raise RuntimeError(f"cell setting {args.cell_setting!r} is not available for this structure")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(converted, ensure_ascii=False, indent=args.indent),
        encoding="utf-8",
    )
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
