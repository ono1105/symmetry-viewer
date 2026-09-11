from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from crystal_viewer.export_pipeline import build_export_payload, export_analysis_to_json


def main() -> int:
    parser = argparse.ArgumentParser(description="Export RenderData and AtomMapping as JSON.")
    parser.add_argument("path", type=Path)
    parser.add_argument("--mode", choices=["crystal", "molecule"], required=True)
    parser.add_argument("--output", "-o", type=Path, default=None)
    parser.add_argument("--tolerance-cart", type=float, default=1e-2)
    parser.add_argument("--indent", type=int, default=2)
    args = parser.parse_args()

    if args.output is None:
        payload = build_export_payload(
            args.path,
            mode=args.mode,
            tolerance_cart=args.tolerance_cart,
        )
        print(json.dumps(payload, ensure_ascii=False, indent=args.indent))
    else:
        output_path = export_analysis_to_json(
            args.path,
            mode=args.mode,
            output_path=args.output,
            tolerance_cart=args.tolerance_cart,
            indent=args.indent,
        )
        print(f"Wrote {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
