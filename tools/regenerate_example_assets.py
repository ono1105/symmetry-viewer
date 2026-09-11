from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from crystal_viewer.export_pipeline import default_json_output_path, export_analysis_to_json
from crystal_viewer.game.catalog import beyond_quiz_vocabulary, puzzle_counts
from crystal_viewer.source_kinds import SOURCE_KIND_CRYSTAL, SOURCE_KIND_MOLECULE


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CIF_DIR = PROJECT_ROOT / "examples/cif"
DEFAULT_MOLECULE_DIR = PROJECT_ROOT / "examples/molecules"
DEFAULT_JSON_DIR = PROJECT_ROOT / "exports/json"
DEFAULT_CATALOG_PATH = PROJECT_ROOT / "examples/example_catalog.json"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Regenerate example JSON exports and the browser example catalog."
    )
    parser.add_argument("--cif-dir", type=Path, default=DEFAULT_CIF_DIR)
    parser.add_argument("--molecule-dir", type=Path, default=DEFAULT_MOLECULE_DIR)
    parser.add_argument("--json-dir", type=Path, default=DEFAULT_JSON_DIR)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG_PATH)
    parser.add_argument("--tolerance-cart", type=float, default=1e-2)
    parser.add_argument("--indent", type=int, default=2)
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete top-level JSON files in --json-dir before regeneration.",
    )
    parser.add_argument(
        "--source",
        type=Path,
        action="append",
        default=None,
        help=(
            "Regenerate only these files instead of globbing the example directories. "
            "Repeatable. Used to refresh the retired CIFs under tests/fixtures/."
        ),
    )
    parser.add_argument(
        "--no-catalog",
        action="store_true",
        help="Skip writing the example catalog (for --source runs outside examples/).",
    )
    args = parser.parse_args()

    json_dir = args.json_dir
    json_dir.mkdir(parents=True, exist_ok=True)
    if args.clean:
        for path in json_dir.glob("*.json"):
            path.unlink()

    if args.source:
        crystal_paths = [path for path in args.source if path.suffix.lower() == ".cif"]
        molecule_paths = [path for path in args.source if path.suffix.lower() == ".xyz"]
        unknown = set(args.source) - set(crystal_paths) - set(molecule_paths)
        if unknown:
            parser.error(f"--source expects .cif or .xyz files, got: {sorted(map(str, unknown))}")
    else:
        crystal_paths = sorted(args.cif_dir.glob("*.cif"))
        molecule_paths = sorted(args.molecule_dir.glob("*.xyz"))

    catalog = {
        SOURCE_KIND_CRYSTAL: regenerate_kind(
            crystal_paths,
            SOURCE_KIND_CRYSTAL,
            json_dir,
            tolerance_cart=args.tolerance_cart,
            indent=args.indent,
        ),
        SOURCE_KIND_MOLECULE: regenerate_kind(
            molecule_paths,
            SOURCE_KIND_MOLECULE,
            json_dir,
            tolerance_cart=args.tolerance_cart,
            indent=args.indent,
        ),
    }
    if not args.no_catalog:
        args.catalog.write_text(
            json.dumps(catalog, ensure_ascii=False, indent=args.indent) + "\n",
            encoding="utf-8",
        )
    print(
        f"Regenerated {len(catalog[SOURCE_KIND_CRYSTAL])} crystal and "
        f"{len(catalog[SOURCE_KIND_MOLECULE])} molecule examples."
    )
    return 0


def regenerate_kind(
    paths: list[Path],
    kind: str,
    json_dir: Path,
    *,
    tolerance_cart: float,
    indent: int,
) -> list[dict]:
    items = []
    for source_path in paths:
        source_relative = source_path.resolve().relative_to(PROJECT_ROOT)
        output_path = default_json_output_path(source_relative, json_dir)
        export_analysis_to_json(
            source_relative,
            mode=kind,
            output_path=output_path,
            tolerance_cart=tolerance_cart,
            indent=indent,
        )
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        metadata = payload["render_data"]["metadata"]
        item = {
            "kind": kind,
            "name": source_path.stem,
            "path": str(source_relative),
            "formula": str(metadata.get("formula") or ""),
            "display_formula": str(metadata.get("display_formula") or metadata.get("formula") or ""),
            "symmetry": catalog_symmetry_label(kind, str(metadata.get("symmetry_label") or "")),
            "point_group": str(metadata.get("point_group_label") or ""),
            # The picker offers a structure only for the quizzes it can actually
            # pose, so a player never reaches a "no questions" dead end.
            "puzzle_counts": puzzle_counts(payload["render_data"]),
            # ...and withholds it entirely when it carries symmetry no quiz can
            # name, which the question counts alone would not reveal.
            "beyond_quiz_vocabulary": beyond_quiz_vocabulary(payload["render_data"]),
        }
        items.append(item)
        # Say the flag out loud: a structure dropping out of every quiz is
        # exactly the kind of thing that should not happen unnoticed.
        analysis_only = "\tanalysis-only" if item["beyond_quiz_vocabulary"] else ""
        # --json-dir may be given relative to the project root, so resolve
        # before asking for the path to report.
        print(
            f"{kind}\t{source_relative}\t"
            f"{output_path.resolve().relative_to(PROJECT_ROOT)}{analysis_only}"
        )
    return items


def catalog_symmetry_label(kind: str, label: str) -> str:
    if kind != SOURCE_KIND_CRYSTAL:
        return label
    number, _, symbol = label.partition(" ")
    if number.isdigit() and symbol:
        return f"No. {number} {symbol}"
    return label


if __name__ == "__main__":
    raise SystemExit(main())
