#!/usr/bin/env bash
# Copy what the public symmetry-viewer repository holds out of this development
# repository:
#
#   packaging/export_release_repo.sh ../symmetry-viewer
#
# The public repository carries the program, the bundled examples and the
# packaging, and builds the Windows and Linux versions on GitHub Actions. The
# report, the development notes and the test suite stay here. Everything in the
# target except its .git directory is replaced, so edit files here and export
# again rather than editing the copy.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${1:?usage: packaging/export_release_repo.sh <target-directory>}"
mkdir -p "$TARGET"
TARGET="$(cd "$TARGET" && pwd)"
if [ "$TARGET" = "$ROOT_DIR" ] || [[ "$TARGET" == "$ROOT_DIR"/* ]]; then
  echo "refusing to export into the development repository itself: $TARGET" >&2
  exit 1
fi
cd "$ROOT_DIR"

PATHS=(
  LICENSE
  requirements.txt
  crystal_viewer
  examples
  exports/json
  scripts
  packaging
  # The server and the worker scripts it starts.
  tools/_bootstrap.py
  tools/view_json_server.py
  tools/export_analysis_json.py
  tools/export_cell_setting_json.py
  # Command-line analysis, used by scripts/setup.sh as a check.
  tools/analyze_structure.py
  tools/analyze_molecule.py
  # How the bundled data was made, including the literature sources.
  tools/generate_example_structures.py
  tools/regenerate_example_assets.py
  tools/generate_itc_operation_table.py
)

# Tracked files plus new ones not yet committed, minus anything gitignored and
# anything deleted from the working tree.
mapfile -t FILES < <(
  git ls-files --cached --others --exclude-standard -- "${PATHS[@]}" \
    | grep -v '^packaging/release/' | sort -u
)

find "$TARGET" -mindepth 1 -maxdepth 1 ! -name .git -exec rm -rf {} +
for file in "${FILES[@]}"; do
  [ -f "$file" ] || continue
  # -p keeps file times: the server re-analyzes an example whose export looks
  # older than its CIF/XYZ, and would write the result into the copy.
  install -p -D -m "$(stat -c %a "$file")" "$file" "$TARGET/$file"
done
install -D -m 644 packaging/release/README.md "$TARGET/README.md"
install -D -m 644 packaging/release/gitignore "$TARGET/.gitignore"
install -D -m 644 packaging/release/build.yml "$TARGET/.github/workflows/build.yml"
install -D -m 644 packaging/release/release-notes.md "$TARGET/.github/release-notes.md"

echo "Exported $(find "$TARGET" -path "$TARGET/.git" -prune -o -type f -print | wc -l) files to $TARGET"
