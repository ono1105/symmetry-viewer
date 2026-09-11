#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON:-python3}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python command not found: $PYTHON_BIN" >&2
  echo "Install Python 3.11+ or set PYTHON=/path/to/python." >&2
  exit 1
fi

if [ ! -d ".venv" ]; then
  "$PYTHON_BIN" -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# Three.js ships in crystal_viewer/web/vendor/three, so no Node.js step is needed.

python tools/analyze_structure.py examples/cif/Halite.cif >/dev/null
python tools/analyze_molecule.py examples/molecules/water.xyz >/dev/null

cat <<'EOF'
Setup complete.

Start the web viewer:
  scripts/serve.sh

Start directly in puzzle mode:
  scripts/serve.sh --mode puzzle
EOF
