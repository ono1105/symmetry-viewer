#!/usr/bin/env bash
# Build the packaged Linux version into dist/:
#
#   dist/symmetry-viewer/                       the program folder
#   dist/symmetry-viewer-linux-x86_64.tar.gz    what gets handed out
#
# Needs curl, tar and an internet connection, nothing else. The build runs on
# its own python-build-standalone CPython rather than the system Python: a
# system Python is linked against this machine's glibc, and the result would
# refuse to start on any distribution older than the one it was built on.
#
# Check the result with:
#   build/venv/bin/python packaging/smoke_test.py dist/symmetry-viewer/symmetry-viewer
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PBS_TAG=20260901
PBS_FILE="cpython-3.14.7+${PBS_TAG}-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz"
PBS_SHA256=3959f92825141e04adf44982d3a83ee57af0877e893b0796e04c1468749d9b04
PBS_URL="https://github.com/astral-sh/python-build-standalone/releases/download/${PBS_TAG}/${PBS_FILE//+/%2B}"

BUILD_DIR=build
PYTHON_DIR="$BUILD_DIR/python"
VENV_DIR="$BUILD_DIR/venv"
DIST_DIR=dist
APP_DIR="$DIST_DIR/symmetry-viewer"
ARCHIVE="$DIST_DIR/symmetry-viewer-linux-x86_64.tar.gz"

mkdir -p "$BUILD_DIR"
if [ "$(cat "$PYTHON_DIR/.pbs-tag" 2>/dev/null)" != "$PBS_TAG" ]; then
  curl -fL --retry 3 -o "$BUILD_DIR/$PBS_FILE" "$PBS_URL"
  echo "$PBS_SHA256  $BUILD_DIR/$PBS_FILE" | sha256sum -c -
  rm -rf "$PYTHON_DIR"
  tar -C "$BUILD_DIR" -xzf "$BUILD_DIR/$PBS_FILE"   # unpacks into build/python
  rm "$BUILD_DIR/$PBS_FILE"
  echo "$PBS_TAG" > "$PYTHON_DIR/.pbs-tag"
fi

if [ "$(cat "$VENV_DIR/.pbs-tag" 2>/dev/null)" != "$PBS_TAG" ]; then
  rm -rf "$VENV_DIR"
  "$PYTHON_DIR/bin/python3" -m venv "$VENV_DIR"
  echo "$PBS_TAG" > "$VENV_DIR/.pbs-tag"
fi
"$VENV_DIR/bin/python" -m pip install --quiet --upgrade pip
"$VENV_DIR/bin/python" -m pip install --quiet -r packaging/requirements-build.txt

rm -rf "$APP_DIR" "$ARCHIVE"
"$VENV_DIR/bin/python" -m PyInstaller --noconfirm --clean \
  --distpath "$DIST_DIR" --workpath "$BUILD_DIR/pyinstaller" \
  packaging/symmetry_view.spec

cp packaging/README_linux.txt "$APP_DIR/README.txt"
cp LICENSE "$APP_DIR/LICENSE.txt"
tar -C "$DIST_DIR" -czf "$ARCHIVE" symmetry-viewer

du -sh "$APP_DIR" "$ARCHIVE"
# The oldest glibc the result starts on. 2.27 when libstdc++ is left out as
# symmetry_view.spec does; README_linux.txt states it, so keep the two in step.
if command -v objdump >/dev/null 2>&1; then
  echo "Needs glibc $(find "$APP_DIR" -type f -name '*.so*' -print0 \
    | xargs -0 objdump -T 2>/dev/null | grep -oE 'GLIBC_2\.[0-9]+' \
    | sort -t. -k2,2n -u | tail -1 | cut -d_ -f2) or newer"
fi
echo "Built $ARCHIVE"
