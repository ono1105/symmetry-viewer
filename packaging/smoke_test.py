"""Check a built symmetry_view end to end through its HTTP API.

    python3 packaging/smoke_test.py dist/symmetry-viewer/symmetry-viewer
    python3 packaging/smoke_test.py --root . -- .venv/bin/python tools/view_json_server.py

Uses only the standard library, so any Python 3.11+ can run it. It starts the
given command with --no-browser and then, as the browser would:

- fetches the page and every script, stylesheet and Three.js file it loads;
- opens every bundled example, and fails if that re-ran the analysis instead of
  reading the bundled export (the export files must stay untouched);
- converts each bundled crystal to its primitive and Bravais cell;
- imports every bundled CIF/XYZ as if it were the user's own file, which runs
  the whole analysis in a worker process, so a module left out of the build
  shows up here;
- uploads one molecule three times, saved with LF, CRLF and CR newlines, and
  checks what the analysis made of it;
- asks each quiz for its questions and writes a GIF.

--quick keeps the cell and import steps to one crystal and one molecule.
--root is the folder holding examples/ and exports/json; for a built
executable it defaults to the _internal folder next to it.
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path


STATIC_PATHS = (
    "/static/browser_ui.css",
    "/static/browser_ui.js",
    "/static/animation_path.js",
    "/static/colors.js",
    "/static/three_loader.js",
    "/static/three_view.js",
    "/static/puzzle.js",
    "/vendor/three/three.module.js",
    "/vendor/three/three.core.js",
    "/vendor/three/addons/controls/TrackballControls.js",
)
QUIZ_PATHS = (
    "/api/puzzle/axis_orders",
    "/api/puzzle/operations",
    "/api/puzzle/composition",
    "/api/puzzle/mapping",
    "/api/puzzle/point_group",
)
# A 1x1 PNG, enough for the GIF writer.
PIXEL_PNG = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="
STARTUP_TIMEOUT_SEC = 120


class SmokeTest:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url
        self.request_id = 0
        self.failures: list[str] = []

    def request(self, path: str, payload: dict | None = None, *, timeout: float = 300) -> tuple[int, bytes]:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(self.base_url + path, data=data)
        if data is not None:
            request.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.status, response.read()
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read()

    def post_ok(self, label: str, path: str, payload: dict) -> bool:
        self.request_id += 1
        started = time.monotonic()
        status, body = self.request(path, {**payload, "request_id": self.request_id})
        elapsed = time.monotonic() - started
        try:
            result = json.loads(body)
        except json.JSONDecodeError:
            result = {}
        if status == 200 and result.get("ok"):
            print(f"  ok    {label} ({elapsed:.1f}s)")
            return True
        error = result.get("error") or body[:300].decode("utf-8", "replace")
        self.fail(f"{label}: HTTP {status}: {error}")
        return False

    def fail(self, message: str) -> None:
        print(f"  FAIL  {message}")
        self.failures.append(message)

    def check_static(self) -> None:
        print("page and assets")
        status, body = self.request("/")
        if status != 200 or b"<html" not in body.lower():
            self.fail(f"/: HTTP {status}")
        for path in STATIC_PATHS:
            status, body = self.request(path)
            if status != 200 or not body:
                self.fail(f"{path}: HTTP {status}, {len(body)} bytes")
        print(f"  checked / and {len(STATIC_PATHS)} assets")

    def examples(self) -> dict[str, list[dict]]:
        status, body = self.request("/api/examples")
        if status != 200:
            self.fail(f"/api/examples: HTTP {status}")
            return {}
        catalog = json.loads(body)
        print(f"examples: {len(catalog.get('crystal', []))} crystals, {len(catalog.get('molecule', []))} molecules")
        return catalog

    def open_examples(self, catalog: dict[str, list[dict]], root: Path) -> None:
        print("open every bundled example (must use the bundled exports)")
        exports = root / "exports" / "json"
        before = export_snapshot(exports)
        for kind in ("crystal", "molecule"):
            for item in catalog.get(kind, []):
                self.post_ok(f"open {item['path']}", "/api/open_example", {"kind": kind, "path": item["path"]})
        after = export_snapshot(exports)
        if not before:
            self.fail(f"no exports found under {exports}")
        elif before != after:
            changed = sorted(name for name in before.keys() | after.keys() if before.get(name) != after.get(name))
            self.fail(f"opening examples rewrote bundled exports: {', '.join(changed)}")

    def convert_cells(self, crystals: list[dict]) -> None:
        print("primitive and Bravais cells")
        for item in crystals:
            if not self.post_ok(f"open {item['path']}", "/api/open_example", {"kind": "crystal", "path": item["path"]}):
                continue
            for mode in ("primitive", "conventional"):
                self.post_ok(f"  {mode}", "/api/cell_setting", {"cell_setting_mode": mode})

    def import_files(self, root: Path, crystals: list[dict], molecules: list[dict]) -> None:
        print("import bundled files as the user's own (full analysis in a worker)")
        for kind, items, path in (
            ("crystal", crystals, "/api/import_cif"),
            ("molecule", molecules, "/api/import_molecule"),
        ):
            for item in items:
                source = root / item["path"]
                content = source.read_text(encoding="utf-8")
                self.post_ok(f"import {source.name}", path, {"filename": source.name, "content": content})

    def check_newlines(self) -> None:
        """Upload the same molecule with each newline convention.

        Notepad saves CRLF, so this is what a Windows user's own file looks
        like. It used to fail there and only there: the upload was written back
        in text mode, Windows turned every "\\n" into "\\r\\n" a second time, and
        the file read back with a blank line between every line, which the XYZ
        parser could not take. The bundled examples cannot cover this -- reading
        one with read_text() strips the carriage returns before the server ever
        sees them -- so build the content here instead. CIF goes through the
        same upload path, so guarding XYZ guards both.

        **On Linux this passes whether or not the fix is in place**, because
        nothing doubles the newlines on the way to disk and Python's universal
        newlines absorb a bare CR on the way back. It is the Windows run of this
        smoke test that does the work. Reproducing the Windows write on Linux
        (normalization removed, newline="\\r\\n") fails on CRLF alone, exactly as
        the machine did, so the check is known to catch the bug it guards.
        """
        print("uploads with each newline convention (a file saved on Windows arrives as CRLF)")
        water = (
            "3\nwater\n"
            "O 0.000000 0.000000 0.117300\n"
            "H 0.000000 0.757200 -0.469200\n"
            "H 0.000000 -0.757200 -0.469200\n"
        )
        for label, content in (
            ("LF", water),
            ("CRLF", water.replace("\n", "\r\n")),
            ("CR", water.replace("\n", "\r")),
        ):
            if not self.post_ok(
                f"import a molecule saved with {label} newlines",
                "/api/import_molecule",
                {"filename": f"newline_{label}.xyz", "content": content},
            ):
                continue
            # "ok" is not enough. A file that reaches the parser mangled can
            # still come back as a molecule -- just the wrong one.
            status, body = self.request("/api/state")
            metadata = (json.loads(body).get("metadata") or {}) if status == 200 else {}
            found = (
                metadata.get("display_formula") or metadata.get("formula"),
                metadata.get("point_group_label"),
                metadata.get("operation_count"),
            )
            if found == ("H2O", "C2v", 4):
                print(f"  ok    {label} -> H2O, C2v, 4 operations")
            else:
                self.fail(f"{label} newlines analyzed as {found}, expected ('H2O', 'C2v', 4)")

    def check_quizzes(self, molecule: dict | None) -> None:
        print("quizzes")
        if molecule is not None:
            self.post_ok(f"open {molecule['path']}", "/api/open_example", {"kind": "molecule", "path": molecule["path"]})
        for path in QUIZ_PATHS:
            status, body = self.request(path)
            try:
                json.loads(body)
            except json.JSONDecodeError:
                status = -1
            if status != 200:
                self.fail(f"{path}: HTTP {status}")
        print(f"  checked {len(QUIZ_PATHS)} quiz endpoints")

    def check_gif(self) -> None:
        print("GIF export")
        status, body = self.request("/api/export_gif", {"frames": [PIXEL_PNG, PIXEL_PNG], "frame_duration_ms": 100})
        if status != 200 or not body.startswith(b"GIF8"):
            self.fail(f"/api/export_gif: HTTP {status}, {body[:80]!r}")
        else:
            print(f"  ok    {len(body)} bytes")


def export_snapshot(directory: Path) -> dict[str, tuple[int, int]]:
    return {path.name: (path.stat().st_mtime_ns, path.stat().st_size) for path in directory.glob("*.json")}


def start(command: list[str]) -> tuple[subprocess.Popen, str]:
    process = subprocess.Popen(
        [*command, "--no-browser", "--port", "0"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    found: list[str] = []
    ready = threading.Event()

    def pump() -> None:
        for line in process.stdout:
            print(f"  [app] {line.rstrip()}")
            match = re.search(r"http://[\d.]+:\d+/", line)
            if match and not found:
                found.append(match.group(0).rstrip("/"))
                ready.set()
        ready.set()

    threading.Thread(target=pump, daemon=True).start()
    if not ready.wait(STARTUP_TIMEOUT_SEC) or not found:
        process.kill()
        raise SystemExit(f"the app did not print its address within {STARTUP_TIMEOUT_SEC}s")
    return process, found[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", nargs="+", help="The built executable, or a command that starts the server.")
    parser.add_argument("--root", type=Path, default=None, help="Folder holding examples/ and exports/json.")
    parser.add_argument("--quick", action="store_true", help="Convert and import only one crystal and one molecule.")
    args = parser.parse_args()

    root = args.root if args.root is not None else Path(args.command[0]).resolve().parent / "_internal"
    started = time.monotonic()
    process, base_url = start(args.command)
    print(f"started {base_url} in {time.monotonic() - started:.1f}s")
    test = SmokeTest(base_url)
    try:
        test.check_static()
        catalog = test.examples()
        crystals = catalog.get("crystal", [])
        molecules = catalog.get("molecule", [])
        if not crystals or not molecules:
            test.fail("the example catalog is missing crystals or molecules")
        test.open_examples(catalog, root)
        some_crystals = crystals[:1] if args.quick else crystals
        some_molecules = molecules[:1] if args.quick else molecules
        test.convert_cells(some_crystals)
        test.import_files(root, some_crystals, some_molecules)
        test.check_newlines()
        test.check_quizzes(molecules[0] if molecules else None)
        test.check_gif()
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()

    print(f"\n{len(test.failures)} failure(s) in {time.monotonic() - started:.0f}s")
    for failure in test.failures:
        print(f"  - {failure}")
    return 1 if test.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
