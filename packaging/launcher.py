"""Entry point of the packaged symmetry-viewer (see symmetry_view.spec).

Starting the executable runs the Web viewer exactly as `scripts/serve.sh` does,
with the same command-line options. The server also starts this executable
again to analyze a CIF/XYZ file or to convert a cell setting in a child process;
those calls arrive as `symmetry-viewer --worker <script> ...` and are handed to
the matching tools/ script here, since the packaged build has no Python to run it.
"""

from __future__ import annotations

import os
import sys
import traceback

import export_analysis_json
import export_cell_setting_json
import view_json_server


WORKERS = {
    "export_analysis_json.py": export_analysis_json.main,
    "export_cell_setting_json.py": export_cell_setting_json.main,
}
# Printed before the server's own "Control panel: <url>" line.
BANNER = """\
symmetry-viewer を起動しています。まもなくブラウザが開きます。
ブラウザが開かないときは、下に表示される http://127.0.0.1:（数字）/ を開いてください。
終了するには、このウィンドウを閉じるか、Ctrl キーを押しながら C キーを押してください。
"""


def never_fail_on_output_encoding() -> None:
    # Through a pipe, Windows encodes output with the ANSI code page. On a system
    # whose code page has no Japanese, printing the banner would stop the program
    # before it starts. The encoding itself stays, since the server decodes a
    # worker's output with that same code page; unencodable text is replaced.
    for stream in (sys.stdout, sys.stderr):
        if stream is not None:
            stream.reconfigure(errors="replace")


def restore_system_library_path() -> None:
    # On Linux the bootloader points LD_LIBRARY_PATH at the bundled libraries.
    # The browser started from here would inherit that and load our copies in
    # place of the system's. This process has already loaded what it needs, and a
    # worker child gets the variable set again by its own bootloader.
    if not sys.platform.startswith("linux"):
        return
    original = os.environ.pop("LD_LIBRARY_PATH_ORIG", None)
    if original is None:
        os.environ.pop("LD_LIBRARY_PATH", None)
    else:
        os.environ["LD_LIBRARY_PATH"] = original


def wait_before_the_window_closes() -> None:
    # Double-clicked on Windows, the console window closes the moment the program
    # exits, taking the error message with it before anyone can read it.
    if sys.platform == "win32" and sys.stdin is not None and sys.stdin.isatty():
        input("\nエラーで終了しました。上の内容を確認したら Enter キーを押してください。")


def main() -> int:
    never_fail_on_output_encoding()
    if len(sys.argv) >= 3 and sys.argv[1] == view_json_server.WORKER_FLAG:
        script = sys.argv[2]
        sys.argv = [script, *sys.argv[3:]]
        return WORKERS[script]()
    restore_system_library_path()
    print(BANNER, flush=True)
    try:
        return view_json_server.main()
    except Exception:
        traceback.print_exc()
        wait_before_the_window_closes()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
