# License texts shipped with the packaged build

`symmetry_view.spec` copies `linux/` or `windows/` (whichever matches the build)
into `_internal/licenses/`, next to the license files it collects from the
installed Python packages (numpy, scipy, pymatgen, ...) and the PyInstaller
bootloader.

These are the ones no installed package carries: the build runs on a
python-build-standalone CPython, which has these libraries compiled in.

## Where they came from

Each `python-build-standalone/` holds files from `python/licenses/` of the
"full" archive of the release pinned in `build_linux.sh` / `build_windows.ps1`
(the `install_only` archives the builds use do not include them), keeping only
the components that end up in the build according to that archive's
`PYTHON.json`:

- `linux/`: `cpython-3.14.7+20260901-x86_64-unknown-linux-gnu-pgo+lto-full.tar.zst`.
  All extension modules are compiled into `libpython3.14.so`.
- `windows/`: `cpython-3.14.7+20260901-x86_64-pc-windows-msvc-pgo-full.tar.zst`.
  Extension modules are separate `.pyd` files next to their DLLs.

| File | Component | Used by | Linux | Windows |
|---|---|---|---|---|
| `LICENSE.cpython.txt` | CPython | everything | yes | yes |
| `LICENSE.openssl-3.txt` | OpenSSL 3 | `_ssl`, `_hashlib` | yes | |
| `LICENSE.openssl-1.1.txt` | OpenSSL 1.1 | `_ssl`, `_hashlib` | | yes |
| `LICENSE.libffi.txt` | libffi | `_ctypes` | yes | yes |
| `LICENSE.expat.txt` | expat | `pyexpat` | yes | yes |
| `LICENSE.bzip2.txt` | bzip2 | `_bz2` | yes | yes |
| `LICENSE.liblzma.txt` | xz / liblzma | `_lzma` | yes | yes |
| `LICENSE.mpdecimal.txt` | mpdecimal | `_decimal` | yes | yes |
| `LICENSE.sqlite.txt` | SQLite | `_sqlite3` | yes | yes |
| `LICENSE.zlib.txt` | zlib | `zlib`, `binascii` | yes | yes |
| `LICENSE.libuuid.txt` | libuuid | `_uuid` | yes | |
| `LICENSE.libedit.txt`, `LICENSE.ncurses.txt` | libedit, ncurses | `readline`, `_curses` | yes | |

Two are not in those directories and were fetched from upstream:

- `LICENSE.zstd.txt`: `_zstd` links zstd (BSD-3-Clause per `PYTHON.json`).
  From https://github.com/facebook/zstd/blob/v1.5.7/LICENSE.
- `LICENSE.hacl-star.txt`: CPython's hash functions come from HACL\*
  (`Modules/_hacl/`, "Licensed under the Apache 2.0 and MIT Licenses").
  From https://github.com/hacl-star/hacl-star/blob/main/LICENSE.

Left out on purpose, because the viewer never imports their extension modules
and the builds do not contain them: Berkeley DB (`_dbm`, Sleepycat license) and
Tcl/Tk and X11 (`_tkinter`). No GPL component is linked (on Linux, `readline`
uses libedit).

When a build script moves to another python-build-standalone release, fetch
that release's full archive for the platform and redo the selection from its
`PYTHON.json`.
