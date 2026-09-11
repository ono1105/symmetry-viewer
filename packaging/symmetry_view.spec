# -*- mode: python -*-
# PyInstaller definition for the packaged symmetry-viewer.
#
# One folder holding the `symmetry-viewer` executable and everything it needs;
# no Python or Node.js on the user's machine. It runs with a console window,
# which shows the address and quits the server when closed. Build it with
# packaging/build_linux.sh or packaging/build_windows.ps1 rather than by hand.
import importlib.metadata as metadata
import sys
from pathlib import Path, PurePosixPath

from PyInstaller.utils.hooks import collect_data_files

ROOT = Path(SPECPATH).resolve().parent
WEB = ROOT / "crystal_viewer" / "web"
APP_NAME = "symmetry-viewer"

datas = [
    # Page, scripts and styles the server reads from disk on each request.
    *[(str(path), "crystal_viewer/web") for pattern in ("*.html", "*.js", "*.css") for path in sorted(WEB.glob(pattern))],
    (str(WEB / "vendor"), "crystal_viewer/web/vendor"),
    (str(ROOT / "crystal_viewer" / "data"), "crystal_viewer/data"),
    (str(ROOT / "crystal_viewer" / "viewer" / "atom_defaults.json"), "crystal_viewer/viewer"),
    # structure_analysis.load_legacy_core() loads this by file path, not import.
    (str(ROOT / "crystal_viewer" / "legacy" / "symmetry_core.py"), "crystal_viewer/legacy"),
    # The bundled examples with their exports, so opening one never re-analyzes.
    (str(ROOT / "examples"), "examples"),
    (str(ROOT / "exports" / "json" / "*.json"), "exports/json"),
    # Element tables, space-group data and the like.
    *collect_data_files("pymatgen"),
]

a = Analysis(
    [str(ROOT / "packaging" / "launcher.py")],
    # tools/ holds the server and the worker scripts the launcher imports.
    pathex=[str(ROOT), str(ROOT / "tools")],
    datas=datas,
    hiddenimports=[
        # Loaded by file path (see above), so its own imports are not seen otherwise.
        "crystal_viewer.legacy.symmetry_core",
        # Molecule.from_file()/Structure.from_file() import the reader for a file
        # extension by name (pymatgen/io/registry.py). Uploads are saved as .xyz
        # and .cif, so those two are the ones the viewer can reach.
        "pymatgen.io.xyz",
        "pymatgen.io.cif",
    ],
    excludes=[
        # pymatgen dependencies that the analysis never imports. Together about
        # 200MB; packaging/smoke_test.py analyzes every bundled example through
        # the built executable to catch one that turns out to be needed.
        "matplotlib",
        # matplotlib's add-on toolkits; useless without it, yet picked up on their own.
        "mpl_toolkits",
        "plotly",
        "sympy",
        "networkx",
        "lxml",
        "bibtexparser",
        "palettable",
        # Development only: the PyVista reference renderer and its GUI.
        "crystal_viewer.viewer.pyvista_controller",
        "crystal_viewer.viewer.native_gui",
        "pyvista",
        "vtk",
        "vtkmodules",
        "tkinter",
        "playwright",
        "IPython",
    ],
    noarchive=False,
)

# Linux: manylinux wheels link against the system's libstdc++ and libgcc_s
# instead of carrying their own, so PyInstaller copies them from the build
# machine. Those copies need that machine's glibc (2.38 on the first build) and
# would stop the program from starting on older distributions. Every desktop
# Linux has both libraries already; without them the build needs glibc 2.27 and
# GLIBCXX 3.4.22. No such files on Windows.
a.binaries = [
    entry for entry in a.binaries
    if not PurePosixPath(entry[0]).name.startswith(("libstdc++.so", "libgcc_s.so"))
]


def license_datas(analysis):
    """License files of everything that ended up in the build."""
    top_level = {name.split(".")[0] for name, _, _ in analysis.pure}
    top_level |= {PurePosixPath(dest).parts[0].split(".")[0] for dest, _, _ in analysis.binaries}
    # Some wheels list a stray __pycache__ in their RECORD (matplotlib does), which
    # would pull in the license of a package that is not in the build.
    top_level.discard("__pycache__")
    by_module = metadata.packages_distributions()
    names = sorted({name for module in top_level for name in by_module.get(module, ())})
    entries = []
    for name in names:
        for file in metadata.distribution(name).files or ():
            parts = PurePosixPath(file).parts
            if not parts[0].endswith(".dist-info"):
                continue
            if not any(word in file.name.upper() for word in ("LICEN", "COPYING", "NOTICE", "AUTHORS")):
                continue
            entries.append((str(PurePosixPath("licenses", name, *parts[1:])), str(file.locate()), "DATA"))
    # CPython and the libraries compiled into the bundled python-build-standalone
    # (OpenSSL, libffi, expat, ...), which no installed distribution carries a
    # license for. One set per platform; see packaging/licenses/README.md.
    kept = ROOT / "packaging" / "licenses" / ("windows" if sys.platform == "win32" else "linux")
    for path in sorted(kept.rglob("*")):
        if path.is_file():
            entries.append((str(PurePosixPath("licenses", *path.relative_to(kept).parts)), str(path), "DATA"))
    # The PyInstaller bootloader inside the executable: GPL, with an exception
    # that allows shipping it with a program under any license.
    for file in metadata.distribution("pyinstaller").files or ():
        if file.name.upper().startswith(("COPYING", "LICENSE")):
            entries.append((f"licenses/pyinstaller/{file.name}", str(file.locate()), "DATA"))
    entries.append(("licenses/three.js/LICENSE", str(WEB / "vendor" / "three" / "LICENSE"), "DATA"))
    return entries


a.datas += license_datas(a)

pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=APP_NAME,
    console=True,
    strip=False,
    upx=False,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    name=APP_NAME,
    strip=False,
    upx=False,
)
