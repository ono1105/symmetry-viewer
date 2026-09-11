"""Vendored legacy symmetry analysis helpers.

Still live, despite the name: crystal analysis runs on `symmetry_core.py`.
`structure_analysis.load_legacy_core()` loads it by file path rather than
importing it, so nothing here shows up as an import and a search for dead code
can easily mistake this package for one. Deleting it breaks every CIF path.
"""
