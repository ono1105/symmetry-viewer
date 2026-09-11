from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from pymatgen.core import Element


ATOM_DEFAULTS_PATH = Path(__file__).with_name("atom_defaults.json")


_BUILTIN_ELEMENT_COLORS = {
    # CPK/Jmol-style defaults. User-defined element/atom colors still override these.
    "H": "#ffffff",
    "He": "#d9ffff",
    "Li": "#cc80ff",
    "Be": "#c2ff00",
    "B": "#ffb5b5",
    "C": "#909090",
    "N": "#3050f8",
    "O": "#ff0d0d",
    "F": "#90e050",
    "Ne": "#b3e3f5",
    "Na": "#ab5cf2",
    "Mg": "#8aff00",
    "Al": "#bfa6a6",
    "Si": "#f0c8a0",
    "P": "#ff8000",
    "S": "#ffff30",
    "Cl": "#1ff01f",
    "Ar": "#80d1e3",
    "K": "#8f40d4",
    "Ca": "#3dff00",
    "Sc": "#e6e6e6",
    "Ti": "#bfc2c7",
    "V": "#a6a6ab",
    "Cr": "#8a99c7",
    "Mn": "#9c7ac7",
    "Fe": "#e06633",
    "Co": "#f090a0",
    "Ni": "#50d050",
    "Cu": "#c88033",
    "Zn": "#7d80b0",
    "Ga": "#c28f8f",
    "Ge": "#668f8f",
    "As": "#bd80e3",
    "Se": "#ffa100",
    "Br": "#a62929",
    "Kr": "#5cb8d1",
    "Rb": "#702eb0",
    "Sr": "#00ff00",
    "Y": "#94ffff",
    "Zr": "#94e0e0",
    "Nb": "#73c2c9",
    "Mo": "#54b5b5",
    "Tc": "#3b9e9e",
    "Ru": "#248f8f",
    "Rh": "#0a7d8c",
    "Pd": "#006985",
    "Ag": "#c0c0c0",
    "Cd": "#ffd98f",
    "In": "#a67573",
    "Sn": "#668080",
    "Sb": "#9e63b5",
    "Te": "#d47a00",
    "I": "#940094",
    "Xe": "#429eb0",
    "Cs": "#57178f",
    "Ba": "#00c900",
    "La": "#70d4ff",
    "Ce": "#ffffc7",
    "Pr": "#d9ffc7",
    "Nd": "#c7ffc7",
    "Pm": "#a3ffc7",
    "Sm": "#8fffc7",
    "Eu": "#61ffc7",
    "Gd": "#45ffc7",
    "Tb": "#30ffc7",
    "Dy": "#1fffc7",
    "Ho": "#00ff9c",
    "Er": "#00e675",
    "Tm": "#00d452",
    "Yb": "#00bf38",
    "Lu": "#00ab24",
    "Hf": "#4dc2ff",
    "Ta": "#4da6ff",
    "W": "#2194d6",
    "Re": "#267dab",
    "Os": "#266696",
    "Ir": "#175487",
    "Pt": "#d0d0e0",
    "Au": "#ffd123",
    "Hg": "#b8b8d0",
    "Tl": "#a6544d",
    "Pb": "#575961",
    "Bi": "#9e4fb5",
    "Po": "#ab5c00",
    "At": "#754f45",
    "Rn": "#428296",
    "Fr": "#420066",
    "Ra": "#007d00",
    "Ac": "#70abfa",
    "Th": "#00baff",
    "Pa": "#00a1ff",
    "U": "#008fff",
    "Np": "#0080ff",
    "Pu": "#006bff",
    "Am": "#545cf2",
    "Cm": "#785ce3",
    "Bk": "#8a4fe3",
    "Cf": "#a136d4",
    "Es": "#b31fd4",
    "Fm": "#b31fba",
    "Md": "#b30da6",
    "No": "#bd0d87",
    "Lr": "#c70066",
}

_BUILTIN_DEFAULT_ATOM_COLOR = "#9aa5b1"
_ELEMENT_RADIUS_CACHE: dict[int, float] = {}

_BUILTIN_ATOM_MESH_STYLE = {
    "smooth_shading": True,
    "lighting": True,
    "ambient": 0.56,
    "diffuse": 0.54,
    "specular": 0.42,
    "specular_power": 42,
}

_BUILTIN_HIGHLIGHT_RADIUS_SCALE = 0.96


def _normalize_hex_color_value(value) -> str | None:
    text = str(value).strip()
    if len(text) == 7 and text.startswith("#"):
        digits = text[1:]
    elif len(text) == 6:
        digits = text
    else:
        return None
    if all(char in "0123456789abcdefABCDEF" for char in digits):
        return "#" + digits.lower()
    return None


def load_atom_style_defaults(path: Path = ATOM_DEFAULTS_PATH) -> dict:
    element_colors = dict(_BUILTIN_ELEMENT_COLORS)
    default_atom_color = _BUILTIN_DEFAULT_ATOM_COLOR
    atom_mesh_style = dict(_BUILTIN_ATOM_MESH_STYLE)
    highlight_radius_scale = _BUILTIN_HIGHLIGHT_RADIUS_SCALE

    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {
            "element_colors": element_colors,
            "default_atom_color": default_atom_color,
            "atom_mesh_style": atom_mesh_style,
            "highlight_radius_scale": highlight_radius_scale,
        }
    except Exception:
        return {
            "element_colors": element_colors,
            "default_atom_color": default_atom_color,
            "atom_mesh_style": atom_mesh_style,
            "highlight_radius_scale": highlight_radius_scale,
        }

    color = _normalize_hex_color_value(loaded.get("default_atom_color", ""))
    if color is not None:
        default_atom_color = color
    for element, color_value in (loaded.get("element_colors") or {}).items():
        color = _normalize_hex_color_value(color_value)
        if color is not None:
            element_colors[str(element)] = color
    for key, value in (loaded.get("atom_mesh_style") or {}).items():
        if key not in atom_mesh_style:
            continue
        if isinstance(atom_mesh_style[key], bool):
            atom_mesh_style[key] = bool(value)
        else:
            try:
                atom_mesh_style[key] = float(value)
            except (TypeError, ValueError):
                pass
    try:
        highlight_radius_scale = float(
            loaded.get("highlight_radius_scale", highlight_radius_scale)
        )
    except (TypeError, ValueError):
        pass
    return {
        "element_colors": element_colors,
        "default_atom_color": default_atom_color,
        "atom_mesh_style": atom_mesh_style,
        "highlight_radius_scale": highlight_radius_scale,
    }


_ATOM_STYLE_DEFAULTS = load_atom_style_defaults()
ELEMENT_COLORS = _ATOM_STYLE_DEFAULTS["element_colors"]
DEFAULT_ATOM_COLOR = _ATOM_STYLE_DEFAULTS["default_atom_color"]
ATOM_MESH_STYLE = _ATOM_STYLE_DEFAULTS["atom_mesh_style"]
HIGHLIGHT_RADIUS_SCALE = _ATOM_STYLE_DEFAULTS["highlight_radius_scale"]


def element_radius_angstrom(atomic_number: int) -> float:
    atomic_number = int(atomic_number)
    cached = _ELEMENT_RADIUS_CACHE.get(atomic_number)
    if cached is not None:
        return cached
    try:
        element = Element.from_Z(atomic_number)
        radius = element.atomic_radius or element.atomic_radius_calculated
        value = float(radius) if radius is not None else 1.0
    except Exception:
        value = 1.0
    value = max(value, 0.15)
    _ELEMENT_RADIUS_CACHE[atomic_number] = value
    return value


def display_atom_radius(atom: dict, render_data: dict) -> float:
    radius = element_radius_angstrom(int(atom["atomic_number"]))
    unit_cell = render_data.get("unit_cell")
    if unit_cell is None:
        return radius
    scale, minimum_radius = display_radius_scale(render_data)
    return max(radius * scale, minimum_radius)


def display_radius_scale(render_data: dict) -> tuple[float, float]:
    atoms = render_data.get("atoms", [])
    if not atoms:
        return (1.0, 0.0)
    max_radius = max(
        element_radius_angstrom(int(item["atomic_number"]))
        for item in atoms
    )
    unit_cell = render_data.get("unit_cell")
    if unit_cell is None:
        return (1.0, 0.0)
    lattice = np.asarray(unit_cell["lattice"], dtype=float)
    lengths = np.linalg.norm(lattice, axis=1)
    shortest = float(np.min(lengths)) if len(lengths) else 0.0
    if shortest <= 1e-9 or max_radius <= 1e-9:
        return (1.0, 0.0)

    max_display_radius = shortest * 0.12
    scale = min(1.0, max_display_radius / max_radius)
    minimum_radius = shortest * 0.012
    return (scale, minimum_radius)


def atom_color(
    atom: dict,
    *,
    element_colors: dict | None = None,
    atom_colors: dict | None = None,
) -> str:
    atom_index = str(atom.get("index", ""))
    if atom_colors and atom_index in atom_colors:
        color = normalize_hex_color(atom_colors[atom_index])
        if color is not None:
            return color
    element = str(atom.get("element", ""))
    if element_colors and element in element_colors:
        color = normalize_hex_color(element_colors[element])
        if color is not None:
            return color
    return ELEMENT_COLORS.get(element, DEFAULT_ATOM_COLOR)


def normalize_hex_color(value) -> str | None:
    return _normalize_hex_color_value(value)


def color_to_rgb(color: str) -> tuple[float, float, float]:
    normalized = normalize_hex_color(color) or DEFAULT_ATOM_COLOR
    return (
        int(normalized[1:3], 16) / 255.0,
        int(normalized[3:5], 16) / 255.0,
        int(normalized[5:7], 16) / 255.0,
    )
