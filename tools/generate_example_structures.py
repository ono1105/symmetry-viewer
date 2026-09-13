"""Generate the bundled teaching examples from published structural data.

The example set is meant to be *recognisable*: the substances a symmetry course
already talks about, small enough that every atom stays visible while an
operation plays.  Rather than collecting downloaded CIFs of varying provenance,
each structure is written here as its space group, lattice constants and Wyckoff
positions, with the source of those numbers in the table.  That keeps the data
auditable, keeps the files free of experimental clutter (partial occupancies,
anisotropic displacement parameters), and lets the whole set be rebuilt with one
command.

    tools/generate_example_structures.py              # write examples/
    tools/generate_example_structures.py --check-only # verify the shipped files

Verification re-reads what was written and checks the space group, atom count
and composition come back as intended.  It deliberately does not compare
coordinates: CifWriter is free to emit any symmetry-equivalent position, and a
molecule's point group is what matters, not the orientation it was built in.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

try:
    import _bootstrap  # noqa: F401
except ModuleNotFoundError:  # imported as tools.generate_example_structures
    from tools import _bootstrap  # noqa: F401

import numpy as np
from pymatgen.core import Composition, Lattice, Molecule, Structure
from pymatgen.io.cif import CifWriter

from crystal_viewer.game.catalog import beyond_quiz_vocabulary
from crystal_viewer.json_export import render_data_to_dict
from crystal_viewer.molecule_analysis import analyze_molecule_file
from crystal_viewer.render_data import render_data_from_molecule
from crystal_viewer.structure_analysis import analyze_cif


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CIF_DIR = PROJECT_ROOT / "examples/cif"
DEFAULT_MOLECULE_DIR = PROJECT_ROOT / "examples/molecules"

# Hand-authored files that predate this script and must not be overwritten.
# Halite carries the generation-operation snapshot test_structure_analysis pins,
# and BaTiO3 is the only example with an empty symmetry-operation loop, which is
# what exercises the rhombohedral fallback in structure_analysis.
PROTECTED = frozenset({"Halite", "BaTiO3"})


@dataclass(frozen=True)
class CrystalSpec:
    stem: str
    space_group: int
    symbol: str
    lattice: Lattice
    species: tuple[str, ...]
    coords: tuple[tuple[float, float, float], ...]
    wyckoff: tuple[str, ...]
    atoms: int
    formula: str
    source: str
    note: str = ""


@dataclass(frozen=True)
class MoleculeSpec:
    stem: str
    point_group: str
    atoms: int
    build: Callable[[], Molecule]
    source: str
    note: str = ""
    # True for a structure the quizzes cannot ask about, because it carries a
    # fold the answer vocabulary has no name for (a 5-fold axis). Declared here
    # and checked against the computed flag, so a structure cannot quietly drop
    # out of every quiz without someone writing it down.
    analysis_only: bool = False


# --- molecular geometry helpers ---------------------------------------------
# Bond lengths are in angstrom. The analyzer only needs the geometry to be
# symmetric to within its 0.3 A tolerance, but building from exact ideal angles
# keeps the point group unambiguous rather than accidentally over-symmetric.

TETRAHEDRAL_DEG = 109.4712206344907  # arccos(-1/3)


def _spherical(radius: float, polar_deg: float, azimuth_deg: float) -> list[float]:
    polar = math.radians(polar_deg)
    azimuth = math.radians(azimuth_deg)
    return [
        radius * math.sin(polar) * math.cos(azimuth),
        radius * math.sin(polar) * math.sin(azimuth),
        radius * math.cos(polar),
    ]


def ethane(dihedral_deg: float = 60.0) -> Molecule:
    """C2H6 about the z axis; 60 deg is staggered (D3d), 0 deg eclipsed (D3h)."""
    cc, ch = 1.535, 1.094
    half = cc / 2
    # Angle of each C-H away from the C-C axis, from the tetrahedral H-C-H angle.
    tilt = 180.0 - TETRAHEDRAL_DEG
    species = ["C", "C"]
    coords = [[0.0, 0.0, half], [0.0, 0.0, -half]]
    for index in range(3):
        top = _spherical(ch, tilt, 120.0 * index)
        bottom = _spherical(ch, 180.0 - tilt, 120.0 * index + dihedral_deg)
        species += ["H", "H"]
        coords.append([top[0], top[1], half + top[2]])
        coords.append([bottom[0], bottom[1], -half + bottom[2]])
    return Molecule(species, coords)


def hydrogen_peroxide() -> Molecule:
    """H2O2 with its 111 deg dihedral: a C2 axis and nothing else."""
    oo, oh, angle_deg, dihedral_deg = 1.475, 0.950, 94.8, 111.5
    half = oo / 2
    # Put the O-O bond on z and rotate each H by half the dihedral about it, so
    # the C2 axis (perpendicular to O-O, bisecting the dihedral) is exact.
    tilt = 180.0 - angle_deg
    top = _spherical(oh, tilt, +dihedral_deg / 2)
    bottom = _spherical(oh, 180.0 - tilt, -dihedral_deg / 2)
    return Molecule(
        ["O", "O", "H", "H"],
        [
            [0.0, 0.0, half],
            [0.0, 0.0, -half],
            [top[0], top[1], half + top[2]],
            [bottom[0], bottom[1], -half + bottom[2]],
        ],
    )


def planar_ethene_frame(substituent: str, bond: float, ch: float = 1.087) -> Molecule:
    """trans-1,2-disubstituted ethene in the xy plane (C2h).

    Carbons sit on the x axis. Every other atom is placed on the +x carbon and
    then copied through the origin, which makes the inversion centre exact; the
    molecular plane is sigma_h and their product is the C2 along z. The
    substituent goes up and the hydrogen down, so no C2 lies in the plane —
    that is what keeps this C2h rather than the D2h of ethene itself.
    """
    cc, angle_deg = 1.330, 121.0
    half = cc / 2
    # Bond direction measured from the C=C bond, which points along -x here.
    angle = math.radians(angle_deg)
    species = ["C", "C"]
    coords = [[half, 0.0, 0.0], [-half, 0.0, 0.0]]
    for element, length, updown in ((substituent, bond, 1.0), ("H", ch, -1.0)):
        dx = -length * math.cos(angle)
        dy = updown * length * math.sin(angle)
        species += [element, element]
        coords.append([half + dx, dy, 0.0])
        coords.append([-half - dx, -dy, 0.0])
    return Molecule(species, coords)


def substituted_methane(substituents: tuple[str, ...], bonds: tuple[float, ...]) -> Molecule:
    """A tetrahedral carbon with the given four substituents."""
    directions = [
        [1.0, 1.0, 1.0],
        [1.0, -1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
    ]
    species = ["C"]
    coords = [[0.0, 0.0, 0.0]]
    for element, bond, direction in zip(substituents, bonds, directions):
        unit = np.asarray(direction, dtype=float) / math.sqrt(3.0)
        species.append(element)
        coords.append(list(unit * bond))
    return Molecule(species, coords)


def trigonal_bipyramid(center: str, axial: str, equatorial: str,
                       axial_bond: float, equatorial_bond: float) -> Molecule:
    species = [center]
    coords = [[0.0, 0.0, 0.0]]
    for sign in (1.0, -1.0):
        species.append(axial)
        coords.append([0.0, 0.0, sign * axial_bond])
    for index in range(3):
        species.append(equatorial)
        coords.append(_spherical(equatorial_bond, 90.0, 120.0 * index))
    return Molecule(species, coords)


def square_pyramid(center: str, apical: str, basal: str, apical_bond: float,
                   basal_bond: float, basal_polar_deg: float) -> Molecule:
    """C4v: four basal bonds tipped below the equator, one apical bond on +z."""
    species = [center, apical]
    coords = [[0.0, 0.0, 0.0], [0.0, 0.0, apical_bond]]
    for index in range(4):
        species.append(basal)
        coords.append(_spherical(basal_bond, basal_polar_deg, 90.0 * index))
    return Molecule(species, coords)


def icosahedron_directions() -> np.ndarray:
    """The 12 unit vectors to the vertices of a regular icosahedron.

    Cyclic permutations of (0, +-1, +-phi). Six 5-fold axes pass through
    opposite pairs — the symmetry that no periodic lattice can have.
    """
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    vertices = []
    for sign_a in (1.0, -1.0):
        for sign_b in (1.0, -1.0):
            vertices.append([0.0, sign_a * 1.0, sign_b * phi])
            vertices.append([sign_a * 1.0, sign_b * phi, 0.0])
            vertices.append([sign_b * phi, 0.0, sign_a * 1.0])
    directions = np.asarray(vertices, dtype=float)
    return directions / np.linalg.norm(directions[0])


def icosahedron_edges(directions: np.ndarray) -> list[tuple[int, int]]:
    """The 30 vertex pairs one edge apart (the shortest non-zero separation)."""
    distances = [
        (i, j, float(np.linalg.norm(directions[i] - directions[j])))
        for i in range(len(directions))
        for j in range(i + 1, len(directions))
    ]
    shortest = min(distance for *_, distance in distances)
    return [(i, j) for i, j, distance in distances if abs(distance - shortest) < 1e-9]


def centered_icosahedron(center: str, shell: str, radius: float) -> Molecule:
    """One atom surrounded by a regular icosahedron of 12 others."""
    directions = icosahedron_directions()
    species = [center] + [shell] * len(directions)
    coords = [[0.0, 0.0, 0.0]] + [list(direction * radius) for direction in directions]
    return Molecule(species, coords)


def mackay_icosahedron(inner: str, middle: str, outer: str, radius: float) -> Molecule:
    """The 54-atom Mackay icosahedron, centre vacant.

    Three shells at 1 : sqrt(2 + 2/sqrt5) : 2 — an icosahedron, the
    icosidodecahedron on its edge midpoints, and a twice-as-large icosahedron.
    The construction makes every inner-to-middle distance exactly `radius`, so
    the cluster has just two nearest-neighbour distances.
    """
    directions = icosahedron_directions()
    species: list[str] = []
    coords: list[list[float]] = []
    for direction in directions:
        species.append(inner)
        coords.append(list(direction * radius))
    for i, j in icosahedron_edges(directions):
        species.append(middle)
        coords.append(list((directions[i] + directions[j]) * radius))
    for direction in directions:
        species.append(outer)
        coords.append(list(direction * 2.0 * radius))
    return Molecule(species, coords)


def anti_dihaloethane(first: str, second: str, first_bond: float,
                      second_bond: float) -> Molecule:
    """anti-1,2-CHXY-CHXY: an inversion centre at the C-C midpoint and nothing else.

    Every atom on one carbon has a partner obtained by negating its position, so
    the group is Ci. The three substituents on each carbon are deliberately all
    different, which is what removes the C2 a symmetric pattern would leave.
    """
    cc, ch = 1.530, 1.094
    half = cc / 2
    tilt = 180.0 - TETRAHEDRAL_DEG
    species = ["C", "C"]
    coords = [[0.0, 0.0, half], [0.0, 0.0, -half]]
    for element, bond, azimuth in ((first, first_bond, 0.0), (second, second_bond, 120.0),
                                   ("H", ch, 240.0)):
        top = _spherical(bond, tilt, azimuth)
        species += [element, element]
        coords.append([top[0], top[1], half + top[2]])
        coords.append([-top[0], -top[1], -half - top[2]])
    return Molecule(species, coords)


# --- the tables --------------------------------------------------------------

CRYSTALS: tuple[CrystalSpec, ...] = (
    CrystalSpec(
        stem="CsCl", space_group=221, symbol="Pm-3m",
        lattice=Lattice.cubic(4.123),
        species=("Cs", "Cl"), coords=((0.0, 0.0, 0.0), (0.5, 0.5, 0.5)),
        wyckoff=("1a", "1b"), atoms=2, formula="CsCl",
        source="Wyckoff, Crystal Structures vol. 1 (1963), CsCl type, a = 4.123 A",
        note="The smallest bundled cell: one atom of each kind, all 48 operations.",
    ),
    CrystalSpec(
        stem="Copper", space_group=225, symbol="Fm-3m",
        lattice=Lattice.cubic(3.6149),
        species=("Cu",), coords=((0.0, 0.0, 0.0),),
        wyckoff=("4a",), atoms=4, formula="Cu",
        source="Straumanis & Yu, Acta Cryst. A25 (1969) 676, a = 3.6149 A",
        note="Face-centred cubic close packing.",
    ),
    CrystalSpec(
        stem="Iron-alpha", space_group=229, symbol="Im-3m",
        lattice=Lattice.cubic(2.8665),
        species=("Fe",), coords=((0.0, 0.0, 0.0),),
        wyckoff=("2a",), atoms=2, formula="Fe",
        source="Wyckoff, Crystal Structures vol. 1 (1963), bcc iron, a = 2.8665 A",
        note="Body-centred cubic: the centring translation is visible on its own.",
    ),
    CrystalSpec(
        stem="Iron-gamma", space_group=225, symbol="Fm-3m",
        lattice=Lattice.cubic(3.5910),
        species=("Fe",), coords=((0.0, 0.0, 0.0),),
        wyckoff=("4a",), atoms=4, formula="Fe",
        source="Wyckoff, Crystal Structures vol. 1 (1963), gamma iron, a = 3.5910 A",
        note="Face-centred cubic gamma iron (austenite): a direct contrast with bcc alpha iron.",
    ),
    CrystalSpec(
        stem="Magnesium", space_group=194, symbol="P6_3/mmc",
        lattice=Lattice.hexagonal(3.2094, 5.2108),
        species=("Mg",), coords=((1 / 3, 2 / 3, 0.25),),
        wyckoff=("2c",), atoms=2, formula="Mg",
        source="Walker & Marezio, Acta Metall. 7 (1959) 769, a = 3.2094, c = 5.2108 A",
        note="Hexagonal close packing, with the 6_3 screw axis.",
    ),
    CrystalSpec(
        stem="Diamond", space_group=227, symbol="Fd-3m",
        lattice=Lattice.cubic(3.5670),
        species=("C",), coords=((0.0, 0.0, 0.0),),
        wyckoff=("8a",), atoms=8, formula="C",
        source="Hom, Kiszenik & Post, J. Appl. Cryst. 8 (1975) 457, a = 3.567 A",
        note=(
            "Richest source of screw axes and d glides in the set. "
            "pymatgen's Fd-3m origin choice puts 8a at (0,0,0), not (1/8,1/8,1/8)."
        ),
    ),
    CrystalSpec(
        stem="Graphite", space_group=194, symbol="P6_3/mmc",
        lattice=Lattice.hexagonal(2.4612, 6.7079),
        species=("C", "C"), coords=((0.0, 0.0, 0.25), (1 / 3, 2 / 3, 0.25)),
        wyckoff=("2b", "2c"), atoms=4, formula="C",
        source="Trucano & Chen, Nature 258 (1975) 136, a = 2.4612, c = 6.7079 A",
        note="Same space group as magnesium on a layered structure.",
    ),
    CrystalSpec(
        stem="Sphalerite", space_group=216, symbol="F-43m",
        lattice=Lattice.cubic(5.4093),
        species=("Zn", "S"), coords=((0.0, 0.0, 0.0), (0.25, 0.25, 0.25)),
        wyckoff=("4a", "4c"), atoms=8, formula="ZnS",
        source="Skinner, Am. Mineral. 46 (1961) 1399, a = 5.4093 A",
        note="Zinc blende: diamond's arrangement without the inversion centre.",
    ),
    CrystalSpec(
        stem="Zincite", space_group=186, symbol="P6_3mc",
        lattice=Lattice.hexagonal(3.2495, 5.2069),
        species=("Zn", "O"), coords=((1 / 3, 2 / 3, 0.0), (1 / 3, 2 / 3, 0.3819)),
        wyckoff=("2b", "2b"), atoms=4, formula="ZnO",
        source="Abrahams & Bernstein, Acta Cryst. B25 (1969) 1233, a = 3.2495, c = 5.2069 A",
        note="Wurtzite: polar 6mm, so mirrors and glides but no inversion.",
    ),
    CrystalSpec(
        stem="Fluorite", space_group=225, symbol="Fm-3m",
        lattice=Lattice.cubic(5.4626),
        species=("Ca", "F"), coords=((0.0, 0.0, 0.0), (0.25, 0.25, 0.25)),
        wyckoff=("4a", "8c"), atoms=12, formula="CaF2",
        source="Wyckoff, Crystal Structures vol. 1 (1963), fluorite type, a = 5.4626 A",
        note="Two site symmetries (m-3m and -43m) in one space group.",
    ),
    CrystalSpec(
        stem="Rutile", space_group=136, symbol="P4_2/mnm",
        lattice=Lattice.tetragonal(4.5937, 2.9587),
        species=("Ti", "O"), coords=((0.0, 0.0, 0.0), (0.3053, 0.3053, 0.0)),
        wyckoff=("2a", "4f"), atoms=6, formula="TiO2",
        source="Howard, Sabine & Dickson, Acta Cryst. B47 (1991) 462, a = 4.5937, c = 2.9587 A",
        note="4_2 screw axis and n glide in a six-atom cell.",
    ),
    CrystalSpec(
        stem="Anatase", space_group=141, symbol="I4_1/amd",
        lattice=Lattice.tetragonal(3.7842, 9.5146),
        species=("Ti", "O"), coords=((0.0, 0.0, 0.0), (0.0, 0.0, 0.2081)),
        wyckoff=("4a", "8e"), atoms=12, formula="TiO2",
        source="RSC ESI c7ta05852g: anatase TiO2, a = 3.7842, c = 9.5146 A, O z = 0.2081",
        note="The tetragonal TiO2 polymorph: contrast it with rutile's P4_2/mnm cell.",
    ),
    CrystalSpec(
        stem="Perovskite", space_group=221, symbol="Pm-3m",
        lattice=Lattice.cubic(3.905),
        species=("Sr", "Ti", "O"),
        coords=((0.0, 0.0, 0.0), (0.5, 0.5, 0.5), (0.5, 0.5, 0.0)),
        wyckoff=("1a", "1b", "3c"), atoms=5, formula="SrTiO3",
        source="Abramov et al., Acta Cryst. B51 (1995) 942, SrTiO3 cubic, a = 3.905 A",
        note="The ideal cubic perovskite; BaTiO3 shows the distorted R3m relative.",
    ),
    CrystalSpec(
        stem="Quartz", space_group=152, symbol="P3_121",
        lattice=Lattice.hexagonal(4.916, 5.4054),
        species=("Si", "O"),
        coords=((0.4697, 0.0, 1 / 3), (0.4135, 0.2669, 0.1191)),
        wyckoff=("3a", "6c"), atoms=9, formula="SiO2",
        source="Levien, Prewitt & Weidner, Am. Mineral. 65 (1980) 920, 1 atm: a = 4.916, c = 5.4054 A",
        note="Chiral: a 3_1 screw axis and no mirror at all.",
    ),
    CrystalSpec(
        stem="Pyrite", space_group=205, symbol="Pa-3",
        lattice=Lattice.cubic(5.4179),
        species=("Fe", "S"), coords=((0.0, 0.0, 0.0), (0.385, 0.385, 0.385)),
        wyckoff=("4a", "8c"), atoms=12, formula="FeS2",
        source="Brostigen & Kjekshus, Acta Chem. Scand. 23 (1969) 2186, a = 5.4179 A",
        note="Cubic without a 4-fold axis: m-3, with a glides and -3 axes.",
    ),
)


MOLECULES: tuple[MoleculeSpec, ...] = (
    MoleculeSpec(
        stem="ethane", point_group="D3d", atoms=8,
        build=lambda: ethane(60.0),
        source="C-C 1.535 A, C-H 1.094 A, tetrahedral angles; staggered conformer",
        note="The eclipsed conformer would be D3h — same molecule, different group.",
    ),
    MoleculeSpec(
        stem="hydrogen_peroxide", point_group="C2", atoms=4,
        build=hydrogen_peroxide,
        source="O-O 1.475 A, O-H 0.950 A, O-O-H 94.8 deg, dihedral 111.5 deg (gas phase)",
        note="Pure rotation only: no mirror, no inversion.",
    ),
    MoleculeSpec(
        stem="trans_dichloroethene", point_group="C2h", atoms=6,
        build=lambda: planar_ethene_frame("Cl", 1.725),
        source="C=C 1.330 A, C-Cl 1.725 A, C-H 1.087 A, 121 deg; trans isomer",
        note="C2 perpendicular to the plane, sigma_h in it, and the inversion they imply.",
    ),
    MoleculeSpec(
        stem="chlorofluoromethane", point_group="Cs", atoms=5,
        build=lambda: substituted_methane(("Cl", "F", "H", "H"), (1.759, 1.378, 1.087, 1.087)),
        source="C-Cl 1.759 A, C-F 1.378 A, C-H 1.087 A, tetrahedral",
        note="A single mirror plane, which swaps the two hydrogens.",
    ),
    MoleculeSpec(
        stem="bromochlorofluoromethane", point_group="C1", atoms=5,
        build=lambda: substituted_methane(
            ("Br", "Cl", "F", "H"), (1.966, 1.759, 1.378, 1.087)),
        source="C-Br 1.966 A, C-Cl 1.759 A, C-F 1.378 A, C-H 1.087 A, tetrahedral",
        note="No symmetry at all: the chiral counterexample. No quiz can use it.",
    ),
    MoleculeSpec(
        stem="phosphorus_pentafluoride", point_group="D3h", atoms=6,
        build=lambda: trigonal_bipyramid("P", "F", "F", 1.577, 1.534),
        source="P-F axial 1.577 A, equatorial 1.534 A (gas-phase electron diffraction)",
        note="D3h as a trigonal bipyramid, against planar BF3's D3h.",
    ),
    MoleculeSpec(
        stem="bromine_pentafluoride", point_group="C4v", atoms=6,
        build=lambda: square_pyramid("Br", "F", "F", 1.689, 1.774, 174.5 - 90.0),
        source="Br-F apical 1.689 A, basal 1.774 A, F(ap)-Br-F(bas) 84.5 deg",
        note="Square pyramid: a 4-fold axis with mirrors but no horizontal one.",
    ),
    MoleculeSpec(
        stem="anti_dibromodichloroethane", point_group="Ci", atoms=8,
        build=lambda: anti_dihaloethane("Br", "Cl", 1.966, 1.759),
        source="C-C 1.530 A, C-Br 1.966 A, C-Cl 1.759 A, C-H 1.094 A, anti conformer",
        note="Inversion centre and nothing else.",
    ),
    # --- icosahedral clusters ------------------------------------------------
    # Building blocks of icosahedral quasicrystals, NOT quasicrystals: a
    # quasicrystal has no unit cell, so it cannot go down the crystal path at
    # all (spglib would answer with some space group regardless). These are
    # finite clusters, and a finite cluster is free to have the 5-fold axes a
    # periodic lattice cannot. Analysis-mode only, since the quizzes have no
    # name for a 5-fold rotation.
    MoleculeSpec(
        stem="al12w_icosahedron", point_group="Ih", atoms=13,
        build=lambda: centered_icosahedron("W", "Al", 2.760),
        source=(
            "Adam & Rich, Acta Cryst. 7 (1954) 813 — WAl12 (cI26, Im-3) is built "
            "from W-centred Al12 icosahedra. Idealised to a regular icosahedron "
            "here; in the crystal the site symmetry is only m-3."
        ),
        note="The smallest object with 5-fold axes: 6 C5, 10 C3, 15 C2, i.",
        analysis_only=True,
    ),
    MoleculeSpec(
        stem="mackay_icosahedron", point_group="Ih", atoms=54,
        build=lambda: mackay_icosahedron("Al", "Al", "Mn", 2.760),
        source=(
            "Mackay, Acta Cryst. 15 (1962) 916 (the Mackay icosahedron); "
            "Elser & Henley, Phys. Rev. Lett. 55 (1985) 2883 (the 54-atom "
            "Mackay icosahedron as the building block of alpha-Al-Mn-Si and of "
            "icosahedral Al-Mn). Shell ratios 1 : sqrt(2+2/sqrt5) : 2, scaled "
            "to a 2.76 A shortest distance."
        ),
        note=(
            "The centre is vacant: the inner shell is too tight to hold an atom, "
            "which is why the count is 54 and not the metallic magic number 55."
        ),
        analysis_only=True,
    ),
)


# --- writing and verification -------------------------------------------------


class VerificationError(Exception):
    """A written example did not come back as the table says it should."""


def build_crystal(spec: CrystalSpec) -> Structure:
    return Structure.from_spacegroup(
        spec.space_group, spec.lattice, list(spec.species), [list(c) for c in spec.coords]
    )


def crystal_header(spec: CrystalSpec) -> str:
    sites = ", ".join(
        f"{element} {wyckoff} {tuple(round(value, 5) for value in coords)}"
        for element, wyckoff, coords in zip(spec.species, spec.wyckoff, spec.coords)
    )
    lines = [
        "# Generated by tools/generate_example_structures.py -- do not hand-edit.",
        f"# {spec.stem}: space group {spec.space_group} {spec.symbol}",
        f"# sites: {sites}",
        f"# source: {spec.source}",
    ]
    if spec.note:
        lines.append(f"# note: {spec.note}")
    return "\n".join(lines) + "\n"


def write_crystal(spec: CrystalSpec, cif_dir: Path) -> Path:
    path = cif_dir / f"{spec.stem}.cif"
    body = str(CifWriter(build_crystal(spec), symprec=0.01))
    path.write_text(crystal_header(spec) + body, encoding="utf-8")
    return path


def write_molecule(spec: MoleculeSpec, molecule_dir: Path) -> Path:
    path = molecule_dir / f"{spec.stem}.xyz"
    molecule = spec.build()
    lines = [str(len(molecule)), f"{spec.stem} {spec.point_group} -- {spec.source}"]
    for site in molecule:
        x, y, z = site.coords
        lines.append(f"{site.specie.symbol} {x:.6f} {y:.6f} {z:.6f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def verify_crystal(spec: CrystalSpec, path: Path) -> None:
    """Re-analyse the written CIF the way the app will read it."""
    result = analyze_cif(path)
    problems = []
    if result.space_group.number != spec.space_group:
        problems.append(f"space group {result.space_group.number} != {spec.space_group}")
    if result.space_group.international != spec.symbol:
        problems.append(f"symbol {result.space_group.international!r} != {spec.symbol!r}")
    if len(result.structure.atoms) != spec.atoms:
        problems.append(f"{len(result.structure.atoms)} atoms != {spec.atoms}")
    # Composition catches a Wyckoff position that multiplied out to the wrong
    # site count, which an atom count alone can miss.
    if Composition(result.structure.formula) != Composition(spec.formula):
        problems.append(f"composition {result.structure.formula!r} != {spec.formula!r}")
    if problems:
        raise VerificationError(f"{path.name}: " + "; ".join(problems))


# The app analyses molecules at 0.3 A, which is loose enough to call a slightly
# wrong geometry more symmetric than it is (a mis-built trans-dichloroethene
# reads as D2h instead of C2h). Requiring the same answer at a tight tolerance
# separates "built correctly" from "rounded into the right group".
TIGHT_MOLECULE_TOLERANCE = 0.05


def verify_molecule(spec: MoleculeSpec, path: Path) -> None:
    result = analyze_molecule_file(path)
    tight = analyze_molecule_file(path, tolerance=TIGHT_MOLECULE_TOLERANCE)
    problems = []
    # The table declares whether a structure is analysis-only; the computed flag
    # decides it. Requiring them to agree means nobody can add a structure that
    # silently vanishes from every quiz, and nobody can mark one analysis-only
    # that the quizzes would happily have used.
    render_data = render_data_to_dict(render_data_from_molecule(result))
    computed_analysis_only = beyond_quiz_vocabulary(render_data)
    if computed_analysis_only != spec.analysis_only:
        problems.append(
            f"analysis_only={spec.analysis_only} but the quiz vocabulary "
            f"{'cannot' if computed_analysis_only else 'can'} name every operation"
        )
    if result.point_group.symbol != spec.point_group:
        problems.append(f"point group {result.point_group.symbol!r} != {spec.point_group!r}")
    if tight.point_group.symbol != spec.point_group:
        problems.append(
            f"point group at {TIGHT_MOLECULE_TOLERANCE} A is "
            f"{tight.point_group.symbol!r}: the geometry only reaches "
            f"{spec.point_group} through the default tolerance"
        )
    if len(result.molecule.atoms) != spec.atoms:
        problems.append(f"{len(result.molecule.atoms)} atoms != {spec.atoms}")
    if problems:
        raise VerificationError(f"{path.name}: " + "; ".join(problems))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cif-dir", type=Path, default=DEFAULT_CIF_DIR)
    parser.add_argument("--molecule-dir", type=Path, default=DEFAULT_MOLECULE_DIR)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Verify the files already in the example directories without writing.",
    )
    args = parser.parse_args()

    for spec in CRYSTALS:
        if spec.stem in PROTECTED:
            raise SystemExit(f"{spec.stem} is hand-authored and must not be generated")
        path = args.cif_dir / f"{spec.stem}.cif"
        if not args.check_only:
            path = write_crystal(spec, args.cif_dir)
        verify_crystal(spec, path)
        print(f"crystal\t{path.relative_to(PROJECT_ROOT)}\t{spec.symbol}\t{spec.atoms} atoms")

    for spec in MOLECULES:
        path = args.molecule_dir / f"{spec.stem}.xyz"
        if not args.check_only:
            path = write_molecule(spec, args.molecule_dir)
        verify_molecule(spec, path)
        print(f"molecule\t{path.relative_to(PROJECT_ROOT)}\t{spec.point_group}\t{spec.atoms} atoms")

    verb = "Verified" if args.check_only else "Wrote"
    print(f"{verb} {len(CRYSTALS)} crystal and {len(MOLECULES)} molecule examples.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
