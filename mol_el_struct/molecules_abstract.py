# This is a standardised file which defines common molecules, so that other source files can just reference this one

import numpy as np
from pyscf import gto

class AbstractMolecule():

    def __init__(self, atoms, basis, unit, spin = None, label = None):
        self.atoms = atoms # list of ["label", pos x, pos y, pos z]
        self.basis = basis
        self.unit = unit
        self.spin = spin

        # "label" is a human-readable string to be used in pyplot. It can use latex formatting.
        self.label = label


    def get_atoms(self, scale = 1.0):
        res = ""
        for atom in self.atoms:
            res += f"{atom[0]}   {atom[1] * scale}   {atom[2] * scale}   {atom[3] * scale}\n"
        return(res)

    def get_gto_Mole(self, scale = 1.0):
        res = gto.Mole()
        if self.spin is None:
            res.build(
                atom = self.get_atoms(scale),
                basis = self.basis,
                unit = self.unit
                )
        else:
            res.build(
                atom = self.get_atoms(scale),
                basis = self.basis,
                unit = self.unit,
                spin = self.spin
                )
        return(res)


# -------------------------- Li2

# Li2: bond length ≈ 2.673 Å = 5.0512375675 Bohr
# placed symmetrically about origin: half-distance ≈ 2.5256187838 Bohr
li2_am = AbstractMolecule(
    atoms = [["Li", 0.0, 0.0, -2.5], ["Li", 0.0, 0.0, 2.5]],
    basis = 'sto-3g',
    unit = 'Bohr',
    label = r'${\rm Li}_2$'
    )

# bigger basis: 6-31g**

# Set to +-2.5 Bohr exactly to match Dima's Hamiltonian. The true value is +-2.5256187838

# -------------------------------- BeH

# BeH: bond length ≈ 1.342 Å = 2.5360122767 Bohr
# place Be at origin and H along +z
beh_am = AbstractMolecule(
    atoms = [["Be", 0.0, 0.0, 0.0], ["H", 0.0, 0.0, 2.5360122767]],
    basis = 'sto-3g',
    unit = 'Bohr',
    spin = 1,
    label = r'BeH'
    )


# ------------------------------ N2

# N2: bond length ≈ 1.098 Å = 2.0749191355 Bohr
# placed symmetrically about origin: half-distance ≈ 1.0374595677 Bohr
n2_am = AbstractMolecule(
    atoms = [["N", 0.0, 0.0, -1.0374595677], ["N", 0.0, 0.0, 1.0374595677]],
    basis = 'sto-3g',
    unit = 'Bohr',
    label = r'${\rm N}_2$'
    )

# ---------------------------------- C2

# C2: bond length ≈ 1.2425 Å = 2.3479845408 Bohr
# placed symmetrically about origin: half-distance ≈ 1.1739922704 Bohr
c2_am = AbstractMolecule(
    atoms = [["C", 0.0, 0.0, -1.1739922704], ["C", 0.0, 0.0, 1.1739922704]],
    basis = 'sto-3g',
    unit = 'Bohr',
    label = r'${\rm C}_2$'
    )

# -------------------------------- NO

# NO: bond length ≈ 1.151 Å = 2.1750746129 Bohr
# placed symmetrically about origin: half-distance ≈ 1.0875373064 Bohr
no_am = AbstractMolecule(
    atoms = [["N", 0.0, 0.0, -1.0875373064], ["O", 0.0, 0.0, 1.0875373064]],
    basis = 'sto-3g',
    unit = 'Bohr',
    spin = 1,
    label = r'NO'
    )


# -------------------------------- BeH2

# BeH2: BeH bonds are of length 1.334 A = 2.5208947 Bohr
# linear geometry: H - Be - H
BeH2_am = AbstractMolecule(
    atoms = [["H", 0.0, 0.0, -2.5208947], ["Be", 0.0, 0.0, 0.0], ["H", 0.0, 0.0, 2.5208947]],
    basis = 'sto-3g',
    unit = 'Bohr',
    spin = 0,
    label = r'${\rm BeH}_2$'
    )

# -------------------------------- F2

# F2: bond length 1.42 A = 2.683411 Bohr
# placed symmetrically about origin: half-distance ≈ 1.3417055 Bohr
F2_am = AbstractMolecule(
    atoms = [["F", 0.0, 0.0, -1.3417055], ["F", 0.0, 0.0, 1.3417055]],
    basis = 'sto-3g',
    unit = 'Bohr',
    spin = 0,
    label = r'${\rm F}_2$'
    )

# -------------------------------- H2O

# H2O: bond length between hydrogen and oxygen is 0.9584 A = 1.8111135 Bohr
# The H - O - H angle is 104.45 degrees = 1.8229964 rad
b = 1.8111135
ha = 1.8229964 / 2.0 # half angle
H2O_am = AbstractMolecule(
    atoms = [["O", 0.0, 0.0, 0.0], ["H", b * np.cos(ha), 0.0, -b * np.sin(ha)], ["H", b * np.cos(ha), 0.0, b * np.sin(ha)]],
    basis = 'sto-3g',
    unit = 'Bohr',
    label = r'${\rm H}_2 {\rm O}$'
    )


mol_catalogue = {
    'Li2' : li2_am,
    'BeH' : beh_am,
    'N2' : n2_am,
    'C2' : c2_am,
    'NO' : no_am,
    'BeH2' : BeH2_am,
    'F2' : F2_am,
    'H2O' : H2O_am
    }




