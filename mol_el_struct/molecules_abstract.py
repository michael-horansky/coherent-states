# This is a standardised file which defines common molecules, so that other source files can just reference this one

from pyscf import gto

class AbstractMolecule():

    def __init__(self, atoms, basis, unit, spin = None):
        self.atoms = atoms # list of ["label", pos x, pos y, pos z]
        self.basis = basis
        self.unit = unit
        self.spin = spin



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
    atoms = [["Li", 0.0, 0.0, -2.5], ["Li", 0.0, 0.0, -2.5]],
    basis = 'sto-3g',
    unit = 'Bohr'
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
    spin = 1
    )


# ------------------------------ N2

# N2: bond length ≈ 1.098 Å = 2.0749191355 Bohr
# placed symmetrically about origin: half-distance ≈ 1.0374595677 Bohr
n2_am = AbstractMolecule(
    atoms = [["N", 0.0, 0.0, -1.0374595677], ["N", 0.0, 0.0, 1.0374595677]],
    basis = 'sto-3g',
    unit = 'Bohr'
    )

# ---------------------------------- C2

# C2: bond length ≈ 1.2425 Å = 2.3479845408 Bohr
# placed symmetrically about origin: half-distance ≈ 1.1739922704 Bohr
c2_am = AbstractMolecule(
    atoms = [["C", 0.0, 0.0, -1.1739922704], ["C", 0.0, 0.0, 1.1739922704]],
    basis = 'sto-3g',
    unit = 'Bohr'
    )

# -------------------------------- NO

# NO: bond length ≈ 1.151 Å = 2.1750746129 Bohr
# placed symmetrically about origin: half-distance ≈ 1.0875373064 Bohr
no_am = AbstractMolecule(
    atoms = [["N", 0.0, 0.0, -1.0875373064], ["O", 0.0, 0.0, 1.0875373064]],
    basis = 'sto-3g',
    unit = 'Bohr',
    spin = 1
    )


mol_catalogue = {
    'Li2' : li2_am,
    'BeH' : beh_am,
    'N2' : n2_am,
    'C2' : c2_am,
    'NO' : no_am
    }




