# -----------------------------------------------------------------------------
# ----------------------------- class MBlueprint ------------------------------
# -----------------------------------------------------------------------------
# MBlueprint (molecular blueprint) describes the physical properties of a
# molecule apart from its atomic geometry.
# The class method get_gto_Mole builds an instance of gto.Mole using the
# provided instance of MGeometry

from pyscf import gto


class MBlueprint():

    def __init__(self, name, atoms, basis, unit, spin = None, label = None):
        self.name = name
        self.atoms = atoms # ordered list of atom labels
        self.basis = basis
        self.unit = unit
        self.spin = spin

        self.N_a = len(self.atoms) # number of atoms

        # "label" is a human-readable string to be used in pyplot. It can use latex formatting.
        if label is None:
            self.label = self.name
        else:
            self.label = label

        # We create a hr_atoms list which labels repeated atoms by their index
        self.hr_atoms = []
        atom_multiplicity = {}
        for i in range(len(self.atoms)):
            if self.atoms[i] not in atom_multiplicity.keys():
                atom_multiplicity[self.atoms[i]] = 1
                self.hr_atoms.append(self.atoms[i])
            else:
                atom_multiplicity[self.atoms[i]] += 1
                self.hr_atoms.append(f"{self.atoms[i]} ({atom_multiplicity[self.atoms[i]]})")

        self.mol_meta = {
            "name" : self.name,
            "atoms" : self.atoms,
            "basis" : self.basis,
            "unit" : self.unit,
            "spin" : self.spin,
            "label" : self.label
            }


    def get_atoms(self, g):
        # g is an instance of MGeometry
        # The position of the first atom is taken as 0, 0, 0
        # The output is a string which may be passed to PySCF
        res = ""

        res += f"{self.atoms[0]}  0  0  0\n"

        for i in range(len(self.atoms) - 1):
            res += f"{self.atoms[i + 1]}   {g.r[i][0]}   {g.r[i][1]}   {g.r[i][2]}\n"
        return(res)

    def get_gto_Mole(self, g):
        # g is an instance of MGeometry
        # The output is an instance of gto.Mole initialised according to the
        # blueprint and the geometry
        res = gto.Mole()
        if self.spin is None:
            res.build(
                atom = self.get_atoms(g),
                basis = self.basis,
                unit = self.unit
                )
        else:
            res.build(
                atom = self.get_atoms(g),
                basis = self.basis,
                unit = self.unit,
                spin = self.spin
                )
        return(res)


bp_catalogue = {}

# Li2

bp_catalogue["Li2"] = MBlueprint(
        name = "Li2",
        atoms = ["Li", "Li"],
        basis = "sto-3g",
        unit = "Bohr",
        label = r'${\rm Li}_2$'
    )

# BeH2

bp_catalogue["BeH2"] = MBlueprint(
        name = "BeH2",
        atoms = ["Be", "H", "H"],
        basis = "sto-3g",
        unit = "Bohr",
        spin = 0,
        label = r'${\rm BeH}_2$'
    )

