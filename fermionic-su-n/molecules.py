# This is a standardised file which defines common molecules, so that other source files can just reference this one

from pyscf import gto

# -------------------------- Li2

# Li2: bond length ≈ 2.673 Å = 5.0512375675 Bohr
# placed symmetrically about origin: half-distance ≈ 2.5256187838 Bohr
li2_mol = gto.Mole()
li2_mol.build(
    atom = '''
    Li   0.0000000000   0.0000000000  -2.5
    Li   0.0000000000   0.0000000000   2.5
    ''',
    basis = 'sto-3g',
    unit = "Bohr"
)

# bigger basis: 6-31g**

# Set to +-2.5 Bohr exactly to match Dima's Hamiltonian. The true value is +-2.5256187838

# -------------------------------- BeH

# BeH: bond length ≈ 1.342 Å = 2.5360122767 Bohr
# place Be at origin and H along +z
beh_mol = gto.Mole()
beh_mol.build(
    atom = '''
    Be   0.0000000000   0.0000000000   0.0000000000
    H    0.0000000000   0.0000000000   2.5360122767
    ''',
    basis = 'sto-3g',
    unit = "Bohr",
    spin = 1  # BeH is a doublet (5 electrons) -> 1 unpaired electron
)



# ------------------------------ N2

# N2: bond length ≈ 1.098 Å = 2.0749191355 Bohr
# placed symmetrically about origin: half-distance ≈ 1.0374595677 Bohr
n2_mol = gto.Mole()
n2_mol.build(
    atom = '''
    N    0.0000000000   0.0000000000  -1.0374595677
    N    0.0000000000   0.0000000000   1.0374595677
    ''',
    basis = 'sto-3g',
    unit = "Bohr"
)

# ---------------------------------- C2

# C2: bond length ≈ 1.2425 Å = 2.3479845408 Bohr
# placed symmetrically about origin: half-distance ≈ 1.1739922704 Bohr
c2_mol = gto.Mole()
c2_mol.build(
    atom = '''
    C    0.0000000000   0.0000000000  -1.1739922704
    C    0.0000000000   0.0000000000   1.1739922704
    ''',
    basis = 'sto-3g',
    unit = "Bohr"
)

# -------------------------------- NO

# NO: bond length ≈ 1.151 Å = 2.1750746129 Bohr
# placed symmetrically about origin: half-distance ≈ 1.0875373064 Bohr
no_mol = gto.Mole()
no_mol.build(
    atom = '''
    N    0.0000000000   0.0000000000  -1.0875373064
    O    0.0000000000   0.0000000000   1.0875373064
    ''',
    basis = 'sto-3g',
    unit = "Bohr",
    spin = 1  # NO is a doublet (15 electrons) -> 1 unpaired electron
)






