import numpy as np
import matplotlib.pyplot as plt

from pyscf import gto, scf, fci
from mol_solver_degenerate import ground_state_solver

import functions

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

benchmark_molecules = {
    "C2" : c2_mol
    } # No BeH or NO because N_alpha doesn't equal N_beta

# First, we find the number of MOs and electrons per spin-subspace for each benchmark molecule
for mol_name, mol in benchmark_molecules.items():

    mol_solver = ground_state_solver(f"bench_mol_{mol_name}_qubit")
    mol_solver.initialise_molecule(mol)

    # For N2 and C2 both, they have 10 MOs.
    # Let's calculate the trimmed full CI energy when retaining only the bottom 2 unoccupied MOs

    trimmed_number_of_MOs = min(mol_solver.mol.nao, mol_solver.S_alpha + 2)
    trimmed_ground_state_full_ci = mol_solver.find_ground_state_on_full_ci(trim_M = trimmed_number_of_MOs)
    print(f"Trimmed ground state energy (MOs = {trimmed_number_of_MOs}): {trimmed_ground_state_full_ci}")
