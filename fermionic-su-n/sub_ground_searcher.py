import numpy as np
import matplotlib.pyplot as plt

from pyscf import gto, scf, fci
from krylov_ground_state import ground_state_solver

import functions


# ------------------------------ N2

# N2: bond length ≈ 1.098 Å = 2.0749191355 Bohr
# placed symmetrically about origin: half-distance ≈ 1.0374595677 Bohr
n2_mol = gto.Mole()
n2_mol.build(
    atom = '''
    N    0.0000000000   0.0000000000  -1.0374595677
    N    0.0000000000   0.0000000000   1.0374595677
    ''',
    basis = 'sto-3g'
)


mol_solver = ground_state_solver(f"debugger_mol_N2")
mol_solver.initialise_molecule(n2_mol)
mol_solver.search_for_states("Thouless", "highest_orbital", lambda x : (mol_solver.H_overlap(x, x) < mol_solver.ci_energy - mol_solver.mol.energy_nuc()))
