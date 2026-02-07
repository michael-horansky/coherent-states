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
    basis = '6-31g**',
    unit = "Bohr"
)

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
    "Li2" : li2_mol
    }
#benchmark_molecules = {
#    "Li2" : li2_mol,
#    "N2" : n2_mol
#    }

plot_x, plot_y = functions.subplot_dimensions(len(benchmark_molecules))

i = 0
for mol_name, mol in benchmark_molecules.items():
    plt.subplot(plot_y, plot_x, i+1)
    plt.title(f"Molecule {mol_name}")
    plt.xlabel("No. configs")
    plt.ylabel("E [Hartree]")

    mol_solver = ground_state_solver(f"bench_mol_{mol_name}_qubit")
    mol_solver.initialise_molecule(mol)
    N_vals_t, energy_levels_t = mol_solver.find_ground_state("sampling", N = 30, lamb = None, sampling_method = "highest_orbital", CS = "Qubit", assume_spin_symmetry = False)
    #N_vals_q, energy_levels_q = mol_solver.find_ground_state("sampling", N = 2, lamb = None, delta = 1e-2, CS = "Qubit")

    # Now for the reference full CI
    hf = scf.RHF(mol)          # Create the HF object
    hf.kernel()  # Perform the SCF calculation
    cisolver = fci.FCI(mol, hf.mo_coeff)  # Full CI solver with HF orbitals
    ci_energy, ci_wavefunction = cisolver.kernel()
    print("Full CI ground state energy =", ci_energy)

    print(mol_solver.print_diagnostic_log())

    plt.plot(N_vals_t, energy_levels_t, "x", label = "Qubit")
    #plt.plot(N_vals_q, energy_levels_q, "x", label = "Qubit")

    plt.axhline(y = ci_energy, label = "full CI")

    plt.legend()

    i += 1

plt.tight_layout()
plt.show()
