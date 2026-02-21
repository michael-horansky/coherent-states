import numpy as np
import matplotlib.pyplot as plt

from pyscf import gto, scf, fci
from molecules import *
from mol_solver_degenerate import ground_state_solver

import functions



benchmark_molecules = {
    #"Li2" : li2_mol
    "C2" : c2_mol
    }
#benchmark_molecules = {
#    "Li2" : li2_mol,
#    "N2" : n2_mol
#    }

trimmed_ground_state_full_ci = {
    "Li2" : -14.65464717398042,
    "C2" : -74.57429774053233

    }

plot_x, plot_y = functions.subplot_dimensions(len(benchmark_molecules))

i = 0
for mol_name, mol in benchmark_molecules.items():
    plt.subplot(plot_y, plot_x, i+1)
    plt.title(f"Molecule {mol_name}")
    plt.xlabel("No. configs")
    plt.ylabel("E [Hartree]")

    mol_solver = ground_state_solver(f"bench_mol_{mol_name}_qubit")
    mol_solver.initialise_molecule(mol)
    """N_vals_t, energy_levels_t = mol_solver.find_ground_state("sampling", N = 20, lamb = None, sampling_method = "highest_orbital_trim", CS = "Thouless", assume_spin_symmetry = False)
    print(mol_solver.print_diagnostic_log())"""
    N_vals_q, energy_levels_q = mol_solver.find_ground_state("sampling", N = 20, lamb = None, sampling_method = "highest_orbital_trim", CS = "Qubit", assume_spin_symmetry = False)
    print(mol_solver.print_diagnostic_log())

    # For Li2, we can just quote the measured trimmed full CI ground state energy
    #trimmed_ground_state_full_ci = -14.65464717398042

    # For C2, we can just quote the measured trimmed full CI ground state energy
    #trimmed_ground_state_full_ci = -74.57429774053233


    #plt.plot(N_vals_t, energy_levels_t, "x", label = "Thouless")
    plt.plot(N_vals_q, energy_levels_q, "x", label = "Qubit")

    plt.axhline(y = mol_solver.ci_energy, label = "full CI")

    if mol_name in trimmed_ground_state_full_ci:
        plt.axhline(y = trimmed_ground_state_full_ci[mol_name], label = "trimmed full CI")

    plt.legend()

    i += 1

plt.tight_layout()
plt.show()
