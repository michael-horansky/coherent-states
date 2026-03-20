import numpy as np
import matplotlib.pyplot as plt

from pyscf import gto, scf, fci
from molecules import *
from mol_solver_degenerate import ground_state_solver
from class_Semaphor import Semaphor

from coherent_states.CS_Thouless import CS_Thouless
from coherent_states.CS_sample import CS_sample

import functions



benchmark_molecules = {
    "Li2" : li2_mol
#    "N2" : n2_mol
    }

trimmed_ground_state_full_ci = {
    "Li2" : -14.65464717398042,
    "C2" : -74.57429774053233

    }


for mol_name, mol in benchmark_molecules.items():

    mol_solver = ground_state_solver(f"{mol_name}_SECS_doubled_states")
    mol_solver.initialise_molecule(mol)
    #mol_solver.load_data(["self_analysis", "measured_datasets"])
    #mol_solver.print_singlet_info()

    # Self-analysis methods
    mol_solver.full_CI_sol()
    mol_solver.find_LE_solution("SECS")


    # Sampling methods
    #N_vals_width, energy_levels_width = mol_solver.find_ground_state("SEGS_width", N = 80, N_sub = 10, dataset_label = "SEGS width")
    #N_vals_phase, energy_levels_phase = mol_solver.find_ground_state("SEGS_phase", N = 3, N_sub = 1, dataset_label = "SEGS phase")


    # Plotting and saving

    ref_energies = []
    if mol_name in trimmed_ground_state_full_ci:
        ref_energies.append({"E" : trimmed_ground_state_full_ci[mol_name], "label" : "trimmed CI", "color" : functions.ref_energy_colors["trimmed CI"], "linestyle" : "dashed"})

    mol_solver.plot_datasets(ref_energies)
    mol_solver.save_data()

