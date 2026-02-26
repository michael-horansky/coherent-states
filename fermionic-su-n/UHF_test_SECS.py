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
    "NO" : no_mol
#    "N2" : n2_mol
    }

trimmed_ground_state_full_ci = {
    "Li2" : -14.65464717398042,
    "C2" : -74.57429774053233

    }


for mol_name, mol in benchmark_molecules.items():

    mol_solver = ground_state_solver(f"{mol_name}_UHF_SECS")
    mol_solver.initialise_molecule(mol, HF_method = "UHF")
    #mol_solver.load_data(["self_analysis", "measured_datasets"])

    # Self-analysis methods
    mol_solver.pyscf_full_CI()
    #cur_SECS_heatmap, cur_SECS_restricted_energy = mol_solver.solve_on_single_excitation_closed_shell()

    mol_solver.print_UHF_low_excitation_info()
    #top_sim_exc_states = mol_solver.get_top_simultaneously_excited_states(10)
    #mol_solver.log.enter(f"Displaying top {len(top_sim_exc_states)} simultaneously-excited states...")
    #for i in range(len(top_sim_exc_states)):
    #    mol_solver.log.write(f"  {i+1}) Coef = {top_sim_exc_states[i][0]:0.4f}; occ. = {top_sim_exc_states[i][1]} (prom {top_sim_exc_states[i][2]})")
    #mol_solver.log.exit()





    # Sampling methods
    #N_vals_width, energy_levels_width = mol_solver.find_ground_state("SEGS_width", N = 80, N_sub = 10, dataset_label = "SEGS width")
    #N_vals_phase, energy_levels_phase = mol_solver.find_ground_state("SEGS_phase", N = 3, N_sub = 1, dataset_label = "SEGS phase")


    # Plotting and saving
    #mol_solver.print_singlet_info()

    ref_energies = []
    if mol_name in trimmed_ground_state_full_ci:
        ref_energies.append({"E" : trimmed_ground_state_full_ci[mol_name], "label" : "trimmed CI", "color" : functions.ref_energy_colors["trimmed CI"], "linestyle" : "dashed"})

    #mol_solver.plot_datasets(ref_energies)
    #mol_solver.save_data()
    mol_solver.log.close_journal()

