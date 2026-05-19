import numpy as np
import matplotlib.pyplot as plt

from pyscf import gto, scf, fci
from molecules import *
from mol_solver_degenerate import ground_state_solver
from utils.class_Semaphor import Semaphor

from coherent_states.CS_Thouless import CS_Thouless
from coherent_states.CS_sample import CS_sample

import utils.functions



benchmark_molecules = {
    "NO" : no_mol
#    "N2" : n2_mol
    }

trimmed_ground_state_full_ci = {
    "Li2" : -14.65464717398042,
    "C2" : -74.57429774053233

    }


mol_solver = ground_state_solver(f"LE_RSOPM_moment_matching_added_no_cov")
mol_solver.initialise_molecule(li2_mol, HF_method = "RHF")
mol_solver.load_data(["self_analysis", "measured_datasets"])


"""
subplot_widths = []
for i in range(len(mol_solvers)):
    subplot_widths.append(mol_solvers[i].S_alpha)"""


"""
fig, ax = plt.subplots(1, 1, figsize=(12, 8)) #, figsize=(8, 6)
#fig, axes = plt.subplots(1, 1, figsize=(12, 4), width_ratios=subplot_widths) #, figsize=(8, 6)
fig.suptitle("Matrix elements of qubit transition operators on low-excitation g.s.")
#fig.subplots_adjust(bottom=0, top=1, left=4, right=5

im, _ = mol_solver.plot_LE_reduction_matrix(spin = "a", log_plot = True, signed = False, ax = ax)
ax.title.set_text(f"Molecule $\\text{{Li}}_2$ (spin $\\alpha$ subspace)")

fig.tight_layout()
plt.show()"""

"""
fig, axes = plt.subplots(1, 2, figsize=(12, 8)) #, figsize=(8, 6)
#fig, axes = plt.subplots(1, 1, figsize=(12, 4), width_ratios=subplot_widths) #, figsize=(8, 6)
fig.suptitle("Reduced simultaneous occupancy proportion matrix")
#fig.subplots_adjust(bottom=0, top=1, left=4, right=5)
list_of_axes = list(axes)

#if np.all(np.round( mol_solver.LE_sol["red"].T, 5 ) == np.round( mol_solver.LE_sol["red"], 5 )):
#    print("\nReduction matrix is symmetric!\n")
#else:
#    print("\nERROR: Reduction matrix is not symmetric!\n")

im, _ = mol_solver.plot_LE_RSOPM(spin = "a", log_plot = True, ax = list_of_axes[0])
list_of_axes[0].title.set_text(f"Molecule Li2 (alpha)")
im, _ = mol_solver.plot_LE_RSOPM(spin = "b", log_plot = True, ax = list_of_axes[1])
list_of_axes[1].title.set_text(f"Molecule Li2 (beta)")

fig.tight_layout()
plt.show()"""

"""
fig, ax = plt.subplots(1, 1, figsize=(12, 8)) #, figsize=(8, 6)
#fig, axes = plt.subplots(1, 1, figsize=(12, 4), width_ratios=subplot_widths) #, figsize=(8, 6)
fig.suptitle("Closed-shell reduction matrix analysis")
#fig.subplots_adjust(bottom=0, top=1, left=4, right=5)

im, _ = mol_solver.plot_LE_CSRM(log_plot = True, ax = ax)
ax.title.set_text(f"Molecule Li2")

fig.tight_layout()
plt.show()"""




# Self-analysis methods
#mol_solver.full_CI_sol()

#cur_SECS_heatmap, cur_SECS_restricted_energy = mol_solver.solve_on_single_excitation_closed_shell()

#mol_solver.print_UHF_low_excitation_info()
#top_sim_exc_states = mol_solver.get_top_simultaneously_excited_states(10)
#mol_solver.log.enter(f"Displaying top {len(top_sim_exc_states)} simultaneously-excited states...")
#for i in range(len(top_sim_exc_states)):
#    mol_solver.log.write(f"  {i+1}) Coef = {top_sim_exc_states[i][0]:0.4f}; occ. = {top_sim_exc_states[i][1]} (prom {top_sim_exc_states[i][2]})")
#mol_solver.log.exit()

#mol_solver.find_LE_solution("SE", diag_alg = "SCF")


# First, let's check if LE sol SE (CISD SCF for UHF) works the way we expect by comparing its output (especially c1, c2) to an explicit calculation
# Then, let's plot the heatmap for < LE | b_i\hc b_j | LE >
# Then, let's try the zombie sampling!


# Sampling methods
#N_vals_width, energy_levels_width = mol_solver.find_ground_state("SEGS_width", N = 80, N_sub = 10, dataset_label = "SEGS width")


#for N_sub_val in [10, 20, 30, 50, 75, 100]:
#    mol_solver.find_ground_state("LE_Zombie_cov_RSOPM_moment_matching", N = 30, N_sub = N_sub_val, dataset_label = f"LEGS_RSOPM_moment_matching_N_sub={N_sub_val}")





#for nocov in [0, 5, 10]:
#    mol_solver.find_ground_state("LE_Zombie_cov_RSOPM_moment_matching", N = 100, N_sub = 50, N_no_cov = nocov, dataset_label = f"LEGS_RSOPM_moment_matching_S_filter_N_no_cov={nocov}")




#mol_solver.find_ground_state("LE_Zombie_cov_RSOPM_moment_matching", N = 100, N_sub = 50, N_no_cov = 0, rs = True, dataset_label = f"LEGS_RSOPM_moment_matching_S_filter_randsign")





#for cov_prop in np.arange(0.1, 0.9, 0.1):
#    mol_solver.find_ground_state("LE_Zombie_cov_SRRM_mirror", N = 10, N_sub = 10, cov_proportion = cov_prop, dataset_label = f"LEGS_SRRM_trim_cp={cov_prop:0.1f}")

# TODO move the moment matching to LE sol properties, so we dont always calculate it again and again! Therefore eta isnt a free param of the sampling!!

# Plotting and saving
#mol_solver.print_singlet_info()

mol_name = "Li2"
ref_energies = []
#if mol_name in trimmed_ground_state_full_ci:
#    ref_energies.append({"E" : trimmed_ground_state_full_ci[mol_name], "label" : "trimmed CI", "color" : functions.ref_energy_colors["trimmed CI"], "linestyle" : "dashed"})

#mol_solver.plot_datasets_against_param("N_sub", reference_energies = ref_energies)
mol_solver.plot_datasets(reference_energies = ref_energies)
#mol_solver.plot_datasets_extra(reference_energies = ref_energies)
mol_solver.save_data()
#mol_solver.log.close_journal()

