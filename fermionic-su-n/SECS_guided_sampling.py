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
    "Li2" : li2_mol,
    "N2" : n2_mol
    }

trimmed_ground_state_full_ci = {
    "Li2" : -14.65464717398042,
    "C2" : -74.57429774053233

    }


# First, we initialise solvers and obtain the SECS heatmaps
mol_solvers = {}
mol_SECS_heatmaps = {}
mol_SECS_restricted_energies = {}

for mol_name, mol in benchmark_molecules.items():

    mol_solver = ground_state_solver(f"mol_{mol_name}_Thouless_on_SECS")
    mol_solver.initialise_molecule(mol)
    mol_solver.pyscf_full_CI()

    mol_solvers[mol_name] = mol_solver

    """print(f"\n----------------- Molecule {mol_name} -----------------")
    print("----- Closed-shell Slater determinants -----")
    closed_shell_proj, closed_shell_N = mol_solver.closed_shell_projection()
    print(f"Norm squared of projection onto all {closed_shell_N} closed-shell states = {closed_shell_proj}")
    print("Top 5 closed shell occupancies are:")
    top_five_closed_shell_states = mol_solver.get_top_closed_shells(5)
    for i in range(len(top_five_closed_shell_states)):
        prom_label = mol_solver.get_prom_label(top_five_closed_shell_states[i][1], hr = True)
        print(f"  {i+1}) Coef = {top_five_closed_shell_states[i][0]:0.4f}; occ. = {top_five_closed_shell_states[i][1]} (prom ({prom_label[0]}) -> ({prom_label[1]}))")
    print("---------------- Other CSFs ----------------")
    single_exc_proj, single_exc_N = mol_solver.single_excitation_singlets_projection()
    print(f"Total square norm of the projection into all {single_exc_N} single-excitation CSF singlets is {single_exc_proj}")
    print(f"\nFor closed shells U single-excitation CSF singlets, we have a {closed_shell_N + single_exc_N}-dim space with norm square projection {closed_shell_proj + single_exc_proj}")"""

    cur_SECS_heatmap, cur_SECS_restricted_energy = mol_solver.solve_on_single_excitation_closed_shell()

    mol_SECS_heatmaps[mol_name] = cur_SECS_heatmap
    mol_SECS_restricted_energies[mol_name] = cur_SECS_restricted_energy











plot_x, plot_y = functions.subplot_dimensions(len(benchmark_molecules))

i = 0
for mol_name, mol in benchmark_molecules.items():
    plt.subplot(plot_y, plot_x, i+1)
    plt.title(f"Molecule {mol_name}")
    plt.xlabel("No. configs")
    plt.ylabel("E [Hartree]")

    mol_solver = mol_solvers[mol_name]

    # We now manually sample Thouless states guided by the SECS heatmap
    """cur_sample = CS_sample(mol_solver, CS_Thouless, add_ref_state = True)

    shape_alpha = (mol_solver.mol.nao - mol_solver.S_alpha, mol_solver.S_alpha)
    shape_beta = (mol_solver.mol.nao - mol_solver.S_beta, mol_solver.S_beta)
    centres_alpha = np.zeros(shape_alpha)
    centres_beta = np.zeros(shape_beta)

    N = 50
    N_subsample = 10

    N_vals_t = [1]
    energy_levels_t = [mol_solver.reference_state_energy]

    msg = f"  Conditioned sampling with ground state search on {N} states, each taken from {N_subsample} random states"
    new_sem_ID = mol_solver.semaphor.create_event(np.linspace(0, N_subsample * ((N + 2) * (N + 1) / 2 - 1) + 1, 1000 + 1), msg)

    for n in range(N):
        # We add the best out of 10 random states
        cur_subsample = []
        for n_sub in range(N_subsample):
            rand_z_alpha = CS_Thouless(mol_solver.mol.nao, mol_solver.S_alpha, np.random.normal(centres_alpha, np.sqrt(mol_SECS_heatmaps[mol_name]), shape_alpha))
            rand_z_beta = CS_Thouless(mol_solver.mol.nao, mol_solver.S_beta, np.random.normal(centres_beta, np.sqrt(mol_SECS_heatmaps[mol_name]), shape_beta))
            cur_subsample.append([rand_z_alpha, rand_z_beta])

        cur_sample.add_best_of_subsample(cur_subsample, semaphor_ID = new_sem_ID)
        N_vals_t.append(cur_sample.N)
        energy_levels_t.append(cur_sample.E_ground[-1])

    solution_benchmark = mol_solver.semaphor.finish_event(new_sem_ID, "    Evaluation")"""
    #N_vals_width, energy_levels_width = mol_solver.find_ground_state("SEGS_width", N = 20, N_sub = 5)
    N_vals_phase, energy_levels_phase = mol_solver.find_ground_state("SEGS_width", N = 3, N_sub = 1)

    #N_vals_t, energy_levels_t = mol_solver.find_ground_state("manual", sample = cur_sample)

    # For Li2, we can just quote the measured trimmed full CI ground state energy
    #trimmed_ground_state_full_ci = -14.65464717398042

    # For C2, we can just quote the measured trimmed full CI ground state energy
    #trimmed_ground_state_full_ci = -74.57429774053233

    # Save the data
    mol_solver.save_data()


    #plt.plot(N_vals_width, energy_levels_width, "x", label = "SEGS width")
    plt.plot(N_vals_phase, energy_levels_phase, "x", label = "SEGS phase")

    plt.axhline(y = mol_solver.reference_state_energy, label = "ref state", color = ref_energy_colors["ref state"])
    plt.axhline(y = mol_solver.ci_energy, label = "full CI", color = ref_energy_colors["full CI"])
    plt.axhline(y = mol_SECS_restricted_energies[mol_name], label = "SECS-restricted CI", color = ref_energy_colors["SECS"])

    #heatmap_state = [CS_Thouless(mol_solver.mol.nao, mol_solver.S_alpha, mol_SECS_heatmaps[mol_name]), CS_Thouless(mol_solver.mol.nao, mol_solver.S_alpha, mol_SECS_heatmaps[mol_name]) ]
    #heatmap_state_energy = mol_solver.H_overlap(heatmap_state, heatmap_state)
    #plt.axhline(y = heatmap_state_energy, label = "SECS CS-state energy", color = "red")

    if mol_name in trimmed_ground_state_full_ci:
        plt.axhline(y = trimmed_ground_state_full_ci[mol_name], label = "trimmed CI", color = ref_energy_colors["trimmed CI"], linestyle="dashed")

    plt.legend()

    i += 1

plt.tight_layout()
plt.show()

