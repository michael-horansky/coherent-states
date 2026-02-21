import numpy as np
import matplotlib.pyplot as plt

from pyscf import gto, scf, fci
from molecules import *
from mol_solver_degenerate import ground_state_solver

import functions




benchmark_molecules = {
    "Li2" : li2_mol,
    "N2" : n2_mol
    }

# ---------------------------- Initialise solvers -----------------------------

mol_solvers = []

for mol_name, mol in benchmark_molecules.items():

    mol_solver = ground_state_solver(f"{mol_name}_full_CI")
    mol_solver.initialise_molecule(mol)

    mol_solvers.append(mol_solver)



# ----------------- Plot heatmap of single-shell excitations ------------------

print("\n-------------------- Full CI solution analysis --------------------")

# Get approximate subplot widths
subplot_widths = []
for i in range(len(mol_solvers)):
    subplot_widths.append(mol_solvers[i].S_alpha)

fig, axes = plt.subplots(1, 2, figsize=(12, 4), width_ratios=subplot_widths) #, figsize=(8, 6)
fig.suptitle("SECS states as components of the full CI solution")
#fig.subplots_adjust(bottom=0, top=1, left=4, right=5)
list_of_axes = list(axes)

bench_i = 0
for mol_name, mol in benchmark_molecules.items():

    #mol_solver = ground_state_solver(f"{mol_name}_full_CI")
    #mol_solver.initialise_molecule(mol)


    #mol_solver.print_ground_state()

    mol_solver = mol_solvers[bench_i]
    print(f"\n----------------- Molecule {mol_name} -----------------")
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
    print(f"\nFor closed shells U single-excitation CSF singlets, we have a {closed_shell_N + single_exc_N}-dim space with norm square projection {closed_shell_proj + single_exc_proj}")


    im, _ = mol_solver.plot_single_excitation_closed_shell_heatmap(ax = list_of_axes[bench_i])
    list_of_axes[bench_i].title.set_text(f"Molecule {mol_name}")

    bench_i += 1

fig.tight_layout()
plt.show()

# ------ Plot approxiomate heatmap based on the SECS-restricted solution ------

print("\n---------------- SECS-restricted solution analysis ----------------")

# We can reuse the subplot widths

fig, axes = plt.subplots(1, 2, figsize=(12, 4), width_ratios=subplot_widths) #, figsize=(8, 6)
fig.suptitle("SECS states as components of the SECS-restricted solution")
#fig.subplots_adjust(bottom=0, top=1, left=4, right=5)
list_of_axes = list(axes)

bench_i = 0
for mol_name, mol in benchmark_molecules.items():

    mol_solver = mol_solvers[bench_i]

    im, _ = mol_solver.plot_SECS_restricted_heatmap(ax = list_of_axes[bench_i])
    list_of_axes[bench_i].title.set_text(f"Molecule {mol_name}")

    bench_i += 1

fig.tight_layout()
plt.show()
