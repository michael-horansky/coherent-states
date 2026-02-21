import numpy as np
import matplotlib.pyplot as plt

from pyscf import gto, scf, fci
from molecules import *
from mol_solver_degenerate import ground_state_solver

import functions


mol_solver = ground_state_solver(f"N2_full_CI")
mol_solver.initialise_molecule(n2_mol)


mol_solver.print_ground_state()

print("----- Closed-shell Slater determinants -----")

"""
print("All closed shell components:")

closed_shell_projection = 0.0

all_single_spin_slater_dets = mol_solver.get_all_slater_determinants_with_fixed_S(10, 3)

for i in range(len(all_single_spin_slater_dets)):
    cur_c = mol_solver.ground_state_component(all_single_spin_slater_dets[i], all_single_spin_slater_dets[i])
    print(f"{i + 1}) {all_single_spin_slater_dets[i]} = {cur_c}")
    closed_shell_projection += cur_c * cur_c"""

closed_shell_proj, closed_shell_N = mol_solver.closed_shell_projection()
print(f"Norm squared of projection onto all {closed_shell_N} closed-shell states = {closed_shell_proj}")
print("---------------- Other CSFs ----------------")
"""
print("Consider |Psi> = 1/sqrt(2) . (|1110...>x|1011...> + |1011...>x|1110...>), i.e. a singlet CSF")
print("Its overlap with the ground state is", (
    mol_solver.ground_state_component(
        [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 1, 1, 0, 0, 0, 0, 0, 0]
        ) +
    mol_solver.ground_state_component(
        [1, 0, 1, 1, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
        )) / np.sqrt(2) )
print("And the antisym version of this (triplet) has zero overlap! So this really is a singlet CSF component :))")"""
single_exc_proj, single_exc_N = mol_solver.single_excitation_singlets_projection()
print(f"Total square norm of the projection into all {single_exc_N} single-excitation CSF singlets is {single_exc_proj}")
print(f"\nFor closed shells U single-excitation CSF singlets, we have a {closed_shell_N + single_exc_N}-dim space with norm square projection {closed_shell_proj + single_exc_proj}")
