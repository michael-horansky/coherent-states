# this little script checks if the outputs of RHF cisd calculations are interpreted correctly by comparing the coefs to an explicit calculation
# This is because the RHF calculation auitomatically transforms the occupancy basis to account for spin symmetry


# ----------------------- RESULTS ------------------------

import json
import numpy as np

from molecules import *
from mol_solver_degenerate import ground_state_solver

def str_to_prom_tuple(prom_str):
    # Restoring on data load

    proms_by_spin = prom_str.lstrip("(").rstrip(")").split(")), ((")
    index_lists_alpha = proms_by_spin[0].lstrip("(").rstrip(")").split("), (")
    index_lists_beta = proms_by_spin[1].lstrip("(").rstrip(")").split("), (")

    if index_lists_alpha[0] == '':
        index_lists_alpha_tuple = ((), ())
    else:
        index_lists_alpha_tuple = tuple([tuple([int(x) for x in occ.rstrip(",").split(", ")]) for occ in index_lists_alpha])
    if index_lists_beta[0] == '':
        index_lists_beta_tuple = ((), ())
    else:
        index_lists_beta_tuple = tuple([tuple([int(x) for x in occ.rstrip(",").split(", ")]) for occ in index_lists_beta])

    return( (
        index_lists_alpha_tuple,
        index_lists_beta_tuple
        ) )




# Li2 calc

scf_filename = "outputs/Zombie_states_testing_SCF_fixed/self_analysis/LE_sol.json"
exp_filename = "outputs/Zombie_states_testing_exp_comparison_with_doubles/self_analysis/LE_sol.json"


mol_solver = ground_state_solver(f"rhf testing DONT SAVE")
mol_solver.initialise_molecule(li2_mol, HF_method = "RHF")

S_alpha = 3
S_beta = 3
nao = 10

# NO calc (just to double-check the UHF)



scf_file = open(scf_filename, "r")
exp_file = open(exp_filename, "r")

scf_data = json.load(scf_file)
exp_data = json.load(exp_file)

print("Li2 LE results loaded")
print(f"  SCF energy: {scf_data["E"]}")
print(f"  exp energy: {exp_data["E"]}")

scf_sol = {str_to_prom_tuple(k): v for k, v in scf_data["sol"].items()}
exp_sol = {str_to_prom_tuple(k): v for k, v in exp_data["sol"].items()}

round_val = 3

print(f"Checking equivalence w.r.t. {round_val} decimal points...")

print("c0 analysis (HF state, prom signature (0, 0))")

scf_c0 = scf_sol[(((), ()), ((), ()))]
exp_c0 = exp_sol[(((), ()), ((), ()))]

if np.round(scf_c0, round_val) == - np.round(exp_c0, round_val):
    print("  c0 components agree!")
    print(f"    -SCF c0 comp: {scf_c0}")
    print(f"    -exp c0 comp: {exp_c0}")
else:
    print("  c0 components DISAGREE")
    print(f"    -SCF c0 comp: {scf_c0}")
    print(f"    -exp c0 comp: {exp_c0}")


print("c1 analysis (prom signatures (1, 0) and (0, 1))")

all_comps_agree = True

for i in range(S_alpha):
    for j in range(nao - S_alpha):
        # | i -> j > on alpha
            if np.round(scf_sol[(((i,), (j,)), ((), ()))] / scf_c0, round_val) != np.round(exp_sol[(((i,), (j,)), ((), ()))] / exp_c0, round_val):
                all_comps_agree = False
                print(f"Disagreement at |{i}->{j}> on alpha ({scf_sol[(((i,), (j,)), ((), ()))] / scf_c0} vs {exp_sol[(((i,), (j,)), ((), ()))] / exp_c0})")
                print(f"[Rounded: {np.round(scf_sol[(((i,), (j,)), ((), ()))]  / scf_c0, round_val)} vs {np.round(exp_sol[(((i,), (j,)), ((), ()))] / exp_c0, round_val)}]")
for i in range(S_beta):
    for j in range(nao - S_beta):
        # | i -> j > on beta
            if np.round(scf_sol[(((), ()), ((i,), (j,)))] / scf_c0, round_val) != np.round(exp_sol[(((), ()), ((i,), (j,)))] / exp_c0, round_val):
                all_comps_agree = False
                print(f"Disagreement at |{i}->{j}> on beta ({scf_sol[(((), ()), ((i,), (j,)))] / scf_c0} vs {exp_sol[(((), ()), ((i,), (j,)))] / exp_c0})")
                print(f"[Rounded: {np.round(scf_sol[(((), ()), ((i,), (j,)))] / scf_c0, round_val)} vs {np.round(exp_sol[(((), ()), ((i,), (j,)))] / exp_c0, round_val)}]")

if all_comps_agree:
    print("  c1 components agree!")
else:
    print("  c1 components DISAGREE")


print("c2 analysis (prom signature (1, 1))")

all_comps_agree = True

scf_tensor = np.zeros((S_alpha, nao - S_alpha, S_beta, nao - S_beta))
for i in range(S_alpha):
    for j in range(nao - S_alpha):
        for k in range(S_beta):
            for l in range(nao - S_beta):
                scf_tensor[i,j,k,l] += scf_sol[(((i,), (j,)), ((k,), (l,)))] / 2.0
                scf_tensor[k,l,i,j] += scf_sol[(((i,), (j,)), ((k,), (l,)))] / 2.0
                #scf_tensor[i,l,k,j] += scf_sol[(((i,), (j,)), ((k,), (l,)))] / 4.0
                #scf_tensor[k,j,i,l] += scf_sol[(((i,), (j,)), ((k,), (l,)))] / 4.0


for i in range(S_alpha):
    for j in range(nao - S_alpha):
        for k in range(S_beta):
            for l in range(nao - S_beta):
                # | i -> j > on alpha, | k -> l > on beta
                scf_val = scf_sol[(((i,), (j,)), ((k,), (l,)))]#scf_tensor[i,j,k,l]

                if np.round(scf_val / scf_c0, round_val) != np.round(exp_sol[(((i,), (j,)), ((k,), (l,)))] / exp_c0, round_val):
                    all_comps_agree = False
                    print(f"Disagreement at |{i}->{j}> on alpha, |{k}->{l}> on beta ({scf_val / scf_c0} vs {exp_sol[(((i,), (j,)), ((k,), (l,)))] / exp_c0})")
                    print(f"[Rounded: {np.round(scf_val / scf_c0, round_val)} vs {np.round(exp_sol[(((i,), (j,)), ((k,), (l,)))] / exp_c0, round_val)}]")

if all_comps_agree:
    print("  c2 components agree!")
else:
    print("  c2 components DISAGREE")


print("---------- Explicit energy calculation ----------")
print("Looks like it works well now, and any disagreement is due to SCF not finding the global minimum")


