
# In this study, we run the usual calculation with a fixed zeta sample for various displacements of the Li atoms (between x0.5 and x2 of the binding length)
# The zeta sample is calculated for the binding length geometry (base displacement)
# Because this adds a new layer of time complexity, this calculation is to be done on Aire, not on a personal machine (except for debugging with small N, N_sub)


import numpy as np
import matplotlib.pyplot as plt

from pyscf import gto, scf, fci
from mol_solver_degenerate import ground_state_solver
from utils.class_Semaphor import Semaphor

from coherent_states.CS_Thouless import CS_Thouless
from coherent_states.CS_sample import CS_sample

import utils.functions

from molecules_abstract import *


cur_molecule = 'Li2'



#nontrivial_separations = [0.5, 0.6, 0.7, 0.85, 1.2, 1.4, 1.7, 2.0]
#nontrivial_separations = [0.8, 0.9, 0.95, 1.05, 1.1, 1.15, 1.3]
nontrivial_separations = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.05, 1.15, 1.2, 1.3, 1.4, 1.7, 2.0]

N = 50
N_sub = 50
rs = "rp"
freeze_basis = False

#atasets_to_load = ["RNCS_50_50_rp", "RNCS_50_50_rp_nf", "RNCS_100_50_rp"] # None for default
datasets_to_load = ["RNCS_50_50_rp"] # None for default

global_ds_label = f"RNCS_{N}_{N_sub}"
if rs != "ai":
    global_ds_label += f"_{rs}"
if freeze_basis == False:
    global_ds_label += "_nf"

# Assemble the molecules we work with
mol_objs = {} # [sep coef] = pyscf.gto.mol object
cur_am = mol_catalogue[cur_molecule]

for separation_coef in [1.0] + nontrivial_separations:
    mol_objs[separation_coef] = cur_am.get_gto_Mole(separation_coef)

# -------------- PLOTTING --------------

if datasets_to_load is None:
    datasets_to_load = [global_ds_label]

E_data = [] # [data i] = {coef, CI, ref, ds : { measurement, extrapolation, extrapolation error} }
E_base_err = {} # [dataset label] = E base err

# Build the solvers for each separation. The first one is at normal distance and determines the basis
mol_solvers = {} # [sep coef] = ground_state_solver object
# Standard sep
mol_solvers[1.0] = ground_state_solver(f"{cur_molecule}_RNCS_dist=1.0")
mol_solvers[1.0].initialise_molecule(mol_objs[1.0], HF_method = "RHF")

# for the base separation, we actually don't care about the _nf suffix.
base_sep_dataset_buf = {}
base_sep_datasets_to_load = []
for ds in datasets_to_load:
    base_sep_dataset_buf[ds] = ds.rstrip('_nf')
    base_sep_datasets_to_load.append(base_sep_dataset_buf[ds])

mol_solvers[1.0].load_data(["self_analysis", "measured_datasets"], base_sep_datasets_to_load)
mol_solvers[1.0].log.close_journal()


for i in range(len(nontrivial_separations[:3])):
    separation_coef = nontrivial_separations[i]
    mol_solvers[separation_coef] = ground_state_solver(f"{cur_molecule}_RNCS_dist={separation_coef}")
    mol_solvers[separation_coef].initialise_molecule(mol_objs[separation_coef], HF_method = "RHF")
    mol_solvers[separation_coef].load_data(["self_analysis", "measured_datasets"], datasets_to_load)
    mol_solvers[separation_coef].log.close_journal()


"""err_func_roster = {
    "cons." : (lambda x : 1.0),
    "$1/\\sqrt{N}$" : None,
    "$1/N$" : (lambda x : 1.0 / x),
    "$\\exp(-N/10)$" : (lambda x : np.exp( - x / 10)),
    "$\\exp(-N/10)/\\sqrt{N}$" : (lambda x : np.exp( - x / 10) / np.sqrt(x))
    }"""
err_func_roster = {"$1/\\sqrt{N}$" : None}

cmap = plt.get_cmap("tab10")

for sep_c in [1.0] + nontrivial_separations[:3]:
    for i_ds in range(len(datasets_to_load)):
        ds = datasets_to_load[i_ds]

        csv_sol = mol_solvers[sep_c].disk_jockey.data_bulks[ds]["result_energy_states"]

        base_err = mol_solvers[1.0].disk_jockey.metadata[ds]["result_energy_states"]["E_base_err"]
        N_cutoff_space = np.arange(1, 48, 1, dtype = int)

        plt.title(f"${{\\rm Li}}_2$ extrapolation by inv. sqrt law")
        plt.xlabel("cutoff N")
        plt.ylabel("E [H]")
        plt.axhline(y = mol_solvers[sep_c].ci_energy, linestyle = "dashed", label = "$E_g$")

        i_err_func = 1
        for err_func_label, err_func in err_func_roster.items():
            E_ext_space = np.zeros(len(N_cutoff_space))
            E_ext_err_space = np.zeros(len(N_cutoff_space))

            for i in range(len(N_cutoff_space)):
                cur_E_ext, cur_E_ext_err = mol_solvers[sep_c].extrapolate_by_inverse_sqrt(csv_sol, N_cutoff_space[i], err_func)
                E_ext_space[i] = cur_E_ext
                E_ext_err_space[i] = cur_E_ext_err * base_err

            # As our FINAL GUESS, we select the first datapoint (lowest cutoff) for which all subsequent datapoints are within its interval of uncertainty
            final_guess = None
            final_guess_index = None
            for i in range(len(N_cutoff_space)):
                is_consistent_with_subsequent_datapoints = True
                for j in range(i + 1, len(N_cutoff_space)):
                    if E_ext_space[j] > E_ext_space[i] + E_ext_err_space[i] or E_ext_space[j] < E_ext_space[i] - E_ext_err_space[i]:
                        is_consistent_with_subsequent_datapoints = False
                        break
                if is_consistent_with_subsequent_datapoints:
                    final_guess = E_ext_space[i]
                    final_guess_index = i
                    break

            plt.errorbar(N_cutoff_space, E_ext_space, yerr = E_ext_err_space, capsize = 3, color = cmap(i_err_func), label = r"$E_{\rm ext.}$ with err. $\propto$ " + err_func_label)
            plt.axhline(y = final_guess, color = cmap(i_err_func), label = f"final guess ({err_func_label})")

            i_err_func += 1

        #plt.plot(N_cutoff_space, E_ext_space, label = r"$E_{\rm ext.}$")
        #plt.plot(N_cutoff_space, E_ext_err_space, label = r"$\sigma_{E_{\rm ext.}}$")

        plt.legend()
        plt.show()






