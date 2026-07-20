
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




#nontrivial_separations = [0.5, 0.6, 0.7, 0.85, 1.2, 1.4, 1.7, 2.0]
#nontrivial_separations = [0.8, 0.9, 0.95, 1.05, 1.1, 1.15, 1.3]
nontrivial_separations = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.05, 1.15, 1.2, 1.3, 1.4, 1.7, 2.0]


cur_molecule = 'H2O'
N = 100
N_sub = 50
rs = "rp"
freeze_basis = True

#atasets_to_load = ["RNCS_50_50_rp", "RNCS_50_50_rp_nf", "RNCS_100_50_rp"] # None for default
datasets_to_load = None #["RNCS_50_50_rp_nf"] # None for default

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
#mol_solvers[1.0].full_CI_sol()
#mol_solvers[1.0].find_LE_solution("SE", diag_alg = "SCF")
#mol_solvers[1.0].find_ground_state("LE_Zombie_cov_RSOPM_moment_matching", N = N, N_sub = N_sub, N_no_cov = 0, rs = rs, dataset_label = global_ds_label)

#mol_solvers[1.0].plot_datasets(reference_energies = [])

# We extract the sample
#basis_sample_z_tensor = mol_solvers[1.0].disk_jockey.data_bulks[global_ds_label]["basis_samples"]
E_data_base_sep = mol_solvers[1.0].get_dataset_info(base_sep_datasets_to_load)
E_data.append({})
for base_key in ["CI", "HF"]:
    E_data[-1][base_key] = E_data_base_sep[base_key]
for ds in datasets_to_load:
    E_data[-1][ds] = E_data_base_sep[base_sep_dataset_buf[ds]]
E_data[-1]["c"] = 1.0

#ref_E_base_err = mol_solvers[1.0].disk_jockey.metadata[base_sep_dataset_buf[ds]]["result_energy_states"]["E_base_err"]

for ds in datasets_to_load:
    E_base_err[ds] = mol_solvers[1.0].disk_jockey.metadata[base_sep_dataset_buf[ds]]["result_energy_states"]["E_base_err"]
    #duration = mol_solvers[1.0].disk_jockey.metadata[ds]["result_energy_states"]["duration"]


mol_solvers[1.0].save_data()


for i in range(len(nontrivial_separations)):
    separation_coef = nontrivial_separations[i]
    mol_solvers[separation_coef] = ground_state_solver(f"{cur_molecule}_RNCS_dist={separation_coef}")
    mol_solvers[separation_coef].initialise_molecule(mol_objs[separation_coef], HF_method = "RHF")
    mol_solvers[separation_coef].load_data(["self_analysis", "measured_datasets"], datasets_to_load)
    #mol_solvers[separation_coef].full_CI_sol()
    #mol_solvers[separation_coef].find_LE_solution("SE", diag_alg = "SCF")
    #mol_solvers[separation_coef].find_ground_state("Qubit_from_z_tensor", z = basis_sample_z_tensor, dataset_label = global_ds_label)

    #E_out[i + 1] = mol_solvers[separation_coef].disk_jockey.metadata[global_ds_label]["result_energy_states"]["E_g"]
    #E_ref[i + 1] = mol_solvers[separation_coef].reference_state_energy
    #E_fullCI[i + 1] = mol_solvers[separation_coef].ci_energy
    #E_data.append([separation_coef, mol_solvers[separation_coef].ci_energy, mol_solvers[separation_coef].reference_state_energy, mol_solvers[separation_coef].disk_jockey.metadata[global_ds_label]["result_energy_states"]["E_g"]])
    E_data.append(mol_solvers[separation_coef].get_dataset_info(datasets_to_load, E_base_err))
    E_data[-1]["c"] = separation_coef
    #mol_solvers[separation_coef].plot_datasets(reference_energies = [])
    mol_solvers[separation_coef].save_data()


coefspace = {}
E_out = {}

E_data.sort(key = lambda x: x["c"], reverse = True)



N_c = 1 + len(nontrivial_separations)
E_cols = {
    "c" : np.zeros(N_c),
    "CI" : np.zeros(N_c),
    "HF" : np.zeros(N_c)
    }
for ds in datasets_to_load:
    E_cols[ds] = {
        "E_g" : np.zeros(N_c),
        "E_extrapolated" : np.zeros(N_c),
        "E_extrapolated_err" : np.zeros(N_c),
        "duration" : np.zeros(N_c)
        }

total_calculation_time = 0.0
for row_i in range(len(E_data)):
    row = E_data[row_i]
    E_cols["c"][row_i] = row["c"]
    E_cols["CI"][row_i] = row["CI"]
    E_cols["HF"][row_i] = row["HF"]
    for ds in datasets_to_load:
        E_cols[ds]["E_g"][row_i] = row[ds]["E_g"]
        E_cols[ds]["E_extrapolated"][row_i] = row[ds]["E_extrapolated"]
        E_cols[ds]["E_extrapolated_err"][row_i] = row[ds]["E_extrapolated_err"]
        E_cols[ds]["duration"][row_i] = row[ds]["duration"]

        print(f"Dataset {ds} was calculated in {utils.functions.dtstr(row[ds]['duration'])}")
        total_calculation_time += row[ds]["duration"]

print(f"Total calculation time: {utils.functions.dtstr(total_calculation_time)}")


H_in_K = 315775.326864009 # tha value of 1 Hartree in Kelvin
def H_to_K(x):
    return(x * H_in_K)
def K_to_H(x):
    return(x / H_in_K)


# -------------- Building the plot

# --- Colour cycle
cmap = plt.get_cmap("tab10")
# cmap(0) reserved for CI, cmap(1) reserved for ref state, cmap(i+2) corresponds to dataset[i]


fig = plt.figure(figsize=(8, 6))
gs = fig.add_gridspec(2, 2)

plt.suptitle(f"{cur_am.label} electronic ground state energy against atomic separation")

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, :])

ax1.set_title("Abs g.s. energy against separation dist.")

ax1.set_xlabel("Atom separation coef.")
ax1.set_ylabel("E [Hartree]")

ax1.plot(E_cols["c"], E_cols["CI"], label = "full CI", color = cmap(0))
ax1.plot(E_cols["c"], E_cols["HF"], label = "ref E", color = cmap(1))
for i_ds in range(len(datasets_to_load)):
    ds = datasets_to_load[i_ds]
    ax1.plot(E_cols["c"], E_cols[ds]["E_g"], label = ds, color = cmap(i_ds + 2))


ax1.legend()




ax2.set_title("Rel g.s. energy against separation dist.")

ax2.set_xlabel("Atom separation coef.")
ax2.set_ylabel(r'$E - E_{\rm CI}$ [Hartree]')

ax2.grid(True)

secax_y = ax2.secondary_yaxis(
    'right', functions=(H_to_K, K_to_H))
secax_y.set_ylabel(r'$E\ [K]$')



ax2.plot(E_cols["c"], E_cols["HF"] - E_cols["CI"], "x", label = "ref E", color = cmap(1))
for i_ds in range(len(datasets_to_load)):
    ds = datasets_to_load[i_ds]
    ax2.plot(E_cols["c"], E_cols[ds]["E_g"] - E_cols["CI"], "x", label = ds, color = cmap(i_ds + 2))
    #ax2.plot(E_cols["c"], E_cols[ds]["E_extrapolated"] - E_cols["CI"], linestyle = "dashed", label = ds + " ext.", color = cmap(i_ds + 2))
    ax2.errorbar(E_cols["c"], E_cols[ds]["E_extrapolated"] - E_cols["CI"], yerr = E_cols[ds]["E_extrapolated_err"], capsize = 2, linestyle = "dashed", label = ds + " ext.", color = cmap(i_ds + 2))

    #ax2.errorbar(E_cols["c"], E_cols[ds]["E_extrapolated"] - E_cols["CI"], yerr = E_cols[ds]["E_extrapolated_err"] * E_base_err[ds], fmt = 'x', capsize = 3, label = f"{ds} (ext.)")


ax2.legend()



ax3.set_title("G.s. energy calc as a correction of H.F. ref state energy")

ax3.set_xlabel("Atom separation coef.")
ax3.set_ylabel(r'$\frac{E - E_{\rm CI}}{E_{\rm ref} - E_{\rm CI}}$ (%)')

ax3.grid(True)

for i_ds in range(len(datasets_to_load)):
    ds = datasets_to_load[i_ds]
    ax3.plot(E_cols["c"], 100 * (E_cols[ds]["E_g"] - E_cols["CI"]) / (E_cols["HF"] - E_cols["CI"]), "x", label = ds, color = cmap(i_ds + 2))
    #ax3.plot(E_cols["c"], 100 * (E_cols[ds]["E_extrapolated"] - E_cols["CI"]) / (E_cols["HF"] - E_cols["CI"]), linestyle = "dashed", label = ds + " ext.", color = cmap(i_ds + 2))
    ax3.errorbar(E_cols["c"], 100 * (E_cols[ds]["E_extrapolated"] - E_cols["CI"]) / (E_cols["HF"] - E_cols["CI"]), yerr = 100 * E_cols[ds]["E_extrapolated_err"] / (E_cols["HF"] - E_cols["CI"]), capsize = 2, linestyle = "dashed", label = ds + " ext.", color = cmap(i_ds + 2))

ax3.legend()

#plt.tight_layout()
plt.show()

# 1 Improve extrapolation
# 2 larger molecules



# 1 Repeat calc ( batch job)
# larger basis (100) for H2O, BeH2
# Only stretch one bond for triatomic mols (break sym)
# Intro: review of stochastic methods: Martin Patterson, Ali Allavi (slater det space random walk)
# Small note about the PNCCS
# JCP communication




