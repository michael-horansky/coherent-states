
# In this study, we run the usual calculation with a fixed zeta sample for various displacements of the Li atoms (between x0.5 and x2 of the binding length)
# The zeta sample is calculated for the binding length geometry (base displacement)
# Because this adds a new layer of time complexity, this calculation is to be done on Aire, not on a personal machine (except for debugging with small N, N_sub)


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from pyscf import gto, scf, fci
from mol_solver_degenerate import ground_state_solver
from utils.class_Semaphor import Semaphor

from coherent_states.CS_Thouless import CS_Thouless
from coherent_states.CS_sample import CS_sample

import functions

from molecules_abstract import *




#nontrivial_separations = [0.5, 0.6, 0.7, 0.85, 1.2, 1.4, 1.7, 2.0]
#nontrivial_separations = [0.8, 0.9, 0.95, 1.05, 1.1, 1.15, 1.3]
nontrivial_separations = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.05, 1.15, 1.2, 1.3, 1.4, 1.7, 2.0]


ds_id = "RNCS"
sys_id = "ccpVDZ"
N_rep = 20
cur_molecule = 'Li2'
N = 50
N_sub = 50
rs = "rp"
freeze_basis = True
number_of_NOs = 16
abf_basis = "cc-pvdz"

if number_of_NOs is not None:
    sys_id += f"_N_MO={number_of_NOs}"

# Assemble the molecules we work with
mol_objs = {} # [sep coef] = pyscf.gto.mol object
cur_am = mol_catalogue[cur_molecule]
cur_am.basis = abf_basis

for separation_coef in [1.0] + nontrivial_separations:
    mol_objs[separation_coef] = cur_am.get_gto_Mole(separation_coef)

# -------------- PLOTTING -------------

ylims = {
    "BeH2" : [0.02, 0.13, -0.031, 0.024],
    "Li2" : [0.005, 0.041, -0.0026, 0.0008]
    }



# One aggregate dataset per s.c.-

datasets_to_load = []
for i in range(N_rep):
    cur_label = f"{ds_id}_i{i}_{N}_{N_sub}"
    if rs != "ai":
        cur_label += f"_{rs}"
    if freeze_basis == False:
        cur_label += "_nf"
    datasets_to_load.append(cur_label)

E_data = [] # [data i] = {coef, CI, ref, ds : { measurement, extrapolation, extrapolation error} }
E_base_err = {} # [dataset label] = E base err

# Build the solvers for each separation. The first one is at normal distance and determines the basis
mol_solvers = {} # [sep coef] = ground_state_solver object
# Standard sep
mol_solvers[1.0] = ground_state_solver(f"{cur_molecule}_{sys_id}_dist=1.0")
mol_solvers[1.0].initialise_molecule(mol_objs[1.0], HF_method = "RHF", N_MO = number_of_NOs)

# for the base separation, we actually don't care about the _nf suffix.
base_sep_dataset_buf = {}
base_sep_datasets_to_load = []
for ds in datasets_to_load:
    base_sep_dataset_buf[ds] = ds.rstrip('_nf')
    base_sep_datasets_to_load.append(base_sep_dataset_buf[ds])

mol_solvers[1.0].load_data(["self_analysis", "measured_datasets"], base_sep_datasets_to_load)

mol_solvers[1.0].print_singlet_info()
#mol_solvers[1.0].full_CI_sol()
#mol_solvers[1.0].find_LE_solution("SE", diag_alg = "SCF")
#mol_solvers[1.0].find_ground_state("LE_Zombie_cov_RSOPM_moment_matching", N = N, N_sub = N_sub, N_no_cov = 0, rs = rs, dataset_label = global_ds_label)

#mol_solvers[1.0].plot_datasets(reference_energies = [])

# We extract the sample
#basis_sample_z_tensor = mol_solvers[1.0].disk_jockey.data_bulks[global_ds_label]["basis_samples"]
E_data_base_sep = mol_solvers[1.0].get_dataset_info(base_sep_datasets_to_load)
E_data.append({})
for base_key in ["FCI", "HF"]:
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
    mol_solvers[separation_coef] = ground_state_solver(f"{cur_molecule}_{sys_id}_dist={separation_coef}")
    mol_solvers[separation_coef].initialise_molecule(mol_objs[separation_coef], HF_method = "RHF", N_MO = number_of_NOs)
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
    "FCI" : np.zeros(N_c),
    "HF" : np.zeros(N_c),
    "E_g" : np.zeros(N_c),
    "E_g_err" : np.zeros(N_c),
    "E_extrapolated" : np.zeros(N_c),
    "E_extrapolated_err" : np.zeros(N_c),
    "duration" : np.zeros(N_c)
    }

total_calculation_time = 0.0

#duration_table = []

for row_i in range(len(E_data)):
    row = E_data[row_i]
    E_cols["c"][row_i] = row["c"]
    E_cols["FCI"][row_i] = row["FCI"]["E"]
    E_cols["HF"][row_i] = row["HF"]

    # Now we aggregate the data
    cur_E_g = []
    cur_E_extrapolated = []
    cur_duration = []


    for ds in datasets_to_load:
        cur_E_g.append(row[ds]["E_g"])
        cur_E_extrapolated.append(row[ds]["E_extrapolated"])
        cur_duration.append(row[ds]["duration"])

        print(f"Dataset {ds} was calculated in {functions.dtstr(row[ds]['duration'])}")
        total_calculation_time += row[ds]["duration"]

    E_cols["E_g"][row_i] = np.average(cur_E_g)
    E_cols["E_g_err"][row_i] = np.std(cur_E_g)
    E_cols["E_extrapolated"][row_i] = np.average(cur_E_extrapolated)
    E_cols["E_extrapolated_err"][row_i] = np.std(cur_E_extrapolated)
    E_cols["duration"][row_i] = np.average(cur_duration)


print(f"Total calculation time: {functions.dtstr(total_calculation_time)}")


H_in_K = 315775.326864009 # tha value of 1 Hartree in Kelvin
def H_to_K(x):
    return(x * H_in_K)
def K_to_H(x):
    return(x / H_in_K)


# -------------- Building the plot

# --- Colour cycle
cmap = plt.get_cmap("tab10")
# cmap(0) reserved for CI, cmap(1) reserved for ref state, cmap(i+2) corresponds to dataset[i]


fig, (ax_top, ax_bot) = plt.subplots(
    2, 1,
    sharex=True,
    figsize = (8, 6),
    gridspec_kw={"height_ratios": [1, 2], "hspace": 0.05}
)

plt.suptitle(fr"{cur_am.label} ground state energy against bond length coef.")

ax_bot.set_xlabel("Atom separation coef.")

for ax in (ax_top, ax_bot):

    #ax.set_ylabel(r'$E - E_{\rm FCI}$ [Hartree]')

    ax.grid(True)

    secax_y = ax.secondary_yaxis(
        'right', functions=(H_to_K, K_to_H))
    #secax_y.set_ylabel(r'$E - E_{\rm FCI}\ [K]$')



    ax.plot(E_cols["c"], E_cols["HF"] - E_cols["FCI"], marker="x", label = "ref E", color = cmap(1))

    ax.errorbar(E_cols["c"], E_cols["E_g"] - E_cols["FCI"], yerr = E_cols["E_g_err"], capsize = 2, label = "MC calc.", color = cmap(2))
    ax.errorbar(E_cols["c"], E_cols["E_extrapolated"] - E_cols["FCI"], yerr = E_cols["E_extrapolated_err"], capsize = 2, linestyle = "dashed", label = "extrapol.", color = cmap(3))

ax_top.set_ylim(ylims[cur_molecule][0], ylims[cur_molecule][1])
ax_bot.set_ylim(ylims[cur_molecule][2], ylims[cur_molecule][3])


ax_top.spines["bottom"].set_visible(False)
ax_bot.spines["top"].set_visible(False)
ax_top.tick_params(
    axis="x",
    which="both",
    bottom=False,
    labelbottom=False
)

# after setting limits, ticks, etc.

d = 0.5

kwargs = dict(
    marker=[(-1, -d), (1, d)],
    markersize=10,
    linestyle="none",
    color="k",
    mec="k",
    mew=1,
    clip_on=False
)

# Existing axis break marks
ax_top.plot([0, 1], [0, 0], transform=ax_top.transAxes, **kwargs)
ax_bot.plot([0, 1], [1, 1], transform=ax_bot.transAxes, **kwargs)

# Add break marks on vertical grid lines

grid_x = np.arange(0.6, 2.05, 0.2)
#grid_x = list(grid_x) + list(ax_bot.get_xticks())

for x in grid_x:
    ax_top.plot(
        [x], [0],
        transform=ax_top.get_xaxis_transform(),
        **kwargs
    )
    ax_bot.plot(
        [x], [1],
        transform=ax_bot.get_xaxis_transform(),
        **kwargs
    )


# labelling axes

ax_label = fig.add_subplot(111, frameon=False)

# Hide everything
ax_label.tick_params(
    left=False, right=False,
    bottom=False, top=False,
    labelleft=False, labelright=False,
    labelbottom=False, labeltop=False
)

# Main labels
#ax_label.set_xlabel("Atom separation coef.")
ax_label.set_ylabel(r"$E-E_{\rm FCI}$ [Hartree]", labelpad=60)
#ax_label.yaxis.set_label_coords(-0.08, 0.5)

# Secondary axis
secax_label = ax_label.secondary_yaxis(
    "right",
    functions=(H_to_K, K_to_H)
)
secax_label.set_ylabel(r"$E-E_{\rm FCI}$ [K]", labelpad=40)
# Hide ticks and tick labels
secax_label.set_yticks([])
secax_label.tick_params(
    right=False,
    labelright=False,
    length=0
)

# Optional: hide the spine as well
secax_label.spines["right"].set_visible(False)

"""handles_top, labels_top = ax_top.get_legend_handles_labels()
handles_bot, labels_bot = ax_bot.get_legend_handles_labels()

ax_bot.legend(
    handles_top + handles_bot,
    labels_top + labels_bot
)"""
ax_bot.legend(loc='lower left')

plt.show()



