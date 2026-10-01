
# In this study, we run the usual calculation with a fixed zeta sample for various displacements of the Li atoms (between x0.5 and x2 of the binding length)
# The zeta sample is calculated for the binding length geometry (base displacement)
# Because this adds a new layer of time complexity, this calculation is to be done on Aire, not on a personal machine (except for debugging with small N, N_sub)

import math
import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt

from pyscf import gto, scf, fci
from mol_solver_degenerate import ground_state_solver
from utils.class_Semaphor import Semaphor

from coherent_states.CS_Thouless import CS_Thouless
from coherent_states.CS_sample import CS_sample

import functions

from molecules_abstract import *

def get_log_ticks(minpower, maxpower):
    majorticks = []
    majorticklabels = []
    minorticks = []

    for power in range(minpower, maxpower + 1, 1):
        majorticks.append(power)
        majorticklabels.append(fr"$10^{{{power}}}$")
        for power_coef in range(1, 10, 1):
            if power_coef in [3, 7]:
                majorticks.append(power + np.log10(power_coef))
                majorticklabels.append(fr"${power_coef}\cdot 10^{{{power}}}$")
            else:
                minorticks.append(power + np.log10(power_coef))

    return(majorticks, majorticklabels, minorticks)




#nontrivial_separations = [0.5, 0.6, 0.7, 0.85, 1.2, 1.4, 1.7, 2.0]
#nontrivial_separations = [0.8, 0.9, 0.95, 1.05, 1.1, 1.15, 1.3]
nontrivial_separations = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.05, 1.1, 1.15, 1.2, 1.3, 1.4, 1.7, 2.0]


ds_id = "RNCS"
N_rep = 20
cur_molecule = 'Li2'
N = 50
N_sub = 50
rs = "rp"
freeze_basis = True

abf_basis = "cc-pvdz"


# One aggregate dataset per s.c.-

datasets_to_load = []
for i in range(N_rep):
    cur_label = f"{ds_id}_i{i}_{N}_{N_sub}"
    if rs != "ai":
        cur_label += f"_{rs}"
    if freeze_basis == False:
        cur_label += "_nf"
    datasets_to_load.append(cur_label)

N_MO_vals = np.array([10, 11, 12, 13, 14, 15, 16], dtype=int)



all_separations_list = sorted([1.0] + nontrivial_separations)
all_separations = np.array(all_separations_list)
base_i = all_separations_list.index(1.0)

X_dim = len(N_MO_vals)
Y_dim = len(all_separations)

X_mesh, Y_mesh = np.meshgrid(all_separations, N_MO_vals)

# -------------------- We prepare the output data objects ---------------------
# All output objects are indexed as [N_MO index][separation_coef index]

E_spaces = {
    "FCI" : np.zeros((X_dim, Y_dim)),
    "HF" : np.zeros((X_dim, Y_dim)),
    "E_g" : np.zeros((X_dim, Y_dim)),
    "E_g_err" : np.zeros((X_dim, Y_dim)),
    "E_extrapolated" : np.zeros((X_dim, Y_dim)),
    "E_extrapolated_err" : np.zeros((X_dim, Y_dim)),
    "duration_FCI" : np.zeros((X_dim, Y_dim)),
    "duration" : np.zeros((X_dim, Y_dim)),
    "duration_err" : np.zeros((X_dim, Y_dim))
}

E_bulk = np.zeros((X_dim, Y_dim, N + 1)) # [N_MO][c][N_trim]


S = None

for N_MO_i in range(len(N_MO_vals)):

    sys_id = f"ccpVDZ_N_MO={N_MO_vals[N_MO_i]}"

    # Assemble the molecules we work with
    mol_objs = {} # [sep coef] = pyscf.gto.mol object
    cur_am = mol_catalogue[cur_molecule]
    cur_am.basis = abf_basis

    for separation_coef in [1.0] + nontrivial_separations:
        mol_objs[separation_coef] = cur_am.get_gto_Mole(separation_coef)



    E_data = [] # [data i] = {coef, CI, ref, ds : { measurement, extrapolation, extrapolation error} }
    E_base_err = {} # [dataset label] = E base err

    # Build the solvers for each separation. The first one is at normal distance and determines the basis
    mol_solvers = {} # [sep coef] = ground_state_solver object
    # Standard sep
    mol_solvers[1.0] = ground_state_solver(f"{cur_molecule}_{sys_id}_dist=1.0")
    mol_solvers[1.0].initialise_molecule(mol_objs[1.0], HF_method = "RHF", N_MO = N_MO_vals[N_MO_i])

    if S is None:
        S = mol_solvers[1.0].S

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
    for base_key in ["FCI", "HF", "LE"]:
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
        mol_solvers[separation_coef].initialise_molecule(mol_objs[separation_coef], HF_method = "RHF", N_MO = N_MO_vals[N_MO_i])
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

    total_calculation_time = 0.0

    #duration_table = []

    for row_i in range(len(E_data)):
        row = E_data[row_i]
        print(N_MO_vals[N_MO_i])
        print(row["c"])
        print(row.keys())

        E_spaces["FCI"][N_MO_i][row_i] = row["FCI"]["E"]
        E_spaces["duration_FCI"][N_MO_i][row_i] = row["FCI"]["duration"]
        E_spaces["HF"][N_MO_i][row_i] = row["HF"]

        # Now we aggregate the data
        cur_E_g = []
        cur_E_extrapolated = []
        cur_duration = []

        for ds in datasets_to_load:
            cur_E_g.append(row[ds]["E_g"])
            cur_E_extrapolated.append(row[ds]["E_extrapolated"])
            cur_duration.append(row[ds]["duration"])
            for i in range(N + 1):
                E_bulk[N_MO_i][row_i][i] += row[ds]["E_bulk"][i] / len(datasets_to_load)

            print(f"Dataset {ds} was calculated in {functions.dtstr(row[ds]['duration'])}")
            total_calculation_time += row[ds]["duration"]

        E_spaces["E_g"][N_MO_i][row_i] = np.average(cur_E_g)
        E_spaces["E_g_err"][N_MO_i][row_i] = np.std(cur_E_g)
        E_spaces["E_extrapolated"][N_MO_i][row_i] = np.average(cur_E_extrapolated)
        E_spaces["E_extrapolated_err"][N_MO_i][row_i] = np.std(cur_E_extrapolated)
        E_spaces["duration"][N_MO_i][row_i] = np.average(cur_duration)
        E_spaces["duration_err"][N_MO_i][row_i] = np.std(cur_duration)


    print(f"Total calculation time: {functions.dtstr(total_calculation_time)}")


H_in_K = 315775.326864009 # tha value of 1 Hartree in Kelvin
def H_to_K(x):
    return(x * H_in_K)
def K_to_H(x):
    return(x / H_in_K)


# We also need to know the number of SACs
spin = 0
def get_dim(N_MO_val):
    return(int((spin + 1) / (N_MO_val + 1) * math.comb(N_MO_val + 1, int((S - spin) / 2)) * math.comb(N_MO_val + 1, int((S + spin) / 2) + 1)))


cmap = plt.get_cmap("tab10")


# ------------ 1st plot: result surface over N_MO, c

ax = plt.subplot(2, 2, 1, projection = "3d")
#ax.set_yscale("log")
ax.set_title("Accuracy against Hilbert space dimension with frozen basis")

ax.set_xlabel("Atom separation coef.")
ax.set_ylabel(r"$\dim{\mathcal{H}}$")
ax.set_zlabel(r'$(E - E_{\rm FCI})/(E_{\rm ref} - E_{\rm FCI})$ [%]')

#ax.set_zlim(-10, 10)

#yticks = np.log10([1e3, 1e4, 1e5, 1e6])
dim_H_yticks, dim_H_yticklabels, dim_H_yminorticks = get_log_ticks(3, 5)
ax.set_yticks(dim_H_yticks)
ax.set_yticklabels(dim_H_yticklabels)
ax.set_yticks(dim_H_yminorticks, minor = True)

Z = 100 * (E_spaces["E_g"] - E_spaces["FCI"]) / (E_spaces["HF"] - E_spaces["FCI"])



ax.plot_surface(
    X_mesh, np.log10(np.vectorize(get_dim)(Y_mesh)), Z,
    linewidth=0.3,
    antialiased=True,
    alpha=0.9,
    label = "MC calc."
)

ax.plot([1.0] * len(N_MO_vals), np.log10(np.vectorize(get_dim)(N_MO_vals)), Z[:,base_i], linewidth = 6, label = "base dist.", color = cmap(1))
ax.plot([1.0] * len(N_MO_vals), np.log10(np.vectorize(get_dim)(N_MO_vals)), [0.0] * len(N_MO_vals), linewidth = 3, linestyle = "dashed", color = cmap(1))


ax.legend()


# ------------ 2nd plot: fixed c = 1.0, duration of calculation against N_MO

ax = plt.subplot(2, 2, 2)


ax.set_title("Calculation at base distance (with sampling)")

color1 = cmap(1)
color2 = cmap(0)

ax.set_xlabel(r"No. of MOs")
ax.set_ylabel(r'$(E - E_{\rm FCI})/(E_{\rm ref} - E_{\rm FCI})$ [%]', color=color1)
ax.set_ylim(0, 15)

ax.errorbar(N_MO_vals, 100 * (E_spaces["E_g"][:,base_i] - E_spaces["FCI"][:,base_i]) / (E_spaces["HF"][:,base_i] - E_spaces["FCI"][:,base_i]),
            yerr = 100 * E_spaces["E_g_err"][:,base_i] / (E_spaces["HF"][:,base_i] - E_spaces["FCI"][:,base_i]), capsize = 2, label = "base dist.", color = color1
            )

ax.tick_params(axis='y', labelcolor=color1)
# Twin ax to show the duration
ax2 = ax.twinx()

ax2.set_ylabel("duration [h]", color=color2)  # we already handled the x-label with ax1
ax2.set_ylim(0, 20)
ax2.errorbar(N_MO_vals, E_spaces["duration"][:,base_i] / 3600, yerr = E_spaces["duration_err"][:,base_i] / 3600, capsize = 2, color=color2, label = "duration")

c, _ = sp.optimize.curve_fit(lambda x, a, b : a * np.power(x, 5) + b, N_MO_vals, E_spaces["duration"][:,base_i] / 3600, p0 = [2 / 1e5, 0])

ax2.plot(N_MO_vals, c[0] * np.power(N_MO_vals, 5) + c[1], color=color2, linestyle = "dashed", label = "$O(M^5)$ fit")

ax2.tick_params(axis='y', labelcolor=color2)

ax.legend(loc=2)
ax2.legend(loc=1)



# ---------- 3rd plot: fixed accuracy, varying N_trim --------------

E_bulk_accuracy = 100 * (1 - (E_bulk - E_spaces["FCI"][..., None]) / (E_spaces["HF"][..., None] - E_spaces["FCI"][..., None]))

accuracy_preserving_N_trim = np.zeros(X_dim)
accuracy_preserving_N_trim[X_dim - 1] = N

ref_accuracy = E_bulk_accuracy[X_dim - 1][base_i][N]

for N_MO_i in range(X_dim - 1):
    # We find the minimal N_trim
    for N_trim in range(N + 1):
        if E_bulk_accuracy[N_MO_i][base_i][N_trim] > ref_accuracy:
            accuracy_preserving_N_trim[N_MO_i] = N_trim
            break


ax = plt.subplot(2, 2, 3)
#ax.set_yscale("log")
ax.set_title(rf"Minimal $N_{{\rm trim}}$ preserving accuracy {ref_accuracy:0.1f}% at relaxed geometry (log)")

ax.set_xticks(dim_H_yticks)
ax.set_xticklabels(dim_H_yticklabels)
ax.set_xticks(dim_H_yminorticks, minor = True)


ax.set_xlabel(r"$\dim{\mathcal{H}}$")
ax.set_ylabel(r"$N_{\rm trim}$")

ax.plot(np.log10(np.vectorize(get_dim)(N_MO_vals)), accuracy_preserving_N_trim)


ax = plt.subplot(2, 2, 4)
#ax.set_yscale("log")
ax.set_title(rf"Minimal $N_{{\rm trim}}$ preserving accuracy {ref_accuracy:0.1f}% at relaxed geometry (log-log)")


N_trim_yticks, N_trim_yticklabels, N_trim_yminorticks = get_log_ticks(1, 2)

ax.set_xticks(dim_H_yticks)
ax.set_xticklabels(dim_H_yticklabels)
ax.set_xticks(dim_H_yminorticks, minor = True)
ax.set_yticks(N_trim_yticks)
ax.set_yticklabels(N_trim_yticklabels)
ax.set_yticks(N_trim_yminorticks, minor = True)


ax.set_xlabel(r"$\dim{\mathcal{H}}$")
ax.set_ylabel(r"$N_{\rm trim}$")

ax.plot(np.log10(np.vectorize(get_dim)(N_MO_vals)), np.log10(accuracy_preserving_N_trim))


plt.show()





