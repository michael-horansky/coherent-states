
# In this study, we run the usual calculation with a fixed zeta sample for various displacements of the Li atoms (between x0.5 and x2 of the binding length)
# The zeta sample is calculated for the binding length geometry (base displacement)
# Because this adds a new layer of time complexity, this calculation is to be done on Aire, not on a personal machine (except for debugging with small N, N_sub)


import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

from pyscf import gto, scf, fci
from mol_solver_degenerate import ground_state_solver
from utils.class_Semaphor import Semaphor

from coherent_states.CS_Thouless import CS_Thouless
from coherent_states.CS_sample import CS_sample

import functions

from molecules_abstract import *


ds_id = "B1SCV"
sys_id = "RNCS"
N_rep = 20
cur_molecule = 'BeH2'
N = 100
N_sub = 50
rs = "rp"
freeze_basis = True

#nontrivial_separations = [0.5, 0.6, 0.7, 0.85, 1.2, 1.4, 1.7, 2.0]
#nontrivial_separations = [0.8, 0.9, 0.95, 1.05, 1.1, 1.15, 1.3]
nontrivial_separations = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.05, 1.15, 1.2, 1.3, 1.4, 1.7, 2.0]


# One aggregate dataset per s.c.-

datasets_to_load = []
for i in range(N_rep):
    cur_label = f"{ds_id}_i{i}_{N}_{N_sub}"
    if rs != "ai":
        cur_label += f"_{rs}"
    if freeze_basis == False:
        cur_label += "_nf"
    datasets_to_load.append(cur_label)

# Assemble the molecules we work with
mol_objs = {} # [sep coef] = pyscf.gto.mol object
cur_am = mol_catalogue[cur_molecule]

for separation_coef in [1.0] + nontrivial_separations:
    mol_objs[separation_coef] = cur_am.get_gto_Mole(separation_coef)

# -------------- PLOTTING --------------


E_data = [] # [data i] = {coef, CI, ref, ds : { measurement, extrapolation, extrapolation error} }
E_base_err = {} # [dataset label] = E base err

# Build the solvers for each separation. The first one is at normal distance and determines the basis
mol_solvers = {} # [sep coef] = ground_state_solver object
# Standard sep
mol_solvers[1.0] = ground_state_solver(f"{cur_molecule}_{sys_id}_dist=1.0")
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
    mol_solvers[separation_coef] = ground_state_solver(f"{cur_molecule}_{sys_id}_dist={separation_coef}")
    mol_solvers[separation_coef].initialise_molecule(mol_objs[separation_coef], HF_method = "RHF")
    mol_solvers[separation_coef].load_data(["self_analysis", "measured_datasets"], datasets_to_load)
    mol_solvers[separation_coef].log.close_journal()


err_func_roster = {"$1/\\sqrt{N}$" : None}

cmap = plt.get_cmap("tab10")


example_csv = mol_solvers[1.0].disk_jockey.data_bulks[datasets_to_load[0]]["result_energy_states"]

N_list = []
for row in example_csv:
    N_list.append(row["N"])

N_space = np.array(N_list)

trim_i = 0
while(N_space[trim_i] < 10):
    trim_i += 1
sqinv_N_space = np.array(1/np.sqrt(N_space))

N_cutoff_space = N_space[:-2]

min_number_of_datapoints = 10


# Plot properties
linfit_xspace = np.linspace(0.0, 1.0, 3)
color_fci = cmap(0)
color_ds = cmap(1)
color_linfit = cmap(2)
color_intercept = cmap(1)
color_interr = cmap(2)

color_optimal_N_min = cmap(3)

for sep_c in [1.0] + nontrivial_separations[:0]:


    # For the E(1/sqrt(N_trim)) plot
    E_space = np.zeros(len(N_list))

    # For the E_ext(N_min) plot
    E_ext_space = np.zeros(len(N_cutoff_space))
    E_ext_err_space = np.zeros(len(N_cutoff_space))

    for i_ds in range(len(datasets_to_load)):
        ds = datasets_to_load[i_ds]

        csv_sol = mol_solvers[sep_c].disk_jockey.data_bulks[ds]["result_energy_states"]

        # 1. We get the E(1/sqrt(N)) trend and the minimal error value

        for row_i in range(len(csv_sol)):
            E_space[row_i] += csv_sol[row_i]["E [H]"] / len(datasets_to_load)


        base_err = mol_solvers[1.0].disk_jockey.metadata[ds]["result_energy_states"]["E_base_err"]

        #plt.title(f"${{\\rm Li}}_2$ extrapolation by inv. sqrt law")
        #plt.xlabel()
        #plt.ylabel()
        #plt.axhline(y = mol_solvers[sep_c].ci_energy, linestyle = "dashed", label = "$E_g$")

        # Taking the points as error-less, we see if the fit error has a minimum
        cur_E_ext_space = np.zeros(len(N_cutoff_space))
        cur_E_ext_err_space = np.zeros(len(N_cutoff_space))

        for i in range(len(N_cutoff_space)):

            cur_E_ext, cur_E_ext_err = mol_solvers[sep_c].extrapolate_by_inverse_sqrt(csv_sol, N_cutoff_space[i])
            cur_E_ext_space[i] = cur_E_ext
            cur_E_ext_err_space[i] = cur_E_ext_err

        E_ext_space += cur_E_ext_space / len(datasets_to_load)
        E_ext_err_space += cur_E_ext_err_space / len(datasets_to_load)

    # Now we calculate the minimal-uncertainty linfit on the aggregate E_space

    """lol_E_ext_err_space = []
    res_list = []

    for i in range(len(N_space[:-min_number_of_datapoints])):
        res = sp.stats.linregress(sqinv_N_space[i:], E_space[i:])
        lol_E_ext_err_space.append(res.intercept_stderr)
        res_list.append(res)

    final_guess_index = np.argmin(lol_E_ext_err_space)
    final_res = res_list[final_guess_index]"""

    final_guess_index = np.argmin(E_ext_err_space[:-min_number_of_datapoints])
    final_res = sp.stats.linregress(sqinv_N_space[final_guess_index:], E_space[final_guess_index:])

    fig = plt.figure(figsize=(8,6))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    ax1.set_title("MC calc. of $E_g$ against sample basis size")
    ax1.set_xlabel(r"Trimmed sample basis size $N_{\rm trim}$")
    ax1.set_ylabel(r"$E$ [H]")

    ax1.plot(sqinv_N_space, E_space, "x", color = color_ds, label = "MC calc.")

    ax1.plot(linfit_xspace, final_res.slope * linfit_xspace + final_res.intercept, color = color_linfit, label = "Error-min. linfit")


    ax1.axhline(y = mol_solvers[sep_c].FCI_sol["E"], linestyle = "dashed", label = r"$E_{\rm FCI}$", color = color_fci)

    ax1.axvline(x = 1/np.sqrt(N_space[final_guess_index]), linestyle = "dashed", color = color_optimal_N_min, label = r"Optimal $N_{\rm min}$")

    ax1.get_yaxis().get_major_formatter().set_useOffset(False)

    # Transformed ticks
    N_ticks = np.array([1, 2, 3, 4, 5, 10, 20, 100])
    N_minorticks = np.concatenate((np.arange(1, 11), 10*np.arange(1, 11)))

    ax1.set_xticks(1 / np.sqrt(N_ticks))
    ax1.set_xticks(1 / np.sqrt(N_minorticks), minor = True)
    ax1.set_xticklabels(N_ticks)

    # Create inset
    """ax1ins = inset_axes(ax1,
                    width="40%",      # or e.g. 2.5 inches
                    height="40%",
                    loc="lower right")


    ax1ins.plot(sqinv_N_space, E_space, "x", color = color_ds, label = "MC calc.")
    ax1ins.plot(linfit_xspace, final_res.slope * linfit_xspace + final_res.intercept, color = color_linfit, label = "Error-min. linfit")
    ax1ins.axvline(x = 1/np.sqrt(N_space[final_guess_index]), linestyle = "dashed", color = color_optimal_N_min, label = r"Optimal $N_{\rm min}$")

    # Zoom limits
    ax1ins.set_xlim(1/np.sqrt(200), 1/np.sqrt(50))
    ax1ins.set_ylim(-15.5893, -15.5873)

    # Draw rectangle + connecting lines
    mark_inset(ax1, ax1ins, loc1=2, loc2=4, fc="none", ec="0.5")"""

    ax1ins = ax1.inset_axes([0.5, 0.02, 0.47, 0.43])

    ax1ins.plot(sqinv_N_space, E_space, "x", color = color_ds, label = "MC calc.")
    ax1ins.plot(linfit_xspace, final_res.slope * linfit_xspace + final_res.intercept, color = color_linfit, label = "Error-min. linfit")
    ax1ins.axvline(x = 1/np.sqrt(N_space[final_guess_index]), linestyle = "dashed", color = color_optimal_N_min, label = r"Optimal $N_{\rm min}$")

    # Zoom limits
    ax1ins.set_xlim(1/np.sqrt(150), 1/np.sqrt(50))
    ax1ins.set_ylim(-15.5891, -15.5873)
    ax1ins.tick_params(
        left=False, bottom=False,
        labelleft=False, labelbottom=False
    )

    ax1.indicate_inset_zoom(ax1ins)

    ax1.legend(loc=2)



    ax2.set_title(r"Extrapolated $E_g$ and uncertainty on linfit against $N_{\rm min}$")

    ax2.set_xlabel(r"Min. basis size considered for the linfit $N_{\rm min}$")
    ax2.set_ylabel(r"$E_{\rm ext}$ [H]", color=color_intercept)
    ax2.plot(N_cutoff_space, E_ext_space, color=color_intercept, label = r"Linfit intercept")
    ax2.tick_params(axis='y', labelcolor=color_intercept)
    ax2.axhline(y = mol_solvers[sep_c].FCI_sol["E"], linestyle = "dashed", label = r"$E_{\rm FCI}$", color = color_fci)
    ax2.axvline(x = N_space[final_guess_index], linestyle = "dashed", color = color_optimal_N_min, label = r"Optimal $N_{\rm min}$")
    ax2.axvline(x = N_space[-10], linestyle = "dashed", color = cmap(4), label = r"Max $N_{\rm min}$")

    ax2_twin = ax2.twinx()  # instantiate a second Axes that shares the same x-axis

    ax2_twin.set_ylabel(r"$\sigma_{\rm ext}$ [H]", color=color_interr)  # we already handled the x-label with ax2
    ax2_twin.plot(N_cutoff_space, E_ext_err_space, color=color_interr, label = "Linfit uncertainty")
    ax2_twin.tick_params(axis='y', labelcolor=color_interr)

    handles_a, labels_a = ax2.get_legend_handles_labels()
    handles_b, labels_b = ax2_twin.get_legend_handles_labels()

    ax2.get_yaxis().get_major_formatter().set_useOffset(False)
    ax2.legend(
        handles_a + handles_b,
        labels_a + labels_b,
        loc=2
    )

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()







