
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


#nontrivial_separations = [0.5, 0.6, 0.7, 0.85, 1.2, 1.4, 1.7, 2.0]
#nontrivial_separations = [0.8, 0.9, 0.95, 1.05, 1.1, 1.15, 1.3]
nontrivial_separations = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.05, 1.1, 1.15, 1.2, 1.3, 1.4, 1.7, 2.0]

N = 50
N_sub = 50
rs = True
if rs:
    global_ds_label = f"RNCS_{N}_{N_sub}_rs"
else:
    global_ds_label = f"RNCS_{N}_{N_sub}"

# Assemble the molecules we work with
mol_objs = {} # [sep coef] = pyscf.gto.mol object
for separation_coef in [1.0] + nontrivial_separations:
    mol_objs[separation_coef] = gto.Mole()
    mol_objs[separation_coef].build(
        atom = f'''
        Li   0.0000000000   0.0000000000  -{2.5 * separation_coef}
        Li   0.0000000000   0.0000000000   {2.5 * separation_coef}
        ''',
        basis = 'sto-3g',
        unit = "Bohr"
    )


# -------------- ONLY CALCULATION ----------
"""

# Build the solvers for each separation. The first one is at normal distance and determines the basis
mol_solvers = {} # [sep coef] = ground_state_solver object



# Standard sep
mol_solvers[1.0] = ground_state_solver(f"Li2_RNCS_dist=1.0")
mol_solvers[1.0].initialise_molecule(mol_objs[1.0], HF_method = "RHF")
mol_solvers[1.0].load_data(["self_analysis"])
#mol_solvers[1.0].full_CI_sol()
#mol_solvers[1.0].find_LE_solution("SE", diag_alg = "SCF")
mol_solvers[1.0].find_ground_state("LE_Zombie_cov_RSOPM_moment_matching", N = N, N_sub = N_sub, N_no_cov = 0, rs = rs, dataset_label = global_ds_label)

# We extract the sample
basis_sample_z_tensor = mol_solvers[1.0].disk_jockey.data_bulks[global_ds_label]["basis_samples"]
E_base_err = mol_solvers[1.0].disk_jockey.metadata[global_ds_label]["result_energy_states"]["E_base_err"]
duration = mol_solvers[1.0].disk_jockey.metadata[global_ds_label]["result_energy_states"]["duration"]

mol_solvers[1.0].save_data()


for separation_coef in nontrivial_separations:
    mol_solvers[separation_coef] = ground_state_solver(f"Li2_RNCS_dist={separation_coef}")
    mol_solvers[separation_coef].initialise_molecule(mol_objs[separation_coef], HF_method = "RHF")
    mol_solvers[separation_coef].load_data(["self_analysis"])
    #mol_solvers[separation_coef].full_CI_sol()
    #mol_solvers[separation_coef].find_LE_solution("SE", diag_alg = "SCF")
    mol_solvers[separation_coef].find_ground_state("Qubit_from_z_tensor", z = basis_sample_z_tensor, dataset_label = global_ds_label)
    mol_solvers[separation_coef].save_data()

"""
# ---------------- ONLY LOADING ----------------





#E_out = np.zeros(1 + len(nontrivial_separations)) # min E as measured for each separation
#E_ref = np.zeros(1 + len(nontrivial_separations)) # reference E
#E_fullCI = np.zeros(1 + len(nontrivial_separations)) # ground state E as calc'd by SCF
E_data = [] # [data i] = [coef, CI, ref, measurement, extrapolation, extrapolation error]

# Build the solvers for each separation. The first one is at normal distance and determines the basis
mol_solvers = {} # [sep coef] = ground_state_solver object
# Standard sep
mol_solvers[1.0] = ground_state_solver(f"Li2_RNCS_dist=1.0")
mol_solvers[1.0].initialise_molecule(mol_objs[1.0], HF_method = "RHF")
mol_solvers[1.0].load_data(["self_analysis", "measured_datasets"])
#mol_solvers[1.0].full_CI_sol()
#mol_solvers[1.0].find_LE_solution("SE", diag_alg = "SCF")
#mol_solvers[1.0].find_ground_state("LE_Zombie_cov_RSOPM_moment_matching", N = N, N_sub = N_sub, N_no_cov = 0, rs = rs, dataset_label = global_ds_label)

#mol_solvers[1.0].plot_datasets(reference_energies = [])

# We extract the sample
#basis_sample_z_tensor = mol_solvers[1.0].disk_jockey.data_bulks[global_ds_label]["basis_samples"]

E_data.append([1.0] + mol_solvers[1.0].get_dataset_info(global_ds_label))

E_base_err = mol_solvers[1.0].disk_jockey.metadata[global_ds_label]["result_energy_states"]["E_base_err"]
#duration = mol_solvers[1.0].disk_jockey.metadata[global_ds_label]["result_energy_states"]["duration"]


mol_solvers[1.0].save_data()


for i in range(len(nontrivial_separations)):
    separation_coef = nontrivial_separations[i]
    mol_solvers[separation_coef] = ground_state_solver(f"Li2_RNCS_dist={separation_coef}")
    mol_solvers[separation_coef].initialise_molecule(mol_objs[separation_coef], HF_method = "RHF")
    mol_solvers[separation_coef].load_data(["self_analysis", "measured_datasets"])
    #mol_solvers[separation_coef].full_CI_sol()
    #mol_solvers[separation_coef].find_LE_solution("SE", diag_alg = "SCF")
    #mol_solvers[separation_coef].find_ground_state("Qubit_from_z_tensor", z = basis_sample_z_tensor, dataset_label = global_ds_label)

    #E_out[i + 1] = mol_solvers[separation_coef].disk_jockey.metadata[global_ds_label]["result_energy_states"]["E_g"]
    #E_ref[i + 1] = mol_solvers[separation_coef].reference_state_energy
    #E_fullCI[i + 1] = mol_solvers[separation_coef].ci_energy
    #E_data.append([separation_coef, mol_solvers[separation_coef].ci_energy, mol_solvers[separation_coef].reference_state_energy, mol_solvers[separation_coef].disk_jockey.metadata[global_ds_label]["result_energy_states"]["E_g"]])
    E_data.append([separation_coef] + mol_solvers[separation_coef].get_dataset_info(global_ds_label))

    #mol_solvers[separation_coef].plot_datasets(reference_energies = [])
    mol_solvers[separation_coef].save_data()



E_data.sort(key = lambda x: x[0], reverse = True)

coefspace = np.zeros(1 + len(nontrivial_separations))

E_out = np.zeros(1 + len(nontrivial_separations)) # min E as measured for each separation
E_ref = np.zeros(1 + len(nontrivial_separations)) # reference E
E_fullCI = np.zeros(1 + len(nontrivial_separations)) # ground state E as calc'd by SCF

E_extrapolated = np.zeros(1 + len(nontrivial_separations)) # inv sqrt law
E_extrapolated_err = np.zeros(1 + len(nontrivial_separations)) # inv sqrt law err

for i in range(1 + len(nontrivial_separations)):
    coefspace[i] = E_data[i][0]
    E_fullCI[i] = E_data[i][1]
    E_ref[i] = E_data[i][2]
    E_out[i] = E_data[i][3]

    E_extrapolated[i] = E_data[i][4]
    E_extrapolated_err[i] = E_data[i][5]


H_in_K = 315775.326864009 # tha value of 1 Hartree in Kelvin
def H_to_K(x):
    return(x * H_in_K)
def K_to_H(x):
    return(x / H_in_K)

fig = plt.figure(figsize=(8, 6))
gs = fig.add_gridspec(2, 2)

plt.suptitle(r'${\rm Li}_2$ electronic ground state energy against atomic separation')

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, :])

ax1.set_title("Abs g.s. energy against separation dist.")

ax1.set_xlabel("Atom separation coef.")
ax1.set_ylabel("E [Hartree]")

ax1.plot(coefspace, E_fullCI, label = "full CI")
ax1.plot(coefspace, E_ref, label = "ref E")
ax1.plot(coefspace, E_out, label = f"RNCS (N = {N}, rep = {N_sub})")


ax1.legend()




ax2.set_title("Rel g.s. energy against separation dist.")

ax2.set_xlabel("Atom separation coef.")
ax2.set_ylabel(r'$E - E_{\rm CI}$ [Hartree]')

secax_y = ax2.secondary_yaxis(
    'right', functions=(H_to_K, K_to_H))
secax_y.set_ylabel(r'$E\ [K]$')



ax2.plot(coefspace, E_ref - E_fullCI, "x", label = "ref E")
ax2.plot(coefspace, E_out - E_fullCI, "x", label = f"RNCS (N = {N}, rep = {N_sub})")

ax2.errorbar(coefspace, E_extrapolated - E_fullCI, yerr = E_extrapolated_err * E_base_err, fmt = 'x', capsize = 3, label = f"$1/\\sqrt{{N}}$ extrapol.")


ax2.legend()



ax3.set_title("G.s. energy calc as a correction of H.F. ref state energy")

ax3.set_xlabel("Atom separation coef.")
ax3.set_ylabel(r'$\frac{E - E_{\rm CI}}{E_{\rm ref} - E_{\rm CI}}$ (%)')



ax3.plot(coefspace, 100 * (E_out - E_fullCI) / (E_ref - E_fullCI), "x", label = f"RNCS (N = {N}, rep = {N_sub})")

ax3.legend()

#plt.tight_layout()
plt.show()




