
###############################################################################
########################## CALCULATION ACCESS POINT ###########################
###############################################################################
# This is an access point for the simulation which calculates the mol-el g.s. E
# for varying separation coefficient for a diatomic molecule.

import numpy as np
import matplotlib.pyplot as plt

from pyscf import gto, scf, fci
from mol_solver_degenerate import ground_state_solver
from utils.class_Semaphor import Semaphor

from coherent_states.CS_Thouless import CS_Thouless
from coherent_states.CS_sample import CS_sample

import functions

from molecules_abstract import *

# AP header
from AP_catalogue import sep_coef_AP
sep_coef_AP.man()

params = sep_coef_AP.process_cmd_args()

cur_molecule = params[0]
if cur_molecule not in mol_catalogue.keys():
    raise Exception(f"Molecule \"{cur_molecule}\" not known.")

N = params[1]
N_sub = params[2]
rs = params[3]
freeze_basis = (params[4] == 1)
load_analysis = (params[5] == 1)
c_restrict = params[6]
ds_id = params[7]
sys_id = params[8]
abf_basis = params[9]



#nontrivial_separations = [0.5, 0.6, 0.7, 0.85, 1.2, 1.4, 1.7, 2.0]
#nontrivial_separations = [0.8, 0.9, 0.95, 1.05, 1.1, 1.15, 1.3]
nontrivial_separations = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.05, 1.1, 1.15, 1.2, 1.3, 1.4, 1.7, 2.0]

global_ds_label = f"{ds_id}_{N}_{N_sub}"
if rs != "ai":
    global_ds_label += f"_{rs}"
if freeze_basis == False:
    global_ds_label += "_nf"

# Assemble the molecules we work with
mol_objs = {} # [sep coef] = pyscf.gto.mol object
cur_am = mol_catalogue[cur_molecule]

cur_am.basis = abf_basis

for separation_coef in [1.0] + nontrivial_separations:
    mol_objs[separation_coef] = cur_am.get_gto_Mole(separation_coef)

# -------------- ONLY CALCULATION ----------


# Build the solvers for each separation. The first one is at normal distance and determines the basis
mol_solvers = {} # [sep coef] = ground_state_solver object

if freeze_basis or (c_restrict == 0 or c_restrict == 1):
    # Standard sep
    mol_solvers[1.0] = ground_state_solver(f"{cur_molecule}_{sys_id}_dist=1.0", yes = True, fancy_printing = False)
    mol_solvers[1.0].initialise_molecule(mol_objs[1.0])
    if load_analysis:
        mol_solvers[1.0].load_data(["self_analysis"])
    else:
        mol_solvers[1.0].full_CI_sol()
        mol_solvers[1.0].find_LE_solution("SE", diag_alg = "SCF")
    mol_solvers[1.0].find_ground_state("LE_Zombie_cov_RSOPM_moment_matching", N = N, N_sub = N_sub, N_no_cov = 0, rs = rs, dataset_label = global_ds_label)

    # We extract the sample
    basis_sample_z_tensor = mol_solvers[1.0].disk_jockey.data_bulks[global_ds_label]["basis_samples"]
    E_base_err = mol_solvers[1.0].disk_jockey.metadata[global_ds_label]["result_energy_states"]["E_base_err"]
    duration = mol_solvers[1.0].disk_jockey.metadata[global_ds_label]["result_energy_states"]["duration"]

    mol_solvers[1.0].save_data()


for i_separation_coef in range(len(nontrivial_separations)):
    if c_restrict == 0 or c_restrict == i_separation_coef + 2:
        separation_coef = nontrivial_separations[i_separation_coef]
        mol_solvers[separation_coef] = ground_state_solver(f"{cur_molecule}_{sys_id}_dist={separation_coef}", yes = True, fancy_printing = False)
        mol_solvers[separation_coef].initialise_molecule(mol_objs[separation_coef])
        if load_analysis:
            mol_solvers[separation_coef].load_data(["self_analysis"])
        else:
            mol_solvers[separation_coef].full_CI_sol()
            mol_solvers[separation_coef].find_LE_solution("SE", diag_alg = "SCF", skip_properties = freeze_basis)
        if freeze_basis:
            mol_solvers[separation_coef].find_ground_state("Qubit_from_z_tensor", z = basis_sample_z_tensor, dataset_label = global_ds_label)
        else:
            mol_solvers[separation_coef].find_ground_state("LE_Zombie_cov_RSOPM_moment_matching", N = N, N_sub = N_sub, N_no_cov = 0, rs = rs, dataset_label = global_ds_label)
        mol_solvers[separation_coef].save_data()






