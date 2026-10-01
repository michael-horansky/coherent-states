
###############################################################################
########################## CALCULATION ACCESS POINT ###########################
###############################################################################
# This is an access point for the simulation which calculates the mol-el g.s. E
# for varying number of MOs

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
from AP_catalogue import N_MO_AP
N_MO_AP.man()

params = N_MO_AP.process_cmd_args()

cur_molecule = params["mol"]
if cur_molecule not in mol_catalogue.keys():
    raise Exception(f"Molecule \"{cur_molecule}\" not known.")

N = params["N"]
N_sub = params["N_sub"]
method = params["method"]
rs = params["sr"]
sym = params["sym"]
load_analysis = (params["load_analysis"] == 1)
ds_id = params["ds_id"]
sys_id = params["sys_id"]
abf_basis = params["basis"]
number_of_NOs = params["N_MO"]
if number_of_NOs == 0:
    number_of_NOs = None



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

mol_solvers[1.0] = ground_state_solver(f"{cur_molecule}_{sys_id}_dist=1.0", yes = True, fancy_printing = True)
mol_solvers[1.0].initialise_molecule(mol_objs[1.0], N_MO = number_of_NOs)
if load_analysis:
    if load_base_dist:
        mol_solvers[1.0].load_data(["self_analysis", "measured_datasets"])
    else:
        mol_solvers[1.0].load_data(["self_analysis"])
else:
    mol_solvers[1.0].full_CI_sol()
    mol_solvers[1.0].find_LE_solution("SE", diag_alg = "SCF")

if not load_base_dist:
    mol_solvers[1.0].find_ground_state(method, N = N, N_sub = N_sub, N_no_cov = 0, rs = rs, dataset_label = global_ds_label)

# We extract the sample
basis_sample_z_tensor = mol_solvers[1.0].disk_jockey.data_bulks[global_ds_label]["basis_samples"]
mol_solvers[1.0].save_data()








