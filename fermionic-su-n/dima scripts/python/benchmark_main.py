import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


from integrals import *
from overlap import *
from H_pncfcs import *

from pyscf import gto, scf, fci
from mol_solver_degenerate import ground_state_solver

from coherent_states.CS_Qubit import CS_Qubit

# -------------------------- Li2

# Li2: bond length ≈ 2.673 Å = 5.0512375675 Bohr
# placed symmetrically about origin: half-distance ≈ 2.5256187838 Bohr
li2_mol = gto.Mole()
li2_mol.build(
    atom = '''
    Li   0.0   0.0  -2.5
    Li   0.0   0.0   2.5
    ''',
    basis = 'sto-3g',
    unit = "Bohr"
)
nalpha, nbeta = li2_mol.nelec
S_alpha = nalpha
S_beta = nbeta

print(f"M = {li2_mol.nao}, S = {S_alpha}")

# bigger basis: 6-31g**

# Set to +-2.5 Bohr exactly to match Dima's Hamiltonian. The true value is +-2.5256187838


# here we sample a standard basis sample which will be used for both my and dima's script

max_N = 10

z_null = np.concatenate((
        np.ones(3, dtype = complex),
        np.zeros(2, dtype=complex)
    ))
z_sample = [z_null]
for i in range(max_N - 1):
    rand_spin_alpha_state = np.concatenate((
            np.random.normal(100.0, 100.0, 2),
            np.random.normal(1.0, 1.0, 1),
            np.random.normal(0.0, 0.1, 2)
        ))
    z_sample.append(rand_spin_alpha_state)




# First, we make it Dima-compliant by doubling the alpha rep to the beta space
z_sample_dima = []
for z_vec in z_sample:
    z_sample_dima.append(np.concatenate((z_vec, z_vec)))

z_sample_michal = []
for z_vec in z_sample:
    cur_CS_Qubit = CS_Qubit(li2_mol.nao, S_alpha, np.concatenate((z_vec, np.zeros(li2_mol.nao - 5))))
    z_sample_michal.append([cur_CS_Qubit, cur_CS_Qubit])




# Michal

mol_solver = ground_state_solver(f"benchmark_with_dima_li2")
mol_solver.initialise_molecule(li2_mol)


# Since we're trimming the CI, we need to manually calculate the full CI energy
"""
trimmed_ground_state_full_ci = mol_solver.find_ground_state_on_full_ci(trim_M = 5)
print("Trimmed ground state energy (MOs = 5):", trimmed_ground_state_full_ci)

# for Li2, trim_M = 5, this gives -14.65464717398042 Hartree
"""
# We just quote the number to save time
trimmed_ground_state_full_ci = -14.65464717398042



N_vals_t, energy_levels_t, S_michal, H_michal = mol_solver.find_ground_state("manual", N = max_N, sample = z_sample_michal)

# Dima

I1, I2 = load_integrals()
ground_state_E_dima = []
for N in N_vals_t:

    # we calculate the overlap matrix and the Hamiltonian matrix on the sample
    H_dima, S_dima = H_pncfcs(np.column_stack(z_sample_dima[:N]), I1, I2)

    # We now solve the generalised eigenvalue problem
    energy_levels, energy_states = sp.linalg.eigh(H_dima, S_dima)
    ground_state_index = np.argmin(energy_levels)

    ground_state_E_dima.append(energy_levels[ground_state_index])


plt.title(f"Li2")
plt.xlabel("No. configs")
plt.ylabel("E [Hartree]")

plt.axhline(y = mol_solver.ci_energy, label = "Michal full CI")
plt.axhline(y = trimmed_ground_state_full_ci, label = "Michal trimmed full CI")

plt.plot(N_vals_t, energy_levels_t, "x", label = "Michal")

plt.plot(N_vals_t, ground_state_E_dima, "x", label = "Dima")


plt.legend()
plt.tight_layout()
plt.show()

