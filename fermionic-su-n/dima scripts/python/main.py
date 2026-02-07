


import numpy as np
import scipy as sp

import matplotlib.pyplot as plt

from integrals import *
from overlap import *
from H_pncfcs import *

# Li2 molecule integrals here

I1, I2 = load_integrals()

max_N = 10

M = 10

# note that the modes are not ordered by energy, but we have the first 5 modes ordered by energy and then the other 5 are the same MOs but opposite spin
# For each spin subspace, we have 3 occupied and 2 unoccupied MOs

# However, we assume that the spin-alpha and spin-beta components are equal for the ground state. So the CS restricts itself into the spin-alpha subspace

z_null = np.concatenate((
        np.ones(3, dtype = complex),
        np.zeros(2, dtype=complex),
        np.ones(3, dtype = complex),
        np.zeros(2, dtype=complex)
    ))

z_null_E, z_null_self_overlap = H_pncfcs(np.column_stack([z_null]), I1, I2, debug = True)

print("< null | H | null > =", z_null_E)
print("< null | null > =", z_null_self_overlap)


ground_state_energy_space = []
N_space = []

# in our sample, we will always have the Hartree-Fock zero-excitation state, and then states sampled randomly according to the paper by Dima
# This method samples each z as follows:
#     -lowest two MOs: normal with mean 100, std 100
#     -highest occupied MO: normal with mean 1, std 1
#     -two unoccupied MOs: normal with mean 0, std 0.1

z_sample = [z_null]
for i in range(max_N - 1):
    rand_spin_alpha_state = np.concatenate((
            np.random.normal(100.0, 100.0, 2),
            np.random.normal(1.0, 1.0, 1),
            np.random.normal(0.0, 0.1, 2)
        ))
    z_sample.append(np.concatenate((rand_spin_alpha_state, rand_spin_alpha_state)))

for N in range(1, max_N + 1):


    # we calculate the overlap matrix and the Hamiltonian matrix on the sample
    H, S = H_pncfcs(np.column_stack(z_sample[:N]), I1, I2)

    # We now solve the generalised eigenvalue problem
    energy_levels, energy_states = sp.linalg.eigh(H, S)
    ground_state_index = np.argmin(energy_levels)

    ground_state_energy_space.append(energy_levels[ground_state_index])
    N_space.append(N)

plt.plot(N_space, ground_state_energy_space, 'x')
plt.show()


