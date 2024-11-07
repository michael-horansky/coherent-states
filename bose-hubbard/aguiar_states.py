import math
import numpy as np
import matplotlib.pyplot as plt

import time

import seaborn as sns
import pandas as pd

def bitfield(n, l):
    # l digits, binary rep of n
    #return [1 if digit=='1' else 0 for digit in bin(n)[2:]]
    return [1 if digit=='1' else 0 for digit in  f'{n:0{l}b}']
def bitfield_to_composition(bf):
    res = []
    cur_sum = 1
    for i in range(len(bf)):
        if bf[i] == 1:
            cur_sum += 1
        else:
            res.append(cur_sum)
            cur_sum = 1
    #if cur_sum > 1:
    res.append(cur_sum)
    return(res)

def compositions_of_k(k):
    if k == 0:
        return([])
    if k == 1:
        return([[1]])
    res = []
    for i in range(np.power(2, k-1)):
        cur_bf = bitfield_to_composition(bitfield(i, k-1))
        res.append(cur_bf)
    return(res)


def from_N_pick_M(N, M):
    # generates all lists of length M with non-repeating integers 0 <= k_i < N in ascending order
    if M == 0:
        return([[]])
    else:
        # we fix the final element and use tail recursion
        result = []
        for k_M in range(M-1, N):
            minors = from_N_pick_M(k_M, M-1)
            for i in range(len(minors)):
                result.append(minors[i] + [k_M])
        return(result)

def N_tuple_sum_K(N, K):
    partitions = from_N_pick_M(N+K-1, N-1)
    results = []
    for partition in partitions:
        results.append([partition[0]])
        for i in range(1, len(partition)):
            results[-1].append(partition[i]-partition[i-1]-1)
        results[-1].append(N+K-2 - partition[-1])
    return(results)




class CS():
    # object Coherent state, namely the SU(2) one

    def __init__(self, S, M, z = np.array([])):
        # M is the number of modes, equivalent to the number of complex parameters

        # here we force an SU(2) state
        self.S = S
        self.M = M #M

        if len(z) == 0:
            z = np.zeros(self.M-1, dtype=complex)
        self.z = z

        # Here we calculate N = 1/sqrt({z|z})
        self.N = np.power(1 / np.sqrt(1 + np.sum(self.z.real * self.z.real + self.z.imag * self.z.imag)), self.S)

    def __str__(self):
        return(f"SU({self.M}) Aguiar CS with S = {self.S}, z = {self.z}")
    def __repr__(self):
        return(self.__str__())

    def overlap(self, other, reduction = 0):
        # calculates the overlap integral < other (r) | self (r) >, where r(eduction) is the number of apostrophes
        coef_product = 1 + np.sum(np.conjugate(other.z) * self.z)
        return(np.power(coef_product, self.S - reduction) * self.N * other.N)

class BH():
    #driven Bose-Hubbard model

    def __init__(self, J_0, J_1, omega, U, K, j_zero, S, M):
        # Hamiltonian parameters
        self.J_0 = J_0
        self.J_1 = J_1
        self.omega = omega
        self.U = U
        self.K = K
        self.j_zero = j_zero


        # Fock parameters
        self.S = S
        self.M = M

        print("A driven Bose-Hubbard model with %i bosons over %i modes has been initialized." % (self.S, self.M))
        print(f"    | J(t) = {self.J_0} + {self.J_1} . cos({self.omega:.2f}t)   (hopping interaction)")
        print(f"    | U = {self.U}                       (on-site interaction)")
        print(f"    | K = {self.K}, j_0 = {self.j_zero}                (harmonic trapping potential)")

    def sample_gridlike(self, max_rect_dist, z_0 = np.array([]), beta = np.sqrt(np.pi)):

        # This is the approach of a 2(M-1)-dimensional complex grid with spacing beta centered around z_0,
        # which ensures that we can locally approximate identity integrals with riemann sums with measure beta^(2M-2)
        if len(z_0) == 0:
            z_0 = np.zeros(self.M-1, dtype=complex)

        self.basis = []
        self.beta = beta

        print("  Sampling the neighbourhood of z_0 in a gridlike fashion...")

        for rect_dist in range(max_rect_dist+1):
            deviations = N_tuple_sum_K(2 * (self.M-1), rect_dist)
            for deviation in deviations:
                # we find out the number of non-zero elements
                nonzero_indices = []
                for i in range(len(deviation)):
                    if deviation[i] > 0:
                        nonzero_indices.append(i)
                #sign_flip_configuration = bitfield()
                for i in range(2 ** len(nonzero_indices)):
                    sign_flip_configuration = bitfield(i, len(nonzero_indices))
                    deviation_copy = deviation.copy()
                    for j in range(len(nonzero_indices)):
                        if sign_flip_configuration[j] == 1:
                            deviation_copy[nonzero_indices[j]] *= -1
                    cur_z = z_0.copy()
                    for j in range(self.M - 1):
                        cur_z[j] += deviation_copy[j] * beta + 1j * deviation_copy[j + self.M - 1] * beta
                    self.basis.append(CS(self.S, self.M, cur_z))

        self.N = len(self.basis)
        self.z_0 = CS(self.S, self.M, z_0)

        print("  Preparing overlap matrices...")

        #TODO calculate <z_j | z_i> and their reductions here

        print(f"A sample of N = {self.N} basis vectors around z_0 = {z_0} with rectangular radius of {max_rect_dist} and spacing of {beta:.2f} has been initialized.")

    def iterate(self, max_t, dt):

        identity_prefactor = math.factorial(self.S + self.M - 1) / math.factorial(self.S)
        # First, we find A(t=0)
        A = np.zeros(self.N, dtype=complex)
        for i in range(self.N):
            A[i] = (identity_prefactor / np.power(1.0 + np.sum(np.conjugate(self.basis[i].z) * self.basis[i].z), self.M)) * np.power(self.beta * self.beta / np.pi, self.M - 1) * self.z_0.overlap(self.basis[i])
        return(A)

z_0 = 0.0+ 1j * 0.0

lol = BH(1, 0.5, 2 * np.pi, 0.1, 0, 0, 5, 2)
lol.sample_gridlike(5, np.array([z_0], dtype=complex), 0.2)

x_vals = []
y_vals = []

for basis_element in lol.basis:
    x_vals.append(basis_element.z[0].real - z_0.real)
    y_vals.append(basis_element.z[0].imag - z_0.imag)
A_vals = lol.iterate(5, 0.001)

Psi_mag = 0.0
for i in range(lol.N):
    for j in range(lol.N):
        Psi_mag += np.conjugate(A_vals[i]) * A_vals[j] * lol.basis[j].overlap(lol.basis[i])
print("< Psi | Psi > =", Psi_mag.real)

df = pd.DataFrame({"Y" : y_vals, "X" : x_vals, "A" : np.sqrt(A_vals.real * A_vals.real + A_vals.imag * A_vals.imag)})
table = df.pivot(index='Y', columns='X', values='A')
ax = sns.heatmap(table)
ax.invert_yaxis()
plt.show()

#TODO check at t=0 if <n_1> is a sensible number (between 0 and 1)

