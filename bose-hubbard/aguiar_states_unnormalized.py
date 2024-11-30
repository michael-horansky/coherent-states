import math
import numpy as np
import matplotlib.pyplot as plt

import time

import seaborn as sns
import pandas as pd

import csv

# here we

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
        return(f"SU({self.M}) unnormalized Aguiar CS with S = {self.S}, z = {self.z}")
    def __repr__(self):
        return(self.__str__())

    def std_z(self, index):
        # This method returns the indexed value of z using the Fock standard
        # std_z(M-1) = 1, std_z(-1) = 0
        if index >= 0 and index < self.M - 1:
            return(self.z[index])
        if index == self.M - 1:
            return(1.0)
        return(0.0)

    def overlap(self, other, reduction = 0):
        # calculates the overlap integral < other (r) | self (r) >, where r(eduction) is the number of apostrophes
        coef_product = 1 + np.sum(np.conjugate(other.z) * self.z)
        return(np.power(coef_product, self.S - reduction)) # notice no N(z)

class BH():
    #driven Bose-Hubbard model

    # ---------------------------------------------------------
    # -------- Initializers, descriptors, destructors ---------
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # ------------------- Physical methods --------------------
    # ---------------------------------------------------------

    # ------------------- Sampling methods --------------------

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

        #print("  Preparing overlap matrices...")

        #TODO calculate <z_j | z_i> and their reductions here

        print(f"A sample of N = {self.N} basis vectors around z_0 = {z_0} with rectangular radius of {max_rect_dist} and spacing of {beta:.2f} has been initialized.")

    # ----------------- Dynamical descriptors -----------------

    # Overlap matrix initializers
    def initialize_overlap_matrices(self, cur_basis, max_reduction = 3):

        N = len(cur_basis)
        self.X = np.zeros((max_reduction + 1, N, N), dtype=complex) # X[r][i][j] = { z_i^(r) | z_j^(r) }
        for r in range(max_reduction + 1):
            for i in range(N):
                for j in range(N):
                    self.X[r][i][j] = cur_basis[j].overlap(cur_basis[i], reduction = r)

    def J(self, t):
        return(self.J_0 + self.J_1 * np.cos(self.omega * t))

    def H(self, t, cur_A, cur_basis):
        # this evaluates < Psi | H | Psi >
        N = len(cur_basis)
        M = cur_basis[0].M
        H = 0.0
        for k in range(N):
            for j in range(N):
                sum1 = 0.0
                for i in range(M-1):
                    sum1 += (np.conjugate(cur_basis[k].std_z(i)) * cur_basis[j].std_z(i+1) + np.conjugate(cur_basis[k].std_z(i+1)) * cur_basis[j].std_z(i)) * self.X[1][k][j]
                sum1 *= (- self.J(t) * self.S)

                sum2 = 0.0
                for i in range(M):
                    sum2 += (np.conjugate(cur_basis[k].std_z(i)) * np.conjugate(cur_basis[k].std_z(i)) * cur_basis[j].std_z(i) * cur_basis[j].std_z(i)) * self.X[2][k][j]
                sum2 *= (self.U * self.S * (self.S - 1) / 2)

                sum3 = 0.0
                for i in range(M):
                    sum3 += (1 + i - self.j_zero) * (1 + i - self.j_zero) * (np.conjugate(cur_basis[k].std_z(i)) * cur_basis[j].std_z(i)) * self.X[1][k][j]
                sum3 *= (self.S * self.K / 2)

                H += np.conjugate(cur_A[k]) * cur_A[j] * (sum1 + sum2 + sum3)

    def R(self, t, cur_A, cur_basis):
        N = len(cur_basis)
        M = cur_basis[0].M

        R = np.zeros(M*N, dtype=complex)

        # R = R_1[N] + R_2[N * (M-1)]
        # R_1[k] = dH / dA_k*
        # R_2[m * N + k] = dH / dz_k,m*

        # First, we fill in R_1
        for k in range(N):
            for j in range(N):
                sum1 = 0.0
                for i in range(M-1):
                    sum1 += (np.conjugate(cur_basis[k].std_z(i)) * cur_basis[j].std_z(i+1) + np.conjugate(cur_basis[k].std_z(i+1)) * cur_basis[j].std_z(i)) * self.X[1][k][j]
                sum1 *= (- self.J(t) * self.S)

                sum2 = 0.0
                for i in range(M):
                    sum2 += (np.conjugate(cur_basis[k].std_z(i)) * np.conjugate(cur_basis[k].std_z(i)) * cur_basis[j].std_z(i) * cur_basis[j].std_z(i)) * self.X[2][k][j]
                sum2 *= (self.U * self.S * (self.S - 1) / 2)

                sum3 = 0.0
                for i in range(M):
                    sum3 += (1 + i - self.j_zero) * (1 + i - self.j_zero) * (np.conjugate(cur_basis[k].std_z(i)) * cur_basis[j].std_z(i)) * self.X[1][k][j]
                sum3 *= (self.S * self.K / 2)

                R[k] += cur_A[j] * (sum1 + sum2 + sum3)

        # Then, we fill in R_2
        for m in range(M-1):
            for k in range(N):
                for j in range(N):
                    term1 = - self.J(t) * self.S * (cur_basis[j].std_z(m+1) + cur_basis[j].std_z(m-1)) * self.X[1][k][j]#cur_basis[j].overlap(cur_basis[k], self.S, reduction = 1)

                    term2 = 0.0
                    for i in range(M-1):
                        term2 += (np.conjugate(cur_basis[k].std_z(i)) * cur_basis[j].std_z(i+1) + np.conjugate(cur_basis[k].std_z(i+1)) * cur_basis[j].std_z(i)) * self.X[2][k][j]
                    term2 *= (- self.J(t) * self.S * (self.S - 1) * cur_basis[j].std_z(m))

                    term3 = self.U * self.S * (self.S - 1) * (np.conjugate(cur_basis[k].std_z(m)) * cur_basis[j].std_z(m) * cur_basis[j].std_z(m)) * self.X[2][k][j]

                    term4 = 0.0
                    for i in range(M):
                        term4 += (np.conjugate(cur_basis[k].std_z(i)) * np.conjugate(cur_basis[k].std_z(i)) * cur_basis[j].std_z(i) * cur_basis[j].std_z(i)) * self.X[3][k][j]
                    term4 *= (self.U * self.S * (self.S - 1) * (self.S - 2) * cur_basis[j].std_z(m) / 2)

                    term5 = (self.K / 2) * self.S * (1 + m - self.j_zero) * (1 + m - self.j_zero) * cur_basis[j].std_z(m) * self.X[1][k][j]

                    term6 = 0.0
                    for i in range(M):
                        term6 += (1 + i - self.j_zero) * (1 + i - self.j_zero) * (np.conjugate(cur_basis[k].std_z(i)) * cur_basis[j].std_z(i)) * self.X[2][k][j]
                    term6 *= self.K * self.S * (self.S - 1) * cur_basis[j].std_z(m) / 2

                    R[N + m * N + k] += np.conjugate(cur_A[k]) * cur_A[j] * (term1 + term2 + term3 + term4 + term5 + term6)

        return(R)

    def Theta(self, t, cur_A, cur_basis):
        # cur_A is an ndarray of complex decomposition coefficients A(t) of length N
        # cur_basis is a list of N instances of CS, each possessing an ndarray of complex parameters of length M
        N = len(cur_basis)
        M = cur_basis[0].M

        m_Theta = np.zeros((M*N, M*N), dtype=complex)

        # First, we fill in X
        for i in range(N):
            for j in range(N):
                m_Theta[i][j] = self.X[0][i][j]
        # Then, we fill in Y and Y^h.c.
        for a in range(N):
            for b in range(M-1):
                for d in range(N):
                    m_Theta[a][N + b * N + d] = self.S * np.conjugate(cur_basis[a].std_z(b)) * cur_A[d] * self.X[1][a][d]
                    m_Theta[N + b * N + d][a] = np.conjugate(m_Theta[a][N + b * N + d])
        # Then, we fill in Z
        for i in range(M-1):
            for j in range(M-1):
                for a in range(N):
                    for b in range(N):
                        # first, we evaluate (F_ij)_ab
                        m_Theta[N + i * N + a][N + j * N + b] = self.S * (self.S - 1) * self.X[2][a][b] * np.conjugate(cur_basis[a].std_z(j)) * cur_basis[b].std_z(i)
                        if i == j:
                            m_Theta[N + i * N + a][N + j * N + b] += self.S * self.X[1][a][b]
                        m_Theta[N + i * N + a][N + j * N + b] *= np.conjugate(cur_A[a]) * cur_A[b]
        return(m_Theta)


    # ---------------- Runtime routine methods ----------------

    def iterate(self, max_t, dt):

        N = len(self.basis)

        # everything is in natural units (hbar=1)
        # maximum time will actually be J_0 * max_t, which will also be the units we display it in

        t_space = np.arange(0.0, self.J_0 * max_t + dt, dt)

        step_N = len(t_space)

        A_evol = np.zeros((step_N, N), dtype=complex)
        basis_evol = np.zeros((step_N, N, self.M-1), dtype=complex)
        E_evol = np.zeros(step_N, dtype=complex)

        def record_state(t_i, cur_it_A, cur_it_basis):
            for i in range(N):
                A_evol[t_i][i] = cur_it_A[i]
                for j in range(self.M-1):
                    basis_evol[t_i][i][j] = cur_it_basis[i].std_z(j)
            E_evol[t_i] = self.H(t_i * dt, cur_it_A, cur_it_basis)


        # We initialize dynamical variables

        print("  Calculating initial decomposition coefficients...", end='', flush=True)
        identity_prefactor = math.factorial(self.S + self.M - 1) / math.factorial(self.S)
        # First, we find A(t=0)
        it_A = np.zeros(self.N, dtype=complex)
        for i in range(self.N):
            it_A[i] = (identity_prefactor / np.power(1.0 + np.sum(np.conjugate(self.basis[i].z) * self.basis[i].z), self.M + self.S)) * np.power(self.beta * self.beta / np.pi, self.M - 1) * self.z_0.overlap(self.basis[i])

        Psi_mag = 0.0
        for i in range(self.N):
            for j in range(self.N):
                Psi_mag += np.conjugate(it_A[i]) * it_A[j] * self.basis[j].overlap(self.basis[i])
        print(f" Done! The direct wavefunction normalization is {Psi_mag.real:.4f}")

        it_basis = []
        for i in range(N):
            it_basis.append(CS(self.S, self.M, self.basis[i].z.copy()))

        self.initialize_overlap_matrices(it_basis, 3)
        record_state(0, it_A, it_basis)

        start_time = time.time()
        progress = 0

        print(f"Iterative simulation of the Bose-Hubbard model on a timescale of t_max = {self.J_0 * max_t}, dt = {dt} ({step_N} steps) at {time.strftime("%H:%M:%S", time.localtime( start_time))}")

        for t_i in range(1, step_N):
            cur_t = (t_i-1) * dt
            # RK4

            #TODO triple check the basis_copy.z += update: shouldnt the slices through k1 be spaced out? as in it_basis_copy[n].z[m] += k1[N + m * N + n]
            #k1
            it_A_copy = it_A.copy()
            it_basis_copy = []
            for i in range(N):
                it_basis_copy.append(CS(self.S, self.M, it_basis[i].z.copy()))
            self.initialize_overlap_matrices(it_basis_copy, 3)
            cur_R = self.R(cur_t, it_A_copy, it_basis_copy)

            cur_Theta_inv = np.linalg.inv(self.Theta(cur_t, it_A_copy, it_basis_copy))
            k1 = - 1j * cur_Theta_inv.dot(cur_R)

            #k2
            it_A_copy += (dt / 2) * k1[:N]
            for n in range(N):
                for m in range(self.M-1):
                    it_basis_copy[n].z[m] += (dt / 2) * k1[N + N * m + n]
            self.initialize_overlap_matrices(it_basis_copy, 3)
            cur_R = self.R(cur_t + dt / 2, it_A_copy, it_basis_copy)
            cur_Theta_inv = np.linalg.inv(self.Theta(cur_t + dt / 2, it_A_copy, it_basis_copy))
            k2 = - 1j * cur_Theta_inv.dot(cur_R)

            #k3
            it_A_copy += (dt / 2) * (k2[:N]-k1[:N])
            for n in range(N):
                for m in range(self.M-1):
                    it_basis_copy[n].z += (dt / 2) * (k2[N + N * m + n]-k1[N + N * m + n])
            self.initialize_overlap_matrices(it_basis_copy, 3)
            cur_R = self.R(cur_t + dt / 2, it_A_copy, it_basis_copy)
            cur_Theta_inv = np.linalg.inv(self.Theta(cur_t + dt / 2, it_A_copy, it_basis_copy))
            k3 = - 1j * cur_Theta_inv.dot(cur_R)

            #k4
            it_A_copy += (dt / 2) * (2 * k3[:N]-k2[:N])
            for n in range(N):
                for m in range(self.M-1):
                    it_basis_copy[n].z += (dt / 2) * (2 * k3[N + N * m + n]-k2[N + N * m + n])
            self.initialize_overlap_matrices(it_basis_copy, 3)
            cur_R = self.R(cur_t + dt, it_A_copy, it_basis_copy)
            cur_Theta_inv = np.linalg.inv(self.Theta(cur_t + dt, it_A_copy, it_basis_copy))
            k4 = - 1j * cur_Theta_inv.dot(cur_R)

            it_A += (dt / 6) * (k1[:N] + 2 * k2[:N] + 2 * k3[:N] + k4[:N])
            for n in range(N):
                for m in range(self.M-1):
                    it_basis[n].z += (dt / 6) * (k1[N + N * m + n] + 2 * k2[N + N * m + n] + 2 * k3[N + N * m + n] + k4[N + N * m + n])

            record_state(t_i, it_A, it_basis)
            if np.floor(t_i / step_N * 100) > progress:
                progress = np.floor(t_i / step_N * 100)
                print("  " + str(progress) + "% done; est. time of finish: " + time.strftime("%H:%M:%S", time.localtime( (time.time()-start_time) * 100 / progress + start_time )), end='\r')


        # This function outputs the following arrays:
        #    1. [t][n] = A_n(t)
        #    2. [t] = <Psi(t) | H | Psi(t)> = <E>(t), for checking whether energy is conserved
        #    3. [t][n] = sum_m |xi_nm(t)|^2 (for checking if CSs stay SU(N)-normalized during their dynamical evolution)
        return(t_space, A_evol, basis_evol, E_evol)

        #print("  Preparing overlap matrices...", end='', flush=True)
        #print(" Done!")





def save_outputs(t_space, A_evol, basis_evol, E_evol):

    output_filename = "changename"

    print("  Writing outputs into outputs/" + output_filename + ".csv", end='', flush=True)

    output_file = open("outputs/" + output_filename + ".csv", "w")
    output_writer = csv.writer(output_file)

    N = len(A_evol[0])
    M = len(basis_evol[0][0])+1

    # number of string lengths (max int is 10^fill -1)
    N_fill = 4
    M_fill = 2

    # header
    header_row = ["t"]
    for n in range(N):
        header_row.append("A_" + str(n).zfill(N_fill))
    for n in range(N):
        for m in range(M - 1):
            header_row.append("z_" + str(n).zfill(N_fill) + "_" + str(m).zfill(M_fill))
    header_row.append("E")
    output_writer.writerow(header_row)



    output_file.close()
    print(" Done!")


z_0 = 0.0+ 1j * 0.0



lol = BH(1, 0.5, 2 * np.pi, 0.1, 0, 0, 5, 2)
lol.sample_gridlike(4, np.array([z_0], dtype=complex), 0.4)
N = len(lol.basis)

x_vals = []
y_vals = []

for basis_element in lol.basis:
    x_vals.append(round(basis_element.z[0].real - z_0.real, 3))
    y_vals.append(round(basis_element.z[0].imag - z_0.imag, 3))
t_space, A_evol, basis_evol, E_evol = lol.iterate(0.008, 0.00002)
#lol.iterate(5, 0.001)

"""Psi_mag = 0.0
for i in range(lol.N):
    for j in range(lol.N):
        Psi_mag += np.conjugate(A_vals[i]) * A_vals[j] * lol.basis[j].overlap(lol.basis[i])
print("< Psi | Psi > =", Psi_mag.real)"""

psi_mag = np.zeros(len(t_space))
avg_n_1 = np.zeros(len(t_space), dtype=complex)

for t_i in range(len(t_space)):
    for a in range(N):
        for b in range(N):
            psi_mag[t_i] += np.conjugate(A_evol[t_i][a]) * A_evol[t_i][b] * CS(lol.S, lol.M, basis_evol[t_i][b]).overlap(CS(lol.S, lol.M, basis_evol[t_i][a]))
            avg_n_1[t_i] += np.conjugate(A_evol[t_i][a]) * A_evol[t_i][b] * np.conjugate(basis_evol[t_i][a][0]) * basis_evol[t_i][b][0] * CS(lol.S, lol.M, basis_evol[t_i][b]).overlap(CS(lol.S, lol.M, basis_evol[t_i][a]), reduction = 1)
plt.plot(t_space, psi_mag, label="$\\langle \\Psi | \\Psi \\rangle}$")
plt.plot(t_space, avg_n_1, label="$\\frac{\\langle N_1 \\rangle}{S}$")
plt.legend()
plt.show()

save_outputs(t_space, A_evol, basis_evol, E_evol)

"""A_vals = A_evol[0]

df = pd.DataFrame({"Y" : y_vals, "X" : x_vals, "A" : np.sqrt(A_vals.real * A_vals.real + A_vals.imag * A_vals.imag)})
table = df.pivot(index='Y', columns='X', values='A')
ax = sns.heatmap(table)
ax.invert_yaxis()
plt.show()"""

#TODO check at t=0 if <n_1> is a sensible number (between 0 and 1)

