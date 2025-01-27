import math
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from pathlib import Path
import inspect # callable handling
from copy import deepcopy
import time

import csv

def subplot_dimensions(number_of_plots):
    # 1x1, 2x1, 3x1, 2x2, 3x2, 4x2, 3x3, 4x3, 5x3
    if number_of_plots == 1:
        return(1, 1)
    if number_of_plots == 2:
        return(2, 1)
    if number_of_plots == 3:
        return(3, 1)
    if number_of_plots == 4:
        return(2, 2)
    if number_of_plots <= 6:
        return(3, 2)
    if number_of_plots <= 8:
        return(4, 2)
    if number_of_plots <= 9:
        return(3, 3)
    if number_of_plots <= 12:
        return(4, 3)
    if number_of_plots <= 15:
        return(5, 3)
    return(int(np.ceil(np.sqrt(number_of_plots))), int(np.ceil(np.sqrt(number_of_plots))))

# -------------------- Coherent state functions --------------------
# We _deliberately_ do not implement instances of coherent states as
# class instances, as the dynamical manipulation is fast only when
# working with ndarrays. Hence these functions provide physical
# methods for overlap integrals etc.

def square_norm(z, S):
    # Returns a real number = { z | z }
    return(np.power(1 + np.sum(z.real * z.real + z.imag * z.imag), S))

def overlap(z1, z2, S, r = 0):
    # Returns a complex number = { z1^(r) | z2^(r) }, where r is the reduction.
    return(np.power(1 + np.sum(np.conjugate(z1) * z2), S - r))


class bosonic_su_n():

    ###########################################################################
    # --------------- STATIC METHODS, CONSTRUCTORS, DESCRIPTORS ---------------
    ###########################################################################

    basis_sampling_methods = {
            "sample_gaussian(z_0 = np.array([]), width = 1.0, conditioning_limit = -1, N_max = 50, max_saturation_steps = 50)" : "J. Chem. Phys. 144, 094106 (2016); see Appendix"
        }

    def __init__(self, ID):
        # Identification
        self.ID = ID

        print("---------------------------- " + str(ID) + " -----------------------------")

        # Data bins are initialized

        # initial conditions. Unchanged when simulation occurs.
        self.basis = []
        self.wavef = [] # Initial decomposition coefficients

        self.output_table = []#np.zeros((N_dtp, 2 + self.N * self.M), dtype=complex) # here we'll store all the results smashed together
        self.t_space = [] # np.zeros(N_dtp)
        self.wavef_evol = []#np.zeros((N_dtp, self.N), dtype=complex)
        self.basis_evol = []#np.zeros((N_dtp, self.N, self.M-1), dtype=complex)
        self.E_evol = []#np.zeros(N_dtp, dtype=complex)

        # Semaphors are initialised here to null values for safety
        self.semaphor_t_space = np.array([0.0])
        self.semaphor_simulation_start_time = 0.0
        self.semaphor_next_flag_t_i = 1

        # State booleans are initialized; useful when choosing savedata format
        self.is_phys_init = False # Are physical properties initialized?
        self.is_basis_init = False # Is basis initialized?
        self.is_basis_evol = False # Is basis evolved?
        self.is_wavef_init = False # Is wavefunction initialized?
        self.is_wavef_evol = False # Is wavefunction evolved?

        self.M = 0
        self.S = 0

        # Typographical constants
        # number of string lengths (max int is 10^fill -1)
        self.N_fill = 4
        self.M_fill = 2

        # Naming conventions
        self.output_subfolder_name = f"{self.ID}_data"
        self.output_config_filename = "config" # Stores configuration details
        self.output_H_A_filename = "H_A_func" # Stores H_A as callable
        self.output_H_B_filename = "H_B_func" # Stores H_B as callable
        self.output_basis_init_filename = "basis_init" # Initial basis
        self.output_wavef_init_filename = "wavef_init" # Initial wavef
        self.output_basis_evol_filename = "basis_evol" # Evolved basis
        self.output_wavef_evol_filename = "wavef_evol" # Evolved wavef


    ###########################################################################
    # ----------------------------- INTERNAL METHODS --------------------------
    ###########################################################################
    # Methods which the user isn't expected to invoke


    # ---------------------------------------------------------
    # ----------------- Physical descriptors ------------------
    # ---------------------------------------------------------
    # Methods used to describe the physics of the system

    # ------------------- Operator methods --------------------

    def decompose_aguiar(self, z):
        # Returns a decomposition of z into self.basis
        if not self.is_basis_init:
            print("ERROR: You have attempted to calculate a wavefunction decomposition before sampling a basis set. You can do this by calling one of the many basis-sampling methods listed below.")
            for m, desc in bosonic_su_n.basis_sampling_methods.items():
                print(f"{m} [{desc}]")
            return(-1)

        decomposition = np.zeros(self.N, dtype=complex)
        for i in range(self.N):
            for a in range(self.N):
                decomposition[i] += self.inverse_overlap_matrix[i][a] * overlap(self.basis[a], z, self.S)
        return(decomposition)

    def decompose_grossmann(self, xi):
        # Returns a decomposition of xi into self.basis
        if not self.is_basis_init:
            print("ERROR: You have attempted to calculate a wavefunction decomposition before sampling a basis set. You can do this by calling one of the many basis-sampling methods listed below.")
            for m, desc in bosonic_su_n.basis_sampling_methods.items():
                print(f"{m} [{desc}]")
            return(-1)

        decomposition = np.zeros(self.N, dtype=complex)
        for i in range(self.N):
            for a in range(self.N):
                cur_overlap = np.power(self.z[self.M - 1] + np.sum(np.conjugate(self.basis[a]) * self.z[:self.M - 1]), self.S )
                decomposition[i] += cur_overlap * self.inverse_overlap_matrix[i][a]
        return(decomposition)

    # ----------------- Interpolation methods -----------------

    def basis_state_at_time(self, t):
        # If basis and t_space are initialised, this will interpolate the basis state at any given time
        # and return both the state and the first time-derivative
        if not self.is_basis_evol:
            print("ERROR: Cannot interpolate the state of the basis at a non-zero time without first having evolved the basis.")
            return(-1)

        if t < self.t_space[0] or t > self.t_space[-1]:
            print(f"ERROR: Cannot interpolate the state of the basis at a time outside of the interval of propagation ({self.t_space[0]}, {self.t_space[-1]}).")
            return(-1)

        for t_i in range(len(self.t_space)):
            if self.t_space[t_i] == t:
                if t_i == 0:
                    time_derivative = (self.basis_evol[t_i+1]-self.basis_evol[t_i]) / (self.t_space[t_i+1] - self.t_space[t_i])
                else:
                    time_derivative = (self.basis_evol[t_i]-self.basis_evol[t_i-1]) / (self.t_space[t_i] - self.t_space[t_i-1])
                return(self.basis_evol[t_i], time_derivative)
            if self.t_space[t_i] > t:
                break
        # t_i is now the index of the first datapoint at a time larger than t
        time_derivative = (self.basis_evol[t_i]-self.basis_evol[t_i-1]) / (self.t_space[t_i] - self.t_space[t_i-1])
        q = (t-self.t_space[t_i-1]) / (self.t_space[t_i]-self.t_space[t_i-1])
        return(self.basis_evol[t_i - 1] + (self.basis_evol[t_i] - self.basis_evol[t_i - 1]) * q, time_derivative)
        # TODO more precisely, we can avoid using finite differences by directly feeding the interpolated value
        # into uncoupled_basis_y_dot. But is it still fast enough? And does it make a difference?

    def two_mode_solution(self, t_space, c_0 = False):
        # General solution to be plotted when M = 2 :))

        # First, we find H_ij(t)
        def func_H_matrix(t):
            H_matrix = np.zeros((self.S+1, self.S+1), dtype=complex)
            for i in range(self.S + 1):
                for j in range(self.S + 1):
                    # Calculating H_ij

                    # First order
                    for a in range(self.M):
                        for b in range(self.M):
                            a_i = i
                            b_i = self.S - i
                            a_j = j
                            b_j = self.S - j
                            coef = 1.0
                            if a == 0:
                                if a_i == 0:
                                    continue
                                coef *= np.sqrt(a_i)
                                a_i -= 1
                            elif a == 1:
                                if b_i == 0:
                                    continue
                                coef *= np.sqrt(b_i)
                                b_i -= 1
                            if b == 0:
                                if a_j == 0:
                                    continue
                                coef *= np.sqrt(a_j)
                                a_j -= 1
                            elif b == 1:
                                if b_j == 0:
                                    continue
                                coef *= np.sqrt(b_j)
                                b_j -= 1
                            if a_i == a_j and b_i == b_j:
                                H_matrix[i][j] += self.H_A(t, a, b) * coef

                    # Second order
                    for a in range(self.M):
                        for b in range(self.M):
                            for c in range(self.M):
                                for d in range(self.M):
                                    a_i = i
                                    b_i = self.S - i
                                    a_j = j
                                    b_j = self.S - j
                                    coef = 1.0
                                    if a == 0:
                                        if a_i == 0:
                                            continue
                                        coef *= np.sqrt(a_i)
                                        a_i -= 1
                                    elif a == 1:
                                        if b_i == 0:
                                            continue
                                        coef *= np.sqrt(b_i)
                                        b_i -= 1
                                    if b == 0:
                                        if a_i == 0:
                                            continue
                                        coef *= np.sqrt(a_i)
                                        a_i -= 1
                                    elif b == 1:
                                        if b_i == 0:
                                            continue
                                        coef *= np.sqrt(b_i)
                                        b_i -= 1
                                    if c == 0:
                                        if a_j == 0:
                                            continue
                                        coef *= np.sqrt(a_j)
                                        a_j -= 1
                                    elif c == 1:
                                        if b_j == 0:
                                            continue
                                        coef *= np.sqrt(b_j)
                                        b_j -= 1
                                    if d == 0:
                                        if a_j == 0:
                                            continue
                                        coef *= np.sqrt(a_j)
                                        a_j -= 1
                                    elif d == 1:
                                        if b_j == 0:
                                            continue
                                        coef *= np.sqrt(b_j)
                                        b_j -= 1
                                    if a_i == a_j and b_i == b_j:
                                        H_matrix[i][j] += self.H_B(t, a, b, c, d) * coef
            return(H_matrix)

        def c_dot(t, c):
            H_matrix = func_H_matrix(t)
            return( - 1j * H_matrix.dot(c))


        if type(c_0) == bool:
            c_0 = np.zeros(S + 1, dtype = complex)
            c_0[0] = 1.0

        sol = sp.integrate.solve_ivp(c_dot, [t_space[0], t_space[-1]], c_0, method = 'RK45', t_eval = t_space)
        N_space = np.zeros(len(t_space)) # N_space[t] = <N_1>/S at t

        for t_i in range(len(t_space)):
            t = t_space[t_i]
            cur_c = np.zeros(self.S+1, dtype=complex)
            for i in range(self.S + 1):
                cur_c[i] = sol.y[i][t_i]

            for i in range(self.S + 1):
                N_space[t_i] += i * (cur_c[i].real * cur_c[i].real + cur_c[i].imag * cur_c[i].imag) / self.S
        return(N_space)





    # ---------------- Uncoupled basis methods ----------------

    def uncoupled_basis_y_dot(self, t, y, reg_timescale = -1):
        # Here y = z, y.shape = (M - 1)

        extended_y = np.zeros(self.M, dtype=complex)
        for i in range(self.M - 1):
            extended_y[i] = y[i]
        extended_y[self.M - 1] = 1.0
        norm_reductions = [0, 0, 0] # [r] = { z^(r+1) | z^(r+1) }
        base_product = 1 + np.sum(y.real * y.real + y.imag * y.imag)
        norm_reductions[2] = np.power(base_product, self.S - 3)
        norm_reductions[1] = norm_reductions[2] * base_product
        norm_reductions[0] = norm_reductions[1] * base_product

        # We now initialize M_ij
        M = np.zeros((self.M - 1, self.M - 1), dtype=complex)
        for i in range(self.M - 1):
            for j in range(self.M - 1):
                M[i][j] += 1j * self.S * (self.S - 1) * norm_reductions[1] * np.conjugate(y[j]) * y[i]
                if i == j:
                    M[i][j] += 1j * self.S * norm_reductions[0]

        # Now we initialize R_i
        R = np.zeros(self.M - 1, dtype=complex)
        for i in range(self.M - 1):
            sum1 = 0.0
            for a in range(self.M):
                for b in range(self.M):
                    sum1 += self.H_A(t,a,b) * np.conjugate(extended_y[a]) * extended_y[b]
            R[i] += self.S * (self.S - 1) * norm_reductions[1] * y[i] * sum1
            sum2 = 0.0
            for b in range(self.M):
                sum2 += self.H_A(t,i,b) * extended_y[b]
            R[i] += self.S * norm_reductions[0] * sum2
            sum3 = 0.0
            for a in range(self.M):
                for b in range(self.M):
                    for c in range(self.M):
                        for d in range(self.M):
                            sum3 += self.H_B(t,a,b,c,d) * np.conjugate(extended_y[a]) * np.conjugate(extended_y[b]) * extended_y[c] * extended_y[d]
            R[i] += 0.5 * self.S * (self.S - 1) * (self.S - 2) * norm_reductions[2] * y[i] * sum3
            sum4 = 0.0
            for a in range(self.M):
                for c in range(self.M):
                    for d in range(self.M):
                        sum4 += (self.H_B(t,i,a,c,d) * self.H_B(t,a,i,c,d)) * np.conjugate(extended_y[a]) * extended_y[c] * extended_y[d]
            R[i] += 0.5 * self.S * (self.S - 1) * norm_reductions[1] * sum4

        # y_dot = M^(-1) . R
        # Regularisation
        if reg_timescale != -1:
            for i in range(self.M - 1):
                M[i][i] += reg_timescale
        M_inv = np.linalg.inv(M)

        # Semaphor : the uncoupled basis y_dot actually doesn't update semaphor, as the semaphor is used to aggregate the progress for all basis vectors
        #self.update_semaphor(t)

        return(M_inv.dot(R))

    def uncoupled_wavef_y_dot(self, t, y, reg_timescale = -1):
        # Here y = A_i, y.shape = (N)

        cur_basis, cur_basis_dot = self.basis_state_at_time(t)
        overlap_matrix = np.zeros((self.N, self.N, 3), dtype=complex) #[i][j][r] = { z^(r) | z^(r) }
        for i in range(self.N):
            for j in range(self.N):
                base_product = 1 + np.sum(np.conjugate(cur_basis[i]) * cur_basis[j])
                overlap_matrix[i][j][2] = np.power(base_product, self.S - 2)
                overlap_matrix[i][j][1] = overlap_matrix[i][j][2] * base_product
                overlap_matrix[i][j][0] = overlap_matrix[i][j][1] * base_product

        def ebe(i, k): # extended basis element
            if k != self.M - 1:
                return(cur_basis[i][k])
            else:
                return(1.0)

        # We now initialize M_ij
        M = np.zeros((self.N, self.N), dtype=complex)
        for i in range(self.N):
            for j in range(self.N):
                M[i][j] = 1j * overlap_matrix[i][j][0]

        # Now we initialize R_i
        R = np.zeros(self.N, dtype=complex)
        for i in range(self.N):
            sum1 = 0.0
            for j in range(self.N):
                sum1 += y[j] * overlap_matrix[i][j][1] * np.sum(np.conjugate(cur_basis[i]) * cur_basis_dot[j])
            sum1 *= -1j * self.S

            sum2 = 0.0
            for j in range(self.N):
                # We calculate the matrix element
                first_order_element_sum = 0.0
                for a in range(self.M):
                    for b in range(self.M):
                        first_order_element_sum += self.H_A(t, a, b) * np.conjugate(ebe(i, a)) * ebe(j, b)
                first_order_element_sum *= self.S * overlap_matrix[i][j][1]

                second_order_element_sum = 0.0
                for a in range(self.M):
                    for b in range(self.M):
                        for c in range(self.M):
                            for d in range(self.M):
                                second_order_element_sum += self.H_B(t,a,b,c,d) * np.conjugate(ebe(i, a)) * np.conjugate(ebe(i, b)) * ebe(j, c) * ebe(j, d)
                second_order_element_sum *= 0.5 * self.S * (self.S - 1) * overlap_matrix[i][j][2]

                sum2 += y[j] * (first_order_element_sum + second_order_element_sum)
            R[i] = sum1 + sum2
        # y_dot = M^(-1) . R
        # Regularisation
        if reg_timescale != -1:
            for i in range(self.M - 1):
                M[i][i] += reg_timescale
        M_inv = np.linalg.inv(M)

        # Semaphor
        self.update_semaphor(t)

        return(M_inv.dot(R))


    # --------------- Fully variational methods ---------------

    def standardise_dynamic_variables(self, cur_basis, cur_wavef):
        # y is the same format as R, so [A_1, A_2 ... A_N, z_1,1, z_2,1 ... z_N,1 ... ]
        # cur_basis is a list of lists [n][m]
        y = np.zeros(self.N * self.M, dtype = complex)
        for n in range(N):
            y[n] = cur_wavef[n]
            for m in range(self.M-1):
                y[self.N + self.N * m + n] = cur_basis[n][m]
        return(y)

    def destandardise_dynamic_variables(self, y):
        cur_basis = np.zeros((self.N, self.M - 1), dtype=complex)
        cur_wavef = np.zeros(self.N, dtype=complex)
        for n in range(N):
            cur_wavef[n] = y[n]
            for m in range(self.M-1):
                cur_basis[n][m] = y[self.N + self.N * m + n]
        return(cur_basis, cur_wavef)


    def calculate_overlap_matrices(self, y, max_reduction = 3):
        # we note that z_n,m = y[N + N * m + n]
        # therefore z_n as a list is = y[N + n : N + n + N * M : N], where n goes from 0 to N-1 inclusive
        X = np.zeros((self.N, self.N, max_reduction + 1), dtype=complex) # X[i][j][r] = { z_i^(r) | z_j^(r) }

        # We optimize the number of operations by noting that { z_i^(r) | z_j^(r) } = { z_i^(r+1) | z_j^(r+1) } * (1+z_i*.z_j)
        for i in range(self.N):
            for j in range(self.N):
                base_inner_product = 1 + np.sum(np.conjugate(y[self.N + i : self.N + i + self.N * self.M : self.N]) * y[self.N + j : self.N + j + self.N * self.M : self.N])
                X[i][j][max_reduction] = np.power(base_inner_product, self.S - max_reduction)
                for delta_r in range(max_reduction):
                    X[i][j][max_reduction - (delta_r + 1)] = X[i][j][max_reduction - delta_r] * base_inner_product
        return(X)

    def calculate_hamiltonian_tensors(self, t):
        cur_H_A = np.zeros((self.M, self.M), dtype=complex)
        cur_H_B = np.zeros((self.M, self.M, self.M, self.M), dtype=complex)
        for a in range(self.M):
            for b in range(self.M):
                cur_H_A[a][b] = self.H_A(t, a, b)
        for a in range(self.M):
            for b in range(self.M):
                for c in range(self.M):
                    for d in range(self.M):
                        cur_H_B[a][b][c][d] = self.H_B(t, a, b, c, d)
        return(cur_H_A, cur_H_B)


    def variational_y_dot(self, t, y, reg_timescale = -1):
        # y is the same format as R, so [A_1, A_2 ... A_N, z_1,1, z_2,1 ... z_N,1 ... z_1,(M-1) ... z_N,(M-1)]

        # We also create a standardised y which includes z_n,M = 1
        # we note that z_n,m = y_std[N + N * m + n]

        y_std = np.ones(self.N * (self.M + 1), dtype = complex)
        y_std[:self.N * self.M] = y

        # we want to read the semaphor right here in this function

        X = self.calculate_overlap_matrices(y, 3)
        cur_H_A, cur_H_B = self.calculate_hamiltonian_tensors(t)

        R = np.zeros(self.M * self.N, dtype=complex)
        m_Theta = np.zeros((self.M*self.N, self.M*self.N), dtype=complex)

        # First we calculate R
        for k in range(self.N):
            for j in range(self.N):
                # First, the Hamiltonian matrix element { z_k | H | z_j }
                H_mel_s1 = 0.0
                for a in range(self.M):
                    for b in range(self.M):
                        H_mel_s1 += cur_H_A[a][b] * np.conjugate(y_std[self.N + self.N * a + k]) * y_std[self.N + self.N * b + j]
                H_mel_s2 = 0.0
                for a in range(self.M):
                    for b in range(self.M):
                        for c in range(self.M):
                            for d in range(self.M):
                                H_mel_s2 += cur_H_B[a][b][c][d] * np.conjugate(y_std[self.N + self.N * a + k]) * np.conjugate(y_std[self.N + self.N * b + k]) * y_std[self.N + self.N * c + j] * y_std[self.N + self.N * d + j]
                H_mel = self.S * X[k][j][1] * H_mel_s1 + 0.5 * self.S * (self.S - 1) * X[k][j][2] * H_mel_s2

                R[k] += y_std[j] * H_mel

                for m in range(self.M-1):
                    H_mel_diff = self.S * (self.S - 1) * X[k][j][2] * y_std[self.N + self.N * m + j] * H_mel_s1 + 0.5 * self.S * (self.S - 1) * (self.S - 2) * X[k][j][3] * y_std[self.N + self.N * m + j] * H_mel_s2
                    term3 = 0.0
                    for b in range(self.M):
                        term3 += cur_H_A[m][b] * y_std[self.N + self.N * b + j]
                    term4 = 0.0
                    for b in range(self.M):
                        for c in range(self.M):
                            for d in range(self.M):
                                term4 += (cur_H_B[m][b][c][d] + cur_H_B[b][m][c][d]) * np.conjugate(y_std[self.N + self.N * b + k]) * y_std[self.N + self.N * c + j] * y_std[self.N + self.N * d + j]
                    H_mel_diff += self.S * X[k][j][1] * term3 + 0.5 * self.S * (self.S - 1) * X[k][j][2] * term4

                    R[self.N + self.N * m + k] += np.conjugate(y_std[k]) * y_std[j] * H_mel_diff

        # Then, we calculate Theta
        # First, we fill in X
        for i in range(self.N):
            for j in range(self.N):
                m_Theta[i][j] = X[i][j][0]
        # Then, we fill in Y and Y^h.c.
        for a in range(self.N):
            for b in range(self.M-1):
                for d in range(self.N):
                    m_Theta[a][self.N + b * self.N + d] = self.S * np.conjugate(y_std[self.N + self.N * b + a]) * y_std[d] * X[a][d][1]
                    m_Theta[self.N + b * self.N + d][a] = np.conjugate(m_Theta[a][self.N + b * self.N + d])
        # Then, we fill in Z
        for i in range(self.M-1):
            for j in range(self.M-1):
                for a in range(self.N):
                    for b in range(self.N):
                        # first, we evaluate (F_ij)_ab
                        m_Theta[self.N + i * self.N + a][self.N + j * self.N + b] = self.S * (self.S - 1) * X[a][b][2] * np.conjugate(y_std[self.N + self.N * j + a]) * y_std[self.N + self.N * i + b]
                        if i == j:
                            m_Theta[self.N + i * self.N + a][self.N + j * self.N + b] += self.S * X[a][b][1]
                        m_Theta[self.N + i * self.N + a][self.N + j * self.N + b] *= np.conjugate(y_std[a]) * y_std[b]

        # Regularisation
        if reg_timescale != -1:
            for i in range(self.M * self.N):
                m_Theta[i][i] += reg_timescale

        # Finally, we calculate y dot
        m_Theta_inv = np.linalg.inv(m_Theta)

        # Semaphor
        self.update_semaphor(t)

        return( - 1j * m_Theta_inv.dot(R))


    # ---------------------------------------------------------
    # ------------------- Semaphor methods --------------------
    # ---------------------------------------------------------

    def update_semaphor(self, t):
        # check if semaphor finished
        if self.semaphor_next_flag_t_i >= len(self.semaphor_t_space):
            print("  Semaphor reached the final flagged timestamp. No further semaphor update necessary.", end='\r')
            return(0)
        if t >= self.semaphor_t_space[self.semaphor_next_flag_t_i]:
            # We find the next smallest unreached semaphor flag
            t_i_new = self.semaphor_next_flag_t_i
            while(self.semaphor_t_space[t_i_new] < t):
                t_i_new += 1
                if t_i_new >= len(self.semaphor_t_space):
                    break
            self.semaphor_next_flag_t_i = t_i_new
            progress_fraction = (self.semaphor_t_space[t_i_new - 1] - self.semaphor_simulation_start_time) / (self.semaphor_t_space[-1] - self.semaphor_simulation_start_time)
            ETA = time.strftime("%H:%M:%S", time.localtime( (time.time()-self.semaphor_start_time) / progress_fraction + self.semaphor_start_time ))
            print("  " + str(int(100 * progress_fraction)).zfill(2) + "% done; est. time of finish: " + ETA, end='\r')






    ###########################################################################
    # ------------------------------- USER METHODS ----------------------------
    ###########################################################################
    # Methods intended to be invoked by the user

    # ---------------------------------------------------------
    # ------------------- Physical methods --------------------
    # ---------------------------------------------------------

    def set_global_parameters(self, M, S):
        self.M = M
        self.S = S
        print("A driven Bose-Hubbard model with %i bosons over %i modes has been initialized." % (self.S, self.M))

    # The general SU(N) Hamiltonian is fully described by the tensors A_a,b and B_a,b,c,d

    def OLD_set_hamiltonian_tensors(self, A, B):
        # TODO optimalize these due to their symmetries? (A twofold and B eightfold) Maybe by allowing them to be callable?
        if self.M == 0:
            print("ERROR: You have attempted to initialize the Hamiltonian tensors before initializing the mode and particle number of the system. You can do this by calling set_global_parameters(M, S).")
            return(-1)

        candidate_A = np.array(A)
        candidate_B = np.array(B)

        if candidate_A.shape != (self.M, self.M):
            print("ERRORL: The A_a,b tensor should be of dimension (M x M).")
            return(-1)

        if candidate_B.shape != (self.M, self.M, self.M, self.M):
            print("ERROR: The B_a,b,c,d tensor should be of dimension (M x M x M x M).")
            return(-1)

        # Test the symmetry
        for a in range(self.M):
            for b in range(self.M):
                if candidate_A[a][b] != np.conjugate(candidate_A[b][a]):
                    print("ERROR: The A_a,b tensor should be Hermitian.")
                    return(-1)

        for a in range(self.M):
            for b in range(self.M):
                for c in range(self.M):
                    for d in range(self.M):
                        if len(set([candidate_B[a][b][c][d], candidate_B[b][a][c][d], candidate_B[a][b][d][c], candidate_B[b][a][d][c] ])) > 1:
                            print("ERROR: The B_a,b,c,d tensor should be symmetric with respect to inverting both the first and the second pair of indices.")
                            return(-1)
                        if candidate_B[a][b][c][d] != np.conjugate(candidate_B[d][c][b][a]):
                            print("ERROR: The B_a,b,c,d tensor should be be Hermitian with respect to the two index pairs like so: B_a,b,c,d = (B_c,d,a,b)*.")
                            return(-1)

        self.H_A = candidate_A
        self.H_B = candidate_B

        self.is_phys_init = True #TODO these have to be callable, otherwise there can be no time dependence!

    def set_hamiltonian_tensors(self, A, B):
        # A(t,a,b) and B(t,a,b,c,d) are CALLABLE objects!!!
        if self.M == 0:
            print("ERROR: You have attempted to initialize the Hamiltonian tensors before initializing the mode and particle number of the system. You can do this by calling set_global_parameters(M, S).")
            return(-1)


        """A_signature = inspect.signature(A)
        if len(A_signature.parameters) != 3:
            print("ERROR: A should be a callable which takes 3 arguments (time and 2 indices).")
            return(-1)

        B_signature = inspect.signature(B)
        if len(A_signature.parameters) != 4:
            print("ERROR: A should be a callable which takes 5 arguments (time and 4 indices).")
            return(-1)"""

        # Test the symmetry
        for a in range(self.M):
            for b in range(self.M):
                if A(0,a,b) != np.conjugate(A(0,b,a)):
                    print("ERROR: The A_a,b tensor should be Hermitian.")
                    return(-1)

        for a in range(self.M):
            for b in range(self.M):
                for c in range(self.M):
                    for d in range(self.M):
                        if len(set([B(0,a,b,c,d), B(0,b,a,c,d), B(0,a,b,d,c), B(0,b,a,d,c) ])) > 1:
                            print("ERROR: The B_a,b,c,d tensor should be symmetric with respect to inverting both the first and the second pair of indices.")
                            return(-1)
                        if B(0,a,b,c,d) != np.conjugate(B(0,d,c,b,a)):
                            print("ERROR: The B_a,b,c,d tensor should be be Hermitian with respect to the two index pairs like so: B_a,b,c,d = (B_c,d,a,b)*.")
                            return(-1)

        self.H_A = A
        self.H_B = B

        self.is_phys_init = True #TODO these have to be callable, otherwise there can be no time dependence!

    # ---------------------------------------------------------
    # ------------------- Sampling methods --------------------
    # ---------------------------------------------------------

    # We will always use the inverted overlap matrix identity, and so sampling may be done with any method.

    def sample_gaussian(self, z_0 = np.array([]), width = 1.0, conditioning_limit = -1, N_max = 50, max_saturation_steps = 50):

        if self.M == 0:
            print("ERROR: You have attempted to sample a basis set before initializing the mode and particle number of the system. You can do this by calling set_global_parameters(M, S).")
            return(-1)

        # z_0 must be of the Aguiar variety
        self.sampling_method = "gaussian"
        self.sampling_z_0 = z_0
        self.sampling_width = width
        self.sampling_conditioning_limit = conditioning_limit
        self.sampling_N_max = N_max
        self.sampling_max_saturation_steps = max_saturation_steps

        if len(z_0) == 0:
            z_0 = np.zeros(self.M-1, dtype=complex)

        new_basis = [z_0]

        print(f"Sampling the neighbourhood of z_0 = {z_0} with a normal distribution of width {width}...")
        steps_since_last_addition = 0

        # Complex positive-definite Hermitian matrix with positive real eigenvalues
        overlap_matrix = [[square_norm(z_0, self.S)]]

        # we omit the normalizations, since what we really care about is the properties of X_ab = {z_a | z_b}, NOT the normalized version

        z_0_std = np.zeros(2 * (self.M - 1))
        for m in range(self.M - 1):
            z_0_std[m] = z_0[m].real
            z_0_std[self.M - 1 + m] = z_0[m].imag

        while(len(new_basis) < N_max):
            # Grab a new candidate

            candidate_basis_vector_std = np.random.normal(z_0_std, width)
            candidate_basis_vector = np.zeros(self.M - 1, dtype=complex)
            for m in range(self.M - 1):
                candidate_basis_vector[m] = candidate_basis_vector_std[m] + candidate_basis_vector_std[self.M - 1 + m] * 1j

            # If conditioning factor specified, see if satisfies
            satisfying = True
            if conditioning_limit != -1:
                candidate_basis = new_basis + [candidate_basis_vector]
                candidate_overlap_matrix = deepcopy(overlap_matrix)
                candidate_overlap_matrix.append([0] * len(candidate_basis))
                for a in range(len(candidate_basis)-1):
                    cur_overlap = overlap(candidate_basis[a], candidate_basis_vector, self.S)
                    candidate_overlap_matrix[a].append(cur_overlap)
                    candidate_overlap_matrix[-1][a] = np.conjugate(cur_overlap)
                candidate_overlap_matrix[-1][-1] = square_norm(candidate_basis_vector, self.S)

                eigenvals, eigenvecs = np.linalg.eig(candidate_overlap_matrix)
                epsilon = np.absolute(max(eigenvals, key=np.absolute)) / np.absolute(min(eigenvals, key=np.absolute))
                if epsilon > conditioning_limit:
                    # Reject
                    satisfying = False
            if satisfying:
                # Accept!
                steps_since_last_addition = 0
                new_basis = candidate_basis
                overlap_matrix = candidate_overlap_matrix
            else:
                steps_since_last_addition += 1

            # Check if saturated
            if max_saturation_steps != -1 and steps_since_last_addition > max_saturation_steps:
                print("  Sampling procedure saturated!")
                break

        self.basis = new_basis

        self.N = len(self.basis)
        print(f"  A sample of N = {self.N} basis vectors around z_0 = {z_0} drawn from a normal distribution of width {self.sampling_width} has been initialized.")

        # Create the identity operator matrix
        X = np.zeros((self.N, self.N), dtype = complex)
        for m in range(self.N):
            for n in range(self.N):
                X[m][n] = overlap(self.basis[m], self.basis[n], self.S)
        self.inverse_overlap_matrix = np.linalg.inv(X)


        self.is_basis_init = True


    # ---------------------------------------------------------
    # ------------------ Simulation methods -------------------
    # ---------------------------------------------------------

    def set_initial_wavefunction(self, initial_wavefunction = [], message = ""):
        # Allows different methods for finding the initial decomposition, as
        # encoded by the 'message' argument

        if not self.is_basis_init:
            print("ERROR: You have attempted to set the initial wavefunction state before sampling a basis set. You can do this by calling one of the many basis-sampling methods listed below.")
            for m, desc in bosonic_su_n.basis_sampling_methods.items():
                print(f"{m} [{desc}]")
            return(-1)

        if message in ["manual", "A_vals"]:
            # initial_wavefunction is just directly equal to the initial decomposition coefficients of the wavefunction
            candidate_initial_wavefunction = np.array(initial_wavefunction)
            if candidate_initial_wavefunction.shape != (self.N,):
                print("ERROR: When using the 'manual' method, initial_wavefunction has to be an array of shape (N).")
                return(-1)
            candidate_wavef = candidate_initial_wavefunction
        elif message in ["aguiar"]:
            # initial_wavefunction describes an Aguiar unnormalized coherent state.
            candidate_initial_wavefunction = np.array(initial_wavefunction)
            if candidate_initial_wavefunction.shape != (self.M - 1,):
                print("ERROR: When using the 'aguiar' method, initial_wavefunction has to be an array of shape (M-1).")
                return(-1)
            candidate_wavef = self.decompose_aguiar(candidate_initial_wavefunction)
        elif message in ["grossmann", "frank"]:
            # initial_wavefunction describes a Grossmann normalized coherent state.
            candidate_initial_wavefunction = np.array(initial_wavefunction)
            if candidate_initial_wavefunction.shape != (self.M,):
                print("ERROR: When using the 'grossmann' method, initial_wavefunction has to be an array of shape (M).")
                return(-1)
            if np.round(np.sum(candidate_initial_wavefunction.real * candidate_initial_wavefunction.real + candidate_initial_wavefunction.imag * candidate_initial_wavefunction.imag), 2) != 1.0:
                print("ERROR: When using the 'grossmann' method, initial_wavefunction has to have norm 1.")
                return(-1)
            candidate_wavef = self.decompose_grossmann(candidate_initial_wavefunction)
        elif message in ["", "NONE"]:
            print("set_initial_wavefunction called without specifying the method of initialization.")
            print("Default option selected: initial wavefunction will be set to the first element of the basis set.")
            candidate_wavef = self.decompose_aguiar(self.basis[0])
            message = "NONE"
        else:
            print("ERROR: Unknown method of wavefunction initialization.")
            return(-1)

        Psi_mag = 0.0
        for i in range(self.N):
            for j in range(self.N):
                Psi_mag += np.conjugate(candidate_wavef[i]) * candidate_wavef[j] * overlap(self.basis[i], self.basis[j], self.S)
        print(f"  Initial wavefunction normalized with magnitude {Psi_mag.real:.4f}. The final decomposition will be naively renormalized...")

        # Naive renormalization
        for i in range(self.N):
            candidate_wavef[i] /= np.sqrt(Psi_mag.real)
        self.wavef = candidate_wavef

        self.wavef_initial_wavefunction = initial_wavefunction
        self.wavef_message = message

        self.is_wavef_init = True
        return(0)


    # From the initialized basis and having inputted the initial wavefunction,
    # the user may choose to simulate the uncoupled basis evolution, the
    # coupled basis evolution, the evolution of the wavefunction decomposition
    # on top of both of these, or the full variational method evolution.
    # Each time, the previous "evolution" dataset is erased

    def simulate_uncoupled_basis(self, max_t, N_dtp, rtol = 1e-3, reg_timescale = -1, N_semaphor = 100):

        # reg_timescale may be a tuple, in which case the first element is the basis timescale and the second element the wavef timescale
        if isinstance(reg_timescale, tuple):
            reg_timescale_basis, reg_timescale_wavef = reg_timescale
        else:
            reg_timescale_basis = reg_timescale
            reg_timescale_wavef = reg_timescale

        print("Simulating state evolution by propagating uncoupled basis vectors one by one and then evolving the decomposition.")

        if not self.is_phys_init:
            print("ERROR: You have attempted to simulate state evolution before specifying the system hamiltonian. You can do this by calling set_hamiltonian_tensors(A, B).")
            return(-1)

        if not self.is_basis_init:
            print("ERROR: You have attempted to simulate state evolution before sampling a basis set. You can do this by calling one of the many basis-sampling methods listed below.")
            for m, desc in bosonic_su_n.basis_sampling_methods.items():
                print(f"{m} [{desc}]")
            return(-1)
        if not self.is_wavef_init:
            print("ERROR: You have attempted to simulate state evolution before specifying the initial wavefunction state. You can do this by calling set_initial_wavefunction(initial_wavefunction, message).")
            return(-1)

        # We propagate each basis vector one by one

        # Set up semaphor for basis propagation
        self.semaphor_t_space = np.arange(self.N)
        self.semaphor_simulation_start_time = 0
        self.semaphor_start_time = time.time()
        self.semaphor_next_flag_t_i = 1
        self.semaphor_ETA = "???"

        simulation_start_time = 0.0 # TODO is data loaded, this should be the last timestamp
        self.t_space = [simulation_start_time]

        self.basis_evol = [np.zeros((self.N, self.M-1), dtype=complex)]
        for i in range(self.N):
            for j in range(self.M-1):
                self.basis_evol[0][i][j] = self.basis[i][j]
        for t_i in range(1, N_dtp+1):
            self.basis_evol.append(np.zeros((self.N, self.M-1), dtype=complex))

        print(f"Uncoupled basis propagation on a timescale of t = ({simulation_start_time} - {max_t}), rtol = {rtol} at {time.strftime("%H:%M:%S", time.localtime( self.semaphor_start_time))}")

        for n in range(self.N):
            y_0 = self.basis[n].copy()
            iterated_solution = sp.integrate.solve_ivp(self.uncoupled_basis_y_dot, [simulation_start_time, max_t], y_0, method = 'RK45', t_eval = np.linspace(simulation_start_time, max_t, N_dtp+1), args = (reg_timescale_basis,), rtol = rtol)
            for t_i in range(1, N_dtp+1):
                for m in range(self.M - 1):
                    self.basis_evol[t_i][n][m] = iterated_solution.y[m][t_i]
                if n == 0:
                    # If this is the first one, we initialize self.t_space
                    self.t_space.append(iterated_solution.t[t_i])
            #self.update_semaphor(n)

        print("  Basis propagation finished at " + time.strftime("%H:%M:%S", time.localtime(time.time())) + "; " + str(N_dtp) + " datapoints saved.                  ")
        self.is_basis_evol = True

        # ---------------- wavef propagation ------------------
        # We propagate the wavefunction
        print("  Propagating wavefunction decomposition...")
        self.wavef_evol = [np.zeros(self.N, dtype=complex)]
        for i in range(self.N):
            self.wavef_evol[0][i] = self.wavef[i]
        for t_i in range(1, N_dtp+1):
            self.wavef_evol.append(np.zeros(self.N, dtype=complex))

        # Set up semaphor for basis propagation
        self.semaphor_t_space = np.linspace(self.t_space[0], self.t_space[-1], N_semaphor + 1)
        self.semaphor_simulation_start_time = self.t_space[0]
        self.semaphor_start_time = time.time()
        self.semaphor_next_flag_t_i = 1
        self.semaphor_ETA = "???"

        print(f"Wavefunction propagation over the evolved uncoupled basis on a timescale of t = ({self.t_space[0]} - {self.t_space[-1]}), rtol = {rtol} at {time.strftime("%H:%M:%S", time.localtime( self.semaphor_start_time))}")

        y_0 = self.wavef.copy()
        iterated_solution = sp.integrate.solve_ivp(self.uncoupled_wavef_y_dot, [self.t_space[0], max_t], y_0, method = 'RK45', t_eval = self.t_space, args = (reg_timescale_wavef,), rtol = rtol)

        for t_i in range(1, N_dtp+1):
            for n in range(self.N):
                self.wavef_evol[t_i][n] = iterated_solution.y[n][t_i]

        print("  Simulation finished at " + time.strftime("%H:%M:%S", time.localtime(time.time())) + "; " + str(N_dtp) + " datapoints saved.                  ")
        self.is_wavef_evol = True




    # ---------------------------------------------------------
    # --------------- Graphical output methods ----------------
    # ---------------------------------------------------------

    def plot_data(self, graph_list = ["expected_mode_occupancy", "initial_decomposition"], save_graph=True):

        print("Graphical output plotting routine initialized.")
        # create a superplot
        superplot_shape_x, superplot_shape_y = subplot_dimensions(len(graph_list))

        plt.figure(figsize=(15, 8))

        # initialize lims; if changed, apply them afterwards
        x_left = -1
        x_right = -1
        y_left = -1
        y_right = -1

        include_legend = True

        for i in range(len(graph_list)):
            plt.subplot(superplot_shape_y, superplot_shape_x, i + 1)
            if graph_list[i] == "initial_decomposition":
                if self.M != 2:
                    print("  ERROR: Attempting to plot 'initial_decomposition' with unsuitable mode number.")
                    continue
                x_vals = np.zeros(self.N)
                y_vals = np.zeros(self.N)
                s_vals = np.zeros(self.N)

                for n in range(self.N):
                    x_vals[n] = self.basis_evol[0][n][0].real
                    y_vals[n] = self.basis_evol[0][n][0].imag
                    s_vals[n] = np.absolute(self.wavef_evol[0][n])

                plt.xlabel("$\\Re\\left(z_n\\right)$")
                plt.ylabel("$\\Im\\left(z_n\\right)$")

                plt.scatter(x_vals, y_vals, 5 + np.power(s_vals, 1/10) * 450)
                include_legend = False
            elif graph_list[i] == 'expected_mode_occupancy':
                print("  Plotting the time evolution of expected mode occupancy...")#, end='', flush=True)
                plt.title("Expected mode occupancy")
                plt.xlabel("t")

                print("    Plotting measured wavefunction magnitude and mode occupancies...")
                psi_mag = np.zeros(len(self.t_space))

                avg_n = []
                for m in range(self.M):
                    avg_n.append(np.zeros(len(self.t_space)))

                for t_i in range(len(self.t_space)):
                    for a in range(self.N):
                        for b in range(self.N):
                            cur_X_1 = overlap(self.basis_evol[t_i][a], self.basis_evol[t_i][b], self.S, r = 1)
                            psi_mag[t_i] += (np.conjugate(self.wavef_evol[t_i][a]) * self.wavef_evol[t_i][b] * overlap(self.basis_evol[t_i][a], self.basis_evol[t_i][b], self.S)).real
                            for m in range(self.M-1):
                                avg_n[m][t_i] += (np.conjugate(self.wavef_evol[t_i][a]) * self.wavef_evol[t_i][b] * np.conjugate(self.basis_evol[t_i][a][m]) * self.basis_evol[t_i][b][m] * cur_X_1).real
                            avg_n[self.M-1][t_i] += (np.conjugate(self.wavef_evol[t_i][a]) * self.wavef_evol[t_i][b] * cur_X_1).real

                # We insert horizontal lines indicating occupancies calculated from z_0
                initial_occupancy = np.zeros(self.M)
                if self.wavef_message == "aguiar":
                    z_0 = np.array(self.wavef_initial_wavefunction)
                    for m in range(self.M - 1):
                        initial_occupancy[m] = (z_0[m].real * z_0[m].real + z_0[m].imag * z_0[m].imag) / (1 + np.sum(np.conjugate(z_0) * z_0).real)
                    initial_occupancy[self.M - 1] = 1 / (1 + np.sum(np.conjugate(z_0) * z_0).real)
                elif self.wavef_message == "grossmann":
                    xi_0 = np.array(self.wavef_initial_wavefunction)
                    xi_0_reduced_overlap = np.power(np.sum(np.conjugate(xi_0) * xi_0), self.S - 1).real
                    for m in range(self.M):
                        initial_occupancy[m] = (xi_0[m].real * xi_0[m].real + xi_0[m].imag * xi_0[m].imag) * xi_0_reduced_overlap
                elif self.wavef_message == "NONE":
                    z_0 = self.basis[0]
                    for m in range(self.M - 1):
                        initial_occupancy[m] = (z_0[m].real * z_0[m].real + z_0[m].imag * z_0[m].imag) / (1 + np.sum(np.conjugate(z_0) * z_0).real)
                    initial_occupancy[self.M - 1] = 1 / (1 + np.sum(np.conjugate(z_0) * z_0).real)
                for m in range(self.M):
                    plt.axhline(y = initial_occupancy[m], linestyle = "dotted", label = "init. $\\langle N_" + str(m+1) + " \\rangle/S$")



                # if two-mode, we insert the algebraic solution
                if self.M == 2:
                    print("    Plotting theoretical wavefunction magnitude and mode occupancies for the two-mode solution...")
                    cur_c_0 = np.zeros(self.S + 1, dtype = complex)
                    for i in range(self.S + 1):
                        for a in range(self.N):
                            cur_c_0[i] += self.wavef_evol[0][a] * np.power(self.basis_evol[0][a], i)
                        cur_c_0[i] *= np.sqrt( math.factorial(self.S) / (math.factorial(i) * math.factorial(self.S-i)) )

                    N_space_t_space = np.linspace(self.t_space[0], self.t_space[-1], 200)
                    res_N_space = self.two_mode_solution(N_space_t_space, cur_c_0)

                    plt.plot(N_space_t_space, res_N_space, linestyle = "dashed", label = "theor. $\\langle N_1 \\rangle/S$")
                    plt.plot(N_space_t_space, res_N_space * (-1) + 1, linestyle = "dashed", label = "theor. $\\langle N_2 \\rangle/S$")


                plt.plot(self.t_space, psi_mag, label="$\\langle \\Psi | \\Psi \\rangle$")
                for m in range(self.M):
                    plt.plot(self.t_space, avg_n[m], label="$\\langle N_" + str(m+1) + " \\rangle/S$")
                #print(" Done!")

            if x_left != -1:
                plt.xlim(x_left, x_right)
            if y_left != -1:
                plt.ylim(y_left, y_right)

            if include_legend:
                plt.legend()

        plt.tight_layout()
        if save_graph:
            Path(f"outputs/{self.output_subfolder_name}").mkdir(parents=True, exist_ok=True)
            plt.savefig(f"outputs/{self.output_subfolder_name}/" + str(self.ID) + "_graph_output.png")
        plt.show()




    # ---------------------------------------------------------
    # ----------------- Data storage methods ------------------
    # ---------------------------------------------------------
    # Loading and saving

    def save_data(self):
        # This creates (or accesses) a subfolder with ID name in outputs/
        # and saves the data which was initialized so far (i.e. we are
        # able to save e.g. configuration only, no evolution needed.)

        # Create subfolder if not exists
        Path(f"outputs/{self.output_subfolder_name}").mkdir(parents=True, exist_ok=True)
        dir_path = f"outputs/{self.output_subfolder_name}/"


        # Saves configuration
        config_file = open(dir_path + self.output_config_filename + ".txt", "w")
        config_file.write(", ".join(str(x) for x in [self.is_phys_init, self.is_basis_init, self.is_wavef_init, self.is_basis_evol, self.is_wavef_evol]) + "\n")
        # The following lines dynamically describe the aforementioned initializations
        if self.is_phys_init:
            config_file.write(", ".join(str(x) for x in [self.M, self.S]) + "\n")
        if self.is_basis_init:
            if self.sampling_method == "gaussian":
                config_file.write(", ".join(str(x) for x in [self.sampling_method, self.sampling_width, self.sampling_conditioning_limit, self.sampling_N_max, self.sampling_max_saturation_steps]) + "\n")
                config_file.write(", ".join(str(x) for x in self.sampling_z_0) + "\n")
        if self.is_wavef_init:
            config_file.write(self.wavef_message + "\n")
            if self.wavef_message != "NONE":
                config_file.write(", ".join(str(x) for x in self.wavef_initial_wavefunction) + "\n")

        config_file.close()


        # If they exist, stores the Hamiltonian tensors
        if self.is_phys_init:
            H_A_file = open(dir_path + self.output_H_A_filename + ".txt", "w")
            H_A_file.write(inspect.getsource(self.H_A))
            H_A_file.close()
            H_B_file = open(dir_path + self.output_H_B_filename + ".txt", "w")
            H_B_file.write(inspect.getsource(self.H_B))
            H_B_file.close()

        # If they exist, stores basis and wavef initialization
        if self.is_basis_init:
            basis_init_file = open(dir_path + self.output_basis_init_filename + ".txt", "w")
            for basis_vector in self.basis:
                basis_init_file.write(", ".join(str(x) for x in basis_vector) + "\n")
            basis_init_file.close()

        if self.is_wavef_init:
            wavef_init_file = open(dir_path + self.output_wavef_init_filename + ".txt", "w")
            wavef_init_file.write(", ".join(str(x) for x in self.wavef) + "\n")
            wavef_init_file.close()

        # If they exist, stores basis and wavef evolution
        if self.is_basis_evol:
            basis_evol_file = open(dir_path + self.output_basis_evol_filename + ".csv", "w")
            basis_evol_writer = csv.writer(basis_evol_file)
            header_row = ["t"]

            for n in range(self.N):
                for m in range(self.M - 1):
                    header_row.append("z_" + str(n).zfill(self.N_fill) + "_" + str(m).zfill(self.M_fill))
            basis_evol_writer.writerow(header_row)

            for t_i in range(len(self.t_space)):
                cur_row = [self.t_space[t_i]]
                for n in range(self.N):
                    for m in range(self.M - 1):
                        cur_row.append(self.basis_evol[t_i][n][m])
                basis_evol_writer.writerow(cur_row)
            basis_evol_file.close()

        if self.is_wavef_evol:
            wavef_evol_file = open(dir_path + self.output_wavef_evol_filename + ".csv", "w")
            wavef_evol_writer = csv.writer(wavef_evol_file)
            header_row = ["t"]

            for n in range(self.N):
                header_row.append("A_" + str(n).zfill(self.N_fill))
            wavef_evol_writer.writerow(header_row)

            for t_i in range(len(self.t_space)):
                cur_row = [self.t_space[t_i]]
                for n in range(self.N):
                    cur_row.append(self.wavef_evol[t_i][n])
                # TODO add physical descriptors such as <E> here?
                wavef_evol_writer.writerow(cur_row)
            wavef_evol_file.close()

    def load_data(self, config_load = [True, True, True, True, True]):
        # config_load specifies which properties should we load. I.e. even if there exists data
        # for basis_evol, if we pass config_load = [True, True, True, False, False], only
        # H_A, H_B, and basis_init and wavef_init will load.

        dir_path = f"outputs/{self.output_subfolder_name}/"
        # First, we load the config
        config_file = open(dir_path + self.output_config_filename + ".txt", 'r')
        config_lines = [line.rstrip('\n') for line in config_file]
        config_file.close()

        first_line_list = config_lines[0].split(", ")
        past_is_phys_init = bool(first_line_list[0])
        past_is_basis_init = bool(first_line_list[1])
        past_is_wavef_init = bool(first_line_list[2])
        past_is_basis_evol = bool(first_line_list[3])
        past_is_wavef_evol = bool(first_line_list[4])

        cur_config_line_index = 1

        if past_is_phys_init and config_load[0]:
            # Load phys
            cur_line_list = config_lines[cur_config_line_index].split(", ")
            cur_config_line_index += 1
            M = int(cur_line_list[0])
            S = int(cur_line_list[1])
            self.set_global_parameters(M, S)

            H_A_file = open(dir_path + self.output_H_A_filename + ".txt", "r")
            H_A_fn_body = H_A_file.read()
            H_A_file.close()
            exec(H_A_fn_body) # This creates the function locally
            H_A_fn_name = H_A_fn_body.split(" ", 1)[-1].split("(")[0]
            self.H_A = eval(H_A_fn_name) # This refers to the function object

            H_B_file = open(dir_path + self.output_H_B_filename + ".txt", "r")
            H_B_fn_body = H_B_file.read()
            H_B_file.close()
            exec(H_B_fn_body) # This creates the function locally
            H_B_fn_name = H_B_fn_body.split(" ", 1)[-1].split("(")[0]
            self.H_B = eval(H_B_fn_name) # This refers to the function object

            self.is_phys_init = True

            print("  Hamiltonian tensors loaded.")

        if past_is_basis_init and config_load[1]:
            # Load basis init
            cur_line_list = config_lines[cur_config_line_index].split(", ")
            cur_config_line_index += 1
            self.sampling_method = cur_line_list[0]
            if self.sampling_method == "gaussian":
                self.sampling_width = float(cur_line_list[1])
                self.sampling_conditioning_limit = float(cur_line_list[2])
                self.sampling_N_max = int(cur_line_list[3])
                self.sampling_max_saturation_steps = int(cur_line_list[4])

                cur_line_list = config_lines[cur_config_line_index].split(", ")
                cur_config_line_index += 1
                self.sampling_z_0 = np.array([complex(x) for x in cur_line_list], dtype=complex)

            basis_init_file = open(dir_path + self.output_basis_init_filename + ".txt", "r")
            basis_init_lines = [line.rstrip('\n') for line in basis_init_file]
            basis_init_file.close()

            self.basis = []
            for basis_vec_line in basis_init_lines:
                self.basis.append(np.array([complex(x) for x in basis_vec_line.split(", ")], dtype=complex))

            self.N = len(self.basis)

            # Create the identity operator matrix
            X = np.zeros((self.N, self.N), dtype = complex)
            for m in range(self.N):
                for n in range(self.N):
                    X[m][n] = overlap(self.basis[m], self.basis[n], self.S)
            self.inverse_overlap_matrix = np.linalg.inv(X)

            self.is_basis_init = True
            if self.sampling_method == "gaussian":
                print(f"  A sample of N = {self.N} basis vectors around z_0 = {self.sampling_z_0} drawn from a normal distribution of width {self.sampling_width} has been loaded.")

        if past_is_wavef_init and config_load[2]:
            # Load wavef init
            self.wavef_message = config_lines[cur_config_line_index]
            cur_config_line_index += 1
            if self.wavef_message != "NONE":
                cur_line_list = config_lines[cur_config_line_index].split(", ")
                cur_config_line_index += 1
                self.wavef_initial_wavefunction = np.array([complex(x) for x in cur_line_list], dtype=complex)

            wavef_init_file = open(dir_path + self.output_wavef_init_filename + ".txt", "r")
            wavef_init_lines = [line.rstrip('\n') for line in wavef_init_file]
            wavef_init_file.close()

            self.wavef = np.array([complex(x) for x in wavef_init_lines[0].split(", ")], dtype=complex)

            self.is_wavef_init = True
            if self.wavef_message == "manual":
                print("  An initial wavefunction decomposition has been loaded from a manual decomposition input.")
            elif self.wavef_message == "aguiar":
                print(f"  An initial wavefunction decomposition has been loaded from a pure Aguiar coherent state | z }} = {self.wavef_initial_wavefunction}.")
            elif self.wavef_message == "grossmann":
                print(f"  An initial wavefunction decomposition has been loaded from a pure Grossmann coherent state | xi > = {self.wavef_initial_wavefunction}.")
            elif self.wavef_message == "NONE":
                if self.is_basis_init:
                    print(f"  An initial wavefunction decomposition has been loaded from a pure Aguar coherent state (the first element of the basis set), | z }} = {self.basis[0]}.")
                else:
                    print(f"  An initial wavefunction decomposition has been loaded from a pure Aguar coherent state (the first element of the basis set), but the basis has not been initialized.")

        if past_is_basis_evol and config_load[3]:

            if not self.is_phys_init:
                print("ERROR: You cannot load basis evolution without loading or setting the physics configuration.")
                return(-1)

            # Load basis evol
            basis_evol_file = open(dir_path + self.output_basis_evol_filename + ".csv", newline='')
            basis_evol_reader = csv.reader(basis_evol_file, delimiter=',', quotechar='"')
            basis_evol_rows = list(basis_evol_reader)
            basis_evol_file.close()

            header_row = basis_evol_rows[0]

            N_dtp = len(basis_evol_rows) - 1
            self.t_space = [] # np.zeros(N_dtp)
            self.basis_evol = [] # np.zeros((N_dtp, self.N, self.M-1), dtype=complex)

            for i in range(N_dtp):
                # we append the empty ndarrays
                self.basis_evol.append(np.zeros((self.N, self.M - 1), dtype = complex))

                self.t_space.append(float(basis_evol_rows[i+1][0]))

                for n in range(self.N):
                    for m in range(self.M-1):
                        self.basis_evol[i][n][m] = complex(basis_evol_rows[i+1][1 + (self.M - 1) * n + m])

            self.is_basis_evol = True
            print(f"  Basis evolution ({N_dtp} datapoints) loaded.")

        if past_is_wavef_evol and config_load[4]:

            if not self.is_phys_init:
                print("ERROR: You cannot load wavefunction evolution without loading or setting the physics configuration.")
                return(-1)
            if not self.is_basis_init:
                print("ERROR: You cannot load wavefunction evolution without loading the basis initialization.")
                return(-1)

            # Load wavef evol
            wavef_evol_file = open(dir_path + self.output_wavef_evol_filename + ".csv", newline='')
            wavef_evol_reader = csv.reader(wavef_evol_file, delimiter=',', quotechar='"')
            wavef_evol_rows = list(wavef_evol_reader)
            wavef_evol_file.close()

            header_row = wavef_evol_rows[0]

            N_dtp = len(wavef_evol_rows) - 1
            if len(self.t_space) not in [0, N_dtp]:
                print(f"ERROR: Attempted to load {N_dtp} wavef_evol datapoints, but the internal length of t_space is already set to {len(self.t_space)}.")
                return(-1)
            self.t_space = [] # np.zeros(N_dtp)
            self.wavef_evol = [] # np.zeros((N_dtp, self.N), dtype=complex)

            for i in range(N_dtp):
                # we append the empty ndarrays
                self.wavef_evol.append(np.zeros(self.N, dtype = complex))
                self.t_space.append(float(wavef_evol_rows[i+1][0]))

                for n in range(self.N):
                    self.wavef_evol[i][n] = complex(wavef_evol_rows[i+1][1 + n])

            self.is_wavef_evol = True
            print(f"  Wavefunction evolution ({N_dtp} datapoints) loaded.")






















