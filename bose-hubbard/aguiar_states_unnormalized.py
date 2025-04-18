import math
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


from copy import deepcopy
import time

import seaborn as sns
import pandas as pd

import csv

from numerical_integration_methods import *

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


# ---------------------- two-mode solution ----------------------
def two_mode_solution(t_space, S, U, J, c_0 = False):

    # If J is callable, we just solve the ivp in a silly way
    if callable(J):
        def c_dot(t, c):
            sussy_matrix = np.zeros((S+1, S+1), dtype=complex)
            for i in range(S + 1):
                sussy_matrix[i][i  ] = - 1j * U / 2 * (i * (i + 1) + (S - i) * (S - i - 1))
                if i != S:
                    sussy_matrix[i][i+1] = 1j * J(t) * np.sqrt(i + 1) * np.sqrt(S - i)
                if i != 0:
                    sussy_matrix[i][i-1] = 1j * J(t) * np.sqrt(i) * np.sqrt(S - i + 1)
            return(np.matmul(sussy_matrix, c))
        sol = sp.integrate.solve_ivp(c_dot, [t_space[0], t_space[-1]], c_0, method = 'RK45', t_eval = t_space)
        N_space = np.zeros(len(t_space)) # N_space[t] = <N_1>/S at t

        for t_i in range(len(t_space)):
            t = t_space[t_i]
            cur_c = np.zeros(S+1, dtype=complex)
            for i in range(S + 1):
                cur_c[i] = sol.y[i][t_i]

            for i in range(S + 1):
                N_space[t_i] += i * (cur_c[i].real * cur_c[i].real + cur_c[i].imag * cur_c[i].imag) / S
        return(N_space)

    if type(c_0) == bool:
        c_0 = np.zeros(S + 1, dtype = complex)
        c_0[0] = 1.0

    sussy_matrix = np.zeros((S+1, S+1), dtype=complex)

    for i in range(S + 1):
        sussy_matrix[i][i  ] = - 1j * U / 2 * (i * (i + 1) + (S - i) * (S - i - 1))
        if i != S:
            sussy_matrix[i][i+1] = 1j * J * np.sqrt(i + 1) * np.sqrt(S - i)
        if i != 0:
            sussy_matrix[i][i-1] = 1j * J * np.sqrt(i) * np.sqrt(S - i + 1)

    # diagonalize

    sussy_eigenvalues, sussy_eigenvectors = np.linalg.eig(sussy_matrix)

    S_mat = sussy_eigenvectors.copy()
    #print(type(S))
    Lambda = np.zeros((S + 1, S + 1), dtype=complex)
    for i in range(S + 1):
        Lambda[i][i] = sussy_eigenvalues[i]

    S_mat_inv = np.linalg.inv(S_mat)

    print("sussy matrix =", sussy_matrix)
    print("S . Lambda . S^-1 =", np.matmul(np.matmul(S_mat, Lambda), S_mat_inv))

    N_space = np.zeros(len(t_space)) # N_space[t] = <N_1>/S at t

    for t_i in range(len(t_space)):
        t = t_space[t_i]

        f_Lambda = np.zeros((S + 1, S + 1), dtype=complex)
        for i in range(S + 1):
            f_Lambda[i][i] = np.exp( t * sussy_eigenvalues[i] )

        cur_transformation = np.matmul(np.matmul(S_mat, f_Lambda), S_mat_inv)

        cur_c = np.matmul(cur_transformation, c_0)

        for i in range(S + 1):
            N_space[t_i] += i * (cur_c[i].real * cur_c[i].real + cur_c[i].imag * cur_c[i].imag) / S
    return(N_space)



class BH():
    #driven Bose-Hubbard model

    # ---------------------------------------------------------
    # -------- Initializers, descriptors, destructors ---------
    # ---------------------------------------------------------

    def __init__(self, ID):
        # Identification
        self.ID = ID

        print("---------------------------- " + str(ID) + " -----------------------------")

        # Data bins are initialized
        self.output_table = []#np.zeros((N_dtp, 2 + self.N * self.M), dtype=complex) # here we'll store all the results smashed together
        self.t_space = []
        self.A_evol = []#np.zeros((N_dtp, self.N), dtype=complex)
        self.basis_evol = []#np.zeros((N_dtp, self.N, self.M-1), dtype=complex)
        self.E_evol = []#np.zeros(N_dtp, dtype=complex)

        # Semaphors are initialised here to null values for safety
        self.semaphor_t_space = np.array([0.0])
        self.semaphor_simulation_start_time = 0.0
        self.semaphor_next_flag_t_i = 1

        # Sampling configuration initialised
        self.sampling_method = "no sampling"


    def save_recent_data(self):

        print("Recording of recent data onto local memory unit initialized.")
        output_filename = "BH_" + str(self.ID)
        config_filename = "BH_" + str(self.ID) + "_config"
        sample_filename = "BH_" + str(self.ID) + "_config_sampling"

        print("  Writing config info into outputs/" + config_filename + ".txt...", end='', flush=True)
        config_file = open("outputs/" + config_filename + ".txt", "w")
        config_file.write(", ".join(str(x) for x in [self.S, self.M, self.N]) + "\n")
        config_file.write(", ".join(str(x) for x in [self.J_0, self.J_1, self.omega, self.U, self.K, self.j_zero]) + "\n")
        config_file.close()
        print(" Done!")

        print("  Writing config sampling info into outputs/" + sample_filename + ".txt...", end='', flush=True)
        sample_file = open("outputs/" + sample_filename + ".txt", "w")
        sample_file.write(self.sampling_method + "\n")
        if self.sampling_method == "gridlike":
            sample_file.write(", ".join(str(x) for x in self.z_0) + "\n")
            sample_file.write(", ".join(str(x) for x in [self.z_0_type, self.basis_distance, self.basis_spacing]) + "\n")
        elif self.sampling_method == "gaussian":
            sample_file.write(str(self.sampling_width) + "\n")
            # We just save the entire basis, since sampling here is stochastic
            for basis_vector in self.basis:
                sample_file.write(", ".join(str(x) for x in basis_vector.z) + "\n")
        sample_file.close()
        print(" Done!")


        print("  Writing output table into outputs/" + output_filename + ".csv...", end='', flush=True)

        output_file = open("outputs/" + output_filename + ".csv", "w")
        output_writer = csv.writer(output_file)

        N = len(self.basis)

        # number of string lengths (max int is 10^fill -1)
        N_fill = 4
        M_fill = 2

        # header
        header_row = ["t"]
        for n in range(N):
            header_row.append("A_" + str(n).zfill(N_fill))
        for m in range(self.M - 1):
            for n in range(N): #TODO shouldn't this be the other way around?
                header_row.append("z_" + str(n).zfill(N_fill) + "_" + str(m).zfill(M_fill))
        header_row.append("E")
        output_writer.writerow(header_row)

        output_writer.writerows(self.output_table)

        output_file.close()
        print(" Done!")

    def load_recent_data(self):
        # Loads configuration info and simulation evolution based on the provided ID
        print("Initializing simulation object from local memory unit initialized.")
        simulation_filename = "BH_" + str(self.ID)
        config_filename = "BH_" + str(self.ID) + "_config"
        sample_filename = "BH_" + str(self.ID) + "_config_sampling"

        # Reading config
        config_file = open("outputs/" + config_filename + ".txt", 'r')
        config_lines = [line.rstrip('\n') for line in config_file]

        first_line_list = config_lines[0].split(", ")
        self.S = int(first_line_list[0])
        self.M = int(first_line_list[1])
        self.N = int(first_line_list[2])

        second_line_list = config_lines[1].split(", ")
        self.J_0    = float(second_line_list[0])
        self.J_1    = float(second_line_list[1])
        self.omega  = float(second_line_list[2])
        self.U      = float(second_line_list[3])
        self.K      = float(second_line_list[4])
        self.j_zero = float(second_line_list[5])

        print("A driven Bose-Hubbard model with %i bosons over %i modes has been initialized." % (self.S, self.M))
        print(f"    | J(t) = {self.J_0} + {self.J_1} . cos({self.omega:.2f}t)   (hopping interaction)")
        print(f"    | U = {self.U}                       (on-site interaction)")
        print(f"    | K = {self.K}, j_0 = {self.j_zero}                (harmonic trapping potential)")

        config_file.close()

        # Reading sampling config

        sample_file = open("outputs/" + sample_filename + ".txt", 'r')
        sample_lines = [line.rstrip('\n') for line in sample_file]

        self.sampling_method = sample_lines[0]
        if self.sampling_method == "gridlike":
            print("Gridlike basis vector sample loading...")

            z_0_list = sample_lines[1].split(", ")
            z_0 = np.array([complex(x) for x in z_0_list], dtype=complex)
            gridlike_param_list = sample_lines[2].split(", ")
            z_0_type = gridlike_param_list[0]
            basis_distance = int(gridlike_param_list[1])
            basis_spacing  = float(gridlike_param_list[2])
            self.sample_gridlike(basis_distance, z_0, basis_spacing, z_0_type)

        if self.sampling_method == "gaussian":
            print("Normally distributed basis vector sample loading...", end='', flush=True)

            self.basis = []
            self.sampling_width = float(sample_lines[1])
            for basis_vec_line in sample_lines[2:]:
                self.basis.append(CS(self.S, self.M, np.array([complex(x) for x in basis_vec_line.split(", ")], dtype=complex)))
            # TODO z_0 should be specifiable in iterate()
            self.z_0 = self.basis[0].z
            self.z_0_type = "aguiar"
            self.N = len(self.basis)
            print(f" Done! Basis of size N = {self.N} initialized.")


        print("Reading recent iterated evolution...", end='', flush=True)

        simulation_file = open("outputs/" + simulation_filename + ".csv", newline='')
        simulation_reader = csv.reader(simulation_file, delimiter=',', quotechar='"')

        simulation_rows = list(simulation_reader)

        header_row = simulation_rows[0]

        N_dtp = len(simulation_rows) - 1
        self.output_table = []#np.zeros((N_dtp, 2 + self.N * self.M), dtype=complex) # here we'll store all the results smashed together
        self.t_space = []#np.zeros(N_dtp)
        self.A_evol = []#np.zeros((N_dtp, self.N), dtype=complex)
        self.basis_evol = []#np.zeros((N_dtp, self.N, self.M-1), dtype=complex)
        self.E_evol = []#np.zeros(N_dtp, dtype=complex)

        for i in range(N_dtp):
            # we append the empty ndarrays
            self.output_table.append([])
            self.A_evol.append(np.zeros(self.N, dtype = complex))
            self.basis_evol.append(np.zeros((self.N, self.M - 1), dtype = complex))

            self.t_space.append(float(simulation_rows[i+1][0]))
            self.output_table[i].append(float(simulation_rows[i+1][0]))
            for n in range(self.N):
                self.A_evol[i][n] = complex(simulation_rows[i+1][1+n])
                self.output_table[i].append(complex(simulation_rows[i+1][1+n]))

            for m in range(self.M-1):
                for n in range(self.N):
                    self.basis_evol[i][n][m] = complex(simulation_rows[i+1][1 + self.N + (self.M - 1) * n + m])
                    self.output_table[i].append(complex(simulation_rows[i+1][1 + self.N + (self.M - 1) * n + m])) # TODO is this the correct format?
            self.E_evol.append(float(simulation_rows[i+1][1 + self.M * self.N]))
            self.output_table[i].append(float(simulation_rows[i+1][1 + self.M * self.N]))

        simulation_file.close()
        print(" Done! " + str(N_dtp) + " datapoints loaded.")

    # ---------------------------------------------------------
    # ------------------- Physical methods --------------------
    # ---------------------------------------------------------

    def set_global_parameters(self, S, M, J_0, J_1, omega, U, K, j_zero):
        # Fock parameters
        self.S = S
        self.M = M
        # Hamiltonian parameters
        self.J_0 = J_0
        self.J_1 = J_1
        self.omega = omega
        self.U = U
        self.K = K
        self.j_zero = j_zero

        print("A driven Bose-Hubbard model with %i bosons over %i modes has been initialized." % (self.S, self.M))
        print(f"    | J(t) = {self.J_0} + {self.J_1} . cos({self.omega:.2f}t)   (hopping interaction)")
        print(f"    | U = {self.U}                       (on-site interaction)")
        print(f"    | K = {self.K}, j_0 = {self.j_zero}                (harmonic trapping potential)")




    # ------------------- Sampling methods --------------------

    def maximal_amplitude_from_pure_z(self, z_0, z_0_type):
        # Finds z_1 such that if | z_0 } is decomposed as A_i(z) | z }, A_i will be maximal at z_1

        """
        def amplitude_gradient(z):
            # z = [x_1, x_2 ... x_(M - 1), y_1, y_2 ... y_(M-1)] where z_i = x_i + j.y_i
            result = np.zeros(2 * (self.M - 1))
            x_dot_x = np.sum(z[:self.M - 1] * z[:self.M - 1])
            y_dot_y = np.sum(z[self.M - 1:] * z[self.M - 1:])
            x_dot_x_0 = np.sum(z[:self.M - 1] * z_0_std[:self.M - 1])
            x_dot_y_0 = np.sum(z[:self.M - 1] * z_0_std[self.M - 1:])
            y_dot_x_0 = np.sum(z[self.M - 1:] * z_0_std[:self.M - 1])
            y_dot_y_0 = np.sum(z[self.M - 1:] * z_0_std[self.M - 1:])

            A = (1+x_dot_x_0+y_dot_y_0)
            B = (x_dot_y_0 - y_dot_x_0)
            C = (1 + x_dot_x + y_dot_y)

            for m in range(self.M - 1):
                result[m             ] = A * z_0_std[m             ] + B * z_0_std[self.M - 1 + m] - 2 * self.M * (A * A + B * B) * z[m             ] / C
                result[self.M - 1 + m] = A * z_0_std[self.M - 1 + m] - B * z_0_std[m             ] - 2 * self.M * (A * A + B * B) * z[self.M - 1 + m] / C
            return(result)

        initial_guess = z_0_std.copy()

        eqn_root = sp.optimize.root(amplitude_gradient, x0 = initial_guess)

        sanitised_result = np.zeros(self.M - 1, dtype=complex)
        for m in range(self.M - 1):
            sanitised_result = eqn_root.x[m] + eqn_root.x[self.M - 1 + m] + 1j
        return(sanitised_result)"""

        def neg_amp_aguiar(z):
            x_dot_x = np.sum(z[:self.M - 1] * z[:self.M - 1])
            y_dot_y = np.sum(z[self.M - 1:] * z[self.M - 1:])
            x_dot_x_0 = np.sum(z[:self.M - 1] * z_0_std[:self.M - 1])
            x_dot_y_0 = np.sum(z[:self.M - 1] * z_0_std[self.M - 1:])
            y_dot_x_0 = np.sum(z[self.M - 1:] * z_0_std[:self.M - 1])
            y_dot_y_0 = np.sum(z[self.M - 1:] * z_0_std[self.M - 1:])

            A = (1+x_dot_x_0+y_dot_y_0)
            B = (x_dot_y_0 - y_dot_x_0)
            C = (1 + x_dot_x + y_dot_y)

            return(-(A * A + B * B) / np.power(C, 2 * self.M))

        def neg_amp_grossmann(z):
            x_dot_x = np.sum(z[:self.M - 1] * z[:self.M - 1])
            y_dot_y = np.sum(z[self.M - 1:] * z[self.M - 1:])
            x_dot_x_0 = np.sum(z[:self.M - 1] * x_0)
            x_dot_y_0 = np.sum(z[:self.M - 1] * y_0)
            y_dot_x_0 = np.sum(z[self.M - 1:] * x_0)
            y_dot_y_0 = np.sum(z[self.M - 1:] * y_0)

            A = (1+x_dot_x_0+y_dot_y_0)
            B = (x_dot_y_0 - y_dot_x_0)
            C = (1 + x_dot_x + y_dot_y)

            return(-(A * A + B * B) / np.power(C, 2 * (1 + self.M / self.S)))

        z_0_std = np.zeros(2 * (self.M - 1))

        if z_0_type == "aguiar":
            for m in range(self.M - 1):
                z_0_std[m] = z_0[m].real
                z_0_std[self.M - 1 + m] = z_0[m].imag
            solution = sp.optimize.minimize(neg_amp_aguiar, x0 = z_0_std)
        if z_0_type == "grossmann":
            x_0 = np.zeros(self.M - 1)
            y_0 = np.zeros(self.M - 1)
            for i in range(self.M - 1):
                x_0 = z_0[i].real
                y_0 = z_0[i].imag
            p = z_0[self.M - 1].real
            q = z_0[self.M - 1].imag
            solution = sp.optimize.minimize(neg_amp_grossmann, x0 = np.zeros(2 * (self.M - 1)))

        sanitised_result = np.zeros(self.M - 1, dtype=complex)
        for m in range(self.M - 1):
            sanitised_result[m] = solution.x[m] + solution.x[self.M - 1 + m] * 1j
        return(sanitised_result)



    def sample_gridlike(self, max_rect_dist, z_0 = np.array([]), beta = np.sqrt(np.pi), z_0_type = "aguiar"):

        # This is the approach of a 2(M-1)-dimensional complex grid with spacing beta centered around z_0,
        # which ensures that we can locally approximate identity integrals with riemann sums with measure beta^(2M-2)
        self.sampling_method = "gridlike"

        self.basis = []
        self.beta = beta

        print(f"Sampling the neighbourhood of z_0 = {z_0} in a gridlike fashion...")#, end='', flush=True)

        # if z_0_type == "grossmann", we take it as an array of length M which represents a Grossmann coherent state
        if len(z_0) == 0:
            z_0 = np.zeros(self.M-1, dtype=complex)
            z_max_A = np.zeros(self.M-1, dtype=complex)
        elif z_0_type in ["aguiar", "grossmann"]:
            # Here we calculate the centre of the sampling
            z_max_A = self.maximal_amplitude_from_pure_z(z_0, z_0_type)
        else:
            print("  Type of initial basis vector unrecognised. Aborting.")
            return(-1)


        print("  Decomposition amplitude maximal at z =", z_max_A)

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
                    cur_z = z_max_A.copy()
                    for j in range(self.M - 1):
                        cur_z[j] += deviation_copy[j] * beta + 1j * deviation_copy[j + self.M - 1] * beta
                    self.basis.append(CS(self.S, self.M, cur_z))

        self.N = len(self.basis)
        self.z_0 = z_0 #CS(self.S, self.M, z_0)
        self.z_0_type = z_0_type
        self.basis_grid_centre = z_max_A
        self.basis_distance = max_rect_dist
        self.basis_spacing = beta

        print(" Done!")

        #TODO calculate <z_j | z_i> and their reductions here

        print(f"  A sample of N = {self.N} basis vectors around z_max_A = {z_max_A} with rectangular radius of {max_rect_dist} and spacing of {beta:.2f} has been initialized.")

    def sample_gaussian(self, z_0 = np.array([]), width = 1.0, conditioning_limit = -1, N_max = 50, max_saturation_steps = 50):

        # z_0 must be of the Aguiar variety
        self.sampling_method = "gaussian"
        self.sampling_width = width

        if len(z_0) == 0:
            z_0 = np.zeros(self.M-1, dtype=complex)

        new_basis = [z_0]

        print(f"Sampling the neighbourhood of z_0 = {z_0} with a normal distribution of width {width}...")
        steps_since_last_addition = 0

        # Complex positive-definite Hermitian matrix with positive real eigenvalues
        overlap_matrix = [[np.power(1 + np.sum(z_0.real * z_0.real + z_0.imag * z_0.imag), self.S)]]

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
                    cur_overlap = np.power(1 + np.sum(np.conjugate(candidate_basis[a]) * candidate_basis_vector), self.S)
                    candidate_overlap_matrix[a].append(cur_overlap)
                    candidate_overlap_matrix[-1][a] = np.conjugate(cur_overlap)
                candidate_overlap_matrix[-1][-1] = np.power(1 + np.sum(candidate_basis_vector.real * candidate_basis_vector.real + candidate_basis_vector.imag * candidate_basis_vector.imag), self.S)

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

        self.basis = []
        for basis_vec in new_basis:
            self.basis.append(CS(self.S, self.M, basis_vec))

        self.N = len(self.basis)
        self.z_0 = z_0
        self.z_0_type = "aguiar"
        print(f"  A sample of N = {self.N} basis vectors around z_0 = {z_0} drawn from a normal distribution of width {self.sampling_width} has been initialized.")






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
        return(H.real)

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

    # ---------------- Standardised formalism -----------------

    def standardise_dynamic_variables(self, cur_A, cur_basis):
        # y is the same format as R, so [A_1, A_2 ... A_N, z_1,1, z_2,1 ... z_N,1 ... ]
        # cur_basis can either be a list of CS() or a list of lists [n][m]
        N = len(cur_A)
        y = np.zeros(N * self.M, dtype = complex)
        for n in range(N):
            y[n] = cur_A[n]
            for m in range(self.M-1):
                if type(cur_basis[n]) == CS:
                    y[N + N * m + n] = cur_basis[n].z[m]
                else:
                    y[N + N * m + n] = cur_basis[n][m]
        return(y)

    def calculate_overlap_matrices(self, y, max_reduction = 3):
        # we note that z_n,m = y[N + N * m + n]
        # therefore z_n as a list is = y[N + n : N + n + N * M : N], where n goes from 0 to N-1 inclusive

        X = np.zeros((max_reduction + 1, self.N, self.N), dtype=complex) # X[r][i][j] = { z_i^(r) | z_j^(r) }

        # We optimize the number of operations by noting that { z_i^(r) | z_j^(r) } = { z_i^(r+1) | z_j^(r+1) } * (1+z_i*.z_j)

        for i in range(self.N):
            for j in range(self.N):
                base_inner_product = 1 + np.sum(np.conjugate(y[self.N + i : self.N + i + self.N * self.M : self.N]) * y[self.N + j : self.N + j + self.N * self.M : self.N])
                #print(base_inner_product)

                X[max_reduction][i][j] = np.power(base_inner_product, self.S - max_reduction)
                for delta_r in range(max_reduction):
                    X[max_reduction - (delta_r + 1)][i][j] = X[max_reduction - delta_r][i][j] * base_inner_product
        return(X)

    def y_dot(self, t, y, reg_timescale = -1):
        # y is the same format as R, so [A_1, A_2 ... A_N, z_1,1, z_2,1 ... z_N,1 ... z_1,(M-1) ... z_N,(M-1)]

        # We also create a standardised y which includes z_n,M = 1
        # we note that z_n,m = y_std[N + N * m + n]

        y_std = np.ones(self.N * (self.M + 1), dtype = complex)
        y_std[:self.N * self.M] = y

        # we want to read the semaphor right here in this function

        X = self.calculate_overlap_matrices(y, 3)

        R = np.zeros(self.M * self.N, dtype=complex)
        m_Theta = np.zeros((self.M*self.N, self.M*self.N), dtype=complex)

        # First we calculate R
        for k in range(self.N):
            for j in range(self.N):
                # First, we fill R_1
                sum1 = 0.0
                for i in range(self.M-1):
                    sum1 += np.conjugate(y_std[self.N + self.N * i + k]) * y_std[self.N + self.N * (i+1) + j] + np.conjugate(y_std[self.N + self.N * (i+1) + k]) * y_std[self.N + self.N * i + j]
                #sum1 *= (- self.J(t) * self.S) * X[1][k][j]

                sum2 = 0.0
                for i in range(self.M):
                    sum2 += np.conjugate(y_std[self.N + self.N * i + k]) * np.conjugate(y_std[self.N + self.N * i + k]) * y_std[self.N + self.N * i + j] * y_std[self.N + self.N * i + j]
                #sum2 *= (self.U * self.S * (self.S - 1) / 2) * X[2][k][j]

                sum3 = 0.0
                for i in range(self.M):
                    sum3 += (1 + i - self.j_zero) * (1 + i - self.j_zero) * (np.conjugate(y_std[self.N + self.N * i + k]) * y_std[self.N + self.N * i + j])
                #sum3 *= (self.S * self.K / 2) * X[1][k][j]

                R[k] += y_std[j] * ((- self.J(t) * self.S) * X[1][k][j] * sum1 + (self.U * self.S * (self.S - 1) / 2) * X[2][k][j] * sum2 + (self.S * self.K / 2) * X[1][k][j] * sum3)

                # Then, we fill in R_2
                for m in range(self.M-1):
                    if m == 0:
                        term1 = - self.J(t) * self.S * y_std[self.N + self.N * (m+1) + j] * X[1][k][j]
                    else:
                        term1 = - self.J(t) * self.S * (y_std[self.N + self.N * (m+1) + j] + y_std[self.N + self.N * (m-1) + j]) * X[1][k][j]

                    term2 = sum1
                    term2 *= (- self.J(t) * self.S * (self.S - 1) * y_std[self.N + self.N * m + j] * X[2][k][j])

                    term3 = self.U * self.S * (self.S - 1) * (np.conjugate(y_std[self.N + self.N * m + k]) * y_std[self.N + self.N * m + j] * y_std[self.N + self.N * m + j]) * X[2][k][j]

                    term4 = sum2
                    term4 *= (self.U * self.S * (self.S - 1) * (self.S - 2) * y_std[self.N + self.N * m + j] / 2) * X[3][k][j]

                    term5 = (self.K / 2) * self.S * (1 + m - self.j_zero) * (1 + m - self.j_zero) * y_std[self.N + self.N * m + j] * X[1][k][j]

                    term6 = sum3
                    term6 *= (self.K * self.S * (self.S - 1) * y_std[self.N + self.N * m + j] / 2) * X[2][k][j]

                    R[self.N + m * self.N + k] += np.conjugate(y_std[k]) * y_std[j] * (term1 + term2 + term3 + term4 + term5 + term6)

        # Then, we calculate Theta
        # First, we fill in X
        for i in range(self.N):
            for j in range(self.N):
                m_Theta[i][j] = X[0][i][j]
        # Then, we fill in Y and Y^h.c.
        for a in range(self.N):
            for b in range(self.M-1):
                for d in range(self.N):
                    m_Theta[a][self.N + b * self.N + d] = self.S * np.conjugate(y_std[self.N + self.N * b + a]) * y_std[d] * X[1][a][d]
                    m_Theta[self.N + b * self.N + d][a] = np.conjugate(m_Theta[a][self.N + b * self.N + d])
        # Then, we fill in Z
        for i in range(self.M-1):
            for j in range(self.M-1):
                for a in range(self.N):
                    for b in range(self.N):
                        # first, we evaluate (F_ij)_ab
                        m_Theta[self.N + i * self.N + a][self.N + j * self.N + b] = self.S * (self.S - 1) * X[2][a][b] * np.conjugate(y_std[self.N + self.N * j + a]) * y_std[self.N + self.N * i + b]
                        if i == j:
                            m_Theta[self.N + i * self.N + a][self.N + j * self.N + b] += self.S * X[1][a][b]
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




    # ---------------- Runtime routine methods ----------------

    def update_semaphor(self, t):
        """if np.floor(t_i / (step_N-1) * 100) > progress:
            progress = int(np.floor(t_i / (step_N-1) * 100))
            ETA = time.strftime("%H:%M:%S", time.localtime( (time.time()-start_time) * 100 / progress + start_time ))"""
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

    def old_iterate(self, max_t, dt, N_dtp):

        # max_t is the terminal simulation time, dt is the timestep, N_dtp is the number of datapoints equidistant in time which are saved
        N = len(self.basis)

        # everything is in natural units (hbar=1)
        # maximum time will actually be J_0 * max_t, which will also be the units we display it in

        self.full_t_space = np.arange(0.0, self.J_0 * max_t + dt, dt)

        step_N = len(self.full_t_space)

        # If the user asks for more datapoints than there are steps, we will limit the datapoint number
        if N_dtp > step_N:
            N_dtp = step_N

        # Output data bins
        self.output_table = []#np.zeros((N_dtp, 2 + self.N * self.M), dtype=complex) # here we'll store all the results smashed together
        self.t_space = np.zeros(N_dtp)
        self.A_evol = np.zeros((N_dtp, N), dtype=complex)
        self.basis_evol = np.zeros((N_dtp, N, self.M-1), dtype=complex)
        self.E_evol = np.zeros(N_dtp, dtype=complex)

        self.N_dtp_saved = 0 # we track the amount of datapoints saved
        def record_state(t_i, cur_it_A, cur_it_basis):
            self.t_space[self.N_dtp_saved] = t_i * dt
            for i in range(N):
                self.A_evol[self.N_dtp_saved][i] = cur_it_A[i]
                for j in range(self.M-1):
                    self.basis_evol[self.N_dtp_saved][i][j] = cur_it_basis[i].std_z(j)
            self.E_evol[self.N_dtp_saved] = self.H(t_i * dt, cur_it_A, cur_it_basis)
            # we save everything to the output table
            self.output_table.append([])
            self.output_table[self.N_dtp_saved].append(t_i * dt)
            for i in range(N):
                self.output_table[self.N_dtp_saved].append(cur_it_A[i])
            for i in range(N):
                for j in range(self.M-1):
                    self.output_table[self.N_dtp_saved].append(cur_it_basis[i].std_z(j))
            self.output_table[self.N_dtp_saved].append(self.H(t_i * dt, cur_it_A, cur_it_basis))
            self.N_dtp_saved += 1


        # We initialize dynamical variables

        # Output table = [t, A_k, z_k_m, E]

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
        ETA = "???"

        print(f"Iterative simulation of the Bose-Hubbard model on a timescale of t_max = {self.J_0 * max_t}, dt = {dt} ({step_N} steps) at {time.strftime("%H:%M:%S", time.localtime( start_time))}")

        for t_i in range(1, step_N):
            cur_t = (t_i-1) * dt #this is previous cycle's time
            # RK4

            #TODO triple check the basis_copy.z += update: shouldnt the slices through k1 be spaced out? as in it_basis_copy[n].z[m] += k1[N + m * N + n]
            #k1
            it_A_copy = it_A.copy()
            it_basis_copy = []
            for i in range(N):
                it_basis_copy.append(CS(self.S, self.M, it_basis[i].z.copy()))
            self.initialize_overlap_matrices(it_basis_copy, 3)
            cur_R = self.R(cur_t, it_A_copy, it_basis_copy)
            cur_Theta = self.Theta(cur_t, it_A_copy, it_basis_copy)

            cur_Theta_inv = np.linalg.inv(cur_Theta)
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

            # Check if state should be recorded and progress updated
            if np.floor(t_i / (step_N-1) * (N_dtp-1)) > (self.N_dtp_saved - 1):
                record_state(t_i, it_A, it_basis)
            if np.floor(t_i / (step_N-1) * 100) > progress:
                progress = int(np.floor(t_i / (step_N-1) * 100))
                ETA = time.strftime("%H:%M:%S", time.localtime( (time.time()-start_time) * 100 / progress + start_time ))
            print("  " + str(progress).zfill(2) + "% done (" + str(t_i) + "/" + str(step_N-1) + "); est. time of finish: " + ETA, end='\r')

        print("  Simulation finished at " + time.strftime("%H:%M:%S", time.localtime(time.time())) + "; " + str(self.N_dtp_saved) + " datapoints saved.                  ")
        # This function outputs the following arrays:
        #    1. [t][n] = A_n(t)
        #    2. [t] = <Psi(t) | H | Psi(t)> = <E>(t), for checking whether energy is conserved
        #    3. [t][n] = sum_m |xi_nm(t)|^2 (for checking if CSs stay SU(N)-normalized during their dynamical evolution)

        #self.

        #return(t_space, A_evol, basis_evol, E_evol)

        #print("  Preparing overlap matrices...", end='', flush=True)
        #print(" Done!")


    def iterate(self, max_t, N_dtp, rtol = 1e-3, reg_timescale = -1, N_semaphor = 100):

        # max_t is the terminal simulation time, dt is the timestep, N_dtp is the number of datapoints equidistant in time which are saved

        # everything is in natural units (hbar=1)
        # maximum time will actually be J_0 * max_t, which will also be the units we display it in

        # TODO read the start time off of the loaded data so you can resume

        # Output data bins
        """self.output_table = []#np.zeros((N_dtp, 2 + self.N * self.M), dtype=complex) # here we'll store all the results smashed together
        self.t_space = np.zeros(N_dtp)
        self.A_evol = np.zeros((N_dtp, self.N), dtype=complex)
        self.basis_evol = np.zeros((N_dtp, self.N, self.M-1), dtype=complex)
        self.E_evol = np.zeros(N_dtp, dtype=complex)"""

        if len(self.t_space) == 0:
            simulation_start_time = 0.0
            # We initialize dynamical variables

            # Output table = [t, A_k, z_k_m, E]

            print("  Calculating initial decomposition coefficients...", end='', flush=True)
            # First, we find A(t=0)
            # TODO make the integral discretisation more sane (not just midle value but trapezoids?)
            it_A = np.zeros(self.N, dtype=complex)
            for i in range(self.N):
                if False and self.z_0_type == "aguiar" and self.sampling_method == "gridlike":
                    identity_prefactor = (math.factorial(self.S + self.M - 1) / math.factorial(self.S)) * np.power(self.beta * self.beta / np.pi, self.M - 1)
                    cur_overlap = np.power(1 + np.sum(np.conjugate(self.basis[i].z) * self.z_0), self.S)
                    it_A[i] = (identity_prefactor / np.power(1.0 + np.sum(np.conjugate(self.basis[i].z) * self.basis[i].z), self.M + self.S)) * cur_overlap
                else:
                    # NOTE this correctly preserves <N> for an aguiar pure initial state, but doesn't agree well with gridlike sampling!
                    X = np.zeros((self.N, self.N), dtype = complex)
                    for m in range(self.N):
                        for n in range(self.N):
                            X[m][n] = np.power(1 + np.sum(np.conjugate(self.basis[m].z) * self.basis[n].z), self.S)
                    inverse_overlap = np.linalg.inv(X)

                    for b in range(self.N):
                        if self.z_0_type == "aguiar":
                            cur_overlap = np.power(1 + np.sum(np.conjugate(self.basis[b].z) * self.z_0), self.S)
                        elif self.z_0_type == "grossmann":
                            cur_overlap = np.power(self.z_0[self.M - 1] + np.sum(np.conjugate(self.basis[b].z) * self.z_0[:self.M - 1]), self.S )
                        it_A[i] += cur_overlap * inverse_overlap[i][b]
                    #cur_overlap = np.power( (self.z_0[self.M - 1] + np.sum(np.conjugate(self.basis[i].z) * self.z_0[:self.M - 1] ) ), self.S )
                    #it_A[i] = (identity_prefactor / np.power(1.0 + np.sum(np.conjugate(self.basis[i].z) * self.basis[i].z), self.M + self.S)) * cur_overlap

            Psi_mag = 0.0
            for i in range(self.N):
                for j in range(self.N):
                    Psi_mag += np.conjugate(it_A[i]) * it_A[j] * self.basis[j].overlap(self.basis[i])
            print(f"  Done! The direct wavefunction normalization is {Psi_mag.real:.4f}")

            # Naive renormalization
            for i in range(self.N):
                it_A[i] /= np.sqrt(Psi_mag.real)

            y_0 = self.standardise_dynamic_variables(it_A, self.basis)

            # We fill out the first datapoint manually
            self.output_table = [[]]
            self.t_space = [simulation_start_time]
            self.output_table[0].append(simulation_start_time)
            self.A_evol = [it_A]
            for i in range(self.N):
                self.output_table[0].append(it_A[i])
            self.basis_evol = [np.zeros((self.N, self.M-1), dtype=complex)]
            for j in range(self.M-1):
                for i in range(self.N):
                    self.basis_evol[0][i][j] = self.basis[i].z[j]
                    self.output_table[0].append(self.basis[i].z[j])
            self.E_evol = [0.0] #TODO add H
            self.output_table[0].append(0.0)

        else:
            # By loading or previous call of the iterate routine, there is data saved already
            print(f"  We will resume the simulation from the timestamp it terminated at previously.")
            simulation_start_time = self.t_space[-1]

            y_0 = self.standardise_dynamic_variables(self.A_evol[-1], self.basis_evol[-1])

            debug_X = self.calculate_overlap_matrices(y_0, 0)

            Psi_mag = 0.0
            for i in range(self.N):
                for j in range(self.N):
                    Psi_mag += np.conjugate(y_0[i]) * y_0[j] * debug_X[0][i][j]
            print(f"  Data loaded! The direct wavefunction normalization is {Psi_mag.real:.4f}")
            print(y_0)





        # TODO sort out the semaphors
        self.semaphor_t_space = np.linspace(simulation_start_time, self.J_0 * max_t, N_semaphor + 1)
        self.semaphor_simulation_start_time = simulation_start_time
        self.semaphor_start_time = time.time()
        self.semaphor_next_flag_t_i = 1
        self.semaphor_ETA = "???"

        print(f"Iterative simulation of the Bose-Hubbard model on a timescale of t = ({simulation_start_time} - {self.J_0 * max_t}), rtol = {rtol} at {time.strftime("%H:%M:%S", time.localtime( self.semaphor_start_time))}")

        iterated_solution = sp.integrate.solve_ivp(self.y_dot, [simulation_start_time, self.J_0 * max_t], y_0, method = 'RK45', t_eval = np.linspace(simulation_start_time, self.J_0 * max_t, N_dtp+1), args = (reg_timescale,), rtol = rtol)
        print("  Simulation finished at " + time.strftime("%H:%M:%S", time.localtime(time.time())) + "; " + str(N_dtp) + " datapoints saved.                  ")

        # Saving datapoints except for the first one

        for t_i in range(1, N_dtp+1):
            self.output_table.append([iterated_solution.t[t_i]])
            self.t_space.append(iterated_solution.t[t_i])
            self.A_evol.append(np.zeros(self.N, dtype=complex))
            for i in range(self.N):
                self.A_evol[-1][i] = iterated_solution.y[i][t_i]
                self.output_table[-1].append(iterated_solution.y[i][t_i])
            self.basis_evol.append(np.zeros((self.N, self.M-1), dtype=complex))
            for j in range(self.M-1):
                for i in range(self.N):
                    self.basis_evol[-1][i][j] = iterated_solution.y[self.N + self.N * j + i][t_i]
                    self.output_table[-1].append(iterated_solution.y[self.N + self.N * j + i][t_i])
            self.E_evol.append(0.0) # TODO calculate from standardised y #self.H(t_i * dt, cur_it_A, cur_it_basis)
            self.output_table[-1].append(0.0)

        #print(self.output_table)

    def iterate_rk4(self, max_t, dt, N_dtp):
        # max_t is the terminal simulation time, dt is the timestep, N_dtp is the number of datapoints equidistant in time which are saved
        N = len(self.basis)

        # everything is in natural units (hbar=1)
        # maximum time will actually be J_0 * max_t, which will also be the units we display it in

        # TODO read the start time off of the loaded data so you can resume

        # Output data bins
        self.output_table = []#np.zeros((N_dtp, 2 + self.N * self.M), dtype=complex) # here we'll store all the results smashed together
        self.t_space = np.zeros(N_dtp)
        self.A_evol = np.zeros((N_dtp, N), dtype=complex)
        self.basis_evol = np.zeros((N_dtp, N, self.M-1), dtype=complex)
        self.E_evol = np.zeros(N_dtp, dtype=complex)


        # We initialize dynamical variables

        # Output table = [t, A_k, z_k_m, E]

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

        y_0 = self.standardise_dynamic_variables(it_A, self.basis)

        # TODO sort out the semaphors
        start_time = time.time()
        progress = 0
        ETA = "???"

        print(f"Iterative simulation of the Bose-Hubbard model on a timescale of t_max = {self.J_0 * max_t}, dt = {dt} at {time.strftime("%H:%M:%S", time.localtime( start_time))}")

        iterated_solution = explicit_runge_kutta(self.y_dot, [0, self.J_0 * max_t, dt], y_0, rk4_tableau, N_dtp)
        #iterated_solution = sp.integrate.solve_ivp(self.y_dot, [0, self.J_0 * max_t], y_0, method = 'RK45', t_eval = np.linspace(0, self.J_0 * max_t, N_dtp), rtol = rtol)

        # Saving datapoints

        for t_i in range(N_dtp):
            self.t_space[t_i] = iterated_solution.t[t_i]
            for i in range(self.N):
                self.A_evol[t_i][i] = iterated_solution.values[t_i][i]
                for j in range(self.M-1):
                    self.basis_evol[t_i][i][j] = iterated_solution.values[t_i][self.N + self.N * j + i]
            self.E_evol[t_i] = 0.0 # TODO calculate from standardised y #self.H(t_i * dt, cur_it_A, cur_it_basis)
            # we save everything to the output table
            self.output_table.append([])
            self.output_table[t_i].append(iterated_solution.t[t_i])
            for i in range(self.N):
                self.output_table[t_i].append(iterated_solution.values[t_i][i])
            for i in range(self.N):
                for j in range(self.M-1):
                    self.output_table[t_i].append(iterated_solution.values[t_i][self.N + self.N * j + i])
            self.output_table[t_i].append(0.0) #TODO H

        print("  Simulation finished at " + time.strftime("%H:%M:%S", time.localtime(time.time())) + "; " + str(N_dtp) + " datapoints saved.                  ")


    def plot_recent_data(self, graph_list = ["expected_mode_occupancy", "initial_basis_heatmap"], save_graph=True):

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
            if graph_list[i] == 'initial_basis_heatmap':
                # NOTE this is only for gridlike samples!!
                if self.sampling_method != "gridlike":
                    print("  ERROR: Attempting to plot 'initial_basis_heatmap' when sample is not gridlike. Did you mean to plot 'initial_decomposition' instead?")
                    continue

                print("  Plotting the heatmap of initial basis decomposition coefficient magnitudes...", end='', flush=True)
                if self.M != 2:
                    print("    This graph can only be plotted for a two-mode system.")
                    continue

                plt.title("Initial basis decomposition coefficient magnitudes")
                x_vals = []
                y_vals = []

                for basis_element in self.basis:
                    x_vals.append(round(basis_element.z[0].real - self.basis_grid_centre[0].real, 3))
                    y_vals.append(round(basis_element.z[0].imag - self.basis_grid_centre[0].imag, 3))
                A_vals = self.A_evol[0]

                df = pd.DataFrame({"Y" : y_vals, "X" : x_vals, "A" : np.sqrt(A_vals.real * A_vals.real + A_vals.imag * A_vals.imag)})
                table = df.pivot(index='Y', columns='X', values='A')
                ax = sns.heatmap(table)
                ax.collections[0].colorbar.set_label("$|A_k|$", rotation=0, labelpad=15)
                ax.invert_yaxis()
                plt.xlabel("$\\Re\\left(z_n-z_0\\right)$")
                plt.ylabel("$\\Im\\left(z_n-z_0\\right)$")
                plt.gca().set_aspect("equal")
                include_legend = False
                print(" Done!")

            elif graph_list[i] == "initial_decomposition":
                if self.M != 2:
                    print("  ERROR: Attempting to plot 'initial_decomposition' with unsuitable mode number.")
                    continue
                x_vals = np.zeros(self.N)
                y_vals = np.zeros(self.N)
                s_vals = np.zeros(self.N)

                for n in range(self.N):
                    x_vals[n] = self.basis_evol[0][n][0].real
                    y_vals[n] = self.basis_evol[0][n][0].imag
                    s_vals[n] = np.absolute(self.A_evol[0][n])

                plt.xlabel("$\\Re\\left(z_n\\right)$")
                plt.ylabel("$\\Im\\left(z_n\\right)$")

                plt.scatter(x_vals, y_vals, 5 + np.power(s_vals, 1/10) * 450)
                include_legend = False



            elif graph_list[i] == 'expected_mode_occupancy':
                print("  Plotting the time evolution of expected mode occupancy...", end='', flush=True)
                plt.title("Expected mode occupancy")
                plt.xlabel("t")
                psi_mag = np.zeros(len(self.t_space))

                avg_n = []
                for m in range(self.M):
                    avg_n.append(np.zeros(len(self.t_space)))

                for t_i in range(len(self.t_space)):
                    for a in range(self.N):
                        for b in range(self.N):
                            cur_X_1 = CS(self.S, self.M, self.basis_evol[t_i][b]).overlap(CS(self.S, self.M, self.basis_evol[t_i][a]), reduction = 1)
                            psi_mag[t_i] += (np.conjugate(self.A_evol[t_i][a]) * self.A_evol[t_i][b] * CS(self.S, self.M, self.basis_evol[t_i][b]).overlap(CS(self.S, self.M, self.basis_evol[t_i][a]))).real
                            for m in range(self.M-1):
                                avg_n[m][t_i] += (np.conjugate(self.A_evol[t_i][a]) * self.A_evol[t_i][b] * np.conjugate(self.basis_evol[t_i][a][m]) * self.basis_evol[t_i][b][m] * cur_X_1).real
                            avg_n[self.M-1][t_i] += (np.conjugate(self.A_evol[t_i][a]) * self.A_evol[t_i][b] * cur_X_1).real

                # We insert horizontal lines indicating occupancies calculated from z_0
                initial_occupancy = np.zeros(self.M)
                if self.z_0_type == "aguiar":

                    for m in range(self.M - 1):
                        initial_occupancy[m] = (self.z_0[m].real * self.z_0[m].real + self.z_0[m].imag * self.z_0[m].imag) / (1 + np.sum(np.conjugate(self.z_0) * self.z_0).real)
                    initial_occupancy[self.M - 1] = 1 / (1 + np.sum(np.conjugate(self.z_0) * self.z_0).real)
                elif self.z_0_type == "grossmann":
                    z_0_reduced_overlap = np.power(np.sum(np.conjugate(self.z_0) * self.z_0), self.S - 1).real
                    for m in range(self.M):
                        initial_occupancy[m] = (self.z_0[m].real * self.z_0[m].real + self.z_0[m].imag * self.z_0[m].imag) * z_0_reduced_overlap
                for m in range(self.M):
                    plt.axhline(y = initial_occupancy[m], linestyle = "dotted", label = "init. $\\langle N_" + str(m+1) + " \\rangle/S$")



                # if two-mode, we insert the algebraic solution
                if self.M == 2:
                    cur_c_0 = np.zeros(self.S + 1, dtype = complex)
                    for i in range(self.S + 1):
                        for a in range(self.N):
                            cur_c_0[i] += self.A_evol[0][a] * np.power(self.basis_evol[0][a], i)
                        cur_c_0[i] *= np.sqrt( math.factorial(self.S) / (math.factorial(i) * math.factorial(self.S-i)) )

                    N_space_t_space = np.linspace(self.t_space[0], self.t_space[-1], 200)
                    res_N_space = two_mode_solution(N_space_t_space, self.S, self.U, self.J, cur_c_0)

                    plt.plot(N_space_t_space, res_N_space, linestyle = "dashed", label = "theor. $\\langle N_1 \\rangle/S$")
                    plt.plot(N_space_t_space, res_N_space * (-1) + 1, linestyle = "dashed", label = "theor. $\\langle N_2 \\rangle/S$")



                plt.plot(self.t_space, psi_mag, label="$\\langle \\Psi | \\Psi \\rangle$")
                for m in range(self.M):
                    plt.plot(self.t_space, avg_n[m], label="$\\langle N_" + str(m+1) + " \\rangle/S$")
                print(" Done!")

            if x_left != -1:
                plt.xlim(x_left, x_right)
            if y_left != -1:
                plt.ylim(y_left, y_right)

            if include_legend:
                plt.legend()

        plt.tight_layout()
        if save_graph:
            plt.savefig("outputs/BH_" + str(self.ID) + "_graph_output.png")
        plt.show()




