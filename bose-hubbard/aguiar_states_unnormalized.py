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

class BH():
    #driven Bose-Hubbard model

    # ---------------------------------------------------------
    # -------- Initializers, descriptors, destructors ---------
    # ---------------------------------------------------------

    def __init__(self, ID):
        # Identification
        self.ID = ID

        print("---------------------------- " + str(ID) + " -----------------------------")


    def save_recent_data(self):

        print("Recording of recent data onto local memory unit initialized.")
        output_filename = "BH_" + str(self.ID)
        config_filename = "BH_" + str(self.ID) + "_config"

        print("  Writing config info into outputs/" + config_filename + ".txt...", end='', flush=True)
        config_file = open("outputs/" + config_filename + ".txt", "w")
        config_file.write(", ".join(str(x) for x in [self.S, self.M, self.N]) + "\n")
        config_file.write(", ".join(str(x) for x in [self.J_0, self.J_1, self.omega, self.U, self.K, self.j_zero]) + "\n")
        config_file.write(", ".join(str(x) for x in self.z_0.z) + "\n")
        config_file.write(", ".join(str(x) for x in [self.basis_distance, self.basis_spacing]) + "\n")
        config_file.close()
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
        for n in range(N):
            for m in range(self.M - 1):
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

        # Reading config
        init_config_file = open("outputs/" + config_filename + ".txt", 'r')
        config_lines = [line.rstrip('\n') for line in init_config_file]

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

        third_line_list = config_lines[2].split(", ")
        z_0 = np.array([complex(x) for x in third_line_list], dtype=complex)
        fourth_line_list = config_lines[3].split(", ")
        basis_distance = int(fourth_line_list[0])
        basis_spacing  = float(fourth_line_list[1])
        self.sample_gridlike(basis_distance, z_0, basis_spacing)

        init_config_file.close()

        print("Reading recent iterated evolution...", end='', flush=True)

        simulation_file = open("outputs/" + simulation_filename + ".csv", newline='')
        simulation_reader = csv.reader(simulation_file, delimiter=',', quotechar='"')

        simulation_rows = list(simulation_reader)

        header_row = simulation_rows[0]

        N_dtp = len(simulation_rows) - 1
        self.output_table = np.zeros((N_dtp, 2 + self.N * self.M), dtype=complex) # here we'll store all the results smashed together
        self.t_space = np.zeros(N_dtp)
        self.A_evol = np.zeros((N_dtp, self.N), dtype=complex)
        self.basis_evol = np.zeros((N_dtp, self.N, self.M-1), dtype=complex)
        self.E_evol = np.zeros(N_dtp, dtype=complex)

        for i in range(N_dtp):
            self.t_space[i] = float(simulation_rows[i+1][0])
            self.output_table[i][0] = float(simulation_rows[i+1][0])
            for n in range(self.N):
                self.A_evol[i][n] = complex(simulation_rows[i+1][1+n])
                self.output_table[i][1+n] = complex(simulation_rows[i+1][1+n])
                for m in range(self.M-1):
                    self.basis_evol[i][n][m] = complex(simulation_rows[i+1][1 + self.N + (self.M - 1) * n + m])
                    self.output_table[i][1 + self.N + (self.M - 1) * n + m] = complex(simulation_rows[i+1][1 + self.N + (self.M - 1) * n + m])
            self.E_evol[i] = float(simulation_rows[i+1][1 + self.M * self.N])
            self.output_table[i][1 + self.M * self.N] = float(simulation_rows[i+1][1 + self.M * self.N])

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

    def sample_gridlike(self, max_rect_dist, z_0 = np.array([]), beta = np.sqrt(np.pi)):

        # This is the approach of a 2(M-1)-dimensional complex grid with spacing beta centered around z_0,
        # which ensures that we can locally approximate identity integrals with riemann sums with measure beta^(2M-2)
        if len(z_0) == 0:
            z_0 = np.zeros(self.M-1, dtype=complex)

        self.basis = []
        self.beta = beta

        print("Sampling the neighbourhood of z_0 in a gridlike fashion...", end='', flush=True)

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
        self.basis_distance = max_rect_dist
        self.basis_spacing = beta

        print(" Done!")

        #TODO calculate <z_j | z_i> and their reductions here

        print(f"  A sample of N = {self.N} basis vectors around z_0 = {z_0} with rectangular radius of {max_rect_dist} and spacing of {beta:.2f} has been initialized.")

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


    # ---------------- Runtime routine methods ----------------

    def iterate(self, max_t, dt, N_dtp):

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
                print("  Plotting the heatmap of initial basis decomposition coefficient magnitudes...", end='', flush=True)
                plt.title("Initial basis decomposition coefficient magnitudes")
                x_vals = []
                y_vals = []

                for basis_element in self.basis:
                    x_vals.append(round(basis_element.z[0].real - self.z_0.z[0].real, 3))
                    y_vals.append(round(basis_element.z[0].imag - self.z_0.z[0].imag, 3))
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




