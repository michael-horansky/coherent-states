import math
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from pathlib import Path
import inspect # callable handling
from functools import partial
from copy import deepcopy
import time

from class_Semaphor import Semaphor
from class_DSM import DSM
import functions

import csv

import warnings

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

def dtstr(seconds, max_depth = 2):
    # Dynamically chooses the right format
    # max_depth is the number of different measurements (e.g. max_depth = 2: "2 days 5 hours")
    if seconds >= 60 * 60 * 24:
        # Days
        if max_depth == 1:
            return(f"{int(np.round(seconds / (60 * 60 * 24)))} days")
        remainder = seconds % (60 * 60 * 24)
        days = int((seconds - remainder) / (60 * 60 * 24))
        return(f"{days} days {dtstr(remainder, max_depth - 1)}")
    if seconds >= 60 * 60:
        # Hours
        if max_depth == 1:
            return(f"{int(np.round(seconds / (60 * 60)))} hours")
        remainder = seconds % (60 * 60)
        hours = int((seconds - remainder) / (60 * 60))
        return(f"{hours} hours {dtstr(remainder, max_depth - 1)}")
    if seconds >= 60:
        # Minutes
        if max_depth == 1:
            return(f"{int(np.round(seconds / 60))} min")
        remainder = seconds % (60)
        minutes = int((seconds - remainder) / (60))
        return(f"{minutes} min {dtstr(remainder, max_depth - 1)}")
    if seconds >= 1:
        # Seconds
        if max_depth == 1:
            return(f"{int(np.round(seconds))} sec")
        remainder = seconds % (1)
        secs = int((seconds - remainder))
        return(f"{secs} sec {dtstr(remainder, max_depth - 1)}")
    # Milliseconds
    return(f"{int(np.round(seconds / 0.001))} ms")

def square_mag(c):
    return(c.real * c.real + c.imag * c.imag)

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

    rk_max_step = 0.1
    rk_max_norm_deviation = 1.5

    basis_sampling_methods = {
            "sample_gaussian(z_0 = np.array([]), width = 1.0, conditioning_limit = -1, N_max = 50, max_saturation_steps = 50)" : "J. Chem. Phys. 144, 094106 (2016); see Appendix"
        }

    data_nodes = {
        "system" : {"config" : "pkl", "H_A_func" : "txt", "H_B_func" : "txt", "fock_solution" : "csv"}, # The description of the physical system
        "setup" : {"basis_init" : "csv", "wavef_init" : "pkl"}, # The specification of the experimental setup
        "solution" : {"basis_evol" : "csv", "wavef_evol" : "csv"} # The solution as found by the CS method
    }

    def __init__(self, ID):
        # Identification
        self.ID = ID

        print("---------------------------- " + str(ID) + " -----------------------------")

        # Data bins are initialized

        # initial conditions. Unchanged when simulation occurs.
        self.basis = [] # This is a LIST of bases, i.e. each element is a basis = np.array((self.N, self.M - 1), dtype=complex)
        #self.wavef = [] # This is a LIST of initial decomposition coefficient arrays, one for each basis

        self.N = [] # A list of basis lengths
        self.inverse_overlap_matrix = [] # A list of overlap operators

        self.t_space = [] # Initialized for the first basis and then used for each subsequent one. np.zeros(N_dtp)
        self.wavef_evol = []# LIST of evolutions, where each element is a wavef evolution = np.zeros((N_dtp, self.N), dtype=complex)
        self.basis_evol = []# LIST of evolutions, where each element is a basis evolution = np.zeros((N_dtp, self.N, self.M-1), dtype=complex)

        self.evol_benchmarks = [] # For each b_i, this is the time in seconds it takes to evolve both the basis AND the wavefunction

        self.solution = [] # Can be found on the Fock basis for small M,S. A list of expected occupancies in time. np.array((N_dtp, self.M), dtype=float)
        self.solution_benchmark = None

        # Config
        self.basis_config = [] # Every element is a dictionary

        # Semaphor
        self.semaphor = Semaphor(time_format = "%H:%M:%S")

        # State booleans are initialized; useful when choosing savedata format
        self.is_phys_init = False # Are physical properties initialized?
        self.is_basis_init = False # Is basis initialized?
        self.is_basis_evol = False # Is basis evolved?
        self.is_wavef_init = False # Is wavefunction initialized?
        self.is_wavef_evol = False # Is wavefunction evolved?
        self.is_solved = False # Is fock-basis solution found?

        self.is_t_space_init = False

        self.M = 0
        self.S = 0

        # Typographical constants
        # number of string lengths (max int is 10^fill -1)
        self.B_fill = 2
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
        self.output_solution_filename = "fock_solution" # Solution on the full Fock basis

        # DSM
        self.disk_jockey = DSM(f"outputs/{self.ID}")
        self.disk_jockey.create_data_nodes(bosonic_su_n.data_nodes)



    ###########################################################################
    # ----------------------------- INTERNAL METHODS --------------------------
    ###########################################################################
    # Methods which the user isn't expected to invoke


    # ---------------------------------------------------------
    # ----------------- Physical descriptors ------------------
    # ---------------------------------------------------------
    # Methods used to describe the physics of the system

    # ------------------- Operator methods --------------------

    def decompose_aguiar(self, z, basis_index):
        # Returns a decomposition of z into self.basis
        if not self.is_basis_init:
            print("ERROR: You have attempted to calculate a wavefunction decomposition before sampling a basis set. You can do this by calling one of the many basis-sampling methods listed below.")
            for m, desc in bosonic_su_n.basis_sampling_methods.items():
                print(f"{m} [{desc}]")
            return(-1)

        decomposition = np.zeros(self.N[basis_index], dtype=complex)
        for i in range(self.N[basis_index]):
            for a in range(self.N[basis_index]):
                decomposition[i] += self.inverse_overlap_matrix[basis_index][i][a] * overlap(self.basis[basis_index][a], z, self.S)
        return(decomposition)

    def decompose_grossmann(self, xi, basis_index):
        # Returns a decomposition of xi into self.basis
        if not self.is_basis_init:
            print("ERROR: You have attempted to calculate a wavefunction decomposition before sampling a basis set. You can do this by calling one of the many basis-sampling methods listed below.")
            for m, desc in bosonic_su_n.basis_sampling_methods.items():
                print(f"{m} [{desc}]")
            return(-1)

        decomposition = np.zeros(self.N[basis_index], dtype=complex)
        for i in range(self.N[basis_index]):
            for a in range(self.N[basis_index]):
                cur_overlap = np.power(self.z[self.M - 1] + np.sum(np.conjugate(self.basis[basis_index][a]) * self.z[:self.M - 1]), self.S )
                decomposition[i] += cur_overlap * self.inverse_overlap_matrix[basis_index][i][a]
        return(decomposition)

    def decompose_aguiar_into_fock(self, z, fock_basis, renormalise = True):
        c = np.zeros(len(fock_basis), dtype = complex)

        if renormalise:
            normalisation_coef = np.power(1 / np.sqrt(1 + np.sum(z.real * z.real + z.imag * z.imag)), self.S)
        else:
            normalisation_coef = 1.0

        for i in range(len(fock_basis)):
            c[i] = np.sqrt( float(math.factorial(self.S)) ) * normalisation_coef
            for m in range(self.M):
                c[i] /= np.sqrt(float(math.factorial(fock_basis[i][m])))
                if m < self.M - 1:
                    c[i] *= np.power(z[m], fock_basis[i][m])
        return(c)

    def decompose_grossmann_into_fock(self, xi, fock_basis):
        c = np.zeros(len(fock_basis), dtype = complex)
        for i in range(len(fock_basis)):
            c[i] = np.sqrt( math.factorial(self.S) )
            for m in range(self.M):
                c[i] /= np.sqrt(math.factorial(fock_basis[i][m]))
                c[i] *= np.power(xi[m], fock_basis[i][m])
        return(c)

    def decompose_init_wavef_into_basis(self, basis_index):
        if self.wavef_message in ["aguiar_pure"]:
            # We also normalise here
            return(self.decompose_aguiar(self.wavef_initial_wavefunction, basis_index) / np.sqrt(square_norm(self.wavef_initial_wavefunction, self.S)))
        elif self.wavef_message in ["aguiar_disc"]:
            total_decomp = np.zeros(self.N[basis_index], dtype=complex)
            for i in range(len(self.wavef_initial_wavefunction[0])):
                total_decomp += self.wavef_initial_wavefunction[1][i] * self.decompose_aguiar(self.wavef_initial_wavefunction[0][i], basis_index)
            return(total_decomp)
        else:
            print("ERROR: Unsupported init_wavef type")

    def decompose_init_wavef_into_fock(self, fock_basis):
        if self.wavef_message in ["aguiar_pure"]:
            return(self.decompose_aguiar_into_fock(self.wavef_initial_wavefunction, fock_basis, renormalise = True))
        elif self.wavef_message in ["aguiar_disc"]:
            total_decomp = np.zeros(len(fock_basis), dtype=complex)
            for i in range(len(self.wavef_initial_wavefunction[0])):
                total_decomp += self.wavef_initial_wavefunction[1][i] * self.decompose_aguiar_into_fock(self.wavef_initial_wavefunction[0][i], fock_basis, renormalise = False)
            return(total_decomp)
        else:
            print("ERROR: Unsupported init_wavef type")





    # ----------------- Interpolation methods -----------------

    def basis_state_at_time(self, t, basis_index):
        # If basis and t_space are initialised, this will interpolate the basis state at any given time
        # and return both the state and the first time-derivative
        """if not self.is_basis_evol:
            print("ERROR: Cannot interpolate the state of the basis at a non-zero time without first having evolved the basis.")
            return(-1)"""

        if t < self.t_space[0] or t > self.t_space[-1]:
            print(f"ERROR: Cannot interpolate the state of the basis at a time outside of the interval of propagation ({self.t_space[0]}, {self.t_space[-1]}).")
            return(-1)

        for t_i in range(len(self.t_space)):
            if self.t_space[t_i] == t:
                if t_i == 0:
                    time_derivative = (self.basis_evol[basis_index][t_i+1]-self.basis_evol[basis_index][t_i]) / (self.t_space[t_i+1] - self.t_space[t_i])
                else:
                    time_derivative = (self.basis_evol[basis_index][t_i]-self.basis_evol[basis_index][t_i-1]) / (self.t_space[t_i] - self.t_space[t_i-1])
                return(self.basis_evol[basis_index][t_i], time_derivative)
            if self.t_space[t_i] > t:
                break
        # t_i is now the index of the first datapoint at a time larger than t
        time_derivative = (self.basis_evol[basis_index][t_i]-self.basis_evol[basis_index][t_i-1]) / (self.t_space[t_i] - self.t_space[t_i-1])
        q = (t-self.t_space[t_i-1]) / (self.t_space[t_i]-self.t_space[t_i-1])
        return(self.basis_evol[basis_index][t_i - 1] + (self.basis_evol[basis_index][t_i] - self.basis_evol[basis_index][t_i - 1]) * q, time_derivative)
        # TODO more precisely, we can avoid using finite differences by directly feeding the interpolated value
        # into uncoupled_basis_y_dot. But is it still fast enough? And does it make a difference?

    # ------------------ Fock-basis solvers -------------------

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
                                        H_matrix[i][j] += 0.5 * self.H_B(t, a, b, c, d) * coef
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

    def fock_solution(self, t_range = -1, N_semaphor = 100):
        # General solution to be plotted

        # If t_space not initialized, it will be initialized according to t_range = [(t_start,) t_stop, N_dtps]
        if not self.is_phys_init:
            print("ERROR: You have attempted to solve the system on the Fock basis before specifying the system hamiltonian. You can do this by calling set_hamiltonian_tensors(A, B).")
            return(-1)

        print("Solving the Hamiltonian on the Fock basis...")


        # We use the time-dependent Schrodinger Equation over the full occupancy number basis
        # d/dt | Psi > = -i H | Psi >
        # | Psi > = c_i | u_i >, where | u_i > is the i-th element of the full occpuancy number basis
        # Hence dc_j/dt = -i c_i < u_j | H | u_i > = -i c_i H_ji
        # In vector form: dc/dt = -i H . c

        def flatten_basis(M, S):
            # Creates a list of all Fock states with M modes and S particles,
            # where each element is a distinct list of M integers summing up to S

            # We fix the first mode occupancy and then find the answer for M - 1 occupancies
            if S == 0:
                return([[0] * M])
            if M == 1:
                return([[S]])
            answer = []
            for first_mode_occupancy in range(S + 1):
                remainder = flatten_basis(M - 1, S - first_mode_occupancy)
                for row in remainder:
                    answer.append([first_mode_occupancy] + row)
            return(answer)

        print("  Finding the basis...")
        fock_basis = flatten_basis(self.M, self.S)
        fock_N = len(fock_basis)

        def c_dot(t, y, semaphor_event_ID = None):
            # We find the H_ij matrix
            H_matrix = np.zeros((fock_N, fock_N), dtype=complex)
            for i in range(fock_N):
                for j in range(fock_N):
                    # Calculating H_ij

                    # First order
                    for a in range(self.M):
                        for b in range(self.M):
                            bra_occupancy = fock_basis[i].copy()
                            ket_occupancy = fock_basis[j].copy()
                            coef = 1.0

                            # a^h.c._a acts on the bra
                            if bra_occupancy[a] == 0:
                                continue
                            coef *= np.sqrt(bra_occupancy[a])
                            bra_occupancy[a] -= 1
                            # a_b acts on the ket
                            if ket_occupancy[b] == 0:
                                continue
                            coef *= np.sqrt(ket_occupancy[b])
                            ket_occupancy[b] -= 1

                            if bra_occupancy == ket_occupancy:
                                H_matrix[i][j] += self.H_A(t, a, b) * coef
                    # Second order
                    for a in range(self.M):
                        for b in range(self.M):
                            for c in range(self.M):
                                for d in range(self.M):
                                    bra_occupancy = fock_basis[i].copy()
                                    ket_occupancy = fock_basis[j].copy()
                                    coef = 1.0

                                    # a^h.c._a acts on the bra
                                    if bra_occupancy[a] == 0:
                                        continue
                                    coef *= np.sqrt(bra_occupancy[a])
                                    bra_occupancy[a] -= 1
                                    # a^h.c._b acts on the bra
                                    if bra_occupancy[b] == 0:
                                        continue
                                    coef *= np.sqrt(bra_occupancy[b])
                                    bra_occupancy[b] -= 1
                                    # a_c acts on the ket
                                    if ket_occupancy[c] == 0:
                                        continue
                                    coef *= np.sqrt(ket_occupancy[c])
                                    ket_occupancy[c] -= 1
                                    # a_d acts on the ket
                                    if ket_occupancy[d] == 0:
                                        continue
                                    coef *= np.sqrt(ket_occupancy[d])
                                    ket_occupancy[d] -= 1

                                    if bra_occupancy == ket_occupancy:
                                        H_matrix[i][j] += 0.5 * self.H_B(t, a, b, c, d) * coef
            self.semaphor.update(semaphor_event_ID, t)
            return( - 1j * H_matrix.dot(y))


        if not self.is_t_space_init:
            if len(t_range) == 2:
                self.t_space = np.linspace(0, t_range[0], t_range[1] + 1)
            elif len(t_range) == 3:
                self.t_space = np.linspace(t_range[0], t_range[1], t_range[2] + 1)
            else:
                print("  ERROR: t_space is a required argument when t_space has not been initialized, and must be a list of form [(t_start,) t_stop, N_dtps]")
                return(-1)
            self.is_t_space_init = True

        N_dtp = len(self.t_space)

        # Decomposition of initial state from self.wavef_initial_wavefunction, self.wavef_message
        # c_i(t = 0) = < u_i | z_0 }

        """if self.wavef_message in ["aguiar"]:
            # initial_wavefunction describes an Aguiar unnormalized coherent state.
            c_0 = self.decompose_aguiar_into_fock(np.array(self.wavef_initial_wavefunction), fock_basis)
        elif self.wavef_message in ["grossmann", "frank"]:
            # initial_wavefunction describes a Grossmann normalized coherent state.
            c_0 = self.decompose_grossmann_into_fock(np.array(self.wavef_initial_wavefunction), fock_basis)
        elif self.wavef_message in ["", "NONE"]:
            c_0 = self.decompose_aguiar_into_fock(self.basis[0][0], fock_basis)
        else:
            print(f"  ERROR: self.wavef_message {self.wavef_message} not recognized.")
            return(-1)"""
        c_0 = self.decompose_init_wavef_into_fock(fock_basis)

        msg = f"  Solving the Schrodinger equation discretised on the full Fock basis on a timescale of t = ({self.t_space[0]} - {self.t_space[-1]})..."
        new_sem_ID = self.semaphor.create_event(np.linspace(self.t_space[0], self.t_space[-1], N_semaphor + 1), msg)

        sol = sp.integrate.solve_ivp(c_dot, [self.t_space[0], self.t_space[-1]], c_0, method = 'RK45', t_eval = self.t_space, args = (new_sem_ID,))

        self.solution_benchmark = self.semaphor.finish_event(new_sem_ID, "    Simulation")

        print("  Translating Fock basis coefficients into mode occupancies...")

        self.solution = []
        for t_i in range(len(self.t_space)):
            cur_c = np.zeros(fock_N, dtype=complex)
            for i in range(fock_N):
                cur_c[i] = sol.y[i][t_i]

            self.solution.append(np.zeros(self.M))
            for m in range(self.M):
                for i in range(fock_N):
                    self.solution[t_i][m] += fock_basis[i][m] * (cur_c[i].real * cur_c[i].real + cur_c[i].imag * cur_c[i].imag) / self.S

        print(f"  Done! Solution on the full occupancy basis found in {functions.dtstr(self.solution_benchmark)}")


        self.is_solved = True


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

    def uncoupled_wavef_y_dot(self, t, y, reg_timescale, basis_index, semaphor_event_ID = None):
        # Here y = A_i, y.shape = (N)

        cur_basis, cur_basis_dot = self.basis_state_at_time(t, basis_index)
        overlap_matrix = np.zeros((self.N[basis_index], self.N[basis_index], 3), dtype=complex) #[i][j][r] = { z^(r) | z^(r) }
        for i in range(self.N[basis_index]):
            for j in range(self.N[basis_index]):
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
        M = np.zeros((self.N[basis_index], self.N[basis_index]), dtype=complex)
        for i in range(self.N[basis_index]):
            for j in range(self.N[basis_index]):
                M[i][j] = 1j * overlap_matrix[i][j][0]

        # Now we initialize R_i
        R = np.zeros(self.N[basis_index], dtype=complex)
        for i in range(self.N[basis_index]):
            sum1 = 0.0
            for j in range(self.N[basis_index]):
                sum1 += y[j] * overlap_matrix[i][j][1] * np.sum(np.conjugate(cur_basis[i]) * cur_basis_dot[j])
            sum1 *= -1j * self.S

            sum2 = 0.0
            for j in range(self.N[basis_index]):
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
            for i in range(self.N[basis_index]):
                M[i][i] += reg_timescale
        M_inv = np.linalg.inv(M)

        # Semaphor
        self.semaphor.update(semaphor_event_ID, t)

        return(M_inv.dot(R))

    # ------------------ Single vector classical trajectory -------------------

    def normalised_z_y_dot(self, t, y, reg_timescale = -1):
        # y[m] = z_m

        y_std = np.ones(self.M, dtype = complex)
        y_std[:self.M - 1] = y
        """
        #   wrong hamiltonian, using { z | H | z } instead of < z | H | z >
        # Overlap matrices
        X = np.zeros(4)
        base_inner_product = 1 + np.sum(y.real * y.real + y.imag * y.imag)
        X[3] = np.power(base_inner_product, self.S - 3)
        X[2] = base_inner_product * X[3]
        X[1] = base_inner_product * X[2]
        X[0] = base_inner_product * X[1]
        # Hamiltonian tensors and differential vector
        cur_H_A, cur_H_B = self.calculate_hamiltonian_tensors(t)
        H_mel_diff = np.zeros(self.M - 1, dtype = complex)
        for j in range(self.M - 1):
            one_body_interaction = 0.0
            for alpha in range(self.M):
                one_body_interaction += self.S * X[1] * cur_H_A[j][alpha] * y_std[alpha]
                for beta in range(self.M):
                    one_body_interaction += self.S * (self.S - 1) * X[2] * y_std[j] * cur_H_A[alpha][beta] * np.conjugate(y_std[alpha]) * y_std[beta]

            two_body_interaction = 0.0
            for alpha in range(self.M):
                for beta in range(self.M):
                    for gamma in range(self.M):
                        two_body_interaction += 0.5 * self.S * (self.S - 1) * X[2] * (cur_H_B[j][alpha][beta][gamma] + cur_H_B[alpha][j][beta][gamma]) * np.conjugate(y_std[alpha]) * y_std[beta] * y_std[gamma]
                        for delta in range(self.M):
                            two_body_interaction += 0.5 * self.S * (self.S - 1) * (self.S - 2) * X[3] * y_std[j] * cur_H_B[alpha][beta][gamma][delta] * np.conjugate(y_std[alpha]) * np.conjugate(y_std[beta]) * y_std[gamma] * y_std[delta]
            H_mel_diff[j] = one_body_interaction + two_body_interaction"""

        # fixed
        base_inner_product = 1 + np.sum(y.real * y.real + y.imag * y.imag)
        # Hamiltonian tensors and differential vector
        cur_H_A, cur_H_B = self.calculate_hamiltonian_tensors(t)
        H_mel_diff = np.zeros(self.M - 1, dtype = complex)
        for j in range(self.M - 1):
            one_body_interaction = 0.0
            for alpha in range(self.M):
                one_body_interaction += (self.S / base_inner_product) * cur_H_A[j][alpha] * y_std[alpha]
                for beta in range(self.M):
                    one_body_interaction -= (self.S / (base_inner_product * base_inner_product)) * y_std[j] * cur_H_A[alpha][beta] * np.conjugate(y_std[alpha]) * y_std[beta]

            two_body_interaction = 0.0
            for alpha in range(self.M):
                for beta in range(self.M):
                    for gamma in range(self.M):
                        two_body_interaction += 0.5 * (self.S * (self.S - 1) / (base_inner_product * base_inner_product)) * (cur_H_B[j][alpha][beta][gamma] + cur_H_B[alpha][j][beta][gamma]) * np.conjugate(y_std[alpha]) * y_std[beta] * y_std[gamma]
                        for delta in range(self.M):
                            two_body_interaction -= (self.S * (self.S - 1) / (base_inner_product * base_inner_product * base_inner_product)) * y_std[j] * cur_H_B[alpha][beta][gamma][delta] * np.conjugate(y_std[alpha]) * np.conjugate(y_std[beta]) * y_std[gamma] * y_std[delta]
            H_mel_diff[j] = one_body_interaction + two_body_interaction



        outer_product_plus_identity = np.zeros((self.M - 1, self.M - 1), dtype = complex)
        prefactor = - 1j * base_inner_product / self.M
        for i in range(self.M - 1):
            outer_product_plus_identity[i][i] = prefactor
            for j in range(self.M - 1):
                outer_product_plus_identity[i][j] += prefactor * np.conjugate(y[j]) * y[i]
        return(outer_product_plus_identity.dot(H_mel_diff))



    # --------------- Fully variational methods ---------------

    def standardise_dynamic_variables(self, cur_basis, cur_wavef):
        # y is the same format as R, so [A_1, A_2 ... A_N, z_1,1, z_2,1 ... z_N,1 ... ]
        # cur_basis is a list of lists [n][m]
        N = len(cur_basis)
        y = np.zeros(N * self.M, dtype = complex)
        for n in range(N):
            y[n] = cur_wavef[n]
            for m in range(self.M-1):
                y[N + N * m + n] = cur_basis[n][m]
        return(y)

    def destandardise_dynamic_variables(self, y):
        N = int(len(y) / self.M)
        cur_basis = np.zeros((N, self.M - 1), dtype=complex)
        cur_wavef = np.zeros(N, dtype=complex)
        for n in range(N):
            cur_wavef[n] = y[n]
            for m in range(self.M-1):
                cur_basis[n][m] = y[N + N * m + n]
        return(cur_basis, cur_wavef)


    def calculate_overlap_matrices(self, y, max_reduction = 3):
        # we note that z_n,m = y[N + N * m + n]
        # therefore z_n as a list is = y[N + n : N + n + N * M : N], where n goes from 0 to N-1 inclusive
        N = int(len(y) / self.M)
        X = np.zeros((N, N, max_reduction + 1), dtype=complex) # X[i][j][r] = { z_i^(r) | z_j^(r) }

        # We optimize the number of operations by noting that { z_i^(r) | z_j^(r) } = { z_i^(r+1) | z_j^(r+1) } * (1+z_i*.z_j)
        for i in range(N):
            for j in range(N):
                base_inner_product = 1 + np.sum(np.conjugate(y[N + i : N + i + N * self.M : N]) * y[N + j : N + j + N * self.M : N])
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


    def variational_y_dot(self, t, y, reg_timescale = -1, semaphor_event_ID = None):
        # y is the same format as R, so [A_1, A_2 ... A_N, z_1,1, z_2,1 ... z_N,1 ... z_1,(M-1) ... z_N,(M-1)]

        # We also create a standardised y which includes z_n,M = 1
        # we note that z_n,m = y_std[N + N * m + n]
        N = int(len(y) / self.M)

        y_std = np.ones(N * (self.M + 1), dtype = complex)
        y_std[:N * self.M] = y

        # we want to read the semaphor right here in this function

        X = self.calculate_overlap_matrices(y, 3)
        cur_H_A, cur_H_B = self.calculate_hamiltonian_tensors(t)

        R = np.zeros(self.M * N, dtype=complex)
        m_Theta = np.zeros((self.M*N, self.M*N), dtype=complex)

        # First we calculate R
        for k in range(N):
            for j in range(N):
                # First, the Hamiltonian matrix element { z_k | H | z_j }
                H_mel_s1 = 0.0
                for a in range(self.M):
                    for b in range(self.M):
                        H_mel_s1 += cur_H_A[a][b] * np.conjugate(y_std[N + N * a + k]) * y_std[N + N * b + j]
                H_mel_s2 = 0.0
                for a in range(self.M):
                    for b in range(self.M):
                        for c in range(self.M):
                            for d in range(self.M):
                                H_mel_s2 += cur_H_B[a][b][c][d] * np.conjugate(y_std[N + N * a + k]) * np.conjugate(y_std[N + N * b + k]) * y_std[N + N * c + j] * y_std[N + N * d + j]
                H_mel = self.S * X[k][j][1] * H_mel_s1 + 0.5 * self.S * (self.S - 1) * X[k][j][2] * H_mel_s2

                R[k] += y_std[j] * H_mel

                for m in range(self.M-1):
                    H_mel_diff = self.S * (self.S - 1) * X[k][j][2] * y_std[N + N * m + j] * H_mel_s1 + 0.5 * self.S * (self.S - 1) * (self.S - 2) * X[k][j][3] * y_std[N + N * m + j] * H_mel_s2
                    term3 = 0.0
                    for b in range(self.M):
                        term3 += cur_H_A[m][b] * y_std[N + N * b + j]
                    term4 = 0.0
                    for b in range(self.M):
                        for c in range(self.M):
                            for d in range(self.M):
                                term4 += (cur_H_B[m][b][c][d] + cur_H_B[b][m][c][d]) * np.conjugate(y_std[N + N * b + k]) * y_std[N + N * c + j] * y_std[N + N * d + j]
                    H_mel_diff += self.S * X[k][j][1] * term3 + 0.5 * self.S * (self.S - 1) * X[k][j][2] * term4

                    R[N + N * m + k] += np.conjugate(y_std[k]) * y_std[j] * H_mel_diff

        # Then, we calculate Theta
        # First, we fill in X
        for i in range(N):
            for j in range(N):
                m_Theta[i][j] = X[i][j][0]
        # Then, we fill in Y and Y^h.c.
        for a in range(N):
            for b in range(self.M-1):
                for d in range(N):
                    m_Theta[a][N + b * N + d] = self.S * np.conjugate(y_std[N + N * b + a]) * y_std[d] * X[a][d][1]
                    m_Theta[N + b * N + d][a] = np.conjugate(m_Theta[a][N + b * N + d])
        # Then, we fill in Z
        for i in range(self.M-1):
            for j in range(self.M-1):
                for a in range(N):
                    for b in range(N):
                        # first, we evaluate (F_ij)_ab
                        m_Theta[N + i * N + a][N + j * N + b] = self.S * (self.S - 1) * X[a][b][2] * np.conjugate(y_std[N + N * j + a]) * y_std[N + N * i + b]
                        if i == j:
                            m_Theta[N + i * N + a][N + j * N + b] += self.S * X[a][b][1]
                        m_Theta[N + i * N + a][N + j * N + b] *= np.conjugate(y_std[a]) * y_std[b]

        # Regularisation
        if reg_timescale != -1:
            for i in range(self.M * N):
                m_Theta[i][i] += reg_timescale

        # Finally, we calculate y dot
        m_Theta_inv = np.linalg.inv(m_Theta)

        # Semaphor
        self.semaphor.update(semaphor_event_ID, t)

        sol = - 1j * m_Theta_inv.dot(R)

        # If N = 1, we enforce the time derivative of the normalisation
        #if N == 1:
        #    sol[0] =  - self.S * X[0][0][1] / np.power(X[0][0][0], 3/2) * np.sum(np.conjugate(y[1:]) * sol[1:]).real

        #return( - 1j * m_Theta_inv.dot(R))
        return(sol)






    ###########################################################################
    # ------------------------------- USER METHODS ----------------------------
    ###########################################################################
    # Methods intended to be invoked by the user

    # -------------------------------------------------------------------------
    # ------------------ Physical system description methods ------------------
    # ------------------------------------------------------------------------

    def set_global_parameters(self, M, S):
        self.M = M
        self.S = S

        print("A driven Bose-Hubbard model with %i bosons over %i modes has been initialized." % (self.S, self.M))

    # The general SU(N) Hamiltonian is fully described by the tensors A_a,b and B_a,b,c,d

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
    # --------------- Experiment setup methods ----------------
    # ---------------------------------------------------------

    def set_initial_wavefunction(self, initial_wavefunction, message = ""):
        # initial_wavefunction is an object interpreted contextually based on
        # the method specified by message, as per the table:
        #   "aguiar_pure" : a z-vec (i.e. complex ndarray of len M - 1), which represents an unnormalised state | z }
        #   "aguiar_disc": a list of two lists, first with z-vecs and second with weights
        #   "aguiar_cont": a function which takes a z-vec and outputs the overlap { z | psi }. NOT SUPPORTED YET

        # The wavef need not be normalised. For aguiar_disc, the normalisation occurs automatically.

        self.wavef_init_supplementary = {} # derived quantities which need not be stored

        if message in ["aguiar_pure"]:
            self.wavef_initial_wavefunction = initial_wavefunction
        elif message in ["aguiar_disc"]:
            # The suer-input method takes the coefficients of the normalised states | z >, although it is stored as the coefficients as the unnormalised states | z }.
            # Hence first we re-weight every coefficient by the norm of the corresponding state
            self.wavef_initial_wavefunction = initial_wavefunction
            for i in range(len(self.wavef_initial_wavefunction[0])):
                self.wavef_initial_wavefunction[1][i] /= np.sqrt(square_norm(self.wavef_initial_wavefunction[0][i], self.S))
            Psi_mag = 0.0
            for i in range(len(self.wavef_initial_wavefunction[0])):
                for j in range(len(self.wavef_initial_wavefunction[0])):
                    Psi_mag += np.conjugate(self.wavef_initial_wavefunction[1][i]) * self.wavef_initial_wavefunction[1][j] * overlap(self.wavef_initial_wavefunction[0][i], self.wavef_initial_wavefunction[0][j], self.S)
            print(f"  Initial wavefunction normalized with square magnitude {Psi_mag.real:.4f}. The final decomposition will be naively renormalized...")

            # Naive renormalization
            self.wavef_init_supplementary["p"] = np.zeros(len(self.wavef_initial_wavefunction[1]))
            for i in range(len(self.wavef_initial_wavefunction[1])):
                self.wavef_initial_wavefunction[1][i] /= np.sqrt(Psi_mag.real)
                self.wavef_init_supplementary["p"][i] = (self.wavef_initial_wavefunction[1][i].real * self.wavef_initial_wavefunction[1][i].real + self.wavef_initial_wavefunction[1][i].imag * self.wavef_initial_wavefunction[1][i].imag) * np.sqrt(square_norm(self.wavef_initial_wavefunction[0][i], self.S))
            self.wavef_init_supplementary["p"] /= np.sum(self.wavef_init_supplementary["p"])



        self.wavef_message = message
        self.is_wavef_init = True

        """if self.message == "aguiar_disc":
            # we calculate p
            self.init_wavef_supplementary["p"] = np.array(initial_wavefunction[1]).real * np.array(initial_wavefunction[1]).real + np.array(initial_wavefunction[1]).imag * np.array(initial_wavefunction[1]).imag
            self.init_wavef_supplementary["p"] /= np.sum(self.init_wavef_supplementary["p"])"""
    # ---------------------------------------------------------
    # ------------------- Sampling methods --------------------
    # ---------------------------------------------------------

    # We will always use the inverted overlap matrix identity, and so sampling may be done with any method.
    # The sampling interprets magnitude squared overlap with init_wavef as the weight function.

    def get_random_basis_vector(self, width):
        # this is the "goofy gaussians" method
        z_0_std = np.zeros(2 * (self.M - 1))
        if self.wavef_message in ["aguiar_pure"]:
            for m in range(self.M - 1):
                z_0_std[m] = self.wavef_initial_wavefunction[m].real
                z_0_std[self.M - 1 + m] = self.wavef_initial_wavefunction[m].imag
        elif self.wavef_message in ["aguar_disc"]:
            # First we randomly choose one of the foci
            z_focal_index = np.random.choice(len(self.wavef_initial_wavefunction[0]), p = self.wavef_init_supplementary["p"])
            for m in range(self.M - 1):
                z_0_std[m] = self.wavef_initial_wavefunction[0][z_focal_index][m].real
                z_0_std[self.M - 1 + m] = self.wavef_initial_wavefunction[0][z_focal_index][m].imag
        candidate_basis_vector_std = np.random.normal(z_0_std, width)
        candidate_basis_vector = np.zeros(self.M - 1, dtype=complex)
        for m in range(self.M - 1):
            candidate_basis_vector[m] = candidate_basis_vector_std[m] + candidate_basis_vector_std[self.M - 1 + m] * 1j
        return(candidate_basis_vector)


    def sample_gaussian(self, width = 1.0, conditioning_limit = -1, N_max = 50, max_saturation_steps = 50, include_every_focal_point = True):

        if self.M == 0:
            print("ERROR: You have attempted to sample a basis set before initializing the mode and particle number of the system. You can do this by calling set_global_parameters(M, S).")
            return(-1)

        # z_0 must be of the Aguiar variety
        self.basis_config.append({
                "method" : "gaussian",
                "width" : width,
                "conditioning_limit" : conditioning_limit,
                "N_max" : N_max,
                "max_saturation_steps" : max_saturation_steps,
                "include_every_focal_point" : include_every_focal_point
            })

        new_basis = []

        if self.wavef_message in ["aguiar_pure"]:
            new_basis.append(self.wavef_initial_wavefunction)
        elif self.wavef_message in ["aguiar_disc"]:
            if include_every_focal_point:
                if len(self.wavef_initial_wavefunction[0]) > N_max:
                    print(f"  Sampling with N_max = {N_max} queued with include_every_focal_point = True, but the number of focal points is {len(self.wavef_initial_wavefunction[0])}. Overriding N_max...")
                for i in range(len(self.wavef_initial_wavefunction[0])):
                    new_basis.append(self.wavef_initial_wavefunction[0][i])

        print(f"Sampling the neighbourhood of every focal point with a normal distribution of width {width}...")
        steps_since_last_addition = 0

        # Complex positive-definite Hermitian matrix with positive real eigenvalues
        overlap_matrix = []
        for i in range(len(new_basis)):
            overlap_matrix.append([])
            for j in range(len(new_basis)):
                if j > i:
                    overlap_matrix[i].append(overlap(new_basis[i], new_basis[j], self.S))
                elif j == i:
                    overlap_matrix[i].append(square_norm(new_basis[i], self.S))
                else:
                    overlap_matrix[i].append(np.conjugate(overlap_matrix[j][i]))

        # we omit the normalizations, since what we really care about is the properties of X_ab = {z_a | z_b}, NOT the normalized version


        while(len(new_basis) < N_max):
            # Grab a new candidate
            candidate_basis_vector = self.get_random_basis_vector(width)

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

        self.basis.append(new_basis)

        self.N.append(len(new_basis))
        print(f"  A sample of N = {self.N[-1]} basis vectors drawn from a normal distribution of width {width} centered randomly around the focal points has been initialized.")

        # Create the identity operator matrix
        X = np.zeros((self.N[-1], self.N[-1]), dtype = complex)
        for m in range(self.N[-1]):
            for n in range(self.N[-1]):
                X[m][n] = overlap(self.basis[-1][m], self.basis[-1][n], self.S)
        self.inverse_overlap_matrix.append(np.linalg.inv(X))


        self.is_basis_init = True


    # ---------------------------------------------------------
    # ------------------ Simulation methods -------------------
    # ---------------------------------------------------------

    """def set_initial_wavefunction(self, initial_wavefunction = [], message = ""):
        # Allows different methods for finding the initial decomposition, as
        # encoded by the 'message' argument

        if not self.is_basis_init:
            print("ERROR: You have attempted to set the initial wavefunction state before sampling a basis set. You can do this by calling one of the many basis-sampling methods listed below.")
            for m, desc in bosonic_su_n.basis_sampling_methods.items():
                print(f"{m} [{desc}]")
            return(-1)

        if message in ["manual", "A_vals"]:
            # initial_wavefunction is just directly equal to the initial decomposition coefficients of the wavefunction
            return(-1) # TODO unsupported for multiple bases; maybe allow to set this manually for the first basis?
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
            candidate_wavef = []
            for b_i in range(len(self.basis)):
                candidate_wavef.append(self.decompose_aguiar(candidate_initial_wavefunction, b_i))
        elif message in ["grossmann", "frank"]:
            # initial_wavefunction describes a Grossmann normalized coherent state.
            candidate_initial_wavefunction = np.array(initial_wavefunction)
            if candidate_initial_wavefunction.shape != (self.M,):
                print("ERROR: When using the 'grossmann' method, initial_wavefunction has to be an array of shape (M).")
                return(-1)
            if np.round(np.sum(candidate_initial_wavefunction.real * candidate_initial_wavefunction.real + candidate_initial_wavefunction.imag * candidate_initial_wavefunction.imag), 2) != 1.0:
                print("ERROR: When using the 'grossmann' method, initial_wavefunction has to have norm 1.")
                return(-1)
            candidate_wavef = []
            for b_i in range(len(self.basis)):
                candidate_wavef.append(self.decompose_grossmann(candidate_initial_wavefunction, b_i))
        elif message in ["", "NONE"]:
            print("set_initial_wavefunction called without specifying the method of initialization.")
            print("Default option selected: initial wavefunction will be set to the first element of the basis set.")
            candidate_wavef = []
            for b_i in range(len(self.basis)):
                candidate_wavef.append(self.decompose_aguiar(self.basis[0][0], b_i))
            message = "NONE"
        else:
            print("ERROR: Unknown method of wavefunction initialization.")
            return(-1)

        for b_i in range(len(self.basis)):
            Psi_mag = 0.0
            for i in range(self.N[b_i]):
                for j in range(self.N[b_i]):
                    Psi_mag += np.conjugate(candidate_wavef[b_i][i]) * candidate_wavef[b_i][j] * overlap(self.basis[b_i][i], self.basis[b_i][j], self.S)
            print(f"  In basis #{b_i + 1}: Initial wavefunction normalized with magnitude {Psi_mag.real:.4f}. The final decomposition will be naively renormalized...")

            # Naive renormalization
            for i in range(self.N[b_i]):
                candidate_wavef[b_i][i] /= np.sqrt(Psi_mag.real)
        self.wavef = candidate_wavef

        self.wavef_initial_wavefunction = initial_wavefunction
        self.wavef_message = message

        self.is_wavef_init = True
        return(0)"""


    # From the initialized basis and having inputted the initial wavefunction,
    # the user may choose to simulate the uncoupled basis evolution, the
    # coupled basis evolution, the evolution of the wavefunction decomposition
    # on top of both of these, or the full variational method evolution.
    # Each time, the previous "evolution" dataset is erased

    def simulate_uncoupled_basis(self, max_t = -1, N_dtp = -1, rtol = 1e-3, reg_timescale = -1, N_semaphor = 100):
        # rtol/reg_timescale may be a tuple, in which case the first element is the basis timescale and the second element the wavef timescale
        if isinstance(rtol, tuple):
            rtol_basis, rtol_wavef = rtol
        else:
            rtol_basis = rtol
            rtol_wavef = rtol

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

        # We propagate each basis vector one by one TODO

        # Set up semaphor for basis propagation
        # TODO

        if not self.is_t_space_init:
            if max_t == -1 or N_dtp == -1:
                print("  ERROR: If t_space has not been previously initialized, max_t and N_dtp are required arguments!")
                return(-1)
            simulation_start_time = 0.0 # TODO is data loaded, this should be the last timestamp
            self.t_space = np.linspace(simulation_start_time, max_t, N_dtp+1)
            self.is_t_space_init = True


        self.basis_evol = []
        self.wavef_evol = []

        self.evol_benchmarks = []

        # From this point on, the new method shines! We shall use a customisable RK45 iterator with exit conditions etc
        for b_i in range(len(self.basis)):

            print(f"# Analyzing basis no. {b_i + 1}")
            cur_start_time = time.time()

            cur_basis_evol = [np.zeros((self.N[b_i], self.M-1), dtype=complex)]
            for i in range(self.N[b_i]):
                for j in range(self.M-1):
                    cur_basis_evol[0][i][j] = self.basis[b_i][i][j]
            for t_i in range(1, N_dtp+1):
                cur_basis_evol.append(np.zeros((self.N[b_i], self.M-1), dtype=complex))

            #print(f"  Uncoupled basis propagation on a timescale of t = ({self.t_space[0]} - {self.t_space[-1]}), rtol = {rtol_basis} at {time.strftime("%H:%M:%S", time.localtime(time.time()))}")

            msg = f"  Uncoupled basis propagation on a timescale of t = ({self.t_space[0]} - {self.t_space[-1]}), rtol = {rtol_basis} at {time.strftime("%H:%M:%S", time.localtime(time.time()))}"
            new_sem_ID = self.semaphor.create_event(np.arange(0, self.N[b_i] + 1, 1), msg)
            basis_solutions = []
            for n in range(self.N[b_i]):
                A_0 = 1 / np.sqrt(functions.square_norm(self.basis[b_i][n], self.S))
                y_0 = np.array([A_0] + list(self.basis[b_i][n].copy()))
                basis_solutions.append(functions.solve_ivp_with_condition(self.variational_y_dot, (self.t_space[0], self.t_space[-1]), y0=y_0, t_eval=self.t_space, args=(reg_timescale_basis,), exit_condition=None, rtol = rtol_basis, max_step = 0.01))
                self.semaphor.update(new_sem_ID, n + 1)
            self.semaphor.finish_event(new_sem_ID, "    Basis propagation")

            # NOTE: When dynamically re-partitioning the basis set, we would always only do the solution between two t_eval (or t_partition) vals, and then re-initialised the solvers
            for n in range(self.N[b_i]):
                for t_i in range(1, N_dtp+1):
                    for m in range(self.M - 1):
                        cur_basis_evol[t_i][n][m] = basis_solutions[n].y[t_i][m + 1]

            self.basis_evol.append(cur_basis_evol)

            """x_data = []
            y_data = []
            y_dot_data = []
            E_data = []"""

            def basis_dependent_wavef_y_dot(t, y, reg_timescale, semaphor_event_ID = None):

                # y[a] = A_a(t)

                #x_data.append(t)
                #y_data.append([])
                #y_dot_data.append([])

                cur_basis = np.zeros((self.N[b_i], self.M - 1), dtype=complex)
                cur_basis_dot = np.zeros((self.N[b_i], self.M - 1), dtype=complex)
                cur_N = np.zeros(self.N[b_i], dtype=complex)
                cur_N_dot = np.zeros(self.N[b_i], dtype=complex)
                for n in range(self.N[b_i]):
                    cur_weighted_basis_state = basis_solutions[n].sol(t)
                    cur_weighted_basis_state_dot = self.variational_y_dot(t, cur_weighted_basis_state)
                    cur_basis[n] = cur_weighted_basis_state[1:]
                    cur_basis_dot[n] = cur_weighted_basis_state_dot[1:]
                    cur_N[n] = cur_weighted_basis_state[0]
                    cur_N_dot[n] = cur_weighted_basis_state_dot[0]

                    #y_data[-1].append(np.sqrt(functions.square_mag(cur_weighted_basis_state[0])) * np.sqrt(functions.square_norm(cur_basis[n], self.S).real))
                    #y_dot_data[-1].append(np.sqrt(functions.square_mag(cur_weighted_basis_state_dot[0])) * np.sqrt(functions.square_norm(cur_basis_dot[n], self.S).real))
                #y_data.append([basis_solutions[0].sol(t)[0].real, basis_solutions[0].sol(t)[1].real, basis_solutions[0].sol(t)[1].imag])
                #y_dot_data.append([self.variational_y_dot(t, basis_solutions[0].sol(t))[0].real, self.variational_y_dot(t, basis_solutions[0].sol(t))[1].real, self.variational_y_dot(t, basis_solutions[0].sol(t))[1].imag])

                def ebe(i, k): # extended basis element
                    if k != self.M - 1:
                        return(cur_basis[i][k])
                    else:
                        return(1.0)

                cur_H_A, cur_H_B = self.calculate_hamiltonian_tensors(t)

                Xi = np.zeros((3, self.N[b_i], self.N[b_i]), dtype=complex)
                zeta = np.zeros((self.N[b_i], self.N[b_i]), dtype=complex)
                epsilon = np.zeros((self.N[b_i], self.N[b_i]), dtype=complex)
                eta = np.zeros((self.N[b_i], self.N[b_i]), dtype=complex)
                for a in range(self.N[b_i]):
                    for b in range(self.N[b_i]):
                        cur_base_overlap = 1 + np.sum(np.conjugate(cur_basis[a]) * cur_basis[b])
                        cur_xi = np.power(np.conjugate(cur_N[a]) * cur_N[b], 1.0 / self.S) * cur_base_overlap
                        Xi[0][a][b] = np.power(cur_xi, self.S)
                        Xi[1][a][b] = Xi[0][a][b] / cur_base_overlap
                        Xi[2][a][b] = Xi[1][a][b] / cur_base_overlap

                        zeta[a][b] = Xi[0][a][b] * cur_N_dot[b] / cur_N[b]

                        epsilon[a][b] = Xi[1][a][b] * (np.sum(np.conjugate(cur_basis[a]) * cur_basis_dot[b]))

                        one_body_interaction = 0.0
                        two_body_interaction = 0.0
                        for alpha in range(self.M):
                            for beta in range(self.M):
                                one_body_interaction += cur_H_A[alpha][beta] * np.conjugate(ebe(a, alpha)) * ebe(b, beta)
                                for gamma in range(self.M):
                                    for delta in range(self.M):
                                        two_body_interaction += cur_H_B[alpha][beta][gamma][delta] * np.conjugate(ebe(a, alpha)) * np.conjugate(ebe(a, beta)) * ebe(b, gamma) * ebe(b, delta)
                        eta[a][b] = self.S * Xi[1][a][b] * one_body_interaction + 0.5 * self.S * (self.S - 1) * Xi[2][a][b] * two_body_interaction

                """E_data.append(0.0)
                for a in range(self.N[b_i]):
                    for b in range(self.N[b_i]):
                        E_data[-1] += np.conjugate(y[a]) * y[b] * eta[a][b]"""


                # Regularisation
                if reg_timescale != -1:
                    Xi_inv = np.linalg.inv(Xi[0] + reg_timescale * np.identity(self.N[b_i], dtype=complex)) # inverse of the non-reduced overlap matrix
                else:
                    Xi_inv = np.linalg.inv(Xi[0]) # inverse of the non-reduced overlap matrix

                M = np.matmul(Xi_inv, -1j * eta - zeta - self.S * epsilon)


                # Semaphor
                self.semaphor.update(semaphor_event_ID, t)

                return(M.dot(y))

            # ---------------- wavef propagation ------------------
            # We propagate the wavefunction
            print("  Propagating wavefunction decomposition...")
            cur_wavef_evol = [np.zeros(self.N[b_i], dtype=complex)]
            cur_wavef_init_decomposition = self.decompose_init_wavef_into_basis(b_i)
            for i in range(self.N[b_i]):
                cur_wavef_evol[0][i] = cur_wavef_init_decomposition[i]
            for t_i in range(1, N_dtp+1):
                cur_wavef_evol.append(np.zeros(self.N[b_i], dtype=complex))

            # Set up semaphor event
            msg = f"  Wavefunction propagation over the evolved uncoupled basis on a timescale of t = ({self.t_space[0]} - {self.t_space[-1]}), rtol = {rtol_wavef}"
            new_sem_ID = self.semaphor.create_event(np.linspace(self.t_space[0], self.t_space[-1], N_semaphor + 1), msg)

            y_0 = cur_wavef_init_decomposition.copy()
            for n in range(self.N[b_i]):
                y_0[n] /= basis_solutions[n].sol(self.t_space[0])[0]
            iterated_solution = sp.integrate.solve_ivp(basis_dependent_wavef_y_dot, [self.t_space[0], self.t_space[-1]], y_0, method = 'RK45', t_eval = self.t_space, args = (reg_timescale_wavef, new_sem_ID), rtol = rtol_wavef)

            """plt.title("Interpolated basis values")
            plt.ylim((-1e2, 1e2))
            plt.plot(x_data, y_data, label = "z")
            plt.plot(x_data, y_dot_data, label = "dz/dt")
            plt.plot(x_data, np.gradient(np.array(y_data), np.array(x_data), axis = 0), label = "interpolated dz/dt", linestyle="dashed")
            plt.plot(x_data, E_data, label = "energy")
            plt.legend()
            plt.show()"""

            # The issue is that after this method concludes, the basis norms are discarded, but they are now a vital part of the solution!
            # Two ways to resolve this:
            #     1. Store an extra set of values for basis norms
            #     2. Absorb N into A
            # Let's firstly do method 2 because it's easier, and if the floating point error persists, we shall try method 1. It works!


            for t_i in range(1, N_dtp+1):
                for n in range(self.N[b_i]):
                    cur_wavef_evol[t_i][n] = iterated_solution.y[n][t_i] * basis_solutions[n].sol(self.t_space[t_i])[0]

            self.semaphor.finish_event(new_sem_ID, "    Simulation")

            self.wavef_evol.append(cur_wavef_evol)

            self.evol_benchmarks.append(time.time() - cur_start_time)
            print("    Total (basis & wavefunction) benchmark: " + functions.dtstr(self.evol_benchmarks[b_i]))

        self.is_basis_evol = True
        self.is_wavef_evol = True


    def simulate_variational(self, max_t = -1, N_dtp = -1, rtol = 1e-3, reg_timescale = -1, N_semaphor = 100):
        print("Simulating state evolution by propagating both the basis vectors and the wavefunction decomposition coefficients coupled in a fully variational method.")

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

        if not self.is_t_space_init:
            if max_t == -1 or N_dtp == -1:
                print("  ERROR: If t_space has not been previously initialized, max_t and N_dtp are required arguments!")
                return(-1)
            simulation_start_time = 0.0 # TODO is data loaded, this should be the last timestamp
            self.t_space = np.linspace(simulation_start_time, max_t, N_dtp+1)
            self.is_t_space_init = True

        self.basis_evol = []
        self.wavef_evol = []

        self.evol_benchmarks = []

        for b_i in range(len(self.basis)):

            print(f"# Analyzing basis no. {b_i + 1}")
            cur_start_time = time.time()

            cur_wavef_init_decomposition = self.decompose_init_wavef_into_basis(b_i)

            # Initialize basis_evol and wavef_evol from initial conditions
            cur_basis_evol = [np.zeros((self.N[b_i], self.M-1), dtype=complex)]
            for i in range(self.N[b_i]):
                for j in range(self.M-1):
                    cur_basis_evol[0][i][j] = self.basis[b_i][i][j]
            for t_i in range(1, N_dtp+1):
                cur_basis_evol.append(np.zeros((self.N[b_i], self.M-1), dtype=complex))
            cur_wavef_evol = [np.zeros(self.N[b_i], dtype=complex)]
            for i in range(self.N[b_i]):
                cur_wavef_evol[0][i] = cur_wavef_init_decomposition[i]
            for t_i in range(1, N_dtp+1):
                cur_wavef_evol.append(np.zeros(self.N[b_i], dtype=complex))

            # Set up semaphor event
            msg = f"Fully variational basis and wavefunction propagation on a timescale of t = ({self.t_space[0]} - {self.t_space[-1]}), rtol = {rtol}, reg_t = {reg_timescale}"
            new_sem_ID = self.semaphor.create_event(np.linspace(self.t_space[0], self.t_space[-1], N_semaphor + 1), msg)

            y_0 = self.standardise_dynamic_variables(self.basis[b_i], cur_wavef_init_decomposition)

            # Set up the RK45 (Dormand-Prince) iterator and iterate
            t_dense = [self.t_space[0]]
            y_dense = [y_0]
            variational_y_dot_partial = partial(self.variational_y_dot, reg_timescale = reg_timescale, semaphor_event_ID = new_sem_ID)
            solver = sp.integrate.RK45(fun = variational_y_dot_partial, t0 = self.t_space[0], y0 = y_0, t_bound=self.t_space[-1], rtol = rtol, max_step=bosonic_su_n.rk_max_step)
            while(solver.status == "running"):
                solver.step()
                t_dense.append(solver.t)
                y_dense.append(solver.y.copy())
                # Check exit conditions!
                """norm_trace = 0.0
                for i in range(self.N[b_i]):
                    norm_trace += (np.conjugate(solver.y[i]) * solver.y[i]) * np.power(1 + np.sum(np.conjugate(solver.y[self.N[b_i] + i : self.N[b_i] + i + self.N[b_i] * self.M : self.N[b_i]]) * solver.y[self.N[b_i] + i : self.N[b_i] + i + self.N[b_i] * self.M : self.N[b_i]]), self.S)
                if norm_trace.real > bosonic_su_n.rk_max_norm_deviation:
                    break"""

            resul_interpolator = sp.interpolate.interp1d(x = np.array(t_dense), y = np.array(y_dense), axis = 0, assume_sorted = True, bounds_error = False)
            for t_i in range(1, N_dtp+1):
                interpolated_y = resul_interpolator(self.t_space[t_i])
                for n in range(self.N[b_i]):
                    cur_wavef_evol[t_i][n] = interpolated_y[n]
                    for m in range(self.M-1):
                        cur_basis_evol[t_i][n][m] = interpolated_y[self.N[b_i] + self.N[b_i] * m + n]

            """iterated_solution = sp.integrate.solve_ivp(self.variational_y_dot, [self.t_space[0], self.t_space[-1]], y_0, method = 'RK45', t_eval = self.t_space, args = (reg_timescale, new_sem_ID), rtol = rtol, max_step = bosonic_su_n.rk_max_step)

            for t_i in range(1, N_dtp+1):
                for n in range(self.N[b_i]):
                    cur_wavef_evol[t_i][n] = iterated_solution.y[n][t_i]
                    for m in range(self.M-1):
                        cur_basis_evol[t_i][n][m] = iterated_solution.y[self.N[b_i] + self.N[b_i] * m + n][t_i]"""

            self.semaphor.finish_event(new_sem_ID, "  Simulation")

            self.basis_evol.append(cur_basis_evol)
            self.wavef_evol.append(cur_wavef_evol)

            self.evol_benchmarks.append(time.time() - cur_start_time)
            print("    Total (basis & wavefunction) benchmark: " + dtstr(self.evol_benchmarks[b_i]))

        self.is_basis_evol = True
        self.is_wavef_evol = True



    # ---------------------------------------------------------
    # -------------------- Output methods ---------------------
    # ---------------------------------------------------------

    # ------------------ Text output methods ------------------

    def summarise_data(self):
        # Summarises obtained data and benchmarks

        if self.is_wavef_evol:
            print(f"System solved on {len(self.basis)} CS basis sets.")
            for b_i in range(len(self.basis)):
                print(f"  Basis {b_i + 1} of size {self.N[b_i]}: solution benchmark is {functions.dtstr(self.evol_benchmarks[b_i])} (basis & wavef)")
        if self.is_solved:
            print(f"System solved on the full occupancy basis; solution benchmark is {functions.dtstr(self.solution_benchmark)}")

    # --------------- Graphical output methods ----------------

    def plot_data(self, graph_list = ["expected_mode_occupancy", "initial_decomposition"], save_graph=True):

        print("Graphical output plotting routine initialized.")

        # Sample a color scheme for the modes
        mode_reference_points = np.linspace(0, 1, self.M + 2)[:-1]
        mode_reference_spacing = mode_reference_points[1] - mode_reference_points[0]
        mode_reference_spacing_fraction = 0.6

        self.ref_color = [] # [b_i]
        self.mode_colors = [] # [m][b_i]
        self.ref_color = plt.cm.rainbow(np.linspace(mode_reference_points[-1], mode_reference_points[-1] + mode_reference_spacing * mode_reference_spacing_fraction, len(self.basis) + 1))
        for m in range(self.M):
            self.mode_colors.append(plt.cm.rainbow(np.linspace(mode_reference_points[m], mode_reference_points[m] + mode_reference_spacing * mode_reference_spacing_fraction, len(self.basis) + 1)))


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

                plt.xlabel("$\\Re\\left(z_n\\right)$")
                plt.ylabel("$\\Im\\left(z_n\\right)$")
                for b_i in range(len(self.basis)):
                    x_vals = np.zeros(self.N[b_i])
                    y_vals = np.zeros(self.N[b_i])
                    s_vals = np.zeros(self.N[b_i])

                    for n in range(self.N[b_i]):
                        x_vals[n] = self.basis_evol[b_i][0][n][0].real
                        y_vals[n] = self.basis_evol[b_i][0][n][0].imag
                        s_vals[n] = np.absolute(self.wavef_evol[b_i][0][n])

                    plt.scatter(x_vals, y_vals, 5 + np.power(s_vals, 1/10) * 450, label = f"Basis n. {b_i + 1}")
                #include_legend = False
            elif graph_list[i] == 'expected_mode_occupancy':
                print("  Plotting the time evolution of expected mode occupancy...")#, end='', flush=True)
                plt.title("Expected mode occupancy")
                plt.xlabel("t")
                plt.ylim([-0.1, 1.5])

                print("    Plotting measured wavefunction magnitude and mode occupancies...")
                psi_mag = []
                avg_n = []
                for b_i in range(len(self.basis)):
                    psi_mag.append(np.zeros(len(self.t_space)))

                    avg_n.append([])
                    for m in range(self.M):
                        avg_n[b_i].append(np.zeros(len(self.t_space)))

                    for t_i in range(len(self.t_space)):
                        for a in range(self.N[b_i]):
                            for b in range(self.N[b_i]):
                                cur_X_1 = overlap(self.basis_evol[b_i][t_i][a], self.basis_evol[b_i][t_i][b], self.S, r = 1)
                                psi_mag[b_i][t_i] += (np.conjugate(self.wavef_evol[b_i][t_i][a]) * self.wavef_evol[b_i][t_i][b] * overlap(self.basis_evol[b_i][t_i][a], self.basis_evol[b_i][t_i][b], self.S)).real
                                for m in range(self.M-1):
                                    avg_n[b_i][m][t_i] += (np.conjugate(self.wavef_evol[b_i][t_i][a]) * self.wavef_evol[b_i][t_i][b] * np.conjugate(self.basis_evol[b_i][t_i][a][m]) * self.basis_evol[b_i][t_i][b][m] * cur_X_1).real
                                avg_n[b_i][self.M-1][t_i] += (np.conjugate(self.wavef_evol[b_i][t_i][a]) * self.wavef_evol[b_i][t_i][b] * cur_X_1).real



                # We insert horizontal lines indicating occupancies calculated from z_0
                """initial_occupancy = np.zeros(self.M)
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
                    plt.axhline(y = initial_occupancy[m], linestyle = "dotted", label = "init. $\\langle N_" + str(m+1) + " \\rangle/S$", color = self.mode_colors[m])"""


                for b_i in range(len(self.basis)):
                    plt.plot(self.t_space, psi_mag[b_i], label=f"$\\langle \\Psi | \\Psi \\rangle$ (N = {self.N[b_i]})", color = self.ref_color[b_i])
                for m in range(self.M):
                    for b_i in range(len(self.basis)):
                        plt.plot(self.t_space, avg_n[b_i][m], label=f"$\\langle N_{m+1} \\rangle/S$ (N = {self.N[b_i]})", color = self.mode_colors[m][b_i])
                    if self.is_solved:
                        cur_solution = []
                        for i in range(len(self.solution)):
                            cur_solution.append(self.solution[i][m])
                        plt.plot(self.t_space, cur_solution, linestyle = "dashed", label = f"theor. $\\langle N_{m+1} \\rangle/S$", color = self.mode_colors[m][-1])

                #print(" Done!")
            elif graph_list[i] == 'basis_expected_mode_occupancy':
                print("  Plotting the time evolution of expected mode occupancy for each basis vector...")#, end='', flush=True)
                plt.title("Basis: Expected mode occupancy")
                plt.xlabel("t")
                plt.ylim([-0.1, 1.5])

                print("    Plotting measured basis magnitude and mode occupancies...")
                for b_i in range(len(self.basis)):
                    for a in range(self.N[b_i]):
                        #magnitude = np.zeros(len(self.t_space))
                        occupancy = []
                        for m in range(self.M):
                            occupancy.append(np.zeros(len(self.t_space)))

                        for t_i in range(len(self.t_space)):
                            #magnitude[t_i] = np.sqrt(square_norm(self.basis_evol[b_i][t_i][a], self.S))
                            for m in range(self.M-1):
                                occupancy[m][t_i] += square_mag(self.basis_evol[b_i][t_i][a][m]) / (1 + np.sum(square_mag(self.basis_evol[b_i][t_i][a])))
                            occupancy[self.M - 1][t_i] += 1 / (1 + np.sum(square_mag(self.basis_evol[b_i][t_i][a])))

                        # Plot
                        for m in range(self.M):
                            plt.plot(self.t_space, occupancy[m], label=f"$\\langle N_{m}^{{|{b_i}; {a}\\rangle}} \\rangle/S$", color = self.mode_colors[m][b_i])

                if self.is_solved:
                    for m in range(self.M):
                        cur_solution = []
                        for i in range(len(self.solution)):
                            cur_solution.append(self.solution[i][m])
                        plt.plot(self.t_space, cur_solution, linestyle = "dashed", label = f"theor. $\\langle N_{m+1} \\rangle/S$", color = self.mode_colors[m][-1])

            elif graph_list[i] == 'basis_phase_space':
                if self.M != 2:
                    print("  ERROR: Attempting to plot 'basis_phase_space' with unsuitable mode number.")
                    continue
                plt.title("Phase space of $|{z}\\}=z$")
                y_space = np.linspace(-8, 8, 20)
                x_space = np.linspace(-11, 11, 20)

                U = np.zeros((len(y_space), len(x_space)))
                V = np.zeros((len(y_space), len(x_space)))

                for x_i in range(len(x_space)):
                    for y_i in range(len(y_space)):
                        cur_z = np.array([x_space[x_i] + 1j * y_space[y_i]])
                        cur_z_dot = self.uncoupled_basis_y_dot(0, cur_z)[0]
                        U[y_i][x_i] = cur_z_dot.real
                        V[y_i][x_i] = cur_z_dot.imag

                plt.quiver(x_space, y_space, U, V)

                plt.xlabel("$\\Re\\left(z\\right)$")
                plt.ylabel("$\\Im\\left(z\\right)$")
                include_legend = False


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




    # -------------------------------------------------------------------------
    # ------------------------- Data storage methods --------------------------
    # -------------------------------------------------------------------------
    # Encoders and decoders for .csv headers
    def encode_basis_evol_header(self, b_i, n, m):
        # Returns the name of the column for the b_i-th basis, n-th vector, m-th parameter
        return("B" + str(b_i).zfill(self.B_fill) + "_z_" + str(n).zfill(self.N_fill) + "_" + str(m).zfill(self.M_fill))

    def encode_wavef_evol_header(self, b_i, n):
        # Returns the name of the column for the b_i-th basis, n-th decomposition coefficient
        return("B" + str(b_i).zfill(self.B_fill) + "_A_" + str(n).zfill(self.N_fill))

    def encode_solution_header(self, m):
        return("S_" + str(m).zfill(self.M_fill))


    # Loading and saving

    def save_data(self, data_groups = None):
        if data_groups is None:
            data_groups = list(bosonic_su_n.data_nodes.keys())
        # If they exist, stores the Hamiltonian tensors
        if "system" in data_groups:
            if self.is_phys_init:
                self.disk_jockey.commit_datum_bulk("config", {"M" : self.M, "S" : self.S})
                self.disk_jockey.commit_datum_bulk("H_A_func", inspect.getsource(self.H_A))
                self.disk_jockey.commit_datum_bulk("H_B_func", inspect.getsource(self.H_B))
            if self.is_wavef_init:
                self.disk_jockey.commit_datum_bulk("wavef_init", self.wavef_initial_wavefunction)
                self.disk_jockey.commit_metadatum("wavef_init", {"wavef_message" : self.wavef_message})
            if self.is_solved:
                header_row = ["t"]
                for m in range(self.M):
                    header_row.append(self.encode_solution_header(m))
                solution_rows = [header_row]
                for t_i in range(len(self.t_space)):
                    cur_row = [self.t_space[t_i]]
                    for m in range(self.M):
                        cur_row.append(self.solution[t_i][m])
                    solution_rows.append(cur_row)
                self.disk_jockey.commit_datum_bulk("fock_solution", solution_rows, header_row = True)
                self.disk_jockey.commit_metadatum("fock_solution", {"benchmark" : self.solution_benchmark})

        if "setup" in data_groups:
            if self.is_basis_init:
                basis_init_header_row = []
                for m in range(self.M - 1):
                    basis_init_header_row.append(f"z_{m}")
                basis_init_rows = [basis_init_header_row]

                for b_i in range(len(self.basis)):
                    for basis_vector in self.basis[b_i]:
                        basis_init_rows.append([])
                        for m in range(self.M - 1):
                            basis_init_rows[-1].append(basis_vector[m])
                self.disk_jockey.commit_datum_bulk("basis_init", basis_init_rows, header_row = True)
                self.disk_jockey.commit_metadatum("basis_init", {"N" : self.N, "basis_config" : self.basis_config})

        if "solution" in data_groups:
            if self.is_basis_evol:
                header_row = ["t"]
                for b_i in range(len(self.basis)):
                    for n in range(self.N[b_i]):
                        for m in range(self.M - 1):
                            header_row.append(self.encode_basis_evol_header(b_i, n, m))
                basis_evol_rows = [header_row]

                for t_i in range(len(self.t_space)):
                    cur_row = [self.t_space[t_i]]
                    for b_i in range(len(self.basis)):
                        for n in range(self.N[b_i]):
                            for m in range(self.M - 1):
                                cur_row.append(self.basis_evol[b_i][t_i][n][m])
                    basis_evol_rows.append(cur_row)
                self.disk_jockey.commit_datum_bulk("basis_evol", basis_evol_rows, header_row = True)


            if self.is_wavef_evol:
                header_row = ["t"]
                for b_i in range(len(self.basis)):
                    for n in range(self.N[b_i]):
                        header_row.append(self.encode_wavef_evol_header(b_i, n))
                wavef_evol_rows = [header_row]

                for t_i in range(len(self.t_space)):
                    cur_row = [self.t_space[t_i]]
                    for b_i in range(len(self.basis)):
                        for n in range(self.N[b_i]):
                            cur_row.append(self.wavef_evol[b_i][t_i][n])
                    # TODO add physical descriptors such as <E> here?
                    wavef_evol_rows.append(cur_row)
                self.disk_jockey.commit_datum_bulk("wavef_evol", wavef_evol_rows, header_row = True)
                self.disk_jockey.commit_metadatum("wavef_evol", {"evol_benchmarks" : self.evol_benchmarks})

        self.disk_jockey.save_data(data_groups)

    def load_data(self, data_groups = None):
        if data_groups is None:
            data_groups = list(bosonic_su_n.data_nodes.keys())
        self.disk_jockey.load_data(data_groups)

        if "system" in data_groups:
            if self.disk_jockey.is_data_initialised["config"]:
                self.set_global_parameters(self.disk_jockey.data_bulks["config"]["M"], self.disk_jockey.data_bulks["config"]["S"])
                if self.disk_jockey.is_data_initialised["H_A_func"] and self.disk_jockey.is_data_initialised["H_B_func"]:
                    H_A_fn_body = self.disk_jockey.data_bulks["H_A_func"]
                    H_B_fn_body = self.disk_jockey.data_bulks["H_B_func"]
                    exec(H_A_fn_body) # This creates the function locally
                    H_A_fn_name = H_A_fn_body.split(" ", 1)[-1].split("(")[0]
                    self.H_A = eval(H_A_fn_name) # This refers to the function object
                    exec(H_B_fn_body) # This creates the function locally
                    H_B_fn_name = H_B_fn_body.split(" ", 1)[-1].split("(")[0]
                    self.H_B = eval(H_B_fn_name) # This refers to the function object
                    self.is_phys_init = True
                    print("  Hamiltonian tensors loaded.")
            if self.disk_jockey.is_data_initialised["wavef_init"]:
                wavef_message = self.disk_jockey.metadata["wavef_init"]["wavef_message"]
                wavef_initial_wavefunction = self.disk_jockey.data_bulks["wavef_init"]
                self.set_initial_wavefunction(wavef_initial_wavefunction, wavef_message)
            if self.disk_jockey.is_data_initialised["fock_solution"]:
                if not self.is_t_space_init:
                    self.t_space = [] # np.zeros(N_dtp)
                self.solution = [] # np.zeros((N_dtp, self.M))
                for row in self.disk_jockey.data_bulks["fock_solution"]:
                    if not self.is_t_space_init:
                        self.t_space.append(float(row["t"]))
                    # we append the empty ndarrays
                    self.solution.append(np.zeros(self.M))
                    for m in range(self.M):
                        self.solution[-1][m] = float(row[self.encode_solution_header(m)])
                self.is_solved = True
                self.is_t_space_init = True
                self.solution_benchmark = self.disk_jockey.metadata["fock_solution"]["benchmark"]
                print(f"  Solution ({len(self.solution)} datapoints) loaded.")

        if "setup" in data_groups:
            if self.disk_jockey.is_data_initialised["basis_init"]:
                self.N = self.disk_jockey.metadata["basis_init"]["N"]
                self.basis_config = self.disk_jockey.metadata["basis_init"]["basis_config"]
                self.basis = functions.cast_to_inhomogeneous_list(self.disk_jockey.data_bulks["basis_init"], self.N, functions.cast_dict_to_list, list_of_keys = [f"z_{m}" for m in range(self.M - 1)], convert_to_ndarray = True)
                # Create the identity operator matrices
                self.inverse_overlap_matrix = []

                for b_i in range(len(self.basis)):
                    X = np.zeros((self.N[b_i], self.N[b_i]), dtype = complex)
                    for m in range(self.N[b_i]):
                        for n in range(self.N[b_i]):
                            X[m][n] = overlap(self.basis[b_i][m], self.basis[b_i][n], self.S)
                    self.inverse_overlap_matrix.append(np.linalg.inv(X))
                self.is_basis_init = True
                for b_i in range(len(self.basis)):
                    if self.basis_config[b_i]["method"] == "gaussian":
                        print(f"  A sample of N = {self.N[b_i]} basis vectors drawn from a normal distribution of width {self.basis_config[b_i]["width"]} has been loaded.")

        if "solution" in data_groups:
            if self.disk_jockey.is_data_initialised["basis_evol"]:
                self.basis_evol = []
                if not self.is_t_space_init:
                    self.t_space = [] # np.zeros(N_dtp)
                    for basis_evol_row in self.disk_jockey.data_bulks["basis_evol"]:
                        self.t_space.append(basis_evol_row["t"])
                    self.is_t_space_init = True

                for b_i in range(len(self.basis)):
                    self.basis_evol.append([])
                    for basis_evol_row in self.disk_jockey.data_bulks["basis_evol"]:
                        cur_basis_evol = np.zeros((self.N[b_i], self.M-1), dtype=complex)
                        for n in range(self.N[b_i]):
                            for m in range(self.M - 1):
                                cur_basis_evol[n][m] = basis_evol_row[self.encode_basis_evol_header(b_i, n, m)]
                        self.basis_evol[b_i].append(cur_basis_evol)
                self.is_basis_evol = True
                print(f"  Basis evolution ({len(self.basis_evol[0])} datapoints) loaded.")
            if self.disk_jockey.is_data_initialised["wavef_evol"]:
                if not self.is_phys_init:
                    print("ERROR: You cannot load wavefunction evolution without loading or setting the physics configuration.")
                    return(-1)
                if not self.is_basis_init:
                    print("ERROR: You cannot load wavefunction evolution without loading the basis initialization.")
                    return(-1)

                if not self.is_t_space_init:
                    self.t_space = [] # np.zeros(N_dtp)
                    for wavef_evol_row in self.disk_jockey.data_bulks["wavef_evol"]:
                        self.t_space.append(wavef_evol_row["t"])
                    self.is_t_space_init = True
                self.wavef_evol = [] # [b_i] = np.zeros((N_dtp, self.N), dtype=complex)
                for b_i in range(len(self.basis)):
                    self.wavef_evol.append([])

                for wavef_evol_row in self.disk_jockey.data_bulks["wavef_evol"]:
                    for b_i in range(len(self.basis)):
                        # we append the empty ndarrays
                        self.wavef_evol[b_i].append(np.zeros(self.N[b_i], dtype = complex))

                        for n in range(self.N[b_i]):
                            self.wavef_evol[b_i][-1][n] = wavef_evol_row[self.encode_wavef_evol_header(b_i, n)]
                self.evol_benchmarks = self.disk_jockey.metadata["wavef_evol"]["evol_benchmarks"]
                self.is_wavef_evol = True
                print(f"  Wavefunction evolution ({len(self.t_space)} datapoints) loaded.")
        if self.is_t_space_init:
            self.t_space = np.array(self.t_space)

        self.summarise_data()



