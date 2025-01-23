import math
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


from copy import deepcopy
import time

import csv

from numerical_integration_methods import *


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
        self.t_space = []
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

    # ---------------------------------------------------------
    # ------------------- Physical methods --------------------
    # ---------------------------------------------------------

    def set_global_parameters(self, M, S):
        self.M = M
        self.S = S
        print("A driven Bose-Hubbard model with %i bosons over %i modes has been initialized." % (self.S, self.M))

    # The general SU(N) Hamiltonian is fully described by the tensors A_a,b and B_a,b,c,d

    def set_hamiltonian_tensors(self, A, B):
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

        self.A = candidate_A
        self.B = candidate_B

        self.is_phys_init = True

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
                    cur_overlap = overlap(candidate_basis[a], candidate_basis_vector, S)
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
    # ------------------- Operator methods --------------------
    # ---------------------------------------------------------

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
            self.wavef = candidate_initial_wavefunction
        elif message in ["aguiar"]:
            # initial_wavefunction describes an Aguiar unnormalized coherent state.
            candidate_initial_wavefunction = np.array(initial_wavefunction)
            if candidate_initial_wavefunction.shape != (self.M - 1,):
                print("ERROR: When using the 'aguiar' method, initial_wavefunction has to be an array of shape (M-1).")
                return(-1)
            self.wavef = self.decompose_aguiar(candidate_initial_wavefunction)
        elif message in ["grossmann", "frank"]:
            # initial_wavefunction describes a Grossmann normalized coherent state.
            candidate_initial_wavefunction = np.array(initial_wavefunction)
            if candidate_initial_wavefunction.shape != (self.M,):
                print("ERROR: When using the 'grossmann' method, initial_wavefunction has to be an array of shape (M).")
                return(-1)
            if np.round(np.sum(candidate_initial_wavefunction.real * candidate_initial_wavefunction.real + candidate_initial_wavefunction.imag * candidate_initial_wavefunction.imag), 2) != 1.0:
                print("ERROR: When using the 'grossmann' method, initial_wavefunction has to have norm 1.")
                return(-1)
            self.wavef = self.decompose_grossmann(candidate_initial_wavefunction)
        elif message == "":
            print("set_initial_wavefunction called without specifying the method of initialization.")
            print("Default option selected: initial wavefunction will be set to the first element of the basis set.")
            self.wavef = self.decompose_aguiar(self.basis[0])
        else:
            print("ERROR: Unknown method of wavefunction initialization.")
            return(-1)
        self.is_wavef_init = True
        return(0)


    # From the initialized basis and having inputted the initial wavefunction,
    # the user may choose to simulate the uncoupled basis evolution, the
    # coupled basis evolution, the evolution of the wavefunction decomposition
    # on top of both of these, or the full variational method evolution.
    # Each time, the previous "evolution" dataset is erased

    def simulate_uncoupled_basis(self, max_t, N_dtp, rtol = 1e-3, reg_timescale = -1, N_semaphor = 100):

        print("Simulating state evolution by propagating uncoupled basis vectors one by one and then evolving the decomposition.")
        if not self.is_basis_init:
            print("ERROR: You have attempted to simulate state evolution before sampling a basis set. You can do this by calling one of the many basis-sampling methods listed below.")
            for m, desc in bosonic_su_n.basis_sampling_methods.items():
                print(f"{m} [{desc}]")
            return(-1)
        if not self.is_wavef_init:
            print("ERROR: You have attempted to simulate state evolution before specifying the initial wavefunction state. You can do this by calling set_initial_wavefunction(initial_wavefunction, message).")
            return(-1)

        # We propagate each basis vector one by one
        print("  Propagating each basis vector one-by-one...")












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





