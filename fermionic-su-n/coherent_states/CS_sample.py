# This class does not inherit from CS_base
# Instead, it provides a method to cook up your own sample of CS states and then pass an instance of this to a ground state solver

#from mol_solver_degenerate import ground_state_solver
from coherent_states.CS_base import CS_Base
import numpy as np
import scipy as sp

class CS_sample:

    def __init__(self, solver, CS_class, add_ref_state = True):
        # Solver is an instance of ground_state_solver
        # CS_class is a class which inherits from CS_base
        self.solver = solver
        self.CS_class = CS_class

        self.M = self.solver.mol.nao
        self.S_alpha = self.solver.S_alpha
        self.S_beta = self.solver.S_beta

        self.N = 0 # size of basis
        # The basis can be described in two ways: =
        #   1. As a (N, 2, ...) array of parameter tensors, where axis 0 determines basis vector and axis 1 determines spin
        #   2. As a list [[b_1^a, b_1^b], ...], where b_i^a/b is an instance of self.CS_class
        # Both shall be retrieved with methods get_z_tensor and get_basis, respectively
        self.basis = []

        # Calculated properties
        self.S = [] # S[a][b] = < a | b >
        self.H = [] # H[a][b] = < a | H | b >
        self.S_cond = [] # S_cond[n] = condition number of the overlap on the first n basis vectors
        self.E_ground = [] # E[n] = ground state energy restricted to the first n basis vectors

        # If not stated otherwise, we populate the sample with the reference state
        if add_ref_state:
            z_ref_alpha = self.CS_class.null_state(self.M, self.S_alpha)
            z_ref_beta = self.CS_class.null_state(self.M, self.S_beta)
            self.add_basis_vector([z_ref_alpha, z_ref_beta], self.solver.reference_state_energy, [], [])
            self.S_cond.append(1.0)
            self.E_ground.append(self.solver.reference_state_energy)


    def add_basis_vector(self, basis_vector, self_E = None, S_part = None, H_part = None):
        # basis_vector = [instance of CS_class, instance of CS_class]
        self.basis.append(basis_vector)

        # self_E = < b | H | b >
        # S_part[a] = < a | b > (length is previous value of N)
        # H_part[a] = < a | H | b > (length is previous value of N)
        # These can be left as None, but further calculations may be impossible
        # Note: we assume the basis vector is normalised

        self.S.append([])
        self.H.append([])

        # Add overlaps with existing basis vectors
        for a in range(self.N):
            self.S[a].append(S_part[a])
            self.S[self.N].append(np.conjugate(S_part[a]))
            self.H[a].append(H_part[a])
            self.H[self.N].append(np.conjugate(H_part[a]))
        # Now we add the self-energy to the corner of H
        self.S[self.N].append(1.0)
        self.H[self.N].append(self_E)

        self.N += 1

    def get_z_tensor(self):
        # Good for storing in files
        # If S_alpha != S_beta, this is inhomogeneous, so we store it as a list of lists of ndarrays
        z_tensor = []
        for n in range(self.N):
            z_tensor.append([self.basis[n][0].z, self.basis[n][1].z])
        return(z_tensor)

    # Some more interesting methods, which may use the solver object and its methods

    def add_best_of_subsample(self, subsample, max_cond = 1e8, semaphor_ID = None, semaphor_offset = None):
        # Selects one vector from subsample which minimises ground state energy
        # when added to the sample
        # subsample can either be a list of basis vectors or a z tensor ndarray


        if semaphor_ID is not None and semaphor_offset is None:
            # we assume that every step has the same sized subsample
            semaphor_offset = len(subsample) * ((self.N + 1) * self.N / 2.0 - 1) + 1

        aug_S = np.zeros((self.N + 1, self.N + 1), dtype=complex)
        aug_H = np.zeros((self.N + 1, self.N + 1), dtype=complex)

        aug_S[:self.N, :self.N] = self.S
        aug_H[:self.N, :self.N] = self.H

        min_ground_state = self.E_ground[self.N - 1]
        best_addition = None
        best_self_E = None
        best_S_part = None
        best_H_part = None

        do_we_create_samples = isinstance(subsample, np.ndarray)

        for i in range(len(subsample)):
            if do_we_create_samples:
                # We have to actually create the thing
                candidate = [self.CS_class(self.M, self.S_alpha, subsample[i][0]), self.CS_class(self.M, self.S_beta, subsample[i][1])]
            else:
                candidate = subsample[i]
            # We augment the overlap matrix
            for a in range(self.N):
                aug_S[a][self.N] = self.basis[a][0].norm_overlap(candidate[0]) * self.basis[a][1].norm_overlap(candidate[1])
                aug_S[self.N][a] = np.conjugate(aug_S[a][self.N])
            aug_S[self.N][self.N] = 1.0
            # We augment the Hamiltonian matrix
            for a in range(self.N):
                aug_H[a][self.N] = self.solver.H_overlap(self.basis[a], candidate)

                if semaphor_ID is not None:
                    self.solver.semaphor.update(semaphor_ID, semaphor_offset + i * (self.N + 1) + a + 1)

                aug_H[self.N][a] = np.conjugate(aug_H[a][self.N])
            aug_H[self.N][self.N] = self.solver.H_overlap(candidate, candidate)

            if semaphor_ID is not None:
                self.solver.semaphor.update(semaphor_ID, semaphor_offset + (i + 1) * (self.N + 1))

            # If candidate is (almost) linearly dependent on old basis, the
            # process breaks down. We firstly condition the sampling based
            # on the condition number of the overlap matrix

            # Now we find the new ground state energy
            energy_levels, _ = sp.linalg.eigh(aug_H, aug_S)
            ground_state_index = np.argmin(energy_levels)
            candidate_E_ground = energy_levels[ground_state_index]
            assert candidate_E_ground < self.E_ground[self.N - 1] # By Cauchy interlacing theorem

            if candidate_E_ground < min_ground_state:
                # This is the best so far
                min_ground_state = candidate_E_ground
                best_addition = candidate
                best_self_E = aug_H[self.N][self.N]
                best_S_part = aug_S[:,self.N].copy()
                best_H_part = aug_H[:,self.N].copy()

        # We now formally add the best candidate to state properties
        self.add_basis_vector(best_addition, best_self_E, S_part = best_S_part, H_part = best_H_part)

        self.S_cond.append(np.linalg.cond(self.S))
        self.E_ground.append(min_ground_state)


