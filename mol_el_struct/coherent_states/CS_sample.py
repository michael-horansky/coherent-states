# This class does not inherit from CS_base
# Instead, it provides a method to cook up your own sample of CS states and then pass an instance of this to a ground state solver

#from mol_solver_degenerate import ground_state_solver
from coherent_states.CS_base import CS_Base
import numpy as np
import scipy as sp

from project_management_utils.Correspondent import Correspondent

EPS = np.finfo(float).eps # machine precision. Used to score parameters by condition number.

def lowest_E_safe(H, S, tol = 1e-14):
    # Safe min eigenval solver
    # H : matrix whose eigenvalue we're solving
    # S : overlap matrix for basis
    try:
        return(sp.linalg.eigh(H, S, eigvals_only=True, subset_by_index=[0, 0])[0].real)
    except np.linalg.LinAlgError:
        w, U = np.linalg.eigh(S)
        keep = w > tol * w.max()
        X = U[:, keep] / np.sqrt(w[keep])
        return(np.linalg.eigvalsh(X.conj().T @ H @ X)[0].real)


class CS_sample:

    CS_sample_Correspondent = Correspondent({"ID" : "CS_Sample", "label" : "CS Sample", "color" : ("magenta", False)})

    def __init__(self, solver, CS_class, add_ref_state = True):
        # Solver is an instance of ground_state_solver
        # CS_class is a class which inherits from CS_base
        self.solver = solver
        self.CS_class = CS_class

        self.M = self.solver.N_MO
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

        # output trackers
        self.number_of_refinements = 0


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

    def create_CS_state(self, Z_line):
        # creates a spin-pair of CS states of the specified class.

        if isinstance(Z_line[0], list) or isinstance(Z_line[0], np.ndarray) and len(Z_line) == 2:
            # The first dimension is the spin-subspace division
            return([self.CS_class(self.M, self.S_alpha, Z_line[0]), self.CS_class(self.M, self.S_beta, Z_line[1])])
        else:
            # Z is flat
            return([self.CS_class(self.M, self.S_alpha, Z_line[:self.solver.N_MO]), self.CS_class(self.M, self.S_beta, Z_line[self.solver.N_MO:])])

    # Some more interesting methods, which may use the solver object and its methods

    """def add_best_of_subsample(self, subsample, max_cond = 1e8, update_semaphor = False, semaphor_offset = None, condition = 'H', reject_high_overlap = False):
        # Selects one vector from subsample which minimises ground state energy
        # when added to the sample
        # subsample can either be a list of basis vectors or a z tensor ndarray
        # condition is either H or S; the candidate which lowers the eigenvalue of the condition matrix the most is selected


        if update_semaphor and semaphor_offset is None:
            # we assume that every step has the same sized subsample
            if condition == "H":
                semaphor_offset = len(subsample) * ((self.N + 1) * self.N / 2.0 - 1) + 1
            if condition == "S":
                semaphor_offset = (self.N + 1) * self.N / 2.0

        aug_S = np.zeros((self.N + 1, self.N + 1), dtype=complex)
        aug_H = np.zeros((self.N + 1, self.N + 1), dtype=complex)

        aug_S[:self.N, :self.N] = self.S
        aug_H[:self.N, :self.N] = self.H

        min_ground_state = self.E_ground[self.N - 1]
        best_addition = None
        best_self_E = None
        best_S_part = None
        best_H_part = None
        best_addition_proj = 1.0

        S_inv = np.linalg.inv(self.S)
        new_overlaps = np.zeros(self.N, dtype = complex)

        do_we_create_samples = isinstance(subsample, np.ndarray)

        if condition == "H":

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

                if reject_high_overlap:
                    candidate_proj = np.dot(np.conjugate(aug_S[:self.N,self.N]), S_inv @ aug_S[:self.N,self.N])
                    if candidate_proj > best_addition_proj:
                        # We reject this state without calculating the new min E because it is not orthogonal enough
                        if update_semaphor:
                            CS_sample.CS_sample_Correspondent.update_semaphor_event(semaphor_offset + (i + 1) * (self.N + 1))
                        continue

                # We augment the Hamiltonian matrix
                for a in range(self.N):
                    aug_H[a][self.N] = self.solver.H_overlap(self.basis[a], candidate)

                    if update_semaphor:
                        CS_sample.CS_sample_Correspondent.update_semaphor_event(semaphor_offset + i * (self.N + 1) + a + 1)

                    aug_H[self.N][a] = np.conjugate(aug_H[a][self.N])
                aug_H[self.N][self.N] = self.solver.H_overlap(candidate, candidate)

                if update_semaphor:
                    CS_sample.CS_sample_Correspondent.update_semaphor_event(semaphor_offset + (i + 1) * (self.N + 1))

                # If candidate is (almost) linearly dependent on old basis, the
                # process breaks down. We firstly condition the sampling based
                # on the condition number of the overlap matrix

                # Now we find the new ground state energy
                energy_levels, _ = sp.linalg.eigh(aug_H, aug_S)
                ground_state_index = np.argmin(energy_levels)
                candidate_E_ground = energy_levels[ground_state_index]
                #assert candidate_E_ground <= self.E_ground[self.N - 1] # By Cauchy interlacing theorem
                #TODO this assertion fails due to high noise for poorly conditioned samples. Solve this!

                if candidate_E_ground < min_ground_state:
                    # This is the best so far
                    min_ground_state = candidate_E_ground
                    best_addition = candidate
                    best_self_E = aug_H[self.N][self.N]
                    best_S_part = aug_S[:,self.N].copy()
                    best_H_part = aug_H[:,self.N].copy()
                    if reject_high_overlap:
                        best_addition_proj = candidate_proj

        if condition == "S":

            smallest_proj = 1.0
            smallest_proj_i = -1

            for i in range(len(subsample)):
                if do_we_create_samples:
                    # We have to actually create the thing
                    candidate = [self.CS_class(self.M, self.S_alpha, subsample[i][0]), self.CS_class(self.M, self.S_beta, subsample[i][1])]
                else:
                    candidate = subsample[i]
                # We find the norm squared of the projection onto the existing sample

                for a in range(self.N):
                    new_overlaps[a] = self.basis[a][0].norm_overlap(candidate[0]) * self.basis[a][1].norm_overlap(candidate[1])

                new_proj = np.dot(np.conjugate(new_overlaps), S_inv @ new_overlaps)

                if new_proj < smallest_proj:
                    smallest_proj = new_proj
                    smallest_proj_i = i
                    best_S_part = np.array(new_overlaps)

            # Now we calculate the H_part
            if do_we_create_samples:
                best_addition = [self.CS_class(self.M, self.S_alpha, subsample[smallest_proj_i][0]), self.CS_class(self.M, self.S_beta, subsample[smallest_proj_i][1])]
            else:
                best_addition = subsample[smallest_proj_i]
            best_H_part = np.zeros(self.N, dtype = complex)
            for a in range(self.N):
                best_H_part[a] = self.solver.H_overlap(self.basis[a], best_addition)
                if update_semaphor:
                    CS_sample.CS_sample_Correspondent.update_semaphor_event(semaphor_offset + a + 1)
            best_self_E = aug_H[self.N][self.N] = self.solver.H_overlap(best_addition, best_addition)
            if update_semaphor:
                CS_sample.CS_sample_Correspondent.update_semaphor_event(semaphor_offset + self.N + 1)


        # We now formally add the best candidate to state properties
        self.add_basis_vector(best_addition, best_self_E, S_part = best_S_part, H_part = best_H_part)

        self.S_cond.append(np.linalg.cond(self.S))
        self.E_ground.append(min_ground_state)"""

    # ------------------ Candidate selection / conditioning -------------------

    # GENERAL STRUCTURED COMPARISON
    #   FEASIBILITY: A state is FEASIBLE iff the augmented overlap matrix cond.
    #                number doesn't exceed some parameter max_cond.
    # General comparison works like so:
    #   1. If only one of the two candidates is feasible, that candidate wins.
    #   2. If both are feasible, the candidate associated with a lower ground
    #      state energy value wins.
    #   3. If both are infeasible, the ground state energy calculation may be
    #      misleading. The less ill-conditioned candidate wins. This also helps
    #      expanding the span of the basis, possibly beating late plateaus.
    def is_better(self, E_new, c_new, E_best, c_best, max_cond):
        # E_new is new candidate's g.s. energy
        # c_new is new candidate's augmented condition number
        # E_best, c_best are that for the currently stored best candidate.
        # max_cond is the
        new_ok, best_ok = c_new <= max_cond, c_best <= max_cond
        if new_ok != best_ok:
            return(new_ok)
        if new_ok:
            return(E_new < E_best)
        return(c_new < c_best)

    # SMOOTH HEURISTIC
    # An alternative approach, we do not need an arbitrary max_cond or a
    # max_tries value. We define a function which combines both g.s. E value
    # and S cond number into one score scalar. The candidate minimising this
    # score wins.
    # The score is based on the error on the machine calculation of the energy
    # eigenvalue. For an almost linearly dependent sample, the error is high,
    # and thus a small energy eigenvalue is less meaningful.
    def score(self, E, cond, k = 10.0):
        # k is a safety parameter.
        # E has a general offset. We get rid of that by subtracting the ref
        # state energy.
        E_no_offset = E - self.E_ground[0]
        return(E_no_offset + k * EPS * cond * abs(E_no_offset)) # The lower the score, the better.

    def add_best_of_subsample(self, sampler_inst, min_tries = 1, max_tries = 100, max_cond = 1e8, S_ID = None, semaphor_offset = None, heuristic = "struct", reject_high_overlap = False):
        # Selects one vector from subsample which minimises ground state energy
        # when added to the sample
        # subsample can either be a list of basis vectors or a z tensor ndarray
        # heuristic describes how we compare the candidates.


        if (S_ID is not None) and (semaphor_offset is None):
            # we assume that every step has the same sized subsample
            if heuristic in ["struct", "score"]:
                semaphor_offset = min_tries * ((self.N + 1) * self.N / 2.0 - 1) + 1
            if heuristic == "S":
                semaphor_offset = (self.N + 1) * self.N / 2.0

        aug_S = np.zeros((self.N + 1, self.N + 1), dtype=complex)
        aug_H = np.zeros((self.N + 1, self.N + 1), dtype=complex)

        aug_S[:self.N, :self.N] = self.S
        aug_H[:self.N, :self.N] = self.H

        min_ground_state = self.E_ground[self.N - 1]
        best_addition = None
        best_self_E = None
        best_S_part = None
        best_H_part = None
        best_addition_proj = 1.0
        best_S_cond = None

        S_inv = np.linalg.inv(self.S)
        new_overlaps = np.zeros(self.N, dtype = complex)

        def accept_candidate(candidate, candidate_E_ground, candidate_cond, candidate_proj):
            nonlocal min_ground_state
            nonlocal best_addition
            nonlocal best_self_E
            nonlocal best_S_part
            nonlocal best_H_part
            nonlocal best_addition_proj
            nonlocal best_S_cond

            # If candidate_E_ground is None, we are accepting an infeasible candidate. This means we do not calculate the energy; if it is None at the end, we calculate the energy for the selected infeasible candidate.
            # An infeasible candidate can never beat a feasible candidate, so once E_ground is calculated, it will never become None again

            best_addition = candidate
            best_S_cond = candidate_cond
            if reject_high_overlap:
                best_addition_proj = candidate_proj
            best_S_part = aug_S[:,self.N].copy()
            if candidate_E_ground is None:
                min_ground_state = None
            else:
                min_ground_state = candidate_E_ground
                best_self_E = aug_H[self.N][self.N]
                best_H_part = aug_H[:,self.N].copy()


        # We always consider the first sample as a potential best sample
        candidate = self.create_CS_state(sampler_inst.get_sample())
        for a in range(self.N):
            aug_S[a][self.N] = self.basis[a][0].norm_overlap(candidate[0]) * self.basis[a][1].norm_overlap(candidate[1])
            aug_S[self.N][a] = np.conjugate(aug_S[a][self.N])
        aug_S[self.N][self.N] = 1.0

        for a in range(self.N):
            aug_H[a][self.N] = self.solver.H_overlap(self.basis[a], candidate)

            if S_ID is not None:
                CS_sample.CS_sample_Correspondent.semaphor_progress(S_ID = S_ID)

            aug_H[self.N][a] = np.conjugate(aug_H[a][self.N])
        aug_H[self.N][self.N] = self.solver.H_overlap(candidate, candidate).real

        if S_ID is not None:
            CS_sample.CS_sample_Correspondent.semaphor_progress(S_ID = S_ID)

        # Now we find the new ground state energy
        candidate_E_ground = lowest_E_safe(aug_H, aug_S)

        accept_candidate(candidate, candidate_E_ground, np.linalg.cond(aug_S), np.dot(np.conjugate(aug_S[:self.N,self.N]), S_inv @ aug_S[:self.N,self.N]))

        N_tries = 0 # Number of attempts. We won't terminate until at least min_tries.

        def semaphor_record_work(dtau = 1):
            if S_ID is not None:
                if N_tries < min_tries:
                    CS_sample.CS_sample_Correspondent.semaphor_progress(dtau, S_ID = S_ID)
                else:
                    CS_sample.CS_sample_Correspondent.semaphor_overtime(dtau, S_ID = S_ID)


        while(N_tries < min_tries or (best_S_cond > max_cond and N_tries < max_tries)):
            N_tries += 1
            candidate = self.create_CS_state(sampler_inst.get_sample())

            if heuristic == "struct":
                # We augment the overlap matrix
                for a in range(self.N):
                    aug_S[a][self.N] = self.basis[a][0].norm_overlap(candidate[0]) * self.basis[a][1].norm_overlap(candidate[1])
                    aug_S[self.N][a] = np.conjugate(aug_S[a][self.N])
                aug_S[self.N][self.N] = 1.0

                candidate_proj = None
                """if reject_high_overlap:
                    candidate_proj = np.dot(np.conjugate(aug_S[:self.N,self.N]), S_inv @ aug_S[:self.N,self.N])
                    if candidate_proj > best_addition_proj:
                        # We reject this state without calculating the new min E because it is not orthogonal enough
                        if update_semaphor:
                            CS_sample.CS_sample_Correspondent.update_semaphor_event(semaphor_offset + (min(N_tries, min_tries) + 1) * (self.N + 1))
                        continue"""

                candidate_cond = np.linalg.cond(aug_S)
                if not np.isfinite(candidate_cond):
                    candidate_cond = np.inf

                candidate_feasible = candidate_cond <= max_cond
                best_feasible = best_S_cond <= max_cond

                if not candidate_feasible:
                    semaphor_record_work(self.N + 1)

                    if best_feasible or candidate_cond >= best_S_cond:
                        # cannot win: reject
                        continue
                    # infeasible candidate wins
                    accept_candidate(candidate, None, candidate_cond, candidate_proj)
                    continue

                # candidate is feasible. We calculate the new H matrix

                # We augment the Hamiltonian matrix
                for a in range(self.N):
                    aug_H[a][self.N] = self.solver.H_overlap(self.basis[a], candidate)

                    semaphor_record_work()

                    aug_H[self.N][a] = np.conjugate(aug_H[a][self.N])
                aug_H[self.N][self.N] = self.solver.H_overlap(candidate, candidate).real

                semaphor_record_work()

                # Now we find the new ground state energy
                candidate_E_ground = lowest_E_safe(aug_H, aug_S)
                #assert candidate_E_ground <= self.E_ground[self.N - 1] # By Cauchy interlacing theorem

                if (not best_feasible):
                    # Candidate is feasible and best is infeasible. We must accept.
                    accept_candidate(candidate, candidate_E_ground, candidate_cond, candidate_proj)
                    continue

                if candidate_E_ground < min_ground_state:
                    # Both feasible, new has better energy
                    accept_candidate(candidate, candidate_E_ground, candidate_cond, candidate_proj)

            elif heuristic == "score":
                for a in range(self.N):
                    aug_S[a][self.N] = self.basis[a][0].norm_overlap(candidate[0]) * self.basis[a][1].norm_overlap(candidate[1])
                    aug_S[self.N][a] = np.conjugate(aug_S[a][self.N])
                aug_S[self.N][self.N] = 1.0

                candidate_cond = np.linalg.cond(aug_S)
                if not np.isfinite(candidate_cond):
                    # infinite condition number = reject
                    semaphor_record_work(self.N + 1)
                    continue

                # We augment the Hamiltonian matrix
                for a in range(self.N):
                    aug_H[a][self.N] = self.solver.H_overlap(self.basis[a], candidate)

                    semaphor_record_work()

                    aug_H[self.N][a] = np.conjugate(aug_H[a][self.N])
                aug_H[self.N][self.N] = self.solver.H_overlap(candidate, candidate).real

                semaphor_record_work()

                candidate_E_ground = lowest_E_safe(aug_H, aug_S)

                if self.score(candidate_E_ground, candidate_cond) < self.score(min_ground_state, best_S_cond):
                    accept_candidate(candidate, candidate_E_ground, candidate_cond, None)

            elif heuristic == "S":
                raise Exception("No!")


        if best_S_cond > max_cond:
            # We exited the loop because we reached a hard cap on max tries. The current candidate is infeasible. We calculate its energy.
            CS_sample.CS_sample_Correspondent.warning(f"Sampler exited sampling due to hard cap on max tries reached. Current condition number: {best_S_cond} (> {max_cond})")

            if min_ground_state is None:
                aug_S[:, self.N] = best_S_part
                aug_S[self.N, :self.N] = np.conjugate(best_S_part[:self.N])
                aug_S[self.N, self.N] = 1.0

                for a in range(self.N):
                    aug_H[a][self.N] = self.solver.H_overlap(self.basis[a], best_addition)
                    aug_H[self.N][a] = np.conjugate(aug_H[a][self.N])
                aug_H[self.N][self.N] = self.solver.H_overlap(best_addition, best_addition).real

                candidate_E_ground = lowest_E_safe(aug_H, aug_S)

                min_ground_state = candidate_E_ground
                best_self_E = aug_H[self.N][self.N]
                best_H_part = aug_H[:,self.N].copy()


        # We now formally add the best candidate to state properties
        self.add_basis_vector(best_addition, best_self_E, S_part = best_S_part, H_part = best_H_part)

        self.S_cond.append(best_S_cond)
        self.E_ground.append(min_ground_state)

    """def refine(self, subsample, min_orth = 1e-6):

        def red_i(i, k):
            # i is the removed candidate
            # k is the reduced index of another basis vector
            if k < i:
                return(k)
            return(k + 1)

        do_we_create_samples = isinstance(subsample[0], np.ndarray)

        min_ground_state = self.E_ground[self.N] # current best E_g

        # Now for each sample element we check the candidates
        for i in range(self.N):
            # We are trying to replace state i here

            # We build the inverted partial S
            cur_S_part = np.delete(np.delete(self.S, i, 1), i, 0)
            S_part_inv = np.linalg.inv(cur_S_part)

            candidate_E_vals = np.zeros(len(subsample[0]))

            for j in range(len(subsample[0])):
                if do_we_create_samples:
                    # We have to actually create the thing
                    candidate = [self.CS_class(self.M, self.S_alpha, subsample[i][j][0]), self.CS_class(self.M, self.S_beta, subsample[i][j][1])]
                else:
                    candidate = subsample[i][j]

                # We check whether orthogonal enough
                S_cur = np.zeros(self.N - 1, dtype = complex)
                for k in range(self.N - 1):
                    S_cur[k] = self.basis[red_i(i, k)][0].norm_overlap(candidate[0]) * self.basis[red_i(i, k)][1].norm_overlap(candidate[1])
                orth = 1.0 - np.conjugate(S_cur.T) @ S_part_inv @ S_cur

                if orth < min_orth:
                    continue
                    # TODO raise some counter here

                new_S = np.ndarray(self.S)
                for k in range(self.N - 1):
                    new_S[red_i(i, k)][i] = S_cur[k]
                    new_S[i][red_i(i, k)] = np.conjugate(S_cur[k])
                new_S[i][i] = 1.0

                # Orthogonal enough. Let's see the H effect.
                new_H = np.ndarray(self.H)
                for k in range(self.N - 1):
                    new_H[red_i(i, k)][i] = self.solver.H_overlap(self.basis[red_i(i, k)], candidate)
                    new_H[i][red_i(i, k)] = np.conjugate(new_H[red_i(i, k)][i])
                new_H[i][i] = self.solver.H_overlap(candidate, candidate)

                energy_levels, _ = sp.linalg.eigh(new_H, new_S)
                ground_state_index = np.argmin(energy_levels)
                candidate_E_vals[j] = energy_levels[ground_state_index]

            best_candidate_i = np.argmin(candidate_E_vals)
            best_candidate_E = candidate_E_vals[best_candidate_i]

            if best_candidate_E < min_ground_state:
                # We accept the improvement
                if do_we_create_samples:
                    # We have to actually create the thing
                    self.basis[i] = [self.CS_class(self.M, self.S_alpha, subsample[i][best_candidate_i][0]), self.CS_class(self.M, self.S_beta, subsample[i][best_candidate_i][1])]
                else:
                    self.basis[i] = subsample[i][best_candidate_i]
                self.H = #TODO
    """


