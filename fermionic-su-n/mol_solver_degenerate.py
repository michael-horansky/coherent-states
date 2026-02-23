import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from pyscf import gto, scf, cc, ao2mo, fci

from coherent_states.CS_Thouless import CS_Thouless
from coherent_states.CS_Qubit import CS_Qubit
from coherent_states.CS_sample import CS_sample

from class_Semaphor import Semaphor
from class_DSM import DSM
import functions


# this solver assumes that the occupancies in the spin-alpha and spin-beta subspaces are restricted separately.


# water molecule
mol = gto.Mole()
mol.build(
    atom = '''O 0 0 0; H  0 1 0; H 0 0 1''',
    basis = 'sto-3g')

# Notice: STO-type orbital basis is a linear combination of GTOs, which are a product of a real-valued radial function and the complex spherical harmonics.
# Since the two-body integrals have full rotational symmetry, and the radial dependence is given by a real-valued function, their values are always real.
# The single-body integrals reflect the angular dependence of the AOs, and as such will be complex-valued.



# For the AOs, PySCF uses Mulliken's notation: https://gqcg-res.github.io/knowdes/two-electron-integrals.html

# YES SYMMETRY

one_body_kin = mol.intor('int1e_kin', hermi = 1)
one_body_nuc = mol.intor('int1e_nuc', hermi = 1)
H_one_body = one_body_kin + one_body_nuc

# As for the s8 symmetry:
#   The exchange i <-> j, k <-> l is trivial (this is the s4 symmetry). This means we can always choose j >= i, l >= k.
H_ERI = mol.intor('int2e', aosym = "s8")

def get_nonascending_pairs(l):
    # for indices in l
    if isinstance(l, int):
        l = list(range(l))
    res = []
    for a in range(len(l)):
        for b in range(a + 1):
            res.append([l[a], l[b]])
    return(res)

ao_pairs = get_nonascending_pairs(mol.nao)
all_indices = get_nonascending_pairs(ao_pairs)

# The all_indices list is used as a translation between the s8-symmetrised flat array of int2e and the indices pqrs


class ground_state_solver():

    ###########################################################################
    # --------------- STATIC METHODS, CONSTRUCTORS, DESCRIPTORS ---------------
    ###########################################################################

    coherent_state_types = {
        "Thouless" : CS_Thouless,
        "Qubit" : CS_Qubit
        }

    data_nodes = {
        "system" : {"log" : "txt", "log_machine" : "plk"}, # User actions summary
        "molecule" : {"mol_structure" : "plk", "mode_structure" : "plk"}, # molecule properties
        "samples" : {"basis_samples" : "plk"}, # The specific samples, useful to reconstruct the ground state
        "results" : {"result_energy_states" : "plk"}, # The solution as found by the CS method
        "diagnostics" : {"diagnostic_log" : "txt"} # Condition numbers, eigenvalue min-max ratios, norms etc
    }

    def __init__(self, ID):

        self.ID = ID

        # Semaphor
        self.semaphor = Semaphor(time_format = "%H:%M:%S")

        # Data Storage Manager
        self.disk_jockey = DSM(f"outputs/{self.ID}")
        self.disk_jockey.create_data_nodes(ground_state_solver.data_nodes)

        self.user_log = f"initialised solver {self.ID}\n"
        self.diagnostics_log = []
        # The structure of the diagnostics log is like so: every element is either a list (for multiple same-level subprocedures) or a dict (for a header : content) pair

        # Self-analysis properties
        self.checklist = [] # List of succesfully performed actions

        self.reference_state_energy = None
        self.ci_energy = None
        self.ci_sol = None # We did not perform full CI

        self.SECS_eta = None
        self.SECS_energy = None

        print("---------------------------- " + str(ID) + " -----------------------------")


    ###########################################################################
    # --------------------------- Internal methods ----------------------------
    ###########################################################################


    # -------------------------------------------------------------------------
    # ---------------------------- Solver methods -----------------------------
    # -------------------------------------------------------------------------

    def std_i(self, i, j, n = 0, N = 1):
        # |i| = M - S labels the row, |j| = S labels the column, |n| = N labels the basis vector
        # output is a standardised index on the flattened array
        return(i * (self.S * N) + j * N + n)

    def encode_system(self, A, Z):
        # A is a list of N amplitudes
        # Z is a list of (M-S,S) matrices
        # Output is a standardised vector

        N = len(A)
        res = np.zeros(N * (1 + (self.M - self.S) * self.S), dtype=complex)

        # Amplitude
        res[:N] = A

        # Params
        for i in range(self.M - self.N):
            for j in range(self.M):
                for n in range(N):
                    res[N + self.std_i(i, j, n, N)] = Z[n][i][j]

        return(res)

    def decode_system(self, Q):
        N = int(len(Q) / (1 + (self.M - self.S) * self.S))
        A = np.zeros(N, dtype=complex)
        Z = np.zeros((N, self.M - self.S, self.S), dtype=complex)
        A = Q[:N]
        for i in range(self.M - self.N):
            for j in range(self.M):
                for n in range(N):
                    Z[n][i][j] = res[N + self.std_i(i, j, n, N)]
        return(A, Z)

    """def Q_dot(self, t, Q, reg_timescale = -1):

        # Full-variational method

        # First, we decode the current Q-array
        cur_A, cur_Z = self.decode_system(Q)
        N = len(cur_A)

        # Now we find all zeroth, first, and second-order transition mels
        X_zero = np.zeros((N, N), dtype=complex) # [n_a][n_b]
        X_one = np.zeros((self.M - self.S, self.S, N, N), dtype=complex) # [i][j][n_a][n_b]
        X_two = np.zeros((self.M - self.S, self.S, self.M - self.S, self.S, N, N), dtype=complex) # [i][j][i'][j'][n_a][n_b]

        for n_a in range(N):
            for n_b in range(N):
                # zeroth-order overlap
                X_zero[n_a][n_b] = self.general_overlap(cur_Z[n_a], cur_Z[n_b], [], [])

                for j_a in range(self.S):
                    for j_b"""
                # first-order overlap


    # ---------------------------- Full CI methods ----------------------------

    def get_all_slater_determinants_with_fixed_S(self, M, S):
        res = [ [1] * S + [0] * (M - S) ]
        while(True):
            next_state = functions.choose_iterator(res[-1])
            if next_state is None:
                break
            res.append(next_state)

        """is_iteration_running = True

        while(is_iteration_running):
            number_of_passed_zeros = 0
            pointer_index = 0

            while(res[-1][pointer_index] == 0):
                number_of_passed_zeros += 1
                pointer_index += 1

                if number_of_passed_zeros == M - S:
                    # We reached the end
                    is_iteration_running = False
                    break
            if is_iteration_running:
                while(res[-1][pointer_index] == 1):
                    pointer_index += 1
                next_basis_state = [1] * (pointer_index - 1 - number_of_passed_zeros) + [0] * (number_of_passed_zeros + 1) + [1] + res[-1][pointer_index+1:]
                res.append(next_basis_state)"""
        return(res)


    def get_full_basis(self, trim_M = None):
        # Returns a list containing all occupancy basis states represented as
        # two-item lists, with each element describing the occupancy in one
        # spin subspace.

        # if trim_M is set to an int S <= trim_M < mol.nao, we will restrict
        # the occupancy basis to the lowest trim_M MOs.

        if trim_M is None:
            M = self.mol.nao
        else:
            M = trim_M
        S_A = self.S_alpha
        S_B = self.S_beta

        full_basis_A = self.get_all_slater_determinants_with_fixed_S(M, S_A)
        full_basis_B = self.get_all_slater_determinants_with_fixed_S(M, S_B)

        """full_basis_A = [ [1] * S_A + [0] * (M - S_A) ]

        is_iteration_running = True

        while(is_iteration_running):
            number_of_passed_zeros = 0
            pointer_index = 0

            while(full_basis_A[-1][pointer_index] == 0):
                number_of_passed_zeros += 1
                pointer_index += 1

                if number_of_passed_zeros == M - S_A:
                    # We reached the end
                    is_iteration_running = False
                    break
            if is_iteration_running:
                while(full_basis_A[-1][pointer_index] == 1):
                    pointer_index += 1
                next_basis_state = [1] * (pointer_index - 1 - number_of_passed_zeros) + [0] * (number_of_passed_zeros + 1) + [1] + full_basis_A[-1][pointer_index+1:]
                full_basis_A.append(next_basis_state)

        full_basis_B = [ [1] * S_B + [0] * (M - S_B) ]

        is_iteration_running = True

        while(is_iteration_running):
            number_of_passed_zeros = 0
            pointer_index = 0

            while(full_basis_B[-1][pointer_index] == 0):
                number_of_passed_zeros += 1
                pointer_index += 1

                if number_of_passed_zeros == M - S_B:
                    # We reached the end
                    is_iteration_running = False
                    break
            if is_iteration_running:
                while(full_basis_B[-1][pointer_index] == 1):
                    pointer_index += 1
                next_basis_state = [1] * (pointer_index - 1 - number_of_passed_zeros) + [0] * (number_of_passed_zeros + 1) + [1] + full_basis_B[-1][pointer_index+1:]
                full_basis_B.append(next_basis_state)"""

        # Now we compose the full basis
        return(full_basis_A, full_basis_B)
        #full_basis = []
        #for i in range(len(full_basis

    def get_exchange_integral_on_occupancy(self, occ_a, occ_b, c = [], a = []):
        # here occ_a, occ_b are restricted to one spin subspace

        # We assume that c, a do not destroy their respective states

        # The Jordan-Wigner string is 1 if total number of occupied modes
        # lower than the annihilated mode if even; -1 if odd.

        # If we order a to be ascending, the jordan-wigner prefactor
        # associated with each operator is independent on the other
        # operators. This introduces the permutation signature.

        reduced_a = occ_a.copy()
        reduced_b = occ_b.copy()

        total_jordan_wigner_string = functions.permutation_signature(c) * functions.permutation_signature(a)
        for i in range(len(c)):
            # we find the jordan-wigner string of c[i]
            if sum(occ_a[:c[i]]) % 2 == 1:
                total_jordan_wigner_string *= -1
            if occ_a[c[i]] == 0:
                # destroys
                return(0.0)
            else:
                reduced_a[c[i]] = 0
        for i in range(len(a)):
            # we find the jordan-wigner string of a[i]
            if sum(occ_b[:a[i]]) % 2 == 1:
                total_jordan_wigner_string *= -1
            if occ_b[a[i]] == 0:
                # destroys
                return(0.0)
            else:
                reduced_b[a[i]] = 0

        # Now we check orthogonality
        for i in range(len(reduced_a)):
            if reduced_a[i] != reduced_b[i]:
                return(0.0)
        return(total_jordan_wigner_string)


    def get_H_overlap_on_occupancy(self, occ_a, occ_b):
        # occ_a, occ_b are two-lists containing one element from each spin
        # basis.

        M = len(occ_a[0]) # we assume this is equal to len(occ_a[1])

        alpha_overlap = self.get_exchange_integral_on_occupancy(occ_a[0], occ_b[0])
        beta_overlap = self.get_exchange_integral_on_occupancy(occ_a[1], occ_b[1])
        W_alpha = np.zeros((M, M), dtype=complex)
        W_beta = np.zeros((M, M), dtype=complex)
        H_one_term = 0.0
        # This is a sum over all mode pairs
        for p in range(M):
            for q in range(M):
                W_alpha[p][q] = self.get_exchange_integral_on_occupancy(occ_a[0], occ_b[0], [p], [q])
                W_beta[p][q]  = self.get_exchange_integral_on_occupancy(occ_a[1], occ_b[1], [p], [q])
                H_one_term += self.mode_exchange_energy([p], [q]) * W_alpha[p][q] * beta_overlap
                H_one_term += self.mode_exchange_energy([p], [q]) * W_beta[p][q] * alpha_overlap
        H_two_term = 0.0
        # equal spin
        c_pairs = functions.subset_indices(np.arange(M), 2)
        a_pairs = functions.subset_indices(np.arange(M), 2)
        upup = 0.0
        downdown = 0.0
        mixed = 0.0
        for c_pair in c_pairs:
            for a_pair in a_pairs:
                i = c_pair[0]
                j = c_pair[1]
                k = a_pair[0]
                l = a_pair[1]
                prefactor_same_spin = 1.0 * (self.mode_exchange_energy([i, j], [k, l]) - self.mode_exchange_energy([i, j], [l, k]))
                # alpha alpha
                H_two_term += prefactor_same_spin * self.get_exchange_integral_on_occupancy(occ_a[0], occ_b[0], [j, i], [l, k]) * beta_overlap
                # beta beta
                H_two_term += prefactor_same_spin * self.get_exchange_integral_on_occupancy(occ_a[1], occ_b[1], [j, i], [l, k]) * alpha_overlap
                upup += prefactor_same_spin * self.get_exchange_integral_on_occupancy(occ_a[0], occ_b[0], [j, i], [l, k]) * beta_overlap
                downdown += prefactor_same_spin * self.get_exchange_integral_on_occupancy(occ_a[1], occ_b[1], [j, i], [l, k]) * alpha_overlap
        # opposite spin
        for i in range(M):
            for j in range(M):
                for k in range(M):
                    for l in range(M):
                        prefactor = 0.5 * self.mode_exchange_energy([i, j], [k, l])
                        # alpha beta
                        H_two_term += prefactor * W_alpha[i][k] * W_beta[j][l]
                        # beta alpha
                        H_two_term += prefactor * W_alpha[j][l] * W_beta[i][k]
                        mixed += prefactor * W_alpha[i][k] * W_beta[j][l] + prefactor * W_alpha[j][l] * W_beta[i][k]
        #print(H_one_term, H_two_term)
        #print("Up-up:", upup)
        #print("Down-down:", downdown)
        #print("Mixed spin:", mixed)
        H_nuc = self.mol.energy_nuc() * alpha_overlap * beta_overlap
        return(H_one_term + H_two_term + H_nuc)

    def find_ground_state_on_full_ci(self, trim_M = None):
        if trim_M is None:
            M = self.mol.nao
        else:
            M = trim_M

        fb_A, fb_B = self.get_full_basis(trim_M)

        # We flatten the basis
        fb = []
        for i in range(len(fb_A)):
            for j in range(len(fb_B)):
                fb.append([fb_A[i], fb_B[j]])




        """print("TESTY NA OVERLAP")
        cur_occ_a = [1, 1, 1, 0, 0]
        cur_occ_b = [1, 1, 0, 1, 0]
        cur_c = [2]
        cur_a = [3]
        print(f"< {cur_occ_a} | ({cur_c})\\hc {cur_a} | {cur_occ_b} > = {self.get_exchange_integral_on_occupancy(cur_occ_a, cur_occ_b, cur_c, cur_a)}")"""

        H = np.zeros((len(fb), len(fb)), dtype=complex)

        msg = f"  Explicit Hamiltonian evaluation on a trimmed full CI"
        new_sem_ID = self.semaphor.create_event(np.linspace(0, len(fb), 100 + 1), msg)


        for i in range(len(fb)):
            for j in range(len(fb)):
                self.semaphor.update(new_sem_ID, i)
                H[i][j] = self.get_H_overlap_on_occupancy(fb[i], fb[j])

        self.semaphor.finish_event(new_sem_ID, "    Evaluation")

        energy_levels, energy_states = np.linalg.eig(H)
        ground_state_index = np.argmin(energy_levels)
        ground_state_energy = energy_levels[ground_state_index]

        return(ground_state_energy.real)





    def find_ground_state_sampling(self, **kwargs):
        # kwargs:
        #     -N: sample size
        #     -lamb: None for no conditioning, or positive float for the conditioning parameter
        #     -sampling_method: magic word for how the CS samples its parameter tensor
        #     -CS: Type of coherent state to use
        #     -assume_spin_symmetry: if true, |z_alpha> = |z_beta> for the entire sample. False by default

        if "assume_spin_symmetry" in kwargs:
            assume_spin_symmetry = kwargs["assume_spin_symmetry"]
        else:
            assume_spin_symmetry = False

        self.user_log += f"find_ground_state_sampling [N = {kwargs["N"]}, lambda = {kwargs["lamb"]}, sampling method = {kwargs["sampling_method"]}, assume spin symmetry = {assume_spin_symmetry}]\n"
        procedure_diagnostic = []


        print(f"Obtaining the ground state with the method \"random sampling\" [CS types = {kwargs["CS"]}, N = {kwargs["N"]}, lambda = {kwargs["lamb"]}, sampling method = {kwargs["sampling_method"]}, assume spin symmetry = {assume_spin_symmetry}]")

        # We sample around the HF null guess, i.e. Z = 0
        # We include one extra basis vector - the null point itself!
        cs_null_param = ground_state_solver.coherent_state_types[kwargs["CS"]].null_state(self.M, self.S)


        #if kwargs["weights"] is None:
        #    kwargs["weights"] = np.ones(cs_null_param.shape)

        """initial_vector_sample = np.zeros((1,) + cs_null_param.shape, dtype=complex)
        additional_vector_sample = np.random.normal(0.0, kwargs["delta"], (kwargs["N"],) + cs_null_param.shape)
        vector_sample = np.concatenate((initial_vector_sample, additional_vector_sample))
        #random_weights = np.concatenate((np.zeros((1,), dtype=complex), np.random.normal(0.0, kwargs["delta"], (kwargs["N"],))))"""
        N = kwargs["N"] + 1

        cur_CS_sample = [[ground_state_solver.coherent_state_types[kwargs["CS"]].null_state(self.mol.nao, self.S_alpha), ground_state_solver.coherent_state_types[kwargs["CS"]].null_state(self.mol.nao, self.S_beta)]]
        for i in range(kwargs["N"]):

            if assume_spin_symmetry:
                new_sample_state = ground_state_solver.coherent_state_types[kwargs["CS"]].random_state(self.mol.nao, self.S_alpha, kwargs["sampling_method"])
                cur_CS_sample.append([new_sample_state, new_sample_state])

            else:
                cur_CS_sample.append([
                    ground_state_solver.coherent_state_types[kwargs["CS"]].random_state(self.mol.nao, self.S_alpha, kwargs["sampling_method"]),
                    ground_state_solver.coherent_state_types[kwargs["CS"]].random_state(self.mol.nao, self.S_beta, kwargs["sampling_method"])
                    ])

        basis_samples_bulk = []
        for i in range(N):
            basis_samples_bulk.append([cur_CS_sample[i][0].z, cur_CS_sample[i][1].z])
        self.disk_jockey.commit_datum_bulk("basis_samples", basis_samples_bulk)

        # Firstly we find the normalisation coefficients
        """norm_coefs = np.zeros(N)
        norm_coefs[0] = 1.0
        for i in range(1, N):
            sig, mag = cur_CS_sample[i].log_norm_coef
            if (sig * np.exp(-mag / 2.0)).imag > 1e-04:
                procedure_diagnostic.append(f"basis norm not real!!")
            norm_coefs[i] = (sig * np.exp(-mag / 2.0)).real

        print("-- Normalisation coefs:", norm_coefs)

        procedure_diagnostic.append(f"basis norm max/min = {np.max(norm_coefs) / np.min(norm_coefs)}")"""

        overlap_matrix = np.zeros((N, N), dtype=complex) # [a][b] = <a|b>
        for i in range(N):
            for j in range(N):
                overlap_matrix[i][j] = cur_CS_sample[i][0].norm_overlap(cur_CS_sample[j][0]) * cur_CS_sample[i][1].norm_overlap(cur_CS_sample[j][1])
        #print("-- Overlap matrix:")
        #print(overlap_matrix)
        print(f"Overlap matrix condition number = {np.linalg.cond(overlap_matrix)}")

        # At what trim number is the condition number maximal?
        S_cond_trim_max = 1
        S_cond_trim_max_val = 1.0
        for N_trim in range(2, N + 1):
            cur_S_cond_trim_val = np.linalg.cond(overlap_matrix[:N_trim, :N_trim])
            if cur_S_cond_trim_val > S_cond_trim_max_val:
                S_cond_trim_max = N_trim
                S_cond_trim_max_val = cur_S_cond_trim_val
        procedure_diagnostic.append(f"overlap max condition val = {S_cond_trim_max_val} at N_eff = {S_cond_trim_max}")

        # We now diagonalise on the vector sample
        H_eff = np.zeros((N, N), dtype=complex)

        msg = f"  Explicit Hamiltonian evaluation"
        new_sem_ID = self.semaphor.create_event(np.linspace(0, N * (N + 1) / 2, 100 + 1), msg)

        for a in range(N):
            # Here the diagonal <Z_a|H|Z_a>
            cur_H_overlap = self.H_overlap(cur_CS_sample[a], cur_CS_sample[a])
            H_eff[a][a] = cur_H_overlap
            self.semaphor.update(new_sem_ID, a * (a + 1) / 2)

            # Here the off-diagonal, using the fact that H_eff is a Hermitian matrix
            for b in range(a):
                # We explicitly calculate <Z_a | H | Z_b>
                cur_H_overlap = self.H_overlap(cur_CS_sample[a], cur_CS_sample[b])
                H_eff[a][b] = cur_H_overlap
                H_eff[b][a] = np.conjugate(H_eff[a][b])
                self.semaphor.update(new_sem_ID, a * (a + 1) / 2 + b + 1)

        self.solution_benchmark = self.semaphor.finish_event(new_sem_ID, "    Evaluation")

        # H_eff diagonal terms
        print("------ Debug and diagnostics")

        for i in range(N):
            print(f"  Z_{i} = [ {repr(cur_CS_sample[i][0].z)}, {repr(cur_CS_sample[i][1].z)} ]")
        for i in range(N):
            print(f"  Z_{i} self-energy: {H_eff[i][i]}")
            if H_eff[i][i] < self.ci_energy:
                print("SMALLER THAN GROUND STATE???")

        def get_partial_sol(N_eff):
            print(f"-------------------- Configuration space = {N_eff} --------------------")
            H_eff_trimmed = H_eff[:N_eff, :N_eff]
            S_trimmed = overlap_matrix[:N_eff, :N_eff]

            if np.all(np.linalg.eigvals(S_trimmed) > 0):
                print("    Overlap matrix is positive definite")
            if np.all(np.round(S_trimmed, 3) - np.round(np.conjugate(S_trimmed.T), 3) == 0):
                print("    Overlap matrix is Hermitian")
            if np.all(np.round(H_eff_trimmed, 3) - np.round(np.conjugate(H_eff_trimmed.T), 3) == 0):
                print("    Effective Hamiltonian matrix is Hermitian")

            cond_S = np.linalg.cond(S_trimmed)
            print(f"    Overlap matrix condition number: {cond_S}")
            print(f"    Overlap matrix eigenvalues: {np.linalg.eigvals(S_trimmed)}")
            print(f"    Energy bound: {np.min(np.linalg.eigvals(H_eff_trimmed) / np.linalg.eigvals(S_trimmed))}")

            energy_levels, energy_states = sp.linalg.eigh(H_eff_trimmed, S_trimmed)
            ground_state_index = np.argmin(energy_levels)

            print("      Ground state as CS superposition:", energy_states[:, ground_state_index])
            ground_state_norm = 0.0
            direct_ground_state_energy = 0.0
            for i in range(N_eff):
                for j in range(N_eff):
                    ground_state_norm += np.conjugate(energy_states[i][ground_state_index]) * energy_states[j][ground_state_index] * S_trimmed[i][j]
                    direct_ground_state_energy += np.conjugate(energy_states[i][ground_state_index]) * energy_states[j][ground_state_index] * H_eff_trimmed[i][j]
            print(f"        Ground state norm = {ground_state_norm}")
            print(f"        Ground state direct energy calc = {direct_ground_state_energy} (after renorm: {direct_ground_state_energy / ground_state_norm})")
            print(f"        Compare to eigh val = {energy_levels[ground_state_index]}")

            return(energy_levels[ground_state_index])

        convergence_sols = []
        N_vals = []
        for N_eff_val in range(1, N + 1):
            N_vals.append(N_eff_val)
            convergence_sols.append(get_partial_sol(N_eff_val))

        self.disk_jockey.commit_datum_bulk("result_energy_states", [N_vals, convergence_sols])
        self.diagnostics_log.append({"find_ground_state_sampling" : procedure_diagnostic})

        return(N_vals, convergence_sols)

    def find_ground_state_manual(self, **kwargs):
        # kwargs:
        #     -sample: an instance of CS_sample

        cur_CS_sample = kwargs["sample"].basis
        N = kwargs["sample"].N

        self.user_log += f"find_ground_state_manual [N = {N}]\n"
        procedure_diagnostic = []


        print(f"Obtaining the ground state with the method \"manual sampling\" [N = {N}]")


        self.disk_jockey.commit_datum_bulk("basis_samples", kwargs["sample"].get_z_tensor())

        overlap_matrix = np.zeros((N, N), dtype=complex) # [a][b] = <a|b>
        for i in range(N):
            for j in range(N):
                overlap_matrix[i][j] = cur_CS_sample[i][0].norm_overlap(cur_CS_sample[j][0]) * cur_CS_sample[i][1].norm_overlap(cur_CS_sample[j][1])
        #print("-- Overlap matrix:")
        #print(overlap_matrix)
        print(f"Overlap matrix condition number = {np.linalg.cond(overlap_matrix)}")

        # At what trim number is the condition number maximal?
        S_cond_trim_max = 1
        S_cond_trim_max_val = 1.0
        for N_trim in range(2, N + 1):
            cur_S_cond_trim_val = np.linalg.cond(overlap_matrix[:N_trim, :N_trim])
            if cur_S_cond_trim_val > S_cond_trim_max_val:
                S_cond_trim_max = N_trim
                S_cond_trim_max_val = cur_S_cond_trim_val
        procedure_diagnostic.append(f"overlap max condition val = {S_cond_trim_max_val} at N_eff = {S_cond_trim_max}")

        # We now diagonalise on the vector sample
        H_eff = np.zeros((N, N), dtype=complex)

        msg = f"  Explicit Hamiltonian evaluation"
        new_sem_ID = self.semaphor.create_event(np.linspace(0, N * (N + 1) / 2, 100 + 1), msg)

        for a in range(N):
            # Here the diagonal <Z_a|H|Z_a>
            cur_H_overlap = self.H_overlap(cur_CS_sample[a], cur_CS_sample[a])
            H_eff[a][a] = cur_H_overlap
            self.semaphor.update(new_sem_ID, a * (a + 1) / 2)

            # Here the off-diagonal, using the fact that H_eff is a Hermitian matrix
            for b in range(a):
                # We explicitly calculate <Z_a | H | Z_b>
                cur_H_overlap = self.H_overlap(cur_CS_sample[a], cur_CS_sample[b])
                H_eff[a][b] = cur_H_overlap
                H_eff[b][a] = np.conjugate(H_eff[a][b])
                self.semaphor.update(new_sem_ID, a * (a + 1) / 2 + b + 1)

        self.solution_benchmark = self.semaphor.finish_event(new_sem_ID, "    Evaluation")

        # H_eff diagonal terms
        print("------ Debug and diagnostics")

        for i in range(N):
            print(f"  Z_{i} = [ {repr(cur_CS_sample[i][0].z)}, {repr(cur_CS_sample[i][1].z)} ]")
        for i in range(N):
            print(f"  Z_{i} self-energy: {H_eff[i][i]}")
            if H_eff[i][i] < self.ci_energy:
                print("SMALLER THAN GROUND STATE???")

        def get_partial_sol(N_eff):
            print(f"-------------------- Configuration space = {N_eff} --------------------")
            H_eff_trimmed = H_eff[:N_eff, :N_eff]
            S_trimmed = overlap_matrix[:N_eff, :N_eff]

            if np.all(np.linalg.eigvals(S_trimmed) > 0):
                print("    Overlap matrix is positive definite")
            if np.all(np.round(S_trimmed, 3) - np.round(np.conjugate(S_trimmed.T), 3) == 0):
                print("    Overlap matrix is Hermitian")
            if np.all(np.round(H_eff_trimmed, 3) - np.round(np.conjugate(H_eff_trimmed.T), 3) == 0):
                print("    Effective Hamiltonian matrix is Hermitian")

            cond_S = np.linalg.cond(S_trimmed)
            print(f"    Overlap matrix condition number: {cond_S}")
            print(f"    Overlap matrix eigenvalues: {np.linalg.eigvals(S_trimmed)}")
            print(f"    Energy bound: {np.min(np.linalg.eigvals(H_eff_trimmed) / np.linalg.eigvals(S_trimmed))}")

            energy_levels, energy_states = sp.linalg.eigh(H_eff_trimmed, S_trimmed)
            ground_state_index = np.argmin(energy_levels)

            print("      Ground state as CS superposition:", energy_states[:, ground_state_index])
            ground_state_norm = 0.0
            direct_ground_state_energy = 0.0
            for i in range(N_eff):
                for j in range(N_eff):
                    ground_state_norm += np.conjugate(energy_states[i][ground_state_index]) * energy_states[j][ground_state_index] * S_trimmed[i][j]
                    direct_ground_state_energy += np.conjugate(energy_states[i][ground_state_index]) * energy_states[j][ground_state_index] * H_eff_trimmed[i][j]
            print(f"        Ground state norm = {ground_state_norm}")
            print(f"        Ground state direct energy calc = {direct_ground_state_energy} (after renorm: {direct_ground_state_energy / ground_state_norm})")
            print(f"        Compare to eigh val = {energy_levels[ground_state_index]}")

            return(energy_levels[ground_state_index])

        convergence_sols = []
        N_vals = []
        for N_eff_val in range(1, N + 1):
            N_vals.append(N_eff_val)
            convergence_sols.append(get_partial_sol(N_eff_val))

        self.disk_jockey.commit_datum_bulk("result_energy_states", [N_vals, convergence_sols])
        self.diagnostics_log.append({"find_ground_state_manual" : procedure_diagnostic})

        return(N_vals, convergence_sols)

    def find_ground_state_SEGS_width(self, **kwargs):
        # kwargs:
        #     -N: Sample size
        #     -N_sub: Subsample size
        #     -CS: Type of CS to use. Not all CS types support SEGS!

        N = kwargs["N"]
        N_subsample = 1
        if "N_sub" in kwargs:
            N_subsample = kwargs["N_sub"]

        CS_type = "Thouless"
        if "CS" in kwargs:
            CS_type = kwargs["CS"]

        print(f"Obtaining the ground state with the method \"SEGS for dist width\" [N = {kwargs["N"]}, N_sub = {N_subsample}, CS = {CS_type}]")

        assert self.S_alpha == self.S_beta # So far only RHF!

        procedure_diagnostic = []

        print("Obtaining SEGS heatmap and reference energy...")
        cur_SECS_heatmap, cur_SECS_restricted_energy = self.solve_on_single_excitation_closed_shell()

        # We now manually sample Thouless states guided by the SECS heatmap
        cur_sample = CS_sample(self, ground_state_solver.coherent_state_types[CS_type], add_ref_state = True)


        # Here, depending on the CS type, we set up the normal distribution parameters
        if CS_type == "Thouless":
            shape_alpha = (self.mol.nao - self.S_alpha, self.S_alpha)
            shape_beta = (self.mol.nao - self.S_beta, self.S_beta)
            centres_alpha = np.zeros(shape_alpha)
            centres_beta = np.zeros(shape_beta)
            widths_alpha =  np.sqrt(cur_SECS_heatmap)
            widths_beta =  np.sqrt(cur_SECS_heatmap)
        elif CS_type == "Qubit":
            # mu_0 = (ref occupancy); std = sum(eta, axis)
            shape_alpha = (self.mol.nao,)
            shape_beta = (self.mol.nao,)
            centres_alpha = np.concatenate((
                        np.ones(self.S_alpha),
                        np.zeros(self.mol.nao - self.S_alpha)
                    ))
            centres_beta = np.concatenate((
                        np.ones(self.S_beta),
                        np.zeros(self.mol.nao - self.S_beta)
                    ))
            widths_alpha = np.concatenate((
                        np.sqrt(np.sum(cur_SECS_heatmap, axis = 1)),
                        np.sqrt(np.sum(cur_SECS_heatmap, axis = 0))
                    ))
            widths_beta = np.concatenate((
                        np.sqrt(np.sum(cur_SECS_heatmap, axis = 1)),
                        np.sqrt(np.sum(cur_SECS_heatmap, axis = 0))
                    ))

        N_vals = [1]
        convergence_sols = [self.reference_state_energy]

        msg = f"  Conditioned sampling with ground state search on {N} states, each taken from {N_subsample} random states"
        new_sem_ID = self.semaphor.create_event(np.linspace(0, N_subsample * ((N + 2) * (N + 1) / 2 - 1) + 1, 1000 + 1), msg)

        for n in range(N):
            # We add the best out of 10 random states
            cur_subsample = []
            for n_sub in range(N_subsample):
                #rand_z_alpha = ground_state_solver.coherent_state_types[CS_type](self.mol.nao, self.S_alpha, np.random.normal(centres_alpha, np.sqrt(cur_SECS_heatmap), shape_alpha))
                #rand_z_beta = ground_state_solver.coherent_state_types[CS_type](self.mol.nao, self.S_beta, np.random.normal(centres_beta, np.sqrt(cur_SECS_heatmap), shape_beta))
                rand_z_alpha = ground_state_solver.coherent_state_types[CS_type](self.mol.nao, self.S_alpha, np.random.normal(centres_alpha, widths_alpha, shape_alpha))
                rand_z_beta = ground_state_solver.coherent_state_types[CS_type](self.mol.nao, self.S_beta, np.random.normal(centres_beta, widths_beta, shape_beta))
                cur_subsample.append([rand_z_alpha, rand_z_beta])

            cur_sample.add_best_of_subsample(cur_subsample, semaphor_ID = new_sem_ID)
            N_vals.append(cur_sample.N)
            convergence_sols.append(cur_sample.E_ground[-1])

        procedure_diagnostic.append(f"Full sample condition number = {cur_sample.S_cond}")

        solution_benchmark = self.semaphor.finish_event(new_sem_ID, "    Evaluation")

        self.disk_jockey.commit_datum_bulk("result_energy_states", [N_vals, convergence_sols])
        self.diagnostics_log.append({"find_ground_state_SEGS_width" : procedure_diagnostic})

        return(N_vals, convergence_sols)

    def find_ground_state_SEGS_phase(self, **kwargs):
        # kwargs:
        #     -N: Sample size
        #     -N_sub: Subsample size

        print(f"Obtaining the ground state with the method \"SEGS with random phase\" [N = {kwargs["N"]}, N_sub = {kwargs["N_sub"]}]")

        assert self.S_alpha == self.S_beta


        procedure_diagnostic = []

        print("---------------- Sampling ")

        print("Obtaining SEGS heatmap and reference energy...")
        cur_SECS_heatmap, cur_SECS_restricted_energy = self.solve_on_single_excitation_closed_shell()

        # We now manually sample Thouless states guided by the SECS heatmap
        cur_sample = CS_sample(self, CS_Thouless, add_ref_state = True)

        shape_alpha = (self.mol.nao - self.S_alpha, self.S_alpha)
        shape_beta = (self.mol.nao - self.S_beta, self.S_beta)
        centres_alpha = np.zeros(shape_alpha)
        centres_beta = np.zeros(shape_beta)

        N = kwargs["N"]
        N_subsample = kwargs["N_sub"]

        N_vals = [1]
        convergence_sols = [self.reference_state_energy]

        msg = f"  Conditioned sampling with ground state search on {N} states, each taken from {N_subsample} random states"
        new_sem_ID = self.semaphor.create_event(np.linspace(0, N_subsample * ((N + 2) * (N + 1) / 2 - 1) + 1, 1000 + 1), msg)

        for n in range(N):
            # We add the best out of 10 random states
            cur_subsample = []
            for n_sub in range(N_subsample):
                #rand_z_alpha = CS_Thouless(self.mol.nao, self.S_alpha, np.random.normal(centres_alpha, np.sqrt(cur_SECS_heatmap), shape_alpha))
                #rand_z_beta = CS_Thouless(self.mol.nao, self.S_beta, np.random.normal(centres_beta, np.sqrt(cur_SECS_heatmap), shape_beta))
                rand_z_alpha = CS_Thouless(self.mol.nao, self.S_alpha, np.sqrt(cur_SECS_heatmap) * np.exp(1j * np.random.random(shape_alpha) * 2.0 * np.pi))
                rand_z_beta = CS_Thouless(self.mol.nao, self.S_beta, np.sqrt(cur_SECS_heatmap) * np.exp(1j * np.random.random(shape_beta) * 2.0 * np.pi))
                cur_subsample.append([rand_z_alpha, rand_z_beta])

            cur_sample.add_best_of_subsample(cur_subsample, semaphor_ID = new_sem_ID)
            N_vals.append(cur_sample.N)
            convergence_sols.append(cur_sample.E_ground[-1])

        procedure_diagnostic.append(f"Full sample condition number = {cur_sample.S_cond}")

        solution_benchmark = self.semaphor.finish_event(new_sem_ID, "    Evaluation")

        self.disk_jockey.commit_datum_bulk("result_energy_states", [N_vals, convergence_sols])
        self.diagnostics_log.append({"find_ground_state_SEGS_phase" : procedure_diagnostic})

        return(N_vals, convergence_sols)



    def find_ground_state_krylov(self, **kwargs):
        # kwargs:
        #     -dt: float; time spacing of the basis sampling process
        print("placeholder", kwargs["dt"])

        # The initial guess is the null matrix, i.e. all pi_1 modes occupied.




    def find_ground_state_imaginary_timeprop(self, **kwargs):
        # kwargs:
        #     -dt: float; timestep
        #     -tol: float; tolerance on the solution or smth
        print("placeholder", kwargs["tol"])


    find_ground_state_methods = {
            "sampling" : find_ground_state_sampling,
            "manual" : find_ground_state_manual, # manual sample
            "SEGS_width" : find_ground_state_SEGS_width, # uses SEGS for width of sample, correlated for UHF
            "SEGS_phase" : find_ground_state_SEGS_phase, # restricts the magnitudes by SEGS, only randomises phases of components
            "krylov" : find_ground_state_krylov,
            "imag_timeprop" : find_ground_state_imaginary_timeprop
        }


    ###########################################################################
    # ----------------------------- User methods ------------------------------
    ###########################################################################

    def initialise_molecule(self, mol):
        # mol is an instance of pyscf.gto.Mole

        if "mol_init" in self.checklist:
            print("Molecule already initialised.")
            return(None)

        print("Initialising molecule...")
        self.mol = mol
        self.M = self.mol.nao * 2
        self.S = self.mol.tot_electrons()

        nalpha, nbeta = self.mol.nelec
        self.S_alpha = nalpha
        self.S_beta = nbeta

        # We create helpful dictionaries to describe the molecule
        self.element_to_basis = [] # [element index] = [type of element, AO index start, AO index end (non-inclusive)]
        cur_aoslice = self.mol.aoslice_by_atom()
        for i in range(len(self.mol.elements)):
            self.element_to_basis.append([self.mol.elements[i], int(cur_aoslice[i][2]), int(cur_aoslice[i][3])])

        print(f"    There are {self.mol.nao} atomic orbitals, each able to hold 2 electrons of opposing spin.")
        print(f"    The molecule is occupied by {self.mol.tot_electrons()} electrons in total: {self.S_alpha} with spin alpha, {self.S_beta} with spin beta.")
        print(f"    The molecule consists of the following atoms: {self.mol.elements}")
        print(f"    The atomic orbitals are ordered as follows: {self.mol.ao_labels()}")
        print(f"    The nuclear repulsion energy is {self.mol.energy_nuc()}")
        print(self.element_to_basis)
        print(gto.charge("O"))

        print("  Calculating 1e integrals...")
        AO_H_one = self.mol.intor('int1e_kin', hermi = 1) + self.mol.intor('int1e_nuc', hermi = 1)

        print("  Calculating 2e integrals...")
        #self.H_two = self.mol.intor('int2e', aosym = "s1")
        AO_H_two_chemist = self.mol.intor('int2e', aosym = "s1")
        # <ij|kl> = (ik|jl)

        # We now construct AO_H_two as the coefficient tensor in second quantisation according to Szabo & Ostlund: Modern Quantum Chemistry p. 95, Eq. 2.232
        # O_2 = 0.5 * sum_ijkl <ij|kl> f\hc_i f\hc_j f_l f_k
        # Using <ij|kl> = (ik|jl) we have
        # O_ijkl = 0.5 (il|jk)
        # By symmetry: (ij|kl) = (ji|kl) = (ij|lk) = (ji|lk)
        # Hence O_ijkl = O_ljki = O_ikjl = O_lkji

        print("  Finding the molecular orbitals using mean-field approximations...")
        mean_field = scf.RHF(mol).run()
        self.MO_coefs = mean_field.mo_coeff
        self.reference_state_energy = mean_field.e_tot

        print("  Transforming 1e and 2e integrals to MO basis...")
        self.MO_H_one = np.matmul(self.MO_coefs.T, np.matmul(AO_H_one, self.MO_coefs))
        MO_H_two_packed = ao2mo.kernel(self.mol, self.MO_coefs)
        MO_H_two_chemist = ao2mo.restore(1, MO_H_two_packed, self.MO_coefs.shape[1]) # In mulliken notation: MO_H_two[p][q][r][s] = (pq|rs)

        self.MO_H_two = MO_H_two_chemist.transpose(0, 2, 1, 3)# - MO_H_two_chemist.transpose(0, 3, 1, 2)

        # We construct the second quantised space as occupied \oplus unoccupied orbitals
        occ_orbs = [i for i, o in enumerate(mean_field.mo_occ) if o > 0]
        # TODO note that by using RHF, we assume N_alpha = N_beta. We can generalise the process by using UHF,
        # but this would mean using separate MO coeffs for alpha and beta subspaces

        self.modes = occ_orbs.copy()
        for i in range(self.mol.nao):
            if i not in occ_orbs:
                self.modes.append(i) # already ordered by mol.mo_energy

        print("  ------------------ NULL STATE analysis ------------------")
        print("  Occupied orbitals:", occ_orbs)





        print("    Antisym tests")
        print("      Sym 0 <-> 3?", (np.round(self.MO_H_two - self.MO_H_two.transpose(3, 1, 2, 0), 2) == 0).all())
        print("      Sym 1 <-> 2", (np.round(self.MO_H_two - self.MO_H_two.transpose(0, 2, 1, 3), 2) == 0).all())

        null_state_energy = 0.0
        null_state_energy_one = 0.0
        for p in range(self.S_alpha):
            # Only occupied states contribute
            null_state_energy_one += 2 *  self.mode_exchange_energy([p], [p])
        print(f" Null state energy one = {null_state_energy_one}")
        null_state_energy_two = 0.0

        # antisym spin-orbital approach
        for p in range(self.S_alpha):
            for q in range(self.S_alpha):
                null_state_energy_two += 2 * MO_H_two_chemist[p][p][q][q] - MO_H_two_chemist[p][q][q][p]

        print(f" Null state energy two = {null_state_energy_two}")

        null_state_energy = null_state_energy_one + null_state_energy_two + self.mol.energy_nuc()
        print(f"  Energy of the null state = {null_state_energy}")
        #null_state_alpha = CS_Thouless(self.mol.nao, self.S_alpha, np.zeros((self.mol.nao - self.S_alpha, self.S_alpha), dtype=complex))
        #null_state_beta = CS_Thouless(self.mol.nao, self.S_beta, np.zeros((self.mol.nao - self.S_beta, self.S_beta), dtype=complex))
        null_state_alpha = CS_Qubit.null_state(self.mol.nao, self.S_alpha)
        null_state_beta = CS_Qubit.null_state(self.mol.nao, self.S_alpha)
        null_state = [null_state_alpha, null_state_beta]
        null_state_direct_self_energy = self.H_overlap(null_state, null_state)
        print(f"  Energy of the null state with the overlap method = {null_state_direct_self_energy}")

        self.user_log += f"initialise_molecule [M = {self.M}, S = {self.S}]"
        mol_structure_bulk = {
            "atom" : self.mol.atom,
            "basis" : self.mol.basis,
            "spin" : self.mol.spin
            }
        mode_structure_bulk = {
            "M" : self.M,
            "S" : self.S,
            "modes" : self.modes,
            "MO_H_one" : self.MO_H_one,
            "MO_H_two" : self.MO_H_two
            }
        self.disk_jockey.commit_datum_bulk("mol_structure", mol_structure_bulk)
        self.disk_jockey.commit_datum_bulk("mode_structure", mode_structure_bulk)

        self.checklist.append("mol_init")

    def pyscf_full_CI(self):
        if "pyscf_full_CI" in self.checklist:
            print("Full CI by PySCF already performed.")
            return(self.ci_energy)

        print("Performing SCF on full CI...")

        cisolver = fci.FCI(self.mol, self.MO_coefs)
        self.ci_energy, self.ci_sol = cisolver.kernel()

        print(f"  Ground state energy as calculated by SCF (full configuration) = {self.ci_energy}")
        return(self.ci_energy)




    def search_for_states(self, state_type, sampling_method, inclusion):
        found_states = 0
        while(True):
            cur_state = [
                ground_state_solver.coherent_state_types[state_type].random_state(self.M, self.S, sampling_method),
                ground_state_solver.coherent_state_types[state_type].random_state(self.M, self.S, sampling_method)
                ]
            if inclusion(cur_state):
                print("Success!")
                print(f"State found with z = [ {repr(cur_state[0].z)}, {repr(cur_state[1].z)} ]")
            found_states += 1
            print(f"Tried {found_states} states\r")

    def print_diagnostic_log(self):
        # returns the log as a string
        output = f"----------------- {self.ID} diagnostic log -----------------\n"
        cur_depth = 0
        tw = 4 # tab width
        def enter_node(element):
            nonlocal output, cur_depth
            if isinstance(element, list):
                for sub_element in element:
                    enter_node(sub_element)
            if isinstance(element, dict):
                for header, sub_element in element.items():
                    output += " " * tw * cur_depth + header + "\n"
                    cur_depth += 1
                    enter_node(sub_element)
                    cur_depth -= 1
            if isinstance(element, str):
                output += " " * tw * cur_depth + element + "\n"

        enter_node(self.diagnostics_log)
        return(output)

    def save_data(self):
        # Save user log
        self.disk_jockey.commit_datum_bulk("log", self.user_log)

        # save diagnostic
        self.disk_jockey.commit_datum_bulk("diagnostic_log", self.print_diagnostic_log())

    def load_data(self):
        # loads data using disk jockey
        pass

    def get_H_two_element(self, p, q, r, s):
        return(self.H_two[p][q][r][s])
        #return(self.H_two[int(p * (p * p * p + 2 * p * p + 3 * p + 2) / 8 + p * q * (p + 1) / 2 + q * (q + 1) / 2   + r * (r + 1) / 2 + s )])

    def mode_exchange_energy(self, m_i, m_f, debug = False):
        # This function translates mode indices to spatial MO indices and returns the coefficient tensor element

        # m_i/f are lists of either one mode index (single electron exchange) or two mode indices (two electron exchange)
        # the indices in m_i/f are SPATIAL, and are interpreted as acting on one spin state only.
        if len(m_i) == 1:
            return(self.MO_H_one[self.modes[m_i[0]]][self.modes[m_f[0]]])
        elif len(m_i) == 2:
            #if self.modes[m_i[0]][1] == self.modes[m_f[1]][1] and self.modes[m_i[1]][1] == self.modes[m_f[0]][1]:
            return(self.MO_H_two[self.modes[m_i[0]]][self.modes[m_i[1]]][self.modes[m_f[0]]][self.modes[m_f[1]]])
        return(0.0)

    def H_overlap(self, pair_a, pair_b):
        # This method calculates <Z_a | H | Z_b>, including the nuclear self-energy
        # here pair_a/b = [state_a/b with spin alpha, state_a/b with spin beta]

        alpha_overlap = pair_a[0].norm_overlap(pair_b[0])
        beta_overlap = pair_a[1].norm_overlap(pair_b[1])

        #print("lolll", alpha_overlap, beta_overlap)

        # To speed up cross-spin two-electron matrix elements, we prepare a matrix of all first-order sequence overlaps
        W_alpha = np.zeros((self.mol.nao, self.mol.nao), dtype=complex) # [i][j] = < alpha | f\hc_i f_j | alpha >
        W_beta = np.zeros((self.mol.nao, self.mol.nao), dtype=complex) # [i][j] = < beta | f\hc_i f_j | beta >

        H_one_term = 0.0
        # This is a sum over all mode pairs
        for p in range(self.mol.nao):
            for q in range(self.mol.nao):

                W_alpha[p][q] = pair_a[0].norm_overlap(pair_b[0], [p], [q])
                W_beta[p][q]  = pair_a[1].norm_overlap(pair_b[1], [p], [q])

                # alpha
                H_one_term += self.mode_exchange_energy([p], [q]) * W_alpha[p][q] * beta_overlap
                # beta
                H_one_term += self.mode_exchange_energy([p], [q]) * W_beta[p][q] * alpha_overlap

        H_two_term = 0.0

        # Same spin (sigma = theta)

        # This is a sum over pairs of strictly ascending mode pairs (for other cases we can use symmetry, which just becomes an extra factor here)
        """c_pairs = functions.subset_indices(np.arange(self.mol.nao), 2)
        a_pairs = functions.subset_indices(np.arange(self.mol.nao), 2)
        for c_pair in c_pairs:
            for a_pair in a_pairs:

                # <ij|kl> -> < c_pair[1], c_pair[0] | a_pair[0], a_pair[1] >

                # alpha (all 4 operators act on hilbert_alpha)
                H_two_term += 2 * self.mode_exchange_energy([c_pair[1], c_pair[0]], a_pair) * pair_a[0].norm_overlap(pair_b[0], c_pair, a_pair) * beta_overlap
                # beta (all 4 operators act on hilbert_beta)
                H_two_term += 2 * self.mode_exchange_energy([c_pair[1], c_pair[0]], a_pair) * pair_a[1].norm_overlap(pair_b[1], c_pair, a_pair) * alpha_overlap

        # Different spin (sigma != theta)

        for p in range(self.mol.nao):
            for q in range(self.mol.nao):
                for r in range(self.mol.nao):
                    for s in range(self.mol.nao):
                        H_two_term += self.mode_exchange_energy([q, p], [r, s]) * pair_a[0].norm_overlap(pair_b[0], [p], [r]) * pair_a[1].norm_overlap(pair_b[1], [q], [s])"""

        # The "no symmetry" approach
        """for i in range(self.mol.nao):
            for j in range(self.mol.nao):
                for k in range(self.mol.nao):
                    for l in range(self.mol.nao):
                        prefactor = 0.5 * self.mode_exchange_energy([i, j], [k, l])

                        # alpha alpha
                        H_two_term += prefactor * pair_a[0].norm_overlap(pair_b[0], [j, i], [l, k]) * beta_overlap

                        # beta beta
                        H_two_term += prefactor * pair_a[1].norm_overlap(pair_b[1], [j, i], [l, k]) * alpha_overlap

                        # alpha beta
                        H_two_term += prefactor * pair_a[0].norm_overlap(pair_b[0], [i], [k]) * pair_a[1].norm_overlap(pair_b[1], [j], [l])

                        # beta alpha
                        H_two_term += prefactor * pair_a[0].norm_overlap(pair_b[0], [j], [l]) * pair_a[1].norm_overlap(pair_b[1], [i], [k])"""

        """c_pairs = functions.subset_indices(np.arange(self.mol.nao), 2)
        a_pairs = functions.subset_indices(np.arange(self.mol.nao), 2)
        for c_pair in c_pairs:
            for a_pair in a_pairs:

                # <ij|kl> -> < c_pair[1], c_pair[0] | a_pair[0], a_pair[1] >

                i = c_pair[0]
                j = c_pair[1]
                k = a_pair[0]
                l = a_pair[1]




                prefactor_same_spin = 0.5 * (self.mode_exchange_energy([i, j], [k, l]) - self.mode_exchange_energy([j, i], [k, l]) - self.mode_exchange_energy([i, j], [l, k]) + self.mode_exchange_energy([j, i], [l, k]) )

                # alpha alpha
                H_two_term += prefactor_same_spin * pair_a[0].norm_overlap(pair_b[0], [j, i], [l, k]) * beta_overlap

                # beta beta
                H_two_term += prefactor_same_spin * pair_a[1].norm_overlap(pair_b[1], [j, i], [l, k]) * alpha_overlap

                # alpha beta
                H_two_term += (
                            0.5 * self.mode_exchange_energy([i, j], [k, l]) * pair_a[0].norm_overlap(pair_b[0], [i], [k]) * pair_a[1].norm_overlap(pair_b[1], [j], [l])
                            + 0.5 * self.mode_exchange_energy([j, i], [k, l]) * pair_a[0].norm_overlap(pair_b[0], [j], [k]) * pair_a[1].norm_overlap(pair_b[1], [i], [l])
                            + 0.5 * self.mode_exchange_energy([i, j], [l, k]) * pair_a[0].norm_overlap(pair_b[0], [i], [l]) * pair_a[1].norm_overlap(pair_b[1], [j], [k])
                            + 0.5 * self.mode_exchange_energy([j, i], [l, k]) * pair_a[0].norm_overlap(pair_b[0], [j], [l]) * pair_a[1].norm_overlap(pair_b[1], [i], [k])
                            )

                # beta alpha
                H_two_term += (
                            0.5 * self.mode_exchange_energy([i, j], [k, l]) * pair_a[0].norm_overlap(pair_b[0], [j], [l]) * pair_a[1].norm_overlap(pair_b[1], [i], [k])
                            + 0.5 * self.mode_exchange_energy([j, i], [k, l]) * pair_a[0].norm_overlap(pair_b[0], [i], [l]) * pair_a[1].norm_overlap(pair_b[1], [j], [k])
                            + 0.5 * self.mode_exchange_energy([i, j], [l, k]) * pair_a[0].norm_overlap(pair_b[0], [j], [k]) * pair_a[1].norm_overlap(pair_b[1], [i], [l])
                            + 0.5 * self.mode_exchange_energy([j, i], [l, k]) * pair_a[0].norm_overlap(pair_b[0], [i], [k]) * pair_a[1].norm_overlap(pair_b[1], [j], [l])
                            )

        # Now for the cross-terms
        ...
        """

        # equal spin
        c_pairs = functions.subset_indices(np.arange(self.mol.nao), 2)
        a_pairs = functions.subset_indices(np.arange(self.mol.nao), 2)
        upup = 0.0
        downdown = 0.0
        mixed = 0.0
        for c_pair in c_pairs:
            for a_pair in a_pairs:

                # <ij|kl> -> < c_pair[1], c_pair[0] | a_pair[0], a_pair[1] >

                i = c_pair[0]
                j = c_pair[1]
                k = a_pair[0]
                l = a_pair[1]

                prefactor_same_spin = 1.0 * (self.mode_exchange_energy([i, j], [k, l]) - self.mode_exchange_energy([i, j], [l, k]))

                # alpha alpha
                H_two_term += prefactor_same_spin * pair_a[0].norm_overlap(pair_b[0], [j, i], [l, k]) * beta_overlap

                # beta beta
                H_two_term += prefactor_same_spin * pair_a[1].norm_overlap(pair_b[1], [j, i], [l, k]) * alpha_overlap

                upup += prefactor_same_spin * pair_a[0].norm_overlap(pair_b[0], [j, i], [l, k]) * beta_overlap
                downdown += prefactor_same_spin * pair_a[1].norm_overlap(pair_b[1], [j, i], [l, k]) * alpha_overlap

        # opposite spin
        for i in range(self.mol.nao):
            for j in range(self.mol.nao):
                for k in range(self.mol.nao):
                    for l in range(self.mol.nao):
                        prefactor = 0.5 * self.mode_exchange_energy([i, j], [k, l])

                        # alpha beta
                        #H_two_term += prefactor * pair_a[0].norm_overlap(pair_b[0], [i], [k]) * pair_a[1].norm_overlap(pair_b[1], [j], [l])
                        H_two_term += prefactor * W_alpha[i][k] * W_beta[j][l]

                        # beta alpha
                        #H_two_term += prefactor * pair_a[0].norm_overlap(pair_b[0], [j], [l]) * pair_a[1].norm_overlap(pair_b[1], [i], [k])
                        H_two_term += prefactor * W_alpha[j][l] * W_beta[i][k]

                        mixed += prefactor * W_alpha[i][k] * W_beta[j][l] + prefactor * W_alpha[j][l] * W_beta[i][k]


        #print(H_one_term, H_two_term)
        #print("Up-up:", upup)
        #print("Down-down:", downdown)
        #print("Mixed spin:", mixed)

        H_nuc = self.mol.energy_nuc() * alpha_overlap * beta_overlap
        return(H_one_term + H_two_term + H_nuc)





    def find_ground_state(self, method, **kwargs):
        if method in self.find_ground_state_methods.keys():
            return(self.find_ground_state_methods[method](self, **kwargs))
        else:
            print(f"ERROR: Unknown ground state method {method}. Available methods: {self.find_ground_state_methods.keys()}")
            return(None)


    # ----- Methods to look at the answer in order to better the approach -----

    def print_ground_state(self):
        assert self.ci_sol is not None

        print("Printing ground state solution on the full CI...")
        print(self.ci_sol)

    def occ_list_to_occ_string(self, occ_list):
        # the string that cistring deals with is exactly like the occ_list, but
        # reversed and also a string
        if isinstance(occ_list, str):
            return(occ_list) # just to regularise user action
        res = ""
        for i in range(len(occ_list) - 1, -1, -1):
            res += str(occ_list[i])
        return(res)

    def occ_idx_to_occ_list(self, idx_alpha, idx_beta):
        occ_alpha = fci.cistring.addr2str(self.mol.nao, self.S_alpha, idx_alpha)
        occ_beta = fci.cistring.addr2str(self.mol.nao, self.S_beta, idx_beta)
        # These are just integers corresponding to the binary value
        occ_alpha_bin = "{0:b}".format(occ_alpha)
        occ_beta_bin = "{0:b}".format(occ_beta)
        # We're still missing leading zeros. Those are just tailing zeros in the list
        alpha_list = []
        for i in range(len(occ_alpha_bin) - 1, -1, -1):
            alpha_list.append(int(occ_alpha_bin[i]))
        alpha_list += [0] * (self.mol.nao - len(alpha_list))
        beta_list = []
        for i in range(len(occ_beta_bin) - 1, -1, -1):
            beta_list.append(int(occ_beta_bin[i]))
        beta_list += [0] * (self.mol.nao - len(beta_list))
        return(alpha_list, beta_list)

    def ground_state_component(self, alpha_occupancy, beta_occupancy):
        assert self.ci_sol is not None
        # alpha/beta_occupancy is an array [occ1, occ2... occ_M] where N is the
        # number of alpha/beta MOs and occupancies sum up to S_alpha/beta.
        alpha_idx = fci.cistring.str2addr(self.mol.nao, self.S_alpha, self.occ_list_to_occ_string(alpha_occupancy))
        beta_idx = fci.cistring.str2addr(self.mol.nao, self.S_beta, self.occ_list_to_occ_string(beta_occupancy))

        # Access coefficient
        return(self.ci_sol[alpha_idx, beta_idx])

    # CSF projections

    def closed_shell_projection(self, trim_M = None):
        # Calculates the norm squared of the projection of the ground state
        # onto the closed-shell-only Hilbert subspace.

        # if trim_M is not None, we trim to the bottom trim_M MOs.
        # cistring "inserts leading zeros" to bitstrings :)
        act_M = self.mol.nao
        if trim_M is not None:
            act_M = min(trim_M, self.mol.nao)

        cur_state = [1] * self.S_alpha + [0] * (act_M - self.S_alpha)
        res = 0
        res_N = 0
        while(True):
            cur_c = self.ground_state_component(cur_state, cur_state)
            res += cur_c * cur_c
            res_N += 1
            cur_state = functions.choose_iterator(cur_state)
            if cur_state is None:
                break
        return(res, res_N)

    def get_prom_label(self, bitlist, trim_M = None, hr = False):
        # Returns a list [[de-occupied MOs], [promoted MOs]] from ref state
        # if hr, this is human-readable (i.e. MO labels are +1)

        cur_S = sum(bitlist)
        hr_cor = 0
        if hr:
            hr_cor = 1

        act_M = self.mol.nao
        if trim_M is not None:
            if trim_M > cur_S: # We need at least one empty shell
                act_M = min(trim_M, self.mol.nao)

        res = [[], []]
        for i in range(cur_S):
            if bitlist[i] == 0:
                res[0].append(i + hr_cor)
        for i in range(cur_S, act_M):
            if bitlist[i] == 1:
                res[1].append(i + hr_cor)

        return(res)


    def get_top_closed_shells(self, N_cs, trim_M = None):
        # Returns a list of top N_cs closed-shells and their squared norms

        act_M = self.mol.nao
        if trim_M is not None:
            if trim_M > self.S_alpha: # We need at least one empty shell
                act_M = min(trim_M, self.mol.nao)

        res = [] #[i] = [n_sq, bitlist]; ordered by n_sq desc.
        for i in range(N_cs):
            res.append([0, None, None])

        cur_state = [1] * self.S_alpha + [0] * (act_M - self.S_alpha)
        while(True):
            cur_c = self.ground_state_component(cur_state, cur_state)
            cur_n_sq = cur_c * cur_c

            if cur_n_sq > res[-1][0]:
                # belongs to the list
                new_i = len(res)
                while(cur_n_sq > res[new_i - 1][0]):
                    new_i -= 1
                    if new_i == 0:
                        break
                res.insert(new_i, [cur_n_sq, cur_state.copy()])
                res.pop()


            cur_state = functions.choose_iterator(cur_state)
            if cur_state is None:
                break
        return(res)

    def single_excitation_singlets_projection(self, trim_M = None):
        # For every occupancy list in alpha, L_A, we consider all occupancy
        # lists L_B for which exactly one electron is promoted to a
        # higher-index unoccupied orbital. Then
        #   CSF singlet = (L_A x L_B + L_B x L_A) / sqrt(2)

        act_M = self.mol.nao
        if trim_M is not None:
            act_M = min(trim_M, self.mol.nao)

        res = 0.0
        number_of_states = 0

        cur_state = [1] * self.S_alpha + [0] * (act_M - self.S_alpha)

        while(True):
            for excited_electron_i in range(self.S_alpha):
                # we find the index of the electron
                j = -1
                found_electrons = 0
                while(found_electrons <= excited_electron_i):
                    j += 1
                    found_electrons += cur_state[j]
                # Now, j points at the n-th electron in cur_state. We can promote to any higher index
                for k in range(j + 1, act_M):
                    if cur_state[k] == 1:
                        # Already occupied
                        continue
                    second_state = cur_state.copy()
                    second_state[j] = 0
                    second_state[k] = 1
                    full_ci_coef = (self.ground_state_component(cur_state, second_state) + self.ground_state_component(second_state, cur_state)) / np.sqrt(2)
                    res += full_ci_coef * full_ci_coef
                    number_of_states += 1
            cur_state = functions.choose_iterator(cur_state)
            if cur_state is None:
                break
        return(res, number_of_states)

    def single_excitation_closed_shell_heatmap(self, trim_M = None):
        # For closed-shell states which differ in shell occupancy from the
        # reference state in only one place, finds heatmap of norms squared as
        # a function of (de-occupied shell, newly-occupied shell).
        # Returns a (M - S, S) ndarray

        act_M = self.mol.nao
        if trim_M is not None:
            if trim_M > self.S_alpha: # We need at least one empty shell
                act_M = min(trim_M, self.mol.nao)

        res = np.zeros((act_M - self.S_alpha, self.S_alpha))

        ref_state = [1] * self.S_alpha + [0] * (act_M - self.S_alpha)

        for a in range(act_M - self.S_alpha):
            for b in range(self.S_alpha):
                cur_state = ref_state.copy()
                cur_state[self.S_alpha + a] = 1
                cur_state[b] = 0
                cur_c = self.ground_state_component(cur_state, cur_state)
                res[a][b] = cur_c * cur_c

        return(res)

    def plot_single_excitation_closed_shell_heatmap(self, ax = None, trim_M = None):

        act_M = self.mol.nao
        if trim_M is not None:
            if trim_M > self.S_alpha: # We need at least one empty shell
                act_M = min(trim_M, self.mol.nao)

        one_exc_closed_shell_hm = self.single_excitation_closed_shell_heatmap(trim_M)

        if ax is None:
            ax = plt.gca()

        # Plot the heatmap
        heatmap = ax.imshow(one_exc_closed_shell_hm, cmap='Wistia', interpolation='none') # extent = functions.m_ext(one_exc_closed_shell_hm)


        row_lab = [f"{i + one_exc_closed_shell_hm.shape[1] + 1}" for i in range(one_exc_closed_shell_hm.shape[0])]
        col_lab = [f"{i + 1}" for i in range(one_exc_closed_shell_hm.shape[1])]

        # Create colorbar
        cbar = ax.figure.colorbar(heatmap, ax=ax)
        cbar.ax.set_ylabel("Norm sq. of sol. component", rotation=-90, va="bottom")

        ax.set_xlabel("MO being promoted from")
        ax.set_ylabel("MO being promoted into")

        ax.set_xticks(np.arange(one_exc_closed_shell_hm.shape[1]), labels=col_lab)
        ax.set_yticks(np.arange(one_exc_closed_shell_hm.shape[0]), labels=row_lab)

        # Grid
        ax.spines[:].set_visible(False)
        ax.set_xticks(np.arange(one_exc_closed_shell_hm.shape[1]+1)-.5, minor=True)
        ax.set_yticks(np.arange(one_exc_closed_shell_hm.shape[0]+1)-.5, minor=True)
        ax.grid(which="minor", color="b", linestyle='-', linewidth=3)
        ax.tick_params(which="minor", bottom=False, left=False)

        return(heatmap, cbar)

    # ------------------------- Approximation methods -------------------------

    def solve_on_single_excitation_closed_shell(self, trim_M = None):
        # returns heatmap, SECS energy

        # Takes a basis consisting of ref state and SECSs, finds effective H,
        # finds solution, and returns norm squared of solution component for
        # every single-excitation.
        # The goal is to approximate the true solution heatmap for the purpose
        # of guiding the CS sampling process.

        if "SECS_sol" in self.checklist:
            # Already solved
            return(self.SECS_eta, self.SECS_energy)


        assert(self.S_alpha == self.S_beta)

        act_M = self.mol.nao
        if trim_M is not None:
            if trim_M > self.S_alpha: # We need at least one empty shell
                act_M = min(trim_M, self.mol.nao)

        ref_state = [1] * self.S_alpha + [0] * (act_M - self.S_alpha)
        basis = [[ref_state, ref_state]]

        for a in range(act_M - self.S_alpha):
            for b in range(self.S_alpha):
                cur_state = ref_state.copy()
                cur_state[self.S_alpha + a] = 1
                cur_state[b] = 0
                basis.append([cur_state, cur_state])

        H = np.zeros((len(basis), len(basis)))
        msg = f"  Explicit Hamiltonian evaluation on SECS basis"


        new_sem_ID = self.semaphor.create_event(np.linspace(0, len(basis) * (len(basis) + 1) / 2, 100 + 1), msg)

        for a in range(len(basis)):
            # Here the diagonal <Z_a|H|Z_a>
            cur_H_overlap = self.get_H_overlap_on_occupancy(basis[a], basis[a])
            assert cur_H_overlap.imag < 1e-08
            H[a][a] = cur_H_overlap.real
            self.semaphor.update(new_sem_ID, a * (a + 1) / 2)

            # Here the off-diagonal, using the fact that H is a Hermitian matrix
            for b in range(a):
                # We explicitly calculate <Z_a | H | Z_b>
                cur_H_overlap = self.get_H_overlap_on_occupancy(basis[a], basis[b])
                assert cur_H_overlap.imag < 1e-08
                H[a][b] = cur_H_overlap.real
                H[b][a] = H[a][b] # np.conjugate(H[a][b])
                self.semaphor.update(new_sem_ID, a * (a + 1) / 2 + b + 1)

        self.semaphor.finish_event(new_sem_ID, "    Evaluation")

        energy_levels, energy_states = np.linalg.eig(H)
        ground_state_index = np.argmin(energy_levels)
        ground_state_energy = energy_levels[ground_state_index]
        ground_state_vector = energy_states[:,ground_state_index] # note that energy_states consist of column vectors, not row vectors

        print(f"Ground state energy: {ground_state_energy} (compare to full CI: {self.ci_energy})")

        # Now, for the heatmap
        res = np.zeros((act_M - self.S_alpha, self.S_alpha))
        i = 1
        for a in range(act_M - self.S_alpha):
            for b in range(self.S_alpha):
                sol_component = ground_state_vector[i]
                res[a][b] = sol_component * sol_component
                i += 1

        # Mark as solved
        self.checklist.append("SECS_sol")

        self.SECS_eta = res
        self.SECS_energy = ground_state_energy

        return(res, ground_state_energy)



    def plot_SECS_restricted_heatmap(self, ax = None, trim_M = None):

        act_M = self.mol.nao
        if trim_M is not None:
            if trim_M > self.S_alpha: # We need at least one empty shell
                act_M = min(trim_M, self.mol.nao)

        one_exc_closed_shell_hm, _ = self.solve_on_single_excitation_closed_shell(trim_M)

        if ax is None:
            ax = plt.gca()

        # Plot the heatmap
        heatmap = ax.imshow(one_exc_closed_shell_hm, cmap='Wistia', interpolation='none') # extent = functions.m_ext(one_exc_closed_shell_hm)


        row_lab = [f"{i + one_exc_closed_shell_hm.shape[1] + 1}" for i in range(one_exc_closed_shell_hm.shape[0])]
        col_lab = [f"{i + 1}" for i in range(one_exc_closed_shell_hm.shape[1])]

        # Create colorbar
        cbar = ax.figure.colorbar(heatmap, ax=ax)
        cbar.ax.set_ylabel("Norm sq. of sol. component", rotation=-90, va="bottom")

        ax.set_xlabel("MO being promoted from")
        ax.set_ylabel("MO being promoted into")

        ax.set_xticks(np.arange(one_exc_closed_shell_hm.shape[1]), labels=col_lab)
        ax.set_yticks(np.arange(one_exc_closed_shell_hm.shape[0]), labels=row_lab)

        # Grid
        ax.spines[:].set_visible(False)
        ax.set_xticks(np.arange(one_exc_closed_shell_hm.shape[1]+1)-.5, minor=True)
        ax.set_yticks(np.arange(one_exc_closed_shell_hm.shape[0]+1)-.5, minor=True)
        ax.grid(which="minor", color="b", linestyle='-', linewidth=3)
        ax.tick_params(which="minor", bottom=False, left=False)

        return(heatmap, cbar)





