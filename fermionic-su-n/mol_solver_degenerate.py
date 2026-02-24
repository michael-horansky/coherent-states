import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from pyscf import gto, scf, cc, ao2mo, fci

from coherent_states.CS_Thouless import CS_Thouless
from coherent_states.CS_Qubit import CS_Qubit
from coherent_states.CS_sample import CS_sample

from class_Journal import Journal
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
        "system" : {"user_actions" : "txt", "log" : "txt"}, # User actions summary, Journal style log
        "molecule" : {"mol_structure" : "pkl", "mode_structure" : "pkl"}, # molecule properties
        "self_analysis" : {"physical_properties" : "json"}, # things which are deterministic but costly to calculate
        #"samples" : {"basis_samples" : "pkl"}, # The specific samples, useful to reconstruct the ground state
        #"results" : {"result_energy_states" : "pkl"}, # The solution as found by the CS method
        "diagnostics" : {"diagnostic_log" : "txt"} # Condition numbers, eigenvalue min-max ratios, norms etc
    }

    def __init__(self, ID, log_verbosity = 5):

        self.ID = ID
        self.log = Journal(log_verbosity)
        # Verbosity key:
        #   0/None: Important, always prints (user methods)
        #   2: Important, major subroutine
        #   5: Minor subroutine
        #   10: Micro-subroutine (expected to repeat many times)
        # Typically, writes within subroutines are one level of verbosity higher than the subroutine itself

        self.log.enter(f"Initialising molecule solver {ID} at log_verbosity = {log_verbosity}")

        # Data Storage Manager
        self.log.write("Initialising Data Storage Manager...", 1)
        self.disk_jockey = DSM(f"outputs/{self.ID}", self.log)
        self.disk_jockey.create_data_nodes(ground_state_solver.data_nodes) # Each solver call adds a new node dynamically

        self.user_actions = f"initialised solver {self.ID}\n"
        self.diagnostics_log = []
        # The structure of the diagnostics log is like so: every element is either a list (for multiple same-level subprocedures) or a dict (for a header : content) pair

        # Self-analysis properties
        self.log.write("Initialising self-analysis properties...", 1)
        self.checklist = [] # List of succesfully performed actions
        self.measured_datasets = [] # list of dataset labels

        self.reference_state_energy = None
        self.ci_energy = None
        self.ci_sol = None # We did not perform full CI

        self.SECS_eta = None
        self.SECS_energy = None

        self.log.exit()


    ###########################################################################
    # --------------------------- Internal methods ----------------------------
    ###########################################################################

    def check_off(self, checklist_element):
        if checklist_element not in self.checklist:
            self.checklist.append(checklist_element)


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

        self.log.enter(f"Getting all Slater determinants for M = {M}, S = {S}", 5)
        res = [ [1] * S + [0] * (M - S) ]
        while(True):
            next_state = functions.choose_iterator(res[-1])
            if next_state is None:
                break
            res.append(next_state)

        self.log.write(f"Found {len(res)} states.", 6)
        self.log.exit()
        return(res)


    def get_full_basis(self, trim_M = None):
        # Returns a list containing all occupancy basis states represented as
        # two-item lists, with each element describing the occupancy in one
        # spin subspace.

        # if trim_M is set to an int S <= trim_M < mol.nao, we will restrict
        # the occupancy basis to the lowest trim_M MOs.

        self.log.enter(f"Getting the full electronic occupancy basis with trim_M = {trim_M}", 5)

        if trim_M is None:
            M = self.mol.nao
        else:
            M = trim_M
        self.log.write(f"Mode number for basis = {M}", 6)
        S_A = self.S_alpha
        S_B = self.S_beta
        self.log.write(f"Total electron numbers for basis: S_alpha = {S_A}, S_beta = {S_B}.", 6)

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
        self.log.exit()
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

        self.log.enter(f"Finding the ground state on full occupancy basis with trim_M = {trim_M}", 2)

        if trim_M is None:
            M = self.mol.nao
        else:
            M = trim_M

        self.log.write(f"Mode number for basis = {M}", 5)

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

        msg = f"Explicit Hamiltonian evaluation on a trimmed full CI"
        #new_sem_ID = self.semaphor.create_event(np.linspace(0, len(fb), 100 + 1), msg)
        self.log.enter(msg, 1, True, tau_space = np.linspace(0, len(fb), 100 + 1))


        self.log.write(f"Evaluating the Hamiltonian on full occupancy basis...", 3)
        for i in range(len(fb)):
            for j in range(len(fb)):
                #self.semaphor.update(new_sem_ID, i)
                self.log.update_semaphor_event(i)
                H[i][j] = self.get_H_overlap_on_occupancy(fb[i], fb[j])

        #self.semaphor.finish_event(new_sem_ID, "Evaluation")
        self.log.exit("Evaluation")


        self.log.enter(f"Diagonalising Hamiltonian matrix...", 3)
        energy_levels, energy_states = np.linalg.eig(H)
        ground_state_index = np.argmin(energy_levels)
        ground_state_energy = energy_levels[ground_state_index]

        self.log.enter(f"Obtained ground state energy = {ground_state_energy}", 3)

        self.log.exit()

        return(ground_state_energy.real)





    def find_ground_state_sampling(self, **kwargs):
        # kwargs:
        #     -N: sample size
        #     -sampling_method: magic word for how the CS samples its parameter tensor
        #     -CS: Type of coherent state to use
        #     -assume_spin_symmetry: if true, |z_alpha> = |z_beta> for the entire sample. False by default
        #     ------------------------
        #     -dataset_label: if present, this will label the dataset in disc jockey. Otherwise, label is generated from other kwargs.

        # -------------------- Parameter initialisation

        assert "N" in kwargs
        N = kwargs["N"] + 1
        assert "sampling_method" in kwargs
        sampling_method = kwargs["sampling_method"]
        assert "CS" in kwargs
        CS_type = kwargs["CS"]

        if "assume_spin_symmetry" in kwargs:
            assume_spin_symmetry = kwargs["assume_spin_symmetry"]
        else:
            assume_spin_symmetry = False

        if "dataset_label" in kwargs:
            dataset_label = kwargs["dataset_label"]
        else:
            dataset_label = f"sampling_{N-1}_{sampling_method}_{CS_type}_{assume_spin_symmetry}"


        self.log.enter(f"Obtaining the ground state with the method \"random sampling\" [CS types = {CS_type}, N = {N - 1}, sampling method = {sampling_method}, assume spin symmetry = {assume_spin_symmetry}]", 0)
        self.log.write(f"Dataset label is \"{dataset_label}\"", 1)



        # Disk jockey node creation and metadata storage
        self.disk_jockey.create_data_nodes({dataset_label : {"basis_samples" : "pkl", "result_energy_states" : "csv"}})
        self.disk_jockey.commit_metadatum(dataset_label, "basis_samples", {"CS" : CS_type, "N" : N - 1, "sampling_method" : sampling_method, "assume_spin_symmetry" : assume_spin_symmetry})
        self.user_actions += f"find_ground_state_sampling [N = {N - 1}, sampling method = {sampling_method}, assume spin symmetry = {assume_spin_symmetry}]\n"
        procedure_diagnostic = []

        # We sample around the HF null guess, i.e. Z = 0
        # We include one extra basis vector - the null point itself!
        self.log.write(f"Sampling the Z-parameter space with provided methods.", 3)
        cur_CS_sample = [[ground_state_solver.coherent_state_types[CS_type].null_state(self.mol.nao, self.S_alpha), ground_state_solver.coherent_state_types[CS_type].null_state(self.mol.nao, self.S_beta)]]
        for i in range(N - 1):

            if assume_spin_symmetry:
                new_sample_state = ground_state_solver.coherent_state_types[CS_type].random_state(self.mol.nao, self.S_alpha, sampling_method)
                cur_CS_sample.append([new_sample_state, new_sample_state])

            else:
                cur_CS_sample.append([
                    ground_state_solver.coherent_state_types[CS_type].random_state(self.mol.nao, self.S_alpha, sampling_method),
                    ground_state_solver.coherent_state_types[CS_type].random_state(self.mol.nao, self.S_beta, sampling_method)
                    ])

        basis_samples_bulk = []
        for i in range(N):
            basis_samples_bulk.append([cur_CS_sample[i][0].z, cur_CS_sample[i][1].z])
        self.disk_jockey.commit_datum_bulk(dataset_label, "basis_samples", basis_samples_bulk)
        self.log.write(f"Basis sample of size {N} committed", 1)

        #procedure_diagnostic.append(f"basis norm max/min = {np.max(norm_coefs) / np.min(norm_coefs)}")

        self.log.write(f"Initialising overlap matrix...", 4)
        overlap_matrix = np.zeros((N, N), dtype=complex) # [a][b] = <a|b>
        for i in range(N):
            for j in range(N):
                overlap_matrix[i][j] = cur_CS_sample[i][0].norm_overlap(cur_CS_sample[j][0]) * cur_CS_sample[i][1].norm_overlap(cur_CS_sample[j][1])
        #print("-- Overlap matrix:")
        #print(overlap_matrix)
        self.log.write(f"Overlap matrix condition number = {np.linalg.cond(overlap_matrix)}", 1)

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

        msg = f"Explicit Hamiltonian evaluation"
        #new_sem_ID = self.semaphor.create_event(np.linspace(0, N * (N + 1) / 2, 100 + 1), msg)
        self.log.enter(msg, 1, True, tau_space = np.linspace(0, N * (N + 1) / 2, 100 + 1))

        for a in range(N):
            # Here the diagonal <Z_a|H|Z_a>
            cur_H_overlap = self.H_overlap(cur_CS_sample[a], cur_CS_sample[a])
            H_eff[a][a] = cur_H_overlap
            #self.semaphor.update(new_sem_ID, a * (a + 1) / 2)
            self.log.update_semaphor_event(a * (a + 1) / 2)

            # Here the off-diagonal, using the fact that H_eff is a Hermitian matrix
            for b in range(a):
                # We explicitly calculate <Z_a | H | Z_b>
                cur_H_overlap = self.H_overlap(cur_CS_sample[a], cur_CS_sample[b])
                H_eff[a][b] = cur_H_overlap
                H_eff[b][a] = np.conjugate(H_eff[a][b])
                #self.semaphor.update(new_sem_ID, a * (a + 1) / 2 + b + 1)
                self.log.update_semaphor_event(a * (a + 1) / 2 + b + 1)

        #self.solution_benchmark = self.semaphor.finish_event(new_sem_ID, "Evaluation")
        self.log.exit("Evaluation")

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
        csv_sol = [] # list of rows, first one being header
        for N_eff_val in range(1, N + 1):
            N_vals.append(N_eff_val)
            convergence_sols.append(get_partial_sol(N_eff_val))
            csv_sol.append({"N" : N_vals[-1], "E [H]" : float(convergence_sols[-1])})

        self.disk_jockey.commit_datum_bulk(dataset_label, "result_energy_states", csv_sol)
        self.diagnostics_log.append({f"find_ground_state_sampling ({dataset_name})" : procedure_diagnostic})

        self.measured_datasets.append(dataset_label)

        self.log.exit()

        return(N_vals, convergence_sols)

    def find_ground_state_manual(self, **kwargs):
        # kwargs:
        #     -sample: an instance of CS_sample
        #     ------------------------
        #     -dataset_label: if present, this will label the dataset in disc jockey. Otherwise, label is generated from other kwargs.

        # -------------------- Parameter initialisation

        assert "sample" in kwargs
        cur_CS_sample = kwargs["sample"].basis

        N = kwargs["sample"].N
        CS_type = kwargs["sample"].CS_class.class_name

        if "dataset_label" in kwargs:
            dataset_label = kwargs["dataset_label"]
        else:
            dataset_label = f"manual_{N}_{CS_type}"

        # Disk jockey node creation and metadata storage
        self.disk_jockey.create_data_nodes({dataset_label : {"basis_samples" : "pkl", "result_energy_states" : "csv"}})
        self.disk_jockey.commit_metadatum(dataset_label, "basis_samples", {"CS" : CS_type, "N" : N, "sampling_method" : "manual"})
        self.user_actions += f"find_ground_state_manual [N = {N}, CS type = {CS_type}]\n"
        procedure_diagnostic = []


        print(f"Obtaining the ground state with the method \"manual sampling\" [N = {N}, CS type = {CS_type}]")


        self.disk_jockey.commit_datum_bulk(dataset_label, "basis_samples", kwargs["sample"].get_z_tensor())

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

        msg = f"Explicit Hamiltonian evaluation"
        #new_sem_ID = self.semaphor.create_event(np.linspace(0, N * (N + 1) / 2, 100 + 1), msg)
        self.log.enter(msg, 1, True, tau_space = np.linspace(0, N * (N + 1) / 2, 100 + 1))

        for a in range(N):
            # Here the diagonal <Z_a|H|Z_a>
            cur_H_overlap = self.H_overlap(cur_CS_sample[a], cur_CS_sample[a])
            H_eff[a][a] = cur_H_overlap
            #self.semaphor.update(new_sem_ID, a * (a + 1) / 2)
            self.log.update_semaphor_event(a * (a + 1) / 2)

            # Here the off-diagonal, using the fact that H_eff is a Hermitian matrix
            for b in range(a):
                # We explicitly calculate <Z_a | H | Z_b>
                cur_H_overlap = self.H_overlap(cur_CS_sample[a], cur_CS_sample[b])
                H_eff[a][b] = cur_H_overlap
                H_eff[b][a] = np.conjugate(H_eff[a][b])
                #self.semaphor.update(new_sem_ID, a * (a + 1) / 2 + b + 1)
                self.log.update_semaphor_event(a * (a + 1) / 2 + b + 1)

        #self.solution_benchmark = self.semaphor.finish_event(new_sem_ID, "    Evaluation")
        self.log.exit("Evaluation")

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
        csv_sol = [] # list of rows
        for N_eff_val in range(1, N + 1):
            N_vals.append(N_eff_val)
            convergence_sols.append(get_partial_sol(N_eff_val))
            csv_sol.append({"N" : N_vals[-1], "E [H]" : float(convergence_sols[-1])})

        self.disk_jockey.commit_datum_bulk(dataset_label, "result_energy_states", csv_sol)
        self.diagnostics_log.append({f"find_ground_state_manual ({dataset_name})" : procedure_diagnostic})

        self.measured_datasets.append(dataset_label)

        return(N_vals, convergence_sols)

    def find_ground_state_SEGS_width(self, **kwargs):
        # kwargs:
        #     -N: Sample size
        #     -N_sub: Subsample size
        #     -CS: Type of CS to use. Not all CS types support SEGS!
        #     ------------------------
        #     -dataset_label: if present, this will label the dataset in disc jockey. Otherwise, label is generated from other kwargs.

        # -------------------- Parameter initialisation

        assert "N" in kwargs
        N = kwargs["N"]

        if "N_sub" in kwargs:
            N_subsample = kwargs["N_sub"]
        else:
            N_subsample = 1

        if "CS" in kwargs:
            CS_type = kwargs["CS"]
        else:
            CS_type = "Thouless"

        if "dataset_label" in kwargs:
            dataset_label = kwargs["dataset_label"]
        else:
            dataset_label = f"SEGS_width_{N}_{N_subsample}_{CS_type}"


        self.log.enter(f"Obtaining the ground state with the method \"SEGS for dist width\" [N = {N}, N_sub = {N_subsample}, CS = {CS_type}]", 1)


        # Disk jockey node creation and metadata storage
        self.disk_jockey.create_data_nodes({dataset_label : {"basis_samples" : "pkl", "result_energy_states" : "csv"}})
        self.disk_jockey.commit_metadatum(dataset_label, "basis_samples", {"CS" : CS_type, "N" : N, "N_sub" : N_subsample})
        self.user_actions += f"find_ground_state_SEGS_width [N = {N}, N_sub = {N_subsample}, CS type = {CS_type}]\n"
        procedure_diagnostic = []

        assert self.S_alpha == self.S_beta # So far only RHF!

        self.log.write(f"Obtaining SEGS heatmap and reference energy...", 1)
        cur_SECS_heatmap, cur_SECS_restricted_energy = self.solve_on_single_excitation_closed_shell()

        # We now manually sample Thouless states guided by the SECS heatmap
        cur_sample = CS_sample(self, ground_state_solver.coherent_state_types[CS_type], add_ref_state = True)


        # Here, depending on the CS type, we set up the normal distribution parameters
        self.log.write(f"Determining proper Z-parameter constants for the sampling process...", 7)
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
        self.log.write(f"Spin-alpha has shape {shape_alpha}; spin-beta has shape {shape_beta}", 3)

        N_vals = [1]
        convergence_sols = [self.reference_state_energy]

        msg = f"Conditioned sampling with ground state search on {N} states, each taken from {N_subsample} random states"
        #new_sem_ID = self.semaphor.create_event(np.linspace(0, N_subsample * ((N + 2) * (N + 1) / 2 - 1) + 1, 1000 + 1), msg)
        self.log.enter(msg, 1, True, tau_space = np.linspace(0, N_subsample * ((N + 2) * (N + 1) / 2 - 1) + 1, 1000 + 1))

        for n in range(N):
            # We add the best out of 10 random states
            cur_subsample = []
            for n_sub in range(N_subsample):
                #rand_z_alpha = ground_state_solver.coherent_state_types[CS_type](self.mol.nao, self.S_alpha, np.random.normal(centres_alpha, np.sqrt(cur_SECS_heatmap), shape_alpha))
                #rand_z_beta = ground_state_solver.coherent_state_types[CS_type](self.mol.nao, self.S_beta, np.random.normal(centres_beta, np.sqrt(cur_SECS_heatmap), shape_beta))
                rand_z_alpha = ground_state_solver.coherent_state_types[CS_type](self.mol.nao, self.S_alpha, np.random.normal(centres_alpha, widths_alpha, shape_alpha))
                rand_z_beta = ground_state_solver.coherent_state_types[CS_type](self.mol.nao, self.S_beta, np.random.normal(centres_beta, widths_beta, shape_beta))
                cur_subsample.append([rand_z_alpha, rand_z_beta])

            cur_sample.add_best_of_subsample(cur_subsample, update_semaphor = True)
            N_vals.append(cur_sample.N)
            convergence_sols.append(cur_sample.E_ground[-1])

        procedure_diagnostic.append(f"Full sample condition number = {cur_sample.S_cond}")

        #solution_benchmark = self.semaphor.finish_event(new_sem_ID, "Evaluation")
        self.log.exit("Evaluation")

        csv_sol = [] # list of rows
        for i in range(len(N_vals)):
            csv_sol.append({"N" : N_vals[i], "E [H]" : float(convergence_sols[i])})


        self.disk_jockey.commit_datum_bulk(dataset_label, "basis_samples", cur_sample.get_z_tensor())
        self.disk_jockey.commit_datum_bulk(dataset_label, "result_energy_states", csv_sol)
        self.diagnostics_log.append({f"find_ground_state_SEGS_width ({dataset_label})" : procedure_diagnostic})

        self.measured_datasets.append(dataset_label)

        self.log.exit()
        return(N_vals, convergence_sols)

    def find_ground_state_SEGS_phase(self, **kwargs):
        # kwargs:
        #     -N: Sample size
        #     -N_sub: Subsample size
        #     -CS: Type of CS to use. Not all CS types support SEGS!
        #     ------------------------
        #     -dataset_label: if present, this will label the dataset in disc jockey. Otherwise, label is generated from other kwargs.

        # -------------------- Parameter initialisation

        assert "N" in kwargs
        N = kwargs["N"]

        if "N_sub" in kwargs:
            N_subsample = kwargs["N_sub"]
        else:
            N_subsample = 1

        if "CS" in kwargs:
            CS_type = kwargs["CS"]
        else:
            CS_type = "Thouless"

        assert CS_type == "Thouless" # no other supporting type so far

        if "dataset_label" in kwargs:
            dataset_label = kwargs["dataset_label"]
        else:
            dataset_label = f"SEGS_phase_{N}_{N_subsample}_{CS_type}"


        self.log.enter(f"Obtaining the ground state with the method \"SEGS with random phase\" [N = {N}, N_sub = {N_subsample}, CS type = {CS_type}]", 1)


        # Disk jockey node creation and metadata storage
        self.disk_jockey.create_data_nodes({dataset_label : {"basis_samples" : "pkl", "result_energy_states" : "csv"}})
        self.disk_jockey.commit_metadatum(dataset_label, "basis_samples", {"CS" : CS_type, "N" : N, "N_sub" : N_subsample})
        self.user_actions += f"find_ground_state_SEGS_phase [N = {N}, N_sub = {N_subsample}, CS type = {CS_type}]\n"
        procedure_diagnostic = []

        assert self.S_alpha == self.S_beta

        self.log.write(f"Obtaining SEGS heatmap and reference energy...", 1)

        cur_SECS_heatmap, cur_SECS_restricted_energy = self.solve_on_single_excitation_closed_shell()

        # We now manually sample Thouless states guided by the SECS heatmap
        cur_sample = CS_sample(self, CS_Thouless, add_ref_state = True)


        self.log.write(f"Determining proper Z-parameter constants for the sampling process...", 7)

        shape_alpha = (self.mol.nao - self.S_alpha, self.S_alpha)
        shape_beta = (self.mol.nao - self.S_beta, self.S_beta)
        centres_alpha = np.zeros(shape_alpha)
        centres_beta = np.zeros(shape_beta)


        self.log.write(f"Spin-alpha has shape {shape_alpha}; spin-beta has shape {shape_beta}", 3)

        N_vals = [1]
        convergence_sols = [self.reference_state_energy]

        msg = f"Conditioned sampling with ground state search on {N} states, each taken from {N_subsample} random states"
        #new_sem_ID = self.semaphor.create_event(np.linspace(0, N_subsample * ((N + 2) * (N + 1) / 2 - 1) + 1, 1000 + 1), msg)
        self.log.enter(msg, 1, True, tau_space = np.linspace(0, N_subsample * ((N + 2) * (N + 1) / 2 - 1) + 1, 1000 + 1))

        for n in range(N):
            # We add the best out of 10 random states
            cur_subsample = []
            for n_sub in range(N_subsample):
                rand_z_alpha = CS_Thouless(self.mol.nao, self.S_alpha, np.sqrt(cur_SECS_heatmap) * np.exp(1j * np.random.random(shape_alpha) * 2.0 * np.pi))
                rand_z_beta = CS_Thouless(self.mol.nao, self.S_beta, np.sqrt(cur_SECS_heatmap) * np.exp(1j * np.random.random(shape_beta) * 2.0 * np.pi))
                cur_subsample.append([rand_z_alpha, rand_z_beta])

            cur_sample.add_best_of_subsample(cur_subsample, update_semaphor = True)
            N_vals.append(cur_sample.N)
            convergence_sols.append(cur_sample.E_ground[-1])

        procedure_diagnostic.append(f"Full sample condition number = {cur_sample.S_cond}")

        #solution_benchmark = self.semaphor.finish_event(new_sem_ID, "Evaluation")
        self.log.exit("Evaluation")

        csv_sol = [] # list of rows
        for i in range(len(N_vals)):
            #csv_sol.append([N_vals[i], convergence_sols[i]])
            csv_sol.append({"N" : N_vals[i], "E [H]" : float(convergence_sols[i])})

        self.disk_jockey.commit_datum_bulk(dataset_label, "result_energy_states", csv_sol)
        self.diagnostics_log.append({f"find_ground_state_SEGS_phase ({dataset_label})" : procedure_diagnostic})

        self.measured_datasets.append(dataset_label)

        self.log.exit()

        return(N_vals, convergence_sols)



    def find_ground_state_krylov(self, **kwargs):
        # kwargs:
        #     -dt: float; time spacing of the basis sampling process
        #     ------------------------
        #     -dataset_label: if present, this will label the dataset in disc jockey. Otherwise, label is generated from other kwargs.
        print("placeholder", kwargs["dt"])

        # The initial guess is the null matrix, i.e. all pi_1 modes occupied.




    def find_ground_state_imaginary_timeprop(self, **kwargs):
        # kwargs:
        #     -dt: float; timestep
        #     -tol: float; tolerance on the solution or smth
        #     ------------------------
        #     -dataset_label: if present, this will label the dataset in disc jockey. Otherwise, label is generated from other kwargs.
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

        self.log.enter("Initialising molecule...", 0)

        if "mol_init" in self.checklist:
            self.log.write("Molecule already initialised.", 1)
            return(None)

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

        self.log.write(f"There are {self.mol.nao} atomic orbitals, each able to hold 2 electrons of opposing spin.", 3)
        self.log.write(f"The molecule is occupied by {self.mol.tot_electrons()} electrons in total: {self.S_alpha} with spin alpha, {self.S_beta} with spin beta.", 3)
        self.log.write(f"The molecule consists of the following atoms: {self.mol.elements}", 3)
        self.log.write(f"The atomic orbitals are ordered as follows: {self.mol.ao_labels()}", 3)
        self.log.write(f"The nuclear repulsion energy is {self.mol.energy_nuc()}", 3)
        #print(gto.charge("O"))

        self.log.write("Calculating 1e integrals...", 1)
        AO_H_one = self.mol.intor('int1e_kin', hermi = 1) + self.mol.intor('int1e_nuc', hermi = 1)

        self.log.write("Calculating 2e integrals...", 1)
        #self.H_two = self.mol.intor('int2e', aosym = "s1")
        AO_H_two_chemist = self.mol.intor('int2e', aosym = "s1")
        # <ij|kl> = (ik|jl)

        # We now construct AO_H_two as the coefficient tensor in second quantisation according to Szabo & Ostlund: Modern Quantum Chemistry p. 95, Eq. 2.232
        # O_2 = 0.5 * sum_ijkl <ij|kl> f\hc_i f\hc_j f_l f_k
        # Using <ij|kl> = (ik|jl) we have
        # O_ijkl = 0.5 (il|jk)
        # By symmetry: (ij|kl) = (ji|kl) = (ij|lk) = (ji|lk)
        # Hence O_ijkl = O_ljki = O_ikjl = O_lkji

        self.log.write("Finding the molecular orbitals using mean-field approximations...", 1)
        mean_field = scf.RHF(mol).run(verbose = 0)
        self.MO_coefs = mean_field.mo_coeff
        self.reference_state_energy = mean_field.e_tot
        self.log.write(f"Done! Reference state energy is {self.reference_state_energy:0.5f}", 1)

        self.log.write("Transforming 1e and 2e integrals to MO basis...", 3)
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

        self.log.enter("Reference state analysis", 4)
        self.log.write(f"Occupied orbitals:", 4)
        self.log.write(f"  -in the spin-alpha subspace: {occ_orbs}", 4)
        self.log.write(f"  -in the spin-beta subspace:  {occ_orbs}", 4)



        #print("    Antisym tests")
        #print("      Sym 0 <-> 3?", (np.round(self.MO_H_two - self.MO_H_two.transpose(3, 1, 2, 0), 2) == 0).all())
        #print("      Sym 1 <-> 2", (np.round(self.MO_H_two - self.MO_H_two.transpose(0, 2, 1, 3), 2) == 0).all())

        """null_state_energy = 0.0
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
        print(f"  Energy of the null state = {null_state_energy}")"""
        #null_state_alpha = CS_Thouless(self.mol.nao, self.S_alpha, np.zeros((self.mol.nao - self.S_alpha, self.S_alpha), dtype=complex))
        #null_state_beta = CS_Thouless(self.mol.nao, self.S_beta, np.zeros((self.mol.nao - self.S_beta, self.S_beta), dtype=complex))
        null_state_alpha = CS_Qubit.null_state(self.mol.nao, self.S_alpha)
        null_state_beta = CS_Qubit.null_state(self.mol.nao, self.S_alpha)
        null_state = [null_state_alpha, null_state_beta]
        null_state_direct_self_energy = self.H_overlap(null_state, null_state).real
        self.log.write(f"Reference state energy with the overlap method = {null_state_direct_self_energy:0.5f}", 4)
        self.log.exit() # exits reference state analysis

        self.user_actions += f"initialise_molecule [M = {self.M}, S = {self.S}]\n"
        mol_structure_bulk = {
            "atom" : self.mol.atom,
            "basis" : self.mol.basis,
            "spin" : self.mol.spin
            }
        mode_structure_bulk = {
            "M" : self.M,
            "S" : self.S,
            "MO_coefs" : self.MO_coefs,
            "modes" : self.modes,
            "MO_H_one" : self.MO_H_one,
            "MO_H_two" : self.MO_H_two
            }
        self.disk_jockey.commit_datum_bulk("molecule", "mol_structure", mol_structure_bulk)
        self.disk_jockey.commit_datum_bulk("molecule", "mode_structure", mode_structure_bulk)

        self.check_off("mol_init")

        self.log.exit()

    def pyscf_full_CI(self):
        self.user_actions += f"pyscf_full_CI\n"

        self.log.enter("Performing SCF on full CI...", 0)

        if "pyscf_full_CI" in self.checklist:
            print("Full CI by PySCF already performed.")
            return(self.ci_energy)

        cisolver = fci.FCI(self.mol, self.MO_coefs)
        self.log.write("FCI solver initialised...", 0)
        self.ci_energy, raw_ci_sol = cisolver.kernel()

        # We convert the raw_ci_sol object (which is an FCIvector) into a dict
        # with tuples as keys (tuples represent occupancy strings)

        self.ci_sol = {}
        norb = cisolver.norb
        n_alpha, n_beta = cisolver.nelec

        self.log.write("Regularising solution as a dict of tuples...", 3)
        for a in range(raw_ci_sol.shape[0]):
            for b in range(raw_ci_sol.shape[1]):
                alpha_occ = self.occ_str_to_occ_tuple("{0:b}".format(fci.cistring.addr2str(norb, n_alpha, a)))
                beta_occ = self.occ_str_to_occ_tuple("{0:b}".format(fci.cistring.addr2str(norb, n_beta, b)))
                key = (alpha_occ, beta_occ)
                self.ci_sol[key] = float(raw_ci_sol[a, b])

        self.log.write(f"Type of solution vector is {type(self.ci_sol)}")

        self.check_off("pyscf_full_CI")

        self.log.write(f"Ground state energy as calculated by SCF (full configuration) = {self.ci_energy}", 0)
        self.log.exit()
        return(self.ci_energy)




    def search_for_states(self, state_type, sampling_method, inclusion):
        self.user_actions += f"search_for_states [{state_type}, {sampling_method}]\n"
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
        self.user_actions += f"save_data\n"
        # Save user log
        self.disk_jockey.commit_datum_bulk("system", "user_actions", self.user_actions)
        self.disk_jockey.commit_datum_bulk("system", "log", self.log.dump())
        self.disk_jockey.commit_metadatum("system", "log", {
            "checklist" : self.checklist,
            "measured_datasets" : self.measured_datasets
            })

        # Save self-analysis results which were performed
        physical_properties_bulk = {}
        if "mol_init" in self.checklist:
            physical_properties_bulk["reference_state_energy"] = self.reference_state_energy
        if "pyscf_full_CI" in self.checklist:
            physical_properties_bulk["ci_energy"] = self.ci_energy
            physical_properties_bulk["ci_sol"] = {str(k): v for k, v in self.ci_sol.items()} # each key is a tuple of tuples
        if "SECS_sol" in self.checklist:
            physical_properties_bulk["SECS_eta"] = self.SECS_eta.tolist()
            physical_properties_bulk["SECS_energy"] = self.SECS_energy
        self.disk_jockey.commit_datum_bulk("self_analysis", "physical_properties", physical_properties_bulk)

        # save diagnostic
        self.disk_jockey.commit_datum_bulk("diagnostics", "diagnostic_log", self.print_diagnostic_log())

        # Save to disk
        self.disk_jockey.save_data()

        self.log.close_journal()

    def load_data(self, what_to_load = None):
        # what_to_load is a list of magic strings
        if what_to_load is None:
            # Default option
            self.load_data(["system", "self_analysis", "measured_datasets"])
        else:
            self.log.enter("Loading data from the disk...", 0)
            self.user_actions += f"load_data\n"
            # loads data using disk jockey

            # Firstly, let's see what we can load!
            self.log.write("Reading files on disk...", 5)
            self.disk_jockey.load_data(["system", "diagnostics"]) # Always by default
            loaded_checklist = self.disk_jockey.metadata["system"]["log"]["checklist"]

            if "self_analysis" in what_to_load and "mol_init" in loaded_checklist:
                self.log.enter("Restoring self-analysis values...", 3)
                if "mol_init" not in self.checklist:
                    self.disk_jockey.load_data(["molecule"])
                self.disk_jockey.load_data(["self_analysis"])

                loaded_phys_properties = self.disk_jockey.data_bulks["self_analysis"]["physical_properties"]

                self.reference_state_energy = loaded_phys_properties["reference_state_energy"]
                self.check_off("mol_init")
                if "pyscf_full_CI" in loaded_checklist:
                    self.ci_energy = loaded_phys_properties["ci_energy"]
                    self.ci_sol = {tuple([tuple([int(x) for x in occ.split(", ")]) for occ in k.strip("()").split("), (")]): v for k, v in loaded_phys_properties["ci_sol"].items()} # evil oneliner
                    self.check_off("pyscf_full_CI")
                    self.log.write("Results from SCF performed on the full CI loaded...", 4)

                if "SECS_sol" in loaded_checklist:
                    self.SECS_eta = np.array(loaded_phys_properties["SECS_eta"])
                    self.SECS_energy = loaded_phys_properties["SECS_energy"]
                    self.check_off("SECS_sol")
                    self.log.write("Results from diagonalisation on the SECS basis loaded...", 4)

                self.log.exit()

            if "measured_datasets" in what_to_load:
                self.log.enter("Restoring measured datasets...", 3)
                for loaded_dataset in self.disk_jockey.metadata["system"]["log"]["measured_datasets"]:
                    if loaded_dataset not in self.measured_datasets:
                        self.log.write(f"Restoring dataset '{loaded_dataset}'...", 4)
                        for dataset_datum in self.disk_jockey.data_nodes[loaded_dataset]:
                            self.disk_jockey.load_datum(loaded_dataset, dataset_datum)
                        self.measured_datasets.append(loaded_dataset)
                    else:
                        self.log.write("WARNING: Attempted to load a dataset which exists in internal checklist. Data from the disk was ignored.", 0)
                self.log.exit()

            self.log.exit()





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
            self.log.write(f"ERROR: Unknown ground state method {method}. Available methods: {self.find_ground_state_methods.keys()}")
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

    def occ_str_to_occ_tuple(self, occ_str):
        # the string is just a binary rep of the address in the FCI vector
        if isinstance(occ_str, tuple):
            return(occ_str) # just to regularise user action
        res = []
        for i in range(len(occ_str) - 1, -1, -1):
            res.append(int(occ_str[i]))
        # we regularise the tuple lengths
        #for j in range() or not
        return(tuple(res))

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
        # alpha/beta_occupancy is a tuple (occ1, occ2... occ_M) where N is the
        # number of alpha/beta MOs and occupancies sum up to S_alpha/beta.
        #alpha_idx = fci.cistring.str2addr(self.mol.nao, self.S_alpha, self.occ_list_to_occ_string(alpha_occupancy))
        #beta_idx = fci.cistring.str2addr(self.mol.nao, self.S_beta, self.occ_list_to_occ_string(beta_occupancy))

        # Access coefficient
        if isinstance(alpha_occupancy, tuple):
            return(self.ci_sol[(alpha_occupancy, beta_occupancy)])
        elif isinstance(alpha_occupancy, list):
            return(self.ci_sol[(tuple(alpha_occupancy), tuple(beta_occupancy))])
        #return(self.ci_sol[alpha_idx, beta_idx])

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
        self.user_actions += f"get_top_closed_shells [N_cs = {N_cs}, trim_M = {trim_M}]\n"
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
        self.user_actions += f"single_excitation_singlets_projection [trim_M = {trim_M}]\n"

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
        self.user_actions += f"single_excitation_closed_shell_heatmap [trim_M = {trim_M}]\n"

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

        self.user_actions += f"plot_single_excitation_closed_shell_heatmap [trim_M = {trim_M}]\n"

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

        self.log.enter("Solving on a single-excitation closed-shell basis...", 1)

        self.user_actions += f"solve_on_single_excitation_closed_shell [trim_M = {trim_M}]\n"

        if "SECS_sol" in self.checklist:
            # Already solved
            self.log.write("Already solved.", 1)
            self.log.exit()
            return(self.SECS_eta, self.SECS_energy)


        assert(self.S_alpha == self.S_beta)

        act_M = self.mol.nao
        if trim_M is not None:
            if trim_M > self.S_alpha: # We need at least one empty shell
                act_M = min(trim_M, self.mol.nao)

        self.log.write(f"Mode number for each spin space = {act_M}", 4)

        ref_state = [1] * self.S_alpha + [0] * (act_M - self.S_alpha)
        basis = [[ref_state, ref_state]]


        self.log.write(f"Obtaining the SECS basis....", 1)

        for a in range(act_M - self.S_alpha):
            for b in range(self.S_alpha):
                cur_state = ref_state.copy()
                cur_state[self.S_alpha + a] = 1
                cur_state[b] = 0
                basis.append([cur_state, cur_state])

        self.log.write(f"SECS basis of length {len(basis)} obtained.", 1)

        H = np.zeros((len(basis), len(basis)))
        msg = f"Explicit Hamiltonian evaluation on SECS basis"


        #new_sem_ID = self.semaphor.create_event(np.linspace(0, len(basis) * (len(basis) + 1) / 2, 100 + 1), msg)
        self.log.enter(msg, 1, True, tau_space = np.linspace(0, len(basis) * (len(basis) + 1) / 2, 100 + 1))

        for a in range(len(basis)):
            # Here the diagonal <Z_a|H|Z_a>
            cur_H_overlap = self.get_H_overlap_on_occupancy(basis[a], basis[a])
            assert cur_H_overlap.imag < 1e-08
            H[a][a] = cur_H_overlap.real
            #self.semaphor.update(new_sem_ID, a * (a + 1) / 2)
            self.log.update_semaphor_event(a * (a + 1) / 2)

            # Here the off-diagonal, using the fact that H is a Hermitian matrix
            for b in range(a):
                # We explicitly calculate <Z_a | H | Z_b>
                cur_H_overlap = self.get_H_overlap_on_occupancy(basis[a], basis[b])
                assert cur_H_overlap.imag < 1e-08
                H[a][b] = cur_H_overlap.real
                H[b][a] = H[a][b] # np.conjugate(H[a][b])
                #self.semaphor.update(new_sem_ID, a * (a + 1) / 2 + b + 1)
                self.log.update_semaphor_event(a * (a + 1) / 2 + b + 1)

        #self.semaphor.finish_event(new_sem_ID, "Evaluation")
        self.log.exit("Evaluation")

        energy_levels, energy_states = np.linalg.eig(H)
        ground_state_index = np.argmin(energy_levels)
        ground_state_energy = energy_levels[ground_state_index]
        ground_state_vector = energy_states[:,ground_state_index] # note that energy_states consist of column vectors, not row vectors

        self.log.write(f"Ground state energy: {ground_state_energy} (compare to full CI: {self.ci_energy})", 1)

        # Now, for the heatmap
        self.log.write(f"Obtaining the single-excitation prevalence matrix...", 2)
        res = np.zeros((act_M - self.S_alpha, self.S_alpha))
        i = 1
        for a in range(act_M - self.S_alpha):
            for b in range(self.S_alpha):
                sol_component = ground_state_vector[i]
                res[a][b] = sol_component * sol_component
                i += 1

        # Mark as solved
        self.check_off("SECS_sol")

        self.SECS_eta = res
        self.SECS_energy = ground_state_energy

        self.log.write(f"Success!", 1)
        self.log.exit()

        return(res, ground_state_energy)



    def plot_SECS_restricted_heatmap(self, ax = None, trim_M = None):

        self.user_actions += f"plot_SECS_restricted_heatmap [trim_M = {trim_M}]\n"

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

    def plot_datasets(self, reference_energies = None):
        # Plots energy against configuration size
        self.log.enter("Plotting obtained measurements...", 1)

        plt.title(f"{self.ID}")
        plt.xlabel("Basis size")
        plt.ylabel("E [Hartree]")

        for i in range(len(self.measured_datasets)):
            self.log.write(f"Collecting data from dataset '{self.measured_datasets[i]}'...", 5)
            ds_val = self.disk_jockey.data_bulks[self.measured_datasets[i]]["result_energy_states"]
            # ds_val is a list of dicts - here we cast it into plottable arrays
            N_space = []
            E_space = []
            for row in ds_val:
                N_space.append(row["N"])
                E_space.append(row["E [H]"])
            plt.plot(N_space, E_space, "x", label = self.measured_datasets[i])

        if "mol_init" in self.checklist:
            plt.axhline(y = self.reference_state_energy, label = "ref state", color = functions.ref_energy_colors["ref state"])
        if "pyscf_full_CI" in self.checklist:
            plt.axhline(y = self.ci_energy, label = "full CI", color = functions.ref_energy_colors["full CI"])
        if "SECS_sol" in self.checklist:
            plt.axhline(y = self.SECS_energy, label = "SECS-restricted CI", color = functions.ref_energy_colors["SECS"])

        if reference_energies is not None:
            for ref_energy in reference_energies:
                ref_e = ref_energy["E"]
                ref_label = ref_energy["label"]
                ref_color = "blue"
                if "color" in ref_energy:
                    ref_color = ref_energy["color"]
                ref_linestyle = "solid"
                if "linestyle" in ref_energy:
                    ref_linestyle = ref_energy["linestyle"]
                plt.axhline(y = ref_e, label = ref_label, color = ref_color, linestyle = ref_linestyle)

        self.log.write(f"Displaying plot...", 5)
        plt.legend()
        plt.tight_layout()
        plt.show()

        self.log.exit()






