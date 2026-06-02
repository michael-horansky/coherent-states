import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from pyscf import gto, scf, cc, ao2mo, ci, fci, pbc

from coherent_states.CS_Thouless import CS_Thouless
from coherent_states.CS_Qubit import CS_Qubit
from coherent_states.CS_sample import CS_sample

from utils.class_Journal import Journal
from utils.class_Disk_Jockey import Disk_Jockey
import utils.functions as functions


def esp(roots, order, omit = []):
    # roots is a list
    # order is an integer <= len(roots)
    # omit is a list of indices on which roots is to be set to zero

    if order == 0:
        return(1.0)
    if order == len(roots):
        return(np.prod(roots))

    partial_esp = np.zeros(order + 1)
    partial_esp[0] = 1.0
    for i in range(len(roots)):
        if i in omit:
            continue
        for j in range(min(i + 1, order), 0, -1):
            partial_esp[j] += roots[i] * partial_esp[j - 1]
    return(partial_esp[order])

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

        # self_analysis contains solutions performed on configuration bases,
        # such as full CI (performed with SCF) or various low-excitation ("LE")
        # bases used to guide the sampling process.
        # Solutions are always stored as dicts in the form
        #     sol[((occ_alpha, occ_beta))] = coef of that state in ground state
        # Metadata for low-excitation solutions specifies what kind of LE basis
        # was used (e.g. singlet CAS with up to 2 open shells for RHF, etc...).
        "self_analysis" : {
            "physical_properties" : "json", # Short properties: energies of solutions below
            "full_CI_sol" : "json",
            "LE_sol" : "json"
            }, # things which are deterministic but costly to calculate
        "diagnostics" : {"diagnostic_log" : "txt"} # Condition numbers, eigenvalue min-max ratios, norms etc
    }

    mean_field_methods = {
        "RHF" : scf.RHF,
        "UHF" : scf.UHF
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
        self.disk_jockey = Disk_Jockey(f"outputs/{self.ID}", self.log)
        self.disk_jockey.create_data_nodes(ground_state_solver.data_nodes) # Each solver call adds a new node dynamically

        self.user_actions = f"initialised solver {self.ID}\n"
        self.diagnostics_log = []
        # The structure of the diagnostics log is like so: every element is either a list (for multiple same-level subprocedures) or a dict (for a header : content) pair

        # ----------- Self-analysis properties
        self.log.write("Initialising self-analysis properties...", 1)
        self.checklist = [] # List of succesfully performed actions
        self.measured_datasets = [] # list of dataset labels

        # mol_init properties
        self.mol = None
        self.mean_field = None
        self.HF_method = None
        self.reference_state_energy = None
        # Full CI properties
        self.ci_energy = None
        self.ci_sol = None # We did not perform full CI
        # Low-excitation solution properties
        self.LE_sol = {
            "E" : None, # Energy of solution
            "sol" : None, # dict[occ tuple] = coefficient
            "exp" : None # expectation value constraint object, context-sensitive form
            }
        self.LE_description = {
            "env" : None, # RHF or UHF, just for verbosity
            "spec" : None, # magic word uniquely specifying process
            "params" : None # parameters used during the solving
            }

        self.log.exit()

        self.find_ground_state_methods = {
            "sampling" : self.find_ground_state_sampling,
            "manual" : self.find_ground_state_manual, # manual sample
            "LEGS_width" : self.find_ground_state_LEGS_width, # uses SEGS for width of sample, correlated for UHF
            "LEGS_phase" : self.find_ground_state_LEGS_phase, # restricts the magnitudes by SEGS, only randomises phases of components
            "LE_first_order" : self.find_ground_state_LEGS_first_order,
            "LE_mixed_spin_covariance" : self.find_ground_state_LEGS_mixed_spin_covariance,
            "LE_Zombie_cov" : self.find_ground_state_LEGS_Zombie_cov,
            "LE_Zombie_cov_ALT" : self.find_ground_state_LEGS_Zombie_cov_ALT,
            "LE_Zombie_cov_SRRM" : self.find_ground_state_LEGS_Zombie_cov_SRRM,
            "LE_Zombie_cov_SRRM_alt" : self.find_ground_state_LEGS_Zombie_cov_SRRM_alt,
            "LE_Zombie_cov_SRRM_mirror" : self.find_ground_state_LEGS_Zombie_cov_SRRM_mirror,
            "LE_Zombie_cov_SOPM" : self.find_ground_state_LEGS_Zombie_cov_SOPM,
            "LE_Zombie_cov_RSOPM" : self.find_ground_state_LEGS_Zombie_cov_RSOPM,
            "LE_Zombie_cov_RSOPM_moment_matching" : self.find_ground_state_LEGS_Zombie_cov_RSOPM_moment_matching,
            "Qubit_from_z_tensor" : self.find_ground_state_from_z_tensor,
            "krylov" : self.find_ground_state_krylov,
            "imag_timeprop" : self.find_ground_state_imaginary_timeprop
        }

        self.low_excitation_methods = { # ["RHF"/"UHF"][spec] = {"method" : solver method, "desc" : human-readable description}
            "RHF" : {
                "SECS" : { # "Single excitation closed-shell states"
                    "method" : self.find_LE_solution_SECS,
                    "desc" : "Includes only closed-shell states with one excitation in each subspace"
                },
                "SEO1" : { # "Single excitation openness-1 states"
                    "method" : self.find_LE_solution_SEO1,
                    "desc" : "Includes singlet SACs with up to two open shells (openness = 1)"
                },
                "SE" : { # "Single excitation states"
                    "method" : self.find_LE_solution_SE,
                    "desc" : "Includes only states with up to one excitation in total"
                }
            },
            "UHF" : {
                "SE" : { # "Single excitation states"
                    "method" : self.find_LE_solution_SE,
                    "desc" : "Includes only states with up to one excitation in total"
                },
                "MSDE" : { # "Mixed-spin double excitation states"
                    "method" : self.find_LE_solution_MSDE,
                    "desc" : "Includes states with up to one excitation per spin-subspace."
                }
            }
        }


    ###########################################################################
    # --------------------------- Internal methods ----------------------------
    ###########################################################################

    def check_off(self, checklist_element):
        if checklist_element not in self.checklist:
            self.checklist.append(checklist_element)

    def spin_label(self):
        # Human-readable label for the molecule spin value
        if "mol_init" in self.checklist:
            if self.mol.spin == 0:
                return("singlet")
            elif self.mol.spin == 1:
                return("doublet")
            elif self.mol.spin == 2:
                return("triplet")
            else:
                return(f"(2S+1)={self.mol.spin}")
        return("[mol not initialised]")


    # -------------------------------------------------------------------------
    # ---------------------------- Solver methods -----------------------------
    # -------------------------------------------------------------------------

    """
    TODO adapt this for differing S_alpha, S_beta

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
        return(A, Z)"""

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

        self.log.exit()
        return(full_basis_A, full_basis_B)

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
                H_one_term += self.mode_exchange_energy([p], [q], "a") * W_alpha[p][q] * beta_overlap
                H_one_term += self.mode_exchange_energy([p], [q], "b") * W_beta[p][q] * alpha_overlap
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
                prefactor_aa = 1.0 * (self.mode_exchange_energy([i, j], [k, l], "a") - self.mode_exchange_energy([i, j], [l, k], "a"))
                prefactor_bb = 1.0 * (self.mode_exchange_energy([i, j], [k, l], "b") - self.mode_exchange_energy([i, j], [l, k], "b"))
                # alpha alpha
                H_two_term += prefactor_aa * self.get_exchange_integral_on_occupancy(occ_a[0], occ_b[0], [j, i], [l, k]) * beta_overlap
                # beta beta
                H_two_term += prefactor_bb * self.get_exchange_integral_on_occupancy(occ_a[1], occ_b[1], [j, i], [l, k]) * alpha_overlap
                upup += prefactor_aa * self.get_exchange_integral_on_occupancy(occ_a[0], occ_b[0], [j, i], [l, k]) * beta_overlap
                downdown += prefactor_bb * self.get_exchange_integral_on_occupancy(occ_a[1], occ_b[1], [j, i], [l, k]) * alpha_overlap
        # opposite spin
        for i in range(M):
            for j in range(M):
                for k in range(M):
                    for l in range(M):
                        prefactor_ab = 0.5 * self.mode_exchange_energy([i, j], [k, l], "ab")
                        prefactor_ba = 0.5 * self.mode_exchange_energy([j, i], [l, k], "ab")
                        # alpha beta
                        H_two_term += prefactor_ab * W_alpha[i][k] * W_beta[j][l]
                        # beta alpha
                        H_two_term += prefactor_ba * W_alpha[j][l] * W_beta[i][k]
                        mixed += prefactor_ab * W_alpha[i][k] * W_beta[j][l] + prefactor_ba * W_alpha[j][l] * W_beta[i][k]
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

        H = np.zeros((len(fb), len(fb)), dtype=complex)

        msg = f"Explicit Hamiltonian evaluation on a trimmed full CI"
        self.log.enter(msg, 1, True, tau_space = np.linspace(0, len(fb), 100 + 1))


        self.log.write(f"Evaluating the Hamiltonian on full occupancy basis...", 3)
        for i in range(len(fb)):
            for j in range(len(fb)):
                self.log.update_semaphor_event(i)
                H[i][j] = self.get_H_overlap_on_occupancy(fb[i], fb[j])

        self.log.exit("Evaluation")

        self.log.enter(f"Diagonalising Hamiltonian matrix...", 3)
        energy_levels, energy_states = np.linalg.eig(H)
        ground_state_index = np.argmin(energy_levels)
        ground_state_energy = energy_levels[ground_state_index].real

        self.log.enter(f"Obtained ground state energy = {ground_state_energy}", 3)

        self.log.exit()
        return(ground_state_energy)





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
        if self.HF_method != "RHF":
            # States cannot be the same in the two subspaces anyway
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
        self.log.enter(msg, 1, True, tau_space = np.linspace(0, N * (N + 1) / 2, 100 + 1))

        for a in range(N):
            # Here the diagonal <Z_a|H|Z_a>
            cur_H_overlap = self.H_overlap(cur_CS_sample[a], cur_CS_sample[a])
            H_eff[a][a] = cur_H_overlap
            self.log.update_semaphor_event(a * (a + 1) / 2)

            # Here the off-diagonal, using the fact that H_eff is a Hermitian matrix
            for b in range(a):
                # We explicitly calculate <Z_a | H | Z_b>
                cur_H_overlap = self.H_overlap(cur_CS_sample[a], cur_CS_sample[b])
                H_eff[a][b] = cur_H_overlap
                H_eff[b][a] = np.conjugate(H_eff[a][b])
                self.log.update_semaphor_event(a * (a + 1) / 2 + b + 1)

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

    def find_ground_state_LEGS_width(self, **kwargs):
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
        self.user_actions += f"find_ground_state_LEGS_width [N = {N}, N_sub = {N_subsample}, CS type = {CS_type}]\n"
        procedure_diagnostic = []

        assert self.S_alpha == self.S_beta # So far only RHF!

        if "LE_sol" not in self.checklist:
            self.log.write(f"ERROR: LEGS method requires LE solution to be known. Aborting...")
            self.log.exit()
            return(None)

        cur_SECS_heatmap = self.LE_sol["exp"]

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
        self.diagnostics_log.append({f"find_ground_state_LEGS_width ({dataset_label})" : procedure_diagnostic})

        self.measured_datasets.append(dataset_label)

        self.log.exit()
        return(N_vals, convergence_sols)

    def find_ground_state_LEGS_phase(self, **kwargs):
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
        self.user_actions += f"find_ground_state_LEGS_phase [N = {N}, N_sub = {N_subsample}, CS type = {CS_type}]\n"
        procedure_diagnostic = []

        assert self.S_alpha == self.S_beta

        if "LE_sol" not in self.checklist:
            self.log.write(f"ERROR: LEGS method requires LE solution to be known. Aborting...")
            self.log.exit()
            return(None)

        cur_SECS_heatmap = self.LE_sol["exp"]

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

        self.disk_jockey.commit_datum_bulk(dataset_label, "basis_samples", cur_sample.get_z_tensor())
        self.disk_jockey.commit_datum_bulk(dataset_label, "result_energy_states", csv_sol)
        self.diagnostics_log.append({f"find_ground_state_LEGS_phase ({dataset_label})" : procedure_diagnostic})

        self.measured_datasets.append(dataset_label)

        self.log.exit()

        return(N_vals, convergence_sols)

    def find_ground_state_LEGS_first_order(self, **kwargs):
        # kwargs:
        #     -N: Sample size
        #     -N_sub: Subsample size
        #     -CS: Type of CS to use. Not all CS types support LEGS!
        #     -sigma: standard deviation of every Z parameter. default 0.05. Can be a number, a list of ndarrays, or a magic word
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

        if "sigma" in kwargs:
            sigma = kwargs["sigma"]
        else:
            sigma = 0.05

        if "dataset_label" in kwargs:
            dataset_label = kwargs["dataset_label"]
        else:
            dataset_label = f"LEGS_first_order_{N}_{N_subsample}_{CS_type}"


        self.log.enter(f"Obtaining the ground state with the method \"LEGS 1st ord.\" [N = {N}, N_sub = {N_subsample}, CS = {CS_type}]", 1)


        # Disk jockey node creation and metadata storage
        self.disk_jockey.create_data_nodes({dataset_label : {"basis_samples" : "pkl", "result_energy_states" : "csv"}})
        self.disk_jockey.commit_metadatum(dataset_label, "basis_samples", {"CS" : CS_type, "N" : N, "N_sub" : N_subsample, "sigma" : sigma})
        self.user_actions += f"find_ground_state_LEGS_first_order [N = {N}, N_sub = {N_subsample}, CS type = {CS_type}, sigma = {sigma}]\n"
        procedure_diagnostic = []

        if "LE_sol" not in self.checklist:
            self.log.write(f"ERROR: LEGS method requires LE solution to be known. Aborting...")
            self.log.exit()
            return(None)

        cur_LE_heatmap = self.LE_sol["exp"] #["g"/"a"/"b"/"ab"] for first order

        # We now manually sample Thouless states guided by the SECS heatmap
        cur_sample = CS_sample(self, ground_state_solver.coherent_state_types[CS_type], add_ref_state = True)

        # We flatten the heatmap
        len_a = self.S_alpha * (self.mol.nao - self.S_alpha)
        len_b = self.S_beta * (self.mol.nao - self.S_beta)
        a_ij_to_i = lambda i,j : i * (self.mol.nao - self.S_alpha) + j
        b_ij_to_i = lambda i,j : self.S_alpha * (self.mol.nao - self.S_alpha) + i * (self.mol.nao - self.S_beta) + j
        means = np.zeros( len_a + len_b )
        stds = sigma + np.zeros( len_a + len_b) #TODO what if sigma is not always the same?

        for i in range(self.mol.nao - self.S_alpha):
            for j in range(self.S_alpha):
                # j -> i on alpha
                means[a_ij_to_i(i, j)] = cur_LE_heatmap["a"][i][j]
        for i in range(self.mol.nao - self.S_beta):
            for j in range(self.S_beta):
                # j -> i on beta
                means[b_ij_to_i(i, j)] = cur_LE_heatmap["b"][i][j]

        product_means = np.outer(means, means) + np.diag(stds * stds) # we start with no covariance except natural widths and then change the off-diagonal block

        for i in range(self.mol.nao - self.S_alpha):
            for j in range(self.S_alpha):
                for k in range(self.mol.nao - self.S_beta):
                    for l in range(self.S_beta):
                        # j -> i on alpha, l -> k on beta
                        product_means[a_ij_to_i(i, j)][b_ij_to_i(k, l)] = cur_LE_heatmap["ab"][i][j][k][l]
                        product_means[b_ij_to_i(k, l)][a_ij_to_i(i, j)] = cur_LE_heatmap["ab"][i][j][k][l]

        cov_matrix = product_means - np.outer(means, means) # This is what we sample by with Cholesky

        eigvals, eigvecs = np.linalg.eigh(cov_matrix)
        min_eigval = min(eigvals)
        if min_eigval <= 0.0:
            self.log.write(f"Warning: Covariance matrix is not positive-semidefinite (min eig = {min_eigval})")

        # We pre-sample the parameters
        rand_X = functions.sample_with_autocorrelation_safe(means, cov_matrix, N * N_subsample)
        raw_Z_sample = [
            np.zeros((N, N_subsample, self.mol.nao - self.S_alpha, self.S_alpha)),
            np.zeros((N, N_subsample, self.mol.nao - self.S_beta, self.S_beta))
            ]
        for n in range(N):
            for n_sub in range(N_subsample):
                for i in range(self.mol.nao - self.S_alpha):
                    for j in range(self.S_alpha):
                        # j -> i on alpha
                        raw_Z_sample[0][n][n_sub][i][j] = rand_X[n * N_subsample + n_sub][a_ij_to_i(i, j)]
                for i in range(self.mol.nao - self.S_beta):
                    for j in range(self.S_beta):
                        # j -> i on beta
                        raw_Z_sample[1][n][n_sub][i][j] = rand_X[n * N_subsample + n_sub][b_ij_to_i(i, j)]


        assert CS_type == "Thouless"

        N_vals = [1]
        convergence_sols = [self.reference_state_energy]

        msg = f"Conditioned sampling with ground state search on {N} states, each taken from {N_subsample} random states"
        #new_sem_ID = self.semaphor.create_event(np.linspace(0, N_subsample * ((N + 2) * (N + 1) / 2 - 1) + 1, 1000 + 1), msg)
        self.log.enter(msg, 1, True, tau_space = np.linspace(0, N_subsample * ((N + 2) * (N + 1) / 2 - 1) + 1, 1000 + 1))

        for n in range(N):
            # We add the best out of 10 random states
            cur_subsample = []
            for n_sub in range(N_subsample):

                rand_z_alpha = ground_state_solver.coherent_state_types[CS_type](self.mol.nao, self.S_alpha, raw_Z_sample[0][n][n_sub])
                rand_z_beta = ground_state_solver.coherent_state_types[CS_type](self.mol.nao, self.S_beta, raw_Z_sample[1][n][n_sub])
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
        self.diagnostics_log.append({f"find_ground_state_LEGS_first_order ({dataset_label})" : procedure_diagnostic})

        self.measured_datasets.append(dataset_label)

        self.log.exit()
        return(N_vals, convergence_sols)

    def find_ground_state_LEGS_mixed_spin_covariance(self, **kwargs):

        # This method takes into account that even if a single-excited state
        # has low presence in the LE sol., it may still be a significant part
        # of the sampling distribution in Z because of its second orders (which
        # are typically mixed-spin double excitations).
        # Here, we take the single-excitation state overlaps as expectation
        # values of Z_ij,s, and mixed-spin double excitation state overlaps as
        # elements of the covariance matrix (whose absolute value also
        # contributes to the variance of Z_ij,s)

        # LE is found via SCF

        # kwargs:
        #     -N: Sample size
        #     -N_sub: Subsample size
        #     -sigma: base standard deviation of every Z parameter. default 0.001. Can be a number, a list of ndarrays, or a magic word
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

        if "sigma" in kwargs:
            sigma = kwargs["sigma"]
        else:
            sigma = 1e-3

        if "dataset_label" in kwargs:
            dataset_label = kwargs["dataset_label"]
        else:
            dataset_label = f"LEGS_mixed_spin_covariance_{N}_{N_subsample}_{CS_type}"

        self.log.enter(f"Obtaining the ground state with the method \"LEGS mixed spin covariance\" [N = {N}, N_sub = {N_subsample}, CS = {CS_type}, sigma : {sigma}]", 1)


        # Disk jockey node creation and metadata storage
        self.disk_jockey.create_data_nodes({dataset_label : {"basis_samples" : "pkl", "result_energy_states" : "csv"}})
        self.disk_jockey.commit_metadatum(dataset_label, "basis_samples", {"CS" : CS_type, "N" : N, "N_sub" : N_subsample, "sigma" : sigma})
        self.user_actions += f"find_ground_state_LEGS_mixed_spin_covariance [N = {N}, N_sub = {N_subsample}, CS type = {CS_type}, sigma : {sigma}]\n"
        procedure_diagnostic = []

        if "LE_sol" not in self.checklist:
            self.log.write(f"ERROR: LEGS method requires LE solution to be known. Aborting...")
            self.log.exit()
            return(None)

        cur_LE_heatmap = self.LE_sol["exp"] #["g"/"a"/"b"/"ab"] for first order

        # We now manually sample Thouless states guided by the SECS heatmap
        cur_sample = CS_sample(self, ground_state_solver.coherent_state_types[CS_type], add_ref_state = True)

        # We flatten the heatmap
        len_a = self.S_alpha * (self.mol.nao - self.S_alpha)
        len_b = self.S_beta * (self.mol.nao - self.S_beta)
        a_ij_to_i = lambda i,j : i * self.S_alpha + j
        b_ij_to_i = lambda i,j : self.S_alpha * (self.mol.nao - self.S_alpha) + i * self.S_beta + j
        means = np.zeros( len_a + len_b )
        stds = sigma + np.zeros( len_a + len_b) #TODO what if sigma is not always the same?

        for i in range(self.mol.nao - self.S_alpha):
            for j in range(self.S_alpha):
                # j -> i on alpha
                means[a_ij_to_i(i, j)] = cur_LE_heatmap["a"][i][j]
        for i in range(self.mol.nao - self.S_beta):
            for j in range(self.S_beta):
                # j -> i on beta
                means[b_ij_to_i(i, j)] = cur_LE_heatmap["b"][i][j]

        product_means = np.outer(means, means) # we start with no covariance except natural widths and then change the off-diagonal block

        for i in range(self.mol.nao - self.S_alpha):
            for j in range(self.S_alpha):
                for k in range(self.mol.nao - self.S_beta):
                    for l in range(self.S_beta):
                        # j -> i on alpha, l -> k on beta
                        product_means[a_ij_to_i(i, j)][b_ij_to_i(k, l)] = cur_LE_heatmap["ab"][i][j][k][l]
                        product_means[b_ij_to_i(k, l)][a_ij_to_i(i, j)] = cur_LE_heatmap["ab"][i][j][k][l]

        cov_matrix = product_means - np.outer(means, means) # This is what we sample by with Cholesky
        # Now we add the diagonal terms, i.e. the variance
        required_variance = np.sum(np.abs(cov_matrix), axis = 1) # minimal diag terms to achieve positive semidefiniteness
        cov_matrix += np.diag(required_variance + stds * stds)

        self.log.write(f"Default variance: {stds * stds}; required additional variance: {required_variance}")

        eigvals, eigvecs = np.linalg.eigh(cov_matrix)
        min_eigval = min(eigvals)
        if min_eigval <= 0.0:
            self.log.write(f"Warning: Covariance matrix is not positive-semidefinite (min eig = {min_eigval})")

        # We pre-sample the parameters
        rand_X = functions.sample_with_autocorrelation_safe(means, cov_matrix, N * N_subsample)
        raw_Z_sample = [
            np.zeros((N, N_subsample, self.mol.nao - self.S_alpha, self.S_alpha)),
            np.zeros((N, N_subsample, self.mol.nao - self.S_beta, self.S_beta))
            ]
        for n in range(N):
            for n_sub in range(N_subsample):
                for i in range(self.mol.nao - self.S_alpha):
                    for j in range(self.S_alpha):
                        # j -> i on alpha
                        raw_Z_sample[0][n][n_sub][i][j] = rand_X[n * N_subsample + n_sub][a_ij_to_i(i, j)]
                for i in range(self.mol.nao - self.S_beta):
                    for j in range(self.S_beta):
                        # j -> i on beta
                        raw_Z_sample[1][n][n_sub][i][j] = rand_X[n * N_subsample + n_sub][b_ij_to_i(i, j)]


        assert CS_type == "Thouless"

        N_vals = [1]
        convergence_sols = [self.reference_state_energy]

        msg = f"Conditioned sampling with ground state search on {N} states, each taken from {N_subsample} random states"
        #new_sem_ID = self.semaphor.create_event(np.linspace(0, N_subsample * ((N + 2) * (N + 1) / 2 - 1) + 1, 1000 + 1), msg)
        self.log.enter(msg, 1, True, tau_space = np.linspace(0, N_subsample * ((N + 2) * (N + 1) / 2 - 1) + 1, 1000 + 1))

        for n in range(N):
            # We add the best out of 10 random states
            cur_subsample = []
            for n_sub in range(N_subsample):

                rand_z_alpha = ground_state_solver.coherent_state_types[CS_type](self.mol.nao, self.S_alpha, raw_Z_sample[0][n][n_sub])
                rand_z_beta = ground_state_solver.coherent_state_types[CS_type](self.mol.nao, self.S_beta, raw_Z_sample[1][n][n_sub])
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
        self.diagnostics_log.append({f"find_ground_state_LEGS_mixed_spin_covariance ({dataset_label})" : procedure_diagnostic})

        self.measured_datasets.append(dataset_label)

        self.log.exit()
        return(N_vals, convergence_sols)

    def find_ground_state_LEGS_Zombie_cov(self, **kwargs):

        # This method uses Zombie (Qubit) states.
        # It looks at the total norm of the states in | LE > which have i-th MO
        # occupied, i.e. < LE | b_i\hc b_i | LE >, and uses this to estimate
        # E[z_i^2]. The means are inferred from the variances by lowering from
        # 1 for pi_1 and leaving them at 0 for pi_0.

        # kwargs:
        #     -N: Sample size
        #     -N_sub: Subsample size
        #     ------------------------
        #     -dataset_label: if present, this will label the dataset in disc jockey. Otherwise, label is generated from other kwargs.

        # -------------------- Parameter initialisation

        assert "N" in kwargs
        N = kwargs["N"]

        if "N_sub" in kwargs:
            N_subsample = kwargs["N_sub"]
        else:
            N_subsample = 1


        if "dataset_label" in kwargs:
            dataset_label = kwargs["dataset_label"]
        else:
            dataset_label = f"LEGS_Zombie_cov_{N}_{N_subsample}"

        self.log.enter(f"Obtaining the ground state with the method \"LEGS Zombie cov\" [N = {N}, N_sub = {N_subsample}]", 1)


        # Disk jockey node creation and metadata storage
        self.disk_jockey.create_data_nodes({dataset_label : {"basis_samples" : "pkl", "result_energy_states" : "csv"}})
        self.disk_jockey.commit_metadatum(dataset_label, "basis_samples", {"N" : N, "N_sub" : N_subsample})
        self.user_actions += f"find_ground_state_LEGS_Zombie_cov [N = {N}, N_sub = {N_subsample}]\n"
        procedure_diagnostic = []

        if "LE_sol" not in self.checklist:
            self.log.write(f"ERROR: LEGS method requires LE solution to be known. Aborting...")
            self.log.exit()
            return(None)

        # We now manually sample Thouless states guided by the LE solution
        cur_sample = CS_sample(self, CS_Qubit, add_ref_state = True)

        # We find the means and the covariances
        spin_idx_dict = {"a" : 0, "b" : 1}
        spat_to_spin_idx = lambda sigma, i : spin_idx_dict[sigma] * self.mol.nao + i
        means = np.zeros( 2 * self.mol.nao )
        cov = np.zeros( (2 * self.mol.nao, 2 * self.mol.nao) )


        self.log.print_matrix(self.LE_sol["red"][:self.mol.nao,:self.mol.nao], "qubit transition matrix", dec_points = 5)




        # initialise means
        for i in range(self.S_alpha):
            means[spat_to_spin_idx("a", i)] = 0.5 * (1.0 + np.sqrt(2.0 * self.LE_sol["red"][spat_to_spin_idx("a", i)][spat_to_spin_idx("a", i)] - 1.0))
        for i in range(self.S_beta):
            means[spat_to_spin_idx("b", i)] = 0.5 * (1.0 + np.sqrt(2.0 * self.LE_sol["red"][spat_to_spin_idx("b", i)][spat_to_spin_idx("b", i)] - 1.0))

        # initialise variances
        for i in range(self.mol.nao):
            norm_coef = 1.0
            if i >= self.S_alpha:
                norm_coef = 1.0 / self.S_alpha
            cov[spat_to_spin_idx("a", i)][spat_to_spin_idx("a", i)] = norm_coef * self.LE_sol["red"][spat_to_spin_idx("a", i)][spat_to_spin_idx("a", i)] - means[spat_to_spin_idx("a", i)] * means[spat_to_spin_idx("a", i)]
        for i in range(self.mol.nao):
            norm_coef = 1.0
            if i >= self.S_beta:
                norm_coef = 1.0 / self.S_beta
            cov[spat_to_spin_idx("b", i)][spat_to_spin_idx("b", i)] = norm_coef * self.LE_sol["red"][spat_to_spin_idx("b", i)][spat_to_spin_idx("b", i)] - means[spat_to_spin_idx("b", i)] * means[spat_to_spin_idx("b", i)]

        # initialise covariances
        # alpha-alpha
        for i in range(self.mol.nao):
            for j in range(i + 1, self.mol.nao):
                if i < self.S_alpha and j < self.S_alpha:
                    # agnostic
                    continue
                elif i < self.S_alpha and j > self.S_alpha:
                    cov[spat_to_spin_idx("a", i)][spat_to_spin_idx("a", j)] = self.LE_sol["red"][spat_to_spin_idx("a", i)][spat_to_spin_idx("a", j)] - means[spat_to_spin_idx("a", i)] * means[spat_to_spin_idx("a", j)]
                    cov[spat_to_spin_idx("a", j)][spat_to_spin_idx("a", i)] = cov[spat_to_spin_idx("a", i)][spat_to_spin_idx("a", j)] # conjugate?
                elif i > self.S_alpha and j > self.S_alpha:
                    cov[spat_to_spin_idx("a", i)][spat_to_spin_idx("a", j)] = 1.0 / self.S_alpha * self.LE_sol["red"][spat_to_spin_idx("a", i)][spat_to_spin_idx("a", j)] - means[spat_to_spin_idx("a", i)] * means[spat_to_spin_idx("a", j)]
                    cov[spat_to_spin_idx("a", j)][spat_to_spin_idx("a", i)] = cov[spat_to_spin_idx("a", i)][spat_to_spin_idx("a", j)] # conjugate?
        # beta-beta
        for i in range(self.mol.nao):
            for j in range(i + 1, self.mol.nao):
                if i < self.S_beta and j < self.S_beta:
                    # agnostic
                    continue
                elif i < self.S_beta and j > self.S_beta:
                    cov[spat_to_spin_idx("b", i)][spat_to_spin_idx("b", j)] = self.LE_sol["red"][spat_to_spin_idx("b", i)][spat_to_spin_idx("b", j)] - means[spat_to_spin_idx("b", i)] * means[spat_to_spin_idx("b", j)]
                    cov[spat_to_spin_idx("b", j)][spat_to_spin_idx("b", i)] = cov[spat_to_spin_idx("b", i)][spat_to_spin_idx("b", j)] # conjugate?
                elif i > self.S_beta and j > self.S_beta:
                    cov[spat_to_spin_idx("b", i)][spat_to_spin_idx("b", j)] = 1.0 / self.S_beta * self.LE_sol["red"][spat_to_spin_idx("b", i)][spat_to_spin_idx("b", j)] - means[spat_to_spin_idx("b", i)] * means[spat_to_spin_idx("b", j)]
                    cov[spat_to_spin_idx("b", j)][spat_to_spin_idx("b", i)] = cov[spat_to_spin_idx("b", i)][spat_to_spin_idx("b", j)] # conjugate?

        # alpha-beta
        # We remain agnostic!

        dec_point = 4

        diagnostic_table = []
        for i in range(self.mol.nao):
            diagnostic_table.append([
                np.round(means[spat_to_spin_idx("a", i)], dec_point),
                np.round(cov[spat_to_spin_idx("a", i)][spat_to_spin_idx("a", i)], dec_point),
                np.round(np.sum(np.abs(cov[spat_to_spin_idx("a", i)])) - cov[spat_to_spin_idx("a", i)][spat_to_spin_idx("a", i)], dec_point),
                np.round(2 * cov[spat_to_spin_idx("a", i)][spat_to_spin_idx("a", i)] - np.sum(np.abs(cov[spat_to_spin_idx("a", i)])), dec_point)
                ])

        self.log.print_table(
            table_name = "LE Zombie diag.",
            column_names = ["mean", "var", "gershgorin disc", "leeway"],
            row_names = np.arange(1, self.mol.nao + 1, 1, dtype = int),
            list_of_rows = diagnostic_table
            )

        self.log.print_matrix(cov[:self.mol.nao,:self.mol.nao], "covariance matrix", dec_points = 5)

        """print("")
        print("- Means -")
        print(means)
        print("- Covariance matrix -")
        print(cov)
        print("- Covariance matric after removing negative eigenvalues -")
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.clip(eigvals, 0, None)
        L = eigvecs * eigvals[None, :]
        print(L)
        print("")"""


        """self.log.write("Gershgorin-proving the covariance matrix by increasing variances by their respective Gershgorin disc radii...")
        for i in range(2 * self.mol.nao):
            cov[i][i] = np.sum(np.abs(cov[i]))"""

        eigvals, eigvecs = np.linalg.eigh(cov)
        min_eigval = min(eigvals)
        if min_eigval <= 0.0:
            self.log.write(f"Warning: Covariance matrix is not positive-semidefinite (min eig = {min_eigval})")

        # We pre-sample the parameters
        rand_X = functions.sample_with_autocorrelation_safe(means, cov, N * N_subsample)
        raw_Z_sample = [
            np.zeros((N, N_subsample, self.mol.nao)),
            np.zeros((N, N_subsample, self.mol.nao))
            ]
        for n in range(N):
            for n_sub in range(N_subsample):
                for i in range(self.mol.nao):
                    raw_Z_sample[0][n][n_sub][i] = rand_X[n * N_subsample + n_sub][spat_to_spin_idx("a", i)]
                    raw_Z_sample[1][n][n_sub][i] = rand_X[n * N_subsample + n_sub][spat_to_spin_idx("b", i)]

        N_vals = [1]
        convergence_sols = [self.reference_state_energy]

        msg = f"Conditioned sampling with ground state search on {N} states, each taken from {N_subsample} random states"
        #new_sem_ID = self.semaphor.create_event(np.linspace(0, N_subsample * ((N + 2) * (N + 1) / 2 - 1) + 1, 1000 + 1), msg)
        self.log.enter(msg, 1, True, tau_space = np.linspace(0, N_subsample * ((N + 2) * (N + 1) / 2 - 1) + 1, 1000 + 1))

        for n in range(N):
            # We add the best out of 10 random states
            cur_subsample = []
            for n_sub in range(N_subsample):

                rand_z_alpha = CS_Qubit(self.mol.nao, self.S_alpha, raw_Z_sample[0][n][n_sub])
                rand_z_beta = CS_Qubit(self.mol.nao, self.S_beta, raw_Z_sample[1][n][n_sub])
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
        self.diagnostics_log.append({f"find_ground_state_LEGS_Zombie_cov ({dataset_label})" : procedure_diagnostic})

        self.measured_datasets.append(dataset_label)

        self.log.exit()
        return(N_vals, convergence_sols)

    def find_ground_state_LEGS_Zombie_cov_ALT(self, **kwargs):

        # This method uses Zombie (Qubit) states.
        # It looks at the total norm of the states in | LE > which have i-th MO
        # occupied, i.e. < LE | b_i\hc b_i | LE >, and uses this to estimate
        # E[z_i^2]. The means are inferred from the variances by lowering from
        # 1 for pi_1 and leaving them at 0 for pi_0.

        # kwargs:
        #     -N: Sample size
        #     -N_sub: Subsample size
        #     ------------------------
        #     -dataset_label: if present, this will label the dataset in disc jockey. Otherwise, label is generated from other kwargs.

        # -------------------- Parameter initialisation

        assert "N" in kwargs
        N = kwargs["N"]

        if "N_sub" in kwargs:
            N_subsample = kwargs["N_sub"]
        else:
            N_subsample = 1


        if "dataset_label" in kwargs:
            dataset_label = kwargs["dataset_label"]
        else:
            dataset_label = f"LEGS_Zombie_cov_{N}_{N_subsample}"

        self.log.enter(f"Obtaining the ground state with the method \"LEGS Zombie cov\" [N = {N}, N_sub = {N_subsample}]", 1)


        # Disk jockey node creation and metadata storage
        self.disk_jockey.create_data_nodes({dataset_label : {"basis_samples" : "pkl", "result_energy_states" : "csv"}})
        self.disk_jockey.commit_metadatum(dataset_label, "basis_samples", {"N" : N, "N_sub" : N_subsample})
        self.user_actions += f"find_ground_state_LEGS_Zombie_cov [N = {N}, N_sub = {N_subsample}]\n"
        procedure_diagnostic = []

        if "LE_sol" not in self.checklist:
            self.log.write(f"ERROR: LEGS method requires LE solution to be known. Aborting...")
            self.log.exit()
            return(None)

        # We now manually sample Thouless states guided by the LE solution
        cur_sample = CS_sample(self, CS_Qubit, add_ref_state = True)

        # We find the means and the covariances
        spin_idx_dict = {"a" : 0, "b" : 1}
        spat_to_spin_idx = lambda sigma, i : spin_idx_dict[sigma] * self.mol.nao + i
        means = np.zeros( 2 * self.mol.nao )
        sq_means = np.zeros( 2 * self.mol.nao )
        variances = np.zeros( 2 * self.mol.nao )
        cov = np.zeros( (2 * self.mol.nao, 2 * self.mol.nao) )


        self.log.print_matrix(self.LE_sol["red"][:self.mol.nao,:self.mol.nao], "qubit transition matrix", dec_points = 5)

        # initialise means and variances Dima style (mean = std)
        self.log.write("Means and variances Dima style (mean = std, and derived from the calculated mean of square)")
        for i in range(self.mol.nao):
            norm_coef = 1.0
            if i >= self.S_alpha:
                norm_coef = 1.0 / self.S_alpha
            sq_means[spat_to_spin_idx("a", i)] = norm_coef * self.LE_sol["red"][spat_to_spin_idx("a", i)][spat_to_spin_idx("a", i)]
            if i < self.S_alpha:
                means[spat_to_spin_idx("a", i)] = np.sqrt(norm_coef * self.LE_sol["red"][spat_to_spin_idx("a", i)][spat_to_spin_idx("a", i)] / 2.0)
            else:
                means[spat_to_spin_idx("a", i)] = 0
            variances[spat_to_spin_idx("a", i)] = sq_means[spat_to_spin_idx("a", i)] - means[spat_to_spin_idx("a", i)] * means[spat_to_spin_idx("a", i)]
        for i in range(self.mol.nao):
            norm_coef = 1.0
            if i >= self.S_beta:
                norm_coef = 1.0 / self.S_beta
            sq_means[spat_to_spin_idx("b", i)] = norm_coef * self.LE_sol["red"][spat_to_spin_idx("b", i)][spat_to_spin_idx("b", i)]
            if i < self.S_beta:
                means[spat_to_spin_idx("b", i)] = np.sqrt(norm_coef * self.LE_sol["red"][spat_to_spin_idx("b", i)][spat_to_spin_idx("b", i)] / 2.0)
            else:
                means[spat_to_spin_idx("b", i)] = 0
            variances[spat_to_spin_idx("b", i)] = sq_means[spat_to_spin_idx("b", i)] - means[spat_to_spin_idx("b", i)] * means[spat_to_spin_idx("b", i)]

        cov = np.diag(variances)

        # initialise covariances
        # alpha-alpha
        for i in range(self.mol.nao):
            for j in range(i + 1, self.mol.nao):
                if i < self.S_alpha and j < self.S_alpha:
                    # agnostic
                    continue
                elif i < self.S_alpha and j > self.S_alpha:
                    cov[spat_to_spin_idx("a", i)][spat_to_spin_idx("a", j)] = self.LE_sol["red"][spat_to_spin_idx("a", i)][spat_to_spin_idx("a", j)] - means[spat_to_spin_idx("a", i)] * means[spat_to_spin_idx("a", j)]
                    cov[spat_to_spin_idx("a", j)][spat_to_spin_idx("a", i)] = cov[spat_to_spin_idx("a", i)][spat_to_spin_idx("a", j)] # conjugate?
                elif i > self.S_alpha and j > self.S_alpha:
                    cov[spat_to_spin_idx("a", i)][spat_to_spin_idx("a", j)] = 1.0 / self.S_alpha * self.LE_sol["red"][spat_to_spin_idx("a", i)][spat_to_spin_idx("a", j)] - means[spat_to_spin_idx("a", i)] * means[spat_to_spin_idx("a", j)]
                    cov[spat_to_spin_idx("a", j)][spat_to_spin_idx("a", i)] = cov[spat_to_spin_idx("a", i)][spat_to_spin_idx("a", j)] # conjugate?
        # beta-beta
        for i in range(self.mol.nao):
            for j in range(i + 1, self.mol.nao):
                if i < self.S_beta and j < self.S_beta:
                    # agnostic
                    continue
                elif i < self.S_beta and j > self.S_beta:
                    cov[spat_to_spin_idx("b", i)][spat_to_spin_idx("b", j)] = self.LE_sol["red"][spat_to_spin_idx("b", i)][spat_to_spin_idx("b", j)] - means[spat_to_spin_idx("b", i)] * means[spat_to_spin_idx("b", j)]
                    cov[spat_to_spin_idx("b", j)][spat_to_spin_idx("b", i)] = cov[spat_to_spin_idx("b", i)][spat_to_spin_idx("b", j)] # conjugate?
                elif i > self.S_beta and j > self.S_beta:
                    cov[spat_to_spin_idx("b", i)][spat_to_spin_idx("b", j)] = 1.0 / self.S_beta * self.LE_sol["red"][spat_to_spin_idx("b", i)][spat_to_spin_idx("b", j)] - means[spat_to_spin_idx("b", i)] * means[spat_to_spin_idx("b", j)]
                    cov[spat_to_spin_idx("b", j)][spat_to_spin_idx("b", i)] = cov[spat_to_spin_idx("b", i)][spat_to_spin_idx("b", j)] # conjugate?

        # alpha-beta
        # We remain agnostic!



        dec_point = 4

        diagnostic_table = []
        for i in range(self.mol.nao):
            diagnostic_table.append([
                np.round(means[spat_to_spin_idx("a", i)], dec_point),
                np.round(cov[spat_to_spin_idx("a", i)][spat_to_spin_idx("a", i)], dec_point),
                np.round(np.sum(np.abs(cov[spat_to_spin_idx("a", i)])) - cov[spat_to_spin_idx("a", i)][spat_to_spin_idx("a", i)], dec_point),
                np.round(2 * cov[spat_to_spin_idx("a", i)][spat_to_spin_idx("a", i)] - np.sum(np.abs(cov[spat_to_spin_idx("a", i)])), dec_point)
                ])

        self.log.print_table(
            table_name = "LE Zombie diag.",
            column_names = ["mean", "var", "gershgorin disc", "leeway"],
            row_names = np.arange(1, self.mol.nao + 1, 1, dtype = int),
            list_of_rows = diagnostic_table
            )

        self.log.print_matrix(cov[:self.mol.nao,:self.mol.nao], "covariance matrix", dec_points = 5)

        """print("")
        print("- Means -")
        print(means)
        print("- Covariance matrix -")
        print(cov)
        print("- Covariance matric after removing negative eigenvalues -")
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.clip(eigvals, 0, None)
        L = eigvecs * eigvals[None, :]
        print(L)
        print("")"""


        """self.log.write("Gershgorin-proving the covariance matrix by increasing variances by their respective Gershgorin disc radii...")
        for i in range(2 * self.mol.nao):
            cov[i][i] = np.sum(np.abs(cov[i]))"""

        eigvals, eigvecs = np.linalg.eigh(cov)
        min_eigval = min(eigvals)
        if min_eigval <= 0.0:
            self.log.write(f"Warning: Covariance matrix is not positive-semidefinite (min eig = {min_eigval})")

        # We pre-sample the parameters
        rand_X = functions.sample_with_autocorrelation_safe(means, cov, N * N_subsample)
        raw_Z_sample = [
            np.zeros((N, N_subsample, self.mol.nao)),
            np.zeros((N, N_subsample, self.mol.nao))
            ]
        for n in range(N):
            for n_sub in range(N_subsample):
                for i in range(self.mol.nao):
                    raw_Z_sample[0][n][n_sub][i] = rand_X[n * N_subsample + n_sub][spat_to_spin_idx("a", i)]
                    raw_Z_sample[1][n][n_sub][i] = raw_Z_sample[0][n][n_sub][i] #rand_X[n * N_subsample + n_sub][spat_to_spin_idx("b", i)]

        N_vals = [1]
        convergence_sols = [self.reference_state_energy]

        msg = f"Conditioned sampling with ground state search on {N} states, each taken from {N_subsample} random states"
        #new_sem_ID = self.semaphor.create_event(np.linspace(0, N_subsample * ((N + 2) * (N + 1) / 2 - 1) + 1, 1000 + 1), msg)
        self.log.enter(msg, 1, True, tau_space = np.linspace(0, N_subsample * ((N + 2) * (N + 1) / 2 - 1) + 1, 1000 + 1))

        for n in range(N):
            # We add the best out of 10 random states
            cur_subsample = []
            for n_sub in range(N_subsample):

                rand_z_alpha = CS_Qubit(self.mol.nao, self.S_alpha, raw_Z_sample[0][n][n_sub])
                rand_z_beta = CS_Qubit(self.mol.nao, self.S_beta, raw_Z_sample[1][n][n_sub])
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
        self.diagnostics_log.append({f"find_ground_state_LEGS_Zombie_cov ({dataset_label})" : procedure_diagnostic})

        self.measured_datasets.append(dataset_label)

        self.log.exit()
        return(N_vals, convergence_sols)

    def find_ground_state_LEGS_Zombie_cov_SRRM(self, **kwargs):

        # This method uses Zombie (Qubit) states.
        # It looks at the total norm of the states in | LE > which have i-th MO
        # occupied, i.e. < LE | b_i\hc b_i | LE >, and uses this to estimate
        # E[z_i^2]. The means are inferred from the variances by lowering from
        # 1 for pi_1 and leaving them at 0 for pi_0.

        # The difference from ALT is that when considering b_i b_j acting on one spin subspace, we sum over the abs values
        # of all contributions from the other spin subspace to account for simultaneous excitations!

        # kwargs:
        #     -N: Sample size
        #     -N_sub: Subsample size
        #     ------------------------
        #     -dataset_label: if present, this will label the dataset in disc jockey. Otherwise, label is generated from other kwargs.

        # -------------------- Parameter initialisation

        assert "N" in kwargs
        N = kwargs["N"]

        if "N_sub" in kwargs:
            N_subsample = kwargs["N_sub"]
        else:
            N_subsample = 1


        if "dataset_label" in kwargs:
            dataset_label = kwargs["dataset_label"]
        else:
            dataset_label = f"LEGS_Zombie_cov_SRRM_{N}_{N_subsample}"

        self.log.enter(f"Obtaining the ground state with the method \"LEGS Zombie cov SRRM\" [N = {N}, N_sub = {N_subsample}]", 1)


        # Disk jockey node creation and metadata storage
        self.disk_jockey.create_data_nodes({dataset_label : {"basis_samples" : "pkl", "result_energy_states" : "csv"}})
        self.disk_jockey.commit_metadatum(dataset_label, "basis_samples", {"N" : N, "N_sub" : N_subsample})
        self.user_actions += f"find_ground_state_LEGS_Zombie_cov_SRRM [N = {N}, N_sub = {N_subsample}]\n"
        procedure_diagnostic = []

        if "LE_sol" not in self.checklist:
            self.log.write(f"ERROR: LEGS method requires LE solution to be known. Aborting...")
            self.log.exit()
            return(None)

        # We now manually sample Thouless states guided by the LE solution
        cur_sample = CS_sample(self, CS_Qubit, add_ref_state = True)

        # We find the means and the covariances
        spin_idx_dict = {"a" : 0, "b" : 1}
        spat_to_spin_idx = lambda sigma, i : spin_idx_dict[sigma] * self.mol.nao + i
        means = np.zeros( 2 * self.mol.nao )
        sq_means = np.zeros( 2 * self.mol.nao )
        variances = np.zeros( 2 * self.mol.nao )
        cov = np.zeros( (2 * self.mol.nao, 2 * self.mol.nao) )


        self.log.print_matrix(self.LE_sol["SRRM"][:self.mol.nao,:self.mol.nao], "qubit transition matrix", dec_points = 5)

        # initialise means and variances Dima style (mean = std)
        self.log.write("Means and variances Dima style (mean = std, and derived from the calculated mean of square)")
        for i in range(self.mol.nao):
            norm_coef = 1.0
            if i >= self.S_alpha:
                norm_coef = 1.0 / self.S_alpha
            sq_means[spat_to_spin_idx("a", i)] = norm_coef * self.LE_sol["SRRM"][spat_to_spin_idx("a", i)][i]
            if i < self.S_alpha:
                means[spat_to_spin_idx("a", i)] = np.sqrt(norm_coef * self.LE_sol["SRRM"][spat_to_spin_idx("a", i)][i] / 2.0)
            else:
                means[spat_to_spin_idx("a", i)] = 0
            variances[spat_to_spin_idx("a", i)] = sq_means[spat_to_spin_idx("a", i)] - means[spat_to_spin_idx("a", i)] * means[spat_to_spin_idx("a", i)]
        for i in range(self.mol.nao):
            norm_coef = 1.0
            if i >= self.S_beta:
                norm_coef = 1.0 / self.S_beta
            sq_means[spat_to_spin_idx("b", i)] = norm_coef * self.LE_sol["SRRM"][spat_to_spin_idx("b", i)][i]
            if i < self.S_beta:
                means[spat_to_spin_idx("b", i)] = np.sqrt(norm_coef * self.LE_sol["SRRM"][spat_to_spin_idx("b", i)][i] / 2.0)
            else:
                means[spat_to_spin_idx("b", i)] = 0
            variances[spat_to_spin_idx("b", i)] = sq_means[spat_to_spin_idx("b", i)] - means[spat_to_spin_idx("b", i)] * means[spat_to_spin_idx("b", i)]

        cov = np.diag(variances)

        # initialise covariances
        # alpha-alpha
        for i in range(self.mol.nao):
            for j in range(i + 1, self.mol.nao):
                if i < self.S_alpha and j < self.S_alpha:
                    # agnostic
                    continue
                elif i < self.S_alpha and j > self.S_alpha:
                    cov[spat_to_spin_idx("a", i)][spat_to_spin_idx("a", j)] = self.LE_sol["SRRM"][spat_to_spin_idx("a", i)][j] - means[spat_to_spin_idx("a", i)] * means[spat_to_spin_idx("a", j)]
                    cov[spat_to_spin_idx("a", j)][spat_to_spin_idx("a", i)] = cov[spat_to_spin_idx("a", i)][spat_to_spin_idx("a", j)] # conjugate?
                elif i > self.S_alpha and j > self.S_alpha:
                    cov[spat_to_spin_idx("a", i)][spat_to_spin_idx("a", j)] = 1.0 / self.S_alpha * self.LE_sol["SRRM"][spat_to_spin_idx("a", i)][j] - means[spat_to_spin_idx("a", i)] * means[spat_to_spin_idx("a", j)]
                    cov[spat_to_spin_idx("a", j)][spat_to_spin_idx("a", i)] = cov[spat_to_spin_idx("a", i)][spat_to_spin_idx("a", j)] # conjugate?
        # beta-beta
        for i in range(self.mol.nao):
            for j in range(i + 1, self.mol.nao):
                if i < self.S_beta and j < self.S_beta:
                    # agnostic
                    continue
                elif i < self.S_beta and j > self.S_beta:
                    cov[spat_to_spin_idx("b", i)][spat_to_spin_idx("b", j)] = self.LE_sol["SRRM"][spat_to_spin_idx("b", i)][j] - means[spat_to_spin_idx("b", i)] * means[spat_to_spin_idx("b", j)]
                    cov[spat_to_spin_idx("b", j)][spat_to_spin_idx("b", i)] = cov[spat_to_spin_idx("b", i)][spat_to_spin_idx("b", j)] # conjugate?
                elif i > self.S_beta and j > self.S_beta:
                    cov[spat_to_spin_idx("b", i)][spat_to_spin_idx("b", j)] = 1.0 / self.S_beta * self.LE_sol["SRRM"][spat_to_spin_idx("b", i)][j] - means[spat_to_spin_idx("b", i)] * means[spat_to_spin_idx("b", j)]
                    cov[spat_to_spin_idx("b", j)][spat_to_spin_idx("b", i)] = cov[spat_to_spin_idx("b", i)][spat_to_spin_idx("b", j)] # conjugate?

        # alpha-beta
        # We remain agnostic!



        dec_point = 4

        diagnostic_table = []
        for i in range(self.mol.nao):
            diagnostic_table.append([
                np.round(means[spat_to_spin_idx("a", i)], dec_point),
                np.round(cov[spat_to_spin_idx("a", i)][spat_to_spin_idx("a", i)], dec_point),
                np.round(np.sum(np.abs(cov[spat_to_spin_idx("a", i)])) - cov[spat_to_spin_idx("a", i)][spat_to_spin_idx("a", i)], dec_point),
                np.round(2 * cov[spat_to_spin_idx("a", i)][spat_to_spin_idx("a", i)] - np.sum(np.abs(cov[spat_to_spin_idx("a", i)])), dec_point)
                ])

        self.log.print_table(
            table_name = "LE Zombie diag.",
            column_names = ["mean", "var", "gershgorin disc", "leeway"],
            row_names = np.arange(1, self.mol.nao + 1, 1, dtype = int),
            list_of_rows = diagnostic_table
            )

        self.log.print_matrix(cov[:self.mol.nao,:self.mol.nao], "covariance matrix", dec_points = 5)

        """print("")
        print("- Means -")
        print(means)
        print("- Covariance matrix -")
        print(cov)
        print("- Covariance matric after removing negative eigenvalues -")
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.clip(eigvals, 0, None)
        L = eigvecs * eigvals[None, :]
        print(L)
        print("")"""

        # -------------- making sure covariances dont overshadow the variances ----------------

        """
        self.log.write("Gershgorin-proving the covariance matrix by increasing variances by their respective Gershgorin disc radii...")
        for i in range(2 * self.mol.nao):
            cov[i][i] = np.sum(np.abs(cov[i]))"""
        self.log.write("Gershgorin-proving the covariance matrix by re-scaling the off-diagonal terms...")
        for i in range(2 * self.mol.nao):
            row_coef = (np.sum(np.abs(cov[i])) - cov[i][i]) / cov[i][i]
            if row_coef > 1.0:
                # negative leeway
                for j in range(2 * self.mol.nao):
                    if i == j:
                        continue
                    cov[i][j] /= row_coef

        diagnostic_table = []
        for i in range(self.mol.nao):
            diagnostic_table.append([
                np.round(means[spat_to_spin_idx("a", i)], dec_point),
                np.round(cov[spat_to_spin_idx("a", i)][spat_to_spin_idx("a", i)], dec_point),
                np.round(np.sum(np.abs(cov[spat_to_spin_idx("a", i)])) - cov[spat_to_spin_idx("a", i)][spat_to_spin_idx("a", i)], dec_point),
                np.round(2 * cov[spat_to_spin_idx("a", i)][spat_to_spin_idx("a", i)] - np.sum(np.abs(cov[spat_to_spin_idx("a", i)])), dec_point)
                ])

        self.log.print_table(
            table_name = "LE Zombie diag.",
            column_names = ["mean", "var", "gershgorin disc", "leeway"],
            row_names = np.arange(1, self.mol.nao + 1, 1, dtype = int),
            list_of_rows = diagnostic_table
            )

        eigvals, eigvecs = np.linalg.eigh(cov)
        min_eigval = min(eigvals)
        if min_eigval <= 0.0:
            self.log.write(f"Warning: Covariance matrix is not positive-semidefinite (min eig = {min_eigval})")

        # We pre-sample the parameters
        rand_X = functions.sample_with_autocorrelation_safe(means, cov, N * N_subsample)
        raw_Z_sample = [
            np.zeros((N, N_subsample, self.mol.nao)),
            np.zeros((N, N_subsample, self.mol.nao))
            ]
        for n in range(N):
            for n_sub in range(N_subsample):
                for i in range(self.mol.nao):
                    raw_Z_sample[0][n][n_sub][i] = rand_X[n * N_subsample + n_sub][spat_to_spin_idx("a", i)]
                    raw_Z_sample[1][n][n_sub][i] = raw_Z_sample[0][n][n_sub][i] #rand_X[n * N_subsample + n_sub][spat_to_spin_idx("b", i)]

        N_vals = [1]
        convergence_sols = [self.reference_state_energy]

        msg = f"Conditioned sampling with ground state search on {N} states, each taken from {N_subsample} random states"
        #new_sem_ID = self.semaphor.create_event(np.linspace(0, N_subsample * ((N + 2) * (N + 1) / 2 - 1) + 1, 1000 + 1), msg)
        self.log.enter(msg, 1, True, tau_space = np.linspace(0, N_subsample * ((N + 2) * (N + 1) / 2 - 1) + 1, 1000 + 1))

        for n in range(N):
            # We add the best out of 10 random states
            cur_subsample = []
            for n_sub in range(N_subsample):

                rand_z_alpha = CS_Qubit(self.mol.nao, self.S_alpha, raw_Z_sample[0][n][n_sub])
                rand_z_beta = CS_Qubit(self.mol.nao, self.S_beta, raw_Z_sample[1][n][n_sub])
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
        self.diagnostics_log.append({f"find_ground_state_LEGS_Zombie_cov_SRRM ({dataset_label})" : procedure_diagnostic})

        self.measured_datasets.append(dataset_label)

        self.log.exit()
        return(N_vals, convergence_sols)

    def find_ground_state_LEGS_Zombie_cov_SRRM_alt(self, **kwargs):

        # This method uses Zombie (Qubit) states.
        # It looks at the total norm of the states in | LE > which have i-th MO
        # occupied, i.e. < LE | b_i\hc b_i | LE >, and uses this to estimate
        # E[z_i^2]. The means are inferred from the variances by lowering from
        # 1 for pi_1 and leaving them at 0 for pi_0.

        # The difference from ALT is that when considering b_i b_j acting on one spin subspace, we sum over the abs values
        # of all contributions from the other spin subspace to account for simultaneous excitations!

        # kwargs:
        #     -N: Sample size
        #     -N_sub: Subsample size
        #     -cov_proportion: Maximum ratio of the Gershgorin disc to the diagonal term in the cov matrix. Default is 0.9
        #     ------------------------
        #     -dataset_label: if present, this will label the dataset in disc jockey. Otherwise, label is generated from other kwargs.

        # -------------------- Parameter initialisation

        assert "N" in kwargs
        N = kwargs["N"]

        if "N_sub" in kwargs:
            N_subsample = kwargs["N_sub"]
        else:
            N_subsample = 1

        if "cov_proportion" in kwargs:
            cov_proportion = kwargs["cov_proportion"]
        else:
            cov_proportion = 0.9


        if "dataset_label" in kwargs:
            dataset_label = kwargs["dataset_label"]
        else:
            dataset_label = f"LEGS_Zombie_cov_SRRM_alt_{N}_{N_subsample}"

        self.log.enter(f"Obtaining the ground state with the method \"LEGS Zombie cov SRRM alt\" [N = {N}, N_sub = {N_subsample}, cov_proportion = {cov_proportion}]", 1)


        # Disk jockey node creation and metadata storage
        self.disk_jockey.create_data_nodes({dataset_label : {"basis_samples" : "pkl", "result_energy_states" : "csv"}})
        self.disk_jockey.commit_metadatum(dataset_label, "basis_samples", {
                "method" : "LEGS_Zombie_cov_SRRM_alt", # required
                "params" : {
                    "N" : N,
                    "N_sub" : N_subsample,
                    "cov_proportion" : cov_proportion
                }
            })
        self.user_actions += f"find_ground_state_LEGS_Zombie_cov_SRRM_alt [N = {N}, N_sub = {N_subsample}, cov_proportion = {cov_proportion}]\n"
        procedure_diagnostic = []

        if "LE_sol" not in self.checklist:
            self.log.write(f"ERROR: LEGS method requires LE solution to be known. Aborting...")
            self.log.exit()
            return(None)

        # We now manually sample Thouless states guided by the LE solution
        cur_sample = CS_sample(self, CS_Qubit, add_ref_state = True)

        # We find the means and the covariances
        spin_idx_dict = {"a" : 0, "b" : 1}
        spat_to_spin_idx = lambda sigma, i : spin_idx_dict[sigma] * self.mol.nao + i
        means = np.zeros( 2 * self.mol.nao )
        sq_means = np.zeros( 2 * self.mol.nao )
        variances = np.zeros( 2 * self.mol.nao )
        cov = np.zeros( (2 * self.mol.nao, 2 * self.mol.nao) )



        dec_point = 4

        # -------------------- Variances --------------------

        # Trying to reproduce dima's approach with all means set to zero
        #variances = np.array(([1.0, 1.0, 1e-4, 1e-6, 1e-6] + [0] * 5) * 2)
        #means = np.sqrt(np.array(([1.0, 1.0, 1e-4, 0.0, 0.0] + [0] * 5) * 2)) # mu = sigma for pi1
        #means = np.sqrt(np.array(np.sqrt(variances))) # mu = sigma
        #variances = np.array(([1.0, 1.0, 1e-4, 1e-6, 1e-6, 1e-6, 1e-7, 1e-8, 1e-8, 1e-7]) * 2)
        #cov = np.diag(variances)

        variances[spat_to_spin_idx("a", 0)] = 1.0
        for i in range(1, self.S_alpha):
            variances[spat_to_spin_idx("a", i)] = (1 - self.LE_sol["red"][spat_to_spin_idx("a", 0)][spat_to_spin_idx("a", 0)]) / (1 - self.LE_sol["red"][spat_to_spin_idx("a", i)][spat_to_spin_idx("a", i)])
        for i in range(self.S_alpha, self.mol.nao):
            variances[spat_to_spin_idx("a", i)] = self.LE_sol["red"][spat_to_spin_idx("a", i)][spat_to_spin_idx("a", i)] / self.S_alpha
        variances[spat_to_spin_idx("b", 0)] = 1.0
        for i in range(1, self.S_beta):
            variances[spat_to_spin_idx("b", i)] = (1 - self.LE_sol["red"][spat_to_spin_idx("b", 0)][spat_to_spin_idx("b", 0)]) / (1 - self.LE_sol["red"][spat_to_spin_idx("b", i)][spat_to_spin_idx("b", i)])
        for i in range(self.S_beta, self.mol.nao):
            variances[spat_to_spin_idx("b", i)] = self.LE_sol["red"][spat_to_spin_idx("b", i)][spat_to_spin_idx("b", i)] / self.S_beta
        cov = np.diag(variances)



        """
        var_alpha = 0.01 # the coefs by which initially unoccupied vars are multiplied



        self.log.print_matrix(self.LE_sol["SRRM"][:self.mol.nao,:self.mol.nao], "qubit transition matrix", dec_points = 5)

        # initialise means and variances Dima style (mean = std)
        self.log.write("Means and variances Dima style (mean = std, and derived from the calculated mean of square)")
        for i in range(self.mol.nao):
            if i < self.S_alpha:
                sq_means[spat_to_spin_idx("a", i)] = self.LE_sol["SRRM"][spat_to_spin_idx("a", i)][i]
                variances[spat_to_spin_idx("a", i)] = sq_means[spat_to_spin_idx("a", i)] - means[spat_to_spin_idx("a", i)] * means[spat_to_spin_idx("a", i)]
            if i >= self.S_alpha:
                sq_means[spat_to_spin_idx("a", i)] = self.LE_sol["SRRM"][spat_to_spin_idx("a", i)][i] / self.S_alpha
                variances[spat_to_spin_idx("a", i)] = var_alpha * (sq_means[spat_to_spin_idx("a", i)] - means[spat_to_spin_idx("a", i)] * means[spat_to_spin_idx("a", i)])
        for i in range(self.mol.nao):
            #norm_coef = 1.0
            #if i >= self.S_beta:
            #    norm_coef = 1.0 / self.S_beta
            #sq_means[spat_to_spin_idx("b", i)] = norm_coef * self.LE_sol["SRRM"][spat_to_spin_idx("b", i)][i]
            #variances[spat_to_spin_idx("b", i)] = sq_means[spat_to_spin_idx("b", i)] - means[spat_to_spin_idx("b", i)] * means[spat_to_spin_idx("b", i)]
            if i < self.S_beta:
                sq_means[spat_to_spin_idx("b", i)] = self.LE_sol["SRRM"][spat_to_spin_idx("b", i)][i]
                variances[spat_to_spin_idx("b", i)] = sq_means[spat_to_spin_idx("b", i)] - means[spat_to_spin_idx("b", i)] * means[spat_to_spin_idx("b", i)]
            if i >= self.S_beta:
                sq_means[spat_to_spin_idx("b", i)] = self.LE_sol["SRRM"][spat_to_spin_idx("b", i)][i] / self.S_beta
                variances[spat_to_spin_idx("b", i)] = var_alpha * (sq_means[spat_to_spin_idx("b", i)] - means[spat_to_spin_idx("b", i)] * means[spat_to_spin_idx("b", i)])

        cov = np.diag(variances)"""
        """
        cov_alpha = 1 # the off-diagonal rescaling free parameter

        # initialise covariances
        # alpha-alpha
        for i in range(self.mol.nao):
            for j in range(i + 1, self.mol.nao):
                if i < self.S_alpha and j < self.S_alpha:
                    # agnostic
                    continue
                elif i < self.S_alpha and j > self.S_alpha:
                    cur_cov = self.LE_sol["SRRM"][spat_to_spin_idx("a", i)][j] - means[spat_to_spin_idx("a", i)] * means[spat_to_spin_idx("a", j)]
                    cov[spat_to_spin_idx("a", i)][spat_to_spin_idx("a", j)] = cov_alpha * cur_cov
                    cov[spat_to_spin_idx("a", j)][spat_to_spin_idx("a", i)] = cov[spat_to_spin_idx("a", i)][spat_to_spin_idx("a", j)] # conjugate?
                elif i > self.S_alpha and j > self.S_alpha:
                    cur_cov = 1.0 / self.S_alpha * self.LE_sol["SRRM"][spat_to_spin_idx("a", i)][j] - means[spat_to_spin_idx("a", i)] * means[spat_to_spin_idx("a", j)]
                    cov[spat_to_spin_idx("a", i)][spat_to_spin_idx("a", j)] = cov_alpha * cur_cov
                    cov[spat_to_spin_idx("a", j)][spat_to_spin_idx("a", i)] = cov[spat_to_spin_idx("a", i)][spat_to_spin_idx("a", j)] # conjugate?
        # beta-beta
        for i in range(self.mol.nao):
            for j in range(i + 1, self.mol.nao):
                if i < self.S_beta and j < self.S_beta:
                    # agnostic
                    continue
                elif i < self.S_beta and j > self.S_beta:
                    cur_cov = self.LE_sol["SRRM"][spat_to_spin_idx("b", i)][j] - means[spat_to_spin_idx("b", i)] * means[spat_to_spin_idx("b", j)]
                    cov[spat_to_spin_idx("b", i)][spat_to_spin_idx("b", j)] = cov_alpha * cur_cov
                    cov[spat_to_spin_idx("b", j)][spat_to_spin_idx("b", i)] = cov[spat_to_spin_idx("b", i)][spat_to_spin_idx("b", j)] # conjugate?
                elif i > self.S_beta and j > self.S_beta:
                    cur_cov = 1.0 / self.S_beta * self.LE_sol["SRRM"][spat_to_spin_idx("b", i)][j] - means[spat_to_spin_idx("b", i)] * means[spat_to_spin_idx("b", j)]
                    cov[spat_to_spin_idx("b", i)][spat_to_spin_idx("b", j)] = cov_alpha * cur_cov
                    cov[spat_to_spin_idx("b", j)][spat_to_spin_idx("b", i)] = cov[spat_to_spin_idx("b", i)][spat_to_spin_idx("b", j)] # conjugate?
        """



        cov_alpha = 1 # the off-diagonal rescaling free parameter

        # initialise covariances
        # alpha-alpha
        for i in range(self.mol.nao):
            for j in range(i + 1, self.mol.nao):
                if i < self.S_alpha and j < self.S_alpha:
                    # agnostic
                    continue
                elif i < self.S_alpha and j > self.S_alpha:
                    cur_cov = - (self.LE_sol["SRRM"][spat_to_spin_idx("a", i)][j] - means[spat_to_spin_idx("a", i)] * means[spat_to_spin_idx("a", j)]) # always neg
                    cov[spat_to_spin_idx("a", i)][spat_to_spin_idx("a", j)] = cov_alpha * cur_cov
                    cov[spat_to_spin_idx("a", j)][spat_to_spin_idx("a", i)] = cov[spat_to_spin_idx("a", i)][spat_to_spin_idx("a", j)] # conjugate?
                elif i > self.S_alpha and j > self.S_alpha:
                    cur_cov = - (1.0 / self.S_alpha * self.LE_sol["SRRM"][spat_to_spin_idx("a", i)][j] - means[spat_to_spin_idx("a", i)] * means[spat_to_spin_idx("a", j)])
                    cov[spat_to_spin_idx("a", i)][spat_to_spin_idx("a", j)] = cov_alpha * cur_cov
                    cov[spat_to_spin_idx("a", j)][spat_to_spin_idx("a", i)] = cov[spat_to_spin_idx("a", i)][spat_to_spin_idx("a", j)] # conjugate?
        # beta-beta
        for i in range(self.mol.nao):
            for j in range(i + 1, self.mol.nao):
                if i < self.S_beta and j < self.S_beta:
                    # agnostic
                    continue
                elif i < self.S_beta and j > self.S_beta:
                    cur_cov = - (self.LE_sol["SRRM"][spat_to_spin_idx("b", i)][j] - means[spat_to_spin_idx("b", i)] * means[spat_to_spin_idx("b", j)])
                    cov[spat_to_spin_idx("b", i)][spat_to_spin_idx("b", j)] = cov_alpha * cur_cov
                    cov[spat_to_spin_idx("b", j)][spat_to_spin_idx("b", i)] = cov[spat_to_spin_idx("b", i)][spat_to_spin_idx("b", j)] # conjugate?
                elif i > self.S_beta and j > self.S_beta:
                    cur_cov = - (1.0 / self.S_beta * self.LE_sol["SRRM"][spat_to_spin_idx("b", i)][j] - means[spat_to_spin_idx("b", i)] * means[spat_to_spin_idx("b", j)])
                    cov[spat_to_spin_idx("b", i)][spat_to_spin_idx("b", j)] = cov_alpha * cur_cov
                    cov[spat_to_spin_idx("b", j)][spat_to_spin_idx("b", i)] = cov[spat_to_spin_idx("b", i)][spat_to_spin_idx("b", j)] # conjugate?

        # alpha-beta
        # We remain agnostic!



        diagnostic_table = []
        for i in range(self.mol.nao):
            diagnostic_table.append([
                np.round(means[spat_to_spin_idx("a", i)], dec_point),
                np.round(np.sqrt(cov[spat_to_spin_idx("a", i)][spat_to_spin_idx("a", i)]), dec_point),
                np.round(cov[spat_to_spin_idx("a", i)][spat_to_spin_idx("a", i)], dec_point),
                np.round(np.sum(np.abs(cov[spat_to_spin_idx("a", i)])) - cov[spat_to_spin_idx("a", i)][spat_to_spin_idx("a", i)], dec_point),
                np.round(2 * cov[spat_to_spin_idx("a", i)][spat_to_spin_idx("a", i)] - np.sum(np.abs(cov[spat_to_spin_idx("a", i)])), dec_point),
                np.round(100 * (2 * cov[spat_to_spin_idx("a", i)][spat_to_spin_idx("a", i)] - np.sum(np.abs(cov[spat_to_spin_idx("a", i)])) ) / cov[spat_to_spin_idx("a", i)][spat_to_spin_idx("a", i)], 1)
                ])

        self.log.print_table(
            table_name = "LE Zombie diag.",
            column_names = ["mean", "std", "var", "gershgorin disc", "leeway", "leeway %"],
            row_names = np.arange(1, self.mol.nao + 1, 1, dtype = int),
            list_of_rows = diagnostic_table
            )

        self.log.print_matrix(cov[:self.mol.nao,:self.mol.nao], "covariance matrix", dec_points = 5)

        # -------------- making sure covariances dont overshadow the variances ----------------


        #self.log.write("Gershgorin-proving the covariance matrix by increasing variances by their respective Gershgorin disc radii...")
        #for i in range(2 * self.mol.nao):
        #    cov[i][i] = np.sum(np.abs(cov[i]))
        self.log.write("Gershgorin-proving the covariance matrix by re-scaling the off-diagonal terms...")
        for i in range(2 * self.mol.nao):
            if np.sum(np.abs(cov[i])) - cov[i][i] == 0.0:
                # No off-diagonal terms, we protect against div by zero
                continue
            row_coef = (cov[i][i] * cov_proportion) / (np.sum(np.abs(cov[i])) - cov[i][i])
            if row_coef < 1.0:
                # negative leeway
                for j in range(2 * self.mol.nao):
                    if i == j:
                        continue
                    cov[i][j] *= row_coef
                    cov[j][i] *= row_coef



        # spin symmetry

        # but let's encode the goofy thing
        #for i in range(self.mol.nao):
        #    cov[spat_to_spin_idx("a", i)][spat_to_spin_idx("b", i)] = np.sqrt(variances[spat_to_spin_idx("a", i)] * variances[spat_to_spin_idx("b", i)]) * (1 - cov_proportion)
        #    cov[spat_to_spin_idx("b", i)][spat_to_spin_idx("a", i)] = cov[spat_to_spin_idx("a", i)][spat_to_spin_idx("b", i)]

        diagnostic_table = []
        for i in range(self.mol.nao):
            diagnostic_table.append([
                np.round(means[spat_to_spin_idx("a", i)], dec_point),
                np.round(np.sqrt(cov[spat_to_spin_idx("a", i)][spat_to_spin_idx("a", i)]), dec_point),
                np.round(cov[spat_to_spin_idx("a", i)][spat_to_spin_idx("a", i)], dec_point),
                np.round(np.sum(np.abs(cov[spat_to_spin_idx("a", i)])) - cov[spat_to_spin_idx("a", i)][spat_to_spin_idx("a", i)], dec_point),
                np.round(2 * cov[spat_to_spin_idx("a", i)][spat_to_spin_idx("a", i)] - np.sum(np.abs(cov[spat_to_spin_idx("a", i)])), dec_point),
                np.round(100 * (2 * cov[spat_to_spin_idx("a", i)][spat_to_spin_idx("a", i)] - np.sum(np.abs(cov[spat_to_spin_idx("a", i)])) ) / cov[spat_to_spin_idx("a", i)][spat_to_spin_idx("a", i)], 1)
                ])

        self.log.print_table(
            table_name = "LE Zombie diag.",
            column_names = ["mean", "std", "var", "gershgorin disc", "leeway", "leeway %"],
            row_names = np.arange(1, self.mol.nao + 1, 1, dtype = int),
            list_of_rows = diagnostic_table
            )

        eigvals, eigvecs = np.linalg.eigh(cov)
        min_eigval = min(eigvals)
        if min_eigval <= 0.0:
            self.log.write(f"Warning: Covariance matrix is not positive-semidefinite (min eig = {min_eigval})")

        # We pre-sample the parameters
        rand_X = functions.sample_with_autocorrelation_safe(means, cov, N * N_subsample)
        raw_Z_sample = [
            np.zeros((N, N_subsample, self.mol.nao)),
            np.zeros((N, N_subsample, self.mol.nao))
            ]
        for n in range(N):
            for n_sub in range(N_subsample):
                for i in range(self.mol.nao):
                    raw_Z_sample[0][n][n_sub][i] = rand_X[n * N_subsample + n_sub][spat_to_spin_idx("a", i)]
                    raw_Z_sample[1][n][n_sub][i] = raw_Z_sample[0][n][n_sub][i] #rand_X[n * N_subsample + n_sub][spat_to_spin_idx("b", i)] #

        N_vals = [1]
        convergence_sols = [self.reference_state_energy]

        msg = f"Conditioned sampling with ground state search on {N} states, each taken from {N_subsample} random states"
        #new_sem_ID = self.semaphor.create_event(np.linspace(0, N_subsample * ((N + 2) * (N + 1) / 2 - 1) + 1, 1000 + 1), msg)
        self.log.enter(msg, 1, True, tau_space = np.linspace(0, N_subsample * ((N + 2) * (N + 1) / 2 - 1) + 1, 1000 + 1))

        for n in range(N):
            # We add the best out of 10 random states
            cur_subsample = []
            for n_sub in range(N_subsample):

                rand_z_alpha = CS_Qubit(self.mol.nao, self.S_alpha, raw_Z_sample[0][n][n_sub])
                rand_z_beta = CS_Qubit(self.mol.nao, self.S_beta, raw_Z_sample[1][n][n_sub])
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
        self.disk_jockey.commit_metadatum(dataset_label, "result_energy_states", {"E_g" : cur_sample.E_ground[-1]})
        self.diagnostics_log.append({f"find_ground_state_LEGS_Zombie_cov_SRRM_alt ({dataset_label})" : procedure_diagnostic})

        self.measured_datasets.append(dataset_label)

        self.log.write(f"Measured datasets: {self.measured_datasets}")

        self.log.exit()
        return(N_vals, convergence_sols)

    def find_ground_state_LEGS_Zombie_cov_SRRM_mirror(self, **kwargs):

        # This method uses Zombie (Qubit) states.
        # It looks at the total norm of the states in | LE > which have i-th MO
        # occupied, i.e. < LE | b_i\hc b_i | LE >, and uses this to estimate
        # E[z_i^2].
        # The means are equal to sigmas, and the sample is then multiplied by a random sign mask

        # The difference from ALT is that when considering b_i b_j acting on one spin subspace, we sum over the abs values
        # of all contributions from the other spin subspace to account for simultaneous excitations!

        # kwargs:
        #     -N: Sample size
        #     -N_sub: Subsample size
        #     -cov_proportion: Maximum ratio of the Gershgorin disc to the diagonal term in the cov matrix. Default is 0.9
        #     ------------------------
        #     -dataset_label: if present, this will label the dataset in disc jockey. Otherwise, label is generated from other kwargs.

        # -------------------- Parameter initialisation

        assert "N" in kwargs
        N = kwargs["N"]

        if "N_sub" in kwargs:
            N_subsample = kwargs["N_sub"]
        else:
            N_subsample = 1

        if "cov_proportion" in kwargs:
            cov_proportion = kwargs["cov_proportion"]
        else:
            cov_proportion = 0.9


        if "dataset_label" in kwargs:
            dataset_label = kwargs["dataset_label"]
        else:
            dataset_label = f"LEGS_Zombie_cov_SRRM_mirror_{N}_{N_subsample}"

        self.log.enter(f"Obtaining the ground state with the method \"LEGS Zombie cov SRRM mirror\" [N = {N}, N_sub = {N_subsample}, cov_proportion = {cov_proportion}]", 1)


        # Disk jockey node creation and metadata storage
        self.disk_jockey.create_data_nodes({dataset_label : {"basis_samples" : "pkl", "result_energy_states" : "csv"}})
        self.disk_jockey.commit_metadatum(dataset_label, "basis_samples", {
                "method" : "LEGS_Zombie_cov_SRRM_mirror", # required
                "params" : {
                    "N" : N,
                    "N_sub" : N_subsample,
                    "cov_proportion" : cov_proportion
                }
            })
        self.user_actions += f"find_ground_state_LEGS_Zombie_cov_SRRM_mirror [N = {N}, N_sub = {N_subsample}, cov_proportion = {cov_proportion}]\n"
        procedure_diagnostic = []

        if "LE_sol" not in self.checklist:
            self.log.write(f"ERROR: LEGS method requires LE solution to be known. Aborting...")
            self.log.exit()
            return(None)

        # We now manually sample Thouless states guided by the LE solution
        cur_sample = CS_sample(self, CS_Qubit, add_ref_state = True)

        # We find the means and the covariances
        spin_idx_dict = {"a" : 0, "b" : 1}
        spat_to_spin_idx = lambda sigma, i : spin_idx_dict[sigma] * self.mol.nao + i
        means = np.zeros( 2 * self.mol.nao )
        sq_means = np.zeros( 2 * self.mol.nao )
        variances = np.zeros( 2 * self.mol.nao )
        cov = np.zeros( (2 * self.mol.nao, 2 * self.mol.nao) )



        dec_point = 4

        # -------------------- Variances --------------------

        # Trying to reproduce dima's approach with all means set to zero
        #variances = np.array(([1.0, 1.0, 1e-4, 1e-6, 1e-6] + [0] * 5) * 2)
        #means = np.sqrt(np.array(np.sqrt(variances))) # mu = sigma
        #variances = np.array(([1.0, 1.0, 1e-4, 1e-6, 1e-6, 1e-6, 1e-7, 1e-8, 1e-8, 1e-7]) * 2)
        #cov = np.diag(variances)


        variances[spat_to_spin_idx("a", 0)] = 1.0
        for i in range(1, self.S_alpha):
            variances[spat_to_spin_idx("a", i)] = (1 - self.LE_sol["red"][spat_to_spin_idx("a", 0)][spat_to_spin_idx("a", 0)]) / (1 - self.LE_sol["red"][spat_to_spin_idx("a", i)][spat_to_spin_idx("a", i)])
        for i in range(self.S_alpha, self.mol.nao):
            variances[spat_to_spin_idx("a", i)] = self.LE_sol["red"][spat_to_spin_idx("a", i)][spat_to_spin_idx("a", i)] / self.S_alpha
        variances[spat_to_spin_idx("b", 0)] = 1.0
        for i in range(1, self.S_beta):
            variances[spat_to_spin_idx("b", i)] = (1 - self.LE_sol["red"][spat_to_spin_idx("b", 0)][spat_to_spin_idx("b", 0)]) / (1 - self.LE_sol["red"][spat_to_spin_idx("b", i)][spat_to_spin_idx("b", i)])
        for i in range(self.S_beta, self.mol.nao):
            variances[spat_to_spin_idx("b", i)] = self.LE_sol["red"][spat_to_spin_idx("b", i)][spat_to_spin_idx("b", i)] / self.S_beta
        cov = np.diag(variances)

        cov_alpha = 1 # the off-diagonal rescaling free parameter

        # initialise covariances
        # alpha-alpha
        for i in range(self.mol.nao):
            for j in range(i + 1, self.mol.nao):
                if i < self.S_alpha and j < self.S_alpha:
                    # agnostic
                    continue
                elif i < self.S_alpha and j >= self.S_alpha:
                    cur_cov = - (self.LE_sol["SRRM"][spat_to_spin_idx("a", i)][j] - means[spat_to_spin_idx("a", i)] * means[spat_to_spin_idx("a", j)]) # always neg
                    cov[spat_to_spin_idx("a", i)][spat_to_spin_idx("a", j)] = cov_alpha * cur_cov
                    cov[spat_to_spin_idx("a", j)][spat_to_spin_idx("a", i)] = cov[spat_to_spin_idx("a", i)][spat_to_spin_idx("a", j)] # conjugate?
                elif i >= self.S_alpha and j >= self.S_alpha:
                    cur_cov = - (1.0 / self.S_alpha * self.LE_sol["SRRM"][spat_to_spin_idx("a", i)][j] - means[spat_to_spin_idx("a", i)] * means[spat_to_spin_idx("a", j)])
                    cov[spat_to_spin_idx("a", i)][spat_to_spin_idx("a", j)] = cov_alpha * cur_cov
                    cov[spat_to_spin_idx("a", j)][spat_to_spin_idx("a", i)] = cov[spat_to_spin_idx("a", i)][spat_to_spin_idx("a", j)] # conjugate?
        # beta-beta
        for i in range(self.mol.nao):
            for j in range(i + 1, self.mol.nao):
                if i < self.S_beta and j < self.S_beta:
                    # agnostic
                    continue
                elif i < self.S_beta and j >= self.S_beta:
                    cur_cov = - (self.LE_sol["SRRM"][spat_to_spin_idx("b", i)][j] - means[spat_to_spin_idx("b", i)] * means[spat_to_spin_idx("b", j)])
                    cov[spat_to_spin_idx("b", i)][spat_to_spin_idx("b", j)] = cov_alpha * cur_cov
                    cov[spat_to_spin_idx("b", j)][spat_to_spin_idx("b", i)] = cov[spat_to_spin_idx("b", i)][spat_to_spin_idx("b", j)] # conjugate?
                elif i >= self.S_beta and j >= self.S_beta:
                    cur_cov = - (1.0 / self.S_beta * self.LE_sol["SRRM"][spat_to_spin_idx("b", i)][j] - means[spat_to_spin_idx("b", i)] * means[spat_to_spin_idx("b", j)])
                    cov[spat_to_spin_idx("b", i)][spat_to_spin_idx("b", j)] = cov_alpha * cur_cov
                    cov[spat_to_spin_idx("b", j)][spat_to_spin_idx("b", i)] = cov[spat_to_spin_idx("b", i)][spat_to_spin_idx("b", j)] # conjugate?

        # alpha-beta
        # We remain agnostic!

        # Now we shift the means to match the sigmas
        means = np.sqrt(np.array(np.sqrt(variances))) # mu = sigma

        diagnostic_table = []
        for i in range(self.mol.nao):
            diagnostic_table.append([
                np.round(means[spat_to_spin_idx("a", i)], dec_point),
                np.round(np.sqrt(cov[spat_to_spin_idx("a", i)][spat_to_spin_idx("a", i)]), dec_point),
                np.round(cov[spat_to_spin_idx("a", i)][spat_to_spin_idx("a", i)], dec_point),
                np.round(np.sum(np.abs(cov[spat_to_spin_idx("a", i)])) - cov[spat_to_spin_idx("a", i)][spat_to_spin_idx("a", i)], dec_point),
                np.round(2 * cov[spat_to_spin_idx("a", i)][spat_to_spin_idx("a", i)] - np.sum(np.abs(cov[spat_to_spin_idx("a", i)])), dec_point),
                np.round(100 * (2 * cov[spat_to_spin_idx("a", i)][spat_to_spin_idx("a", i)] - np.sum(np.abs(cov[spat_to_spin_idx("a", i)])) ) / cov[spat_to_spin_idx("a", i)][spat_to_spin_idx("a", i)], 1)
                ])

        self.log.print_table(
            table_name = "LE Zombie diag.",
            column_names = ["mean", "std", "var", "gershgorin disc", "leeway", "leeway %"],
            row_names = np.arange(1, self.mol.nao + 1, 1, dtype = int),
            list_of_rows = diagnostic_table
            )

        self.log.print_matrix(cov[:self.mol.nao,:self.mol.nao], "covariance matrix", dec_points = 5)

        # -------------- making sure covariances dont overshadow the variances ----------------


        #self.log.write("Gershgorin-proving the covariance matrix by increasing variances by their respective Gershgorin disc radii...")
        #for i in range(2 * self.mol.nao):
        #    cov[i][i] = np.sum(np.abs(cov[i]))
        self.log.write("Gershgorin-proving the covariance matrix by re-scaling the off-diagonal terms...")
        for i in range(2 * self.mol.nao):
            if np.sum(np.abs(cov[i])) - cov[i][i] == 0.0:
                # No off-diagonal terms, we protect against div by zero
                continue
            row_coef = (cov[i][i] * cov_proportion) / (np.sum(np.abs(cov[i])) - cov[i][i])
            if row_coef < 1.0:
                # negative leeway
                for j in range(2 * self.mol.nao):
                    if i == j:
                        continue
                    cov[i][j] *= row_coef
                    cov[j][i] *= row_coef



        # spin symmetry

        # but let's encode the goofy thing
        #for i in range(self.mol.nao):
        #    cov[spat_to_spin_idx("a", i)][spat_to_spin_idx("b", i)] = np.sqrt(variances[spat_to_spin_idx("a", i)] * variances[spat_to_spin_idx("b", i)]) * (1 - cov_proportion)
        #    cov[spat_to_spin_idx("b", i)][spat_to_spin_idx("a", i)] = cov[spat_to_spin_idx("a", i)][spat_to_spin_idx("b", i)]

        diagnostic_table = []
        for i in range(self.mol.nao):
            diagnostic_table.append([
                np.round(means[spat_to_spin_idx("a", i)], dec_point),
                np.round(np.sqrt(cov[spat_to_spin_idx("a", i)][spat_to_spin_idx("a", i)]), dec_point),
                np.round(cov[spat_to_spin_idx("a", i)][spat_to_spin_idx("a", i)], dec_point),
                np.round(np.sum(np.abs(cov[spat_to_spin_idx("a", i)])) - cov[spat_to_spin_idx("a", i)][spat_to_spin_idx("a", i)], dec_point),
                np.round(2 * cov[spat_to_spin_idx("a", i)][spat_to_spin_idx("a", i)] - np.sum(np.abs(cov[spat_to_spin_idx("a", i)])), dec_point),
                np.round(100 * (2 * cov[spat_to_spin_idx("a", i)][spat_to_spin_idx("a", i)] - np.sum(np.abs(cov[spat_to_spin_idx("a", i)])) ) / cov[spat_to_spin_idx("a", i)][spat_to_spin_idx("a", i)], 1)
                ])

        self.log.print_table(
            table_name = "LE Zombie diag.",
            column_names = ["mean", "std", "var", "gershgorin disc", "leeway", "leeway %"],
            row_names = np.arange(1, self.mol.nao + 1, 1, dtype = int),
            list_of_rows = diagnostic_table
            )

        eigvals, eigvecs = np.linalg.eigh(cov)
        min_eigval = min(eigvals)
        if min_eigval <= 0.0:
            self.log.write(f"Warning: Covariance matrix is not positive-semidefinite (min eig = {min_eigval})")

        # We pre-sample the parameters
        rand_X = functions.sample_with_autocorrelation_safe(means, cov, N * N_subsample)

        # We now randomly sign the parameters
        sign_mask = np.random.randint(0, 2, (N * N_subsample, self.mol.nao * 2)) * 2 - 1
        sign_mask[:,:self.S_alpha] = np.ones((N * N_subsample, self.S_alpha))

        rand_X = rand_X * sign_mask

        raw_Z_sample = [
            np.zeros((N, N_subsample, self.mol.nao)),
            np.zeros((N, N_subsample, self.mol.nao))
            ]
        for n in range(N):
            for n_sub in range(N_subsample):
                for i in range(self.mol.nao):
                    raw_Z_sample[0][n][n_sub][i] = rand_X[n * N_subsample + n_sub][spat_to_spin_idx("a", i)]
                    raw_Z_sample[1][n][n_sub][i] = raw_Z_sample[0][n][n_sub][i] #rand_X[n * N_subsample + n_sub][spat_to_spin_idx("b", i)] #

        N_vals = [1]
        convergence_sols = [self.reference_state_energy]

        msg = f"Conditioned sampling with ground state search on {N} states, each taken from {N_subsample} random states"
        #new_sem_ID = self.semaphor.create_event(np.linspace(0, N_subsample * ((N + 2) * (N + 1) / 2 - 1) + 1, 1000 + 1), msg)
        self.log.enter(msg, 1, True, tau_space = np.linspace(0, N_subsample * ((N + 2) * (N + 1) / 2 - 1) + 1, 1000 + 1))

        for n in range(N):
            # We add the best out of 10 random states
            cur_subsample = []
            for n_sub in range(N_subsample):

                rand_z_alpha = CS_Qubit(self.mol.nao, self.S_alpha, raw_Z_sample[0][n][n_sub])
                rand_z_beta = CS_Qubit(self.mol.nao, self.S_beta, raw_Z_sample[1][n][n_sub])
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
        self.disk_jockey.commit_metadatum(dataset_label, "result_energy_states", {"E_g" : cur_sample.E_ground[-1]})
        self.diagnostics_log.append({f"find_ground_state_LEGS_Zombie_cov_SRRM_mirror ({dataset_label})" : procedure_diagnostic})

        self.measured_datasets.append(dataset_label)

        self.log.write(f"Measured datasets: {self.measured_datasets}")

        self.log.exit()
        return(N_vals, convergence_sols)

    def find_ground_state_LEGS_Zombie_cov_SOPM(self, **kwargs):

        # This method uses Zombie (Qubit) states.
        # It looks at the total norm of the states in | LE > which have i-th MO
        # occupied, i.e. < LE | b_i\hc b_i | LE >, and uses this to estimate
        # E[z_i^2].
        # The means are equal to sigmas, and the sample is then multiplied by a random sign mask

        # The difference from ALT is that when considering b_i b_j acting on one spin subspace, we sum over the abs values
        # of all contributions from the other spin subspace to account for simultaneous excitations!

        # kwargs:
        #     -N: Sample size
        #     -N_sub: Subsample size
        #     ------------------------
        #     -dataset_label: if present, this will label the dataset in disc jockey. Otherwise, label is generated from other kwargs.

        # -------------------- Parameter initialisation

        assert "N" in kwargs
        N = kwargs["N"]

        if "N_sub" in kwargs:
            N_subsample = kwargs["N_sub"]
        else:
            N_subsample = 1

        if "dataset_label" in kwargs:
            dataset_label = kwargs["dataset_label"]
        else:
            dataset_label = f"LEGS_Zombie_cov_SOPM_{N}_{N_subsample}"

        self.log.enter(f"Obtaining the ground state with the method \"LEGS Zombie cov SOPM\" [N = {N}, N_sub = {N_subsample}]", 1)


        # Disk jockey node creation and metadata storage
        self.disk_jockey.create_data_nodes({dataset_label : {"basis_samples" : "pkl", "result_energy_states" : "csv"}})
        self.disk_jockey.commit_metadatum(dataset_label, "basis_samples", {
                "method" : "LEGS_Zombie_cov_SOPM", # required
                "params" : {
                    "N" : N,
                    "N_sub" : N_subsample
                }
            })
        self.user_actions += f"find_ground_state_LEGS_Zombie_cov_SOPM [N = {N}, N_sub = {N_subsample}]\n"
        procedure_diagnostic = []

        if "LE_sol" not in self.checklist:
            self.log.write(f"ERROR: LEGS method requires LE solution to be known. Aborting...")
            self.log.exit()
            return(None)

        # We now manually sample Thouless states guided by the LE solution
        cur_sample = CS_sample(self, CS_Qubit, add_ref_state = True)

        # We find the means and the covariances
        spin_idx_dict = {"a" : 0, "b" : 1}
        spat_to_spin_idx = lambda sigma, i : spin_idx_dict[sigma] * self.mol.nao + i
        means = np.zeros( 2 * self.mol.nao )
        sq_means = np.zeros( 2 * self.mol.nao )
        variances = np.zeros( 2 * self.mol.nao )
        cov = np.zeros( (2 * self.mol.nao, 2 * self.mol.nao) )



        dec_point = 4

        # -------------------- Variances --------------------

        # Trying to reproduce dima's approach with all means set to zero
        #variances = np.array(([1.0, 1.0, 1e-4, 1e-6, 1e-6] + [0] * 5) * 2)
        #means = np.sqrt(np.array(np.sqrt(variances))) # mu = sigma
        #variances = np.array(([1.0, 1.0, 1e-4, 1e-6, 1e-6, 1e-6, 1e-7, 1e-8, 1e-8, 1e-7]) * 2)
        #cov = np.diag(variances)


        variances[spat_to_spin_idx("a", 0)] = 1.0
        for i in range(1, self.S_alpha):
            variances[spat_to_spin_idx("a", i)] = (1 - self.LE_sol["SOPM"][spat_to_spin_idx("a", 0)][spat_to_spin_idx("a", 0)]) / (1 - self.LE_sol["SOPM"][spat_to_spin_idx("a", i)][spat_to_spin_idx("a", i)])
        for i in range(self.S_alpha, self.mol.nao):
            variances[spat_to_spin_idx("a", i)] = self.LE_sol["SOPM"][spat_to_spin_idx("a", i)][spat_to_spin_idx("a", i)] / self.S_alpha
        variances[spat_to_spin_idx("b", 0)] = 1.0
        for i in range(1, self.S_beta):
            variances[spat_to_spin_idx("b", i)] = (1 - self.LE_sol["SOPM"][spat_to_spin_idx("b", 0)][spat_to_spin_idx("b", 0)]) / (1 - self.LE_sol["SOPM"][spat_to_spin_idx("b", i)][spat_to_spin_idx("b", i)])
        for i in range(self.S_beta, self.mol.nao):
            variances[spat_to_spin_idx("b", i)] = self.LE_sol["SOPM"][spat_to_spin_idx("b", i)][spat_to_spin_idx("b", i)] / self.S_beta
        cov = np.diag(variances)


        cov_alpha = 1 # the off-diagonal rescaling free parameter

        # initialise covariances

        # note that the off-diagonal terms in SOPM are just correlations.
        # So cov_ij = SOPM_ij * sqrt(var_i * var_j)

        # alpha-alpha
        for i in range(self.mol.nao):
            for j in range(i + 1, self.mol.nao):
                cur_cov = self.LE_sol["SOPM"][spat_to_spin_idx("a", i)][spat_to_spin_idx("a", j)] * np.sqrt(variances[spat_to_spin_idx("a", i)] * variances[spat_to_spin_idx("a", j)])
                cov[spat_to_spin_idx("a", i)][spat_to_spin_idx("a", j)] = cov_alpha * cur_cov
                cov[spat_to_spin_idx("a", j)][spat_to_spin_idx("a", i)] = cov[spat_to_spin_idx("a", i)][spat_to_spin_idx("a", j)] # conjugate?
        # beta-beta
        for i in range(self.mol.nao):
            for j in range(i + 1, self.mol.nao):
                cur_cov = self.LE_sol["SOPM"][spat_to_spin_idx("b", i)][spat_to_spin_idx("b", j)] * np.sqrt(variances[spat_to_spin_idx("b", i)] * variances[spat_to_spin_idx("b", j)])
                cov[spat_to_spin_idx("b", i)][spat_to_spin_idx("b", j)] = cov_alpha * cur_cov
                cov[spat_to_spin_idx("b", j)][spat_to_spin_idx("b", i)] = cov[spat_to_spin_idx("b", i)][spat_to_spin_idx("b", j)] # conjugate?

        # alpha-beta
        # We remain agnostic! Because even tho this should work perfectly, I'm not sure how the gershgorin stuff will work

        diagnostic_table = []
        for i in range(self.mol.nao):
            diagnostic_table.append([
                np.round(means[spat_to_spin_idx("a", i)], dec_point),
                np.round(np.sqrt(cov[spat_to_spin_idx("a", i)][spat_to_spin_idx("a", i)]), dec_point),
                np.round(cov[spat_to_spin_idx("a", i)][spat_to_spin_idx("a", i)], dec_point),
                np.round(np.sum(np.abs(cov[spat_to_spin_idx("a", i)])) - cov[spat_to_spin_idx("a", i)][spat_to_spin_idx("a", i)], dec_point),
                np.round(2 * cov[spat_to_spin_idx("a", i)][spat_to_spin_idx("a", i)] - np.sum(np.abs(cov[spat_to_spin_idx("a", i)])), dec_point),
                np.round(100 * (2 * cov[spat_to_spin_idx("a", i)][spat_to_spin_idx("a", i)] - np.sum(np.abs(cov[spat_to_spin_idx("a", i)])) ) / cov[spat_to_spin_idx("a", i)][spat_to_spin_idx("a", i)], 1)
                ])

        self.log.print_table(
            table_name = "LE Zombie diag.",
            column_names = ["mean", "std", "var", "gershgorin disc", "leeway", "leeway %"],
            row_names = np.arange(1, self.mol.nao + 1, 1, dtype = int),
            list_of_rows = diagnostic_table
            )

        self.log.print_matrix(cov[:self.mol.nao,:self.mol.nao], "covariance matrix", dec_points = 5)

        # -------------- making sure covariances dont overshadow the variances ----------------

        eigvals, eigvecs = np.linalg.eigh(cov)
        min_eigval = min(eigvals)
        if min_eigval <= 0.0:
            self.log.write(f"Warning: Covariance matrix is not positive-semidefinite (min eig = {min_eigval})")

        # We pre-sample the parameters
        rand_X = functions.sample_with_autocorrelation_safe(means, cov, N * N_subsample)

        raw_Z_sample = [
            np.zeros((N, N_subsample, self.mol.nao)),
            np.zeros((N, N_subsample, self.mol.nao))
            ]
        for n in range(N):
            for n_sub in range(N_subsample):
                for i in range(self.mol.nao):
                    raw_Z_sample[0][n][n_sub][i] = rand_X[n * N_subsample + n_sub][spat_to_spin_idx("a", i)]
                    raw_Z_sample[1][n][n_sub][i] = raw_Z_sample[0][n][n_sub][i] #rand_X[n * N_subsample + n_sub][spat_to_spin_idx("b", i)] #

        N_vals = [1]
        convergence_sols = [self.reference_state_energy]

        msg = f"Conditioned sampling with ground state search on {N} states, each taken from {N_subsample} random states"
        #new_sem_ID = self.semaphor.create_event(np.linspace(0, N_subsample * ((N + 2) * (N + 1) / 2 - 1) + 1, 1000 + 1), msg)
        self.log.enter(msg, 1, True, tau_space = np.linspace(0, N_subsample * ((N + 2) * (N + 1) / 2 - 1) + 1, 1000 + 1))

        for n in range(N):
            # We add the best out of 10 random states
            cur_subsample = []
            for n_sub in range(N_subsample):

                rand_z_alpha = CS_Qubit(self.mol.nao, self.S_alpha, raw_Z_sample[0][n][n_sub])
                rand_z_beta = CS_Qubit(self.mol.nao, self.S_beta, raw_Z_sample[1][n][n_sub])
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
        self.disk_jockey.commit_metadatum(dataset_label, "result_energy_states", {"E_g" : cur_sample.E_ground[-1]})
        self.diagnostics_log.append({f"find_ground_state_LEGS_Zombie_cov_SOPM ({dataset_label})" : procedure_diagnostic})

        self.measured_datasets.append(dataset_label)

        self.log.write(f"Measured datasets: {self.measured_datasets}")

        self.log.exit()
        return(N_vals, convergence_sols)

    def find_ground_state_LEGS_Zombie_cov_RSOPM_moment_matching(self, **kwargs):

        # This method uses Zombie (Qubit) states.
        # It looks at the total norm of the states in | LE > which have i-th MO
        # occupied, i.e. < LE | b_i\hc b_i | LE >, and uses this to estimate
        # E[z_i^2].
        # The means are equal to sigmas, and the sample is then multiplied by a random sign mask

        # The difference from ALT is that when considering b_i b_j acting on one spin subspace, we sum over the abs values
        # of all contributions from the other spin subspace to account for simultaneous excitations!

        # kwargs:
        #     -N: Sample size
        #     -N_sub: Subsample size
        #     -N_no_cov: Number of configs per subsample for which correlation is ignored
        #     -rs: if True, the parameter signs are randomised. Default False
        #     -alpha: strength of correlation. Default 1.0
        #     -eta: step-size for moment matching; default 1e-8
        #     ------------------------
        #     -dataset_label: if present, this will label the dataset in disc jockey. Otherwise, label is generated from other kwargs.

        # -------------------- Parameter initialisation

        assert "N" in kwargs
        N = kwargs["N"]

        if "N_sub" in kwargs:
            N_subsample = kwargs["N_sub"]
        else:
            N_subsample = 1

        if "N_no_cov" in kwargs:
            N_no_cov = min(kwargs["N_no_cov"], N_subsample)
        else:
            N_no_cov = 0

        if "rs" in kwargs:
            randomise_signs = kwargs["rs"]
        else:
            randomise_signs = False
        if randomise_signs:
            rs_label = "(rs)"
        else:
            rs_label = "(no_rs)"

        if "alpha" in kwargs:
            alpha = kwargs["alpha"]
        else:
            alpha = 1.0

        if "eta" in kwargs:
            eta = kwargs["eta"]
        else:
            eta = 1e-1

        if "dataset_label" in kwargs:
            dataset_label = kwargs["dataset_label"]
        else:
            dataset_label = f"LEGS_Zombie_cov_RSOPM_moment_matching_{N}_{N_subsample}_{N_no_cov}_{rs_label}_{alpha}"

        self.log.enter(f"Obtaining the ground state with the method \"LEGS Zombie cov RSOPM moment matching\" [N = {N}, N_sub = {N_subsample}, N_no_cov = {N_no_cov}, rs = {randomise_signs}, alpha = {alpha}, eta = {eta}]", 1)


        # Disk jockey node creation and metadata storage
        self.disk_jockey.create_data_nodes({dataset_label : {"basis_samples" : "pkl", "result_energy_states" : "csv"}})
        self.disk_jockey.commit_metadatum(dataset_label, "basis_samples", {
                "method" : "LEGS_Zombie_cov_RSOPM_moment_matching", # required
                "params" : {
                    "N" : N,
                    "N_sub" : N_subsample,
                    "N_no_cov" : N_no_cov,
                    "rs" : randomise_signs,
                    "alpha" : alpha,
                    "eta" : eta
                }
            })
        self.user_actions += f"find_ground_state_LEGS_Zombie_cov_RSOPM_moment_matching [N = {N}, N_sub = {N_subsample}, N_no_cov = {N_no_cov}, rs = {randomise_signs}, alpha = {alpha}, eta = {eta}]\n"
        procedure_diagnostic = []

        if "LE_sol" not in self.checklist:
            self.log.write(f"ERROR: LEGS method requires LE solution to be known. Aborting...")
            self.log.exit()
            return(None)

        # We now manually sample Thouless states guided by the LE solution
        cur_sample = CS_sample(self, CS_Qubit, add_ref_state = True)

        # We find the means and the covariances
        spin_idx_dict = {"a" : 0, "b" : 1}
        spat_to_spin_idx = lambda sigma, i : spin_idx_dict[sigma] * self.mol.nao + i
        means = np.zeros( 2 * self.mol.nao )
        sq_means = np.zeros( 2 * self.mol.nao )
        variances = np.zeros( 2 * self.mol.nao )
        cov = np.zeros( (2 * self.mol.nao, 2 * self.mol.nao) )



        dec_point = 4

        # -------------------- Variances --------------------

        # We use moment-matching to estimate the no-covariance variances

        # initial guess
        # should we actually use the hole approach from SOPM or just straight up RSOPM?
        # idea: use SOPM to get <z_i^2>, then transform the first S values using the hole approach
        # BUT! The constraint sum A_i only works for RSOPM, and only if the reduced LE sol is properly normalised!
        variances = self.LE_sol["RNCS"] ** 2 # <z_i^2>, taking <z_i> = 0
        cov = np.diag(variances)

        # initialise covariances

        # note that the off-diagonal terms in SOPM are just correlations.
        # So cov_ij = SOPM_ij * sqrt(var_i * var_j)

        # alpha-alpha
        for i in range(self.mol.nao):
            for j in range(i + 1, self.mol.nao):
                cur_cov = alpha * np.sqrt(self.LE_sol["RSOPM"][spat_to_spin_idx("a", i)][spat_to_spin_idx("a", j)] * variances[spat_to_spin_idx("a", i)] * variances[spat_to_spin_idx("a", j)])
                cov[spat_to_spin_idx("a", i)][spat_to_spin_idx("a", j)] = cur_cov
                cov[spat_to_spin_idx("a", j)][spat_to_spin_idx("a", i)] = cov[spat_to_spin_idx("a", i)][spat_to_spin_idx("a", j)] # conjugate?
        # beta-beta
        for i in range(self.mol.nao):
            for j in range(i + 1, self.mol.nao):
                cur_cov = alpha * np.sqrt(self.LE_sol["RSOPM"][spat_to_spin_idx("b", i)][spat_to_spin_idx("b", j)] * variances[spat_to_spin_idx("b", i)] * variances[spat_to_spin_idx("b", j)])
                cov[spat_to_spin_idx("b", i)][spat_to_spin_idx("b", j)] = cur_cov
                cov[spat_to_spin_idx("b", j)][spat_to_spin_idx("b", i)] = cov[spat_to_spin_idx("b", i)][spat_to_spin_idx("b", j)] # conjugate?

        # alpha-beta
        # We remain agnostic! Because even tho this should work perfectly, I'm not sure how the gershgorin stuff will work

        for i in range(self.mol.nao):
            for j in range(self.mol.nao):
                cur_cov = alpha * np.sqrt(self.LE_sol["RSOPM"][spat_to_spin_idx("a", i)][spat_to_spin_idx("b", j)] * variances[spat_to_spin_idx("a", i)] * variances[spat_to_spin_idx("b", j)])
                cov[spat_to_spin_idx("a", i)][spat_to_spin_idx("b", j)] = cur_cov
                cov[spat_to_spin_idx("b", j)][spat_to_spin_idx("a", i)] = cov[spat_to_spin_idx("a", i)][spat_to_spin_idx("b", j)] # conjugate?

        if self.HF_method == "RHF":
            diagnostic_table = []
            for i in range(self.mol.nao):
                diagnostic_table.append([
                    np.round(means[spat_to_spin_idx("a", i)], dec_point),
                    np.round(np.sqrt(cov[spat_to_spin_idx("a", i)][spat_to_spin_idx("a", i)]), dec_point),
                    np.round(cov[spat_to_spin_idx("a", i)][spat_to_spin_idx("a", i)], dec_point),
                    np.round(np.sum(np.abs(cov[spat_to_spin_idx("a", i)])) - cov[spat_to_spin_idx("a", i)][spat_to_spin_idx("a", i)], dec_point),
                    np.round(2 * cov[spat_to_spin_idx("a", i)][spat_to_spin_idx("a", i)] - np.sum(np.abs(cov[spat_to_spin_idx("a", i)])), dec_point),
                    np.round(100 * (2 * cov[spat_to_spin_idx("a", i)][spat_to_spin_idx("a", i)] - np.sum(np.abs(cov[spat_to_spin_idx("a", i)])) ) / cov[spat_to_spin_idx("a", i)][spat_to_spin_idx("a", i)], 1)
                    ])

            self.log.print_table(
                table_name = "LE Zombie diag.",
                column_names = ["mean", "std", "var", "gershgorin disc", "leeway", "leeway %"],
                row_names = np.arange(1, self.mol.nao + 1, 1, dtype = int),
                list_of_rows = diagnostic_table
                )

            self.log.print_matrix(cov[:self.mol.nao,:self.mol.nao], "covariance matrix", dec_points = 5)
        elif self.HF_method == "UHF":
            diagnostic_table = []
            for s in ["a", "b"]:
                for i in range(self.mol.nao):
                    diagnostic_table.append([
                        np.round(means[spat_to_spin_idx(s, i)], dec_point),
                        np.round(np.sqrt(cov[spat_to_spin_idx(s, i)][spat_to_spin_idx(s, i)]), dec_point),
                        np.round(cov[spat_to_spin_idx(s, i)][spat_to_spin_idx(s, i)], dec_point),
                        np.round(np.sum(np.abs(cov[spat_to_spin_idx(s, i)])) - cov[spat_to_spin_idx(s, i)][spat_to_spin_idx(s, i)], dec_point),
                        np.round(2 * cov[spat_to_spin_idx(s, i)][spat_to_spin_idx(s, i)] - np.sum(np.abs(cov[spat_to_spin_idx(s, i)])), dec_point),
                        np.round(100 * (2 * cov[spat_to_spin_idx(s, i)][spat_to_spin_idx(s, i)] - np.sum(np.abs(cov[spat_to_spin_idx(s, i)])) ) / cov[spat_to_spin_idx(s, i)][spat_to_spin_idx(s, i)], 1)
                        ])

            self.log.print_table(
                table_name = "LE Zombie diag.",
                column_names = ["mean", "std", "var", "gershgorin disc", "leeway", "leeway %"],
                row_names = np.arange(1, 2 * self.mol.nao + 1, 1, dtype = int),
                list_of_rows = diagnostic_table
                )

            self.log.print_matrix(cov, "covariance matrix", dec_points = 5)

        # -------------- making sure covariances dont overshadow the variances ----------------

        eigvals, eigvecs = np.linalg.eigh(cov)
        min_eigval = min(eigvals)
        if min_eigval <= 0.0:
            self.log.write(f"Warning: Covariance matrix is not positive-semidefinite (min eig = {min_eigval})")

        # We pre-sample the parameters
        # Some with cov...
        rand_X = functions.sample_with_autocorrelation_safe(means, cov, N * (N_subsample - N_no_cov))
        # ...and some without
        rand_X_no_cov = np.random.randn(N * N_no_cov, 2 * self.mol.nao) * np.sqrt(variances) + means

        if randomise_signs:
            self.log.write("Randomising parameter signs...")
            rand_X *= functions.randsign_mask(rand_X.shape)
            rand_X_no_cov *= functions.randsign_mask(rand_X_no_cov.shape)

        raw_Z_sample = [
            np.zeros((N, N_subsample, self.mol.nao)),
            np.zeros((N, N_subsample, self.mol.nao))
            ]

        if self.HF_method == "RHF":
            # We assume |Z_alpha> = |Z_beta>
            self.log.write("Affixing |Z_alpha> = |Z_beta> for each basis state...")
        elif self.HF_method == "UHF":
            # We do not assume |Z_alpha> = |Z_beta>
            self.log.write("Alpha and beta spin subspace parameters sampled only based on covariance")

        for n in range(N):
            # First, the cov elements...
            for n_sub in range(N_subsample - N_no_cov):
                for i in range(self.mol.nao):
                    raw_Z_sample[0][n][n_sub][i] = rand_X[n * (N_subsample - N_no_cov) + n_sub][spat_to_spin_idx("a", i)]
                    if self.HF_method == "RHF":
                        raw_Z_sample[1][n][n_sub][i] = raw_Z_sample[0][n][n_sub][i]
                    elif self.HF_method == "UHF":
                        raw_Z_sample[1][n][n_sub][i] = rand_X[n * (N_subsample - N_no_cov) + n_sub][spat_to_spin_idx("b", i)]
            # ...then the no cov elements.
            for n_sub in range(N_no_cov):
                for i in range(self.mol.nao):
                    raw_Z_sample[0][n][N_subsample - N_no_cov + n_sub][i] = rand_X_no_cov[n * N_no_cov + n_sub][spat_to_spin_idx("a", i)]
                    if self.HF_method == "RHF":
                        raw_Z_sample[1][n][N_subsample - N_no_cov + n_sub][i] = raw_Z_sample[0][n][N_subsample - N_no_cov + n_sub][i]
                    elif self.HF_method == "UHF":
                        raw_Z_sample[1][n][N_subsample - N_no_cov + n_sub][i] = rand_X_no_cov[n * N_no_cov + n_sub][spat_to_spin_idx("b", i)]


        N_vals = [1]
        convergence_sols = [self.reference_state_energy]

        matrix_condition = "H"

        msg = f"Conditioned sampling with ground state search on {N} states, each taken from {N_subsample} random states"
        #new_sem_ID = self.semaphor.create_event(np.linspace(0, N_subsample * ((N + 2) * (N + 1) / 2 - 1) + 1, 1000 + 1), msg)
        if matrix_condition == "H":
            max_tau =  N_subsample * ((N + 2) * (N + 1) / 2 - 1) + 1
        elif matrix_condition == "S":
            max_tau = ((N + 2) * (N + 1) / 2 - 1) + 1
        self.log.enter(msg, 1, True, tau_space = np.linspace(0, max_tau, 1000 + 1))

        for n in range(N):
            # We add the best out of 10 random states
            cur_subsample = []
            for n_sub in range(N_subsample):

                rand_z_alpha = CS_Qubit(self.mol.nao, self.S_alpha, raw_Z_sample[0][n][n_sub])
                rand_z_beta = CS_Qubit(self.mol.nao, self.S_beta, raw_Z_sample[1][n][n_sub])
                cur_subsample.append([rand_z_alpha, rand_z_beta])

            cur_sample.add_best_of_subsample(cur_subsample, update_semaphor = True, condition = matrix_condition, reject_high_overlap = True)
            N_vals.append(cur_sample.N)
            convergence_sols.append(cur_sample.E_ground[-1])

        procedure_diagnostic.append(f"Full sample condition number = {cur_sample.S_cond}")

        #solution_benchmark = self.semaphor.finish_event(new_sem_ID, "Evaluation")
        calc_duration = self.log.exit("Evaluation")

        csv_sol = [] # list of rows
        for i in range(len(N_vals)):
            csv_sol.append({"N" : N_vals[i], "E [H]" : float(convergence_sols[i])})



        # We estimate the error on a single-basis-state energy estimate
        N_err_est = 100

        self.log.enter("Estimating the error on an N=1 dataset", semaphored = True, tau_space = np.linspace(0, N_err_est, 100 + 1))

        E_err_min = []
        for n in range(N_err_est):
            cur_err_sample = CS_sample(self, CS_Qubit, add_ref_state = True)
            cur_rand_X = functions.sample_with_autocorrelation_safe(means, cov, N_subsample - N_no_cov)
            # ...and some without
            cur_rand_X_no_cov = np.random.randn(N_no_cov, 2 * self.mol.nao) * np.sqrt(variances) + means

            if randomise_signs:
                cur_rand_X *= functions.randsign_mask(cur_rand_X.shape)
                cur_rand_X_no_cov *= functions.randsign_mask(cur_rand_X_no_cov.shape)

            cur_raw_Z_sample = [
                np.zeros((N_subsample, self.mol.nao)),
                np.zeros((N_subsample, self.mol.nao))
                ]

            for n_sub in range(N_subsample - N_no_cov):
                for i in range(self.mol.nao):
                    cur_raw_Z_sample[0][n_sub][i] = cur_rand_X[n_sub][spat_to_spin_idx("a", i)]
                    if self.HF_method == "RHF":
                        cur_raw_Z_sample[1][n_sub][i] = cur_raw_Z_sample[0][n_sub][i]
                    elif self.HF_method == "UHF":
                        cur_raw_Z_sample[1][n_sub][i] = cur_rand_X[n_sub][spat_to_spin_idx("b", i)]
            # ...then the no cov elements.
            for n_sub in range(N_no_cov):
                for i in range(self.mol.nao):
                    cur_raw_Z_sample[0][N_subsample - N_no_cov + n_sub][i] = cur_rand_X_no_cov[n_sub][spat_to_spin_idx("a", i)]
                    if self.HF_method == "RHF":
                        cur_raw_Z_sample[1][N_subsample - N_no_cov + n_sub][i] = cur_raw_Z_sample[0][N_subsample - N_no_cov + n_sub][i]
                    elif self.HF_method == "UHF":
                        cur_raw_Z_sample[1][N_subsample - N_no_cov + n_sub][i] = cur_rand_X_no_cov[n_sub][spat_to_spin_idx("b", i)]

            cur_subsample = []
            for n_sub in range(N_subsample):

                rand_z_alpha = CS_Qubit(self.mol.nao, self.S_alpha, cur_raw_Z_sample[0][n_sub])
                rand_z_beta = CS_Qubit(self.mol.nao, self.S_beta, cur_raw_Z_sample[1][n_sub])
                cur_subsample.append([rand_z_alpha, rand_z_beta])

            cur_err_sample.add_best_of_subsample(cur_subsample, condition = matrix_condition, reject_high_overlap = True)
            E_err_min.append(cur_err_sample.E_ground[-1])

            self.log.update_semaphor_event(n + 1)

        self.log.exit("Error evaluation")

        E_base_err = np.std(E_err_min)
        self.log.write(f"Error on a single-state basis: {E_base_err}")

        # --------------- We extrapolate using inverse sqrt law ----------------
        E_zero, E_zero_err = self.extrapolate_by_inverse_sqrt(csv_sol)

        self.log.write(f"Value of lin fit at x = 0: {E_zero} +- {E_zero_err}.sigma_0, where sigma_0 is the error on a single datapoint.")



        self.disk_jockey.commit_datum_bulk(dataset_label, "basis_samples", cur_sample.get_z_tensor())
        self.disk_jockey.commit_datum_bulk(dataset_label, "result_energy_states", csv_sol)
        self.disk_jockey.commit_metadatum(dataset_label, "result_energy_states", {
            "E_g" : cur_sample.E_ground[-1],
            "E_base_err" : E_base_err,
            "E_extrapolated" : E_zero,
            "E_extrapolated_err" : E_zero_err,
            "duration" : calc_duration
            })
        self.diagnostics_log.append({f"find_ground_state_LEGS_Zombie_cov_RSOPM ({dataset_label})" : procedure_diagnostic})

        self.measured_datasets.append(dataset_label)

        self.log.write(f"Measured datasets: {self.measured_datasets}")

        self.log.exit()
        return(N_vals, convergence_sols)

    def find_ground_state_LEGS_Zombie_cov_RSOPM(self, **kwargs):

        # This method uses Zombie (Qubit) states.
        # It looks at the total norm of the states in | LE > which have i-th MO
        # occupied, i.e. < LE | b_i\hc b_i | LE >, and uses this to estimate
        # E[z_i^2].
        # The means are equal to sigmas, and the sample is then multiplied by a random sign mask

        # The difference from ALT is that when considering b_i b_j acting on one spin subspace, we sum over the abs values
        # of all contributions from the other spin subspace to account for simultaneous excitations!

        # kwargs:
        #     -N: Sample size
        #     -N_sub: Subsample size
        #     ------------------------
        #     -dataset_label: if present, this will label the dataset in disc jockey. Otherwise, label is generated from other kwargs.

        # -------------------- Parameter initialisation

        assert "N" in kwargs
        N = kwargs["N"]

        if "N_sub" in kwargs:
            N_subsample = kwargs["N_sub"]
        else:
            N_subsample = 1

        if "dataset_label" in kwargs:
            dataset_label = kwargs["dataset_label"]
        else:
            dataset_label = f"LEGS_Zombie_cov_RSOPM_{N}_{N_subsample}"

        self.log.enter(f"Obtaining the ground state with the method \"LEGS Zombie cov RSOPM\" [N = {N}, N_sub = {N_subsample}]", 1)


        # Disk jockey node creation and metadata storage
        self.disk_jockey.create_data_nodes({dataset_label : {"basis_samples" : "pkl", "result_energy_states" : "csv"}})
        self.disk_jockey.commit_metadatum(dataset_label, "basis_samples", {
                "method" : "LEGS_Zombie_cov_RSOPM", # required
                "params" : {
                    "N" : N,
                    "N_sub" : N_subsample
                }
            })
        self.user_actions += f"find_ground_state_LEGS_Zombie_cov_RSOPM [N = {N}, N_sub = {N_subsample}]\n"
        procedure_diagnostic = []

        if "LE_sol" not in self.checklist:
            self.log.write(f"ERROR: LEGS method requires LE solution to be known. Aborting...")
            self.log.exit()
            return(None)

        # We now manually sample Thouless states guided by the LE solution
        cur_sample = CS_sample(self, CS_Qubit, add_ref_state = True)

        # We find the means and the covariances
        spin_idx_dict = {"a" : 0, "b" : 1}
        spat_to_spin_idx = lambda sigma, i : spin_idx_dict[sigma] * self.mol.nao + i
        means = np.zeros( 2 * self.mol.nao )
        sq_means = np.zeros( 2 * self.mol.nao )
        variances = np.zeros( 2 * self.mol.nao )
        cov = np.zeros( (2 * self.mol.nao, 2 * self.mol.nao) )



        dec_point = 4

        # -------------------- Variances --------------------

        # Trying to reproduce dima's approach with all means set to zero
        #variances = np.array(([1.0, 1.0, 1e-4, 1e-6, 1e-6] + [0] * 5) * 2)
        #means = np.sqrt(np.array(np.sqrt(variances))) # mu = sigma
        #variances = np.array(([1.0, 1.0, 1e-4, 1e-6, 1e-6, 1e-6, 1e-7, 1e-8, 1e-8, 1e-7]) * 2)
        #cov = np.diag(variances)


        for i in range(self.S_alpha):
            variances[spat_to_spin_idx("a", i)] = self.LE_sol["RSOPM"][spat_to_spin_idx("a", i)][spat_to_spin_idx("a", i)]
        for i in range(self.S_alpha, self.mol.nao):
            variances[spat_to_spin_idx("a", i)] = self.LE_sol["RSOPM"][spat_to_spin_idx("a", i)][spat_to_spin_idx("a", i)]
        for i in range(self.S_beta):
            variances[spat_to_spin_idx("b", i)] = self.LE_sol["RSOPM"][spat_to_spin_idx("b", i)][spat_to_spin_idx("b", i)]
        for i in range(self.S_beta, self.mol.nao):
            variances[spat_to_spin_idx("b", i)] = self.LE_sol["RSOPM"][spat_to_spin_idx("b", i)][spat_to_spin_idx("b", i)]
        cov = np.diag(variances)


        cov_alpha = 1 # the off-diagonal rescaling free parameter

        # initialise covariances

        # note that the off-diagonal terms in RSOPM are just correlations.
        # So cov_ij = SOPM_ij * sqrt(var_i * var_j)

        # alpha-alpha
        """for i in range(self.mol.nao):
            for j in range(i + 1, self.mol.nao):
                cur_cov = self.LE_sol["RSOPM"][spat_to_spin_idx("a", i)][spat_to_spin_idx("a", j)] * np.sqrt(variances[spat_to_spin_idx("a", i)] * variances[spat_to_spin_idx("a", j)])
                cov[spat_to_spin_idx("a", i)][spat_to_spin_idx("a", j)] = cov_alpha * cur_cov
                cov[spat_to_spin_idx("a", j)][spat_to_spin_idx("a", i)] = cov[spat_to_spin_idx("a", i)][spat_to_spin_idx("a", j)] # conjugate?
        # beta-beta
        for i in range(self.mol.nao):
            for j in range(i + 1, self.mol.nao):
                cur_cov = self.LE_sol["RSOPM"][spat_to_spin_idx("b", i)][spat_to_spin_idx("b", j)] * np.sqrt(variances[spat_to_spin_idx("b", i)] * variances[spat_to_spin_idx("b", j)])
                cov[spat_to_spin_idx("b", i)][spat_to_spin_idx("b", j)] = cov_alpha * cur_cov
                cov[spat_to_spin_idx("b", j)][spat_to_spin_idx("b", i)] = cov[spat_to_spin_idx("b", i)][spat_to_spin_idx("b", j)] # conjugate?

        # alpha-beta
        # We remain agnostic! Because even tho this should work perfectly, I'm not sure how the gershgorin stuff will work
        """
        diagnostic_table = []
        for i in range(self.mol.nao):
            diagnostic_table.append([
                np.round(means[spat_to_spin_idx("a", i)], dec_point),
                np.round(np.sqrt(cov[spat_to_spin_idx("a", i)][spat_to_spin_idx("a", i)]), dec_point),
                np.round(cov[spat_to_spin_idx("a", i)][spat_to_spin_idx("a", i)], dec_point),
                np.round(np.sum(np.abs(cov[spat_to_spin_idx("a", i)])) - cov[spat_to_spin_idx("a", i)][spat_to_spin_idx("a", i)], dec_point),
                np.round(2 * cov[spat_to_spin_idx("a", i)][spat_to_spin_idx("a", i)] - np.sum(np.abs(cov[spat_to_spin_idx("a", i)])), dec_point),
                np.round(100 * (2 * cov[spat_to_spin_idx("a", i)][spat_to_spin_idx("a", i)] - np.sum(np.abs(cov[spat_to_spin_idx("a", i)])) ) / cov[spat_to_spin_idx("a", i)][spat_to_spin_idx("a", i)], 1)
                ])

        self.log.print_table(
            table_name = "LE Zombie diag.",
            column_names = ["mean", "std", "var", "gershgorin disc", "leeway", "leeway %"],
            row_names = np.arange(1, self.mol.nao + 1, 1, dtype = int),
            list_of_rows = diagnostic_table
            )

        self.log.print_matrix(cov[:self.mol.nao,:self.mol.nao], "covariance matrix", dec_points = 5)

        # -------------- making sure covariances dont overshadow the variances ----------------

        eigvals, eigvecs = np.linalg.eigh(cov)
        min_eigval = min(eigvals)
        if min_eigval <= 0.0:
            self.log.write(f"Warning: Covariance matrix is not positive-semidefinite (min eig = {min_eigval})")

        # We pre-sample the parameters
        rand_X = functions.sample_with_autocorrelation_safe(means, cov, N * N_subsample)

        raw_Z_sample = [
            np.zeros((N, N_subsample, self.mol.nao)),
            np.zeros((N, N_subsample, self.mol.nao))
            ]
        for n in range(N):
            for n_sub in range(N_subsample):
                for i in range(self.mol.nao):
                    raw_Z_sample[0][n][n_sub][i] = rand_X[n * N_subsample + n_sub][spat_to_spin_idx("a", i)]
                    raw_Z_sample[1][n][n_sub][i] = raw_Z_sample[0][n][n_sub][i] #rand_X[n * N_subsample + n_sub][spat_to_spin_idx("b", i)] #

        N_vals = [1]
        convergence_sols = [self.reference_state_energy]

        msg = f"Conditioned sampling with ground state search on {N} states, each taken from {N_subsample} random states"
        #new_sem_ID = self.semaphor.create_event(np.linspace(0, N_subsample * ((N + 2) * (N + 1) / 2 - 1) + 1, 1000 + 1), msg)
        self.log.enter(msg, 1, True, tau_space = np.linspace(0, N_subsample * ((N + 2) * (N + 1) / 2 - 1) + 1, 1000 + 1))

        for n in range(N):
            # We add the best out of 10 random states
            cur_subsample = []
            for n_sub in range(N_subsample):

                rand_z_alpha = CS_Qubit(self.mol.nao, self.S_alpha, raw_Z_sample[0][n][n_sub])
                rand_z_beta = CS_Qubit(self.mol.nao, self.S_beta, raw_Z_sample[1][n][n_sub])
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
        self.disk_jockey.commit_metadatum(dataset_label, "result_energy_states", {"E_g" : cur_sample.E_ground[-1]})
        self.diagnostics_log.append({f"find_ground_state_LEGS_Zombie_cov_RSOPM ({dataset_label})" : procedure_diagnostic})

        self.measured_datasets.append(dataset_label)

        self.log.write(f"Measured datasets: {self.measured_datasets}")

        self.log.exit()
        return(N_vals, convergence_sols)

    def find_ground_state_from_z_tensor(self, **kwargs):

        # This method uses Zombie (Qubit) states.
        # We provide a fixed z-tensor and the method uses it. The ref state is added by default (since it is the same for all geometries)

        # kwargs:
        #     -z (the z-tensor in the form [basis index][spin index][param index])
        #     ------------------------
        #     -dataset_label: if present, this will label the dataset in disc jockey. Otherwise, label is generated from other kwargs.

        # -------------------- Parameter initialisation

        assert "z" in kwargs
        z_tensor = kwargs["z"]

        if "dataset_label" in kwargs:
            dataset_label = kwargs["dataset_label"]
        else:
            dataset_label = f"manual_sample"

        self.log.enter(f"Obtaining the ground state with the method \"manual_sample\" [len of z: {len(z_tensor)}]", 1)


        # Disk jockey node creation and metadata storage
        self.disk_jockey.create_data_nodes({dataset_label : {"basis_samples" : "pkl", "result_energy_states" : "csv"}})
        self.disk_jockey.commit_metadatum(dataset_label, "basis_samples", {
                "method" : "manual_sample", # required
                "params" : {
                }
            })
        self.user_actions += f"find_ground_state_from_z_tensor\n"
        procedure_diagnostic = []

        N = len(z_tensor)

        cur_sample = [] # list of CS_Qubit

        N_vals = []
        convergence_sols = []

        msg = f"Evaluating the Hamiltonian overlap integrals on the {N} basis states provided"
        self.log.enter(msg, 1, True, tau_space = np.linspace(0, N * (N + 1) / 2, 1000 + 1))

        H_matrix = np.zeros((N, N), dtype = complex)
        S_matrix = np.identity(N, dtype = complex)

        for n in range(N):
            # We add the next basis state
            cur_z_alpha = CS_Qubit(self.mol.nao, self.S_alpha, z_tensor[n][0])
            cur_z_beta = CS_Qubit(self.mol.nao, self.S_beta, z_tensor[n][1])
            cur_sample.append([cur_z_alpha, cur_z_beta])

            # We calculate its self energy and overlaps with previously added states
            for a in range(n):
                S_matrix[a][n] = cur_sample[a][0].norm_overlap(cur_sample[n][0]) * cur_sample[a][1].norm_overlap(cur_sample[n][1])
                S_matrix[n][a] = np.conjugate(S_matrix[a][n])

            for a in range(n):
                H_matrix[a][n] = self.H_overlap(cur_sample[a], cur_sample[n])
                H_matrix[n][a] = np.conjugate(H_matrix[a][n])

                self.log.update_semaphor_event(n * (n + 1) / 2 + a + 1)
            H_matrix[n][n] = self.H_overlap(cur_sample[n], cur_sample[n])
            self.log.update_semaphor_event(n * (n + 1) / 2 + n + 1)


            N_vals.append(n + 1)
            energy_levels, _ = sp.linalg.eigh(H_matrix[:n + 1, :n + 1], S_matrix[:n + 1, :n + 1])
            ground_state_index = np.argmin(energy_levels)
            convergence_sols.append(energy_levels[ground_state_index])


        procedure_diagnostic.append(f"Full sample condition number = {np.linalg.cond(S_matrix)}")

        #solution_benchmark = self.semaphor.finish_event(new_sem_ID, "Evaluation")
        calc_duration = self.log.exit("Evaluation")

        csv_sol = [] # list of rows
        for i in range(len(N_vals)):
            csv_sol.append({"N" : N_vals[i], "E [H]" : float(convergence_sols[i])})

        # Extrapolate the g.s. by inverse sqrt (the rel error has to later be multiplied by the error on a single d.p. which comes with the provided z tensor)
        E_zero, E_zero_err = self.extrapolate_by_inverse_sqrt(csv_sol)
        self.log.write(f"Value of lin fit at x = 0: {E_zero} +- {E_zero_err}.sigma_0, where sigma_0 is the error on a single datapoint.")

        self.disk_jockey.commit_datum_bulk(dataset_label, "basis_samples", z_tensor)
        self.disk_jockey.commit_datum_bulk(dataset_label, "result_energy_states", csv_sol)
        self.disk_jockey.commit_metadatum(dataset_label, "result_energy_states", {
            "E_g" : convergence_sols[-1],
            "E_extrapolated" : E_zero,
            "E_extrapolated_err" : E_zero_err,
            "duration" : calc_duration
            })
        self.diagnostics_log.append({f"find_ground_state_from_z_tensor ({dataset_label})" : procedure_diagnostic})

        self.measured_datasets.append(dataset_label)

        self.log.write(f"Measured datasets: {self.measured_datasets}")

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


    # Helper methods

    def extrapolate_by_inverse_sqrt(self, dataset_dict, min_N = 10):
        sigma_zero = 1.0 # dummy val of inherent err

        N_space = []
        E_space = []
        trim_i = 0
        for row in dataset_dict:
            N_space.append(row["N"])
            E_space.append(row["E [H]"])

        while(N_space[trim_i] < min_N):
            trim_i += 1
            if trim_i == len(N_space):
                trim_i = 0
                break


        sqinv_N_space = np.array(1/np.sqrt(N_space))
        sqinv_N_space_yerr = sigma_zero * sqinv_N_space # the lower the N value, the higher the uncertainty! propto 1/sqrt(N). Res err is quoted as a proportion to the unknown coef
        popt, pcov = sp.optimize.curve_fit(
            lambda x, l, c: l * x + c,
            sqinv_N_space[trim_i:],
            E_space[trim_i:],
            sigma=sqinv_N_space_yerr[trim_i:],
            absolute_sigma=True
        )
        l_best, c_best = popt
        l_err, c_err = np.sqrt(np.diag(pcov))

        E_zero = c_best
        E_zero_err = c_err / sigma_zero

        return(E_zero, E_zero_err)

    def get_dataset_info(self, dataset_label):
        if dataset_label not in self.disk_jockey.metadata.keys():
            return(None)
        return([
            self.ci_energy,
            self.reference_state_energy,
            self.disk_jockey.metadata[dataset_label]["result_energy_states"]["E_g"],
            self.disk_jockey.metadata[dataset_label]["result_energy_states"]["E_extrapolated"],
            self.disk_jockey.metadata[dataset_label]["result_energy_states"]["E_extrapolated_err"],
            self.disk_jockey.metadata[dataset_label]["result_energy_states"]["duration"]
            ])

    ###########################################################################
    # ----------------------------- User methods ------------------------------
    ###########################################################################

    def initialise_molecule(self, mol, HF_method = "default"):
        # mol is an instance of pyscf.gto.Mole
        # HF_method is a magic word which determines how we calculate MOs:
        #   -"RHF": Restricted Hartree-Fock
        #   -"UHF": Unrestricted Hartree-Fock
        #   -"default": RHF if singlet mol, UHF otherwise

        self.log.enter("Initialising molecule...", 0)

        if "mol_init" in self.checklist:
            self.log.write("Molecule already initialised.", 1)
            return(None)

        self.log.enter("Building molecule...", 1)

        # TODO move constructing molecule here so we can store just the parameters!

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
        AO_H_two_chemist = self.mol.intor('int2e')
        # <ij|kl> = (ik|jl)

        self.log.exit()

        self.log.enter("Performing mean-field calculations to determine the molecular orbitals...", 1)

        if HF_method == "RHF":
            self.HF_method = HF_method
            self.log.write("Mean field method: Restricted Hartree-Fock (selected by user)")
            if self.mol.spin != 0:
                self.log.write("WARNING: The molecule is not a singlet, RHF is unsuitable.")
        elif HF_method == "UHF":
            self.HF_method = HF_method
            self.log.write("Mean field method: Unrestricted Hartree-Fock (selected by user)")
        elif HF_method == "default":
            if self.mol.spin == 0:
                self.HF_method = "RHF"
                self.log.write("Mean field method: Restricted Hartree-Fock (determined automatically for a singlet molecule)")
            else:
                self.HF_method = "UHF"
                self.log.write(f"Mean field method: Unrestricted Hartree-Fock (determined automatically for a {self.spin_label()} molecule)")

        # We now construct AO_H_two as the coefficient tensor in second quantisation according to Szabo & Ostlund: Modern Quantum Chemistry p. 95, Eq. 2.232
        # O_2 = 0.5 * sum_ijkl <ij|kl> f\hc_i f\hc_j f_l f_k
        # Using <ij|kl> = (ik|jl) we have
        # O_ijkl = 0.5 (il|jk)
        # By symmetry: (ij|kl) = (ji|kl) = (ij|lk) = (ji|lk)
        # Hence O_ijkl = O_ljki = O_ikjl = O_lkji


        # We initialise self.MO_coefs alpha/beta, self.MO_H_one,two a/b/ab as dicts with spin in key
        self.MO_coefs = {}
        self.MO_H_one = {}
        self.MO_H_two = {}

        # MO_H_two has three elements:
        #   ["a"]_ijkl = <i,a j,a | k,a l,a>
        #   ["b"]_ijkl = <i,b j,b | k,b l,b>
        #   ["ab"]_ijkl = <i,a j,b | k,a l,b>
        #   The second spin-mixed term is obtained by a double transpose; ["ba"]_ijkl = ["ab"]_jilk

        if self.HF_method == "RHF":
            # Everything is the same in both subspaces
            self.log.write("Finding the molecular orbitals using mean-field approximations...", 1)
            self.mean_field = scf.RHF(self.mol).run(verbose = 0)

            self.MO_coefs["a"] = self.mean_field.mo_coeff
            self.MO_coefs["b"] = self.MO_coefs["a"]
            self.reference_state_energy = self.mean_field.e_tot
            self.log.write(f"Done! Reference state energy is {self.reference_state_energy:0.5f}", 1)

            self.log.write("Transforming 1e and 2e integrals to MO basis...", 3)
            self.MO_H_one["a"] = np.matmul(self.MO_coefs["a"].T, np.matmul(AO_H_one, self.MO_coefs["a"]))
            self.MO_H_one["b"] = self.MO_H_one["a"]
            MO_H_two_packed = ao2mo.kernel(AO_H_two_chemist, self.MO_coefs["a"])
            MO_H_two_chemist = ao2mo.restore(1, MO_H_two_packed, self.MO_coefs["a"].shape[1]) # In mulliken notation: MO_H_two[p][q][r][s] = (pq|rs)

            self.MO_H_two["a"] = MO_H_two_chemist.transpose(0, 2, 1, 3)# - MO_H_two_chemist.transpose(0, 3, 1, 2)
            self.MO_H_two["b"] = self.MO_H_two["a"]
            self.MO_H_two["ab"] = self.MO_H_two["a"]

            # We construct the second quantised space as occupied \oplus unoccupied orbitals
            occ_orbs_alpha = [i for i, o in enumerate(self.mean_field.mo_occ) if o > 0]
            occ_orbs_beta = occ_orbs_alpha

        elif self.HF_method == "UHF":
            # Coefs and exchange intergrals differ for the two subspaces
            self.log.write("Finding the molecular orbitals using mean-field approximations...", 1)
            mf_object = scf.UHF(self.mol)

            mf_object.init_guess = 'atom'  # Atomic initial guess. If doesn't work, run RHF first and then use its result as the initial guess
            #mf_rhf = scf.RHF(self.mol).run(verbose=0)
            #mf_uhf = scf.UHF(self.mol)
            #mf_uhf.init_guess = 'atom'
            #dm0 = mf_rhf.make_rdm1()
            #mean_field = mf_uhf.kernel(dm0=dm0, verbose=0)
            mf_object.conv_tol = 1e-10 # Tighter convergence

            self.mean_field = mf_object.run(verbose = 0)
            self.MO_coefs["a"] = self.mean_field.mo_coeff[0]
            self.MO_coefs["b"] = self.mean_field.mo_coeff[1]

            assert self.MO_coefs["a"].shape[1] == self.MO_coefs["b"].shape[1]
            # This is not required but if we remove the constraint we need to
            # firstly restore symmetry, making the mixed H_two["ab"] non-square

            self.reference_state_energy = self.mean_field.e_tot
            self.log.write(f"Done! Reference state energy is {self.reference_state_energy:0.5f}", 1)

            self.log.write("Transforming 1e and 2e integrals to MO basis...", 3)
            self.MO_H_one["a"] = np.matmul(self.MO_coefs["a"].T, np.matmul(AO_H_one, self.MO_coefs["a"]))
            self.MO_H_one["b"] = np.matmul(self.MO_coefs["b"].T, np.matmul(AO_H_one, self.MO_coefs["b"]))
            MO_H_two_packed_alpha = ao2mo.kernel(AO_H_two_chemist, self.MO_coefs["a"])
            MO_H_two_packed_beta = ao2mo.kernel(AO_H_two_chemist, self.MO_coefs["b"])
            MO_H_two_packed_ab = ao2mo.kernel(AO_H_two_chemist, (self.MO_coefs["a"], self.MO_coefs["a"], self.MO_coefs["b"], self.MO_coefs["b"])) # in chemist's the spins are (aa|bb)
            # Removes symmetry to make number access fast
            MO_H_two_chemist_alpha = ao2mo.restore(1, MO_H_two_packed_alpha, self.MO_coefs["a"].shape[1])
            MO_H_two_chemist_beta = ao2mo.restore(1, MO_H_two_packed_beta, self.MO_coefs["b"].shape[1])
            MO_H_two_chemist_ab = ao2mo.restore(1, MO_H_two_packed_ab, self.MO_coefs["a"].shape[1])

            self.MO_H_two["a"] = MO_H_two_chemist_alpha.transpose(0, 2, 1, 3)
            self.MO_H_two["b"] = MO_H_two_chemist_beta.transpose(0, 2, 1, 3)
            self.MO_H_two["ab"] = MO_H_two_chemist_ab.transpose(0, 2, 1, 3)

            # We construct the second quantised space as occupied \oplus unoccupied orbitals
            occ_orbs_alpha = [i for i, o in enumerate(self.mean_field.mo_occ[0]) if o > 0]
            occ_orbs_beta = [i for i, o in enumerate(self.mean_field.mo_occ[1]) if o > 0]

        # TODO note that by using RHF, we assume N_alpha = N_beta. We can generalise the process by using UHF,
        # but this would mean using separate MO coeffs for alpha and beta subspaces

        self.log.exit()

        self.log.enter("Reference state analysis", 4)
        self.log.write(f"Occupied orbitals:", 4)
        self.log.write(f"  -in the spin-alpha subspace: {occ_orbs_alpha}", 4)
        self.log.write(f"  -in the spin-beta subspace:  {occ_orbs_beta}", 4)

        self.log.write("Testing each CS type Hamiltonian overlap evaluation against the reference state...")
        for CS_type in self.coherent_state_types.keys():
            null_state_alpha = self.coherent_state_types[CS_type].null_state(self.mol.nao, self.S_alpha)
            null_state_beta = self.coherent_state_types[CS_type].null_state(self.mol.nao, self.S_beta)
            null_state = [null_state_alpha, null_state_beta]
            null_state_direct_self_energy = self.H_overlap(null_state, null_state).real
            if np.round(null_state_direct_self_energy, 5) == np.round(self.reference_state_energy, 5):
                nse_comment = "agrees"
            else:
                nse_comment = "disagrees"
            self.log.write(f"  -For {CS_type}: E_ref = {null_state_direct_self_energy:0.5f}, which {nse_comment} with the true value", 4)
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
            "MO_H_one" : self.MO_H_one,
            "MO_H_two" : self.MO_H_two
            }
        self.disk_jockey.commit_datum_bulk("molecule", "mol_structure", mol_structure_bulk)
        self.disk_jockey.commit_datum_bulk("molecule", "mode_structure", mode_structure_bulk)

        self.check_off("mol_init")

        self.log.exit()

    def full_CI_sol(self):
        self.user_actions += f"full_CI_sol\n"

        self.log.enter("Performing SCF on full CI...", 0)

        if "full_CI_sol" in self.checklist:
            print("Full CI by PySCF already performed.")
            return(self.ci_energy)

        if self.HF_method == "RHF":
            cisolver = fci.FCI(self.mol, self.MO_coefs["a"])
        elif self.HF_method == "UHF":
            cisolver = fci.FCI(self.mol, (self.MO_coefs["a"], self.MO_coefs["b"]))
        else:
            self.log.write("WARNING: Unknown HF method, cannot run full CI calculation.")
            return(None)

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

        self.check_off("full_CI_sol")

        self.log.write(f"Ground state energy as calculated by SCF (full configuration) = {self.ci_energy:0.5f}", 0)
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





    def get_H_two_element(self, p, q, r, s):
        return(self.H_two[p][q][r][s])
        #return(self.H_two[int(p * (p * p * p + 2 * p * p + 3 * p + 2) / 8 + p * q * (p + 1) / 2 + q * (q + 1) / 2   + r * (r + 1) / 2 + s )])

    def mode_exchange_energy(self, m_i, m_f, spin = None, debug = False):
        # This function translates mode indices to spatial MO indices and returns the coefficient tensor element

        # If Hilbert space is spin-symmetrical, the spin argument is not necessary
        # Otherwise, spin is either "a", "b", or "ab" (to access "ba", do double transpose on "ab")

        if spin is None:
            if self.HF_method == "RHF":
                spin = "a" # doesn't matter
            else:
                self.log.write("WARNING: Called mode_exchange_energy without specifying spin when Hilbert space is not assumed to be spin-symmetrical. Returning None...", 0)
                return(None)

        # m_i/f are lists of either one mode index (single electron exchange) or two mode indices (two electron exchange)
        # the indices in m_i/f are SPATIAL, and are interpreted as acting on one spin state only.
        if len(m_i) == 1:
            return(self.MO_H_one[spin][m_i[0]][m_f[0]])
        elif len(m_i) == 2:
            #if m_i[0][1] == m_f[1][1] and m_i[1][1] == m_f[0][1]:
            return(self.MO_H_two[spin][m_i[0]][m_i[1]][m_f[0]][m_f[1]])
        return(0.0)

    def H_overlap(self, pair_a, pair_b):
        # This method calculates <Z_a | H | Z_b>, including the nuclear self-energy
        # here pair_a/b = [state_a/b with spin alpha, state_a/b with spin beta]

        alpha_overlap = pair_a[0].norm_overlap(pair_b[0])
        beta_overlap = pair_a[1].norm_overlap(pair_b[1])

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
                H_one_term += self.mode_exchange_energy([p], [q], "a") * W_alpha[p][q] * beta_overlap
                # beta
                H_one_term += self.mode_exchange_energy([p], [q], "b") * W_beta[p][q] * alpha_overlap

        H_two_term = 0.0

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

                prefactor_same_spin_alpha = 1.0 * (self.mode_exchange_energy([i, j], [k, l], "a") - self.mode_exchange_energy([i, j], [l, k], "a"))
                prefactor_same_spin_beta = 1.0 * (self.mode_exchange_energy([i, j], [k, l], "b") - self.mode_exchange_energy([i, j], [l, k], "b"))

                # alpha alpha
                H_two_term += prefactor_same_spin_alpha * pair_a[0].norm_overlap(pair_b[0], [j, i], [l, k]) * beta_overlap

                # beta beta
                H_two_term += prefactor_same_spin_beta * pair_a[1].norm_overlap(pair_b[1], [j, i], [l, k]) * alpha_overlap

                upup += prefactor_same_spin_alpha * pair_a[0].norm_overlap(pair_b[0], [j, i], [l, k]) * beta_overlap
                downdown += prefactor_same_spin_beta * pair_a[1].norm_overlap(pair_b[1], [j, i], [l, k]) * alpha_overlap

        # opposite spin
        for i in range(self.mol.nao):
            for j in range(self.mol.nao):
                for k in range(self.mol.nao):
                    for l in range(self.mol.nao):
                        #prefactor = 0.5 * self.mode_exchange_energy([i, j], [k, l])

                        prefactor_alpha_beta = 0.5 * self.mode_exchange_energy([i, j], [k, l], "ab")
                        prefactor_beta_alpha = 0.5 * self.mode_exchange_energy([j, i], [l, k], "ab")

                        # alpha beta
                        #H_two_term += prefactor * pair_a[0].norm_overlap(pair_b[0], [i], [k]) * pair_a[1].norm_overlap(pair_b[1], [j], [l])
                        H_two_term += prefactor_alpha_beta * W_alpha[i][k] * W_beta[j][l]

                        # beta alpha
                        #H_two_term += prefactor * pair_a[0].norm_overlap(pair_b[0], [j], [l]) * pair_a[1].norm_overlap(pair_b[1], [i], [k])
                        H_two_term += prefactor_beta_alpha * W_alpha[j][l] * W_beta[i][k]

                        mixed += prefactor_alpha_beta * W_alpha[i][k] * W_beta[j][l] + prefactor_beta_alpha * W_alpha[j][l] * W_beta[i][k]


        #print(H_one_term, H_two_term)
        #print("Up-up:", upup)
        #print("Down-down:", downdown)
        #print("Mixed spin:", mixed)

        H_nuc = self.mol.energy_nuc() * alpha_overlap * beta_overlap
        return(H_one_term + H_two_term + H_nuc)


    def find_ground_state(self, method, **kwargs):
        if method in self.find_ground_state_methods.keys():
            return(self.find_ground_state_methods[method](**kwargs))
        else:
            self.log.write(f"ERROR: Unknown ground state method {method}. Available methods: {self.find_ground_state_methods.keys()}")
            return(None)

    ###########################################################################
    ##################### Low-excitation guided sampling ######################
    ###########################################################################

    # --------------------- Occupancy bitstring labelling ---------------------

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

    def occ_tuple_restore(self, occ_tuple_repr):
        # Restores a tuple which was converted to str directly
        return( tuple([tuple([int(x) for x in occ.split(", ")]) for occ in occ_tuple_repr.strip("()").split("), (")]) )

    def occ_list_to_occ_tuple(self, occ_list):
        if isinstance(occ_list, tuple):
            return(occ_list) # just to regularise user action
        # There may be trailing zeros. Let's get rid of them with a cursed one-liner
        return(tuple(occ_list[:len(occ_list) - occ_list[::-1].index(1)]))

    def occ_tuple_to_list(self, occ_tuple):
        occ_a, occ_b = occ_tuple
        list_a = list(occ_a)
        list_b = list(occ_b)
        return([ list_a + [0] * (self.mol.nao - len(list_a)), list_b + [0] * (self.mol.nao - len(list_b)) ])


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

    def get_prom_label(self, bitlist, trim_M = None, hr = False):
        # Returns a tuple [[de-occupied MOs], [promoted MOs]] from ref state
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
        for i in range(cur_S, min(act_M, len(bitlist))):
            if bitlist[i] == 1:
                res[1].append(i + hr_cor)

        return(res)

    # Prom tuple methods
    # a prom tuple labels a low-excitation state by the following format: ((a, b), (c, d))
    # where a, b are m-tuples and c, d are n-tuples.
    # This labels the state obtained by promoting electrons from indices in a into b and from c into d in the alpha and beta spin spaces, respectively

    def occ_list_to_prom_tuple(self, occ_a, occ_b):
        prom_label_a = self.get_prom_label(occ_a)
        for i in range(len(prom_label_a[1])):
            prom_label_a[1][i] -= self.S_alpha
        prom_label_b = self.get_prom_label(occ_b)
        for i in range(len(prom_label_b[1])):
            prom_label_b[1][i] -= self.S_beta
        return( ( (tuple(prom_label_a[0]), tuple(prom_label_a[1])), (tuple(prom_label_b[0]), tuple(prom_label_b[1])) ) )

    def str_to_prom_tuple(self, prom_str):
        # Restoring on data load

        proms_by_spin = prom_str.lstrip("(").rstrip(")").split(")), ((")
        index_lists_alpha = proms_by_spin[0].lstrip("(").rstrip(")").split("), (")
        index_lists_beta = proms_by_spin[1].lstrip("(").rstrip(")").split("), (")

        if index_lists_alpha[0] == '':
            index_lists_alpha_tuple = ((), ())
        else:
            index_lists_alpha_tuple = tuple([tuple([int(x) for x in occ.rstrip(",").split(", ")]) for occ in index_lists_alpha])
        if index_lists_beta[0] == '':
            index_lists_beta_tuple = ((), ())
        else:
            index_lists_beta_tuple = tuple([tuple([int(x) for x in occ.rstrip(",").split(", ")]) for occ in index_lists_beta])

        return( (
            index_lists_alpha_tuple,
            index_lists_beta_tuple
            ) )

    def get_ref_state(self):
        # Returns a list of lists, useful for further modification
        return([[1] * self.S_alpha + [0] * (self.mol.nao - self.S_alpha), [1] * self.S_beta + [0] * (self.mol.nao - self.S_beta)])

    # -------------------------------------------------------------------------
    # ------------------ Accessing the full CI ground state -------------------
    # -------------------------------------------------------------------------
    # Note that ci_sol is a dict with 2-tuples of occ-tuples as keys.

    def print_ground_state(self):
        assert self.ci_sol is not None

        print("Printing ground state solution on the full CI...")
        print(self.ci_sol)

    def ground_state_component(self, alpha_occupancy, beta_occupancy):
        assert self.ci_sol is not None

        # Access coefficient
        if isinstance(alpha_occupancy, tuple):
            return(self.ci_sol[(alpha_occupancy, beta_occupancy)])
        elif isinstance(alpha_occupancy, list):
            return(self.ci_sol[(self.occ_list_to_occ_tuple(alpha_occupancy), self.occ_list_to_occ_tuple(beta_occupancy) )])
        #return(self.ci_sol[alpha_idx, beta_idx])

    # ------------------------------ RHF methods ------------------------------

    # Closed shell methods (RHF SACs with 0 openness)

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

    # Two-open-shell methods (RHF SACs with openness 1)

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

    # ------------------------------ UHF methods ------------------------------

    def get_simultaneously_excited_states(self, top_N = None):
        # returns
        #   -a list of top states (as 2-tuples of occ tuples) with one
        #    excitation per spin subspace
        #   -the total number of such states
        #   -The norm squared of the ci_sol projection onto these states (None
        #    if unknown)

        self.log.enter(f"Obtaining all states with one excitation on both spin subspaces...", 8)

        # First, check if there are any such states
        if min(self.S_alpha, self.mol.nao - self.S_alpha) < 1 or min(self.S_beta, self.mol.nao - self.S_beta) < 1:
            self.log.write("No excitation allowed on one of the spin subspaces.", 9)
            self.log.exit()
            return([])

        res = []

        for i in range(self.S_alpha):
            for j in range(self.mol.nao - self.S_alpha):
                for k in range(self.S_beta):
                    for l in range(self.mol.nao - self.S_beta):
                        # Note that we omit the trailing zeros to agree with the ci_sol convention
                        res.append((
                                (1,) * i + (0,) + (1,) * (self.S_alpha - 1 - i) + (0,) * j + (1,),
                                (1,) * k + (0,) + (1,) * (self.S_beta - 1 - k) + (0,) * l + (1,)
                            ))

        self.log.write(f"Obtained {len(res)} such states.", 9)
        self.log.exit()
        return(res)

    def get_states_by_excitation_number(self, N_exc_a, N_exc_b, form = "tuple"):
        # returns a list of all states (as 2-tuples of occ tuples) with the
        # exact number of excitations from the reference state.
        # N_exc_a: number of excitations in the spin-alpha subspace
        # N_exc_b: number of excitations in the spin-beta subspace

        self.log.enter(f"Obtaining all states with {N_exc_a} excitations on spin-alpha and {N_exc_b} excitations on spin-beta...", 4)

        # First, check if there are any such states
        if min(self.S_alpha, self.mol.nao - self.S_alpha) < N_exc_a or min(self.S_beta, self.mol.nao - self.S_beta) < N_exc_b:
            self.log.write("Number of excitations is too large. No such states exist.", 5)
            self.log.exit()
            return([])

        a_from = functions.subset_indices(np.arange(self.S_alpha), N_exc_a)
        a_to = functions.subset_indices(np.arange(self.S_alpha, self.mol.nao), N_exc_a)
        b_from = functions.subset_indices(np.arange(self.S_beta), N_exc_b)
        b_to = functions.subset_indices(np.arange(self.S_beta, self.mol.nao), N_exc_b)

        res = []

        for i in range(len(a_from)):
            for j in range(len(a_to)):
                for k in range(len(b_from)):
                    for l in range(len(b_to)):
                        cur_state = [[1] * self.S_alpha + [0] * (1 + max(a_to[j]) - self.S_alpha), [1] * self.S_beta + [0] * (1 + max(b_to[l]) - self.S_beta)]
                        for a_i in range(N_exc_a):
                            cur_state[0][a_from[i][a_i]] = 0
                            cur_state[0][a_to[j][a_i]] = 1
                        for b_i in range(N_exc_b):
                            cur_state[1][b_from[k][b_i]] = 0
                            cur_state[1][b_to[l][b_i]] = 1
                        if form == "tuple":
                            res.append((self.occ_list_to_occ_tuple(cur_state[0]), self.occ_list_to_occ_tuple(cur_state[1])))
                        else:
                            res.append(cur_state)

        self.log.write(f"Obtained {len(res)} such states.", 5)
        self.log.exit()
        return(res)

    def get_top_single_excitation_states(self, N_top = 10):
        # returns
        #   -a list of lists of top states (as 2-tuples of occ tuples) with one
        #    excitation in total (no fluff if all states), one per subspace
        #   -the total number of such states
        #   -The norm squared of the ci_sol projection onto these states (None
        #    if unknown)

        if N_top is None:
            self.log.enter(f"Obtaining all states with one excitation in total...", 4)
        else:
            N_top_a = min(N_top, self.S_alpha * (self.mol.nao - self.S_alpha))
            N_top_b = min(N_top, self.S_beta * (self.mol.nao - self.S_beta))
            self.log.enter(f"Obtaining top ({N_top_a}, {N_top_b}) states with one excitation in spin (alpha, beta) as measured by overlap in true ground state...", 4)

        if "full_CI_sol" not in self.checklist:
            self.log.write("WARNING: Full CI ground state solution has not been found yet. Aborting...")
            self.log.exit()
            return(None)

        res = [[], []] #[0/1][i] = [n_sq, key, prom_label]; ordered by n_sq desc.
        if N_top is not None:
            for i in range(N_top_a):
                res[0].append([0, None, None])
            for i in range(N_top_b):
                res[1].append([0, None, None])

        total_norm_squared_projection_a = 0.0
        total_norm_squared_projection_b = 0.0

        base_state_alpha = (1,) * self.S_alpha
        base_state_beta = (1,) * self.S_beta

        for i in range(self.S_alpha):
            for j in range(self.mol.nao - self.S_alpha):
                # Note that we omit the trailing zeros to agree with the ci_sol convention
                cur_state_alpha = (1,) * i + (0,) + (1,) * (self.S_alpha - 1 - i) + (0,) * j + (1,)
                cur_c = self.ground_state_component(cur_state_alpha, base_state_beta)
                cur_n_sq = cur_c * cur_c

                total_norm_squared_projection_a += cur_n_sq

                if N_top is not None:
                    if cur_n_sq > res[0][-1][0]:
                        # belongs to the list
                        new_i = len(res[0])
                        while(cur_n_sq > res[0][new_i - 1][0]):
                            new_i -= 1
                            if new_i == 0:
                                break
                        res[0].insert(new_i, [cur_n_sq, cur_state_alpha, f"({i+1} -> {j + self.S_alpha + 1})"])
                        res[0].pop()
                else:
                    # Not discriminating
                    res[0].append(cur_state_beta)
        for k in range(self.S_beta):
            for l in range(self.mol.nao - self.S_beta):
                # Note that we omit the trailing zeros to agree with the ci_sol convention
                cur_state_beta = (1,) * k + (0,) + (1,) * (self.S_beta - 1 - k) + (0,) * l + (1,)
                cur_c = self.ground_state_component(base_state_alpha, cur_state_beta)
                cur_n_sq = cur_c * cur_c

                total_norm_squared_projection_b += cur_n_sq

                if N_top is not None:
                    if cur_n_sq > res[1][-1][0]:
                        # belongs to the list
                        new_i = len(res[1])
                        while(cur_n_sq > res[1][new_i - 1][0]):
                            new_i -= 1
                            if new_i == 0:
                                break
                        res[1].insert(new_i, [cur_n_sq, cur_state_beta, f"({k+1} -> {l + self.S_beta + 1})"])
                        res[1].pop()
                else:
                    # Not discriminating
                    res[1].append(cur_state_beta)


        self.log.write(f"Obtained ({len(res[0])}, {len(res[1])}) such states.", 5)
        self.log.exit()
        return(res, (self.S_alpha * (self.mol.nao - self.S_alpha), self.S_beta * (self.mol.nao - self.S_beta)), (total_norm_squared_projection_a, total_norm_squared_projection_b))

    def get_top_simultaneously_excited_states(self, N_top = 10):
        # returns
        #   -a list of top states (as 2-tuples of occ tuples) with one
        #    excitation per spin subspace (no fluff if all states)
        #   -the total number of such states
        #   -The norm squared of the ci_sol projection onto these states (None
        #    if unknown)

        if N_top is None:
            self.log.enter(f"Obtaining all states with one excitation on both spin subspaces...", 4)
        else:
            N_top = min(N_top, self.S_alpha * (self.mol.nao - self.S_alpha) * self.S_beta * (self.mol.nao - self.S_beta))
            self.log.enter(f"Obtaining top {N_top} states with one excitation on both spin subspaces as measured by overlap in true ground state...", 4)

        if "full_CI_sol" not in self.checklist:
            self.log.write("WARNING: Full CI ground state solution has not been found yet. Aborting...")
            self.log.exit()
            return(None)

        res = [] #[i] = [n_sq, key, prom_label]; ordered by n_sq desc.
        if N_top is not None:
            for i in range(N_top):
                res.append([0, None, None])

        total_norm_squared_projection = 0.0

        for i in range(self.S_alpha):
            for j in range(self.mol.nao - self.S_alpha):
                for k in range(self.S_beta):
                    for l in range(self.mol.nao - self.S_beta):
                        # Note that we omit the trailing zeros to agree with the ci_sol convention
                        cur_state_alpha = (1,) * i + (0,) + (1,) * (self.S_alpha - 1 - i) + (0,) * j + (1,)
                        cur_state_beta = (1,) * k + (0,) + (1,) * (self.S_beta - 1 - k) + (0,) * l + (1,)
                        cur_c = self.ground_state_component(cur_state_alpha, cur_state_beta)
                        cur_n_sq = cur_c * cur_c

                        total_norm_squared_projection += cur_n_sq

                        if N_top is not None:
                            if cur_n_sq > res[-1][0]:
                                # belongs to the list
                                new_i = len(res)
                                while(cur_n_sq > res[new_i - 1][0]):
                                    new_i -= 1
                                    if new_i == 0:
                                        break
                                res.insert(new_i, [cur_n_sq, (cur_state_alpha, cur_state_beta), f"({i+1} -> {j + self.S_alpha + 1}), ({k+1} -> {l + self.S_beta + 1})"])
                                res.pop()
                        else:
                            # Not discriminating
                            res.append((cur_state_alpha, cur_state_beta))


        self.log.write(f"Obtained {len(res)} such states.", 5)
        self.log.exit()
        return(res, self.S_alpha * (self.mol.nao - self.S_alpha) * self.S_beta * (self.mol.nao - self.S_beta), total_norm_squared_projection)



    # -------------------------------------------------------------------------
    # ----------------- Obtaining the low-excitation solution -----------------
    # -------------------------------------------------------------------------
    # Note that an LE solution has two components:
    #   1. The actual ground state projection as a superposition onto occups
    #   2. The expectation value constraint for parameter matrices

    # The LE solution has the following standard structure:
    # self.LE_sol = {
    #     "E" : ground state energy (float),
    #     "sol" : solution as a dictionary [(excitation tuple a, excitation tuple b)] = coefficient
    # }
    # self.LE_description = {
    #     "scope" : list of tuples (a,b); if (a,b) is present, the solution contains states with a excitations on alpha and b excitations on beta,
    #     "params" : param dict,
    #     "label" : human readable method label
    # }


    # ------------------ Finding the projected ground-state -------------------
    # Note: SCF has methods for this (CIS, CISD)
    # Every method in this category:
    #   -Returns None
    #   -Initialises self.LE_sol
    #   -Calculates derived properties of LE_sol and stores them

    # ---------- RHF methods

    def find_LE_solution_SECS(self, **kwargs):
        # kwargs:
        #     -trim_M: integer of bottom spatial MOs to be considered.
        #              default value: all MOs considered

        # Takes a basis consisting of ref state and SECSs, finds effective H,
        # finds solution, and returns norm squared of solution component for
        # every single-excitation.
        # The goal is to approximate the true solution heatmap for the purpose
        # of guiding the CS sampling process.

        # Parameter regularisation
        act_M = self.mol.nao
        if "trim_M" in kwargs:
            if kwargs["trim_M"] > self.S_alpha: # We need at least one empty shell
                act_M = min(kwargs["trim_M"], self.mol.nao)

        self.LE_description["scope"] = [(0, 0), (1, 1)]
        self.LE_description["params"] = {"trim_M" : act_M}
        self.LE_description["label"] = "Closed-shell states differing from the HF state by one shell at most"


        self.log.enter("Solving on a single-excitation closed-shell basis...", 1)

        self.user_actions += f"find_LE_solution_SECS [MOs considered: {act_M}]\n"

        assert(self.S_alpha == self.S_beta)

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

        # Mark as solved
        self.check_off("LE_sol")

        self.LE_sol["E"] = ground_state_energy
        self.LE_sol["sol"] = {}
        for i in range(len(basis)):
            self.LE_sol[self.occ_list_to_prom_tuple(basis[i][0], basis[i][1])] = ground_state_vector[i]

        # Derived properties
        if "skip_properties" in kwargs:
            if kwargs["skip_properties"] == False:
                self.derive_LE_property_reduction_matrix()
        else:
            self.derive_LE_property_reduction_matrix()

        self.log.write(f"Success!", 1)
        self.log.exit()
        return(None)

    def find_LE_solution_SEO1(self, **kwargs):
        pass


    # ---------- UHF methods


    def find_LE_solution_SE(self, **kwargs):
        # Single excitation
        # kwargs:
        #     -diag_alg: algorithm to find the projected ground state. Options:
        #          "exp" : explicit diagonalisation [default]
        #          "SCF" : SCF performed on CIS

        # -------------------- Parameter initialisation

        if "diag_alg" in kwargs:
            diag_alg = kwargs["diag_alg"]
        else:
            diag_alg = "SCF"
        hr_diag_alg_label = {
            "exp" : "explicit diagonalisation",
            "SCF" : "SCF on a CISD basis"
            }[diag_alg]


        self.LE_description["scope"] = [(0, 0), (1, 0), (0, 1), (1, 1)]
        self.LE_description["params"] = {"diag_alg" : diag_alg}
        self.LE_description["label"] = "at most one excitation in each spin subspace"

        self.log.enter(f"Solving on a single-excitation closed-shell basis using {hr_diag_alg_label}...", 1)

        self.user_actions += f"find_LE_solution_SE [diag_alg = '{diag_alg}']\n"



        if diag_alg == "exp":

            self.log.write(f"Obtaining the SE basis....", 1)

            ref_state_a = (1,) * self.S_alpha
            ref_state_b = (1,) * self.S_beta
            basis = [(ref_state_a, ref_state_b)]

            for j in range(self.mol.nao - self.S_alpha):
                for i in range(self.S_alpha):
                    # Note that we omit the trailing zeros to agree with the ci_sol convention
                    basis.append(( (1,) * i + (0,) + (1,) * (self.S_alpha - 1 - i) + (0,) * j + (1,),  ref_state_b))

            for l in range(self.mol.nao - self.S_beta):
                for k in range(self.S_beta):
                    # Note that we omit the trailing zeros to agree with the ci_sol convention
                    basis.append((ref_state_a, (1,) * k + (0,) + (1,) * (self.S_beta - 1 - k) + (0,) * l + (1,)))

            basis += self.get_states_by_excitation_number(1, 1)

            for i in range(self.S_alpha):
                for j in range(i + 1, self.S_alpha):
                    for k in range(self.mol.nao - self.S_alpha):
                        for l in range(k + 1, self.mol.nao - self.S_alpha):
                            a_occ = [1] * self.S_alpha + [0] * (l + 1)
                            a_occ[i] = 0
                            a_occ[j] = 0
                            a_occ[k + self.S_alpha] = 1
                            a_occ[l + self.S_alpha] = 1
                            basis.append((tuple(a_occ), ref_state_b))
            for i in range(self.S_beta):
                for j in range(i + 1, self.S_beta):
                    for k in range(self.mol.nao - self.S_beta):
                        for l in range(k + 1, self.mol.nao - self.S_beta):
                            b_occ = [1] * self.S_beta + [0] * (l + 1)
                            b_occ[i] = 0
                            b_occ[j] = 0
                            b_occ[k + self.S_beta] = 1
                            b_occ[l + self.S_beta] = 1
                            basis.append((ref_state_a, tuple(b_occ)))

            self.log.write(f"SE basis of length {len(basis)} obtained.")

            H = np.zeros((len(basis), len(basis)))
            msg = f"Explicit Hamiltonian evaluation on SE basis"

            self.log.enter(msg, 1, True, tau_space = np.linspace(0, len(basis) * (len(basis) + 1) / 2, 100 + 1))

            for a in range(len(basis)):
                # Here the diagonal <Z_a|H|Z_a>
                cur_H_overlap = self.get_H_overlap_on_occupancy(self.occ_tuple_to_list(basis[a]), self.occ_tuple_to_list(basis[a]))
                assert cur_H_overlap.imag < 1e-08
                H[a][a] = cur_H_overlap.real
                self.log.update_semaphor_event(a * (a + 1) / 2)

                # Here the off-diagonal, using the fact that H is a Hermitian matrix
                for b in range(a):
                    # We explicitly calculate <Z_a | H | Z_b>
                    cur_H_overlap = self.get_H_overlap_on_occupancy(self.occ_tuple_to_list(basis[a]), self.occ_tuple_to_list(basis[b]))
                    assert cur_H_overlap.imag < 1e-08
                    H[a][b] = cur_H_overlap.real
                    H[b][a] = H[a][b] # np.conjugate(H[a][b])
                    self.log.update_semaphor_event(a * (a + 1) / 2 + b + 1)

            self.log.exit("Evaluation")

            energy_levels, energy_states = np.linalg.eig(H)
            ground_state_index = np.argmin(energy_levels)
            ground_state_energy = energy_levels[ground_state_index]
            ground_state_vector = energy_states[:,ground_state_index] # note that energy_states consist of column vectors, not row vectors

            if "full_CI_sol" in self.checklist:
                self.log.write(f"Ground state energy: {ground_state_energy:0.5f} (compare to full CI: {self.ci_energy:0.5f})", 1)
            else:
                self.log.write(f"Ground state energy: {ground_state_energy:0.5f}", 1)

            # Now, for the heatmap
            self.log.write(f"Obtaining the low-excitation prevalence matrix...", 2)

            res_sol = {(((), ()), ((), ())) : ground_state_vector[0]}

            basis_i = 1
            for a in range(self.mol.nao - self.S_alpha):
                for b in range(self.S_alpha):
                    res_sol[ (((b,), (a,)), ((), ())) ] = ground_state_vector[basis_i]
                    basis_i += 1
            for a in range(self.mol.nao - self.S_beta):
                for b in range(self.S_beta):
                    res_sol[ (((), ()), ((b,), (a,))) ] = ground_state_vector[basis_i]
                    basis_i += 1

            # order must match the generator
            for i in range(self.S_alpha):
                for j in range(self.mol.nao - self.S_alpha):
                    for k in range(self.S_beta):
                        for l in range(self.mol.nao - self.S_beta):
                            res_sol[ (((i,), (j,)), ((k,), (l,))) ] = ground_state_vector[basis_i]
                            basis_i += 1

            # alpha alpha
            for i in range(self.S_alpha):
                for j in range(i + 1, self.S_alpha):
                    for k in range(self.mol.nao - self.S_alpha):
                        for l in range(k + 1, self.mol.nao - self.S_alpha):
                            res_sol[ (((i, j), (k, l)), ((), ())) ] = ground_state_vector[basis_i]
                            basis_i += 1
            # beta beta
            for i in range(self.S_beta):
                for j in range(i + 1, self.S_beta):
                    for k in range(self.mol.nao - self.S_beta):
                        for l in range(k + 1, self.mol.nao - self.S_beta):
                            res_sol[ (((), ()), ((i, j), (k, l))) ] = ground_state_vector[basis_i]
                            basis_i += 1

            # Mark as solved
            self.check_off("LE_sol")

            self.LE_sol["E"] = ground_state_energy
            self.LE_sol["sol"] = res_sol

        elif diag_alg == "SCF":
            self.log.enter("SCF on CISD basis...", 3)

            cisd_solver = self.mean_field.CISD().run()

            if not cisd_solver.converged:
                self.log.write("ERROR: SCF solver failed to converge. Aborting...")
                self.log.exit()
                self.log.exit()
                return(None)

            self.log.write(f"Obtained CISD solution (basis length = {len(cisd_solver.ci)})")

            if "full_CI_sol" in self.checklist:
                self.log.write(f"Ground state energy: {cisd_solver.e_tot:0.5f} (compare to full CI: {self.ci_energy:0.5f})", 1)
            else:
                self.log.write(f"Ground state energy: {cisd_solver.e_tot:0.5f}", 1)


            if self.HF_method == "UHF":

                # We characterise the coef array by excitations
                c0, c1, c2 = cisd_solver.cisdvec_to_amplitudes(cisd_solver.ci)

                # The shapes of these look different based on the HF method, with extra symmetry assumed for RHF

                c1_a, c1_b = c1
                c2_aa, c2_ab, c2_bb = c2

                self.log.write("Excitation-based Slater determinant sub-bases have lengths:")
                self.log.write(f"  For zero excitation: one state only (overlap {c0:0.5f})")
                self.log.write(f"  For one excitation: {c1_a.shape} on alpha, {c1_b.shape} on beta")
                self.log.write(f"  For two excitations: {c2_aa.shape} on alpha-alpha, {c2_bb.shape} on beta-beta, {c2_ab.shape} on mixed-spin excitations")

                # Now, for the heatmap
                self.log.write(f"Obtaining the low-excitation prevalence matrix...", 2)


                # We do not need to project onto the (no same-spin double excitation) basis,
                # since the fraction cn / c0 remains the same

                res_sol = {(((), ()), ((), ())) : c0}

                for a in range(self.mol.nao - self.S_alpha):
                    for b in range(self.S_alpha):
                        res_sol[ (((b,), (a,)), ((), ())) ] = c1_a[b][a]
                for a in range(self.mol.nao - self.S_beta):
                    for b in range(self.S_beta):
                        res_sol[ (((), ()), ((b,), (a,))) ] = c1_b[b][a]

                for i in range(self.mol.nao - self.S_alpha):
                    for j in range(self.S_alpha):
                        for k in range(self.mol.nao - self.S_beta):
                            for l in range(self.S_beta):
                                res_sol[ (((j,), (i,)), ((l,), (k,))) ] = c2_ab[j][l][i][k]

            elif self.HF_method == "RHF":
                fci_coefs = ci.cisd.to_fcivec(cisd_solver.ci, self.mol.nao, (self.S_alpha, self.S_beta))
                #fci_coefs = ci.cisd.to_fcivec(cisd_solver.ci, cisd_solver.norb, cisd_solver.nelec)
                #fci_coefs = cc.cc2ci.fci_coefs(cisd_solver)
                # fci_coefs is the same kind of object as the output of a full FCI calculation

                res_sol = {}

                self.log.write("Regularising solution as a dict of tuples...", 3)
                # We omit entries which are not singlet or doublet excitations, since they are by definition zero in the CISD sol
                HF_occ = "1" * self.S_alpha
                res_sol[(((), ()), ((), ()))] = float(fci_coefs[fci.cistring.str2addr(self.mol.nao, self.S_alpha, HF_occ), fci.cistring.str2addr(self.mol.nao, self.S_beta, HF_occ)])

                # singlets
                for i in range(self.S_alpha):
                    for j in range(self.mol.nao - self.S_alpha):
                        promoted_occ = self.occ_list_to_occ_string([1] * i + [0] + [1] * (self.S_alpha - 1 - i) + [0] * j + [1])
                        res_sol[(((i,), (j,)), ((), ()))] = float(fci_coefs[fci.cistring.str2addr(self.mol.nao, self.S_alpha, promoted_occ), fci.cistring.str2addr(self.mol.nao, self.S_beta, HF_occ)])
                        res_sol[(((), ()), ((i,), (j,)))] = float(fci_coefs[fci.cistring.str2addr(self.mol.nao, self.S_alpha, HF_occ), fci.cistring.str2addr(self.mol.nao, self.S_beta, promoted_occ)])

                # doublets
                for i in range(self.S_alpha):
                    for j in range(self.mol.nao - self.S_alpha):
                        for k in range(self.S_beta):
                            for l in range(self.mol.nao - self.S_beta):
                                alpha_occ = self.occ_list_to_occ_string([1] * i + [0] + [1] * (self.S_alpha - 1 - i) + [0] * j + [1])
                                beta_occ = self.occ_list_to_occ_string([1] * k + [0] + [1] * (self.S_alpha - 1 - k) + [0] * l + [1])
                                res_sol[(((i,), (j,)), ((k,), (l,)))] = float(fci_coefs[fci.cistring.str2addr(self.mol.nao, self.S_alpha, alpha_occ), fci.cistring.str2addr(self.mol.nao, self.S_beta, beta_occ)])

                """for a in range(fci_coefs.shape[0]):
                    for b in range(fci_coefs.shape[1]):
                        alpha_occ = self.occ_str_to_occ_tuple("{0:b}".format(fci.cistring.addr2str(self.mol.nao, self.S_alpha, a)))
                        beta_occ = self.occ_str_to_occ_tuple("{0:b}".format(fci.cistring.addr2str(self.mol.nao, self.S_beta, b)))

                        # We omit entries which are not singlet or doublet excitations, since they are by definition zero in the CISD sol
                        #if

                        key = self.occ_list_to_prom_tuple(alpha_occ, beta_occ)# (self.get_prom_label(alpha_occ), self.get_prom_label(beta_occ))
                        res_sol[key] = float(fci_coefs[a, b])"""


                """t1addrs, t1signs = ci.cisd.tn_addrs_signs(self.mol.nao, self.S_alpha, 1)
                t2addrs, t2signs = ci.cisd.tn_addrs_signs(self.mol.nao, self.S_alpha, 2)

                # singlets

                cis_a = fci_coefs[t1addrs, 0] * t1signs
                cis_b = fci_coefs[0, t1addrs] * t1signs

                # doublets

                cid_aa = fci_coefs[t2addrs, 0] * t2signs
                cid_bb = fci_coefs[0, t2addrs] * t2signs
                cid_ab = np.einsum('ij,i,j->ij', fci_coefs[t1addrs[:,None], t1addrs], t1signs, t1signs)

                res_sol = {(((), ()), ((), ())) : fci_coefs[0, 0]}

                idx = 0
                for b in range(self.S_alpha):
                    for a in range(self.mol.nao - self.S_alpha):
                        # | b -> a >
                        #promoted_occ = self.occ_list_to_occ_string([1] * b + [0] + [1] * (self.S_alpha - 1 - b) + [0] * a + [1])
                        res_sol[ (((b,), (a,)), ((), ())) ] = cis_a[idx]
                        res_sol[ (((), ()), ((b,), (a,))) ] = cis_b[idx]
                        idx += 1

                idx_a = 0

                for i in range(self.S_alpha):
                    for j in range(self.mol.nao - self.S_alpha):
                        idx_b = 0
                        for k in range(self.S_beta):
                            for l in range(self.mol.nao - self.S_beta):
                                # | i - > j, k -> l >
                                #alpha_occ = self.occ_list_to_occ_string([1] * i + [0] + [1] * (self.S_alpha - 1 - i) + [0] * j + [1])
                                #beta_occ = self.occ_list_to_occ_string([1] * k + [0] + [1] * (self.S_alpha - 1 - k) + [0] * l + [1])
                                res_sol[ (((i,), (j,)), ((k,), (l,))) ] = cid_ab[idx_a, idx_b]
                                idx_b += 1
                        idx_a += 1"""


                """self.log.write("Excitation-based Slater determinant sub-bases have lengths:")
                self.log.write(f"  For zero excitation: one state only (overlap {c0:0.5f})")
                self.log.write(f"  For one excitation: {c1.shape} spin-adapted configurations of the form | i -> j > = 1/sqrt(2) . (a_i,alpha\\hc a_j,alpha + a_i,beta\\hc a_j,beta) | HF >")
                self.log.write(f"  For two excitations: {c2.shape} singlet-coupled combinations of the form | ij -> kl > 1/2 . (a_k,alpha\\hc a_l,beta\\hc - a_l,alpha\\hc a_k,beta\\hc) . (a_i,alpha a_j,beta - a_j,alpha a_i,beta) | HF >")

                # Now, for the heatmap
                self.log.write(f"Obtaining the low-excitation prevalence matrix...", 2)


                # We do not need to project onto the (no same-spin double excitation) basis,
                # since the fraction cn / c0 remains the same

                res_sol = {(((), ()), ((), ())) : c0}

                for a in range(self.mol.nao - self.S_alpha):
                    for b in range(self.S_alpha):
                        # | b -> a >
                        amp = c1[b][a] / np.sqrt(2)
                        res_sol[ (((b,), (a,)), ((), ())) ] = amp
                        res_sol[ (((), ()), ((b,), (a,))) ] = amp

                for i in range(self.S_alpha):
                    for j in range(self.S_beta):
                        for k in range(self.mol.nao - self.S_alpha):
                            for l in range(self.mol.nao - self.S_beta):
                                # | ij -> kl >
                                res_sol[ (((i,), (k,)), ((j,), (l,))) ] = c2[i][j][k][l]"""

            # Mark as solved
            self.check_off("LE_sol")

            self.LE_sol["E"] = cisd_solver.e_tot
            self.LE_sol["sol"] = res_sol

            self.log.exit()

        # Derived properties
        if "skip_properties" in kwargs:
            if kwargs["skip_properties"] == False:
                self.derive_LE_property_reduction_matrix()
        else:
            self.derive_LE_property_reduction_matrix()

        self.log.write(f"Success!", 1)
        self.log.exit()
        return(None)


    def find_LE_solution_MSDE(self, **kwargs):
        # Double excitation with mixed spins


        self.LE_description["scope"] = [(0, 0), (1, 0), (0, 1), (1, 1)]
        self.LE_description["params"] = {}
        self.LE_description["label"] = "at most one excitation in each spin subspace"

        self.log.enter(f"Solving on a mixed-spin double-excitation basis using SCF...", 1)

        self.user_actions += f"find_LE_solution_MSDE [def_var = {def_var}]\n"

        # Finding the LE solution
        self.log.enter("SCF on CISD basis...", 3)
        cisd_solver = self.mean_field.CISD().run()
        if not cisd_solver.converged:
            self.log.write("ERROR: SCF solver failed to converge. Aborting...")
            self.log.exit()
            self.log.exit()
            return(None)
        self.log.write(f"Obtained CISD solution (basis length = {len(cisd_solver.ci)})")

        if "full_CI_sol" in self.checklist:
            self.log.write(f"Ground state energy: {cisd_solver.e_tot:0.5f} (compare to full CI: {self.ci_energy:0.5f})", 1)
        else:
            self.log.write(f"Ground state energy: {cisd_solver.e_tot:0.5f}", 1)

        # We characterise the coef array by excitations
        c0, c1, c2 = cisd_solver.cisdvec_to_amplitudes(cisd_solver.ci)

        # The shapes of these look different based on the HF method, with extra symmetry assumed for RHF

        if self.HF_method == "UHF":
            c1_a, c1_b = c1
            c2_aa, c2_ab, c2_bb = c2

            self.log.write("Excitation-based Slater determinant sub-bases have lengths:")
            self.log.write(f"  For zero excitation: one state only (overlap {c0:0.5f})")
            self.log.write(f"  For one excitation: {c1_a.shape} on alpha, {c1_b.shape} on beta")
            self.log.write(f"  For two excitations: {c2_aa.shape} on alpha-alpha, {c2_bb.shape} on beta-beta, {c2_ab.shape} on mixed-spin excitations")

            # Now, for the heatmap
            self.log.write(f"Obtaining the low-excitation prevalence matrix...", 2)


            # We do not need to project onto the (no same-spin double excitation) basis,
            # since the fraction cn / c0 remains the same

            res_sol = {(((), ()), ((), ())) : c0}

            for a in range(self.mol.nao - self.S_alpha):
                for b in range(self.S_alpha):
                    res_sol[ (((b,), (a,)), ((), ())) ] = c1_a[b][a]
            for a in range(self.mol.nao - self.S_beta):
                for b in range(self.S_beta):
                    res_sol[ (((), ()), ((b,), (a,))) ] = c1_b[b][a]

            for i in range(self.mol.nao - self.S_alpha):
                for j in range(self.S_alpha):
                    for k in range(self.mol.nao - self.S_beta):
                        for l in range(self.S_beta):
                            res_sol[ (((j,), (i,)), ((l,), (k,))) ] = c2_ab[j][l][i][k]

        elif self.HF_method == "RHF":
            self.log.write("Excitation-based Slater determinant sub-bases have lengths:")
            self.log.write(f"  For zero excitation: one state only (overlap {c0:0.5f})")
            self.log.write(f"  For one excitation: {c1.shape} spin-adapted configurations of the form | i -> j > = 1/sqrt(2) . (a_i,alpha\\hc a_j,alpha + a_i,beta\\hc a_j,beta) | HF >")
            self.log.write(f"  For two excitations: {c2.shape} singlet-coupled combinations of the form | ij -> kl > 1/2 . (a_k,alpha\\hc a_l,beta\\hc - a_l,alpha\\hc a_k,beta\\hc) . (a_i,alpha a_j,beta - a_j,alpha a_i,beta) | HF >")

            # Now, for the heatmap
            self.log.write(f"Obtaining the low-excitation prevalence matrix...", 2)


            # We do not need to project onto the (no same-spin double excitation) basis,
            # since the fraction cn / c0 remains the same

            res_sol = {(((), ()), ((), ())) : c0}

            for a in range(self.mol.nao - self.S_alpha):
                for b in range(self.S_alpha):
                    # | b -> a >
                    amp = c1[b][a] / np.sqrt(2)
                    res_sol[ (((b,), (a,)), ((), ())) ] = amp
                    res_sol[ (((), ()), ((b,), (a,))) ] = amp

            for i in range(self.S_alpha):
                for j in range(self.S_beta):
                    for k in range(self.mol.nao - self.S_alpha):
                        for l in range(self.mol.nao - self.S_beta):
                            # | ij -> kl >
                            res_sol[ (((i,), (k,)), ((j,), (l,))) ] = c2[i][j][k][l]

        # Derived properties
        if "skip_properties" in kwargs:
            if kwargs["skip_properties"] == False:
                self.derive_LE_property_reduction_matrix()
        else:
            self.derive_LE_property_reduction_matrix()

        # Mark as solved
        self.check_off("LE_sol")

        self.LE_sol["E"] = cisd_solver.e_tot
        self.LE_sol["sol"] = res_sol

        self.log.exit()


    # Derived properties calculation

    def derive_LE_property_reduction_matrix(self):
        # Calculates < LE | b_i\hc b_j | LE >, where b is the Qubit (bosonic) annihilation operator and i,j span all spin-orbitals

        assert "sol" in self.LE_sol

        self.log.enter("Deriving the reduction matrix for the LE solution...")

        self.log.write("Calculating the norm of the solution on singlets and mixed-spin doublets...")

        LE_norm = 0.0
        LE_reduced_norm = 0.0 # sans phi_0
        LE_CS_norm = 0.0

        # no excitation
        LE_norm += self.LE_sol["sol"][(((), ()), ((), ()))] * self.LE_sol["sol"][(((), ()), ((), ()))]
        LE_CS_norm += self.LE_sol["sol"][(((), ()), ((), ()))] * self.LE_sol["sol"][(((), ()), ((), ()))]

        # singlets
        if (1, 0) in self.LE_description["scope"]:
            for a in range(self.mol.nao - self.S_alpha):
                for b in range(self.S_alpha):
                    cur_z = self.LE_sol["sol"][ (((b,), (a,)), ((), ())) ]
                    LE_norm += cur_z * cur_z
                    LE_reduced_norm += cur_z * cur_z
        if (0, 1) in self.LE_description["scope"]:
            for a in range(self.mol.nao - self.S_beta):
                for b in range(self.S_beta):
                    cur_z = self.LE_sol["sol"][ (((), ()), ((b,), (a,))) ]
                    LE_norm += cur_z * cur_z
                    LE_reduced_norm += cur_z * cur_z

        if (1, 1) in self.LE_description["scope"]:
            # doublets
            for i in range(self.mol.nao - self.S_alpha):
                for j in range(self.S_alpha):
                    for k in range(self.mol.nao - self.S_beta):
                        for l in range(self.S_beta):
                            cur_z = self.LE_sol["sol"][ (((j,), (i,)), ((l,), (k,))) ]
                            LE_norm += cur_z * cur_z
                            LE_reduced_norm += cur_z * cur_z
                            if i == k and j == l:
                                LE_CS_norm += cur_z * cur_z

        self.log.write(f"Low excitation solution norm squared:")
        self.log.write(f"  -for all singlet and mixed-doublet excitations: <LE | LE> = {LE_norm:0.5f}")
        self.log.write(f"  -for all singlet and mixed-doublet excitations, ignoring the HF state: <LE | LE> = {LE_reduced_norm:0.5f}")
        self.log.write(f"  -for closed-shell mixed-doublet excitations: <LE | LE> = {LE_CS_norm:0.5f}")


        spin_idx_dict = {"a" : 0, "b" : 1}
        spat_to_spin_idx = lambda sigma, i : spin_idx_dict[sigma] * self.mol.nao + i

        """
        self.log.enter("Calculating reduction matrix", 1, True, tau_space = np.linspace(0, 4 * self.mol.nao * self.mol.nao, 1000 + 1))

        self.LE_sol["red"] = np.zeros((2 * self.mol.nao, 2 * self.mol.nao))


        for left_sigma in ["a", "b"]:
            for left_i in range(self.mol.nao):
                for right_sigma in ["a", "b"]:
                    for right_i in range(self.mol.nao):
                        self.log.update_semaphor_event(2 * self.mol.nao * spat_to_spin_idx(left_sigma, left_i) + spat_to_spin_idx(right_sigma, right_i))

                        for left_prom, left_z in self.LE_sol["sol"].items():
                            left_prom_a, left_prom_b = left_prom
                            left_a_from, left_a_to = left_prom_a
                            left_b_from, left_b_to = left_prom_b
                            left_a_occ = [1] * self.S_alpha + [0] * (self.mol.nao - self.S_alpha)
                            left_b_occ = [1] * self.S_beta + [0] * (self.mol.nao - self.S_beta)
                            for left_a_from_i in left_a_from:
                                left_a_occ[left_a_from_i] = 0
                            for left_a_to_i in left_a_to:
                                left_a_occ[self.S_alpha + left_a_to_i] = 1
                            for left_b_from_i in left_b_from:
                                left_b_occ[left_b_from_i] = 0
                            for left_b_to_i in left_b_to:
                                left_b_occ[self.S_beta + left_b_to_i] = 1

                            if left_sigma == "a":
                                if left_a_occ[left_i] == 0:
                                    # destroys
                                    continue
                                else:
                                    left_a_occ[left_i] = 0
                            if left_sigma == "b":
                                if left_b_occ[left_i] == 0:
                                    # destroys
                                    continue
                                else:
                                    left_b_occ[left_i] = 0

                            for right_prom, right_z in self.LE_sol["sol"].items():
                                right_prom_a, right_prom_b = right_prom
                                right_a_from, right_a_to = right_prom_a
                                right_b_from, right_b_to = right_prom_b
                                right_a_occ = [1] * self.S_alpha + [0] * (self.mol.nao - self.S_alpha)
                                right_b_occ = [1] * self.S_beta + [0] * (self.mol.nao - self.S_beta)
                                for right_a_from_i in right_a_from:
                                    right_a_occ[right_a_from_i] = 0
                                for right_a_to_i in right_a_to:
                                    right_a_occ[self.S_alpha + right_a_to_i] = 1
                                for right_b_from_i in right_b_from:
                                    right_b_occ[right_b_from_i] = 0
                                for right_b_to_i in right_b_to:
                                    right_b_occ[self.S_beta + right_b_to_i] = 1


                                #print(f"right prom: {right_prom}; rihgt a occ: {right_a_occ}; right b occ: {right_b_occ}")

                                if right_sigma == "a":
                                    if right_a_occ[right_i] == 0:
                                        # destroys
                                        continue
                                    else:
                                        right_a_occ[right_i] = 0
                                if right_sigma == "b":
                                    if right_b_occ[right_i] == 0:
                                        # destroys
                                        continue
                                    else:
                                        right_b_occ[right_i] = 0

                                if np.all(left_a_occ == right_a_occ) and np.all(left_b_occ == right_b_occ):
                                    self.LE_sol["red"][spat_to_spin_idx(left_sigma, left_i)][spat_to_spin_idx(right_sigma, right_i)] += left_z * right_z / LE_norm

        self.log.exit("Calculation")

        self.log.enter("Calculating closed-shell reduction matrix", 1, True, tau_space = np.linspace(0, self.mol.nao * self.mol.nao, 1000 + 1))

        self.LE_sol["CSRM"] = np.zeros((self.mol.nao, self.mol.nao))
        # here, [i][j] corresponds to a simultaneous transition j -> i on both spin subspaces

        for left_i in range(self.mol.nao):
            for right_i in range(self.mol.nao):
                self.log.update_semaphor_event(self.mol.nao * left_i + right_i)

                for left_prom, left_z in self.LE_sol["sol"].items():
                    left_prom_a, left_prom_b = left_prom
                    left_a_from, left_a_to = left_prom_a
                    left_b_from, left_b_to = left_prom_b
                    left_a_occ = [1] * self.S_alpha + [0] * (self.mol.nao - self.S_alpha)
                    left_b_occ = [1] * self.S_beta + [0] * (self.mol.nao - self.S_beta)
                    for left_a_from_i in left_a_from:
                        left_a_occ[left_a_from_i] = 0
                    for left_a_to_i in left_a_to:
                        left_a_occ[self.S_alpha + left_a_to_i] = 1
                    for left_b_from_i in left_b_from:
                        left_b_occ[left_b_from_i] = 0
                    for left_b_to_i in left_b_to:
                        left_b_occ[self.S_beta + left_b_to_i] = 1

                    if left_a_occ[left_i] == 0:
                        # destroys
                        continue
                    else:
                        left_a_occ[left_i] = 0
                    if left_b_occ[left_i] == 0:
                        # destroys
                        continue
                    else:
                        left_b_occ[left_i] = 0

                    for right_prom, right_z in self.LE_sol["sol"].items():
                        right_prom_a, right_prom_b = right_prom
                        right_a_from, right_a_to = right_prom_a
                        right_b_from, right_b_to = right_prom_b
                        right_a_occ = [1] * self.S_alpha + [0] * (self.mol.nao - self.S_alpha)
                        right_b_occ = [1] * self.S_beta + [0] * (self.mol.nao - self.S_beta)
                        for right_a_from_i in right_a_from:
                            right_a_occ[right_a_from_i] = 0
                        for right_a_to_i in right_a_to:
                            right_a_occ[self.S_alpha + right_a_to_i] = 1
                        for right_b_from_i in right_b_from:
                            right_b_occ[right_b_from_i] = 0
                        for right_b_to_i in right_b_to:
                            right_b_occ[self.S_beta + right_b_to_i] = 1


                        #print(f"right prom: {right_prom}; rihgt a occ: {right_a_occ}; right b occ: {right_b_occ}")

                        if right_a_occ[right_i] == 0:
                            # destroys
                            continue
                        else:
                            right_a_occ[right_i] = 0
                        if right_b_occ[right_i] == 0:
                            # destroys
                            continue
                        else:
                            right_b_occ[right_i] = 0

                        if np.all(left_a_occ == right_a_occ) and np.all(left_b_occ == right_b_occ):
                            self.LE_sol["CSRM"][left_i][right_i] += left_z * right_z / LE_CS_norm

        self.log.exit("Calculation")

        self.log.enter("Calculating transition prevalence matrix", 1, True, tau_space = np.linspace(0, (self.mol.nao - self.S_alpha) * self.S_alpha + (self.mol.nao - self.S_beta) * self.S_beta, 1000 + 1))


        self.LE_sol["TPM"] = np.zeros((2 * self.mol.nao, self.mol.nao)) # [spin * M + i, j]

        # spin a
        for prom_i in range(self.S_alpha):
            for prom_j in range(self.mol.nao - self.S_alpha):
                self.log.update_semaphor_event((self.mol.nao - self.S_alpha) * prom_i + prom_j)
                # We need to select all coefs from sol which are characterised by i -> j on spin sigma
                for state_prom, state_z in self.LE_sol["sol"].items():
                    state_prom_a, state_prom_b = state_prom
                    if state_prom_a == ((prom_i,), (prom_j,)):
                        self.LE_sol["TPM"][spat_to_spin_idx("a", prom_i), prom_j + self.S_alpha] += state_z * state_z
                        self.LE_sol["TPM"][spat_to_spin_idx("a", prom_j + self.S_alpha), prom_i] += state_z * state_z
        # spin b
        for prom_i in range(self.S_beta):
            for prom_j in range(self.mol.nao - self.S_beta):
                self.log.update_semaphor_event((self.mol.nao - self.S_alpha) * self.S_alpha + (self.mol.nao - self.S_beta) * prom_i + prom_j)
                # We need to select all coefs from sol which are characterised by i -> j on spin sigma
                for state_prom, state_z in self.LE_sol["sol"].items():
                    state_prom_a, state_prom_b = state_prom
                    if state_prom_a == ((prom_i,), (prom_j,)):
                        self.LE_sol["TPM"][spat_to_spin_idx("b", prom_i), prom_j + self.S_beta] += state_z * state_z
                        self.LE_sol["TPM"][spat_to_spin_idx("b", prom_j + self.S_beta), prom_i] += state_z * state_z



        self.log.exit("Calculation")


        self.log.enter("Calculating spin-reduced reduction matrix", 1, True, tau_space = np.linspace(0, 2 * self.mol.nao * self.mol.nao, 1000 + 1))

        self.LE_sol["SRRM"] = np.zeros((2 * self.mol.nao, self.mol.nao))


        for prom_sigma in ["a", "b"]:
            for prom_i in range(self.mol.nao):
                for prom_j in range(self.mol.nao):
                    self.log.update_semaphor_event(self.mol.nao * spat_to_spin_idx(prom_sigma, prom_i) + prom_j)

                    for left_prom, left_z in self.LE_sol["sol"].items():
                        left_prom_a, left_prom_b = left_prom
                        left_a_from, left_a_to = left_prom_a
                        left_b_from, left_b_to = left_prom_b
                        left_a_occ = [1] * self.S_alpha + [0] * (self.mol.nao - self.S_alpha)
                        left_b_occ = [1] * self.S_beta + [0] * (self.mol.nao - self.S_beta)
                        for left_a_from_i in left_a_from:
                            left_a_occ[left_a_from_i] = 0
                        for left_a_to_i in left_a_to:
                            left_a_occ[self.S_alpha + left_a_to_i] = 1
                        for left_b_from_i in left_b_from:
                            left_b_occ[left_b_from_i] = 0
                        for left_b_to_i in left_b_to:
                            left_b_occ[self.S_beta + left_b_to_i] = 1

                        if prom_sigma == "a":
                            if left_a_occ[prom_i] == 0:
                                # destroys
                                continue
                            else:
                                left_a_occ[prom_i] = 0
                        if prom_sigma == "b":
                            if left_b_occ[prom_i] == 0:
                                # destroys
                                continue
                            else:
                                left_b_occ[prom_i] = 0

                        for right_prom, right_z in self.LE_sol["sol"].items():
                            right_prom_a, right_prom_b = right_prom
                            right_a_from, right_a_to = right_prom_a
                            right_b_from, right_b_to = right_prom_b
                            right_a_occ = [1] * self.S_alpha + [0] * (self.mol.nao - self.S_alpha)
                            right_b_occ = [1] * self.S_beta + [0] * (self.mol.nao - self.S_beta)
                            for right_a_from_i in right_a_from:
                                right_a_occ[right_a_from_i] = 0
                            for right_a_to_i in right_a_to:
                                right_a_occ[self.S_alpha + right_a_to_i] = 1
                            for right_b_from_i in right_b_from:
                                right_b_occ[right_b_from_i] = 0
                            for right_b_to_i in right_b_to:
                                right_b_occ[self.S_beta + right_b_to_i] = 1


                            #print(f"right prom: {right_prom}; rihgt a occ: {right_a_occ}; right b occ: {right_b_occ}")

                            if prom_sigma == "a":
                                if right_a_occ[prom_j] == 0:
                                    # destroys
                                    continue
                                else:
                                    right_a_occ[prom_j] = 0
                            if prom_sigma == "b":
                                if right_b_occ[prom_j] == 0:
                                    # destroys
                                    continue
                                else:
                                    right_b_occ[prom_j] = 0

                            # We only consider the overlap on the spin subspace on which the transition occurs
                            if prom_sigma == "a":
                                if np.all(left_a_occ == right_a_occ):
                                    self.LE_sol["SRRM"][spat_to_spin_idx(prom_sigma, prom_i)][prom_j] += np.abs(left_z * right_z / LE_norm)
                            if prom_sigma == "b":
                                if np.all(left_b_occ == right_b_occ):
                                    self.LE_sol["SRRM"][spat_to_spin_idx(prom_sigma, prom_i)][prom_j] += np.abs(left_z * right_z / LE_norm)

        self.log.exit("Calculation")


        self.log.enter("Calculating simultanous occupancy proportion matrix", 1, True, tau_space = np.linspace(0, 4 * self.mol.nao * self.mol.nao, 1000 + 1))

        self.LE_sol["SOPM"] = np.zeros((2 * self.mol.nao, 2 * self.mol.nao))

        for i_sigma in ["a", "b"]:
            for i in range(self.mol.nao):
                for j_sigma in ["a", "b"]:
                    for j in range(self.mol.nao):
                        self.log.update_semaphor_event(2 * self.mol.nao * spat_to_spin_idx(i_sigma, i) + spat_to_spin_idx(j_sigma, j))
                        for cur_prom, cur_z in self.LE_sol["sol"].items():
                            cur_prom_a, cur_prom_b = cur_prom
                            cur_a_from, cur_a_to = cur_prom_a
                            cur_b_from, cur_b_to = cur_prom_b
                            cur_a_occ = [1] * self.S_alpha + [0] * (self.mol.nao - self.S_alpha)
                            cur_b_occ = [1] * self.S_beta + [0] * (self.mol.nao - self.S_beta)
                            for cur_a_from_i in cur_a_from:
                                cur_a_occ[cur_a_from_i] = 0
                            for cur_a_to_i in cur_a_to:
                                cur_a_occ[self.S_alpha + cur_a_to_i] = 1
                            for cur_b_from_i in cur_b_from:
                                cur_b_occ[cur_b_from_i] = 0
                            for cur_b_to_i in cur_b_to:
                                cur_b_occ[self.S_beta + cur_b_to_i] = 1

                            if i_sigma == "a" and cur_a_occ[i] == 0:
                                continue
                            if i_sigma == "b" and cur_b_occ[i] == 0:
                                continue
                            if j_sigma == "a" and cur_a_occ[j] == 0:
                                continue
                            if j_sigma == "b" and cur_b_occ[j] == 0:
                                continue

                            self.LE_sol["SOPM"][spat_to_spin_idx(i_sigma, i)][spat_to_spin_idx(j_sigma, j)] += cur_z * cur_z / LE_norm



        self.log.exit("Calculation")"""

        self.log.enter("Calculating reduced simultanous occupancy proportion matrix", 1, True, tau_space = np.linspace(0, 4 * self.mol.nao * self.mol.nao, 1000 + 1))

        self.LE_sol["RSOPM"] = np.zeros((2 * self.mol.nao, 2 * self.mol.nao))

        for i_sigma in ["a", "b"]:
            for i in range(self.mol.nao):
                for j_sigma in ["a", "b"]:
                    for j in range(self.mol.nao):
                        self.log.update_semaphor_event(2 * self.mol.nao * spat_to_spin_idx(i_sigma, i) + spat_to_spin_idx(j_sigma, j))
                        for cur_prom, cur_z in self.LE_sol["sol"].items():
                            if cur_prom == (((), ()), ((), ())):
                                continue
                            cur_prom_a, cur_prom_b = cur_prom
                            cur_a_from, cur_a_to = cur_prom_a
                            cur_b_from, cur_b_to = cur_prom_b
                            cur_a_occ = [1] * self.S_alpha + [0] * (self.mol.nao - self.S_alpha)
                            cur_b_occ = [1] * self.S_beta + [0] * (self.mol.nao - self.S_beta)
                            for cur_a_from_i in cur_a_from:
                                cur_a_occ[cur_a_from_i] = 0
                            for cur_a_to_i in cur_a_to:
                                cur_a_occ[self.S_alpha + cur_a_to_i] = 1
                            for cur_b_from_i in cur_b_from:
                                cur_b_occ[cur_b_from_i] = 0
                            for cur_b_to_i in cur_b_to:
                                cur_b_occ[self.S_beta + cur_b_to_i] = 1

                            if i_sigma == "a" and cur_a_occ[i] == 0:
                                continue
                            if i_sigma == "b" and cur_b_occ[i] == 0:
                                continue
                            if j_sigma == "a" and cur_a_occ[j] == 0:
                                continue
                            if j_sigma == "b" and cur_b_occ[j] == 0:
                                continue

                            self.LE_sol["RSOPM"][spat_to_spin_idx(i_sigma, i)][spat_to_spin_idx(j_sigma, j)] += cur_z * cur_z / LE_reduced_norm



        self.log.exit("Calculation")


        # Now we calculate the reduced null-coherent state (RNCS)
        # The RNCS is simply a Qubit CS for which every expected occupancy
        # matches the LE solution after projecting out |HF>

        self.log.enter("Calculating the reduced null-coherent state...")

        # Firstly, we renormalise RSOPM to satisfy the constraint
        RSOPM_tr = np.trace(self.LE_sol["RSOPM"])
        self.log.write(f"Trace of RSOPM = {RSOPM_tr}; expected value is S = {self.S_alpha + self.S_beta}; renormalising...")
        RSOPM_renorm = self.LE_sol["RSOPM"] * (self.S_alpha + self.S_beta) / RSOPM_tr

        def cur_scale(c_y):
            return(esp(np.exp(c_y), self.S_alpha + self.S_beta))

        # Now, we construct the initial values
        y_0 = np.zeros(2 * self.mol.nao)
        for i in range(2 * self.mol.nao):
            y_0[i] = np.log(RSOPM_renorm[i][i])
        y_0_norm_sq = cur_scale(y_0)
        self.log.write(f"Norm squared of initial guess is {y_0_norm_sq}. Renormalising...")
        """"
        z^2 = e^y
        {z|z} = e_S(z^2) = e_S(e^y)
        Let y = y + c
        then e^y = e^c.e^y
        then {z|z} = {z|z} . e^Sc
        We want e^Sc = 1 / cur scale
        hence c = -ln(cur_scale) / S
        """

        y_0 -= np.log(y_0_norm_sq) / (self.S_alpha + self.S_beta)
        y_0_norm_sq = cur_scale(y_0)
        self.log.write(f"Norm squared of initial guess was renormalised to {y_0_norm_sq}.")

        # --------------- Gradient descent to find the solution -----------------
        # Parameters
        eta = 0.1
        max_err = 1e-6
        self.log.write("Parameters for the gradient descent:")
        self.log.write(f"  -eta (step size) = {eta}")
        self.log.write(f"  -epsilon (max allowed error) = {max_err}")

        cur_y = np.array(y_0)

        # Now, we converge the solution
        #cur_err = 10 * max_err # a bogus val to enter the while loop
        # Cannot track max step because that converges to counteract the renorm step!
        #self.log.enter("Calculating z_i to match reduced mode occupancies", 1, True, tau_space = np.linspace(0, 1, 1000 + 1))

        # We calculate initial error
        y_step = np.zeros(2 * self.mol.nao)
        cur_norm_sq = cur_scale(cur_y)
        for i in range(2 * self.mol.nao):
            y_step[i] = RSOPM_renorm[i][i] - np.exp(cur_y[i]) * esp(np.exp(cur_y), self.S_alpha + self.S_beta - 1, omit = [i]) / cur_norm_sq

        # We calculate the err size
        init_err = np.max(np.abs(y_step))
        cur_err = init_err

        self.log.enter("Calculating z_i to match reduced mode occupancies", 1, True, tau_space = np.linspace(0, np.log(np.sqrt(init_err / max_err)), 1000 + 1))


        init_err = None
        while(cur_err > max_err):
            # We calculate the step and execute it
            cur_norm_sq = cur_scale(cur_y)
            for i in range(2 * self.mol.nao):
                y_step[i] = RSOPM_renorm[i][i] - np.exp(cur_y[i]) * esp(np.exp(cur_y), self.S_alpha + self.S_beta - 1, omit = [i]) / cur_norm_sq
            cur_y += eta * y_step

            # We calculate the err size
            cur_err = np.max(np.abs(y_step))#np.sqrt(np.max(y_step ** 2))
            if init_err is None:
                init_err = cur_err
            else:
                #self.log.update_semaphor_event((init_err - cur_err) / (init_err - max_err))
                #self.log.update_semaphor_event(np.exp(-(init_err - cur_err) / (init_err - max_err)))
                self.log.update_semaphor_event(np.log(np.sqrt(init_err / cur_err)))

            # We project y onto the norm = 1 surface
            cur_y -= np.log(cur_scale(cur_y)) / (self.S_alpha + self.S_beta)

        self.log.exit("Calculation")

        # Now we convert back to the null-CS
        z_null_sq = np.exp(cur_y)
        self.LE_sol["RNCS"] = np.sqrt(z_null_sq)
        # The diagnostic
        A_i_actual = np.zeros(2 * self.mol.nao)
        final_norm_sq = esp(z_null_sq, self.S_alpha + self.S_beta)
        self.log.write(f"Final norm squared = {final_norm_sq}")
        for i in range(2 * self.mol.nao):
            A_i_actual[i] = z_null_sq[i] * esp(z_null_sq, self.S_alpha + self.S_beta - 1, omit = [i]) / final_norm_sq

        diagnostic_table = []
        diagnostic_row_names = []
        for i in range(self.mol.nao):
            diagnostic_table.append([
                np.round(RSOPM_renorm[i][i], 6),
                np.round(A_i_actual[i], 6),
                np.round(100 * (1 - A_i_actual[i] / RSOPM_renorm[i][i]), 1)
                ])
            diagnostic_row_names.append(f"{i + 1}(a)")
        for i in range(self.mol.nao, 2 * self.mol.nao):
            diagnostic_table.append([
                np.round(RSOPM_renorm[i][i], 6),
                np.round(A_i_actual[i], 6),
                np.round(100 * (1 - A_i_actual[i] / RSOPM_renorm[i][i]), 1)
                ])
            diagnostic_row_names.append(f"{i + 1 - self.mol.nao}(b)")

        self.log.print_table(
            table_name = "<S_i> diagnostic>",
            column_names = ["< LE | S_i | LE >", "< Z | S_i | Z >", "Err %"],
            row_names = diagnostic_row_names,
            list_of_rows = diagnostic_table
            )

        self.log.exit()


        self.log.exit()



    # Entry function

    def find_LE_solution(self, spec, **kwargs):
        # requires HF_method to be established, i.e. mol_init
        self.log.enter(f"Obtaining low-excitation solution with env. {self.HF_method}, spec {spec}...", 1)

        if "mol_init" not in self.checklist or self.HF_method is None:
            self.log.write("WARNING: Molecule not initialised. Aborting...")
            self.log.exit()
            return(None)

        if "LE_sol" in self.checklist:
            self.log.write("WARNING: LE solution initialised before.")
            if self.LE_description["env"] != self.HF_method or self.LE_description["spec"] != spec:
                self.log.write(f"Previous solution found with a different spec ({self.LE_description["env"]}: {self.LE_description["spec"]})")
            self.log.exit()
            return(None)

        if spec not in self.low_excitation_methods[self.HF_method]:
            self.log.write(f"ERROR: Unknown spec '{spec}'. Methods available for {self.HF_method}: {self.low_excitation_methods[self.HF_method].keys()}")
            self.log.exit()
            return(None)

        self.LE_description["env"] = self.HF_method
        self.LE_description["spec"] = spec

        self.low_excitation_methods[self.HF_method][spec]["method"](**kwargs)

        self.log.exit()
        return(None)

    ###########################################################################
    ############################# Output methods ##############################
    ###########################################################################

    # ------------------------ Data storage management ------------------------

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
        if "mol_init" in self.checklist:
            self.disk_jockey.commit_datum_bulk("self_analysis", "physical_properties", {
                "HF_method" : self.HF_method,
                "reference_state_energy" : self.reference_state_energy
                })
        if "full_CI_sol" in self.checklist:
            self.disk_jockey.commit_datum_bulk("self_analysis", "full_CI_sol", {
                "E" : self.ci_energy,
                "sol" : {str(k): v for k, v in self.ci_sol.items()} # each key is a tuple of tuples
                })
        if "LE_sol" in self.checklist:
            LE_sol_encoded = {
                "E" : self.LE_sol["E"],
                "sol" : {str(k): v for k, v in self.LE_sol["sol"].items()}
                }
            for prop in ["red", "CSRM", "TPM", "SRRM", "SOPM", "RSOPM", "RNCS"]:
                if prop in self.LE_sol:
                    LE_sol_encoded[prop] = self.LE_sol[prop].tolist()
            self.disk_jockey.commit_datum_bulk("self_analysis", "LE_sol", LE_sol_encoded)
            """self.disk_jockey.commit_datum_bulk("self_analysis", "LE_sol", {
                "E" : self.LE_sol["E"],
                "sol" : {str(k): v for k, v in self.LE_sol["sol"].items()},
                #"red" : self.LE_sol["red"].tolist(),
                #"CSRM" : self.LE_sol["CSRM"].tolist(),
                #"TPM" : self.LE_sol["TPM"].tolist(),
                #"SRRM" : self.LE_sol["SRRM"].tolist(),
                #"SOPM" : self.LE_sol["SOPM"].tolist(),
                "RSOPM" : self.LE_sol["RSOPM"].tolist(),
                "RNCS" : self.LE_sol["RNCS"].tolist()
                })"""
            self.disk_jockey.commit_metadatum("self_analysis", "LE_sol", self.LE_description)

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

                self.HF_method = loaded_phys_properties["HF_method"]
                self.reference_state_energy = loaded_phys_properties["reference_state_energy"]
                self.check_off("mol_init")

                if "full_CI_sol" in loaded_checklist:
                    loaded_full_CI_sol = self.disk_jockey.data_bulks["self_analysis"]["full_CI_sol"]
                    self.ci_energy = loaded_full_CI_sol["E"]
                    self.ci_sol = {self.occ_tuple_restore(k): v for k, v in loaded_full_CI_sol["sol"].items()}
                    self.check_off("full_CI_sol")
                    self.log.write("Results from SCF performed on the full CI loaded...", 4)

                if "LE_sol" in loaded_checklist:
                    loaded_LE_sol = self.disk_jockey.data_bulks["self_analysis"]["LE_sol"]
                    self.LE_description = self.disk_jockey.metadata["self_analysis"]["LE_sol"]
                    # We need to re-tuple the scope
                    self.LE_description["scope"] = [tuple(prom_label) for prom_label in self.LE_description["scope"]]
                    self.LE_sol = {
                        "E" : loaded_LE_sol["E"],
                        "sol" : {self.str_to_prom_tuple(k): v for k, v in loaded_LE_sol["sol"].items()}
                        }
                    for prop in ["red", "CSRM", "TPM", "SRRM", "SOPM", "RSOPM", "RNCS"]:
                        if prop in loaded_LE_sol:
                            self.LE_sol[prop] = np.array(loaded_LE_sol[prop])
                    """self.LE_sol = {
                        "E" : loaded_LE_sol["E"],
                        "sol" : {self.str_to_prom_tuple(k): v for k, v in loaded_LE_sol["sol"].items()},
                        #"red" : np.array(loaded_LE_sol["red"]),
                        #"CSRM" : np.array(loaded_LE_sol["CSRM"]),
                        #"TPM" : np.array(loaded_LE_sol["TPM"]),
                        #"SRRM" : np.array(loaded_LE_sol["SRRM"]),
                        #"SOPM" : np.array(loaded_LE_sol["SOPM"]),
                        "RSOPM" : np.array(loaded_LE_sol["RSOPM"]),
                        "RNCS" : np.array(loaded_LE_sol["RNCS"])
                        }"""
                    self.check_off("LE_sol")
                    self.log.write(f"Results from diagonalisation on a low-excitation basis loaded...", 4)
                    self.log.write(f"  -environment: {self.LE_description["env"]}", 4)
                    self.log.write(f"  -specification: {self.LE_description["spec"]}", 4)
                    self.log.write(f"  -scope: {self.LE_description["scope"]}", 4)
                    self.log.write(f"  -description: {self.LE_description["label"]}", 4)
                    self.log.write(f"  -parameters:", 4)
                    for param_name, param_val in self.LE_description["params"].items():
                        self.log.write(f"    -{param_name}: {param_val}", 4)

                self.log.exit()

            if "measured_datasets" in what_to_load:
                self.log.enter("Restoring measured datasets...", 3)
                for loaded_dataset in self.disk_jockey.metadata["system"]["log"]["measured_datasets"]:
                    if loaded_dataset not in self.measured_datasets:
                        self.log.enter(f"Loading dataset '{loaded_dataset}'...", 4)
                        for dataset_datum in self.disk_jockey.data_nodes[loaded_dataset]:
                            self.disk_jockey.load_datum(loaded_dataset, dataset_datum)
                        self.measured_datasets.append(loaded_dataset)
                        self.log.write(f"Results from dataset '{loaded_dataset}' loaded.", 4)
                        self.log.write(f"  -sampling method: {self.disk_jockey.metadata[loaded_dataset]["basis_samples"]["method"]}", 4)
                        self.log.write(f"  -parameters:", 4)
                        for param_name, param_val in self.disk_jockey.metadata[loaded_dataset]["basis_samples"]["params"].items():
                            self.log.write(f"    -{param_name}: {param_val}", 4)

                        self.log.exit()
                    else:
                        self.log.write("WARNING: Attempted to load a dataset which exists in internal checklist. Data from the disk was ignored.", 0)
                self.log.exit()

            self.log.exit()


    # ------------------------------- Plotting --------------------------------

    def plot_datasets(self, reference_energies = None):
        # Plots energy against configuration size
        self.log.enter("Plotting obtained measurements...", 1)

        plt.title(f"[{self.ID}] Ground state estimate with Monte Carlo")
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
        if "full_CI_sol" in self.checklist:
            plt.axhline(y = self.ci_energy, label = "full CI", color = functions.ref_energy_colors["full CI"])
        if "LE_sol" in self.checklist:
            # LE ground state
            plt.axhline(y = self.LE_sol["E"], label = "LE CI", color = functions.ref_energy_colors["LE CI"])
            # LE mean-value uncorrelated state
            #LE_no_cor = [CS_Thouless(self.mol.nao, self.S_alpha, self.LE_sol["exp"]["a"]), CS_Thouless(self.mol.nao, self.S_beta, self.LE_sol["exp"]["b"])]
            #LE_no_cor_E = self.H_overlap(LE_no_cor, LE_no_cor).real
            #plt.axhline(y = LE_no_cor_E, label = "LE-mean CS", color = functions.ref_energy_colors["LE CI"], linestyle = "dashed")



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

        # We add a second y axis which shows the difference from the real ground state (if given) in Kelvin
        # If full CI sol is known, we set it to 0. Otherwise, we set the HF state energy to 0
        if "full_CI_sol" in self.checklist:
            ref_E = self.ci_energy
        else:
            ref_E = self.reference_state_energy
        H_in_K = 315775.326864009 # tha value of 1 Hartree in Kelvin
        def H_to_K(x):
            return((x - ref_E) * H_in_K)
        def K_to_H(x):
            return(ref_E + x / H_in_K)
        secax_y = plt.gca().secondary_yaxis(
            'right', functions=(H_to_K, K_to_H))
        secax_y.set_ylabel(r'$E\ [K]$')

        self.log.write(f"Displaying plot...", 5)
        plt.legend()
        plt.tight_layout()
        plt.show()

        self.log.exit()

    def plot_datasets_extra(self, reference_energies = None):
        # Plots energy against configuration size
        self.log.enter("Plotting obtained measurements...", 1)

        plt.title(f"[{self.ID}] Ground state estimate with Monte Carlo")
        plt.xlabel("1/sqrt(Basis size)")
        plt.ylabel("E [Hartree]")

        linfit_xspace = np.linspace(0.0, 1.0, 3)
        min_N = 10 # do not consider lower values of N for the fit
        sigma_zero = 0.01 # dummy val of inherent err

        for i in range(len(self.measured_datasets)):
            self.log.write(f"Collecting data from dataset '{self.measured_datasets[i]}'...", 5)
            ds_val = self.disk_jockey.data_bulks[self.measured_datasets[i]]["result_energy_states"]
            # ds_val is a list of dicts - here we cast it into plottable arrays
            N_space = []
            E_space = []
            trim_i = 0
            for row in ds_val:
                N_space.append(row["N"])
                E_space.append(row["E [H]"])

            while(N_space[trim_i] < min_N):
                trim_i += 1
                if trim_i == len(N_space):
                    trim_i = 0
                    break


            sqinv_N_space = np.array(1/np.sqrt(N_space))
            sqinv_N_space_yerr = sigma_zero * sqinv_N_space # the lower the N value, the higher the uncertainty! propto 1/sqrt(N). Res err is quoted as a proportion to the unknown coef
            popt, pcov = sp.optimize.curve_fit(
                lambda x, l, c: l * x + c,
                sqinv_N_space[trim_i:],
                E_space[trim_i:],
                sigma=sqinv_N_space_yerr[trim_i:],
                absolute_sigma=True
            )
            l_best, c_best = popt
            l_err, c_err = np.sqrt(np.diag(pcov))

            E_zero = c_best
            E_zero_err = c_err / sigma_zero

            self.log.write(f"Value of lin fit at x = 0: {E_zero} +- {E_zero_err}.sigma_0, where sigma_0 is the error on a single datapoint.")

            plt.errorbar(sqinv_N_space, E_space, yerr = sqinv_N_space_yerr, fmt = 'x', capsize = 3, label = self.measured_datasets[i])
            plt.plot(linfit_xspace, l_best * linfit_xspace + c_best, label = 'Linear fit')
            #plt.errorbar(np.zeros(1), np.array([E_zero]), yerr = np.array([E_zero_err * sigma_zero]), fmt = 'x', capsize = 3)

            """

            trim_N_space = np.array(1/np.sqrt(N_space))[20:]
            trim_E_space = np.array(E_space)[20:]

            E_fit = np.polyfit(trim_N_space, trim_E_space, deg = 1)
            E_fit_func = lambda x : E_fit[0] * x + E_fit[1]

            plt.plot(1/np.sqrt(N_space), E_space, "x", label = self.measured_datasets[i])
            plt.plot(np.concatenate((np.zeros(1), trim_N_space)), E_fit_func(np.concatenate((np.zeros(1), trim_N_space))), label = 'Linear fit')

            self.log.write(f"Value of lin fit at x = 0: {E_fit_func(0.0)}")"""

        if "mol_init" in self.checklist:
            plt.axhline(y = self.reference_state_energy, label = "ref state", color = functions.ref_energy_colors["ref state"])
        if "full_CI_sol" in self.checklist:
            plt.axhline(y = self.ci_energy, label = "full CI", color = functions.ref_energy_colors["full CI"])
        if "LE_sol" in self.checklist:
            # LE ground state
            plt.axhline(y = self.LE_sol["E"], label = "LE CI", color = functions.ref_energy_colors["LE CI"])
            # LE mean-value uncorrelated state
            #LE_no_cor = [CS_Thouless(self.mol.nao, self.S_alpha, self.LE_sol["exp"]["a"]), CS_Thouless(self.mol.nao, self.S_beta, self.LE_sol["exp"]["b"])]
            #LE_no_cor_E = self.H_overlap(LE_no_cor, LE_no_cor).real
            #plt.axhline(y = LE_no_cor_E, label = "LE-mean CS", color = functions.ref_energy_colors["LE CI"], linestyle = "dashed")



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

        # We add a second y axis which shows the difference from the real ground state (if given) in Kelvin
        # If full CI sol is known, we set it to 0. Otherwise, we set the HF state energy to 0
        if "full_CI_sol" in self.checklist:
            ref_E = self.ci_energy
        else:
            ref_E = self.reference_state_energy
        H_in_K = 315775.326864009 # tha value of 1 Hartree in Kelvin
        def H_to_K(x):
            return((x - ref_E) * H_in_K)
        def K_to_H(x):
            return(ref_E + x / H_in_K)
        secax_y = plt.gca().secondary_yaxis(
            'right', functions=(H_to_K, K_to_H))
        secax_y.set_ylabel(r'$E\ [K]$')


        self.log.write(f"Displaying plot...", 5)
        plt.legend()
        plt.tight_layout()
        plt.show()

        self.log.exit()

    def plot_datasets_against_param(self, param_name, param_label = None, reference_energies = None):
        # plots only the final value of each dataset against a parameter value in the dataset's metadata
        # each dataset has to contain the given parameter name in its basis_samples metadata
        # param_label is a HR label
        if param_label is None:
            param_label = param_name

        self.log.enter(f"Plotting obtained measurements against parameter '{param_name}'...", 1)

        plt.title(f"[{self.ID}] Ground state estimate against {param_label}")
        plt.xlabel(param_label)
        plt.ylabel("E [Hartree]")

        param_space = []
        ground_state_energy_space = []

        for i in range(len(self.measured_datasets)):
            self.log.write(f"Collecting data from dataset '{self.measured_datasets[i]}'...", 5)
            ds_val = self.disk_jockey.data_bulks[self.measured_datasets[i]]["result_energy_states"]
            # ds_val is a list of dicts - here we cast it into plottable arrays
            N_space = []
            E_space = []
            for row in ds_val:
                N_space.append(row["N"])
                E_space.append(row["E [H]"])

            max_N_i = np.argmax(N_space)
            min_E = E_space[max_N_i]

            param_space.append(self.disk_jockey.metadata[self.measured_datasets[i]]["basis_samples"]["params"][param_name])
            ground_state_energy_space.append(min_E)

        if "mol_init" in self.checklist:
            plt.axhline(y = self.reference_state_energy, label = "ref state", color = functions.ref_energy_colors["ref state"])
        if "full_CI_sol" in self.checklist:
            plt.axhline(y = self.ci_energy, label = "full CI", color = functions.ref_energy_colors["full CI"])
        if "LE_sol" in self.checklist:
            # LE ground state
            plt.axhline(y = self.LE_sol["E"], label = "LE CI", color = functions.ref_energy_colors["LE CI"])
            # LE mean-value uncorrelated state
            #LE_no_cor = [CS_Thouless(self.mol.nao, self.S_alpha, self.LE_sol["exp"]["a"]), CS_Thouless(self.mol.nao, self.S_beta, self.LE_sol["exp"]["b"])]
            #LE_no_cor_E = self.H_overlap(LE_no_cor, LE_no_cor).real
            #plt.axhline(y = LE_no_cor_E, label = "LE-mean CS", color = functions.ref_energy_colors["LE CI"], linestyle = "dashed")



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

        plt.scatter(param_space, ground_state_energy_space, label = "$E_{min}$ in datasets", marker = "x")

        self.log.write(f"Displaying plot...", 5)
        plt.legend()
        plt.tight_layout()
        plt.show()

        self.log.exit()

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

    def plot_SECS_restricted_heatmap(self, ax = None, trim_M = None):

        self.user_actions += f"plot_SECS_restricted_heatmap [trim_M = {trim_M}]\n"

        act_M = self.mol.nao
        if trim_M is not None:
            if trim_M > self.S_alpha: # We need at least one empty shell
                act_M = min(trim_M, self.mol.nao)

        #self.find_LE_solution_SECS(trim_M)
        assert "LE_sol" in self.checklist
        assert self.LE_description["env"] == "RHF" and self.LE_description["spec"] == "SECS"

        one_exc_closed_shell_hm = self.LE_sol["exp"]

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

    def plot_LE_reduction_matrix(self, spin, log_plot = True, signed = True, ax = None):
        self.user_actions += f"plot_LE_reduction_matrix\n"

        assert "LE_sol" in self.checklist
        assert "red" in self.LE_sol

        if ax is None:
            ax = plt.gca()

        if spin == "a":
            submatrix = np.array(self.LE_sol["red"][:self.mol.nao, :self.mol.nao])
        elif spin == "b":
            submatrix = np.array(self.LE_sol["red"][self.mol.nao:, self.mol.nao:])
        if log_plot and signed:
            submatrix = - np.log(np.abs(submatrix) + 1e-20) * np.sign(submatrix + 1e-20)
            heatmap_v_min = -10
            heatmap_v_max = 10
            heatmap_v_label = "Matrix element of transition; $\\pm \\ln |\\langle \\text{LE g.s.} | b^\\dag_i b_j | \\text{LE g.s.} \\rangle |$ (pos. sign for positive val.)"
        if log_plot and not signed:
            submatrix = np.log(np.abs(submatrix) + 1e-20)
            heatmap_v_min = -10
            heatmap_v_max = 0
            heatmap_v_label = "Matrix element of transition; $\\ln |\\langle \\text{LE g.s.} | b^\\dag_i b_j | \\text{LE g.s.} \\rangle |$"
        if not log_plot and signed:
            #submatrix = np.abs(submatrix)
            heatmap_v_min = -1
            heatmap_v_max = 1
            heatmap_v_label = "Matrix element of transition; $\\langle \\text{LE g.s.} | b^\\dag_i b_j | \\text{LE g.s.} \\rangle$"
        if not log_plot and not signed:
            submatrix = np.abs(submatrix)
            heatmap_v_min = 0
            heatmap_v_max = 1
            heatmap_v_label = "Matrix element of transition; $|\\langle \\text{LE g.s.} | b^\\dag_i b_j | \\text{LE g.s.} \\rangle |$"

        # Plot the heatmap
        heatmap = ax.imshow(submatrix,
                            cmap='Wistia',
                            interpolation='none',
                            vmin=heatmap_v_min,
                            vmax=heatmap_v_max
            ) # extent = functions.m_ext(one_exc_closed_shell_hm)


        row_lab = [f"{i + 1}" for i in range(submatrix.shape[0])]
        col_lab = [f"{i + 1}" for i in range(submatrix.shape[1])]

        # Create colorbar
        cbar = ax.figure.colorbar(heatmap, ax=ax)
        cbar.ax.set_ylabel(heatmap_v_label, rotation=-90, va="bottom")

        ax.set_xlabel("Left-side annihilation operator index $i$")
        ax.set_ylabel("Right-side annihilation operator index $j$")

        ax.set_xticks(np.arange(submatrix.shape[1]), labels=col_lab)
        ax.set_yticks(np.arange(submatrix.shape[0]), labels=row_lab)

        # Grid
        ax.spines[:].set_visible(False)
        ax.set_xticks(np.arange(submatrix.shape[1]+1)-.5, minor=True)
        ax.set_yticks(np.arange(submatrix.shape[0]+1)-.5, minor=True)
        ax.grid(which="minor", color="b", linestyle='-', linewidth=3)
        ax.tick_params(which="minor", bottom=False, left=False)

        return(heatmap, cbar)

    def plot_LE_CSRM(self, log_plot = True, signed = True, ax = None):
        self.user_actions += f"plot_LE_CSRM\n"

        assert "LE_sol" in self.checklist
        assert "CSRM" in self.LE_sol

        self.log.enter("Plotting the closed-shell reduction matrix...")
        self.log.print_matrix(self.LE_sol["CSRM"], "CSRM")

        if ax is None:
            ax = plt.gca()

        submatrix = np.array(self.LE_sol["CSRM"])
        if log_plot and signed:
            submatrix = - np.log(np.abs(submatrix) + 1e-20) * np.sign(submatrix + 1e-20)
            heatmap_v_min = -10
            heatmap_v_max = 10
            heatmap_v_label = "Matrix element of transition; $\\pm \\ln |\\langle \\text{LE g.s.} | b^\\dag_i b_j | \\text{LE g.s.} \\rangle |$ (pos. sign for positive val.)"
        if log_plot and not signed:
            submatrix = np.log(np.abs(submatrix) + 1e-20)
            heatmap_v_min = -10
            heatmap_v_max = 0
            heatmap_v_label = "Matrix element of transition; $\\ln |\\langle \\text{LE g.s.} | b^\\dag_i b_j | \\text{LE g.s.} \\rangle |$"
        if not log_plot and signed:
            #submatrix = np.abs(submatrix)
            heatmap_v_min = -1
            heatmap_v_max = 1
            heatmap_v_label = "Matrix element of transition; $\\langle \\text{LE g.s.} | b^\\dag_i b_j | \\text{LE g.s.} \\rangle$"
        if not log_plot and not signed:
            submatrix = np.abs(submatrix)
            heatmap_v_min = 0
            heatmap_v_max = 1
            heatmap_v_label = "Matrix element of transition; $|\\langle \\text{LE g.s.} | b^\\dag_i b_j | \\text{LE g.s.} \\rangle |$"

        # Plot the heatmap
        heatmap = ax.imshow(submatrix,
                            cmap='Wistia',
                            interpolation='none',
                            vmin=heatmap_v_min,
                            vmax=heatmap_v_max
            ) # extent = functions.m_ext(one_exc_closed_shell_hm)


        row_lab = [f"{i + 1}" for i in range(submatrix.shape[0])]
        col_lab = [f"{i + 1}" for i in range(submatrix.shape[1])]

        # Create colorbar
        cbar = ax.figure.colorbar(heatmap, ax=ax)
        cbar.ax.set_ylabel(heatmap_v_label, rotation=-90, va="bottom")

        ax.set_xlabel("Left-side annihilation operator index $i$")
        ax.set_ylabel("Right-side annihilation operator index $j$")

        ax.set_xticks(np.arange(submatrix.shape[1]), labels=col_lab)
        ax.set_yticks(np.arange(submatrix.shape[0]), labels=row_lab)

        # Grid
        ax.spines[:].set_visible(False)
        ax.set_xticks(np.arange(submatrix.shape[1]+1)-.5, minor=True)
        ax.set_yticks(np.arange(submatrix.shape[0]+1)-.5, minor=True)
        ax.grid(which="minor", color="b", linestyle='-', linewidth=3)
        ax.tick_params(which="minor", bottom=False, left=False)

        self.log.exit()

        return(heatmap, cbar)

    def plot_LE_TPM(self, spin = "a", log_plot = True, ax = None):
        self.user_actions += f"plot_LE_TPM\n"

        self.log.enter("Plotting the transition prevalence matrix...", 5)

        assert "LE_sol" in self.checklist
        assert "TPM" in self.LE_sol

        if ax is None:
            ax = plt.gca()

        if spin == "a":
            submatrix = np.array(self.LE_sol["TPM"][:self.mol.nao, :])
        elif spin == "b":
            submatrix = np.array(self.LE_sol["TPM"][self.mol.nao:, :])

        if log_plot:
            submatrix = np.log(submatrix + 1e-20)
            heatmap_v_min = -10
            heatmap_v_max = 0
            heatmap_v_label = "Matrix element of transition; $\\ln |\\langle \\text{LE g.s.} | b^\\dag_i b_j | \\text{LE g.s.} \\rangle |$"
        else:
            heatmap_v_min = 0
            heatmap_v_max = 1
            heatmap_v_label = "Matrix element of transition; $|\\langle \\text{LE g.s.} | b^\\dag_i b_j | \\text{LE g.s.} \\rangle |$"

        # Plot the heatmap
        heatmap = ax.imshow(submatrix,
                            cmap='Wistia',
                            interpolation='none',
                            vmin=heatmap_v_min,
                            vmax=heatmap_v_max
            ) # extent = functions.m_ext(one_exc_closed_shell_hm)


        row_lab = [f"{i + 1}" for i in range(submatrix.shape[0])]
        col_lab = [f"{i + 1}" for i in range(submatrix.shape[1])]

        # Create colorbar
        cbar = ax.figure.colorbar(heatmap, ax=ax)
        cbar.ax.set_ylabel(heatmap_v_label, rotation=-90, va="bottom")

        ax.set_xlabel("Left-side annihilation operator index $i$")
        ax.set_ylabel("Right-side annihilation operator index $j$")

        ax.set_xticks(np.arange(submatrix.shape[1]), labels=col_lab)
        ax.set_yticks(np.arange(submatrix.shape[0]), labels=row_lab)

        # Grid
        ax.spines[:].set_visible(False)
        ax.set_xticks(np.arange(submatrix.shape[1]+1)-.5, minor=True)
        ax.set_yticks(np.arange(submatrix.shape[0]+1)-.5, minor=True)
        ax.grid(which="minor", color="b", linestyle='-', linewidth=3)
        ax.tick_params(which="minor", bottom=False, left=False)

        self.log.exit()

        return(heatmap, cbar)

    def plot_LE_SRRM(self, spin = "a", log_plot = True, ax = None):
        self.user_actions += f"plot_LE_SRRM\n"

        self.log.enter("Plotting the spin-reduced reduction matrix...", 5)

        assert "LE_sol" in self.checklist
        assert "SRRM" in self.LE_sol

        if ax is None:
            ax = plt.gca()

        if spin == "a":
            submatrix = np.array(self.LE_sol["SRRM"][:self.mol.nao, :])
        elif spin == "b":
            submatrix = np.array(self.LE_sol["SRRM"][self.mol.nao:, :])

        if log_plot:
            submatrix = np.log(submatrix + 1e-20)
            heatmap_v_min = -10
            heatmap_v_max = 0
            heatmap_v_label = "Matrix element of transition; $\\ln |\\langle \\text{LE g.s.} | b^\\dag_i b_j | \\text{LE g.s.} \\rangle |$"
        else:
            heatmap_v_min = 0
            heatmap_v_max = 1
            heatmap_v_label = "Matrix element of transition; $|\\langle \\text{LE g.s.} | b^\\dag_i b_j | \\text{LE g.s.} \\rangle |$"

        # Plot the heatmap
        heatmap = ax.imshow(submatrix,
                            cmap='Wistia',
                            interpolation='none',
                            vmin=heatmap_v_min,
                            vmax=heatmap_v_max
            ) # extent = functions.m_ext(one_exc_closed_shell_hm)


        row_lab = [f"{i + 1}" for i in range(submatrix.shape[0])]
        col_lab = [f"{i + 1}" for i in range(submatrix.shape[1])]

        # Create colorbar
        cbar = ax.figure.colorbar(heatmap, ax=ax)
        cbar.ax.set_ylabel(heatmap_v_label, rotation=-90, va="bottom")

        ax.set_xlabel("Left-side annihilation operator index $i$")
        ax.set_ylabel("Right-side annihilation operator index $j$")

        ax.set_xticks(np.arange(submatrix.shape[1]), labels=col_lab)
        ax.set_yticks(np.arange(submatrix.shape[0]), labels=row_lab)

        # Grid
        ax.spines[:].set_visible(False)
        ax.set_xticks(np.arange(submatrix.shape[1]+1)-.5, minor=True)
        ax.set_yticks(np.arange(submatrix.shape[0]+1)-.5, minor=True)
        ax.grid(which="minor", color="b", linestyle='-', linewidth=3)
        ax.tick_params(which="minor", bottom=False, left=False)

        self.log.exit()

        return(heatmap, cbar)

    def plot_LE_SOPM(self, spin = "a", log_plot = True, ax = None):
        self.user_actions += f"plot_LE_SOPM\n"

        self.log.enter("Plotting the spin-reduced reduction matrix...", 5)

        assert "LE_sol" in self.checklist
        assert "SOPM" in self.LE_sol

        if ax is None:
            ax = plt.gca()

        if spin == "a":
            submatrix = np.array(self.LE_sol["SOPM"][:self.mol.nao, :self.mol.nao])
        elif spin == "b":
            submatrix = np.array(self.LE_sol["SOPM"][self.mol.nao:, self.mol.nao:])

        if log_plot:
            submatrix = np.log(submatrix + 1e-20)
            heatmap_v_min = -10
            heatmap_v_max = 0
            heatmap_v_label = "Matrix element of transition; $\\ln |\\langle \\text{LE g.s.} | \\hat{S}_{ii} T_{jj} | \\text{LE g.s.} \\rangle |$"
        else:
            heatmap_v_min = 0
            heatmap_v_max = 1
            heatmap_v_label = "Matrix element of transition; $|\\langle \\text{LE g.s.} | \\hat{T}_{ii} T_{jj} | \\text{LE g.s.} \\rangle |$"

        # Plot the heatmap
        heatmap = ax.imshow(submatrix,
                            cmap='Wistia',
                            interpolation='none',
                            vmin=heatmap_v_min,
                            vmax=heatmap_v_max
            ) # extent = functions.m_ext(one_exc_closed_shell_hm)


        row_lab = [f"{i + 1}" for i in range(submatrix.shape[0])]
        col_lab = [f"{i + 1}" for i in range(submatrix.shape[1])]

        # Create colorbar
        cbar = ax.figure.colorbar(heatmap, ax=ax)
        cbar.ax.set_ylabel(heatmap_v_label, rotation=-90, va="bottom")

        ax.set_xlabel("Left-side annihilation operator index $i$")
        ax.set_ylabel("Right-side annihilation operator index $j$")

        ax.set_xticks(np.arange(submatrix.shape[1]), labels=col_lab)
        ax.set_yticks(np.arange(submatrix.shape[0]), labels=row_lab)

        # Grid
        ax.spines[:].set_visible(False)
        ax.set_xticks(np.arange(submatrix.shape[1]+1)-.5, minor=True)
        ax.set_yticks(np.arange(submatrix.shape[0]+1)-.5, minor=True)
        ax.grid(which="minor", color="b", linestyle='-', linewidth=3)
        ax.tick_params(which="minor", bottom=False, left=False)

        self.log.exit()

        return(heatmap, cbar)

    def plot_LE_RSOPM(self, spin = "a", log_plot = True, ax = None):
        self.user_actions += f"plot_LE_RSOPM\n"

        self.log.enter("Plotting the reduced simultanous occupancy proportion matrix...", 5)

        assert "LE_sol" in self.checklist
        assert "RSOPM" in self.LE_sol

        if ax is None:
            ax = plt.gca()

        if spin == "a":
            submatrix = np.array(self.LE_sol["RSOPM"][:self.mol.nao, :self.mol.nao])
        elif spin == "b":
            submatrix = np.array(self.LE_sol["RSOPM"][self.mol.nao:, self.mol.nao:])

        if log_plot:
            submatrix = np.log(submatrix + 1e-20)
            heatmap_v_min = -10
            heatmap_v_max = 0
            heatmap_v_label = "Matrix element of transition; $\\ln |\\langle \\text{LE g.s.} | \\hat{S}_{ii} T_{jj} | \\text{LE g.s.} \\rangle |$"
        else:
            heatmap_v_min = 0
            heatmap_v_max = 1
            heatmap_v_label = "Matrix element of transition; $|\\langle \\text{LE g.s.} | \\hat{T}_{ii} T_{jj} | \\text{LE g.s.} \\rangle |$"

        # Plot the heatmap
        heatmap = ax.imshow(submatrix,
                            cmap='Wistia',
                            interpolation='none',
                            vmin=heatmap_v_min,
                            vmax=heatmap_v_max
            ) # extent = functions.m_ext(one_exc_closed_shell_hm)


        row_lab = [f"{i + 1}" for i in range(submatrix.shape[0])]
        col_lab = [f"{i + 1}" for i in range(submatrix.shape[1])]

        # Create colorbar
        cbar = ax.figure.colorbar(heatmap, ax=ax)
        cbar.ax.set_ylabel(heatmap_v_label, rotation=-90, va="bottom")

        ax.set_xlabel("Left-side annihilation operator index $i$")
        ax.set_ylabel("Right-side annihilation operator index $j$")

        ax.set_xticks(np.arange(submatrix.shape[1]), labels=col_lab)
        ax.set_yticks(np.arange(submatrix.shape[0]), labels=row_lab)

        # Grid
        ax.spines[:].set_visible(False)
        ax.set_xticks(np.arange(submatrix.shape[1]+1)-.5, minor=True)
        ax.set_yticks(np.arange(submatrix.shape[0]+1)-.5, minor=True)
        ax.grid(which="minor", color="b", linestyle='-', linewidth=3)
        ax.tick_params(which="minor", bottom=False, left=False)

        self.log.exit()

        return(heatmap, cbar)

    # ------------------------------- Printing --------------------------------

    def print_singlet_info(self, top_N_closed_shells = 5):
        self.log.enter("Reporting on singlet Spin-Adapted Configurations...", 0)
        if "full_CI_sol" not in self.checklist:
            self.log.write("Full CI solution not known, aborting...")
            self.log.exit()
            return(None)
        self.log.enter("Closed-shell Slater determinants")
        closed_shell_proj, closed_shell_N = self.closed_shell_projection()
        self.log.write(f"Norm squared of projection onto all {closed_shell_N} closed-shell states = {closed_shell_proj:0.5f}")
        self.log.write(f"Top {top_N_closed_shells} closed shell occupancies are:")
        top_N_closed_shell_states = self.get_top_closed_shells(top_N_closed_shells)
        for i in range(top_N_closed_shells):
            prom_label = self.get_prom_label(top_N_closed_shell_states[i][1], hr = True)
            self.log.write(f"  {i+1}) Coef = {top_N_closed_shell_states[i][0]:0.4f}; occ. = {top_N_closed_shell_states[i][1]} (prom ({prom_label[0]}) -> ({prom_label[1]}))")
        self.log.exit()
        self.log.enter("Two-open-shell singlet SACs")
        single_exc_proj, single_exc_N = self.single_excitation_singlets_projection()
        self.log.write(f"Total square norm of the projection into all {single_exc_N} two-open-shell SAC singlets is {single_exc_proj:0.5f}")
        self.log.exit()
        self.log.write(f"The total space of singlet SACs with up to two open shells is {closed_shell_N + single_exc_N}-dimensional; with norm square projection {closed_shell_proj + single_exc_proj:0.5f}")
        self.log.exit()

    def print_UHF_low_excitation_info(self, top_N_single_excitation = 5, top_N_simulatenous_excitation = 10):
        self.log.enter("Reporting on low-excitation configurations for inequal MOs...", 0)
        if "full_CI_sol" not in self.checklist:
            self.log.write("Full CI solution not known, aborting...")
            self.log.exit()
            return(None)

        ref_a, ref_b = self.get_ref_state()
        ref_state_component = self.ground_state_component(ref_a, ref_b)
        ref_state_proj = ref_state_component * ref_state_component
        self.log.write(f"No-excitation (reference) state norm squared ground state projection = {ref_state_proj:0.5f}")


        self.log.enter("Single-excitation Slater determinants")
        single_exc_top, single_exc_N, single_exc_proj = self.get_top_single_excitation_states(top_N_single_excitation)
        single_exc_N_a, single_exc_N_b = single_exc_N
        single_exc_proj_a, single_exc_proj_b = single_exc_proj
        self.log.write(f"For all single-excitation Slater determinants:")
        self.log.write(f"  -Exc. on spin alpha: {single_exc_N_a} states with norm squared projection {single_exc_proj_a:0.5f}")
        self.log.write(f"  -Exc. on spin beta: {single_exc_N_b} states with norm squared projection {single_exc_proj_b:0.5f}")
        self.log.write(f"  -In total: {single_exc_N_a + single_exc_N_b} states with norm squared projection {single_exc_proj_a + single_exc_proj_b:0.5f}")
        self.log.write(f"Top {len(single_exc_top[0])} Slater determinants with one excited spin-alpha electron:")
        for i in range(len(single_exc_top[0])):
            self.log.write(f"  {i+1}) Coef = {single_exc_top[0][i][0]:0.4f}; promotion {single_exc_top[0][i][2]}")
        self.log.write(f"Top {len(single_exc_top[1])} Slater determinants with one excited spin-beta electron:")
        for i in range(len(single_exc_top[1])):
            self.log.write(f"  {i+1}) Coef = {single_exc_top[1][i][0]:0.4f}; promotion {single_exc_top[1][i][2]}")
        self.log.exit()

        self.log.enter("Simultaneourly excited Slater deteminants")
        sim_exc_top, sim_exc_N, sim_exc_proj = self.get_top_simultaneously_excited_states(top_N_simulatenous_excitation)
        self.log.write(f"Total square norm of the projection into all {sim_exc_N} simultaneously-excited Slater determinants: {sim_exc_proj:0.5f}")
        self.log.write(f"Top {len(sim_exc_top)} simultaneously-excited Slater determinants:")
        for i in range(len(sim_exc_top)):
            self.log.write(f"  {i+1}) Coef = {sim_exc_top[i][0]:0.4f}; promotion {sim_exc_top[i][2]}")
        self.log.exit()

        self.log.write(f"The total space of states with up to one excitation on either spin subspace is {1 + single_exc_N_a + single_exc_N_b + sim_exc_N}-dimensional; with norm square projection {ref_state_proj + single_exc_proj_a + single_exc_proj_b + sim_exc_proj:0.5f}")
        self.log.exit()






