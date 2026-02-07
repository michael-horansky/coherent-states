import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from pyscf import gto, scf, cc, ao2mo, fci

from coherent_states.CS_Thouless import CS_Thouless
from coherent_states.CS_Qubit import CS_Qubit

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

        CS_sample = [[ground_state_solver.coherent_state_types[kwargs["CS"]].null_state(self.mol.nao, self.S_alpha), ground_state_solver.coherent_state_types[kwargs["CS"]].null_state(self.mol.nao, self.S_beta)]]
        for i in range(kwargs["N"]):

            if assume_spin_symmetry:
                new_sample_state = ground_state_solver.coherent_state_types[kwargs["CS"]].random_state(self.mol.nao, self.S_alpha, kwargs["sampling_method"])
                CS_sample.append([new_sample_state, new_sample_state])

            else:
                CS_sample.append([
                    ground_state_solver.coherent_state_types[kwargs["CS"]].random_state(self.mol.nao, self.S_alpha, kwargs["sampling_method"]),
                    ground_state_solver.coherent_state_types[kwargs["CS"]].random_state(self.mol.nao, self.S_beta, kwargs["sampling_method"])
                    ])

        basis_samples_bulk = []
        for i in range(N):
            basis_samples_bulk.append([CS_sample[i][0].z, CS_sample[i][1].z])
        self.disk_jockey.commit_datum_bulk("basis_samples", basis_samples_bulk)

        # Firstly we find the normalisation coefficients
        """norm_coefs = np.zeros(N)
        norm_coefs[0] = 1.0
        for i in range(1, N):
            sig, mag = CS_sample[i].log_norm_coef
            if (sig * np.exp(-mag / 2.0)).imag > 1e-04:
                procedure_diagnostic.append(f"basis norm not real!!")
            norm_coefs[i] = (sig * np.exp(-mag / 2.0)).real

        print("-- Normalisation coefs:", norm_coefs)

        procedure_diagnostic.append(f"basis norm max/min = {np.max(norm_coefs) / np.min(norm_coefs)}")"""

        overlap_matrix = np.zeros((N, N), dtype=complex) # [a][b] = <a|b>
        for i in range(N):
            for j in range(N):
                overlap_matrix[i][j] = CS_sample[i][0].norm_overlap(CS_sample[j][0]) * CS_sample[i][1].norm_overlap(CS_sample[j][1])
        print("-- Overlap matrix:")
        print(overlap_matrix)
        print(f"[Overlap matrix condition number = {np.linalg.cond(overlap_matrix)}]")

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

        msg = f"  Explicit Hamiltonian evaluation begins"
        new_sem_ID = self.semaphor.create_event(np.linspace(0, N * (N + 1) / 2, 100 + 1), msg)

        for a in range(N):
            # Here the diagonal <Z_a|H|Z_a>
            cur_H_overlap = self.H_overlap(CS_sample[a], CS_sample[a])
            H_eff[a][a] = cur_H_overlap
            self.semaphor.update(new_sem_ID, a * (a + 1) / 2)

            # Here the off-diagonal, using the fact that H_eff is a Hermitian matrix
            for b in range(a):
                # We explicitly calculate <Z_a | H | Z_b>
                cur_H_overlap = self.H_overlap(CS_sample[a], CS_sample[b])
                H_eff[a][b] = cur_H_overlap
                H_eff[b][a] = np.conjugate(H_eff[a][b])
                self.semaphor.update(new_sem_ID, a * (a + 1) / 2 + b + 1)

        self.solution_benchmark = self.semaphor.finish_event(new_sem_ID, "    Evaluation")

        # H_eff diagonal terms
        print("------ Debug and diagnostics")

        for i in range(N):
            print(f"  Z_{i} = [ {repr(CS_sample[i][0].z)}, {repr(CS_sample[i][1].z)} ]")
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


    def find_ground_state_krylov(self, **kwargs):
        # kwargs:
        #     -dt: float; time spacing of the basis sampling process
        print("skibidfi", kwargs["dt"])

        # The initial guess is the null matrix, i.e. all pi_1 modes occupied.




    def find_ground_state_imaginary_timeprop(self, **kwargs):
        # kwargs:
        #     -dt: float; timestep
        #     -tol: float; tolerance on the solution or smth
        print("skibidfi", kwargs["tol"])


    find_ground_state_methods = {
            "sampling" : find_ground_state_sampling,
            "krylov" : find_ground_state_krylov,
            "imag_timeprop" : find_ground_state_imaginary_timeprop
        }


    ###########################################################################
    # ----------------------------- User methods ------------------------------
    ###########################################################################

    def initialise_molecule(self, mol):
        # mol is an instance of pyscf.gto.Mole

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
        MO_coefs = mean_field.mo_coeff

        print("  Transforming 1e and 2e integrals to MO basis...")
        self.MO_H_one = np.matmul(MO_coefs.T, np.matmul(AO_H_one, MO_coefs))
        MO_H_two_packed = ao2mo.kernel(self.mol, MO_coefs)
        MO_H_two_chemist = ao2mo.restore(1, MO_H_two_packed, MO_coefs.shape[1]) # In mulliken notation: MO_H_two[p][q][r][s] = (pq|rs)

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

        """for p in range(self.S_alpha):
            for q in range(self.S_alpha):
                if p != q:
                    null_state_energy_two += 4 * self.mode_exchange_energy([p, q], [q, p])
                #null_state_energy_two += 0.5 * (self.MO_H_two[p][q][p][q] - self.MO_H_two[p][q][q][p])"""
        # we are looking for quadruplets of the type pqqp and pqpq. These are equivalent (and hence double up) when p, q act on the same spin
        # When p,q act on different spins, only one of the two options is permitted (e.g. p_up q_down q_down p_up contributes, but p_up q_down q_up p_down is trivially zero)
        # Also, iff p,q act on different spins, then they may act on the same spatial modes -> p_up p_down p_down p_up

        # Equal spin contribution
        """for p in range(self.S_alpha):
            for q in range(self.S_alpha):
                if p != q:
                    null_state_energy_two += 4 * self.mode_exchange_energy([p, q], [q, p])"""

        # Differing spin contribution
        """for p in range(self.S_alpha):
            for q in range(self.S_alpha):
                null_state_energy_two += 1 * self.mode_exchange_energy([p, q], [q, p])"""

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

        cisolver = fci.FCI(self.mol, MO_coefs)
        self.ci_energy, self.ci_sol = cisolver.kernel()

        print(f"  Ground state energy as calculated by SCF (full configuration) = {self.ci_energy}")


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
        print("Up-up:", upup)
        print("Down-down:", downdown)
        print("Mixed spin:", mixed)

        H_nuc = self.mol.energy_nuc() * alpha_overlap * beta_overlap
        return(H_one_term + H_two_term + H_nuc)





    def find_ground_state(self, method, **kwargs):
        if method in self.find_ground_state_methods.keys():
            return(self.find_ground_state_methods[method](self, **kwargs))
        else:
            print(f"ERROR: Unknown ground state method {method}. Available methods: {self.find_ground_state_methods.keys()}")
            return(None)





