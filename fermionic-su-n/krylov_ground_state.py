import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from pyscf import gto, scf, cc, ao2mo, fci

from coherent_states.CS_Thouless import CS_Thouless
from coherent_states.CS_Qubit import CS_Qubit

from class_Semaphor import Semaphor
from class_DSM import DSM
import functions


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

        self.user_log += f"find_ground_state_sampling [N = {kwargs["N"]}, lambda = {kwargs["lamb"]}, sampling method = {kwargs["sampling_method"]}]\n"
        procedure_diagnostic = []


        print(f"Obtaining the ground state with the method \"random sampling\" [CS types = {kwargs["CS"]}, N = {kwargs["N"]}, lambda = {kwargs["lamb"]}, sampling method = {kwargs["sampling_method"]}]")

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

        CS_sample = [ground_state_solver.coherent_state_types[kwargs["CS"]].null_state(self.M, self.S)]
        for i in range(kwargs["N"]):
            CS_sample.append(ground_state_solver.coherent_state_types[kwargs["CS"]].random_state(self.M, self.S, kwargs["sampling_method"]))
            #CS_sample.append(ground_state_solver.coherent_state_types[kwargs["CS"]](self.M, self.S, cs_null_param + vector_sample[i] * kwargs["weights"]))

        basis_samples_bulk = []
        for i in range(N):
            basis_samples_bulk.append(CS_sample[i].z)
        self.disk_jockey.commit_datum_bulk("basis_samples", basis_samples_bulk)

        # Firstly we find the normalisation coefficients
        norm_coefs = np.zeros(N)
        norm_coefs[0] = 1.0
        for i in range(1, N):
            sig, mag = CS_sample[i].log_norm_coef
            if (sig * np.exp(-mag / 2.0)).imag > 1e-04:
                procedure_diagnostic.append(f"basis norm not real!!")
            norm_coefs[i] = (sig * np.exp(-mag / 2.0)).real

        print("-- Normalisation coefs:", norm_coefs)

        procedure_diagnostic.append(f"basis norm max/min = {np.max(norm_coefs) / np.min(norm_coefs)}")

        overlap_matrix = np.zeros((N, N), dtype=complex) # [a][b] = <a|b>
        for i in range(N):
            for j in range(N):
                overlap_matrix[i][j] = CS_sample[i].norm_overlap(CS_sample[j])
        print("-- Overlap matrix:")
        print(overlap_matrix)

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

        max_overlap_diagnostic = 0.0
        for a in range(N):
            # Here the diagonal <Z_a|H|Z_a>
            cur_H_overlap, overlap_diagnostic = self.H_overlap(CS_sample[a], CS_sample[a], True)
            H_eff[a][a] = cur_H_overlap
            if overlap_diagnostic > max_overlap_diagnostic:
                max_overlap_diagnostic = overlap_diagnostic
            self.semaphor.update(new_sem_ID, a * (a + 1) / 2)

            # Here the off-diagonal, using the fact that H_eff is a Hermitian matrix
            for b in range(a):
                # We explicitly calculate <Z_a | H | Z_b>
                cur_H_overlap, overlap_diagnostic = self.H_overlap(CS_sample[a], CS_sample[b], True)
                H_eff[a][b] = cur_H_overlap
                H_eff[b][a] = np.conjugate(H_eff[a][b])
                if overlap_diagnostic > max_overlap_diagnostic:
                    max_overlap_diagnostic = overlap_diagnostic
                self.semaphor.update(new_sem_ID, a * (a + 1) / 2 + b + 1)

        procedure_diagnostic.append(f"overlap max update diagnostic = {max_overlap_diagnostic}")

        self.solution_benchmark = self.semaphor.finish_event(new_sem_ID, "    Evaluation")

        # H_eff diagonal terms
        for i in range(N):
            print(f"  Z_{i} self-energy: {H_eff[i][i]+self.mol.energy_nuc()}")
            if H_eff[i][i]+self.mol.energy_nuc() < self.ci_energy:
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

            return(energy_levels[ground_state_index] + self.mol.energy_nuc())

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

        # We create helpful dictionaries to describe the molecule
        self.element_to_basis = [] # [element index] = [type of element, AO index start, AO index end (non-inclusive)]
        cur_aoslice = self.mol.aoslice_by_atom()
        for i in range(len(self.mol.elements)):
            self.element_to_basis.append([self.mol.elements[i], int(cur_aoslice[i][2]), int(cur_aoslice[i][3])])

        print(f"    There are {self.mol.nao} atomic orbitals, each able to hold 2 electrons of opposing spin.")
        print(f"    The molecule is occupied by {self.mol.tot_electrons()} electrons in total.")
        print(f"    The molecule consists of the following atoms: {self.mol.elements}")
        print(f"    The atomic orbitals are ordered as follows: {self.mol.ao_labels()}")
        print(self.element_to_basis)
        print(gto.charge("O"))

        print("  Calculating 1e integrals...")
        AO_H_one = self.mol.intor('int1e_kin', hermi = 1) + self.mol.intor('int1e_nuc', hermi = 1)

        print("  Calculating 2e integrals...")
        #self.H_two = self.mol.intor('int2e', aosym = "s1")
        AO_H_two = self.mol.intor('int2e', aosym = "s1")
        # <ij|g|kl> = (ik|jl)
        # slater-condon: <psi|g|psi> = 1/2 sum_i sum_j (<ij|g|ij> - <ij|g|ji>)

        print("  Finding the molecular orbitals using mean-field approximations...")
        mean_field = scf.RHF(mol).run()
        MO_coefs = mean_field.mo_coeff

        print("  Transforming 1e and 2e integrals to MO basis...")
        self.MO_H_one = np.matmul(MO_coefs.T, np.matmul(AO_H_one, MO_coefs))
        MO_H_two_packed = ao2mo.kernel(self.mol, MO_coefs)
        MO_H_two_no_spin = ao2mo.restore(1, MO_H_two_packed, MO_coefs.shape[1]) # In mulliken notation: MO_H_two[p][q][r][s] = (pq|rs)


        # Let's try ordering the MOs by their 1e self-energy and occupy the lowest S ones
        MO_self_energies = [] # [[energy of MO 0, 0], [energy of MO 1, 1]...]
        for i in range(self.mol.nao):
            MO_self_energies.append([self.MO_H_one[i][i], i])

        MO_self_energies.sort(key = lambda x : x[0])

        # Now we occupy the lowest S guys
        self.modes = [] # [mode index] = [mo_index, spin]def
        for i in range(self.mol.nao):
            self.modes.append([MO_self_energies[i][1], 0])
            self.modes.append([MO_self_energies[i][1], 1])

        self.MO_H_two = np.zeros((self.M, self.M, self.M, self.M), dtype = complex)
        for p in range(self.mol.nao):
            for q in range(self.mol.nao):
                for r in range(self.mol.nao):
                    for s in range(self.mol.nao):
                        corresponding_integral = MO_H_two_no_spin[MO_self_energies[p][1]][MO_self_energies[q][1]][MO_self_energies[r][1]][MO_self_energies[s][1]]
                        self.MO_H_two[p * 2    ][r * 2    ][q * 2    ][s * 2    ] = corresponding_integral
                        self.MO_H_two[p * 2 + 1][r * 2    ][q * 2 + 1][s * 2    ] = corresponding_integral
                        self.MO_H_two[p * 2    ][r * 2 + 1][q * 2    ][s * 2 + 1] = corresponding_integral
                        self.MO_H_two[p * 2 + 1][r * 2 + 1][q * 2 + 1][s * 2 + 1] = corresponding_integral
        self.MO_H_two = self.MO_H_two.transpose(0, 1, 3, 2) - self.MO_H_two # Slater-Condon antisymmetrisation
        """def spatial_idx(P):
            return MO_self_energies[P // 2][1]
        def spin_idx(P):
            return P % 2

        # build antisymmetrized spin-orbital integrals:
        for P in range(self.M):
            p = spatial_idx(P); sp = spin_idx(P)
            for Q in range(self.M):
                q = spatial_idx(Q); sq = spin_idx(Q)
                for R in range(self.M):
                    r = spatial_idx(R); sr = spin_idx(R)
                    for S in range(self.M):
                        s = spatial_idx(S); ss = spin_idx(S)

                        val = 0.0
                        # (pq|rs) term if spins match P<->R and Q<->S
                        if sp == sr and sq == ss:
                            val += MO_H_two_no_spin[p][q][r][s]
                        # subtract (pq|sr) term if spins match P<->S and Q<->R
                        if sp == ss and sq == sr:
                            val -= MO_H_two_no_spin[p][q][s][r]

                        self.MO_H_two[P, Q, R, S] = val"""


        print("    Antisym tests")
        print("      Hermitian?", (np.round(self.MO_H_two - self.MO_H_two.transpose(2, 3, 0, 1), 2) == 0).all())
        print("      Antisym on first pair", (np.round(self.MO_H_two + self.MO_H_two.transpose(1, 0, 2, 3), 2) == 0).all())
        print("      Antisym on second pair", (np.round(self.MO_H_two + self.MO_H_two.transpose(0, 1, 3, 2), 2) == 0).all())

        """print("Occupied MO energies:")
        for i in range(self.S):
            print(self.MO_H_one[self.modes[i][0]][self.modes[i][0]])
        print("Unoccupied MO energies:")
        for i in range(self.S, self.M):
            print(self.MO_H_one[self.modes[i][0]][self.modes[i][0]])"""


        null_state_energy = 0.0
        null_state_energy_one = 0.0
        for p in range(self.S):
            # Only occupied states contribute
            null_state_energy_one += self.mode_exchange_energy([p], [p])
        print(f" Null state energy one = {null_state_energy_one}")
        null_state_energy_two = 0.0
        for p in range(self.S):
            for q in range(self.S):
                null_state_energy_two += 0.5 * self.mode_exchange_energy([p, q], [q, p])
                #null_state_energy_two += 0.5 * (self.MO_H_two[p][q][p][q] - self.MO_H_two[p][q][q][p])

        print(f" Null state energy two = {null_state_energy_two}")

        null_state_energy = null_state_energy_one + null_state_energy_two + self.mol.energy_nuc()
        print(f"  Energy of the null state = {null_state_energy}")
        null_state = CS_Thouless(self.M, self.S, np.zeros((self.M - self.S, self.S), dtype=complex))
        null_state_direct_self_energy = self.H_overlap(null_state, null_state)
        print(f"  Energy of the null state with the overlap method = {null_state_direct_self_energy + self.mol.energy_nuc()}")

        hf = scf.RHF(self.mol)          # Create the HF object
        hf.kernel()  # Perform the SCF calculation
        cisolver = fci.FCI(mol, hf.mo_coeff)
        self.ci_energy, _ = cisolver.kernel()


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
            cur_state = ground_state_solver.coherent_state_types[state_type].random_state(self.M, self.S, sampling_method)
            if inclusion(cur_state):
                print("Success!")
                print(f"State found with z = {repr(cur_state.z)}")
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
        return(0.5 * self.H_two[p][q][r][s])
        #return(self.H_two[int(p * (p * p * p + 2 * p * p + 3 * p + 2) / 8 + p * q * (p + 1) / 2 + q * (q + 1) / 2   + r * (r + 1) / 2 + s )])

    def mode_exchange_energy(self, m_i, m_f, debug = False):
        # m_i/f are lists of either one mode index (single electron exchange) or two mode indices (two electron exchange)
        # this function translates mode index exchanges into mo orbital exchanges, which are known
        if len(m_i) == 1:
            #if self.modes[m_i[0]][1] == self.modes[m_f[0]][1]:
            return(self.MO_H_one[self.modes[m_i[0]][0]][self.modes[m_f[0]][0]])
        elif len(m_i) == 2:
            #if self.modes[m_i[0]][1] == self.modes[m_f[1]][1] and self.modes[m_i[1]][1] == self.modes[m_f[0]][1]:
            return(self.MO_H_two[m_i[0]][m_i[1]][m_f[0]][m_f[1]])
            physicist_p = self.modes[m_i[0]][0]
            physicist_q = self.modes[m_i[1]][0]
            physicist_r = self.modes[m_f[0]][0]
            physicist_s = self.modes[m_f[1]][0]
            # We translate from Mulliken into physicist's notation using the equation (PQ|RS)=<PR|QS>
            p = physicist_p
            q = physicist_q
            r = physicist_r
            s = physicist_s

            if debug:
                print(p, q, r, s)

            spin_p = self.modes[m_i[0]][1]
            spin_q = self.modes[m_i[1]][1]
            spin_r = self.modes[m_f[0]][1]
            spin_s = self.modes[m_f[1]][1]
            # Here p,q,r,s are in the Mulliken notation, and thus we can use the standard ordering to access the symmetrised AO integral
            if spin_p == spin_s and spin_q == spin_r:
                return(self.H_two_full[p][q][r][s])
                #return(self.get_H_two_element(p, q, r, s))
        return(0.0)

    def H_overlap(self, state_a, state_b, include_diagnostic = False):
        # This method only uses the instance's H_one, H_two, to calculate <Z_a | H | Z_b>
        # state_a, state_b are instances of any class inheriting from CS_Base

        # Firstly we prepare update information
        master_matrix_det, master_matrix_inv, master_matrix_alt_inv, overlap_diagnostic = state_a.get_update_information(state_b)

        H_one_term = 0.0
        # This is a sum over all mode pairs
        for p in range(self.M):
            for q in range(self.M):
                #H_one_term += self.mode_exchange_energy([p], [q]) * state_a.overlap_update(state_b, [p], [q], master_matrix_det, master_matrix_inv, master_matrix_alt_inv) #self.general_overlap(Z_a, Z_b, [p], [q])
                H_one_term += self.mode_exchange_energy([p], [q]) * state_a.norm_overlap(state_b, [p], [q])
        #print(f" H_one = {H_one_term}")

        H_two_term = 0.0
        # This is a sum over pairs of strictly ascending mode pairs (for other cases we can use symmetry, which just becomes an extra factor here)
        c_pairs = functions.subset_indices(np.arange(self.M), 2)
        a_pairs = functions.subset_indices(np.arange(self.M), 2)
        for c_pair in c_pairs:
            for a_pair in a_pairs:
                # c_pair = [q, p] (inverted order!)
                # a_pair = [r, s]
                #core_term = self.mode_exchange_energy([c_pair[1], c_pair[0]], a_pair) * state_a.overlap_update(state_b, c_pair, a_pair, master_matrix_det, master_matrix_inv, master_matrix_alt_inv) #self.general_overlap(Z_a, Z_b, c_pair, a_pair)
                core_term = self.mode_exchange_energy([c_pair[1], c_pair[0]], a_pair) * state_a.norm_overlap(state_b, c_pair, a_pair)
                #print(self.general_overlap(Z_a, Z_b, c_pair, a_pair))
                # an extra contribution is from the 4 different order swaps, which yields 2(core_term + core_term*), however, we also have a pre-factor of 0.5
                H_two_term += (core_term)
                """if np.round(np.abs(core_term), 2) != 0:
                    self.mode_exchange_energy([c_pair[1], c_pair[0]], a_pair, debug = True)
                    print(f" {[c_pair[1], c_pair[0]], a_pair} = {core_term}")"""
        #print(H_two_term)
        #print(f" H_two = {H_two_term}")
        if include_diagnostic:
            return(H_one_term + H_two_term, overlap_diagnostic) #TODO the mulliken -> physicist notation conversion is incomplete, and that's why it is not yet antisymmetrised. SUBTRACT THE DIAGONAL TERM (NOT TRUE)
        return(H_one_term + H_two_term)





    def find_ground_state(self, method, **kwargs):
        if method in self.find_ground_state_methods.keys():
            return(self.find_ground_state_methods[method](self, **kwargs))
        else:
            print(f"ERROR: Unknown ground state method {method}. Available methods: {self.find_ground_state_methods.keys()}")
            return(None)





