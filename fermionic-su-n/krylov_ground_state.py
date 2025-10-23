import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from pyscf import gto, scf, cc

from class_Semaphor import Semaphor
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

    def __init__(self, ID):

        self.ID = ID

        # Semaphor
        self.semaphor = Semaphor(time_format = "%H:%M:%S")

        print("---------------------------- " + str(ID) + " -----------------------------")


    ###########################################################################
    # --------------------------- Internal methods ----------------------------
    ###########################################################################

    # ------------------------ Coherent state methods -------------------------

    def overlap(self, Z_a, Z_b):
        # Calculates {Z_a | Z_b}, the unnormalised overlap
        # Note that the norm of a vector | Z } is equal to sqrt({Z | Z})

        # Z_a, Z_b are (M-S, S)-signed matrices
        if self.M - self.S > self.S:
            # We prefer I_S + Z_a\hc.Z_b
            return(np.linalg.det(np.identity(self.S) + np.matmul(np.conjugate(Z_a.T), Z_b)))
        else:
            # We prefer I_(M-S) + Z_b.Z_a\hc
            return(np.linalg.det(np.identity(self.M - self.S) + np.matmul(Z_b, np.conjugate(Z_a.T))))

    def general_overlap(self, Z_a, Z_b, c, a):
        # c and a contain distinct values, otherwise this is trivially zero
        if (len(c) > len(set(c)) or len(a) > len(set(a))):
            return(0.0)
        varsigma_a = []
        tau_a = []
        varsigma_b = []
        tau_b = []
        tau_cup = []
        sigma_intersection = []
        for i in range(len(c)):
            # using len(c) = len(a)
            if c[i] < self.S:
                # pi_1
                if c[i] not in a:
                    varsigma_a.append(c[i])
                else:
                    sigma_intersection.append(c[i])
            else:
                # pi_0
                tau_a.append(c[i] - self.S)
                if c[i] - self.S not in tau_cup:
                    tau_cup.append(c[i] - self.S)
            if a[i] < self.S:
                # pi_1
                if a[i] not in c:
                    varsigma_b.append(a[i])
            else:
                # pi_0
                tau_b.append(a[i] - self.S)
                if a[i] - self.S not in tau_cup:
                    tau_cup.append(a[i] - self.S)
        sigma_a = varsigma_a + sigma_intersection
        sigma_b = varsigma_b + sigma_intersection
        sigma_cup = varsigma_a + varsigma_b + sigma_intersection
        const_sign = functions.sign(self.S * (len(tau_a) + len(tau_b)) + (len(varsigma_a) - 1) * (len(varsigma_b) - 1) + 1 + sum(varsigma_a) + len(varsigma_a) + sum(varsigma_b) + len(varsigma_b) + functions.eta(sigma_intersection, varsigma_a + varsigma_b))
        # Now for the normal ordering sign
        const_sign *= functions.permutation_signature(c) * functions.permutation_signature(a)
        if len(tau_a) <= len(tau_b):
            fast_M = np.zeros((len(tau_b) + self.S - len(sigma_a), len(tau_b) + self.S - len(sigma_a)), dtype=complex)
            fast_M[:len(tau_b), len(tau_a):len(tau_a)+len(varsigma_a)] = np.take(np.take(Z_b, tau_b, axis = 0), varsigma_a, axis = 1)
            fast_M[:len(tau_b), len(tau_a)+len(varsigma_a):] = np.take(functions.reduced_matrix(Z_b, [], sigma_cup), tau_b, axis = 0)

            fast_M[len(tau_b):len(tau_b) + len(varsigma_b),:len(tau_a)] = np.conjugate(np.take(np.take(Z_a, tau_a, axis = 0), varsigma_b, axis = 1).T)
            fast_M[len(tau_b) + len(varsigma_b):, :len(tau_a)] = np.conjugate(np.take(functions.reduced_matrix(Z_a, [], sigma_cup), tau_a, axis = 0).T)

            fast_M[len(tau_b):len(tau_b) + len(varsigma_b), len(tau_a):len(tau_a)+len(varsigma_a)] = np.matmul(np.conjugate(np.take(functions.reduced_matrix(Z_a, tau_cup, []), varsigma_b, axis = 1).T), np.take(functions.reduced_matrix(Z_b, tau_cup, []), varsigma_a, axis = 1))
            fast_M[len(tau_b):len(tau_b) + len(varsigma_b), len(tau_a)+len(varsigma_a):] = np.matmul(np.conjugate(np.take(functions.reduced_matrix(Z_a, tau_cup, []), varsigma_b, axis = 1).T), functions.reduced_matrix(Z_b, tau_cup, sigma_cup))
            fast_M[len(tau_b) + len(varsigma_b):, len(tau_a):len(tau_a)+len(varsigma_a)] = np.matmul(np.conjugate(functions.reduced_matrix(Z_a, tau_cup, sigma_cup).T), np.take(functions.reduced_matrix(Z_b, tau_cup, []), varsigma_a, axis = 1))
            fast_M[len(tau_b) + len(varsigma_b):, len(tau_a)+len(varsigma_a):] = np.identity(self.S - len(sigma_cup)) + np.matmul(np.conjugate(functions.reduced_matrix(Z_a, tau_cup, sigma_cup).T), functions.reduced_matrix(Z_b, tau_cup, sigma_cup))
            cb_sign = functions.sign(len(tau_b) * (1 + len(tau_b) - len(tau_a)))
            return(const_sign * cb_sign * np.linalg.det(fast_M))
        else:
            fast_M = np.zeros((self.M - self.S + len(tau_a) + len(varsigma_a) - len(tau_cup), self.M - self.S + len(tau_a) + len(varsigma_a) - len(tau_cup)), dtype=complex)
            fast_M[:len(varsigma_b), len(varsigma_a):len(varsigma_a)+len(tau_a)] = np.take(np.take(np.conjugate(Z_a).T, varsigma_b, axis=0), tau_a, axis = 1)
            fast_M[:len(varsigma_b), len(varsigma_a)+len(tau_a):] = np.take(functions.reduced_matrix(np.conjugate(Z_a).T, [], tau_cup), varsigma_b, axis=0)
            fast_M[len(varsigma_b):len(varsigma_b)+len(tau_b), :len(varsigma_a)] = np.take(np.take(Z_b, tau_b, axis=0), varsigma_a, axis = 1)
            fast_M[len(varsigma_b)+len(tau_b):, :len(varsigma_a)] = np.take(functions.reduced_matrix(Z_b, tau_cup, []), varsigma_a, axis=1)

            fast_M[len(varsigma_b):len(varsigma_b)+len(tau_b), len(varsigma_a):len(varsigma_a)+len(tau_a)] = np.matmul(np.take(functions.reduced_matrix(Z_b, [], sigma_cup), tau_b, axis = 0), np.take(functions.reduced_matrix(np.conjugate(Z_a).T, sigma_cup, []), tau_a, axis=1))
            fast_M[len(varsigma_b):len(varsigma_b)+len(tau_b), len(varsigma_a)+len(tau_a):] = np.matmul(np.take(functions.reduced_matrix(Z_b, [], sigma_cup), tau_b, axis = 0), functions.reduced_matrix(np.conjugate(Z_a).T, sigma_cup, tau_cup))
            fast_M[len(varsigma_b)+len(tau_b):, len(varsigma_a):len(varsigma_a)+len(tau_a)] = np.matmul(functions.reduced_matrix(Z_b, tau_cup, sigma_cup), np.take(functions.reduced_matrix(np.conjugate(Z_a).T, sigma_cup, []), tau_a, axis=1))
            fast_M[len(varsigma_b)+len(tau_b):, len(varsigma_a)+len(tau_a):] = np.identity(self.M - self.S - len(tau_cup)) + np.matmul(functions.reduced_matrix(Z_b, tau_cup, sigma_cup), functions.reduced_matrix(np.conjugate(Z_a).T, sigma_cup, tau_cup))
            cb_sign = functions.sign(len(varsigma_b) * (1 + len(varsigma_b) - len(varsigma_a)))
            return(const_sign * cb_sign * np.linalg.det(fast_M))

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
        #     -delta: width of the sampling distribution (normal, uncoupled ofc)
        print(f"Obtaining the ground state with the method \"random sampling\" [N = {kwargs["N"]}, lambda = {kwargs["lamb"]}, delta = {kwargs["delta"]}]")

        # We sample around the HF null guess, i.e. Z = 0
        # We include one extra basis vector - the null point itself!
        initial_vector_sample = np.zeros((1, self.M - self.S, self.S), dtype=complex)
        additional_vector_sample = np.random.normal(0.0, kwargs["delta"], (kwargs["N"], self.M - self.S, self.S))
        vector_sample = np.concatenate((initial_vector_sample, additional_vector_sample))
        N = kwargs["N"] + 1

        # Firstly we find the normalisation coefficients
        norm_coefs = np.zeros(N)
        norm_coefs[0] = 1.0
        for i in range(1, N):
            norm_coefs[i] = 1.0 / np.sqrt(self.overlap(vector_sample[i], vector_sample[i]))

        overlap_matrix = np.zeros((N, N), dtype=complex) # [a][b] = <a|b>
        for i in range(N):
            for j in range(N):
                overlap_matrix[i][j] = self.overlap(vector_sample[i], vector_sample[j]) * norm_coefs[i] * norm_coefs[j]
        print(overlap_matrix)

        # We now diagonalise on the vector sample
        H_eff = np.zeros((N, N), dtype=complex)

        msg = f"  Explicit Hamiltonian evaluation begins..."
        new_sem_ID = self.semaphor.create_event(np.linspace(0, N * (N + 1) / 2, 100 + 1), msg)

        for a in range(N):
            # Here the diagonal <Z_a|H|Z_a>
            H_eff[a][a] = self.H_overlap(vector_sample[a], vector_sample[a]) * norm_coefs[a] * norm_coefs[a] + self.mol.energy_nuc()
            self.semaphor.update(new_sem_ID, a * (a + 1) / 2)
            print(f" Diagonal term {a+1}: {H_eff[a][a]}")

            # Here the off-diagonal, using the fact that H_eff is a Hermitian matrix
            for b in range(a):
                # We explicitly calculate <Z_a | H | Z_b>
                H_eff[a][b] = self.H_overlap(vector_sample[a], vector_sample[b]) * norm_coefs[a] * norm_coefs[b] + self.mol.energy_nuc()
                H_eff[b][a] = np.conjugate(H_eff[a][b])
                self.semaphor.update(new_sem_ID, a * (a + 1) / 2 + b + 1)

        self.solution_benchmark = self.semaphor.finish_event(new_sem_ID, "    Evaluation")

        print(H_eff)

        def get_partial_sol(N_eff):
            H_eff_trimmed = H_eff[:N_eff, :N_eff]
            S_trimmed = overlap_matrix[:N_eff, :N_eff]
            energy_levels, energy_states = sp.linalg.eigh(H_eff_trimmed, S_trimmed)
            """unnorm_energy_levels, unnorm_energy_states = np.linalg.eig(H_eff_trimmed)
            # The eigen-states are linear combinations of other states with non-unitary norm, and thus their energy values have to be renormalised as well
            energy_levels = []
            energy_states = []

            for i in range(N_eff):
                cur_vector = unnorm_energy_states[:,i]
                cur_norm = 0.0
                for a in range(N_eff):
                    for b in range(N_eff):
                        cur_norm += np.conjugate(cur_vector[a]) * cur_vector[b] * overlap_matrix[a][b] * norm_coefs[a] * norm_coefs[b]
                energy_states.append(cur_vector / np.sqrt(cur_norm))
                energy_levels.append(unnorm_energy_levels[i])"""
            ground_state_index = np.argmin(energy_levels)
            return(energy_levels[ground_state_index])



        """unnorm_energy_levels, unnorm_energy_states = np.linalg.eig(H_eff)

        # The eigen-states are linear combinations of other states with non-unitary norm, and thus their energy values have to be renormalised as well

        energy_levels = []
        energy_states = []

        for i in range(N):
            cur_vector = unnorm_energy_states[:,i]
            cur_norm = 0.0
            for a in range(N):
                for b in range(N):
                    cur_norm += np.conjugate(cur_vector[a]) * cur_vector[b] * overlap_matrix[a][b]
            energy_states.append(cur_vector / cur_norm)
            energy_levels.append(unnorm_energy_levels[i] / cur_norm)


        ground_state_index = np.argmin(energy_levels)

        print(f"Ground state found with energy {energy_levels[ground_state_index]}")"""

        convergence_sols = []
        N_vals = []
        for N_eff_val in range(1, N + 1):
            N_vals.append(N_eff_val)
            convergence_sols.append(get_partial_sol(N_eff_val))
        plt.plot(N_vals, convergence_sols, "x")
        plt.show()


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
        print(f"    The atomic orbitals (modes) are ordered as follows: {self.mol.ao_labels()}")
        print(self.element_to_basis)
        print(gto.charge("O"))

        print("  Calculating 1e integrals...")
        self.H_one = self.mol.intor('int1e_kin', hermi = 1) + self.mol.intor('int1e_nuc', hermi = 1)

        print("  Calculating 2e integrals...")
        #self.H_two = self.mol.intor('int2e', aosym = "s1")
        exchange_integrals_mulliken = self.mol.intor('int2e', aosym = "s1")
        # <ij|g|kl> = (ik|jl)
        # slater-condon: <psi|g|psi> = 1/2 sum_i sum_j (<ij|g|ij> - <ij|g|ji>)
        self.H_two = exchange_integrals_mulliken - exchange_integrals_mulliken.transpose(0,2,1,3)

        print("  Creating canonical order for spin orbitals...")

        print("    Initial guess: molecule-independent atomic orbitals")
        # For each atom, its basis functions are ordered by self-energy from
        # lowest to highest (doubling each one to account for spin degeneracy),
        # and then the first Z modes are considered as occupied, where Z is the
        # atom's proton number. The occupied and unoccupied modes among all
        # atoms are meshed together into pi_1 and pi_0 and ordered by energy.

        pi_1_provisional = [] # every element is [self-energy, orbital index, spin (0 or 1)]
        pi_0_provisional = []

        for atom_i in range(len(self.mol.elements)):
            proton_number = gto.charge(self.mol.elements[atom_i])
            ao_indices = []
            for i in range(self.element_to_basis[atom_i][1], self.element_to_basis[atom_i][2], 1):
                ao_indices.append([self.H_one[i][i], i])
            ao_indices.sort(key = lambda x : x[0])
            unallocated_charges = proton_number
            cur_ao_index = 0
            while(unallocated_charges > 0):
                if unallocated_charges > 1:
                    pi_1_provisional += [ao_indices[cur_ao_index] + [0], ao_indices[cur_ao_index] + [1]]
                    unallocated_charges -= 2
                else:
                    pi_1_provisional += [ao_indices[cur_ao_index] + [0]] # By default we have the spin alpha state occupied
                    pi_0_provisional += [ao_indices[cur_ao_index] + [1]]
                    unallocated_charges -= 1
                cur_ao_index += 1
            while(cur_ao_index < len(ao_indices)):
                pi_0_provisional += [ao_indices[cur_ao_index] + [0], ao_indices[cur_ao_index] + [1]]
                cur_ao_index += 1

        pi_1_provisional.sort(key = lambda x : x[0])
        pi_0_provisional.sort(key = lambda x : x[0])

        self.modes = [] # [mode index] = [ao index, spin]
        for i in range(len(pi_1_provisional)):
            self.modes.append(pi_1_provisional[i][1:3])
        for i in range(len(pi_0_provisional)):
            self.modes.append(pi_0_provisional[i][1:3])

        print(self.modes)


        self.H_two_full = np.zeros((self.M, self.M, self.M, self.M), dtype = complex) # [mode p][mode q][mode r][mode s] = <ij|g|ij> - <ij|g|ji>
        for p in range(self.M):
            for q in range(self.M):
                for r in range(self.M):
                    for s in range(self.M):
                        # for each, the props are self.modes[p] = [p_ao, p_spin] etc
                        # We require p_spin = s_spin and q_spin = r_spin
                        if set([self.modes[p][1], self.modes[q][1]]) == set([self.modes[r][1], self.modes[s][1]]):
                            self.H_two_full[p][q][r][s] = exchange_integrals_mulliken[self.modes[p][0]][self.modes[s][0]][self.modes[q][0]][self.modes[r][0]]

        null_state_energy = 0.0
        null_state_energy_one = 0.0
        for p in range(self.S):
            # Only occupied states contribute
            null_state_energy_one += self.mode_exchange_energy([p], [p])
        print(f" Null state energy one = {null_state_energy_one}")
        c_pairs = functions.subset_indices(np.arange(self.S), 2)
        null_state_energy_two = 0.0
        for p in range(self.S):
            for q in range(self.S):
                #core_term = self.mode_exchange_energy([p, p], [q, q])
                #core_term =
                #print(core_term)
                #null_state_energy += core_term

                """ao_p = self.modes[p][0]
                spin_p = self.modes[p][1]
                ao_q = self.modes[q][0]
                spin_q = self.modes[q][1]

                #core_term = exchange_integrals_mulliken[ao_p][ao_p][ao_q][ao_q]
                #if spin_p == spin_q:
                #    core_term -= exchange_integrals_mulliken[ao_p][ao_q][ao_q][ao_p]
                core_term = self.H_two[ao_p][ao_p][ao_q][ao_q]
                null_state_energy += core_term"""
                #print(f"{p}, {q}: {core_term}")

                if p == q:
                    # repeated spin-orbital creation operator is zero
                    continue

                null_state_energy_two += 0.5 * self.H_two_full[q][p][p][q]
                #print(p, q, 0.5 * self.H_two_full[q][p][p][q])

        print(f" Null state energy two = {null_state_energy_two}")

        null_state_energy = null_state_energy_one + null_state_energy_two + self.mol.energy_nuc()
        print(f"  Energy of the null state = {null_state_energy}")
        print(f"  Energy of the null state with the overlap method = {self.H_overlap(np.zeros((self.M - self.S, self.S), dtype=complex), np.zeros((self.M - self.S, self.S), dtype=complex)) + self.mol.energy_nuc()}")

        # We print the diagonal elements of H_one
        print("  Occupied:")
        for p in range(self.S):
            print(f"    H_one_{p},{p} = {self.mode_exchange_energy([p], [p])}")
        print("  Unoccupied:")
        for p in range(self.S, self.M):
            print(f"    H_one_{p},{p} = {self.mode_exchange_energy([p], [p])}")

    def get_H_two_element(self, p, q, r, s):
        return(0.5 * self.H_two[p][q][r][s])
        #return(self.H_two[int(p * (p * p * p + 2 * p * p + 3 * p + 2) / 8 + p * q * (p + 1) / 2 + q * (q + 1) / 2   + r * (r + 1) / 2 + s )])

    def mode_exchange_energy(self, m_i, m_f, debug = False):
        # m_i/f are lists of either one mode index (single electron exchange) or two mode indices (two electron exchange)
        # this function translates mode index exchanges into ao orbital exchanges, which are known
        if len(m_i) == 1:
            if self.modes[m_i[0]][1] == self.modes[m_f[0]][1]:
                return(self.H_one[self.modes[m_i[0]][0]][self.modes[m_f[0]][0]])
        elif len(m_i) == 2:
            if self.modes[m_i[0]][1] == self.modes[m_f[1]][1] and self.modes[m_i[1]][1] == self.modes[m_f[0]][1]:
                return(self.H_two_full[m_i[0]][m_i[1]][m_f[0]][m_f[1]])
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

    def H_overlap(self, Z_a, Z_b):
        # This method only uses the instance's H_one, H_two, to calculate <Z_a | H | Z_b>
        # Z_a, Z_b are (M-S, S)-signed matrices
        H_one_term = 0.0
        # This is a sum over all mode pairs
        for p in range(self.M):
            for q in range(self.M):
                H_one_term += self.mode_exchange_energy([p], [q]) * self.general_overlap(Z_a, Z_b, [p], [q])
        #print(f" H_one = {H_one_term}")

        H_two_term = 0.0
        # This is a sum over pairs of strictly ascending mode pairs (for other cases we can use symmetry, which just becomes an extra factor here)
        c_pairs = functions.subset_indices(np.arange(self.M), 2)
        a_pairs = functions.subset_indices(np.arange(self.M), 2)
        for c_pair in c_pairs:
            for a_pair in a_pairs:
                # c_pair = [q, p] (inverted order!)
                # a_pair = [r, s]
                core_term = self.mode_exchange_energy([c_pair[1], c_pair[0]], a_pair) * self.general_overlap(Z_a, Z_b, c_pair, a_pair)
                #print(self.general_overlap(Z_a, Z_b, c_pair, a_pair))
                # an extra contribution is from the 4 different order swaps, which yields 2(core_term + core_term*), however, we also have a pre-factor of 0.5
                H_two_term += (core_term)
                """if np.round(np.abs(core_term), 2) != 0:
                    self.mode_exchange_energy([c_pair[1], c_pair[0]], a_pair, debug = True)
                    print(f" {[c_pair[1], c_pair[0]], a_pair} = {core_term}")"""
        #print(H_two_term)
        #print(f" H_two = {H_two_term}")
        return(H_one_term + H_two_term) #TODO the mulliken -> physicist notation conversion is incomplete, and that's why it is not yet antisymmetrised. SUBTRACT THE DIAGONAL TERM (NOT TRUE)





    def find_ground_state(self, method, **kwargs):
        if method in self.find_ground_state_methods.keys():
            self.find_ground_state_methods[method](self, **kwargs)
        else:
            print(f"ERROR: Unknown ground state method {method}. Available methods: {self.find_ground_state_methods.keys()}")





