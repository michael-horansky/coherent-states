import numpy as np

from pyscf import gto, scf, cc

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

        print("---------------------------- " + str(ID) + " -----------------------------")


    ###########################################################################
    # --------------------------- Internal methods ----------------------------
    ###########################################################################

    # ------------------------ Coherent state methods -------------------------

    def general_overlap(self, Z_a, Z_b, c, a):
        # assumes c and a are both ascending. That's the only assumption. I'll add the perm sign soon, then no assumptions will be made (except for non-repeated indices i guess? easy to test for tbh)
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
        if len(tau_a) <= len(tau_b):
            fast_M = np.zeros((len(tau_b) + self.S - len(sigma_a), len(tau_b) + self.S - len(sigma_a)), dtype=complex)
            fast_M[:len(tau_b), len(tau_a):len(tau_a)+len(varsigma_a)] = np.take(np.take(Z_b, tau_b, axis = 0), varsigma_a, axis = 1)
            fast_M[:len(tau_b), len(tau_a)+len(varsigma_a):] = np.take(reduced_matrix(Z_b, [], sigma_cup), tau_b, axis = 0)

            fast_M[len(tau_b):len(tau_b) + len(varsigma_b),:len(tau_a)] = np.conjugate(np.take(np.take(Z_a, tau_a, axis = 0), varsigma_b, axis = 1).T)
            fast_M[len(tau_b) + len(varsigma_b):, :len(tau_a)] = np.conjugate(np.take(reduced_matrix(Z_a, [], sigma_cup), tau_a, axis = 0).T)

            fast_M[len(tau_b):len(tau_b) + len(varsigma_b), len(tau_a):len(tau_a)+len(varsigma_a)] = np.matmul(np.conjugate(np.take(reduced_matrix(Z_a, tau_cup, []), varsigma_b, axis = 1).T), np.take(reduced_matrix(Z_b, tau_cup, []), varsigma_a, axis = 1))
            fast_M[len(tau_b):len(tau_b) + len(varsigma_b), len(tau_a)+len(varsigma_a):] = np.matmul(np.conjugate(np.take(reduced_matrix(Z_a, tau_cup, []), varsigma_b, axis = 1).T), reduced_matrix(Z_b, tau_cup, sigma_cup))
            fast_M[len(tau_b) + len(varsigma_b):, len(tau_a):len(tau_a)+len(varsigma_a)] = np.matmul(np.conjugate(reduced_matrix(Z_a, tau_cup, sigma_cup).T), np.take(reduced_matrix(Z_b, tau_cup, []), varsigma_a, axis = 1))
            fast_M[len(tau_b) + len(varsigma_b):, len(tau_a)+len(varsigma_a):] = np.identity(self.S - len(sigma_cup)) + np.matmul(np.conjugate(reduced_matrix(Z_a, tau_cup, sigma_cup).T), reduced_matrix(Z_b, tau_cup, sigma_cup))
            cb_sign = functions.sign(len(tau_b) * (1 + len(tau_b) - len(tau_a)))
            return(const_sign * cb_sign * np.linalg.det(fast_M))
        else:
            fast_M = np.zeros((self.M - self.S + len(tau_a) + len(varsigma_a) - len(tau_cup), self.M - self.S + len(tau_a) + len(varsigma_a) - len(tau_cup)), dtype=complex)
            fast_M[:len(varsigma_b), len(varsigma_a):len(varsigma_a)+len(tau_a)] = np.take(np.take(np.conjugate(Z_a).T, varsigma_b, axis=0), tau_a, axis = 1)
            fast_M[:len(varsigma_b), len(varsigma_a)+len(tau_a):] = np.take(reduced_matrix(np.conjugate(Z_a).T, [], tau_cup), varsigma_b, axis=0)
            fast_M[len(varsigma_b):len(varsigma_b)+len(tau_b), :len(varsigma_a)] = np.take(np.take(Z_b, tau_b, axis=0), varsigma_a, axis = 1)
            fast_M[len(varsigma_b)+len(tau_b):, :len(varsigma_a)] = np.take(reduced_matrix(Z_b, tau_cup, []), varsigma_a, axis=1)

            fast_M[len(varsigma_b):len(varsigma_b)+len(tau_b), len(varsigma_a):len(varsigma_a)+len(tau_a)] = np.matmul(np.take(reduced_matrix(Z_b, [], sigma_cup), tau_b, axis = 0), np.take(reduced_matrix(np.conjugate(Z_a).T, sigma_cup, []), tau_a, axis=1))
            fast_M[len(varsigma_b):len(varsigma_b)+len(tau_b), len(varsigma_a)+len(tau_a):] = np.matmul(np.take(reduced_matrix(Z_b, [], sigma_cup), tau_b, axis = 0), reduced_matrix(np.conjugate(Z_a).T, sigma_cup, tau_cup))
            fast_M[len(varsigma_b)+len(tau_b):, len(varsigma_a):len(varsigma_a)+len(tau_a)] = np.matmul(reduced_matrix(Z_b, tau_cup, sigma_cup), np.take(reduced_matrix(np.conjugate(Z_a).T, sigma_cup, []), tau_a, axis=1))
            fast_M[len(varsigma_b)+len(tau_b):, len(varsigma_a)+len(tau_a):] = np.identity(self.M - self.S - len(tau_cup)) + np.matmul(reduced_matrix(Z_b, tau_cup, sigma_cup), reduced_matrix(np.conjugate(Z_a).T, sigma_cup, tau_cup))
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

    def Q_dot(self, t, Q, reg_timescale = -1):

        # Full-variational method

        # First, we decode the current Q-array
        cur_A, cur_Z = self.decode_system(Q)
        N = len(cur_A)

        # Now we find all zeroth, first, and second-order transition mels
        X_zero = np.zeros((N, N), dtype=complex) # [n_a][n_b]
        X_one = np.zeros((self.M - self.S, self.S, N, N), dtype=complex) # [i][j][n_a][n_b]
        X_two = np.zeros((self.M - self.S, self.S, self.M - self.S, self.S,, N, N), dtype=complex) # [i][j][i'][j'][n_a][n_b]

        for n_a in range(N):
            for n_b in range(N):
                # zeroth-order overlap
                X_zero[n_a][n_b] = self.general_overlap(cur_Z[n_a], cur_Z[n_b], [], [])

                for j_a in range(self.S):
                    for j_b
                # first-order overlap




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

        print(f"    There are {self.mol.nao} atomic orbitals (i.e. modes), each able to hold 2 electrons of opposing spin.")
        print(f"    The molecule is occupied by {self.mol.tot_electrons()} electrons in total.")
        print(f"    The molecule consists of the following atoms: {self.mol.elements}")
        print(f"    The atomic orbitals (modes) are ordered as follows: {self.mol.ao_labels()}")
        print(self.element_to_basis)
        print(gto.charge("O"))

        print("  Calculating 1e integrals...")
        self.H_one = self.mol.intor('int1e_kin', hermi = 1) + self.mol.intor('int1e_nuc', hermi = 1)

        print("  Calculating 2e integrals...")
        self.H_two = self.mol.intor('int2e', aosym = "s8")

        print("  Creating canonical order for atomic orbitals...")

        print("    Initial guess: molecule-independent atomic orbitals")
        # For each atom, its basis functions are ordered by self-energy from
        # lowest to highest (doubling each one to account for spin degeneracy),
        # and then the first Z modes are considered as occupied, where Z is the
        # atom's proton number. The occupied and unoccupied modes among all
        # atoms are meshed together into pi_1 and pi_0 and ordered by energy.

        pi_1_provisional = []
        pi_0_provisional = []

        for atom_i in range(len(self.mol.elements)):
            proton_number = gto.charge(self.mol.elements[atom_i])
            ao_indices = []
            for i in range(self.element_to_basis[atom_i][1], self.element_to_basis[atom_i][2], 1):
                ao_indices.append([i, self.H_one[i][i]])
            ao_indices.sort(key = lambda x : x[1])
            unallocated_charges = proton_number
            cur_ao_index = 0
            while(unallocated_charges > 0):
                if unallocated_charges > 1:
                    pi_1_provisional += [ao_indices[cur_ao_index], ao_indices[cur_ao_index]]
                    unallocated_charges -= 2
                else:
                    pi_1_provisional += [ao_indices[cur_ao_index]]
                    pi_0_provisional += [ao_indices[cur_ao_index]]
                    unallocated_charges -= 1
                cur_ao_index += 1
            while(cur_ao_index < len(ao_indices)):
                pi_0_provisional += [ao_indices[cur_ao_index], ao_indices[cur_ao_index]]
                cur_ao_index += 1

        pi_1_provisional.sort(key = lambda x : x[1])
        pi_0_provisional.sort(key = lambda x : x[1])

        self.modes = [] # [mode index] = ao index
        for i in range(len(pi_1_provisional)):
            self.modes.append(pi_1_provisional[i][0])
        for i in range(len(pi_0_provisional)):
            self.modes.append(pi_0_provisional[i][0])

        print(self.modes)





    def find_ground_state(self, method, **kwargs):
        if method in self.find_ground_state_methods.keys():
            self.find_ground_state_methods[method](self, **kwargs)
        else:
            print(f"ERROR: Unknown ground state method {method}. Available methods: {self.find_ground_state_methods.keys()}")





