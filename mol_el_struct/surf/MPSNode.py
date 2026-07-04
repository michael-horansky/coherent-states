# -----------------------------------------------------------------------------
# ------------------------------- class MPSNode -------------------------------
# -----------------------------------------------------------------------------
# MPSNode (molecular potential surface node) specifies a particular realisation
# of a molecular blueprint for a particular molecular geometry.
# It possesses the mean-field properties:
#   -Hartree-Fock method (restricted/unrestricted)
#   -Molecular orbital coefficients
#   -Molecular orbital one-electron exchange integrals
#   -Molecular orbital two-electron exchange integrals
# These can be initialised by calculation or manually.
# MPSNode also possesses an FCI calculation method, which calculates the first
# few lowest-energy Hamiltonian eigenstates and their corresponding energies,
# represented in the full occupancy basis.

import math
import numpy as np

from pyscf import scf, ao2mo, fci

class MPSNode():
    # Molecular potential surface node
    # Corresponds to one geometry point of one molecule blueprint, stores the mean field properties there as well as calculated eigenstates

    def __init__(self, mol_bp, g):
        # mol_bp is an instance of MBlueprint
        # g is an instance of MGeometry
        self.mol_bp = mol_bp
        self.g = g

        self.HF_known = False # FCI cannot be run and H overlap integrals cannot be accessed unless HF is known

    # -------------------------------------------------------------------------
    # --------- Methods to obtain the MOs and the exchange integrals ----------
    # -------------------------------------------------------------------------

    def get_MO_coefs(self, spin = None, i = None):
        # if i is None, we're accessing the entire array
        if self.HF_method == "RHF":
            if i is None:
                return(self.MO_coefs)
            else:
                return(self.MO_coefs[i])
        if self.HF_method == "UHF":
            assert spin is not None
            if i is None:
                return(self.MO_coefs[spin])
            else:
                return(self.MO_coefs[spin][i])

    def get_MO_H_one(self, spin, i, j):
        if self.HF_method == "RHF":
            return(self.MO_H_one[i][j])
        if self.HF_method == "UHF":
            return(self.MO_H_one[spin][i][j])

    def get_MO_H_two(self, spin, i, j, k, l):
        if self.HF_method == "RHF":
            return(self.MO_H_two[i][j][k][l])
        if self.HF_method == "UHF":
            return(self.MO_H_two[spin][i][j][k][l])


    # ------------------------- Hartree-Fock methods --------------------------

    def run_HF(self, HF_method = "default"):

        self.mol = self.mol_bp.get_gto_Mole(self.g)

        self.N_orb = self.mol.nao
        nalpha, nbeta = self.mol.nelec
        self.S_alpha = nalpha
        self.S_beta = nbeta

        self.E_nuc = self.mol.energy_nuc()

        if HF_method == "default":
            if self.mol.spin == 0:
                self.HF_method = "RHF"
            else:
                self.HF_method = "UHF"
        else:
            self.HF_method = HF_method

        AO_H_one = self.mol.intor('int1e_kin', hermi = 1) + self.mol.intor('int1e_nuc', hermi = 1)
        AO_H_two_chemist = self.mol.intor('int2e')

        """
        self.MO_coefs = {}
        self.MO_H_one = {}
        self.MO_H_two = {}"""

        # MO_H_two has three elements:
        #   ["a"]_ijkl = <i,a j,a | k,a l,a>
        #   ["b"]_ijkl = <i,b j,b | k,b l,b>
        #   ["ab"]_ijkl = <i,a j,b | k,a l,b>
        #   The second spin-mixed term is obtained by a double transpose; ["ba"]_ijkl = ["ab"]_jilk

        """if self.HF_method == "RHF":
            # Everything is the same in both subspaces
            self.log.write("Finding the molecular orbitals using mean-field approximations...", 1)
            mean_field = scf.RHF(self.mol).run(verbose = 0)

            self.MO_coefs["a"] = mean_field.mo_coeff
            self.MO_coefs["b"] = self.MO_coefs["a"]
            self.reference_state_energy = mean_field.e_tot

            self.MO_H_one["a"] = np.matmul(self.MO_coefs["a"].T, np.matmul(AO_H_one, self.MO_coefs["a"]))
            self.MO_H_one["b"] = self.MO_H_one["a"]
            MO_H_two_packed = ao2mo.kernel(AO_H_two_chemist, self.MO_coefs["a"])
            MO_H_two_chemist = ao2mo.restore(1, MO_H_two_packed, self.MO_coefs["a"].shape[1]) # In mulliken notation: MO_H_two[p][q][r][s] = (pq|rs)

            self.MO_H_two["a"] = MO_H_two_chemist.transpose(0, 2, 1, 3)# - MO_H_two_chemist.transpose(0, 3, 1, 2)
            self.MO_H_two["b"] = self.MO_H_two["a"]
            self.MO_H_two["ab"] = self.MO_H_two["a"]"""

        if self.HF_method == "RHF":
            # Everything is the same in both subspaces
            mean_field = scf.RHF(self.mol).run(verbose = 0)

            self.MO_coefs = mean_field.mo_coeff
            self.reference_state_energy = mean_field.e_tot

            self.MO_H_one = np.matmul(self.get_MO_coefs().T, np.matmul(AO_H_one, self.get_MO_coefs()))
            MO_H_two_packed = ao2mo.kernel(AO_H_two_chemist, self.get_MO_coefs())
            MO_H_two_chemist = ao2mo.restore(1, MO_H_two_packed, self.get_MO_coefs().shape[1]) # In mulliken notation: MO_H_two[p][q][r][s] = (pq|rs)

            self.MO_H_two = MO_H_two_chemist.transpose(0, 2, 1, 3)

        elif self.HF_method == "UHF":
            # For UHF, MO coefs and exchange integrals are signed by spin
            self.MO_coefs = {}
            self.MO_H_one = {}
            self.MO_H_two = {}

            # Coefs and exchange intergrals differ for the two subspaces
            mf_object = scf.UHF(self.mol)

            mf_object.init_guess = 'atom'  # Atomic initial guess. If doesn't work, run RHF first and then use its result as the initial guess
            mf_object.conv_tol = 1e-10 # Tighter convergence

            mean_field = mf_object.run(verbose = 0)
            self.MO_coefs["a"] = mean_field.mo_coeff[0]
            self.MO_coefs["b"] = mean_field.mo_coeff[1]

            assert self.MO_coefs["a"].shape[1] == self.MO_coefs["b"].shape[1]
            # This is not required but if we remove the constraint we need to
            # firstly restore symmetry, making the mixed H_two["ab"] non-square

            self.reference_state_energy = mean_field.e_tot

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

        self.HF_known = True

    def load_HF(self, mean_field_info, MO_coefs, E_nuc, MO_H_one, MO_H_two, reference_state_energy):
        # If HF_method = "RHF", MO_coefs, MO_H_one, MO_H_two do not have
        # ["a"/"b"(/"ab")] indices.
        # If HF_method = "UHF", they do.

        # Global properties (const along the surface)
        self.mol = None
        self.HF_method = mean_field_info["HF_method"]
        self.N_orb = mean_field_info["N_orb"]
        self.S_alpha = mean_field_info["S_alpha"]
        self.S_beta = mean_field_info["S_beta"]

        # Local properties (geometry-dependent)
        self.MO_coefs = MO_coefs
        self.E_nuc = E_nuc
        self.MO_H_one = MO_H_one
        self.MO_H_two = MO_H_two
        self.reference_state_energy = reference_state_energy

        self.HF_known = True


    # ----------- Hamiltonian overlap integral calculation methods ------------

    def mode_exchange_energy(self, m_i, m_f, spin = None, debug = False):
        # This function translates mode indices to spatial MO indices and returns the coefficient tensor element

        # If Hilbert space is spin-symmetrical, the spin argument is not necessary
        # Otherwise, spin is either "a", "b", or "ab" (to access "ba", do double transpose on "ab")

        if spin is None:
            if self.HF_method == "RHF":
                spin = "a" # doesn't matter
            else:
                return(None)

        # m_i/f are lists of either one mode index (single electron exchange) or two mode indices (two electron exchange)
        # the indices in m_i/f are SPATIAL, and are interpreted as acting on one spixn state only.
        if len(m_i) == 1:
            return(self.get_MO_H_one(spin, m_i[0], m_f[0]))
        elif len(m_i) == 2:
            return(self.get_MO_H_two(spin, m_i[0], m_i[1], m_f[0], m_f[1]))
        return(0.0)

    def H_overlap(self, pair_a, pair_b):
        # This method calculates <Z_a | H | Z_b>, including the nuclear self-energy
        # here pair_a/b = [state_a/b with spin alpha, state_a/b with spin beta]

        alpha_overlap = pair_a[0].norm_overlap(pair_b[0])
        beta_overlap = pair_a[1].norm_overlap(pair_b[1])

        # To speed up cross-spin two-electron matrix elements, we prepare a matrix of all first-order sequence overlaps
        W_alpha = np.zeros((self.N_orb, self.N_orb), dtype=complex) # [i][j] = < alpha | f\hc_i f_j | alpha >
        W_beta = np.zeros((self.N_orb, self.N_orb), dtype=complex) # [i][j] = < beta | f\hc_i f_j | beta >

        H_one_term = 0.0
        # This is a sum over all mode pairs
        for p in range(self.N_orb):
            for q in range(self.N_orb):

                W_alpha[p][q] = pair_a[0].norm_overlap(pair_b[0], [p], [q])
                W_beta[p][q]  = pair_a[1].norm_overlap(pair_b[1], [p], [q])

                # alpha
                H_one_term += self.mode_exchange_energy([p], [q], "a") * W_alpha[p][q] * beta_overlap
                # beta
                H_one_term += self.mode_exchange_energy([p], [q], "b") * W_beta[p][q] * alpha_overlap

        H_two_term = 0.0

        # equal spin
        c_pairs = functions.subset_indices(np.arange(self.N_orb), 2)
        a_pairs = functions.subset_indices(np.arange(self.N_orb), 2)
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

        # opposite spin
        for i in range(self.N_orb):
            for j in range(self.N_orb):
                for k in range(self.N_orb):
                    for l in range(self.N_orb):
                        prefactor_alpha_beta = 0.5 * self.mode_exchange_energy([i, j], [k, l], "ab")
                        prefactor_beta_alpha = 0.5 * self.mode_exchange_energy([j, i], [l, k], "ab")

                        # alpha beta
                        H_two_term += prefactor_alpha_beta * W_alpha[i][k] * W_beta[j][l]

                        # beta alpha
                        H_two_term += prefactor_beta_alpha * W_alpha[j][l] * W_beta[i][k]


        H_nuc = self.E_nuc * alpha_overlap * beta_overlap
        return(H_one_term + H_two_term + H_nuc)

    # -------------------------------------------------------------------------
    # ---- Methods to obtain the FCI solutions in the low-energy spectrum -----
    # -------------------------------------------------------------------------

    # ------------------- Occupancy basis labelling methods -------------------

    def addr_to_index(self, addr_A, addr_B):
        # zips up the indices on the two spin subspaces into canonical index

        return( addr_A * math.comb(self.N_orb, self.S_beta) + addr_B )

    def index_to_occ_string(self, i):

        addr_A = i // math.comb(self.N_orb, self.S_beta)
        addr_B = i % math.comb(self.N_orb, self.S_beta)

        occ_A = fci.cistring.addr2str(self.N_orb, self.S_alpha, addr_A)
        occ_B = fci.cistring.addr2str(self.N_orb, self.S_beta, addr_B)

    # ------------------------ FCI calculation methods ------------------------

    def run_FCI(self, N_surf):


        if not self.HF_known:
            # We run the HF calculation out of necessity
            self.run_HF("default")
        if self.mol is None:
            # This happens when HF was loaded from disk
            self.mol = self.mol_bp.get_gto_Mole(self.g)


        FCI_energies = np.zeros((N_surf))
        FCI_coefs = np.zeros((N_surf, math.comb(self.N_orb, self.S_alpha) * math.comb(self.N_orb, self.S_beta)))

        if self.HF_method == "RHF":
            cisolver = fci.FCI(self.mol, self.get_MO_coefs())
        elif self.HF_method == "UHF":
            cisolver = fci.FCI(self.mol, (self.get_MO_coefs("a"), self.get_MO_coefs("b")))

        FCI_E, raw_FCI_sol = cisolver.kernel(nroots = N_surf)

        if N_surf == 1:
            FCI_E = [FCI_E]
            raw_FCI_sol = [raw_FCI_sol]

        for i_surf in range(N_surf):
            FCI_energies[i_surf] = FCI_E[i_surf]
            for a in range(raw_FCI_sol[i_surf].shape[0]):
                for b in range(raw_FCI_sol[i_surf].shape[1]):
                    FCI_coefs[i_surf][self.addr_to_index(a, b)] = raw_FCI_sol[i_surf][a, b]

        return(FCI_energies, FCI_coefs)

