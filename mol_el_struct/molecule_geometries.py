# This file separates molecule geometry from other aspects of molecule
# construction. Tools for traversing subspaces of the geometry space using
# restrictions and parametrisation are given.

import numpy as np
from pyscf import gto

class MBlueprint():

    def __init__(self, atoms, basis, unit, spin = None, label = None):
        self.atoms = atoms # ordered list of atom labels
        self.basis = basis
        self.unit = unit
        self.spin = spin

        self.N_a = len(self.atoms) # number of atoms

        # "label" is a human-readable string to be used in pyplot. It can use latex formatting.
        self.label = label

        # We create a hr_atoms list which labels repeated atoms by their index
        self.hr_atoms = []
        atom_multiplicity = {}
        for i in range(len(self.atoms)):
            if self.atoms[i] not in atom_multiplicity.keys():
                atom_multiplicity[self.atoms[i]] = 1
                self.hr_atoms.append(self.atoms[i])
            else:
                atom_multiplicity[self.atoms[i]] += 1
                self.hr_atoms.append(f"{self.atoms[i]} ({atom_multiplicity[self.atoms[i]]})")

        self.mol_meta = {
            "label" : self.label,
            "atoms" : self.atoms,
            "basis" : self.basis,
            "unit" : self.unit,
            "spin" : self.spin
            }


    def get_atoms(self, g):
        # g is an instance of MGeometry
        # The position of the first atom is taken as 0, 0, 0
        # The output is a string which may be passed to PySCF
        res = ""

        res += f"{self.atoms[0]}  0  0  0\n"

        for i in range(len(self.atoms) - 1):
            res += f"{self.atoms[i + 1]}   {g.r[i][0]}   {g.r[i][1]}   {g.r[i][2]}\n"
        return(res)

    def get_gto_Mole(self, g):
        # g is an instance of MGeometry
        # The output is an instance of gto.Mole initialised according to the
        # blueprint and the geometry
        res = gto.Mole()
        if self.spin is None:
            res.build(
                atom = self.get_atoms(g),
                basis = self.basis,
                unit = self.unit
                )
        else:
            res.build(
                atom = self.get_atoms(g),
                basis = self.basis,
                unit = self.unit,
                spin = self.spin
                )
        return(res)




class MGeometry():

    def __init__(self, r):
        # r is a list of positions
        # r[i] is the position of the i-th atom relative to the zeroth atom

        self.N_a = len(r) + 1
        self.r = np.array(r)


    # Binary deformations

    def __add__(self, other):
        # Addition adds the atom positions elementwise
        # It is useful to deform a ref=.geometry by a basis of deformations
        assert isinstance(other, MGeometry)
        return(MGeometry(self.r + other.r))

    def __mul__(self, scale):
        # Rescales or positions by a scalar float
        assert isinstance(other, float) or isinstance(other, int) or isinstance(other, float) or isinstance(other, np.float32) or isinstance(other, np.float64) or isinstance(other, np.int32) or isinstance(other, np.int64)
        return(MGeometry(self.r * other))


class MPSNode():
    # Molecular potential surface node
    # Corresponds to one geometry point of one molecule blueprint, stores the mean field properties there as well as calculated eigenstates

    def __init__(self, mol_bp, g):
        # mol_bp is an instance of MBlueprint
        # g is an instance of MGeometry
        self.mol_bp = mol_bp
        self.g = g

        self.surfaces = {} # surface label : {"E" : energy, "sol" : coefficient dict}

    # --------- Methods to obtain the MOs and the exchange integrals ----------

    def run_HF(self, HF_method = "default"):

        self.mol = self.mol_bp.get_gto_Mole(self.g)

        self.N_orb = self.mol.nao

        if HF_method == "default":
            if self.mol.spin == 0:
                self.HF_method = "RHF"
            else:
                self.HF_method = "UHF"
        else:
            self.HF_method = HF_method

        AO_H_one = self.mol.intor('int1e_kin', hermi = 1) + self.mol.intor('int1e_nuc', hermi = 1)
        AO_H_two_chemist = self.mol.intor('int2e')

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

            self.MO_H_one["a"] = np.matmul(self.MO_coefs["a"].T, np.matmul(AO_H_one, self.MO_coefs["a"]))
            self.MO_H_one["b"] = self.MO_H_one["a"]
            MO_H_two_packed = ao2mo.kernel(AO_H_two_chemist, self.MO_coefs["a"])
            MO_H_two_chemist = ao2mo.restore(1, MO_H_two_packed, self.MO_coefs["a"].shape[1]) # In mulliken notation: MO_H_two[p][q][r][s] = (pq|rs)

            self.MO_H_two["a"] = MO_H_two_chemist.transpose(0, 2, 1, 3)# - MO_H_two_chemist.transpose(0, 3, 1, 2)
            self.MO_H_two["b"] = self.MO_H_two["a"]
            self.MO_H_two["ab"] = self.MO_H_two["a"]

        elif self.HF_method == "UHF":
            # Coefs and exchange intergrals differ for the two subspaces
            mf_object = scf.UHF(self.mol)

            mf_object.init_guess = 'atom'  # Atomic initial guess. If doesn't work, run RHF first and then use its result as the initial guess
            mf_object.conv_tol = 1e-10 # Tighter convergence

            self.mean_field = mf_object.run(verbose = 0)
            self.MO_coefs["a"] = self.mean_field.mo_coeff[0]
            self.MO_coefs["b"] = self.mean_field.mo_coeff[1]

            assert self.MO_coefs["a"].shape[1] == self.MO_coefs["b"].shape[1]
            # This is not required but if we remove the constraint we need to
            # firstly restore symmetry, making the mixed H_two["ab"] non-square

            self.reference_state_energy = self.mean_field.e_tot

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

    def add_surface(self, label, energy, coef):
        # Here, coef is an instance of FCIvector


        self.surfaces[label] = {
                "E" : energy,
                "sol" : coef
            }

    # Hamiltonian overlap integral calculation methods

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

                        mixed += prefactor_alpha_beta * W_alpha[i][k] * W_beta[j][l] + prefactor_beta_alpha * W_alpha[j][l] * W_beta[i][k]


        H_nuc = self.mol.energy_nuc() * alpha_overlap * beta_overlap
        return(H_one_term + H_two_term + H_nuc)



