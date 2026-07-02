import math
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


# Notice: STO-type orbital basis is a linear combination of GTOs, which are a product of a real-valued radial function and the complex spherical harmonics.
# Since the two-body integrals have full rotational symmetry, and the radial dependence is given by a real-valued function, their values are always real.
# The single-body integrals reflect the angular dependence of the AOs, and as such will be complex-valued.



# For the AOs, PySCF uses Mulliken's notation: https://gqcg-res.github.io/knowdes/two-electron-integrals.html


class potential_surface_extrapolator():

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

        "surfaces" : {
            "geometry" : "csv",
            "base_node" : "csv",
            "mean_field_MO_coefs" : "csv",
            "mean_field_H_one",
            "mean_field_H_two"
            },

        "diagnostics" : {"diagnostic_log" : "txt"} # Condition numbers, eigenvalue min-max ratios, norms etc
    }

    mean_field_methods = {
        "RHF" : scf.RHF,
        "UHF" : scf.UHF
    }


    def __init__(self, ID, log_verbosity = 5):

        self.ID = ID
        self.log = Journal(log_verbosity, print_on_the_fly = True)
        # Verbosity key:
        #   0/None: Important, always prints (user methods)
        #   2: Important, major subroutine
        #   5: Minor subroutine
        #   10: Micro-subroutine (expected to repeat many times)
        # Typically, writes within subroutines are one level of verbosity higher than the subroutine itself

        self.log.enter(f"Initialising molecule solver {ID} at log_verbosity = {log_verbosity}")

        # Data Storage Manager
        self.log.write("Initialising Data Storage Manager...", 1)
        #self.disk_jockey = Disk_Jockey(f"outputs/{self.ID}", self.log)
        #self.disk_jockey.create_data_nodes(potential_surface_extrapolator.data_nodes) # Each solver call adds a new node dynamically

        self.user_actions = f"initialised solver {self.ID}\n"
        self.diagnostics_log = []
        # The structure of the diagnostics log is like so: every element is either a list (for multiple same-level subprocedures) or a dict (for a header : content) pair

        # ----------- Self-analysis properties
        self.log.write("Initialising self-analysis properties...", 1)
        self.checklist = [] # List of succesfully performed actions
        self.measured_datasets = [] # list of dataset labels

        # Potential surface structure properties
        self.structure = None

        self.log.exit()

        self.find_potential_surfaces_methods = {
            "frozen_basis" : self.find_potential_surfaces_frozen_basis
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
            M = self.N_orb
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


    # -------------------- Find potential surfaces methods --------------------

    def find_potential_surfaces_frozen_basis(self, **kwargs):

        # This method takes the base eigenstates decomposed into CSs (of
        # arbitrary composition), and freezes the CS parameters as well as the
        # decomposition coefficients. By modifying the geometry, the CSs also
        # deform (since the MOs deform).

        # kwargs:
        #     ------------------------
        #     -dataset_label: if present, this will label the dataset in disc jockey. Otherwise, label is generated from other kwargs.

        # -------------------- Parameter initialisation
        if "dataset_label" in kwargs:
            dataset_label = kwargs["dataset_label"]
        else:
            dataset_label = f"frozen_basis"


    ###########################################################################
    # ----------------------------- User methods ------------------------------
    ###########################################################################

    def initialise_structure(self, structure, HF_method = "default"):
        # structure is an instance of MGStructure (or of its subclass)

        # This method runs HF and FCI for every node in the structure

        self.log.enter("Initialising molecular geometry structure...")

        self.structure = structure

        self.log.write("Molecule blueprint:")
        self.log.print_itemize({
            "name" : self.structure.mol_bp.name,
            "atoms" : ', '.join(self.structure.mol_bp.hr_atoms),
            "basis" : self.structure.mol_bp.basis,
            "unit" : self.structure.mol_bp.unit
            })

        if self.structure.mol_bp.spin is not None:
            self.log.write(f"  -spin: determined automatically")
        else:
            self.log.write(f"  -spin: {self.structure.mol_bp.spin}")

        self.log.write("Geometry structure:")
        self.log.write(f"  {self.structure.geometry_meta['class']}: {self.structure.geometry_meta['desc']}")

        self.log.write(f"Number of potential surfaces investigated: {self.structure.base_node_meta['N_surf']}")

        self.log.enter("Running mean-field calculations for all nodes", semaphored = True, tau_space = np.linspace(0, self.structure.N_nodes + 1, 100 + 1))

        # We run HF for every node TODO unless it was loaded from disk NOTE save this for a dedicated load_structure method?
        for i in range(self.structure.N_nodes):
            self.structure.nodes[i].run_HF(HF_method)
            self.log.update_semaphor_event(i + 1)

        self.log.exit("Mean field calculation")


        # Now every node has its MOs and exchange integrals

        # We store some global properties of the molecule
        self.N_orb = self.structure.nodes[0].mol.nao
        nalpha, nbeta = self.structure.nodes[0].mol.nelec
        self.S_alpha = nalpha
        self.S_beta = nbeta

        self.structure.nodes_meta["HF_method"] = self.structure.nodes[0].HF_method
        self.structure.nodes_meta["N_orb"] = self.N_orb
        self.structure.nodes_meta["S_alpha"] = self.S_alpha
        self.structure.nodes_meta["S_beta"] = self.S_beta

        self.log.write("Hilbert space:")
        self.log.print_itemize(self.structure.nodes_meta)

        self.log.exit()

    def run_FCI_on_structure(self):

        self.log.enter("Running FCI for every node in the geometry structure...")

        N_surf = self.structure.base_node_meta["N_surf"]

        self.log.write("Allocating memory for results...")

        FCI_energies = np.zeros((N_surf, self.structure.N_nodes))
        FCI_coefs = np.zeros((N_surf, self.structure.N_nodes, math.comb(self.N_orb, self.S_alpha) * math.comb(self.N_orb, self.S_beta))) # real coefs for FCI

        self.log.enter("Performing FCI", semaphored = True, tau_space = np.linspace(0, self.structure.N_nodes + 1, 1000 + 1))

        for i in range(self.structure.N_nodes):
            cur_FCI_energies, cur_FCI_coefs = self.structure.nodes[i].run_FCI(N_surf)
            for i_surf in range(N_surf):
                FCI_energies[i_surf, i] = cur_FCI_energies[i_surf]
                FCI_coefs[i_surf, i] = cur_FCI_coefs[i_surf]
            self.log.update_semaphor_event(i + 1)

            """node = self.structure.nodes[i]
            if node.HF_method == "RHF":
                cisolver = fci.FCI(node.mol, node.MO_coefs["a"])
            elif node.HF_method == "UHF":
                cisolver = fci.FCI(node.mol, (node.MO_coefs["a"], node.MO_coefs["b"]))

            self.log.write("FCI solver initialised...", 0)
            FCI_E, raw_FCI_sol = cisolver.kernel(nroots = N_surf)

            if N_surf == 1:
                FCI_E = [FCI_E]
                raw_FCI_sol = [raw_FCI_sol]

            # We convert the raw_ci_sol object (which is an FCIvector) into a dict
            # with tuples as keys (tuples represent occupancy strings)


            for i_surf in range(N_surf):

                sol_array = np.zeros( raw_FCI_sol[i_surf].shape[0] * raw_FCI_sol[i_surf].shape[1] )
                for a in range(raw_FCI_sol[i_surf].shape[0]):
                    for b in range(raw_FCI_sol[i_surf].shape[1]):
                        sol_array[self.addr_to_index(a, b)] = raw_FCI_sol[i_surf][a, b]

                FCI_energies[i_surf, i] = FCI_E[i_surf]
                FCI_coefs[i_surf, i] = sol_array"""

        self.log.exit("FCI calculation")

        self.structure.nodes_meta["FCI_known"] = True

        for i_surf in range(N_surf):
            self.structure.add_surface(f"FCI[{i_surf}]", FCI_energies[i_surf], "occupancy", FCI_coefs[i_surf], i_surf, "FCI")

        self.log.write(f"{N_surf} lowest-energy FCI surfaces saved.")

        self.log.exit()


    r"""
    def initialise_base_molecule(self, mol_bp, base_g, N_surf, HF_method = "default"):
        # mol is an instance of MBlueprint
        # base_g is an instance of MGeometry
        # N_surf is an integer, and specifies the number of surfaces we shall
        # analyse. Equivalently, the number of lowest-energy E-eigenstates.

        # 1 Store the provided descriptions of the base molecule
        # 2 Calculate the base molecule HF properties
        # 3 Calculate

        self.log.enter("Initialising base molecule...", 0)

        if "mol_init" in self.checklist:
            self.log.write("Molecule already initialised.", 1)
            return(None)

        # 1 Building the base molecule and describing it

        self.log.enter("Building base molecule...")
        self.mol_bp = mol_bp
        self.base_g = base_g

        self.base_node = MPSNode()


        self.base_mol = self.mol_bp.get_gto_Mole(self.base_g)


        self.M = self.base_mol.nao * 2
        self.S = self.base_mol.tot_electrons()

        nalpha, nbeta = self.base_mol.nelec
        self.S_alpha = nalpha
        self.S_beta = nbeta

        self.log.write(f"There are {self.base_mol.nao} atomic orbitals, each able to hold 2 electrons of opposing spin.", 3)
        self.log.write(f"The molecule is occupied by {self.base_mol.tot_electrons()} electrons in total: {self.S_alpha} with spin alpha, {self.S_beta} with spin beta.", 3)
        self.log.write(f"The molecule consists of the following atoms: {', '.join(self.mol_bp.atoms)}", 3)
        self.log.write(f"The atomic orbitals are ordered as follows: {self.base_mol.ao_labels()}", 3)
        self.log.write(f"The nuclear repulsion energy is {self.base_mol.energy_nuc()}", 3)

        self.log.exit()

        # 2 Calculate the


        self.log.write("Calculating 1e integrals...", 1)
        AO_H_one = self.base_mol.intor('int1e_kin', hermi = 1) + self.base_mol.intor('int1e_nuc', hermi = 1)

        self.log.write("Calculating 2e integrals...", 1)
        #self.H_two = self.base_mol.intor('int2e', aosym = "s1")
        AO_H_two_chemist = self.base_mol.intor('int2e')
        # <ij|kl> = (ik|jl)

        self.log.exit()

        self.log.enter("Performing mean-field calculations to determine the molecular orbitals...", 1)

        if HF_method == "RHF":
            self.HF_method = HF_method
            self.log.write("Mean field method: Restricted Hartree-Fock (selected by user)")
            if self.base_mol.spin != 0:
                self.log.write("WARNING: The molecule is not a singlet, RHF is unsuitable.")
        elif HF_method == "UHF":
            self.HF_method = HF_method
            self.log.write("Mean field method: Unrestricted Hartree-Fock (selected by user)")
        elif HF_method == "default":
            if self.base_mol.spin == 0:
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
            self.mean_field = scf.RHF(self.base_mol).run(verbose = 0)

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
            mf_object = scf.UHF(self.base_mol)

            mf_object.init_guess = 'atom'  # Atomic initial guess. If doesn't work, run RHF first and then use its result as the initial guess
            #mf_rhf = scf.RHF(self.base_mol).run(verbose=0)
            #mf_uhf = scf.UHF(self.base_mol)
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
            null_state_alpha = self.coherent_state_types[CS_type].null_state(self.base_mol.nao, self.S_alpha)
            null_state_beta = self.coherent_state_types[CS_type].null_state(self.base_mol.nao, self.S_beta)
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
            "atom" : self.base_mol.atom,
            "basis" : self.base_mol.basis,
            "spin" : self.base_mol.spin
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
    """

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


    # --------------------- Occupancy bitstring labelling ---------------------



    """
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
        return([ list_a + [0] * (self.N_orb - len(list_a)), list_b + [0] * (self.N_orb - len(list_b)) ])


    def occ_idx_to_occ_list(self, idx_alpha, idx_beta):
        occ_alpha = fci.cistring.addr2str(self.N_orb, self.S_alpha, idx_alpha)
        occ_beta = fci.cistring.addr2str(self.N_orb, self.S_beta, idx_beta)
        # These are just integers corresponding to the binary value
        occ_alpha_bin = "{0:b}".format(occ_alpha)
        occ_beta_bin = "{0:b}".format(occ_beta)
        # We're still missing leading zeros. Those are just tailing zeros in the list
        alpha_list = []
        for i in range(len(occ_alpha_bin) - 1, -1, -1):
            alpha_list.append(int(occ_alpha_bin[i]))
        alpha_list += [0] * (self.N_orb - len(alpha_list))
        beta_list = []
        for i in range(len(occ_beta_bin) - 1, -1, -1):
            beta_list.append(int(occ_beta_bin[i]))
        beta_list += [0] * (self.N_orb - len(beta_list))
        return(alpha_list, beta_list)

    def get_prom_label(self, bitlist, trim_M = None, hr = False):
        # Returns a tuple [[de-occupied MOs], [promoted MOs]] from ref state
        # if hr, this is human-readable (i.e. MO labels are +1)

        cur_S = sum(bitlist)
        hr_cor = 0
        if hr:
            hr_cor = 1

        act_M = self.N_orb
        if trim_M is not None:
            if trim_M > cur_S: # We need at least one empty shell
                act_M = min(trim_M, self.N_orb)

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
    """

    def get_ref_state(self):
        # Returns a list of lists, useful for further modification
        return([[1] * self.S_alpha + [0] * (self.N_orb - self.S_alpha), [1] * self.S_beta + [0] * (self.N_orb - self.S_beta)])



    ###########################################################################
    ############################# Output methods ##############################
    ###########################################################################

    # ------------------------ Data storage management ------------------------

    def save_data(self):
        self.user_actions += f"save_data\n"
        # ---- Save user log
        self.disk_jockey.commit_datum_bulk("system", "user_actions", self.user_actions)
        self.disk_jockey.commit_datum_bulk("system", "log", self.log.dump())

        # ---- Save the node structure
        # Save the geometry
        self.disk_jockey.commit_datum_bulk("surfaces", "geometry", TODO) # csv row : node index, col : r_0_x, r_0_y, r_0_z, r_1_x ...
        self.disk_jockey.commit_metadatum("surfaces", "geometry", self.structure.geometry_meta)
        # Save the base node data
        self.disk_jockey.commit_datum_bulk("surfaces", "base_node", TODO) # csv row : N_surf index, col : E, coef 0, coef 1 ...
        self.disk_jockey.commit_metadatum("surfaces", "base_node", self.structure.base_node_meta)
        # Save the node mean-field properties
        self.disk_jockey.commit_datum_bulk("surfaces", "nodes", TODO) # csv row : node index, col : MO_coef_i, H_one_a/b, H_two_a/b/ab
        self.disk_jockey.commit_metadatum("surfaces", "nodes", self.structure.nodes_meta)
        # Save the surface calculation
        for surface_label in self.structure.nodes_meta["surface_labels"]:
            self.disk_jockey.commit_datum_bulk("surfaces", surface_label, TODO) # csv row : node index, col : E, coef 0, coef 1 ...
            self.disk_jockey.commit_metadatum("surfaces", surface_label, self.structure.surfaces_meta[surface_label])




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
            self.disk_jockey.commit_metadatum("self_analysis", "LE_sol", self.LE_description)

        # save diagnostic
        self.disk_jockey.commit_datum_bulk("diagnostics", "diagnostic_log", self.print_diagnostic_log())

        # Save to disk
        self.disk_jockey.save_data()

        self.log.close_journal()

    def load_data(self, what_to_load = None, load_specific_datasets = None):
        # what_to_load is a list of magic strings
        # load_specific_datasets specifies dataset labels which are loaded


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


            if load_specific_datasets is None:
                load_specific_datasets = self.disk_jockey.metadata["system"]["log"]["measured_datasets"]

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
                for loaded_dataset in load_specific_datasets:
                    if loaded_dataset not in self.disk_jockey.data_nodes.keys():
                        self.log.write(f"WARNING: Requested dataset '{loaded_dataset}' not found on disk.")
                    elif loaded_dataset in self.measured_datasets:
                        self.log.write(f"WARNING: Requested dataset '{loaded_dataset}' initialised during computation session. Loading aborted to not overwrite session data.")
                    else:
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
                self.log.exit()

            self.log.exit()


    # ------------------------------- Plotting --------------------------------



    # --------------------------------------------------------------------------
    # ------------------------------ User methods ------------------------------
    # --------------------------------------------------------------------------

    def find_potential_surfaces(self, method, **kwargs):
            if method in self.find_potential_surfaces_methods.keys():
                return(self.find_potential_surfaces_methods[method](**kwargs))
            else:
                self.log.write(f"ERROR: Unknown potential surface analysis method {method}. Available methods: {self.find_potential_surfaces_methods.keys()}")
                return(None)




