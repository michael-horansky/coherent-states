import math
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib import cm

from pyscf import gto, scf, cc, ao2mo, ci, fci, pbc

from coherent_states.CS_Thouless import CS_Thouless
from coherent_states.CS_Qubit import CS_Qubit
from coherent_states.CS_sample import CS_sample

from utils.class_Journal import Journal
from utils.class_Disk_Jockey import Disk_Jockey
import utils.functions as functions

from surf.MBlueprint import bp_catalogue

from surf.MGStructure import MGStructure
from surf.MGGrid import MGGrid


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

    structure_types = {
        "MGStructure" : MGStructure,
        "MGGrid" : MGGrid
    }

    data_nodes = {
        "system" : {"user_actions" : "txt", "log" : "txt"}, # User actions summary, Journal style log

        "structure" : {
            "info" : "json", # basic info about structure, including mol bp
            "geometry" : "pkl", # data of unknown format (depends on subclass)
            "mean_field" : "json", # HF_method and Hilbert space parameters
            "mean_field_MO_coefs" : "nda",
            "mean_field_E_nuc" : "nda",
            "mean_field_H_one" : "nda",
            "mean_field_H_two" : "nda",
            "mean_field_reference_state_energy" : "nda",
            "base_sol_energies" : "nda",
            "base_sol_coefs" : "nda",
            "base_sol_spins" : "nda"
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

        self.log.enter(f"Initialising potential surface extrapolator {ID} at log_verbosity = {log_verbosity}")

        # Data Storage Manager
        self.log.write("Initialising Data Storage Manager...", 1)
        self.disk_jockey = Disk_Jockey(f"surf_outputs/{self.ID}", self.log)
        self.disk_jockey.create_data_nodes(potential_surface_extrapolator.data_nodes) # Each solver call adds a new node dynamically

        self.user_actions = f"initialised PSE {self.ID}\n"
        self.diagnostics_log = []
        # The structure of the diagnostics log is like so: every element is either a list (for multiple same-level subprocedures) or a dict (for a header : content) pair

        # ----------- Self-analysis properties
        self.log.write("Initialising self-analysis properties...", 1)
        self.checklist = [] # List of succesfully performed actions

        # Potential surface structure properties
        self.structure = None

        self.log.exit()

        self.find_potential_surfaces_methods = {
            "frozen_basis" : self.find_potential_surfaces_frozen_basis
        }

    ###########################################################################
    # --------------------------- Internal methods ----------------------------
    ###########################################################################

    # ------------------------ Data management methods ------------------------

    def register_surface(self, label, E, basis, coef, meta):

        # Add surface to self.structure and to disk_jockey

        self.structure.add_surface(label, E, basis, coef, meta)

        self.disk_jockey.create_data_nodes({label : {
            "energy" : "nda",
            "coef" : "nda"
            }})
        self.disk_jockey.commit_datum_bulk(label, "energy", self.structure.surfaces[label].E)
        self.disk_jockey.commit_metadatum(label, "energy", meta)

        # We need to flatten the coef array. We do not assume anything about
        # its shape beyond the first (node) axis. Different bases have
        # different shapes. Instead, we read the shape and store it as meta.

        #coef_shape = list(self.structure.surfaces[label].coef.shape[1:])
        #self.disk_jockey.commit_datum_bulk(label, "coef", np.reshape( self.structure.surfaces[label].coef, (self.structure.N_nodes, np.prod(coef_shape)) ) )
        self.disk_jockey.commit_datum_bulk(label, "coef", self.structure.surfaces[label].coef)
        self.disk_jockey.commit_metadatum(label, "coef", {
            "basis" : self.structure.surfaces[label].basis
            })

    def load_surface(self, label):
        # Reads surface from disk_jockey and adds it to self.structure
        #coef_shape = self.disk_jockey.metadata[label]["coef"]["coef_shape"]
        #deflattened_coef = np.reshape( self.disk_jockey.data_bulks[label]["coef"], [self.structure.N_nodes] + coef_shape )

        self.structure.add_surface(
            label,
            self.disk_jockey.data_bulks[label]["energy"],
            self.disk_jockey.metadata[label]["coef"]["basis"],
            self.disk_jockey.data_bulks[label]["coef"],
            self.disk_jockey.metadata[label]["energy"]
            )







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

    # -------------------------------------------------------------------------
    # ----------------------- Structure initialisation ------------------------
    # -------------------------------------------------------------------------

    # ----------------- Automatic calculation initialisations -----------------

    def init_structure_automatically(self, cls, mol_name, HF_method = "default", N_surf = 1, **kwargs):
        # cls is a key in structure_types
        # mol_name is a key in bp_catalogue
        # HF_method is a magic word (RHF/UHF/default)
        # N_surf is the number of lowest-energy potential surfaces to consider
        # kwargs are kwargs for the relevant init_geometry method

        self.log.enter("Initialising structure by automatic calculation...")

        # 1. Register structure

        structure_class = potential_surface_extrapolator.structure_types[cls]
        self.log.write(f"Structure class: {cls}. {structure_class.desc}")

        mol_bp = bp_catalogue[mol_name]
        self.log.write("Molecule blueprint:")
        self.log.print_itemize({
            "name" : mol_bp.name,
            "atoms" : ', '.join(mol_bp.hr_atoms),
            "basis" : mol_bp.basis,
            "unit" : mol_bp.unit
            })

        self.structure = structure_class(mol_bp, self.log)

        self.disk_jockey.commit_datum_bulk("structure", "info", self.structure.info)

        # 2. Initialise geometry

        self.structure.init_geometry(**kwargs)
        self.log.write(f"Structure geometry loaded ({self.structure.N_nodes} nodes).")

        self.disk_jockey.commit_datum_bulk("structure", "geometry", kwargs)

        # 3. Initialise nodes

        self.log.enter("HF calculation", semaphored = True, tau_space = np.linspace(0, self.structure.N_nodes, 1000 + 1))
        self.structure.init_nodes_with_HF(HF_method)
        self.log.exit("HF calculation")
        self.log.write("Mean-field properties loaded for each node:")
        self.log.print_itemize({
            "Hartree-Fock method" : self.structure.mean_field["HF_method"],
            "Number of spatial molecular orbitals" : self.structure.mean_field["N_orb"],
            "Electrons with spin alpha" : self.structure.mean_field["S_alpha"],
            "Electrons with spin beta" : self.structure.mean_field["S_beta"],
            })

        self.disk_jockey.commit_datum_bulk("structure", "mean_field", self.structure.mean_field)
        self.disk_jockey.commit_datum_bulk("structure", "mean_field_MO_coefs", self.structure.encode_MO_coefs())
        self.disk_jockey.commit_datum_bulk("structure", "mean_field_E_nuc", self.structure.get_E_nuc_surface())
        self.disk_jockey.commit_datum_bulk("structure", "mean_field_H_one", self.structure.encode_H_one())
        self.disk_jockey.commit_datum_bulk("structure", "mean_field_H_two", self.structure.encode_H_two())
        self.disk_jockey.commit_datum_bulk("structure", "mean_field_reference_state_energy", self.structure.get_reference_state_energy_surface())

        # 4. Initialise base node solution

        self.structure.find_base_sol(N_surf)
        self.log.write(f"Base node solution loaded (for the {self.structure.N_surf} lowest-energy surfaces)")

        self.disk_jockey.commit_datum_bulk("structure", "base_sol_energies", self.structure.base_sol_energies)
        self.disk_jockey.commit_datum_bulk("structure", "base_sol_coefs", self.structure.base_sol_coefs)
        self.disk_jockey.commit_datum_bulk("structure", "base_sol_spins", self.structure.base_sol_spins)


    # ----------------------- Load structure from disk ------------------------

    def load_structure(self):
        self.log.enter("Loading molecular geometry structure...")

        # ---------- Build structure

        # 1. Register structure

        structure_info = self.disk_jockey.data_bulks["structure"]["info"]
        self.log.write(f"Structure class: {structure_info['class']}. {structure_info['desc']}")

        mol_bp = bp_catalogue[structure_info["mol_bp"]["name"]]
        self.log.write("Molecule blueprint:")
        self.log.print_itemize({
            "name" : mol_bp.name,
            "atoms" : ', '.join(mol_bp.hr_atoms),
            "basis" : mol_bp.basis,
            "unit" : mol_bp.unit
            })

        self.structure = potential_surface_extrapolator.structure_types[structure_info["class"]](mol_bp)

        # 2. Initialise geometry

        self.structure.init_geometry(**self.disk_jockey.data_bulks["structure"]["geometry"])
        self.log.write(f"Structure geometry loaded ({self.structure.N_nodes} nodes).")

        # 3. Initialise nodes

        decoded_MO_coefs = self.structure.decode_MO_coefs(self.disk_jockey.data_bulks["structure"]["mean_field"]["HF_method"], self.disk_jockey.data_bulks["structure"]["mean_field_MO_coefs"])
        decoded_H_one = self.structure.decode_H_one(self.disk_jockey.data_bulks["structure"]["mean_field"]["HF_method"], self.disk_jockey.data_bulks["structure"]["mean_field_H_one"])
        decoded_H_two = self.structure.decode_H_two(self.disk_jockey.data_bulks["structure"]["mean_field"]["HF_method"], self.disk_jockey.data_bulks["structure"]["mean_field_H_two"])
        self.structure.load_nodes(
            self.disk_jockey.data_bulks["structure"]["mean_field"],
            decoded_MO_coefs,
            self.disk_jockey.data_bulks["structure"]["mean_field_E_nuc"],
            decoded_H_one,
            decoded_H_two,
            self.disk_jockey.data_bulks["structure"]["mean_field_reference_state_energy"]
            )

        self.log.write("Mean-field properties loaded for each node:")
        self.log.print_itemize({
            "Hartree-Fock method" : self.structure.mean_field["HF_method"],
            "Number of spatial molecular orbitals" : self.structure.mean_field["N_orb"],
            "Electrons with spin alpha" : self.structure.mean_field["S_alpha"],
            "Electrons with spin beta" : self.structure.mean_field["S_beta"],
            })

        # 4. Initialise base node solution

        self.structure.load_base_sol(
            self.disk_jockey.data_bulks["structure"]["base_sol_energies"],
            self.disk_jockey.data_bulks["structure"]["base_sol_coefs"],
            self.disk_jockey.data_bulks["structure"]["base_sol_spins"]
            )

        self.log.write(f"Base node solution loaded (for the {self.structure.N_surf} lowest-energy surfaces)")

        # ---------- Load surfaces

        self.log.enter("Loading surfaces...")
        for surface_label in self.disk_jockey.metadata["structure"]["info"]["surface_labels"]:
            self.load_surface(surface_label)
            self.log.write(f"Loaded surface {surface_label}:")
            self.log.print_itemize({
                "surface index" : self.structure.surfaces[surface_label].meta["i_surf"],
                "calc. method" : self.structure.surfaces[surface_label].meta["method"],
                "calc. duration" : functions.dtstr(self.structure.surfaces[surface_label].meta["duration"])
                })
        self.log.exit()

        self.log.exit()

    def run_FCI_on_structure(self):

        self.log.enter("Running FCI for every node in the geometry structure...")

        FCI_energies, FCI_coefs, FCI_meta = self.structure.find_FCI_surfaces()

        for i_surf in range(self.structure.N_surf):
            self.register_surface(f"FCI[{i_surf}]", FCI_energies[i_surf], {"type" : "occupancy"}, FCI_coefs[i_surf], FCI_meta[i_surf])

        self.log.exit()



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

    def encode_geometry(self):
        flattened_geometry = np.array((self.structure.N_nodes, 3 * (self.structure.mol_bp.N_a - 1)), dtype = float)

        for node_i in range(self.structure.N_nodes):
            for r_i in range(self.structure.mol_bp.N_a - 1):
                for dim_i in range(3):
                    flattened_geometry[node_i][r_i * 3 + dim_i] = self.structure.geometries[node_i].r[r_i][dim_i]
        return(flattened_geometry)

    def decode_geometry(self):
        pass

    def save_data(self):
        self.user_actions += f"save_data\n"
        # ---- Save user log
        self.disk_jockey.commit_datum_bulk("system", "user_actions", self.user_actions)
        self.disk_jockey.commit_datum_bulk("system", "log", self.log.dump())

        # Save list of surfaces
        self.disk_jockey.commit_metadatum("structure", "info", {"surface_labels" : list(self.structure.surfaces.keys())})

        # save diagnostic
        self.disk_jockey.commit_datum_bulk("diagnostics", "diagnostic_log", self.print_diagnostic_log())

        # Save to disk
        self.disk_jockey.save_data()

        self.log.close_journal()

    def load_data(self, what_to_load = None):
        # what_to_load is a list of magic strings

        self.log.enter("Loading data from the disk...", 0)
        self.user_actions += f"load_data\n"

        self.disk_jockey.load_data()

        self.load_structure()

        self.log.exit()


    # ------------------------------- Plotting --------------------------------

    def plot_grid_potential_surfaces(self, surface_labels = None, ax = None, i_d = None, i_surf = None):
        # if surface_labels is None, we plot all known surfaces
        # i_d is the list of dimensions indices along which the plot is drawn.
        # By default this is [0] for 1D grids and [0, 1] for higher-dimensional
        # grids. The pivot is always base_g.

        assert isinstance(self.structure, MGGrid)

        self.user_actions += f"plot_grid_potential_surfaces\n"

        self.log.enter("Plotting potential surfaces...", 5)

        if surface_labels is None:
            surface_labels = list(self.structure.surfaces.keys())

        if ax is None:
            ax = plt.gca()

        if i_d is None:
            # We choose the default option
            if self.structure.D == 1:
                i_d = [0]
            else:
                i_d = [0, 1]

        pivot_i_r = self.structure.find(self.structure.base_can_i)
        subgrid = self.structure.get_subgrid_through_pivot(pivot_i_r, i_d)

        if len(i_d) == 1:
            # 1D graph

            xspace = []
            yspaces = {}
            for sl in surface_labels:
                yspaces[sl] = []

            for i_r in subgrid:
                xspace.append(i_r[i_d[0]])
                can_i = np.vdot(i_r, self.structure.weights)

                for sl in surface_labels:
                    yspaces[sl].append(self.structure.surfaces[sl].E[can_i])

            for sl in surface_labels:
                ax.plot(xspace, yspaces[sl], label = sl)

        if len(i_d) == 2:
            # 2D graph

            xspace = np.arange(self.structure.spans[i_d[0]], dtype = int)
            yspace = np.arange(self.structure.spans[i_d[1]], dtype = int)
            X, Y = np.meshgrid(xspace, yspace)
            X_hr, Y_hr = np.meshgrid(xspace + self.structure.lows[i_d[0]], yspace + self.structure.lows[i_d[1]])

            index_matrix = (np.vdot(np.delete(pivot_i_r, i_d), np.delete(self.structure.weights, i_d))
                + X * self.structure.weights[i_d[0]]
                + Y * self.structure.weights[i_d[1]]
                )

            colors = cm.tab10.colors

            for sl in surface_labels:
                if i_surf is not None:
                    if self.structure.surfaces[sl].meta["i_surf"] not in i_surf:
                        continue

                Z = self.structure.surfaces[sl].E[index_matrix] #np.zeros((self.structure.spans[i_d[1]], self.structure.spans[i_d[0]]))

                if "fancy_label" in self.structure.surfaces[sl].meta:
                    plot_label = self.structure.surfaces[sl].meta["fancy_label"]
                else:
                    plot_label = sl

                #functions.plot_3d_surface(ax, X_hr, Y_hr, Z, plot_label)
                #ax.plot_surface(X_hr, Y_hr, Z, cmap="coolwarm", linewidth=0, antialiased=False, label = plot_label)

                col = colors[self.structure.surfaces[sl].meta["i_surf"] % len(colors)]

                ax.plot_surface(
                    X_hr, Y_hr, Z,
                    color=col,
                    linewidth=0.3,
                    antialiased=True,
                    alpha=0.9,
                    label = plot_label
                )

        ax.legend()

        self.log.exit()



    # --------------------------------------------------------------------------
    # ------------------------------ User methods ------------------------------
    # --------------------------------------------------------------------------

    def find_potential_surfaces(self, method, **kwargs):
            if method in self.find_potential_surfaces_methods.keys():
                return(self.find_potential_surfaces_methods[method](**kwargs))
            else:
                self.log.write(f"ERROR: Unknown potential surface analysis method {method}. Available methods: {self.find_potential_surfaces_methods.keys()}")
                return(None)




