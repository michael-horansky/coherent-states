

from pyscf import gto, scf, cc, ao2mo, ci, fci, pbc

from data_objects.class_Data_Object import Data_Object
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

class DO_Molecule(Data_Object):

    # -------------------------------------------------------------------------
    # --------------------------- Class properties ----------------------------
    # -------------------------------------------------------------------------

    object_type = "molecules"
    data_nodes = {
        "props" : { # physical properties of molecule
                "build" : "json", # instructions on how to build the molecule. Atom geometry, basis, spin, HF_method etc
                "mol" : "json", # derived basic mol properties. M, S_a, S_b, e_tot (HF E), e_labels etc
                "MO" : "json", # MO coefs
                "H" : "json" #one- and two-electron exchange integrals in physicist's notation
            },
        "analysis" : { # solutions obtained for the molecule. Meta can include kwargs (min_err etc) and diagnostics
                "FCI_sol" : "json", # Full CI solution
                "LE_sol" : "json" # Low-excitation solution and its properties
            }
        }

    # -------------------------------------------------------------------------
    # ---------------- Constructors, destructors, descriptors -----------------
    # -------------------------------------------------------------------------

    # ------------------------------ Constructor ------------------------------

    def __init__(self, ID, j = None):
        # ID is the identification label
        # j is a Journal instance passed on to DJ
        super().__init__(ID, j)

        self.pyscf_mol = None # Not to be stored, just for inside use

    # ------------------------------ Descriptors ------------------------------

    def spin_label(self, set_spin = None):
        # Human-readable label for the molecule spin value
        if set_spin is None:
            set_spin = self.dj.data_bulks['props']['build']['spin']
        if set_spin == 0:
            return("singlet")
        elif set_spin == 1:
            return("doublet")
        elif set_spin == 2:
            return("triplet")
        else:
            return(f"(2S+1)={set_spin}")

    def print_LE_sol_analysis(self):
        if self.dj.is_data_initialised['analysis']['LE_sol']:

            self.j.write("The following properties of the low-excitation solution are known:")
            for label in self.dj.data_bulks['analysis']['LE_sol']['props'].keys():
                self.j.write(f'  -{label}')

            if 'RNCS' in self.dj.data_bulks['analysis']['LE_sol']['props'].keys():
                self.j.write(f'RNCS properties:')
                M = self.dj.data_bulks['props']['mol']['M']
                S_a = self.dj.data_bulks['props']['mol']['S_a']
                S_b = self.dj.data_bulks['props']['mol']['S_b']
                z_null_sq = self.dj.data_bulks['analysis']['LE_sol']['props']['RNCS'] ** 2
                cur_RSOPM = self.dj.data_bulks['analysis']['LE_sol']['props']['RSOPM']
                # The diagnostic
                A_i_actual = np.zeros(2 * M)
                final_norm_sq = esp(z_null_sq, S_a + S_b)
                self.j.write(f"  -Final norm squared = {final_norm_sq}")
                for i in range(2 * M):
                    A_i_actual[i] = z_null_sq[i] * esp(z_null_sq, S_a + S_b - 1, omit = [i]) / final_norm_sq

                diagnostic_table = []
                diagnostic_row_names = []
                for i in range(M):
                    diagnostic_table.append([
                        np.round(cur_RSOPM[i][i], 6),
                        np.round(A_i_actual[i], 6),
                        np.round(100 * (1 - A_i_actual[i] / cur_RSOPM[i][i]), 1)
                        ])
                    diagnostic_row_names.append(f"{i + 1}(a)")
                for i in range(M, 2 * M):
                    diagnostic_table.append([
                        np.round(cur_RSOPM[i][i], 6),
                        np.round(A_i_actual[i], 6),
                        np.round(100 * (1 - A_i_actual[i] / cur_RSOPM[i][i]), 1)
                        ])
                    diagnostic_row_names.append(f"{i + 1 - M}(b)")

                self.j.print_table(
                    table_name = "<S_i> diagnostic>",
                    column_names = ["< LE | S_i | LE >", "< Z | S_i | Z >", "Err %"],
                    row_names = diagnostic_row_names,
                    list_of_rows = diagnostic_table
                    )

    def print_mol_desc(self):

        if self.dj.is_data_initialised["props"]["build"]:
            self.j.write("Build:", 3)
            self.j.write("  -atom:")
            atom_geometry_list = self.dj.data_bulks['props']['build']['atom'].split("\n")
            for i in range(len(atom_geometry_list)):
                self.j.write(f"    {atom_geometry_list[i].strip()}", 3)
            self.j.write(f"  -basis: {self.dj.data_bulks['props']['build']['basis']}", 3)
            self.j.write(f"  -spin: {self.dj.data_bulks['props']['build']['spin']} ({self.spin_label()} molecule)", 3)
            self.j.write(f"  -HF method: {self.dj.data_bulks['props']['build']['HF_method']}", 3)

        if self.dj.is_data_initialised["props"]["mol"]:
            self.j.write("Molecule:", 3)
            self.j.write(f"  -{self.dj.data_bulks['props']['mol']['M']} atomic orbitals, each able to hold 2 electrons of opposing spin.", 3)
            self.j.write(f"  -{self.dj.data_bulks['props']['mol']['S']} electrons in total: {self.dj.data_bulks['props']['mol']['S_a']} with spin alpha, {self.dj.data_bulks['props']['mol']['S_b']} with spin beta.", 3)
            self.j.write(f"  -constituent atoms: {', '.join(self.dj.data_bulks['props']['mol']['elements'])}", 3)
            self.j.write(f"  -atomic orbital order: {', '.join(self.dj.data_bulks['props']['mol']['AO_labels'])}", 3)
            self.j.write(f"  -nuclear repulsion energy is {self.dj.data_bulks['props']['mol']['E_nuc']:0.5f}", 3)
            self.j.write(f"  -HF state energy is {self.dj.data_bulks['props']['mol']['E_HF']:0.5f}", 3)

        if self.dj.is_data_initialised["analysis"]["FCI_sol"] or self.dj.is_data_initialised["analysis"]["LE_sol"]:
            self.j.write("Analysis:", 3)
            if self.dj.is_data_initialised["analysis"]["FCI_sol"]:
                self.j.write(f"  -FCI ground state energy: {self.dj.data_bulks['analysis']['FCI_sol']['E']}", 3)
            if self.dj.is_data_initialised["analysis"]["LE_sol"]:
                self.j.write(f"  -LE solution:", 3)
                self.j.write(f"    -method: {self.dj.metadata['analysis']['LE_sol']['method']}", 3)
                self.j.write(f"    -scope: ({'), ('.join(self.dj.metadata['analysis']['LE_sol']['scope'])})", 3)
                self.j.write(f"    -ground state energy: {self.dj.data_bulks['analysis']['LE_sol']['E']}", 3)
                self.print_LE_sol_analysis()
        else:
            self.j.write("No analysis performed on this molecule")

    # --------------------- Converters for property types ---------------------

    def occ_str_to_occ_tuple(self, occ_str):
        # the string is just a binary rep of the address in the FCI vector
        if isinstance(occ_str, tuple):
            return(occ_str) # just to regularise user action
        res = []
        for i in range(len(occ_str) - 1, -1, -1):
            res.append(int(occ_str[i]))
        return(tuple(res))

    def occ_list_to_occ_tuple(self, occ_list):
        if isinstance(occ_list, tuple):
            return(occ_list) # just to regularise user action
        # There may be trailing zeros. Let's get rid of them with a cursed one-liner
        return(tuple(occ_list[:len(occ_list) - occ_list[::-1].index(1)]))

    def occ_list_to_occ_string(self, occ_list):
        # the string that cistring deals with is exactly like the occ_list, but
        # reversed and also a string
        if isinstance(occ_list, str):
            return(occ_list) # just to regularise user action
        res = ""
        for i in range(len(occ_list) - 1, -1, -1):
            res += str(occ_list[i])
        return(res)

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

    def occ_tuple_restore(self, occ_tuple_repr):
        # Restores a tuple which was converted to str directly
        return( tuple([tuple([int(x) for x in occ.split(", ")]) for occ in occ_tuple_repr.strip("()").split("), (")]) )

    # TODO: replace these methods with a single class OBV (occupancy basis vector), which calculates all conventions once and stores them as properties

    # -------------------------------------------------------------------------
    # ---------------------------- Storage access -----------------------------
    # -------------------------------------------------------------------------

    def load_object(self, data_groups = None):
        self.j.enter(f"Loading molecule '{self.ID}' from disk...", 5)
        self.dj.load_data(data_groups)

        # Here, we regularise object types
        if self.dj.is_data_initialised['analysis']['FCI_sol']:
            self.dj.data_bulks['analysis']['FCI_sol']['sol'] = {self.occ_tuple_restore(k): v for k, v in self.dj.data_bulks['analysis']['FCI_sol']['sol'].items()}
        if self.dj.is_data_initialised['analysis']['LE_sol']:
            self.dj.data_bulks['analysis']['LE_sol']['sol'] = {self.str_to_prom_tuple(k): v for k, v in self.dj.data_bulks['analysis']['LE_sol']['sol'].items()}
            self.dj.data_bulks['analysis']['LE_sol']['desc']['scope'] = [tuple(prom_label) for prom_label in self.dj.data_bulks['analysis']['LE_sol']['desc']['scope']]

            for label, val in self.dj.data_bulks['analysis']['LE_sol']['props'].items():
                if isinstance(val, np.ndarray):
                    self.dj.data_bulks['analysis']['LE_sol']['props'][label] = np.ndarray(val)

        self.print_mol_desc()

        self.j.exit()

    def save_object(self, data_groups = None):
        # Here, we regularise object types

        if self.dj.is_data_initialised['analysis']['FCI_sol']:
            self.dj.data_bulks['analysis']['FCI_sol']['sol'] = {str(k): v for k, v in self.dj.data_bulks['analysis']['FCI_sol']['sol'].items()}
        if self.dj.is_data_initialised['analysis']['LE_sol']:
            self.dj.data_bulks['analysis']['LE_sol']['sol'] = {str(k): v for k, v in self.dj.data_bulks['analysis']['LE_sol']['sol'].items()}

            for label, val in self.dj.data_bulks['analysis']['LE_sol']['props'].items():
                if isinstance(val, np.ndarray):
                    self.dj.data_bulks['analysis']['LE_sol']['props'][label] = val.tolist()



        self.dj.save_data(data_groups)

    # -------------------------------------------------------------------------
    # --------------------------- Data calculation ----------------------------
    # -------------------------------------------------------------------------

    def build_pyscf_mol_from_dj(self):
        # If molecule was loaded from storage but pyscf calculations need to be
        # run, this method will build self.pyscf_mol and self.mean_field.
        self.j.write("PySCF Mole object not built. Building...", 5)
        self.pyscf_mol = gto.Mole()
        self.pyscf_mol.build(
            atom = self.dj.data_bulks['props']['build']['atom'],
            basis = self.dj.data_bulks['props']['build']['basis'],
            unit = self.dj.data_bulks['props']['build']['unit'],
            spin = self.dj.data_bulks['props']['build']['spin']
        )
        if self.dj.data_bulks['props']['build']['HF_method'] == "RHF":
            self.mean_field = scf.RHF(self.pyscf_mol).run(verbose = 0)
        elif self.dj.data_bulks['props']['build']['HF_method'] == "UHF":
            mf_object = scf.UHF(self.pyscf_mol)
            mf_object.init_guess = 'atom'
            mf_object.conv_tol = 1e-10
            self.mean_field = mf_object.run(verbose = 0)

    # -------------------------- 'props' calculation --------------------------

    def init_molecule(self, build):
        # build is a dict which is directly being stored in props/build.json
        self.j.enter("Building molecule...")

        self.j.enter("Constructing pyscf.gto.Mole instance...")
        self.pyscf_mol = gto.Mole()

        atom = build["atom"]

        if "basis" in build:
            basis = build["basis"]
        else:
            basis = 'sto-3g'

        if "unit" in build:
            unit = build["unit"]
        else:
            unit = 'Bohr'

        if "spin" in build:
            spin = build["spin"]

            self.pyscf_mol.build(
                atom = atom,
                basis = basis,
                unit = unit,
                spin = spin
            )
        else:
            self.pyscf_mol.build(
                atom = atom,
                basis = basis,
                unit = unit
            )
            spin = self.pyscf_mol.spin

        self.j.write("Calculating 1e integrals...", 1)
        AO_H_one = self.pyscf_mol.intor('int1e_kin', hermi = 1) + self.pyscf_mol.intor('int1e_nuc', hermi = 1)

        self.j.write("Calculating 2e integrals...", 1)
        AO_H_two_chemist = self.pyscf_mol.intor('int2e')

        self.j.exit()


        self.j.enter("Performing mean-field calculations to determine the molecular orbitals...", 1)

        if "HF_method" in build:
            HF_method = build["HF_method"]
            if HF_method == "RHF":
                self.j.write("Mean field method: Restricted Hartree-Fock (selected by user)")
                if spin != 0:
                    self.j.write("WARNING: The molecule is not a singlet, RHF is unsuitable.")
            else:
                self.j.write("Mean field method: Unrestricted Hartree-Fock (selected by user)")
                if spin == 0:
                    self.j.write("NOTE: The molecule is a singlet, RHF may be more suitable.")
        else:
            if spin == 0:
                HF_method = "RHF"
                self.j.write("Mean field method: Restricted Hartree-Fock (determined automatically for a singlet molecule)")
            else:
                HF_method = "UHF"
                self.j.write(f"Mean field method: Unrestricted Hartree-Fock (determined automatically for a {self.spin_label(spin)} molecule)")

        # We now construct AO_H_two as the coefficient tensor in second quantisation according to Szabo & Ostlund: Modern Quantum Chemistry p. 95, Eq. 2.232
        # O_2 = 0.5 * sum_ijkl <ij|kl> f\hc_i f\hc_j f_l f_k
        # Using <ij|kl> = (ik|jl) we have
        # O_ijkl = 0.5 (il|jk)
        # By symmetry: (ij|kl) = (ji|kl) = (ij|lk) = (ji|lk)
        # Hence O_ijkl = O_ljki = O_ikjl = O_lkji


        # We initialise MO_coefs alpha/beta, MO_H_one,two a/b/ab as dicts with spin in key
        MO_coefs = {}
        MO_H_one = {}
        MO_H_two = {}

        # MO_H_two has three elements:
        #   ["a"]_ijkl = <i,a j,a | k,a l,a>
        #   ["b"]_ijkl = <i,b j,b | k,b l,b>
        #   ["ab"]_ijkl = <i,a j,b | k,a l,b>
        #   The second spin-mixed term is obtained by a double transpose; ["ba"]_ijkl = ["ab"]_jilk

        if HF_method == "RHF":
            # Everything is the same in both subspaces
            self.j.write("Finding the molecular orbitals using mean-field approximations...", 1)
            self.mean_field = scf.RHF(self.pyscf_mol).run(verbose = 0)

            MO_coefs["a"] = self.mean_field.mo_coeff
            MO_coefs["b"] = MO_coefs["a"]
            self.j.write(f"Done! Reference state energy is {self.mean_field.e_tot:0.5f}", 1)

            self.j.write("Transforming 1e and 2e integrals to MO basis...", 3)
            MO_H_one["a"] = np.matmul(MO_coefs["a"].T, np.matmul(AO_H_one, MO_coefs["a"]))
            MO_H_one["b"] = MO_H_one["a"]
            MO_H_two_packed = ao2mo.kernel(AO_H_two_chemist, MO_coefs["a"])
            MO_H_two_chemist = ao2mo.restore(1, MO_H_two_packed, MO_coefs["a"].shape[1]) # In mulliken notation: MO_H_two[p][q][r][s] = (pq|rs)

            MO_H_two["a"] = MO_H_two_chemist.transpose(0, 2, 1, 3)# - MO_H_two_chemist.transpose(0, 3, 1, 2)
            MO_H_two["b"] = MO_H_two["a"]
            MO_H_two["ab"] = MO_H_two["a"]

        elif HF_method == "UHF":
            # Coefs and exchange intergrals differ for the two subspaces
            self.j.write("Finding the molecular orbitals using mean-field approximations...", 1)
            mf_object = scf.UHF(self.pyscf_mol)

            mf_object.init_guess = 'atom'  # Atomic initial guess. If doesn't work, run RHF first and then use its result as the initial guess
            #mf_rhf = scf.RHF(self.pyscf_mol).run(verbose=0)
            #mf_uhf = scf.UHF(self.pyscf_mol)
            #mf_uhf.init_guess = 'atom'
            #dm0 = mf_rhf.make_rdm1()
            #mean_field = mf_uhf.kernel(dm0=dm0, verbose=0)
            mf_object.conv_tol = 1e-10 # Tighter convergence

            self.mean_field = mf_object.run(verbose = 0)
            MO_coefs["a"] = self.mean_field.mo_coeff[0]
            MO_coefs["b"] = self.mean_field.mo_coeff[1]

            assert MO_coefs["a"].shape[1] == MO_coefs["b"].shape[1]
            # This is not required but if we remove the constraint we need to
            # firstly restore symmetry, making the mixed H_two["ab"] non-square

            self.j.write(f"Done! Reference state energy is {self.mean_field.e_tot:0.5f}", 1)

            self.j.write("Transforming 1e and 2e integrals to MO basis...", 3)
            MO_H_one["a"] = np.matmul(MO_coefs["a"].T, np.matmul(AO_H_one, MO_coefs["a"]))
            MO_H_one["b"] = np.matmul(MO_coefs["b"].T, np.matmul(AO_H_one, MO_coefs["b"]))
            MO_H_two_packed_alpha = ao2mo.kernel(AO_H_two_chemist, MO_coefs["a"])
            MO_H_two_packed_beta = ao2mo.kernel(AO_H_two_chemist, MO_coefs["b"])
            MO_H_two_packed_ab = ao2mo.kernel(AO_H_two_chemist, (MO_coefs["a"], MO_coefs["a"], MO_coefs["b"], MO_coefs["b"])) # in chemist's the spins are (aa|bb)
            # Removes symmetry to make number access fast
            MO_H_two_chemist_alpha = ao2mo.restore(1, MO_H_two_packed_alpha, MO_coefs["a"].shape[1])
            MO_H_two_chemist_beta = ao2mo.restore(1, MO_H_two_packed_beta, MO_coefs["b"].shape[1])
            MO_H_two_chemist_ab = ao2mo.restore(1, MO_H_two_packed_ab, MO_coefs["a"].shape[1])

            MO_H_two["a"] = MO_H_two_chemist_alpha.transpose(0, 2, 1, 3)
            MO_H_two["b"] = MO_H_two_chemist_beta.transpose(0, 2, 1, 3)
            MO_H_two["ab"] = MO_H_two_chemist_ab.transpose(0, 2, 1, 3)

        self.j.exit()

        self.j.write("Collecting calculated molecule properties...", 5)

        M = self.pyscf_mol.nao
        S_a, S_b = self.pyscf_mol.nelec
        S = S_a + S_b
        elements = self.pyscf_mol.elements
        AO_labels = [x.strip() for x in self.pyscf_mol.ao_labels()]
        E_nuc = self.pyscf_mol.energy_nuc()
        E_HF = self.pyscf_mol.e_tot

        self.dj.commit_datum_bulk("props", "build", {
                "atom" : atom,
                "basis" : basis,
                "unit" : unit,
                "spin" : spin,
                "HF_method" : HF_method
            })

        self.dj.commit_datum_bulk("props", "mol", {
                "M" : M,
                "S" : S,
                "S_a" : S_a,
                "S_b" : S_b,
                "elements" : elements,
                "AO_labels" : AO_labels,
                "E_nuc" : E_nuc,
                "E_HF" : E_HF
            })

        self.dj.commit_datum_bulk("props", "MO", MO_coefs)

        self.dj.commit_datum_bulk("props", "H", {
                "one" : MO_H_one,
                "two" : MO_H_two
            })

        self.j.exit()

    # ------------------------ 'analysis' calculation -------------------------

    # ---- Full CI calculation (using SCF)

    def find_FCI_sol(self):
        self.j.enter("Performing SCF on full CI...", 0)

        if self.dj.is_data_initialised["analysis"]["FCI_sol"]:
            print("Full CI by PySCF already performed. Aborting...")
            return(None)
        if not self.dj.is_data_initialised["props"]["build"]:
            print("Molecule not initialised. Aborting...")
            return(None)

        if self.pyscf_mol is None:
            self.build_pyscf_mol_from_dj()

        if self.dj.data_bulks['props']['build']['HF_method'] == "RHF":
            cisolver = fci.FCI(self.pyscf_mol, self.MO_coefs["a"])
        elif self.dj.data_bulks['props']['build']['HF_method'] == "UHF":
            cisolver = fci.FCI(self.pyscf_mol, (self.dj.data_bulks['props']['MO']['a'], self.dj.data_bulks['props']['MO']['b']))
        else:
            self.j.write("WARNING: Unknown HF method, cannot run full CI calculation.")
            return(None)

        self.j.write("FCI solver initialised...", 0)
        ci_energy, raw_ci_sol = cisolver.kernel()

        # We convert the raw_ci_sol object (which is an FCIvector) into a dict
        # with tuples as keys (tuples represent occupancy strings)

        ci_sol = {}
        norb = cisolver.norb
        n_alpha, n_beta = cisolver.nelec

        self.j.write("Regularising solution as a dict of tuples...", 3)
        for a in range(raw_ci_sol.shape[0]):
            for b in range(raw_ci_sol.shape[1]):
                alpha_occ = self.occ_str_to_occ_tuple("{0:b}".format(fci.cistring.addr2str(norb, n_alpha, a)))
                beta_occ = self.occ_str_to_occ_tuple("{0:b}".format(fci.cistring.addr2str(norb, n_beta, b)))
                key = (alpha_occ, beta_occ)
                ci_sol[key] = float(raw_ci_sol[a, b])

        self.dj.commit_datum_bulk('analysis', 'FCI_sol', {
                'E' : ci_energy,
                'sol' : ci_sol
            })

        self.j.write(f"Ground state energy as calculated by SCF (full configuration) = {ci_energy:0.5f}", 0)
        self.j.exit()

    # ---- Low-excitation calculation (using SCF or explicit diagonalisation)

    def get_states_by_excitation_number(self, N_exc_a, N_exc_b, form = "tuple"):
        # returns a list of all states (as 2-tuples of occ tuples) with the
        # exact number of excitations from the reference state.
        # N_exc_a: number of excitations in the spin-alpha subspace
        # N_exc_b: number of excitations in the spin-beta subspace

        self.j.enter(f"Obtaining all states with {N_exc_a} excitations on spin-alpha and {N_exc_b} excitations on spin-beta...", 4)

        M = self.dj.data_bulks['props']['mol']['M']
        S_a = self.dj.data_bulks['props']['mol']['S_a']
        S_b = self.dj.data_bulks['props']['mol']['S_b']

        # First, check if there are any such states
        if min(S_a, M - S_a) < N_exc_a or min(S_b, M - S_b) < N_exc_b:
            self.j.write("Number of excitations is too large. No such states exist.", 5)
            self.j.exit()
            return([])

        a_from = functions.subset_indices(np.arange(S_a), N_exc_a)
        a_to = functions.subset_indices(np.arange(S_a, M), N_exc_a)
        b_from = functions.subset_indices(np.arange(S_b), N_exc_b)
        b_to = functions.subset_indices(np.arange(S_b, M), N_exc_b)

        res = []

        for i in range(len(a_from)):
            for j in range(len(a_to)):
                for k in range(len(b_from)):
                    for l in range(len(b_to)):
                        cur_state = [[1] * S_a + [0] * (1 + max(a_to[j]) - S_a), [1] * S_b + [0] * (1 + max(b_to[l]) - S_b)]
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

        self.j.write(f"Obtained {len(res)} such states.", 5)
        self.j.exit()
        return(res)

    def find_LE_sol(self, **kwargs):
        # kwargs:
        #     -diag_alg: algorithm to find the projected ground state. Options:
        #          "exp" : explicit diagonalisation
        #          "SCF" : SCF performed on CIS [default]

        # -------------------- Parameter initialisation

        if "diag_alg" in kwargs:
            diag_alg = kwargs["diag_alg"]
        else:
            diag_alg = "SCF"
        hr_diag_alg_label = {
            "exp" : "explicit diagonalisation",
            "SCF" : "SCF on a CISD basis"
            }[diag_alg]

        LE_description = {}

        LE_description["scope"] = [(0, 0), (1, 0), (0, 1), (2, 0), (0, 2), (1, 1)]
        LE_description["params"] = {"diag_alg" : diag_alg}
        LE_description["label"] = "at most two excitations"

        self.j.enter(f"Solving on low-excitation basis using {hr_diag_alg_label}...", 1)

        M = self.dj.data_bulks['props']['mol']['M']
        S_a = self.dj.data_bulks['props']['mol']['S_a']
        S_b = self.dj.data_bulks['props']['mol']['S_b']

        LE_sol = {
            'E' : None,
            'sol' : None
            }

        # We kinda want to design a method which only uses CISD here, since explicit diagonalisation is always costly!
        """
        if diag_alg == "exp":

            self.j.write(f"Obtaining the SE basis....", 1)

            ref_state_a = (1,) * S_a
            ref_state_b = (1,) * S_b
            basis = [(ref_state_a, ref_state_b)]

            for j in range(M - S_a):
                for i in range(S_a):
                    # Note that we omit the trailing zeros to agree with the ci_sol convention
                    basis.append(( (1,) * i + (0,) + (1,) * (S_a - 1 - i) + (0,) * j + (1,),  ref_state_b))

            for l in range(M - S_b):
                for k in range(S_b):
                    # Note that we omit the trailing zeros to agree with the ci_sol convention
                    basis.append((ref_state_a, (1,) * k + (0,) + (1,) * (S_b - 1 - k) + (0,) * l + (1,)))

            basis += self.get_states_by_excitation_number(1, 1)

            for i in range(S_a):
                for j in range(i + 1, S_a):
                    for k in range(M - S_a):
                        for l in range(k + 1, M - S_a):
                            a_occ = [1] * S_a + [0] * (l + 1)
                            a_occ[i] = 0
                            a_occ[j] = 0
                            a_occ[k + S_a] = 1
                            a_occ[l + S_a] = 1
                            basis.append((tuple(a_occ), ref_state_b))
            for i in range(S_b):
                for j in range(i + 1, S_b):
                    for k in range(M - S_b):
                        for l in range(k + 1, M - S_b):
                            b_occ = [1] * S_b + [0] * (l + 1)
                            b_occ[i] = 0
                            b_occ[j] = 0
                            b_occ[k + S_b] = 1
                            b_occ[l + S_b] = 1
                            basis.append((ref_state_a, tuple(b_occ)))

            self.j.write(f"SE basis of length {len(basis)} obtained.")

            H = np.zeros((len(basis), len(basis)))
            msg = f"Explicit Hamiltonian evaluation on SE basis"

            self.j.enter(msg, 1, True, tau_space = np.linspace(0, len(basis) * (len(basis) + 1) / 2, 100 + 1))

            for a in range(len(basis)):
                # Here the diagonal <Z_a|H|Z_a>
                cur_H_overlap = self.get_H_overlap_on_occupancy(self.occ_tuple_to_list(basis[a]), self.occ_tuple_to_list(basis[a]))
                assert cur_H_overlap.imag < 1e-08
                H[a][a] = cur_H_overlap.real
                self.j.update_semaphor_event(a * (a + 1) / 2)

                # Here the off-diagonal, using the fact that H is a Hermitian matrix
                for b in range(a):
                    # We explicitly calculate <Z_a | H | Z_b>
                    cur_H_overlap = self.get_H_overlap_on_occupancy(self.occ_tuple_to_list(basis[a]), self.occ_tuple_to_list(basis[b]))
                    assert cur_H_overlap.imag < 1e-08
                    H[a][b] = cur_H_overlap.real
                    H[b][a] = H[a][b] # np.conjugate(H[a][b])
                    self.j.update_semaphor_event(a * (a + 1) / 2 + b + 1)

            self.j.exit("Evaluation")

            energy_levels, energy_states = np.linalg.eig(H)
            ground_state_index = np.argmin(energy_levels)
            ground_state_energy = energy_levels[ground_state_index]
            ground_state_vector = energy_states[:,ground_state_index] # note that energy_states consist of column vectors, not row vectors

            if "full_CI_sol" in self.checklist:
                self.j.write(f"Ground state energy: {ground_state_energy:0.5f} (compare to full CI: {self.ci_energy:0.5f})", 1)
            else:
                self.j.write(f"Ground state energy: {ground_state_energy:0.5f}", 1)

            # Now, for the heatmap
            self.j.write(f"Obtaining the low-excitation prevalence matrix...", 2)

            res_sol = {(((), ()), ((), ())) : ground_state_vector[0]}

            basis_i = 1
            for a in range(M - S_a):
                for b in range(S_a):
                    res_sol[ (((b,), (a,)), ((), ())) ] = ground_state_vector[basis_i]
                    basis_i += 1
            for a in range(M - S_b):
                for b in range(S_b):
                    res_sol[ (((), ()), ((b,), (a,))) ] = ground_state_vector[basis_i]
                    basis_i += 1

            # order must match the generator
            for i in range(S_a):
                for j in range(M - S_a):
                    for k in range(S_b):
                        for l in range(M - S_b):
                            res_sol[ (((i,), (j,)), ((k,), (l,))) ] = ground_state_vector[basis_i]
                            basis_i += 1

            # alpha alpha
            for i in range(S_a):
                for j in range(i + 1, S_a):
                    for k in range(M - S_a):
                        for l in range(k + 1, M - S_a):
                            res_sol[ (((i, j), (k, l)), ((), ())) ] = ground_state_vector[basis_i]
                            basis_i += 1
            # beta beta
            for i in range(S_b):
                for j in range(i + 1, S_b):
                    for k in range(M - S_b):
                        for l in range(k + 1, M - S_b):
                            res_sol[ (((), ()), ((i, j), (k, l))) ] = ground_state_vector[basis_i]
                            basis_i += 1

            # Mark as solved
            self.check_off("LE_sol")

            self.LE_sol["E"] = ground_state_energy
            self.LE_sol["sol"] = res_sol"""

        if diag_alg == "SCF":
            self.j.enter("SCF on CISD basis...", 3)

            if self.pyscf_mol is None:
                self.build_pyscf_mol_from_dj()

            cisd_solver = self.mean_field.CISD().run()

            if not cisd_solver.converged:
                self.j.write("ERROR: SCF solver failed to converge. Aborting...")
                self.j.exit()
                self.j.exit()
                return(None)

            self.j.write(f"Obtained CISD solution (basis length = {len(cisd_solver.ci)})")

            if self.dj.is_data_initialised['analysis']['FCI_sol']:
                self.j.write(f"Ground state energy: {cisd_solver.e_tot:0.5f} (compare to full CI: {self.dj.data_bulks['analysis']['FCI_sol']['E']:0.5f})", 1)
            else:
                self.j.write(f"Ground state energy: {cisd_solver.e_tot:0.5f}", 1)


            if self.dj.data_bulks['props']['build']['HF_method'] == "UHF":

                # We characterise the coef array by excitations
                c0, c1, c2 = cisd_solver.cisdvec_to_amplitudes(cisd_solver.ci)

                # The shapes of these look different based on the HF method, with extra symmetry assumed for RHF

                c1_a, c1_b = c1
                c2_aa, c2_ab, c2_bb = c2

                self.j.write("Excitation-based Slater determinant sub-bases have lengths:")
                self.j.write(f"  For zero excitation: one state only (overlap {c0:0.5f})")
                self.j.write(f"  For one excitation: {c1_a.shape} on alpha, {c1_b.shape} on beta")
                self.j.write(f"  For two excitations: {c2_aa.shape} on alpha-alpha, {c2_bb.shape} on beta-beta, {c2_ab.shape} on mixed-spin excitations")

                # Now, for the heatmap
                self.j.write(f"Obtaining the low-excitation prevalence matrix...", 2)


                # We do not need to project onto the (no same-spin double excitation) basis,
                # since the fraction cn / c0 remains the same

                res_sol = {(((), ()), ((), ())) : c0}

                for a in range(M - S_a):
                    for b in range(S_a):
                        res_sol[ (((b,), (a,)), ((), ())) ] = c1_a[b][a]
                for a in range(M - S_b):
                    for b in range(S_b):
                        res_sol[ (((), ()), ((b,), (a,))) ] = c1_b[b][a]

                for i in range(M - S_a):
                    for j in range(S_a):
                        for k in range(M - S_b):
                            for l in range(S_b):
                                res_sol[ (((j,), (i,)), ((l,), (k,))) ] = c2_ab[j][l][i][k]

            elif self.dj.data_bulks['props']['build']['HF_method'] == "RHF":
                fci_coefs = ci.cisd.to_fcivec(cisd_solver.ci, M, (S_a, S_b))
                #fci_coefs = ci.cisd.to_fcivec(cisd_solver.ci, cisd_solver.norb, cisd_solver.nelec)
                #fci_coefs = cc.cc2ci.fci_coefs(cisd_solver)
                # fci_coefs is the same kind of object as the output of a full FCI calculation

                res_sol = {}

                self.j.write("Regularising solution as a dict of tuples...", 3)
                # We omit entries which are not singlet or doublet excitations, since they are by definition zero in the CISD sol
                HF_occ = "1" * S_a
                res_sol[(((), ()), ((), ()))] = float(fci_coefs[fci.cistring.str2addr(M, S_a, HF_occ), fci.cistring.str2addr(M, S_b, HF_occ)])

                # singlets
                for i in range(S_a):
                    for j in range(M - S_a):
                        promoted_occ = self.occ_list_to_occ_string([1] * i + [0] + [1] * (S_a - 1 - i) + [0] * j + [1])
                        res_sol[(((i,), (j,)), ((), ()))] = float(fci_coefs[fci.cistring.str2addr(M, S_a, promoted_occ), fci.cistring.str2addr(M, S_b, HF_occ)])
                        res_sol[(((), ()), ((i,), (j,)))] = float(fci_coefs[fci.cistring.str2addr(M, S_a, HF_occ), fci.cistring.str2addr(M, S_b, promoted_occ)])

                # doublets
                for i in range(S_a):
                    for j in range(M - S_a):
                        for k in range(S_b):
                            for l in range(M - S_b):
                                alpha_occ = self.occ_list_to_occ_string([1] * i + [0] + [1] * (S_a - 1 - i) + [0] * j + [1])
                                beta_occ = self.occ_list_to_occ_string([1] * k + [0] + [1] * (S_a - 1 - k) + [0] * l + [1])
                                res_sol[(((i,), (j,)), ((k,), (l,)))] = float(fci_coefs[fci.cistring.str2addr(M, S_a, alpha_occ), fci.cistring.str2addr(M, S_b, beta_occ)])

            # Mark as solved

            LE_sol["E"] = cisd_solver.e_tot
            LE_sol["sol"] = res_sol

            self.j.exit()

        # Derived properties
        LE_sol['desc'] = LE_description
        LE_sol['props'] = self.derive_LE_properties(LE_sol)

        self.dj.commit_datum_bulk('analysis', 'LE_sol', LE_sol)

        self.j.write(f"Success!", 1)
        self.j.exit()
        return(None)

    def derive_LE_properties(self, cur_LE_sol):

        assert "sol" in cur_LE_sol

        self.j.enter("Deriving the reduction matrix for the LE solution...")

        LE_sol_props = {}

        self.j.write("Calculating the norm of the solution on singlets and mixed-spin doublets...")


        M = self.dj.data_bulks['props']['mol']['M']
        S_a = self.dj.data_bulks['props']['mol']['S_a']
        S_b = self.dj.data_bulks['props']['mol']['S_b']

        LE_norm = 0.0
        LE_reduced_norm = 0.0 # sans phi_0
        LE_CS_norm = 0.0

        # no excitation
        LE_norm += cur_LE_sol["sol"][(((), ()), ((), ()))] * cur_LE_sol["sol"][(((), ()), ((), ()))]
        LE_CS_norm += cur_LE_sol["sol"][(((), ()), ((), ()))] * cur_LE_sol["sol"][(((), ()), ((), ()))]

        # singlets
        """if (1, 0) in cur_LE_description["scope"]:
            for a in range(M - S_a):
                for b in range(S_a):
                    cur_z = cur_LE_sol["sol"][ (((b,), (a,)), ((), ())) ]
                    LE_norm += cur_z * cur_z
                    LE_reduced_norm += cur_z * cur_z
        if (0, 1) in cur_LE_description["scope"]:
            for a in range(M - S_b):
                for b in range(S_b):
                    cur_z = cur_LE_sol["sol"][ (((), ()), ((b,), (a,))) ]
                    LE_norm += cur_z * cur_z
                    LE_reduced_norm += cur_z * cur_z

        if (1, 1) in cur_LE_description["scope"]:
            # doublets
            for i in range(M - S_a):
                for j in range(S_a):
                    for k in range(M - S_b):
                        for l in range(S_b):
                            cur_z = cur_LE_sol["sol"][ (((j,), (i,)), ((l,), (k,))) ]
                            LE_norm += cur_z * cur_z
                            LE_reduced_norm += cur_z * cur_z
                            if i == k and j == l:
                                LE_CS_norm += cur_z * cur_z"""
        if (1, 1) in cur_LE_sol['desc']['scope'] and self.dj.data_bulks['props']['build']['HF_method'] == 'RHF':
            # doublets
            for i in range(M - S_a):
                for j in range(S_a):
                    cur_z = cur_LE_sol["sol"][ (((j,), (i,)), ((j,), (i,))) ]
                    LE_CS_norm += cur_z * cur_z
        for cur_prom, cur_z in cur_LE_sol["sol"].items():
            if cur_prom == (((), ()), ((), ())):
                continue
            LE_norm += cur_z * cur_z
            LE_reduced_norm += cur_z * cur_z


        self.j.write(f"Low excitation solution norm squared:")
        self.j.write(f"  -for all singlet and mixed-doublet excitations: <LE | LE> = {LE_norm:0.5f}")
        self.j.write(f"  -for all singlet and mixed-doublet excitations, ignoring the HF state: <LE | LE> = {LE_reduced_norm:0.5f}")
        if self.dj.data_bulks['props']['build']['HF_method'] == 'RHF':
            self.j.write(f"  -for closed-shell mixed-doublet excitations: <LE | LE> = {LE_CS_norm:0.5f}")
        self.j.enter("Calculating reduction matrix", 1, True, tau_space = np.linspace(0, 4 * M * M, 1000 + 1))

        spin_idx_dict = {"a" : 0, "b" : 1}
        spat_to_spin_idx = lambda sigma, i : spin_idx_dict[sigma] * M + i

        LE_sol_props["red"] = np.zeros((2 * M, 2 * M))


        for left_sigma in ["a", "b"]:
            for left_i in range(M):
                for right_sigma in ["a", "b"]:
                    for right_i in range(M):
                        self.j.update_semaphor_event(2 * M * spat_to_spin_idx(left_sigma, left_i) + spat_to_spin_idx(right_sigma, right_i))

                        for left_prom, left_z in cur_LE_sol["sol"].items():
                            left_prom_a, left_prom_b = left_prom
                            left_a_from, left_a_to = left_prom_a
                            left_b_from, left_b_to = left_prom_b
                            left_a_occ = [1] * S_a + [0] * (M - S_a)
                            left_b_occ = [1] * S_b + [0] * (M - S_b)
                            for left_a_from_i in left_a_from:
                                left_a_occ[left_a_from_i] = 0
                            for left_a_to_i in left_a_to:
                                left_a_occ[S_a + left_a_to_i] = 1
                            for left_b_from_i in left_b_from:
                                left_b_occ[left_b_from_i] = 0
                            for left_b_to_i in left_b_to:
                                left_b_occ[S_b + left_b_to_i] = 1

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

                            for right_prom, right_z in cur_LE_sol["sol"].items():
                                right_prom_a, right_prom_b = right_prom
                                right_a_from, right_a_to = right_prom_a
                                right_b_from, right_b_to = right_prom_b
                                right_a_occ = [1] * S_a + [0] * (M - S_a)
                                right_b_occ = [1] * S_b + [0] * (M - S_b)
                                for right_a_from_i in right_a_from:
                                    right_a_occ[right_a_from_i] = 0
                                for right_a_to_i in right_a_to:
                                    right_a_occ[S_a + right_a_to_i] = 1
                                for right_b_from_i in right_b_from:
                                    right_b_occ[right_b_from_i] = 0
                                for right_b_to_i in right_b_to:
                                    right_b_occ[S_b + right_b_to_i] = 1


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
                                    LE_sol_props["red"][spat_to_spin_idx(left_sigma, left_i)][spat_to_spin_idx(right_sigma, right_i)] += left_z * right_z / LE_norm
        """
        # Now for the more intelligent, faster calculation
        # alpha-alpha

        # pi1-pi1
        for i in range(S_a):
            # diagonal


            for j in range(S_b):
                # we just forbid transitionin
        """
        self.j.exit("Calculation")

        if self.dj.data_bulks['props']['build']['HF_method'] == 'RHF':

            self.j.enter("Calculating closed-shell reduction matrix", 1, True, tau_space = np.linspace(0, M * M, 1000 + 1))

            LE_sol_props["CSRM"] = np.zeros((M, M))
            # here, [i][j] corresponds to a simultaneous transition j -> i on both spin subspaces

            for left_i in range(M):
                for right_i in range(M):
                    self.j.update_semaphor_event(M * left_i + right_i)

                    for left_prom, left_z in cur_LE_sol["sol"].items():
                        left_prom_a, left_prom_b = left_prom
                        left_a_from, left_a_to = left_prom_a
                        left_b_from, left_b_to = left_prom_b
                        left_a_occ = [1] * S_a + [0] * (M - S_a)
                        left_b_occ = [1] * S_b + [0] * (M - S_b)
                        for left_a_from_i in left_a_from:
                            left_a_occ[left_a_from_i] = 0
                        for left_a_to_i in left_a_to:
                            left_a_occ[S_a + left_a_to_i] = 1
                        for left_b_from_i in left_b_from:
                            left_b_occ[left_b_from_i] = 0
                        for left_b_to_i in left_b_to:
                            left_b_occ[S_b + left_b_to_i] = 1

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

                        for right_prom, right_z in cur_LE_sol["sol"].items():
                            right_prom_a, right_prom_b = right_prom
                            right_a_from, right_a_to = right_prom_a
                            right_b_from, right_b_to = right_prom_b
                            right_a_occ = [1] * S_a + [0] * (M - S_a)
                            right_b_occ = [1] * S_b + [0] * (M - S_b)
                            for right_a_from_i in right_a_from:
                                right_a_occ[right_a_from_i] = 0
                            for right_a_to_i in right_a_to:
                                right_a_occ[S_a + right_a_to_i] = 1
                            for right_b_from_i in right_b_from:
                                right_b_occ[right_b_from_i] = 0
                            for right_b_to_i in right_b_to:
                                right_b_occ[S_b + right_b_to_i] = 1


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
                                LE_sol_props["CSRM"][left_i][right_i] += left_z * right_z / LE_CS_norm

            self.j.exit("Calculation")

        self.j.enter("Calculating transition prevalence matrix", 1, True, tau_space = np.linspace(0, (M - S_a) * S_a + (M - S_b) * S_b, 1000 + 1))

        spin_idx_dict = {"a" : 0, "b" : 1}
        spat_to_spin_idx = lambda sigma, i : spin_idx_dict[sigma] * M + i

        LE_sol_props["TPM"] = np.zeros((2 * M, M)) # [spin * M + i, j]

        # spin a
        for prom_i in range(S_a):
            for prom_j in range(M - S_a):
                self.j.update_semaphor_event((M - S_a) * prom_i + prom_j)
                # We need to select all coefs from sol which are characterised by i -> j on spin sigma
                for state_prom, state_z in cur_LE_sol["sol"].items():
                    state_prom_a, state_prom_b = state_prom
                    if state_prom_a == ((prom_i,), (prom_j,)):
                        LE_sol_props["TPM"][spat_to_spin_idx("a", prom_i), prom_j + S_a] += state_z * state_z
                        LE_sol_props["TPM"][spat_to_spin_idx("a", prom_j + S_a), prom_i] += state_z * state_z
        # spin b
        for prom_i in range(S_b):
            for prom_j in range(M - S_b):
                self.j.update_semaphor_event((M - S_a) * S_a + (M - S_b) * prom_i + prom_j)
                # We need to select all coefs from sol which are characterised by i -> j on spin sigma
                for state_prom, state_z in cur_LE_sol["sol"].items():
                    state_prom_a, state_prom_b = state_prom
                    if state_prom_a == ((prom_i,), (prom_j,)):
                        LE_sol_props["TPM"][spat_to_spin_idx("b", prom_i), prom_j + S_b] += state_z * state_z
                        LE_sol_props["TPM"][spat_to_spin_idx("b", prom_j + S_b), prom_i] += state_z * state_z



        self.j.exit("Calculation")


        self.j.enter("Calculating spin-reduced reduction matrix", 1, True, tau_space = np.linspace(0, 2 * M * M, 1000 + 1))

        LE_sol_props["SRRM"] = np.zeros((2 * M, M))


        for prom_sigma in ["a", "b"]:
            for prom_i in range(M):
                for prom_j in range(M):
                    self.j.update_semaphor_event(M * spat_to_spin_idx(prom_sigma, prom_i) + prom_j)

                    for left_prom, left_z in cur_LE_sol["sol"].items():
                        left_prom_a, left_prom_b = left_prom
                        left_a_from, left_a_to = left_prom_a
                        left_b_from, left_b_to = left_prom_b
                        left_a_occ = [1] * S_a + [0] * (M - S_a)
                        left_b_occ = [1] * S_b + [0] * (M - S_b)
                        for left_a_from_i in left_a_from:
                            left_a_occ[left_a_from_i] = 0
                        for left_a_to_i in left_a_to:
                            left_a_occ[S_a + left_a_to_i] = 1
                        for left_b_from_i in left_b_from:
                            left_b_occ[left_b_from_i] = 0
                        for left_b_to_i in left_b_to:
                            left_b_occ[S_b + left_b_to_i] = 1

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

                        for right_prom, right_z in cur_LE_sol["sol"].items():
                            right_prom_a, right_prom_b = right_prom
                            right_a_from, right_a_to = right_prom_a
                            right_b_from, right_b_to = right_prom_b
                            right_a_occ = [1] * S_a + [0] * (M - S_a)
                            right_b_occ = [1] * S_b + [0] * (M - S_b)
                            for right_a_from_i in right_a_from:
                                right_a_occ[right_a_from_i] = 0
                            for right_a_to_i in right_a_to:
                                right_a_occ[S_a + right_a_to_i] = 1
                            for right_b_from_i in right_b_from:
                                right_b_occ[right_b_from_i] = 0
                            for right_b_to_i in right_b_to:
                                right_b_occ[S_b + right_b_to_i] = 1


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
                                    LE_sol_props["SRRM"][spat_to_spin_idx(prom_sigma, prom_i)][prom_j] += np.abs(left_z * right_z / LE_norm)
                            if prom_sigma == "b":
                                if np.all(left_b_occ == right_b_occ):
                                    LE_sol_props["SRRM"][spat_to_spin_idx(prom_sigma, prom_i)][prom_j] += np.abs(left_z * right_z / LE_norm)

        self.j.exit("Calculation")


        self.j.enter("Calculating simultanous occupancy proportion matrix", 1, True, tau_space = np.linspace(0, 4 * M * M, 1000 + 1))

        LE_sol_props["SOPM"] = np.zeros((2 * M, 2 * M))

        for i_sigma in ["a", "b"]:
            for i in range(M):
                for j_sigma in ["a", "b"]:
                    for j in range(M):
                        self.j.update_semaphor_event(2 * M * spat_to_spin_idx(i_sigma, i) + spat_to_spin_idx(j_sigma, j))
                        for cur_prom, cur_z in cur_LE_sol["sol"].items():
                            cur_prom_a, cur_prom_b = cur_prom
                            cur_a_from, cur_a_to = cur_prom_a
                            cur_b_from, cur_b_to = cur_prom_b
                            cur_a_occ = [1] * S_a + [0] * (M - S_a)
                            cur_b_occ = [1] * S_b + [0] * (M - S_b)
                            for cur_a_from_i in cur_a_from:
                                cur_a_occ[cur_a_from_i] = 0
                            for cur_a_to_i in cur_a_to:
                                cur_a_occ[S_a + cur_a_to_i] = 1
                            for cur_b_from_i in cur_b_from:
                                cur_b_occ[cur_b_from_i] = 0
                            for cur_b_to_i in cur_b_to:
                                cur_b_occ[S_b + cur_b_to_i] = 1

                            if i_sigma == "a" and cur_a_occ[i] == 0:
                                continue
                            if i_sigma == "b" and cur_b_occ[i] == 0:
                                continue
                            if j_sigma == "a" and cur_a_occ[j] == 0:
                                continue
                            if j_sigma == "b" and cur_b_occ[j] == 0:
                                continue

                            LE_sol_props["SOPM"][spat_to_spin_idx(i_sigma, i)][spat_to_spin_idx(j_sigma, j)] += cur_z * cur_z / LE_norm



        self.j.exit("Calculation")

        self.j.enter("Calculating reduced simultanous occupancy proportion matrix", 1, True, tau_space = np.linspace(0, 4 * M * M, 1000 + 1))

        LE_sol_props["RSOPM"] = np.zeros((2 * M, 2 * M))

        for i_sigma in ["a", "b"]:
            for i in range(M):
                for j_sigma in ["a", "b"]:
                    for j in range(M):
                        self.j.update_semaphor_event(2 * M * spat_to_spin_idx(i_sigma, i) + spat_to_spin_idx(j_sigma, j))
                        for cur_prom, cur_z in cur_LE_sol["sol"].items():
                            if cur_prom == (((), ()), ((), ())):
                                continue
                            cur_prom_a, cur_prom_b = cur_prom
                            cur_a_from, cur_a_to = cur_prom_a
                            cur_b_from, cur_b_to = cur_prom_b
                            cur_a_occ = [1] * S_a + [0] * (M - S_a)
                            cur_b_occ = [1] * S_b + [0] * (M - S_b)
                            for cur_a_from_i in cur_a_from:
                                cur_a_occ[cur_a_from_i] = 0
                            for cur_a_to_i in cur_a_to:
                                cur_a_occ[S_a + cur_a_to_i] = 1
                            for cur_b_from_i in cur_b_from:
                                cur_b_occ[cur_b_from_i] = 0
                            for cur_b_to_i in cur_b_to:
                                cur_b_occ[S_b + cur_b_to_i] = 1

                            if i_sigma == "a" and cur_a_occ[i] == 0:
                                continue
                            if i_sigma == "b" and cur_b_occ[i] == 0:
                                continue
                            if j_sigma == "a" and cur_a_occ[j] == 0:
                                continue
                            if j_sigma == "b" and cur_b_occ[j] == 0:
                                continue

                            LE_sol_props["RSOPM"][spat_to_spin_idx(i_sigma, i)][spat_to_spin_idx(j_sigma, j)] += cur_z * cur_z / LE_reduced_norm



        self.j.exit("Calculation")

        # Finally, we renormalise RSOPM to satisfy the constraint (this should be automatic but floating point imprecision may get the best of us)
        RSOPM_tr = np.trace(LE_sol_props["RSOPM"])
        self.j.write(f"Trace of RSOPM = {RSOPM_tr}; expected value is S = {S_a + S_b}; renormalising...")
        LE_sol_props["RSOPM"] = LE_sol_props["RSOPM"] * (S_a + S_b) / RSOPM_tr


        # Now we calculate the reduced null-coherent state (RNCS)
        # The RNCS is simply a Qubit CS for which every expected occupancy
        # matches the LE solution after projecting out |HF>

        self.j.enter("Calculating the reduced null-coherent state...")

        def cur_scale(c_y):
            return(esp(np.exp(c_y), S_a + S_b))

        # Now, we construct the initial values
        y_0 = np.zeros(2 * M)
        for i in range(2 * M):
            y_0[i] = np.log(LE_sol_props["RSOPM"][i][i])
        y_0_norm_sq = cur_scale(y_0)
        self.j.write(f"Norm squared of initial guess is {y_0_norm_sq}. Renormalising...")
        """"
        z^2 = e^y
        {z|z} = e_S(z^2) = e_S(e^y)
        Let y = y + c
        then e^y = e^c.e^y
        then {z|z} = {z|z} . e^Sc
        We want e^Sc = 1 / cur scale
        hence c = -ln(cur_scale) / S
        """

        y_0 -= np.log(y_0_norm_sq) / (S_a + S_b)
        y_0_norm_sq = cur_scale(y_0)
        self.j.write(f"Norm squared of initial guess was renormalised to {y_0_norm_sq}.")

        # --------------- Gradient descent to find the solution -----------------
        # Parameters
        eta = 0.1
        max_err = 1e-3
        self.j.write("Parameters for the gradient descent:")
        self.j.write(f"  -eta (step size) = {eta}")
        self.j.write(f"  -epsilon (max allowed error) = {max_err}")

        cur_y = np.array(y_0)

        # Now, we converge the solution
        cur_err = 10 * max_err # a bogus val to enter the while loop
        # Cannot track max step because that converges to counteract the renorm step!
        self.j.enter("Calculating z_i to match reduced mode occupancies", 1, True, tau_space = np.linspace(0, 1, 1000 + 1))
        init_err = None
        while(cur_err > max_err):
            # We calculate the step and execute it
            y_step = np.zeros(2 * M)
            cur_norm_sq = cur_scale(cur_y)
            for i in range(2 * M):
                y_step[i] = LE_sol_props["RSOPM"][i][i] - np.exp(cur_y[i]) * esp(np.exp(cur_y), S_a + S_b - 1, omit = [i]) / cur_norm_sq
            cur_y += eta * y_step

            # We calculate the err size
            cur_err = np.sqrt(np.sum(y_step ** 2))
            if init_err is None:
                init_err = cur_err
            else:
                self.j.update_semaphor_event((init_err - cur_err) / (init_err - max_err))

            # We project y onto the norm = 1 surface
            cur_y -= np.log(cur_scale(cur_y)) / (S_a + S_b)

        self.j.exit("Calculation")

        # Now we convert back to the null-CS
        z_null_sq = np.exp(cur_y)
        LE_sol_props["RNCS"] = np.sqrt(z_null_sq)
        # The diagnostic
        A_i_actual = np.zeros(2 * M)
        final_norm_sq = esp(z_null_sq, S_a + S_b)
        self.j.write(f"Final norm squared = {final_norm_sq}")
        for i in range(2 * M):
            A_i_actual[i] = z_null_sq[i] * esp(z_null_sq, S_a + S_b - 1, omit = [i]) / final_norm_sq

        diagnostic_table = []
        diagnostic_row_names = []
        for i in range(M):
            diagnostic_table.append([
                np.round(LE_sol_props["RSOPM"][i][i], 6),
                np.round(A_i_actual[i], 6),
                np.round(100 * (1 - A_i_actual[i] / LE_sol_props["RSOPM"][i][i]), 1)
                ])
            diagnostic_row_names.append(f"{i + 1}(a)")
        for i in range(M, 2 * M):
            diagnostic_table.append([
                np.round(LE_sol_props["RSOPM"][i][i], 6),
                np.round(A_i_actual[i], 6),
                np.round(100 * (1 - A_i_actual[i] / LE_sol_props["RSOPM"][i][i]), 1)
                ])
            diagnostic_row_names.append(f"{i + 1 - M}(b)")

        self.j.print_table(
            table_name = "<S_i> diagnostic>",
            column_names = ["< LE | S_i | LE >", "< Z | S_i | Z >", "Err %"],
            row_names = diagnostic_row_names,
            list_of_rows = diagnostic_table
            )

        self.j.exit()


        self.j.exit()

        return(LE_sol_props)












