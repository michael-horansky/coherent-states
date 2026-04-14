

from pyscf import gto

from data_objects.class_Data_Object import Data_Object
from utils.class_Disk_Jockey import Disk_Jockey


class DO_Molecule(Data_Object):

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

    def __init__(self, ID, j = None):
        # ID is the identification label
        # j is a Journal instance passed on to DJ
        super().__init__(ID, j)

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

    def print_mol_desc(self):
        self.j.write(f"Build:", 3)

        self.j.write("  -atom:")
        atom_geometry_list = self.dj.data_bulks['props']['build']['atom'].split("\n")
        for i in range(len(atom_geometry_list)):
            self.j.write(f"    {atom_geometry_list[i].strip()}", 3)
        self.j.write(f"  -basis: {self.dj.data_bulks['props']['build']['basis']}", 3)
        self.j.write(f"  -spin: {self.dj.data_bulks['props']['build']['spin']} ({self.spin_label()} molecule)", 3)
        self.j.write(f"  -HF method: {self.dj.data_bulks['props']['build']['HF_method']}", 3)

        self.j.write("Molecule:", 3)
        self.j.write(f"  -{self.dj.data_bulks['props']['mol']['M']} atomic orbitals, each able to hold 2 electrons of opposing spin.", 3)
        self.j.write(f"  -{self.dj.data_bulks['props']['mol']['S']} electrons in total: {self.dj.data_bulks['props']['mol']['S_a']} with spin alpha, {self.dj.data_bulks['props']['mol']['S_b']} with spin beta.", 3)
        self.j.write(f"  -constituent atoms: {self.dj.data_bulks['props']['mol']['elements'].join(',')}", 3)
        AO_hr = ', '.join([x.strip() for x in self.dj.data_bulks['props']['mol']['AO_labels']])
        self.j.write(f"  -atomic orbital order: {AO_hr}", 3)
        self.j.write(f"  -nuclear repulsion energy is {self.dj.data_bulks['props']['mol']['E_nuc']:0.5f}", 3)
        self.j.write(f"  -HF state energy is {self.dj.data_bulks['props']['mol']['E_HF']:0.5f}", 3)

        self.j.write("Analysis:", 3)
        self.j.write(f"  -FCI ground state energy: {self.dj.data_bulks['analysis']['FCI_sol']['E']}", 3)
        self.j.write(f"  -LE solution:", 3)
        self.j.write(f"    -method: {self.dj.metadata['analysis']['LE_sol']['method']}", 3)
        self.j.write(f"    -scope: ({self.dj.metadata['analysis']['LE_sol']['scope'].join('), (')})", 3)
        self.j.write(f"    -ground state energy: {self.dj.data_bulks['analysis']['LE_sol']['E']}", 3)

    def load_object(self, data_groups = None):
        self.j.enter(f"Loading molecule '{self.ID}' from disk...", 5)
        self.dj.load_data(data_groups)

        self.print_mol_desc()

        self.j.exit()

    def save_object(self, data_groups = None):
        self.dj.save_data(data_groups)

    def init_molecule(self, build):
        # build is a dict which is directly being stored in props/build.json
        self.j.enter("Building molecule...")

        self.j.enter("Constructing pyscf.gto.Mole instance...")
        pyscf_mol = gto.Mole()

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

            pyscf_mol.build(
                atom = atom,
                basis = basis,
                unit = unit,
                spin = spin
            )
        else:
            pyscf_mol.build(
                atom = atom,
                basis = basis,
                unit = unit
            )
            spin = pyscf_mol.spin

        self.j.write("Calculating 1e integrals...", 1)
        AO_H_one = pyscf_mol.intor('int1e_kin', hermi = 1) + pyscf_mol.intor('int1e_nuc', hermi = 1)

        self.j.write("Calculating 2e integrals...", 1)
        AO_H_two_chemist = pyscf_mol.intor('int2e')

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
            log.write("Finding the molecular orbitals using mean-field approximations...", 1)
            mean_field = scf.RHF(mol).run(verbose = 0)

            MO_coefs["a"] = mean_field.mo_coeff
            MO_coefs["b"] = MO_coefs["a"]
            reference_state_energy = mean_field.e_tot
            log.write(f"Done! Reference state energy is {reference_state_energy:0.5f}", 1)

            log.write("Transforming 1e and 2e integrals to MO basis...", 3)
            MO_H_one["a"] = np.matmul(MO_coefs["a"].T, np.matmul(AO_H_one, MO_coefs["a"]))
            MO_H_one["b"] = MO_H_one["a"]
            MO_H_two_packed = ao2mo.kernel(AO_H_two_chemist, MO_coefs["a"])
            MO_H_two_chemist = ao2mo.restore(1, MO_H_two_packed, MO_coefs["a"].shape[1]) # In mulliken notation: MO_H_two[p][q][r][s] = (pq|rs)

            MO_H_two["a"] = MO_H_two_chemist.transpose(0, 2, 1, 3)# - MO_H_two_chemist.transpose(0, 3, 1, 2)
            MO_H_two["b"] = MO_H_two["a"]
            MO_H_two["ab"] = MO_H_two["a"]

            # We construct the second quantised space as occupied \oplus unoccupied orbitals
            occ_orbs_alpha = [i for i, o in enumerate(mean_field.mo_occ) if o > 0]
            occ_orbs_beta = occ_orbs_alpha

        elif HF_method == "UHF":
            # Coefs and exchange intergrals differ for the two subspaces
            log.write("Finding the molecular orbitals using mean-field approximations...", 1)
            mf_object = scf.UHF(mol)

            mf_object.init_guess = 'atom'  # Atomic initial guess. If doesn't work, run RHF first and then use its result as the initial guess
            #mf_rhf = scf.RHF(mol).run(verbose=0)
            #mf_uhf = scf.UHF(mol)
            #mf_uhf.init_guess = 'atom'
            #dm0 = mf_rhf.make_rdm1()
            #mean_field = mf_uhf.kernel(dm0=dm0, verbose=0)
            mf_object.conv_tol = 1e-10 # Tighter convergence

            mean_field = mf_object.run(verbose = 0)
            MO_coefs["a"] = mean_field.mo_coeff[0]
            MO_coefs["b"] = mean_field.mo_coeff[1]

            assert MO_coefs["a"].shape[1] == MO_coefs["b"].shape[1]
            # This is not required but if we remove the constraint we need to
            # firstly restore symmetry, making the mixed H_two["ab"] non-square

            reference_state_energy = mean_field.e_tot
            log.write(f"Done! Reference state energy is {reference_state_energy:0.5f}", 1)

            log.write("Transforming 1e and 2e integrals to MO basis...", 3)
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

            # We construct the second quantised space as occupied \oplus unoccupied orbitals
            occ_orbs_alpha = [i for i, o in enumerate(mean_field.mo_occ[0]) if o > 0]
            occ_orbs_beta = [i for i, o in enumerate(mean_field.mo_occ[1]) if o > 0]

        # TODO note that by using RHF, we assume N_alpha = N_beta. We can generalise the process by using UHF,
        # but this would mean using separate MO coeffs for alpha and beta subspaces

        self.log.exit()






        M = pyscf_mol.nao
        S_a, S_b = pyscf_mol.nelec
        S = S_a + S_b
        elements = pyscf_mol.elements
        AO_labels = pyscf_mol.ao_labels()
        E_nuc = pyscf_mol.energy_nuc()

        self.dj.commit_datum_bulk("props", "build", {
                "atom" : atom,
                "basis" : basis,
                "unit" : unit,
                "spin" : spin,
                "HF_method" : HF_method
            })








