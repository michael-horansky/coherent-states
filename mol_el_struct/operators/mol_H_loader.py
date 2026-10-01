# This script uses PySCF to access the one- and two-electron exchange integrals
# for a pyscf.gto.Mole object and translates them into a SpinOperator object.

import numpy as np
from utils.class_Journal import color
from SpinOperator import SpinOperator

from pyscf import scf, ao2mo

import functions






        # Maybe also have data_group_metadata, with one entry per data group? So the user doesn't have to remember what each data group is

    # -------------- Journal methods ---------------
    def log_write(self, msg, v = -1):
        if self.journal_instance is not None:
            self.journal_instance.write(msg, v, agent = "DJ")

    def log_enter(self, msg, v = -1):
        if self.journal_instance is not None:
            self.journal_instance.enter(msg, v, agent = "DJ")

    def log_exit(self):
        if self.journal_instance is not None:
            self.journal_instance.exit()


def load_Hamiltonian(mol, requested_HF_method = "default", N_MO = None, log = None):
    # N_MO restricts the number of molecular orbitals. If None, the full basis as determined by SCF is used.

    # We register an agent for this loading process
    if log is not None: log.register_agent_tag("HLoad", {"label" : "H loader", "color" : color.CYAN})

    # We calculate exchange integrals
    if log is not None: log.enter("Calculating exchange integrals...", 3, agent = "HLoad")

    if log is not None: log.write("Calculating one-electron exchange integrals...", 4, agent = "HLoad")
    AO_H_one = mol.intor('int1e_kin', hermi = 1) + mol.intor('int1e_nuc', hermi = 1)

    if log is not None: log.write("Calculating two-electron exchange integrals...", 4, agent = "HLoad")
    #self.H_two = self.mol.intor('int2e', aosym = "s1")
    AO_H_two_chemist = self.mol.intor('int2e')
    # <ij|kl> = (ik|jl)

    if log is not None: log.exit()

    if log is not None: log.enter("Performing mean-field calculations to determine the molecular orbitals...", 3, agent = "HLoad")

    if requested_HF_method == "RHF":
        HF_method = requested_HF_method
        if log is not None: log.write("Mean field method: Restricted Hartree-Fock (selected by user)", 4, agent = "HLoad")
        if mol.spin != 0:
            if log is not None: log.warning("The molecule is not a singlet, RHF is unsuitable.", agent = "HLoad")
    elif requested_HF_method == "UHF":
        HF_method = requested_HF_method
        if log is not None: log.write("Mean field method: Unrestricted Hartree-Fock (selected by user)", 4, agent = "HLoad")
    elif requested_HF_method == "default":
        if mol.spin == 0:
            HF_method = "RHF"
            if log is not None: log.write("Mean field method: Restricted Hartree-Fock (determined automatically for a singlet molecule)", 4, agent = "HLoad")
        else:
            HF_method = "UHF"
            if mol.spin == 0:
                spin_label = "singlet"
            elif mol.spin == 1:
                spin_label = "doublet"
            elif mol.spin == 2:
                spin_label = "triplet"
            else:
                spin_label = f"(2S+1)={mol.spin}"
            if log is not None: log.write(f"Mean field method: Unrestricted Hartree-Fock (determined automatically for a {spin_label} molecule)", 4, agent = "HLoad")
    else:
        raise Exception(f"Unknown Hartree-Fock method: {requested_HF_method}. Valid options: 'RHF', 'UHF', 'default'.")

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
        if log is not None: log.write("Finding the molecular orbitals using the mean-field approximation...", 4, agent = "HLoad")
        mean_field = scf.RHF(mol).run(verbose = 0)

        MO_coefs["a"] = mean_field.mo_coeff
        # We trim the number of MOs
        if N_MO is not None:
            MO_coefs["a"] = MO_coefs["a"][:, :N_MO]

        MO_coefs["b"] = MO_coefs["a"]
        reference_state_energy = mean_field.e_tot
        if log is not None: log.write(f"Done! Reference state energy is {reference_state_energy:0.5f}", 4, agent = "HLoad")

        if log is not None: log.write("Transforming 1e and 2e integrals to MO basis...", 4, agent = "HLoad")
        MO_H_one["a"] = np.matmul(MO_coefs["a"].T, np.matmul(AO_H_one, MO_coefs["a"]))
        MO_H_one["b"] = MO_H_one["a"]
        MO_H_two_packed = ao2mo.kernel(AO_H_two_chemist, MO_coefs["a"])
        MO_H_two_chemist = ao2mo.restore(1, MO_H_two_packed, MO_coefs["a"].shape[1]) # In mulliken notation: MO_H_two[p][q][r][s] = (pq|rs)

        MO_H_two["a"] = MO_H_two_chemist.transpose(0, 2, 1, 3)# - MO_H_two_chemist.transpose(0, 3, 1, 2)
        MO_H_two["b"] = MO_H_two["a"]
        MO_H_two["ab"] = MO_H_two["a"]

    elif HF_method == "UHF":
        # Coefs and exchange intergrals differ for the two subspaces
        if log is not None: log.write("Finding the molecular orbitals using the mean-field approximation...", 4, agent = "HLoad")
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

        # We trim the number of MOs
        if N_MO is not None:
            MO_coefs["a"] = MO_coefs["a"][:, :N_MO]
            MO_coefs["b"] = MO_coefs["b"][:, :N_MO]

        reference_state_energy = mean_field.e_tot
        if log is not None: log.write(f"Done! Reference state energy is {reference_state_energy:0.5f}", 4, agent = "HLoad")

        if log is not None: log.write("Transforming 1e and 2e integrals to MO basis...", 4, agent = "HLoad")
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

    if log is not None: log.exit()

    if log is not None: log.enter("Constructing a SpinOperator object equivalent to the Hamiltonian...", 3, agent = "HLoad")

    N_MO = MO_H_one["a"].shape[0]
    def so_to_m(s, i):
        # Converts spin-orbital index to mode index
        if s == 'a':
            return(i)
        elif s == 'b':
            return(N_MO + i)

    def m_to_so(i):
        # Converts mode index to spin-orbital index
        if i < N_MO:
            return('a', i)
        else:
            return('b', i - N_MO)

    H_terms = {}

    if log is not None: log.write("Preparing the nuclear energy term...", 4, agent = "HLoad")

    H_terms[(frozenset(), frozenset())] = mol.energy_nuc()

    if log is not None: log.write("Preparing the one-electron exchange terms...", 4, agent = "HLoad")

    for i in range(N_MO):
        for j in range(N_MO):
            for s in ['a', 'b']:
                H_terms[(frozenset({so_to_m(s, i)}), frozenset({so_to_m(s, j)}))] = MO_H_one[s][i][j]

    if log is not None: log.write("Preparing the two-electron exchange terms...", 4, agent = "HLoad")

    # equal spin
    c_pairs = functions.subset_indices(np.arange(N_MO), 2)
    a_pairs = functions.subset_indices(np.arange(N_MO), 2)

    for c_pair in c_pairs:
        for a_pair in a_pairs:

            # <ij|kl> -> < c_pair[1], c_pair[0] | a_pair[0], a_pair[1] >
            i = c_pair[0]
            j = c_pair[1]
            k = a_pair[0]
            l = a_pair[1]

            for s in ['a', 'b']:
                H_terms[(frozenset({so_to_m(s, i), so_to_m(s, j)}), frozenset({so_to_m(s, k), so_to_m(s, l)}))] = MO_H_two[s][i][j][k][l] - MO_H_two[s][i][j][l][k]

    # mixed spin
    for i in range(self.N_MO):
        for j in range(self.N_MO):
            for k in range(self.N_MO):
                for l in range(self.N_MO):
                    H_terms[(frozenset({so_to_m('a', i), so_to_m('b', j)}), frozenset({so_to_m('a', k), so_to_m('b', l)}))] = 0.5 * MO_H_two['ab'][i][j][k][l]
                    H_terms[(frozenset({so_to_m('b', i), so_to_m('a', j)}), frozenset({so_to_m('b', k), so_to_m('a', l)}))] = 0.5 * MO_H_two['ab'][j][i][l][k]

    H_operator = SpinOperator(H_terms, N_MO, N_MO, sym = (HF_method == "RHF"))


