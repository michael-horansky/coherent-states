"""
slater_condon_spin_orb.py

- Build spin-orbital one- and two-electron integrals from PySCF AO integrals (aosym='s1').
- Compute determinant energy using Slater-Condon.
- Build antisymmetrized two-electron tensor B_{pqrs} in spin-orbital basis.

Example usage at the bottom shows comparison of the determinant energy for your "occupied-first"
mode ordering to PySCF HF energy (note: if your determinant isn't the HF determinant the numbers
will differ, but the Slater-Condon evaluation will be correct).

Author: Copied-style example for your workflow.
"""

import numpy as np
from pyscf import gto, scf

def build_spin_orbital_h(h_ao):
    """
    Build spin-orbital one-electron integrals from spatial AO h_ao (nao x nao).
    spin-orbital index ordering: i_spin = 2*ao + spin (spin=0 alpha, 1 beta).
    Returns: h_spin (nspin x nspin), where nspin = 2*nao
    """
    nao = h_ao.shape[0]
    nspin = 2 * nao
    h_spin = np.zeros((nspin, nspin))
    # block-diagonal: h_spin[(2p+s),(2q+s)] = h_ao[p,q] for s in {0,1}
    for p in range(nao):
        for q in range(nao):
            val = h_ao[p, q]
            h_spin[2*p + 0, 2*q + 0] = val  # alpha block
            h_spin[2*p + 1, 2*q + 1] = val  # beta  block
    return h_spin

def build_spin_orbital_eri_from_spatial(eri_ao):
    """
    Convert spatial AO two-electron integrals eri_ao[p,q,r,s] (chemists' notation (pq|rs))
    into spin-orbital eri_spin[P,Q,R,S] such that
    (p_sigma q_tau | r_ups s_xi) = (pq|rs) * delta_{sigma,ups} * delta_{tau,xi}

    Returns: eri_spin with shape (nspin,nspin,nspin,nspin)
    """
    nao = eri_ao.shape[0]
    nspin = 2 * nao
    eri_spin = np.zeros((nspin, nspin, nspin, nspin))
    for p in range(nao):
        for q in range(nao):
            for r in range(nao):
                for s in range(nao):
                    val = eri_ao[p, q, r, s]
                    # assign to spin-orbital indices with spin matches: sigma==ups, tau==xi
                    for sig in (0, 1):
                        for tau in (0, 1):
                            P = 2*p + sig
                            Q = 2*q + tau
                            R = 2*r + sig  # must match sigma (ups)
                            S = 2*s + tau  # must match tau (xi)
                            eri_spin[P, Q, R, S] = val
    return eri_spin

def antisymmetrize_two_body(eri_spin):
    """
    Given spin-orbital eri_spin in chemists' notation (pq|rs),
    compute antisymmetrized tensor B_{pqrs} = (pq|rs) - (pr|qs),
    returning B with same shape.
    """
    # eri_spin shape (n,n,n,n)
    # B[p,q,r,s] = eri[p,q,r,s] - eri[p,r,q,s]
    B = eri_spin - eri_spin.transpose(0,2,1,3)
    return B

def det_energy_slater_condon(h_spin, eri_spin, occ_list):
    """
    Compute energy of a Slater determinant with occupied spin-orbital indices 'occ_list'
    using Slater-Condon:
    E = sum_{i in occ} h_{ii} + 0.5 * sum_{i in occ} sum_{j in occ} [ (ii|jj) - (ij|ji) ]
    where (pq|rs) are spin-orbital integrals stored in eri_spin[p,q,r,s].
    """
    E1 = 0.0
    E2 = 0.0
    occ = list(occ_list)
    # one-electron
    for i in occ:
        E1 += h_spin[i, i]
    # two-electron: double sum (including i==j); include 0.5 to avoid double counting
    for i in occ:
        for j in occ:
            coul = eri_spin[i, i, j, j]    # (ii|jj)
            exch = eri_spin[i, j, j, i]    # (ij|ji)
            E2 += (coul - exch)
    E2 *= 0.5
    return E1 + E2

# -------------------------
# Example runnable snippet
# -------------------------
if __name__ == "__main__":
    # Build molecule (example: water - replace with your molecule)
    mol = gto.Mole()
    mol.build(
    atom = '''H 0 0 0; H  0 0 1''',
    basis = 'sto-3g')

    # One- and two-electron AO integrals
    h_core_ao = mol.intor('int1e_kin') + mol.intor('int1e_nuc')   # nao x nao
    eri_ao = mol.intor('int2e', aosym='s1')                      # full (pq|rs), shape (nao,nao,nao,nao)

    nao = mol.nao
    nspin = 2 * nao

    # Build spin-orbital integrals
    h_spin = build_spin_orbital_h(h_core_ao)      # nspin x nspin
    eri_spin = build_spin_orbital_eri_from_spatial(eri_ao)  # nspin^4

    # Choose an occupied list for your "null" determinant.
    # Example: occupy the first Nspin_occ spin-orbitals (this assumes you ordered your modes that way).
    # Here we use mol.tot_electrons() as number of electrons (spin-orbitals occupied)
    nocc = mol.tot_electrons()     # number of electrons = number of occupied spin-orbitals
    occ_list = list(range(nocc))   # e.g., [0,1,2,...,nocc-1]  <-- adapt to your mode ordering

    print("Occupied AOs:")
    for i in occ_list:
        spin = i % 2
        ao = int((i - spin) / 2)
        print(mol.ao_labels()[ao])

    # Compute determinant energy via Slater-Condon
    E_det = det_energy_slater_condon(h_spin, eri_spin, occ_list)
    print("Nuclear binding energy               : ", mol.energy_nuc())
    print("Determinant energy (Slater-Condon)   : ", E_det, "Hartree")
    print("Total null state energy:             : ", E_det + mol.energy_nuc(), "Hartree")

    # For comparison compute RHF energy from PySCF (this performs SCF in MO basis)
    mf = scf.RHF(mol).run()
    print("PySCF RHF total energy                : ", mf.e_tot, "Hartree")

    # Also construct antisymmetrized two-electron tensor in spin-orbital basis
    B_spin = antisymmetrize_two_body(eri_spin)
    # Example check: for spin-orbital diagonal p==q==r==s, B should be zero
    print("Max abs of B_spin tensor: ", np.max(np.abs(B_spin)))
