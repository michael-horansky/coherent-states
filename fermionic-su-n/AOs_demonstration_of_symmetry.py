import numpy as np

from pyscf import gto, scf, cc

# water molecule
mol = gto.Mole()
mol.build(
    atom = '''O 0 0 0; H  0 1 0; H 0 0 1''',
    basis = 'sto-3g')

# Notice: STO-type orbital basis is a linear combination of GTOs, which are a product of a real-valued radial function and the complex spherical harmonics.
# Since the two-body integrals have full rotational symmetry, and the radial dependence is given by a real-valued function, their values are always real.
# The single-body integrals reflect the angular dependence of the AOs, and as such will be complex-valued.


#one_body_kin = gto.getints('int1e_kin', mol._atm, mol._bas, mol._env)
#one_body_nuc = gto.getints('int1e_nuc', mol._atm, mol._bas, mol._env)
#H_one_body = one_body_kin + one_body_nuc
#print(H_one_body.shape)

print(f"There are {mol.nao} atomic orbitals (i.e. modes)")

# For the AOs, PySCF uses Mulliken's notation: https://gqcg-res.github.io/knowdes/two-electron-integrals.html


# NO SYMMETRY

one_body_kin = mol.intor('int1e_kin')
one_body_nuc = mol.intor('int1e_nuc')
H_one_body = one_body_kin + one_body_nuc
print(H_one_body.shape)

# Two-body: electron repulsion integrals
# Not gonna use this: twice as many as there should be? Maybe accounts for spin up and spin down for each one??
# H_ERI = gto.getints('int2e', mol._atm, mol._bas, mol._env)
# print(H_ERI.shape)

H_ERI = mol.intor('int2e', aosym = "s1")
print(H_ERI.shape)

# YES SYMMETRY

herm_one_body_kin = mol.intor('int1e_kin', hermi = 1)
herm_one_body_nuc = mol.intor('int1e_nuc', hermi = 1)
herm_H_one_body = herm_one_body_kin + herm_one_body_nuc
print(herm_H_one_body.shape)

print((np.round(H_one_body, 5) == np.round(herm_H_one_body, 5)).all()) # Works!

# As for the s8 symmetry:
#   The exchange i <-> j, k <-> l is trivial (this is the s4 symmetry). This means we can always choose j >= i, l >= k.
sym_H_ERI = mol.intor('int2e', aosym = "s8")
print(sym_H_ERI.shape)



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

print(len(all_indices))

print(all_indices)

all_agree = True

for index_set_i in range(len(all_indices)):
    index_set = all_indices[index_set_i]
    print(f"Index {index_set}")
    print(f"  Symmetrised value: {sym_H_ERI[index_set_i]}")
    p = index_set[0][0]
    q = index_set[0][1]
    r = index_set[1][0]
    s = index_set[1][1]
    print(f"  Unsymmetrised values:")
    print(f"    (pq | rs) = {H_ERI[p][q][r][s]}, (qp | rs) = {H_ERI[q][p][r][s]}, (pq | sr) = {H_ERI[p][q][s][r]}, (qp | sr) = {H_ERI[q][p][s][r]}")
    print(f"    (rs | pq) = {H_ERI[r][s][p][q]}, (sr | pq) = {H_ERI[s][r][p][q]}, (rs | qp) = {H_ERI[r][s][q][p]}, (sr | qp) = {H_ERI[s][r][q][p]}")

    decimals = 4
    if len({np.round(sym_H_ERI[index_set_i], decimals), np.round(H_ERI[p][q][r][s], decimals), np.round(H_ERI[q][p][r][s], decimals), np.round(H_ERI[p][q][s][r], decimals), np.round(H_ERI[q][p][s][r], decimals), np.round(H_ERI[r][s][p][q], decimals), np.round(H_ERI[s][r][p][q], decimals), np.round(H_ERI[r][s][q][p], decimals), np.round(H_ERI[s][r][q][p], decimals)}) > 1:
        all_agree = False

    print(f"  Symmetrised value accessed explicitly: {sym_H_ERI[int(p * (p * p * p + 2 * p * p + 3 * p + 2) / 8 + p * q * (p + 1) / 2 + q * (q + 1) / 2   + r * (r + 1) / 2 + s )]}")
    if sym_H_ERI[int(p * (p * p * p + 2 * p * p + 3 * p + 2) / 8 + p * q * (p + 1) / 2 + q * (q + 1) / 2   + r * (r + 1) / 2 + s )] != sym_H_ERI[index_set_i]:
        all_agree = False

if all_agree:
    print("All agree!")

""" this is for the s4 property
x = 0
for a in range(mol.nao):
    for b in range(a, mol.nao):
        for c in range(mol.nao):
            for d in range(c, mol.nao):
                x += 1
print(x)"""
