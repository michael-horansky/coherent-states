

from pyscf import gto, scf, fci
from krylov_ground_state import ground_state_solver


# -------------------------- Li2

# Li2: bond length ≈ 2.673 Å = 5.0512375675 Bohr
# placed symmetrically about origin: half-distance ≈ 2.5256187838 Bohr
li2_mol = gto.Mole()
li2_mol.build(
    atom = '''
    Li   0.0000000000   0.0000000000  -2.5256187838
    Li   0.0000000000   0.0000000000   2.5256187838
    ''',
    basis = 'sto-3g'
)


# -------------------------------- BeH

# BeH: bond length ≈ 1.342 Å = 2.5360122767 Bohr
# place Be at origin and H along +z
beh_mol = gto.Mole()
beh_mol.build(
    atom = '''
    Be   0.0000000000   0.0000000000   0.0000000000
    H    0.0000000000   0.0000000000   2.5360122767
    ''',
    basis = 'sto-3g',
    spin = 1  # BeH is a doublet (5 electrons) -> 1 unpaired electron
)



# ------------------------------ N2

# N2: bond length ≈ 1.098 Å = 2.0749191355 Bohr
# placed symmetrically about origin: half-distance ≈ 1.0374595677 Bohr
n2_mol = gto.Mole()
n2_mol.build(
    atom = '''
    N    0.0000000000   0.0000000000  -1.0374595677
    N    0.0000000000   0.0000000000   1.0374595677
    ''',
    basis = 'sto-3g'
)

# ---------------------------------- C2

# C2: bond length ≈ 1.2425 Å = 2.3479845408 Bohr
# placed symmetrically about origin: half-distance ≈ 1.1739922704 Bohr
c2_mol = gto.Mole()
c2_mol.build(
    atom = '''
    C    0.0000000000   0.0000000000  -1.1739922704
    C    0.0000000000   0.0000000000   1.1739922704
    ''',
    basis = 'sto-3g'
)

# -------------------------------- NO

# NO: bond length ≈ 1.151 Å = 2.1750746129 Bohr
# placed symmetrically about origin: half-distance ≈ 1.0875373064 Bohr
no_mol = gto.Mole()
no_mol.build(
    atom = '''
    N    0.0000000000   0.0000000000  -1.0875373064
    O    0.0000000000   0.0000000000   1.0875373064
    ''',
    basis = 'sto-3g',
    spin = 1  # NO is a doublet (15 electrons) -> 1 unpaired electron
)

benchmark_molecules = [li2_mol]


for i in range(len(benchmark_molecules)):
    mol_solver = ground_state_solver(f"bench_mol_{i}")
    mol_solver.initialise_molecule(benchmark_molecules[i])
    mol_solver.find_ground_state("sampling", N = 10, lamb = None, delta = 1e-2)
