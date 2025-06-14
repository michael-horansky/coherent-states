

from pyscf import gto
from krylov_ground_state import ground_state_solver

# water molecule
water_mol = gto.Mole()
water_mol.build(
    atom = '''O 0 0 0; H  0 1 0; H 0 0 1''',
    basis = 'sto-3g')

hydrogen_mol = gto.Mole()
hydrogen_mol.build(
    atom = '''H 0 0 0; H  0 0 1''',
    basis = 'sto-3g')

hydrogen_solver = ground_state_solver("hydrogen_gas")
hydrogen_solver.initialise_molecule(water_mol)
hydrogen_solver.find_ground_state("krylov", dt = 5)




