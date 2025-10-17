

from pyscf import gto, scf, fci
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
#hydrogen_solver.find_ground_state("sampling", N = 10, lamb = None, delta = 1e-3)



hf = scf.RHF(hydrogen_mol)          # Create the HF object
hf.kernel()  # Perform the SCF calculation
#energy = hf.kernel()                # Run the SCF calculation and get energy

#print(f"Ground-state electronic energy: {energy} Hartree")

# Step 3: Perform Full CI calculation
cisolver = fci.FCI(hydrogen_mol, hf.mo_coeff)  # Full CI solver with HF orbitals
ci_energy, ci_wavefunction = cisolver.kernel()
print(f"Full CI ground-state energy: {ci_energy} Hartree")
