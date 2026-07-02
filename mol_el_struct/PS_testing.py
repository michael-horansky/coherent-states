
import numpy as np
import matplotlib.pyplot as plt

from potential_surface_extrapolator import potential_surface_extrapolator


from surf.MGGrid import MGGrid

from surf.MGeometry import MGeometry
from surf.MBlueprint import bp_catalogue

mol_name = "Li2"
N_surf = 2

mol_bp = bp_catalogue[mol_name]

base_g = MGeometry([[5.0, 0.0, 0.0]])

dg = [MGeometry([[0.1, 0.0, 0.0]])]
ranges = [(-3, 3)]

grid = MGGrid(mol_bp, base_g, dg, ranges, N_surf)



PSSolver = potential_surface_extrapolator(f"{mol_name}_PS_solver")

PSSolver.initialise_structure(grid)
PSSolver.run_FCI_on_structure()

plt.title(f"Potential surfaces on {mol_bp.name}")

plt.xlabel("r_x")
plt.ylabel("E [H]")


xspace = np.zeros(grid.N_nodes)

for i_node in range(grid.N_nodes):
    xspace[i_node] = grid.nodes[i_node].g.r[0][0]

plt.plot(xspace, grid.get_reference_state_energy_surface(), label = "$E_{\\rm HF}$")
for i in range(N_surf):
    plt.plot(xspace, grid.surfaces[f"FCI[{i}]"].E, label = f"$E_{{{i + 1}}}^{{\\rm (FCI)}}$")


plt.legend()
plt.show()

