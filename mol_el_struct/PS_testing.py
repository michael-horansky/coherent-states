
import numpy as np
import matplotlib.pyplot as plt

from potential_surface_extrapolator import potential_surface_extrapolator


from surf.MGGrid import MGGrid

from surf.MGeometry import MGeometry
from surf.MBlueprint import bp_catalogue



"""
mol_name = "Li2"
N_surf = 2

base_g = np.array([[5.0, 0.0, 0.0]])

dg = np.array([[[0.1, 0.0, 0.0]]])
ranges = [(-4, 4)]
"""

mol_name = "H2O"
N_surf = 5

b = 1.8111135
ha = 1.8229964 / 2.0 # half angle
base_g = [
    [b * np.cos(ha), -b * np.sin(ha), 0.0],
    [b * np.cos(ha), b * np.sin(ha), 0.0]
    ]

dg = np.array([
    [[0.05 * b * np.cos(ha), 0.0, 0.0], [0.05 * b * np.cos(ha), 0.0, 0.0]], # Moving O closer to the centre of the hydrogen atoms
    [[0.0, -0.05 * b * np.sin(ha), 0.0], [0.0, 0.05 * b * np.sin(ha), 0.0]] # Stretching the distance between the hydrogen atoms
    ])

ranges = [(-10, 10), (-10, 10)]
#ranges = [(-1, 1), (-1, 1)]

PSSolver = potential_surface_extrapolator(f"{mol_name}_PS_solver")

PSSolver.load_data()

#PSSolver.init_structure_automatically("MGGrid", mol_name, N_surf = N_surf, base_g = base_g, dg = dg, ranges = ranges)
#PSSolver.run_FCI_on_structure()
#PSSolver.save_data()

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

PSSolver.plot_grid_potential_surfaces(ax = ax, i_d = [0, 1], i_surf = [2, 3])
plt.show()


"""
plt.title(f"Potential surfaces on {mol_name}")

plt.xlabel("r_x")
plt.ylabel("E [H]")


xspace = np.zeros(PSSolver.structure.N_nodes)

for i_node in range(PSSolver.structure.N_nodes):
    xspace[i_node] = PSSolver.structure.nodes[i_node].g.r[0][0]

plt.plot(xspace, PSSolver.structure.get_reference_state_energy_surface(), label = "$E_{\\rm HF}$")
for i in range(N_surf):
    plt.plot(xspace, PSSolver.structure.surfaces[f"FCI[{i}]"].E, label = f"$E_{{{i + 1}}}^{{\\rm (FCI)}}$")


plt.legend()
plt.show()
"""
