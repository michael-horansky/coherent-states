


from surf.MGGrid import MGGrid

from surf.MGeometry import MGeometry
from surf.MBlueprint import bp_catalogue

mol_name = "BeH2"
N_surf = 2

mol_bp = bp_catalogue[mol_name]

base_g = MGeometry([[-2.5208947, 0.0, 0.0], [2.5208947, 0.0, 0.0]])

dg = [
    MGeometry([[0.1, 0.0, 0.0], [0.0, 0.0, 0.0]]), # shifts the left H
    MGeometry([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])  # shifts the right H
    ]
ranges = [(-2, 2), (-2, 2)]

grid = MGGrid(mol_bp, base_g, dg, ranges, N_surf)

print("Spans:", grid.spans)

print("------------------- Canonisation of geometry index testing")
for i in range(len(grid.nodes)):
    print(f"{i}: {grid.find(i)}")
