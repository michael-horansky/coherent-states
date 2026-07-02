
import numpy as np

from surf.MPSNode import MPSNode
from surf.MPSurface import MPSurface

# Parent class. Not specific. Can be used to create arbitrary, unstructured grids

class MGStructure():

    def __init__(self, mol_bp, base_g, geometries, connections = None, base_sol = 1):
        # mol_bp is an instance of MBlueprint

        # base_g is an instance of MGeometry

        # base_sol is the array of FCI solutions at the base node. It is a list
        # of length N_surf, such that each element is a dict of the form
        # base_sol[i_surf] = {
        #         "E" : energy at the base node,
        #         "coef" : an ndarray in the canonical ordering
        #     }
        # If int, base_sol is instead interpreted as N_surf.

        # geometries is a list of instances of MGeometry

        # connections is a list of the form [canonical index of g] = [list of canonical indices of connected gs]
        self.mol_bp = mol_bp
        self.base_can_i = 0

        self.geometries = geometries
        self.nodes = []
        for i in range(len(self.geometries)):
            self.nodes.append(MPSNode(self.mol_bp, self.geometries[i]))

        self.N_nodes = len(self.nodes)

        if connections is None:
            self.create_automatic_connections()
        else:
            self.connections = connections

        if isinstance(base_sol, int):
            self.find_base_sol(base_sol)
        else:
            self.base_sol = base_sol

        self.init_metadata("MGStructure", "Manual initialisation.")

    def init_metadata(self, cls, desc, params = None):
        # Metadata
        self.geometry_meta = {
            "class" : cls,
            "desc" : desc,
            "mol_bp" : self.mol_bp.mol_meta
            }
        if params is not None:
            self.geometry_meta["params"] = params

        self.base_node_meta = {
            "N_surf" : len(self.base_sol),
            "base_can_i" : self.base_can_i
            }

        self.nodes_meta = {
            "HF_method" : None,
            "FCI_known" : False,
            "surface_labels" : []
            }
        self.surfaces = {} # list of instances of MPSurface
        self.surfaces_meta = {} # surface_label : meta dict



    def get(self, i):
        # This method is used to create a natural indexing, here translated to canonical indexing. Subclasses will redefine this.
        if i >= 0 and i < len(self.nodes):
            return(i)
        return(None)

    def find(self, i):
        # This method translates canonical index to natural index
        return(i)

    def neighbours(self, i):
        # returns canonical indices of all neighbours of node at i
        return(self.connections[i])

        # NOTE: when making subclasses, we should skip using a connections list
        # if possible, as it requires a lot of memory. Regular structure often
        # allows us to directly overload this method.

    def create_automatic_connections(self):
        # Starting from base_g, we connect each geometry to all of its closest yet untraversed geometries
        pass

    def add_surface(self, label, E, basis, coef, i_surf, method):
        # label is the surface label
        # i_surf is the index of the surface this surface is meant to calculate
        # method is any json-serializable object

        self.surfaces[label] = MPSurface(E, basis, coef)

        self.nodes_meta["surface_labels"].append(label)
        self.surfaces_meta[label] = {
            "i_surf" : i_surf,
            "method" : method
            }

    def find_base_sol(self, N_surf):

        node = self.nodes[self.base_can_i]

        base_FCI_energies, base_FCI_coefs = self.nodes[self.base_can_i].run_FCI(N_surf)

        self.base_sol = []
        for i in range(N_surf):
            self.base_sol.append({
                "E" : base_FCI_energies[i],
                "coef" : base_FCI_coefs[i]
                })

    def get_reference_state_energy_surface(self):
        reference_state_energy_surface = np.zeros(self.N_nodes)
        for i in range(self.N_nodes):
            reference_state_energy_surface[i] = self.nodes[i].reference_state_energy
        return(reference_state_energy_surface)
