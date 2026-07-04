# -----------------------------------------------------------------------------
# ----------------------------- class MGStructure -----------------------------
# -----------------------------------------------------------------------------
# MGStructure (molecular geometry structure) is the general collection of
# molecules occupying different points in the space of molecular geometries.
# MGStructure has two main properties:
#   1. geometries: list of MGeometry instances
#   2. nodes: list of MPSNode instances
# MGStructure provides the following useful methods:
#   1. Geometrical methods: get and find to translate between canonical indices
#      and geometric parameters; neighbours to impose nearest-connections on
#      the structure, guiding transfer trajectories.
#      These methods should all be overloaded by subclasses.
#   2. Surface management: add_surface, find_base_sol, etc.
# An instance of MGStructure or its subclass is fully initialised in four
# steps:
#   1. __init__(mol_bp): Registers the structure, specifies molecule blueprint
#      and basic metadata.
#   2. init_geometry(geometry_data): Initialises all geometries. Here, the
#      initialisation is manual; subclasses should override this method to
#      exploit further assumptions about the structure geometry.
#   3. Any init_nodes method: Initialises all nodes. Here, two methods are
#      provided: automatic initialisation (which runs HF on each node) and
#      manual initialisation (which provides mean-field results, useful when
#      loading from disk). Subclasses may add more methods, but this is not
#      expected.
#   4. Any init_base_sol method: Initialises the base solution (FCI at base
#      node). Can be either done automatically or loaded manually.
import math
import numpy as np

from surf.MGeometry import MGeometry
from surf.MPSNode import MPSNode
from surf.MPSurface import MPSurface

# Parent class. Not specific. Can be used to create arbitrary, unstructured grids

class MGStructure():

    # Class properties
    cls = "MGStructure"
    desc = "Manual initialisation."

    # -------------------------------------------------------------------------
    # ------------------------ Initialisation methods -------------------------
    # -------------------------------------------------------------------------

    # ------------------------ 1. Basic initialisation ------------------------

    def __init__(self, mol_bp, log = None):
        # mol_bp is an instance of MBlueprint
        # log is an instance of Journal (or None for no printing)
        self.mol_bp = mol_bp
        self.log = log

        self.geometries = [] # list of instances of MGeometry
        self.nodes = [] # list of instances of MPSNode
        self.surfaces = {} # list of instances of MPSurface

        self.info = {
            "class" : type(self).cls,
            "desc" : type(self).desc,
            "mol_bp" : self.mol_bp.mol_meta
            }

    # ---------------------- 2. Geometry initialisation -----------------------

    def init_geometry(self, **kwargs):
        # Loads geometries and infers the number of nodes. First node is base.
        # kwargs:
        #   -geometry_tensor : an ndarray of atomic positions with
        #       -shape = (N_nodes, N_atoms - 1, 3)
        #       -dtype = float
        #   -connections : list of indices of connected nodes for each node

        geometry_tensor = kwargs["geometry_tensor"]
        if "connections" in kwargs:
            connections = kwargs["connections"]
        else:
            connections = None

        self.N_nodes = len(geometry_tensor)
        self.base_can_i = 0

        for i_node in range(self.N_nodes):
            self.geometries.append(MGeometry(geometry_tensor[i_node]))
            self.nodes.append(MPSNode(self.mol_bp, self.geometries[i_node]))

        # Initialise connections
        if connections is None:
            self.create_automatic_connections()
        else:
            self.connections = connections

        self.geometry_meta = {
            "N_nodes" : self.N_nodes
            }

    def dump_geometry(self):
        geometry_tensor = []
        for i_node in range(self.N_nodes):
            geometry_tensor.append(self.geometries[i_node].r.tolist())
        return({
            "geometry_tensor" : geometry_tensor,
            "connections" : self.connections
            })

    # ------------------------ 3. Node initialisation -------------------------

    def load_nodes(self, mean_field_info, MO_coefs, E_nuc, MO_H_one, MO_H_two, reference_state_energy):
        for i_node in range(self.N_nodes):
            self.nodes[i_node].load_HF(mean_field_info, MO_coefs[i_node], E_nuc[i_node], MO_H_one[i_node], MO_H_two[i_node], reference_state_energy[i_node])

        # Save global properties as metadata
        self.mean_field = mean_field_info

    def init_nodes_with_HF(self, HF_method = "default"):
        for i in range(self.N_nodes):
            self.nodes[i].run_HF(HF_method)
            if self.log is not None:
                self.log.update_semaphor_event(i + 1)
        self.mean_field = {
            "HF_method" : self.nodes[0].HF_method,
            "N_orb" : self.nodes[0].N_orb,
            "S_alpha" : self.nodes[0].S_alpha,
            "S_beta" : self.nodes[0].S_beta
            }

    # -------------------- 4. Base solution initialisation --------------------

    def load_base_sol(self, energies, coefs, spin_mult):
        # energies: H eigenvals for each root
        # coefs: occ basis rep for each root
        # spin_mult: spin multiplicity 2S + 1 for each root

        self.N_surf = len(energies)

        self.base_sol_energies = energies
        self.base_sol_coefs = coefs
        self.base_sol_spin_mult = spin_mult


    def find_base_sol(self, N_surf):

        self.N_surf = N_surf

        self.base_sol_energies, self.base_sol_coefs, self.base_sol_spin_mult = self.nodes[self.base_can_i].run_FCI(self.N_surf)

    # ------------------------- FCI surface analysis


    def find_FCI_surfaces(self):
        # The master method which:
        #   1. Finds N_surf lowest-energy eigenstates at each geometry node
        #   2. Applies labelling which characterises the continuous surfaces
        #      emergent from the FCI calculation:
        #        2a) We firstly characterise the eigenstates by their S2 e.v.
        #        2b) If the parameter space has two or fewer dimensions (e.g.
        #            diatomic molecule which preserves the spatial symmetry
        #            group), only conical degeneracies are allowed by the von
        #            Neumann-Wigner theorem, for which the energy ordering
        #            coincides with the continuity labelling, which is trivial.
        #            Otherwise, proper overlap transport analysis must be used.
        # This method returns (energies, coefs, meta)

        energies, coefs, spin_mult, duration = self.nodewise_FCI_calculation()

        self.log.write(f"Spin multiplicities: {spin_mult}")

        nodewise_surf_order = self.classify_FCI_surfaces(energies, coefs, spin_mult)

        #ordered_energies = np.zeros((self.N_surf, self.N_nodes))
        #ordered_coefs = np.zeros((self.N_surf, self.N_nodes, math.comb(self.mean_field["N_orb"], self.mean_field["S_alpha"]) * math.comb(self.mean_field["N_orb"], self.mean_field["S_beta"])))
        #ordered_spin_mult = np.zeros((self.N_surf, self.N_nodes))

        meta = []

        for i_s in range(self.N_surf):
            meta.append({
                "i_surf" : i_s,
                "spin_mult" : int(self.base_sol_spin_mult[i_s]),
                "method" : "FCI",
                "duration" : duration
                })

        return(
            energies[nodewise_surf_order.T, np.arange(self.N_nodes)],
            coefs[nodewise_surf_order.T, np.arange(self.N_nodes)],
            meta
            )

    def nodewise_FCI_calculation(self):

        self.log.write("Allocating memory for results...")

        FCI_energies = np.zeros((self.N_surf, self.N_nodes))
        FCI_coefs = np.zeros((self.N_surf, self.N_nodes, math.comb(self.mean_field["N_orb"], self.mean_field["S_alpha"]) * math.comb(self.mean_field["N_orb"], self.mean_field["S_beta"]))) # real coefs for FCI
        FCI_spin_mult = np.zeros((self.N_surf, self.N_nodes), dtype = int)

        self.log.enter("Performing FCI", semaphored = True, tau_space = np.linspace(0, self.N_nodes, 1000 + 1))

        for i in range(self.N_nodes):
            cur_FCI_energies, cur_FCI_coefs, cur_FCI_spin_mult = self.nodes[i].run_FCI(self.N_surf)
            for i_surf in range(self.N_surf):
                FCI_energies[i_surf, i] = cur_FCI_energies[i_surf]
                FCI_coefs[i_surf, i] = cur_FCI_coefs[i_surf]
                FCI_spin_mult[i_surf, i] = cur_FCI_spin_mult[i_surf]
            self.log.update_semaphor_event(i + 1)

        duration = self.log.exit("FCI calculation")

        return(FCI_energies, FCI_coefs, FCI_spin_mult, duration)

    def classify_FCI_surfaces(self, energies, coefs, spin_mult):

        # First, we look at the indices by spin subspace for the reference node
        base_index_in_mult_subspace = {}
        for i_s in range(self.N_surf):
            if self.base_sol_spin_mult[i_s] not in base_index_in_mult_subspace.keys():
                base_index_in_mult_subspace[self.base_sol_spin_mult[i_s]] = [i_s]
            else:
                base_index_in_mult_subspace[self.base_sol_spin_mult[i_s]].append(i_s)

        nodewise_surf_order = np.zeros((self.N_nodes, self.N_surf), dtype = int)

        for i_n in range(self.N_nodes):

            index_in_mult_subspace = {}

            for i_s in range(self.N_surf):
                if spin_mult[i_s, i_n] not in index_in_mult_subspace.keys():
                    nodewise_surf_order[i_n, i_s] = base_index_in_mult_subspace[spin_mult[i_s, i_n]][0]
                    index_in_mult_subspace[spin_mult[i_s, i_n]] = 1
                else:
                    nodewise_surf_order[i_n, i_s] = base_index_in_mult_subspace[spin_mult[i_s, i_n]][index_in_mult_subspace[spin_mult[i_s, i_n]]]
                    index_in_mult_subspace[spin_mult[i_s, i_n]] += 1
        return(nodewise_surf_order)


    # -------------------------------------------------------------------------
    # -------------------------- Geometrical methods --------------------------
    # -------------------------------------------------------------------------

    # -------------------------- Canonical indexing ---------------------------

    def get(self, i):
        # This method is used to create a natural indexing, here translated to canonical indexing. Subclasses will redefine this.
        if i >= 0 and i < len(self.nodes):
            return(i)
        return(None)

    def find(self, i):
        # This method translates canonical index to natural index
        return(i)

    # ----------------------------- Neighbourhood -----------------------------

    def neighbours(self, i):
        # returns canonical indices of all neighbours of node at i
        return(self.connections[i])

        # NOTE: when making subclasses, we should skip using a connections list
        # if possible, as it requires a lot of memory. Regular structure often
        # allows us to directly overload this method.

    def create_automatic_connections(self):
        # Starting from base_g, we connect each geometry to all of its closest yet untraversed geometries
        pass

    # ---------------------- Surface management methods -----------------------

    def add_surface(self, label, E, basis, coef, meta):
        self.surfaces[label] = MPSurface(label, E, basis, coef, meta)

        # Note: we do not add the label to any list, since this method is also
        # used to overwrite loaded surfaces. As such, we always read
        # surfaces.keys() when communicating with disk.

    # -------------------- Property encoding and decoding ---------------------

    def encode_MO_coefs(self):
        if self.mean_field["HF_method"] == "RHF":
            res = np.zeros((self.N_nodes, self.mean_field["N_orb"], self.mean_field["N_orb"]), dtype = self.nodes[0].MO_coefs.dtype)
            for i in range(self.N_nodes):
                res[i, :, :] = self.nodes[i].MO_coefs
        elif self.mean_field["HF_method"] == "UHF":
            res = np.zeros((self.N_nodes, 2, self.mean_field["N_orb"], self.mean_field["N_orb"]), dtype = self.nodes[0].MO_coefs["a"].dtype)
            for i in range(self.N_nodes):
                res[i, 0, :, :] = self.nodes[i].MO_coefs["a"]
                res[i, 1, :, :] = self.nodes[i].MO_coefs["b"]
        return(res)

    def decode_MO_coefs(self, HF_method, encoded_MO_coefs):
        if HF_method == "RHF":
            return(encoded_MO_coefs)
        elif HF_method == "UHF":
            res = []
            for i in range(self.N_nodes):
                res.append({
                    "a" : encoded_MO_coefs[i][0],
                    "b" : encoded_MO_coefs[i][1]
                    })
            return(res)

    def encode_H_one(self):
        if self.mean_field["HF_method"] == "RHF":
            res = np.zeros((self.N_nodes, self.mean_field["N_orb"], self.mean_field["N_orb"]), dtype = self.nodes[0].MO_H_one.dtype)
            for i in range(self.N_nodes):
                res[i, :, :] = self.nodes[i].MO_H_one
        elif self.mean_field["HF_method"] == "UHF":
            res = np.zeros((self.N_nodes, 2, self.mean_field["N_orb"], self.mean_field["N_orb"]), dtype = self.nodes[0].MO_H_one["a"].dtype)
            for i in range(self.N_nodes):
                res[i, 0, :, :] = self.nodes[i].MO_H_one["a"]
                res[i, 1, :, :] = self.nodes[i].MO_H_one["b"]
        return(res)

    def decode_H_one(self, HF_method, encoded_H_one):
        if HF_method == "RHF":
            return(encoded_H_one)
        elif HF_method == "UHF":
            res = []
            for i in range(self.N_nodes):
                res.append({
                    "a" : encoded_H_one[i][0],
                    "b" : encoded_H_one[i][1]
                    })
            return(res)

    def encode_H_two(self):
        if self.mean_field["HF_method"] == "RHF":
            res = np.zeros((self.N_nodes, self.mean_field["N_orb"], self.mean_field["N_orb"], self.mean_field["N_orb"], self.mean_field["N_orb"]), dtype = self.nodes[0].MO_H_two.dtype)
            for i in range(self.N_nodes):
                res[i, :, :, :, :] = self.nodes[i].MO_H_two
        elif self.mean_field["HF_method"] == "UHF":
            res = np.zeros((self.N_nodes, 3, self.mean_field["N_orb"], self.mean_field["N_orb"], self.mean_field["N_orb"], self.mean_field["N_orb"]), dtype = self.nodes[0].MO_H_two["a"].dtype)
            for i in range(self.N_nodes):
                res[i, 0, :, :, :, :] = self.nodes[i].MO_H_two["a"]
                res[i, 1, :, :, :, :] = self.nodes[i].MO_H_two["b"]
                res[i, 2, :, :, :, :] = self.nodes[i].MO_H_two["ab"]
        return(res)

    def decode_H_two(self, HF_method, encoded_H_two):
        if HF_method == "RHF":
            return(encoded_H_two)
        elif HF_method == "UHF":
            res = []
            for i in range(self.N_nodes):
                res.append({
                    "a" : encoded_H_two[i][0],
                    "b" : encoded_H_two[i][1],
                    "ab" : encoded_H_two[i][2]
                    })
            return(res)



    def get_E_nuc_surface(self):
        E_nuc_surface = np.zeros(self.N_nodes)
        for i in range(self.N_nodes):
            E_nuc_surface[i] = self.nodes[i].E_nuc
        return(E_nuc_surface)

    def get_reference_state_energy_surface(self):
        reference_state_energy_surface = np.zeros(self.N_nodes)
        for i in range(self.N_nodes):
            reference_state_energy_surface[i] = self.nodes[i].reference_state_energy
        return(reference_state_energy_surface)


