# -----------------------------------------------------------------------------
# ------------------------------- class MGGrid --------------------------------
# -----------------------------------------------------------------------------
# MGGrid (molecular geometry grid) is a subclass of MGStructure which assumes
# that the nodes are arranged in a D-dimensional crystallic structure. The
# geometries are initialised from the lattice vectors (corresponding to certain
# deformations of the base node geometry) and the number of lattice cells along
# every direction.

import numpy as np

from surf.MGStructure import MGStructure
from surf.MGeometry import MGeometry
from surf.MPSNode import MPSNode


class MGGrid(MGStructure):

    # Class properties
    cls = "MGGrid"
    desc = "A D-dim grid built from a reference geometry and D linearly-independent deformation vectors."

    # -------------------------------------------------------------------------
    # ------------------------ Initialisation methods -------------------------
    # -------------------------------------------------------------------------

    # ------------------------ 1. Basic initialisation ------------------------

    def __init__(self, mol_bp, log = None):
        super().__init__(mol_bp, log)

        self.info = {
            "class" : type(self).cls,
            "desc" : type(self).desc,
            "mol_bp" : self.mol_bp.mol_meta
            }

    # ---------------------- 2. Geometry initialisation -----------------------

    def init_geometry(self, **kwargs):
        # kwargs:
        #   -base_g : (N_a - 1, 3) ndarray of dtype float,
        #   -dg : (D, N_a - 1, 3) ndarray of dtype float,
        #   -ranges : (D, 2) list of integers

        base_g = kwargs["base_g"]
        dg = kwargs["dg"]
        if "ranges" in kwargs:
            ranges = kwargs["ranges"]
        else:
            ranges = None

        self.D = len(dg) # grid dimension

        self.dg = []
        for d in range(self.D):
            self.dg.append(MGeometry(dg[d]))

        # We initialise ranges and spans, which enables canonical indexing
        if ranges is None:
            self.ranges = [(-5, 5)] * self.D # default
        else:
            self.ranges = ranges

        self.spans = [] # for each i, this counts the width of the interval
        self.lows = [] # lower range, for ease of accessing
        for i in range(self.D):
            min_c, max_c = self.ranges[i]
            self.spans.append(max_c - min_c + 1)
            self.lows.append(min_c)

        self.weights = [1] * self.D # weights_i = spans_{i + 1} * ... * spans_{D - 1}
        for i in range(self.D - 2, -1, -1):
            self.weights[i] = self.weights[i + 1] * self.spans[i + 1]


        # We find the canonical index of the base node
        self.base_can_i = self.get(np.zeros(self.D, dtype = int))

        # Initialise the geometries
        self.N_nodes = np.prod(self.spans)
        for can_i in range(self.N_nodes):
            cur_r = self.find(can_i)

            cur_g_r = np.array(base_g, copy = True)
            for i_r in range(self.D):
                cur_g_r += self.dg[i_r].r * cur_r[i_r]
            self.geometries.append(MGeometry(cur_g_r))
            self.nodes.append(MPSNode(self.mol_bp, self.geometries[can_i]))


        self.geometry_meta = {
            "N_nodes" : self.N_nodes
            }

    def dump_geometry(self):
        dg_tensor = []
        for d in range(self.D):
            dg_tensor.append(self.dg[d].r.tolist())
        return({
            "base_g" : self.geometries[self.base_can_i].r.tolist(),
            "dg" : dg_tensor,
            "ranges" : self.ranges
            })


    # ------------------------ 3. Node initialisation -------------------------
    # Not overridden

    # -------------------- 4. Base solution initialisation --------------------
    # Not overridden

    # -------------------------------------------------------------------------
    # -------------------------- Geometrical methods --------------------------
    # -------------------------------------------------------------------------

    # -------------------------- Canonical indexing ---------------------------

    def get_i_r(self, r):
        # returns a vector of indices of the lattice cell along each dimension
        i_r = []
        for i in range(self.D):
            min_c, max_c = self.ranges[i]
            if r[i] < min_c or r[i] > max_c:
                return(None)
            i_r.append(r[i] - min_c)
        return(i_r)

    def get(self, r):
        # r is a list of coefs.
        i_r = self.get_i_r(r)
        if i_r is None:
            # not in the grid
            return(None)
        # Now we know that r points at a member of the grid
        # The natural -> canonical indexing map is
        #   canonical i = i_r[0] * spans[1] * ... * spans[D - 1] + i_r[1] * spans[2] * ... * spans[D - 1] + ... + i_r[D - 2] * spans[D - 1] + i_r[D - 1]
        can_i = 0
        for i in range(self.D):
            can_i *= self.spans[i]
            can_i += i_r[i]
        return(can_i)

    def find(self, i):
        r = np.zeros(self.D, dtype = int)
        for i_rep in range(self.D - 1, -1, -1):
            remainder = i % self.spans[i_rep]
            r[i_rep] = self.lows[i_rep] + remainder
            i = i // self.spans[i_rep]
        return(r)

    def get_subgrid_through_pivot(self, pivot_r, free_dimensions):
        # Returns a list of i_r vectors belonging to the subgrid

        # pivot_r is an r-vector of a point in the subgrod
        # free_dimensions is a list of dimension indices which are not kept
        # constant in the grid
        pivot_i_r = self.get_i_r(pivot_r)

        def traverse_first_free_dimension(i_r, remaining_free_dimensions):
            if len(remaining_free_dimensions) == 0:
                # No more free dimensions, the i_r vector is fully fixed
                return([i_r])
            res = []
            for i_d in range(self.spans[remaining_free_dimensions[0]]):
                cur_i_r = i_r.copy()
                cur_i_r[remaining_free_dimensions[0]] = i_d
                res += traverse_first_free_dimension(cur_i_r, remaining_free_dimensions[1:])
            return(res)
        return(traverse_first_free_dimension(pivot_i_r, free_dimensions))


    # ----------------------------- Neighbourhood -----------------------------

    def neighbours(self, i):
        # returns canonical indices of all neighbours of node at i
        home_r = self.find(i)

        neigh_i = []
        for i in range(self.D):
            cur_delta = np.zeros(self.D, dtype = int)
            cur_delta[i] = 1

            lower_i = self.get(home_r - cur_delta)
            higher_i = self.get(home_r + cur_delta)

            if lower_i is not None:
                neigh_i.append(lower_i)
            if higher_i is not None:
                neigh_i.append(higher_i)
        return(neigh_i)

