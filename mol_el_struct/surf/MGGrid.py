
import numpy as np

from surf.MGStructure import MGStructure
from surf.MGeometry import MGeometry
from surf.MPSNode import MPSNode

# MGGrid for a D-dimensional grid

class MGGrid(MGStructure):

    def __init__(self, mol_bp, base_g, dg, ranges = None, base_sol = 1):
        # mol_bp is an instance of MBlueprint
        # base_g is an instance of MGeometry
        # dg is a list of instances of MGeometry
        # ranges is a list of tuples (min coef, max coef) of len equal to dg
        self.mol_bp = mol_bp

        self.D = len(dg) # grid dimension
        self.dg = dg

        if ranges is None:
            self.ranges = [(-5, 5)] * self.D # default
        else:
            self.ranges = ranges

        self.spans = [] # for each i, this counts the width of the interval
        for i in range(self.D):
            min_c, max_c = self.ranges[i]
            self.spans.append(max_c - min_c + 1)

        # we find the canonical index of the base node
        self.base_can_i = self.get(np.zeros(self.D, dtype = int))

        # initialise the geometries

        self.geometries = [None] * np.prod(self.spans)
        self.nodes = [None] * np.prod(self.spans)

        for can_i in range(len(self.geometries)):
            cur_r = self.find(can_i)

            cur_g_r = np.array(base_g.r, copy = True)
            for i_r in range(self.D):
                cur_g_r += self.dg[i_r].r * cur_r[i_r]
            self.geometries[can_i] = MGeometry(cur_g_r)
            self.nodes[can_i] = MPSNode(self.mol_bp, self.geometries[can_i])

        self.N_nodes = len(self.nodes)

        if isinstance(base_sol, int):
            self.find_base_sol(base_sol)
        else:
            self.base_sol = base_sol

        dg_repr = []
        for i in range(self.D):
            dg_repr.append(self.dg[i].r.tolist())

        self.init_metadata("MGGrid", "A D-dim grid built from a reference geometry and D linearly-independent deformation vectors.", {
            "dg" : dg_repr,
            "ranges" : self.ranges
            })


    def get(self, r):
        # r is a list of coefs.
        i_r = [] # assigns an index within the interval of each component of r
        for i in range(self.D):
            min_c, max_c = self.ranges[i]
            if r[i] < min_c or r[i] > max_c:
                return(None)
            i_r.append(r[i] - min_c)
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
            min_c, max_c = self.ranges[i_rep]
            r[i_rep] = min_c + remainder
            i = i // self.spans[i_rep]
        return(r)

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

