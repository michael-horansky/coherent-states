# -----------------------------------------------------------------------------
# ----------------------------- class SpinOperator ----------------------------
# -----------------------------------------------------------------------------
# This is a child class of Operator, which is restricted to a particular
# Hilbert space of the form
#   H = H[spin up] otimes H[spin down]
# where the number of modes in each spin-subspace is fixed as M_a and M_b.

# The canonical ordering of all modes (spin-orbitals) is:
#   1. All spin-up modes
#   2. All spin-down modes

from Operator import Operator

class SpinOperator(Operator):

    def __init__(self, init_object, M_a, M_b, sym = False):
        # init_object initialises phi according to Operator.__init__()
        # M_a, M_b are the mode numbers for the spin-up and spin-down subspaces
        # sym = True when the operator is spin-symmetric, False otherwise.

        # Initialise phi
        super().__init__(init_object)

        # Initialise additional Hilbert space properties
        self.M_a = M_a
        self.M_b = M_b

        self.sym = sym

        # When evaluating an overlap integral, we can decompose the operator
        # into partial elements within the two spin-subspaces.
        # We keep track of the overlaps which need to be pre-computed for each
        # subspace to minimise computation complexity.
        self.update_partial_overlap_list()

    def so_to_m(self, s, i):
        # Converts spin-orbital index to mode index
        if s == 'a':
            assert 0 <= i and i < self.M_a
            return(i)
        elif s == 'b':
            assert 0 <= i and i < self.M_b
            return(self.M_a + i)
        raise Exception(f"Invalid spin provided: '{s}'. Valid values are 'a' for spin-up and 'b' for spin-down.")

    def m_to_so(self, i):
        # Converts mode index to spin-orbital index
        assert 0 <= i and i < self.M_a + self.M_b
        if i < self.M_a:
            return('a', i)
        else:
            return('b', i - self.M_a)

    def decompose_key(self, key):
        # key is a tuple of frozensets
        # returns a dict in the form {'a'/'b' : partial key}
        a_c = set()
        a_a = set()
        b_c = set()
        b_a = set()

        full_c, full_a = key
        for c_i in full_c:
            if c_i < self.M_a:
                a_c.add(c_i)
            else:
                b_c.add(c_i)
        for a_i in full_a:
            if a_i < self.M_a:
                a_a.add(a_i)
            else:
                b_a.add(a_i)

        return({'a' : (frozenset(a_c), frozenset(a_a)), 'b' : (frozenset(b_c), frozenset(b_a))})



    def update_partial_overlap_list(self):
        # First, we reset the list
        self.partial_overlap_list = {'a' : set(), 'b' : set()} # {'a'/'b' : set of partial keys}

        # Then, we add all which needs to be pre-calculated during overlap calc
        for key in self.phi:
            decomposed_key = self.decompose_key(key)
            for spin_label in ['a', 'b']:
                if decomposed_key[spin_label] not in self.partial_overlap_list[spin_label]:
                    self.partial_overlap_list[spin_label].add(decomposed_key[spin_label])

    def simplify(self):
        super().simplify()
        self.update_partial_overlap_list()



    # --------------------- Overlap with coherent states ----------------------

    def mel(self, psi_1, psi_2):
        # here psi_1/2 are two 2-element lists or tuples of objects which
        # posses the norm_overlap methods (such as any CS objects).

        # norm_overlap has the form CS1.norm_overlap(CS2, c, a), where c, a are
        # lists of indices on which creation and annihilation operators act in
        # normal order for the ket and inverted order on the bra.

        # Firstly we pre-compute all the required partial overlaps, then put
        # them together.

        psi = [
            {'a' : psi_1[0], 'b' : psi_1[1]},
            {'a' : psi_2[0], 'b' : psi_2[1]}
            ]

        partial_overlaps = {} # ['a'/'b'][partial key] = overlap value
        for spin_label in ['a', 'b']:
            for req_partial_overlap in self.partial_overlap_list[spin_label]:
                partial_overlaps[spin_label][req_partial_overlap] = psi[0][spin_label].norm_overlap(psi[1][spin_label], sorted(req_partial_overlap[0]), sorted(req_partial_overlap[1]))

        res = 0.0
        for ops, coef in self.phi.items():
            decomposed_ops = self.decompose_key(ops)
            cur_overlap = 1.0
            for spin_label in ['a', 'b']:
                cur_overlap *= partial_overlaps[spin_label][decomposed_ops[spin_label]]
            res += cur_overlap * coef

        return(res)

