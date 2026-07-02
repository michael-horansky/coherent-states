# -----------------------------------------------------------------------------
# ------------------------------- class MPSNode -------------------------------
# -----------------------------------------------------------------------------
# MPSurface (molecular potential surface) stores a calculated potential surface
# in a way which makes it easy to store or work with.
# It provides methods to dump it into csv and json serializable formats or load
# it back from such formats.

class MPSurface():

    def __init__(self, E, basis, coef):

        # E and coef are both ndarrays whose shape begins with (N_nodes, ...)
        # and are ordered by the node order
        # basis is a dict of the form
        # basis = {
        #     "type" : basis type,
        #     ...
        # }
        # Type is either a magic keyword ("occupancy") or a CS label. In the
        # latter case, basis["vectors"] is a list of parameters specifying the
        # basis and the order of the coef listing.

        self.E = E
        self.coef = coef
        self.basis = basis

