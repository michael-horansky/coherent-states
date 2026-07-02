# -----------------------------------------------------------------------------
# ------------------------------ class MGeometry ------------------------------
# -----------------------------------------------------------------------------
# MGeometry (molecular geometry) describes only the atomic geometry of an
# otherwise unspecified molecule. For a molecule with N atoms, MGeometry.r is
# an (N - 1, 3) array specifying the relative positions of atoms 2, 3... N
# relative to atom 1 (which is taken to be at the coordinate space origin).
# Instances of this class may be
#   -added/subtracted (which adds/subtracts their r arrays elementwise)
#   -multiplied by real numbers (which rescales the r array elementwise)

import numpy as np

class MGeometry():

    def __init__(self, r):
        # r is a list of positions
        # r[i] is the position of the i-th atom relative to the zeroth atom

        self.N_a = len(r) + 1
        self.r = np.array(r)


    # Binary deformations

    def __add__(self, other):
        # Addition adds the atom positions elementwise
        # It is useful to deform a ref=.geometry by a basis of deformations
        assert isinstance(other, MGeometry)
        return(MGeometry(self.r + other.r))

    def __sub__(self, other):
        assert isinstance(other, MGeometry)
        return(MGeometry(self.r - other.r))

    # Scalar deformations

    def __mul__(self, scale):
        # Rescales or positions by a scalar float
        assert isinstance(other, float) or isinstance(other, int) or isinstance(other, float) or isinstance(other, np.float32) or isinstance(other, np.float64) or isinstance(other, np.int32) or isinstance(other, np.int64)
        return(MGeometry(self.r * other))

    def __div__(self, scale):
        # Rescales or positions by a scalar float
        assert isinstance(other, float) or isinstance(other, int) or isinstance(other, float) or isinstance(other, np.float32) or isinstance(other, np.float64) or isinstance(other, np.int32) or isinstance(other, np.int64)
        return(MGeometry(self.r / other))


