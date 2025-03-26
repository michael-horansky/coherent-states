# -----------------------------------------------------------------------------
# ------------ UnNormalised de Aguiar Semi-Decoupled Basis method -------------
# -----------------------------------------------------------------------------

from unnormalized_aguiar_solver import bosonic_su_n

class UNDASDB(bosonic_su_n):

    def __init__(self, ID):

        super().__init__(ID)

        self.decoupling_tolerance = [] #[b_i] = user-set tolerance for partitioning the b_i-th basis


    # Firstly: We shall use dense output
