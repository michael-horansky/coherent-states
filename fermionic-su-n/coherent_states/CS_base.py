# -----------------------------------------------------------------------------
# ------------------------ Coherent state - base class ------------------------
# -----------------------------------------------------------------------------
# This is the class which every fermionic CS class has to inherit from.
# This class is unspecified and cannot be used on its own.

class CS_Base():

    @staticmethod
    def null_state(M, S):
        # returns a parameter matrix which corresponds to the Hartree
        # no-excitation state
        return(None)

    def __init__(self, M, S, z):
        # z is the parameter object. In general, it is a multi-dimensional
        # complex ndarray

        self.M = M
        self.S = S

        self.z = z

    def overlap(self, other, c = [], a = []):
        # This method evaluates <self | creation\hc annihilation | other>
        # Note that creation_sequence indices are in inverted order!
        # In other words, both sequences are inputted as if they are
        # annihilation sequences acting on their respective kets

        # other is an instance of the same CS class
        # both operator sequences are 1D lists of mode indices

        return(None)

