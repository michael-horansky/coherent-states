import numpy as np

# -----------------------------------------------------------------------------
# ------------------------ Coherent state - base class ------------------------
# -----------------------------------------------------------------------------
# This is the class which every fermionic CS class has to inherit from.
# This class is unspecified and cannot be used on its own.

class CS_Base():

    sampling_methods = ["uniform", "highest_orbital"]

    @classmethod
    def null_state(cls, M, S):
        # returns a parameter matrix which corresponds to the Hartree
        # no-excitation state
        return(None)

    @classmethod
    def random_state(cls, M, S, sampling_method):
        return(None)


    def __init__(self, M, S, z, S_spin = None):
        # z is the parameter object. In general, it is a multi-dimensional
        # complex ndarray

        self.M = M
        self.S = S
        self.S_spin = S_spin # either None or a tuple (S_alpha, S_beta)

        self.z = z

        log_norm_sign, log_norm_mag = self.slog_overlap(self)
        self.log_norm_coef = (log_norm_sign, log_norm_mag)

        # Descriptor properties
        self.class_name = "base"

    def overlap(self, other, c = [], a = []):
        # This method evaluates <self | creation\hc annihilation | other>
        # Note that creation_sequence indices are in inverted order!
        # In other words, both sequences are inputted as if they are
        # annihilation sequences acting on their respective kets

        # other is an instance of the same CS class
        # both operator sequences are 1D lists of mode indices

        return(None)

    def overlap_update(self, other, c = [], a = [], master_matrix_det = None, master_matrix_inv = None, master_matrix_alt_inv = None):
        return(self.overlap(other, c, a))

    def get_update_information(self, other):
        # returns master_matrix_det, master_matrix_inv, master_matrix_alt_inv, diagnostic

        return(None, None, None, None)

    def slog_overlap(self, other, c = [], a = []):
        return(None, None)

    def norm_overlap(self, other, c = [], a = []):
        # normalised overlap which prevents ftp error
        slog_overlap_sign, slog_overlap_mag = self.slog_overlap(other, c, a)
        sign_a, mag_a = self.log_norm_coef
        sign_b, mag_b = other.log_norm_coef
        return((slog_overlap_sign * sign_a * sign_b) * np.exp(slog_overlap_mag - (mag_a + mag_b) / 2.0))

