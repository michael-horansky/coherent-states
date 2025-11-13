# -----------------------------------------------------------------------------
# --------------- Coherent state - Qubit (spin-coherent) states ---------------
# -----------------------------------------------------------------------------
# This is the fermionic CS class enconding the qubit (spin-coherent) states.
# Its decomposition is given by the partitions of the (M)-parameter vector.

from coherent_states.CS_base import CS_Base
import numpy as np

def esp(roots, order):
    # roots is a list
    # order is an integer <= len(roots)

    if order == 0:
        return(1.0)
    if order == len(roots):
        return(np.prod(roots))

    partial_esp = np.zeros(order + 1, dtype = complex)
    partial_esp[0] = 1.0
    for i in range(len(roots)):
        for j in range(min(i + 1, order), 0, -1):
            partial_esp[j] += roots[i] * partial_esp[j - 1]
    return(partial_esp[order])


class CS_Qubit(CS_Base):

    @classmethod
    def null_state(cls, M, S):
        # returns an instance of this class which corresponds to the Hartree no-excitation state
        return(cls(M, S, np.concatenate((np.ones(S, dtype = complex), np.zeros(M - S, dtype=complex)))))

    @classmethod
    def random_state(cls, M, S, sampling_method):
        if sampling_method == "uniform":
            #random_z = np.random.normal(0.0, 1.0, M) + 1j * np.random.normal(0.0, 1.0, M)
            random_z = np.concatenate((np.random.normal(1.0, 1.0, S), np.random.normal(1.0, 1.0, M - S)))
            return(cls(M, S, random_z))
        if sampling_method == "highest_orbital":
            random_z = np.concatenate((np.random.normal(100.0, 100.0, S-2), np.random.normal(1.0, 1.0, 2), np.random.normal(0.0, 0.1, M - S)))
            return(cls(M, S, random_z))
        return(None)

    def __init__(self, M, S, z):
        super().__init__(M, S, z)

    def overlap(self, other, c = [], a = []):
        # We assume that M, S match

        if len(c) != len(a):
            return(0.0)

        if len(c) == 0:
            return(esp(np.conjugate(self.z) * other.z, self.S))

        # c and a contain distinct values, otherwise this is trivially zero
        if (len(c) > len(set(c)) or len(a) > len(set(a))):
            return(0.0)

        # Let's go for the general overlap
        z_a = []
        z_b = []

        prefactor = 1.0
        cur_sign = 1.0
        cur_pos_a = self.M - 1
        cur_pos_b = self.M - 1
        for i in range(len(c) - 1, -1, -1):
            prefactor *= np.conjugate(self.z[c[i]]) * other.z[a[i]]

            for j in range(cur_pos_a, c[i], -1):
                # note we omit c[i] itself, as it is set to zero
                z_a.insert(0, cur_sign * self.z[j])
            cur_pos_a = c[i] - 1
            for j in range(cur_pos_b, a[i], -1):
                # note we omit c[i] itself, as it is set to zero
                z_b.insert(0, cur_sign * other.z[j])
            cur_pos_b = a[i] - 1
            cur_sign *= -1
        for j in range(cur_pos_a, -1, -1):
            z_a.insert(0, cur_sign * self.z[j])
        for j in range(cur_pos_b, -1, -1):
            z_b.insert(0, cur_sign * other.z[j])

        return(prefactor * esp(np.conjugate(z_a) * z_b, self.S - len(c)))

    def overlap_update(self, other, c = [], a = [], master_matrix_det = None, master_matrix_inv = None, master_matrix_alt_inv = None):
        return(self.overlap(other, c, a))

    def get_update_information(self, other):
        # returns master_matrix_det, master_matrix_inv, master_matrix_alt_inv

        return(None, None, None)




