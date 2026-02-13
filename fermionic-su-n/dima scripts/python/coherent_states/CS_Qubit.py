# -----------------------------------------------------------------------------
# --------------- Coherent state - Qubit (spin-coherent) states ---------------
# -----------------------------------------------------------------------------
# This is the fermionic CS class enconding the qubit (spin-coherent) states.
# Its decomposition is given by the partitions of the (M)-parameter vector.

from coherent_states.CS_base import CS_Base
import numpy as np

# Function for permutation signature (parity)
def permutation_signature(arr):
    # arr is a list of distinct values
    def merge_sort(arr):
        if len(arr) <= 1:
            return arr, 0
        mid = len(arr) // 2
        left, inv_left = merge_sort(arr[:mid])
        right, inv_right = merge_sort(arr[mid:])
        merged, inv_split = merge_and_count(left, right)
        return merged, inv_left + inv_right + inv_split

    def merge_and_count(left, right):
        merged = []
        i = j = inv_count = 0
        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                merged.append(left[i])
                i += 1
            else:
                merged.append(right[j])
                inv_count += len(left) - i
                j += 1
        merged += left[i:]
        merged += right[j:]
        return merged, inv_count

    _, inversions = merge_sort(arr)
    return 1 if inversions % 2 == 0 else -1

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

def complex_logsumexp(sgn_a, log_a, sgn_b, log_b):
    # returns sign and logsumexp of the two provided numbers represented in log-polar space

    # if sgn_a/b = 0, then a/b = 0 and the result should be equal to the other value
    if sgn_a == 0:
        return(sgn_b, log_b)
    if sgn_b == 0:
        return(sgn_a, log_a)

    # regularising
    if log_a < log_b:
        log_a, log_b = log_b, log_a
        sgn_a, sgn_b = sgn_b, sgn_a

    exp_delta = np.exp(log_b - log_a) * (sgn_b / sgn_a)

    # if 1.0 + exp_delta == 0, then we return (1, -np.inf)
    if np.abs(1.0 + exp_delta) == 0.0:
        return(0.0, -np.inf)

    return (
        sgn_a * np.sign(1.0 + exp_delta),
        log_a + np.log(np.abs(1.0 + exp_delta))
    )

def slog_esp(roots, order):
    # calculates the signed natural log of the elementary symmetric polynomial over [roots] of the given order

    # Firstly, we omit all zero terms
    if np.any(roots == 0.0):
        nonzero_roots = []
        for i in range(len(roots)):
            if roots[i] != 0.0:
                nonzero_roots.append(roots[i])
        return(slog_esp(nonzero_roots, order))

    # Then, we find the slog values of roots
    log_roots = np.zeros(len(roots))#np.log(np.abs(roots))
    for i in range(len(roots)):
        if roots[i] == 0.0:
            log_roots[i] = -np.inf
        else:
            log_roots[i] = np.log(np.abs(roots[i]))
    sgn_roots = np.sign(roots)


    # Then, we take care of edge cases
    if order == 0:
        return(1.0, 0.0) # log(1)
    if order == len(roots):
        return(np.prod(sgn_roots), np.sum(log_roots))
    if order > len(roots):
        return(0.0, -np.inf)

    log_partial_esp = -np.inf * np.ones(order + 1)
    sgn_partial_esp = np.ones(order + 1, dtype=complex)

    log_partial_esp[0] = 0.0  # log(1)
    sgn_partial_esp[0] = 1.0

    for i in range(len(roots)):
        for j in range(min(i + 1, order), 0, -1):
            # term = x_i * E[j-1]
            term_log = log_roots[i] + log_partial_esp[j-1]
            term_sgn = sgn_roots[i] * sgn_partial_esp[j-1]

            # E[j] += term
            sgn_partial_esp[j], log_partial_esp[j] = complex_logsumexp(
                sgn_partial_esp[j], log_partial_esp[j],
                term_sgn, term_log
            )

    return(sgn_partial_esp[order], log_partial_esp[order])


class CS_Qubit(CS_Base):

    @classmethod
    def null_state(cls, M, S):
        # returns an instance of this class which corresponds to the Hartree no-excitation state
        return(cls(M, S, np.concatenate((np.ones(S, dtype = complex), np.zeros(M - S, dtype=complex)))))

    @classmethod
    def random_state(cls, M, S, sampling_method, assume_real = True):

        if assume_real:
            if sampling_method == "uniform":
                #random_z = np.random.normal(0.0, 1.0, M) + 1j * np.random.normal(0.0, 1.0, M)
                random_z = np.concatenate((
                        np.random.normal(1.0, 1.0, S),
                        np.random.normal(1.0, 1.0, M - S)
                    ))
                return(cls(M, S, random_z))
            if sampling_method == "highest_two_orbitals":
                random_z = np.concatenate((
                        np.random.normal(100.0, 100.0, S-2),
                        np.random.normal(1.0, 1.0, 2),
                        np.random.normal(0.0, 0.1, M - S)
                    ))
                return(cls(M, S, random_z))
            if sampling_method == "highest_orbital":
                #print(f"Stable MOs: {S-1}; unstable MO: 1; Free MOs: {M-S}")
                random_z = np.concatenate((
                        np.random.normal(100.0, 100.0, S-1),
                        np.random.normal(1.0, 1.0, 1),
                        np.random.normal(0.0, 0.1, M - S)
                    ))
                return(cls(M, S, random_z))
        else:
            if sampling_method == "uniform":
                #random_z = np.random.normal(0.0, 1.0, M) + 1j * np.random.normal(0.0, 1.0, M)
                random_z = np.concatenate((
                        np.random.normal(1.0, 1.0, S) * np.exp(1j * np.random.random(S) * 2.0 * np.pi),
                        np.random.normal(1.0, 1.0, M - S) * np.exp(1j * np.random.random(M - S) * 2.0 * np.pi)
                    ))
                return(cls(M, S, random_z))
            if sampling_method == "highest_two_orbitals":
                random_z = np.concatenate((
                        np.random.normal(100.0, 100.0, S-2) * np.exp(1j * np.random.random(S - 2) * 2.0 * np.pi),
                        np.random.normal(1.0, 1.0, 2) * np.exp(1j * np.random.random(2) * 2.0 * np.pi),
                        np.random.normal(0.0, 0.1, M - S) * np.exp(1j * np.random.random(M - S) * 2.0 * np.pi)
                    ))
                return(cls(M, S, random_z))
            if sampling_method == "highest_orbital":
                #print(f"Stable MOs: {S-1}; unstable MO: 1; Free MOs: {M-S}")
                random_z = np.concatenate((
                        np.random.normal(100.0, 100.0, S-1) * np.exp(1j * np.random.random(S - 1) * 2.0 * np.pi),
                        np.random.normal(1.0, 1.0, 1) * np.exp(1j * np.random.random(1) * 2.0 * np.pi),
                        np.random.normal(0.0, 0.1, M - S) * np.exp(1j * np.random.random(M - S) * 2.0 * np.pi)
                    ))
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


    def slog_overlap(self, other, c = [], a = []):
        # We assume that M, S match

        # firstly, we need to order a and c to be strictly ascending.
        if not (all(x<y for x, y in zip(c, c[1:])) and all(x<y for x, y in zip(a, a[1:]))):
            perm_sign = permutation_signature(c) * permutation_signature(a)
            norm_sgn, norm_log = self.slog_overlap(other, sorted(c), sorted(a))
            return(norm_sgn * perm_sign, norm_log)

        if len(c) != len(a):
            return(1, -np.inf)

        if len(c) == 0:
            return(slog_esp(np.conjugate(self.z) * other.z, self.S))

        # c and a contain distinct values, otherwise this is trivially zero
        if (len(c) > len(set(c)) or len(a) > len(set(a))):
            return(1, -np.inf)

        # Let's go for the general overlap
        z_a = []
        z_b = []

        #print("lol", c, a)

        prefactor = 1.0
        cur_sign = 1.0
        cur_pos_a = self.M - 1
        cur_pos_b = self.M - 1
        for i in range(len(c) - 1, -1, -1):
            prefactor *= np.conjugate(self.z[c[i]]) * other.z[a[i]]

            for j in range(cur_pos_a, c[i], -1):
                # note we omit c[i] itself, as it is set to zero
                z_a.insert(0, cur_sign * self.z[j])
            z_a.insert(0, 0.0)
            cur_pos_a = c[i] - 1
            for j in range(cur_pos_b, a[i], -1):
                # note we omit c[i] itself, as it is set to zero
                z_b.insert(0, cur_sign * other.z[j])
            z_b.insert(0, 0.0)
            cur_pos_b = a[i] - 1
            cur_sign *= -1
        for j in range(cur_pos_a, -1, -1):
            z_a.insert(0, cur_sign * self.z[j])
        for j in range(cur_pos_b, -1, -1):
            z_b.insert(0, cur_sign * other.z[j])


        """if len(a) > 0:
            print(f"i = {c[0]}, j = {a[0]}")
            print("z_a =", z_a)
            print("z_b =", z_b)
            print("z_a*.z_b =", np.conjugate(z_a) * z_b)"""


        if prefactor == 0.0:
            # overlap is zero
            return(0.0, -np.inf)

        reduced_overlap_sgn, reduced_overlap_log = slog_esp(np.conjugate(z_a) * z_b, self.S - len(c))
        prefactor_sgn = np.sign(prefactor)
        prefactor_log = np.log(np.abs(prefactor))

        return(prefactor_sgn * reduced_overlap_sgn, prefactor_log + reduced_overlap_log)




