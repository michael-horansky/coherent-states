# -----------------------------------------------------------------------------
# ------------------ Coherent state - SU(M) Thouless states -------------------
# -----------------------------------------------------------------------------
# This is the fermionic CS class enconding full SU(M) Thouless coherent states.
# Its decomposition is based on Slater subdeterminants of the parameter matrix.

from coherent_states.CS_base import CS_Base
import numpy as np

# Auxiliary functions here

def reduced_matrix(m, row_indices, column_indices):
    return(np.delete(np.delete(m, row_indices, axis = 0), column_indices, axis = 1))

def eta(i, l, offset = 0):
    # number of elements in l smaller than i
    if isinstance(i, int) or isinstance(i, np.int64):
        res = 0
        for obj in l:
            if obj < i + offset:
                res += 1
        return(res)
    else:
        res = 0
        for obj in i:
            res += eta(obj, l, offset)
        return(res)

def sign(i):
    if i % 2 == 0:
        return(1.0)
    else:
        return(-1.0)

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



class CS_Thouless(CS_Base):


    @staticmethod
    def null_state(M, S):
        # returns an instance of this class which corresponds to the Hartree no-excitation state
        return(np.zeros((M - S, S), dtype=complex))

    def __init__(self, M, S, z):
        super().__init__(M, S, z)

    def overlap(self, other, c = [], a = []):
        # We assume that M, S match
        if len(c) != len(a):
            return(0.0)

        if len(c) == 0:
            # This is just <self | other>
            if self.M - self.S > self.S:
                # We prefer I_S + Z_a\hc.Z_b
                return(np.linalg.det(np.identity(self.S) + np.matmul(np.conjugate(self.z.T), other.z)))
            else:
                # We prefer I_(M-S) + Z_b.Z_a\hc
                return(np.linalg.det(np.identity(self.M - self.S) + np.matmul(other.z, np.conjugate(self.z.T))))

        # General overlap, oh boy

        # c and a contain distinct values, otherwise this is trivially zero
        if (len(c) > len(set(c)) or len(a) > len(set(a))):
            return(0.0)
        varsigma_a = []
        tau_a = []
        varsigma_b = []
        tau_b = []
        tau_cup = []
        sigma_intersection = []
        for i in range(len(c)):
            # using len(c) = len(a)
            if c[i] < self.S:
                # pi_1
                if c[i] not in a:
                    varsigma_a.append(c[i])
                else:
                    sigma_intersection.append(c[i])
            else:
                # pi_0
                tau_a.append(c[i] - self.S)
                if c[i] - self.S not in tau_cup:
                    tau_cup.append(c[i] - self.S)
            if a[i] < self.S:
                # pi_1
                if a[i] not in c:
                    varsigma_b.append(a[i])
            else:
                # pi_0
                tau_b.append(a[i] - self.S)
                if a[i] - self.S not in tau_cup:
                    tau_cup.append(a[i] - self.S)
        sigma_a = varsigma_a + sigma_intersection
        sigma_b = varsigma_b + sigma_intersection
        sigma_cup = varsigma_a + varsigma_b + sigma_intersection
        const_sign = sign(self.S * (len(tau_a) + len(tau_b)) + (len(varsigma_a) - 1) * (len(varsigma_b) - 1) + 1 + sum(varsigma_a) + len(varsigma_a) + sum(varsigma_b) + len(varsigma_b) + eta(sigma_intersection, varsigma_a + varsigma_b))
        # Now for the normal ordering sign
        const_sign *= permutation_signature(c) * permutation_signature(a)
        if len(tau_a) <= len(tau_b):
            fast_M = np.zeros((len(tau_b) + self.S - len(sigma_a), len(tau_b) + self.S - len(sigma_a)), dtype=complex)
            fast_M[:len(tau_b), len(tau_a):len(tau_a)+len(varsigma_a)] = np.take(np.take(other.z, tau_b, axis = 0), varsigma_a, axis = 1)
            fast_M[:len(tau_b), len(tau_a)+len(varsigma_a):] = np.take(reduced_matrix(other.z, [], sigma_cup), tau_b, axis = 0)

            fast_M[len(tau_b):len(tau_b) + len(varsigma_b),:len(tau_a)] = np.conjugate(np.take(np.take(self.z, tau_a, axis = 0), varsigma_b, axis = 1).T)
            fast_M[len(tau_b) + len(varsigma_b):, :len(tau_a)] = np.conjugate(np.take(reduced_matrix(self.z, [], sigma_cup), tau_a, axis = 0).T)

            fast_M[len(tau_b):len(tau_b) + len(varsigma_b), len(tau_a):len(tau_a)+len(varsigma_a)] = np.matmul(np.conjugate(np.take(reduced_matrix(self.z, tau_cup, []), varsigma_b, axis = 1).T), np.take(reduced_matrix(other.z, tau_cup, []), varsigma_a, axis = 1))
            fast_M[len(tau_b):len(tau_b) + len(varsigma_b), len(tau_a)+len(varsigma_a):] = np.matmul(np.conjugate(np.take(reduced_matrix(self.z, tau_cup, []), varsigma_b, axis = 1).T), reduced_matrix(other.z, tau_cup, sigma_cup))
            fast_M[len(tau_b) + len(varsigma_b):, len(tau_a):len(tau_a)+len(varsigma_a)] = np.matmul(np.conjugate(reduced_matrix(self.z, tau_cup, sigma_cup).T), np.take(reduced_matrix(other.z, tau_cup, []), varsigma_a, axis = 1))
            fast_M[len(tau_b) + len(varsigma_b):, len(tau_a)+len(varsigma_a):] = np.identity(self.S - len(sigma_cup)) + np.matmul(np.conjugate(reduced_matrix(self.z, tau_cup, sigma_cup).T), reduced_matrix(other.z, tau_cup, sigma_cup))
            cb_sign = sign(len(tau_b) * (1 + len(tau_b) - len(tau_a)))
            return(const_sign * cb_sign * np.linalg.det(fast_M))
        else:
            fast_M = np.zeros((self.M - self.S + len(tau_a) + len(varsigma_a) - len(tau_cup), self.M - self.S + len(tau_a) + len(varsigma_a) - len(tau_cup)), dtype=complex)
            fast_M[:len(varsigma_b), len(varsigma_a):len(varsigma_a)+len(tau_a)] = np.take(np.take(np.conjugate(self.z).T, varsigma_b, axis=0), tau_a, axis = 1)
            fast_M[:len(varsigma_b), len(varsigma_a)+len(tau_a):] = np.take(reduced_matrix(np.conjugate(self.z).T, [], tau_cup), varsigma_b, axis=0)
            fast_M[len(varsigma_b):len(varsigma_b)+len(tau_b), :len(varsigma_a)] = np.take(np.take(other.z, tau_b, axis=0), varsigma_a, axis = 1)
            fast_M[len(varsigma_b)+len(tau_b):, :len(varsigma_a)] = np.take(reduced_matrix(other.z, tau_cup, []), varsigma_a, axis=1)

            fast_M[len(varsigma_b):len(varsigma_b)+len(tau_b), len(varsigma_a):len(varsigma_a)+len(tau_a)] = np.matmul(np.take(reduced_matrix(other.z, [], sigma_cup), tau_b, axis = 0), np.take(reduced_matrix(np.conjugate(self.z).T, sigma_cup, []), tau_a, axis=1))
            fast_M[len(varsigma_b):len(varsigma_b)+len(tau_b), len(varsigma_a)+len(tau_a):] = np.matmul(np.take(reduced_matrix(other.z, [], sigma_cup), tau_b, axis = 0), reduced_matrix(np.conjugate(self.z).T, sigma_cup, tau_cup))
            fast_M[len(varsigma_b)+len(tau_b):, len(varsigma_a):len(varsigma_a)+len(tau_a)] = np.matmul(reduced_matrix(other.z, tau_cup, sigma_cup), np.take(reduced_matrix(np.conjugate(self.z).T, sigma_cup, []), tau_a, axis=1))
            fast_M[len(varsigma_b)+len(tau_b):, len(varsigma_a)+len(tau_a):] = np.identity(self.M - self.S - len(tau_cup)) + np.matmul(reduced_matrix(other.z, tau_cup, sigma_cup), reduced_matrix(np.conjugate(self.z).T, sigma_cup, tau_cup))
            cb_sign = sign(len(varsigma_b) * (1 + len(varsigma_b) - len(varsigma_a)))
            return(const_sign * cb_sign * np.linalg.det(fast_M))




