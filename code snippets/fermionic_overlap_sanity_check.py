
import numpy as np

def subset_indices(N, k):
    # returns a list of all subsequences of <N> of length k
    if k == 0:
        return([[]])
    if len(N) == 0:
        return([[]])
    res = []
    for k_1 in range(0, len(N) - k + 1):
        minor = subset_indices(N[k_1+1:], k-1)
        #if len(minor) > 0:
        for m in minor:
            res.append([N[k_1]] + m)
        #else:
        #    res.append([N[k_1]])
    return(res)

def random_complex_matrix(m, n, interval = None):
    if interval is None:
        scale = 1.0
        offset = 0.0
    else:
        int_min, int_max = interval
        scale = int_max - int_min
        offset = int_min
    return( (np.random.random((m, n)) + 1j * np.random.random((m, n))) * scale + offset)

def reduced_matrix(m, row_indices, column_indices):
    return(np.delete(np.delete(m, row_indices, axis = 0), column_indices, axis = 1))

def flip_lower_signs(m, row_sign_sequence = [], column_sign_sequence = []):
    res = m.copy()
    if len(row_sign_sequence) > 0:
        cur_parity = sign(len(row_sign_sequence))
        for i in range(max(row_sign_sequence)):
            if i in row_sign_sequence:
                cur_parity *= -1
            # apply
            res[i] *= cur_parity
    if len(column_sign_sequence) > 0:
        cur_parity = sign(len(column_sign_sequence))
        for i in range(max(column_sign_sequence)):
            if i in column_sign_sequence:
                cur_parity *= -1
            # apply
            res[:,i] *= cur_parity
    return(res)

def R(i, pi):
    if pi == 1:
        res = np.identity(S)
        res[i][i] = 0.0
        return(res)
    if pi == 0:
        res = np.identity(M-S)
        res[i-S][i-S] = 0.0
        return(res)

def Q(i, pi):
    if pi == 1:
        res = np.identity(S)
        for j in range(i):
            res[j][j] = -1.0
        return(res)
    if pi == 0:
        res = np.identity(M-S)
        for j in range(i-S, M-S):
            res[j][j] = -1.0
        return(res)


def sign(i):
    if i % 2 == 0:
        return(1.0)
    else:
        return(-1.0)

def cumsign(i):
    # = sign(i(i+1)/2)
    if i % 4 in [0, 3]:
        return(1.0)
    else:
        return(-1.0)

class occupancy_basis_state():

    def __init__(self, occupied_modes):
        self.occupied_modes = occupied_modes

    def __repr__(self):
        if len(self.occupied_modes) == 0:
            return(f"|vac.>")
        return(f"|{",".join(["1" if x in self.occupied_modes else "0" for x in range(1 + max(self.occupied_modes))])},0...>")
        #return(str(self.occupied_modes))

    def overlap(self, other):
        if self.occupied_modes == other.occupied_modes:
            return(1.0)
        return(0.0)

class occupancy_basis_superposition():

    def __init__(self, superposition):
        self.superposition = superposition # list of [coef, occupancy basis state]

    def __repr__(self):
        return(" + ".join([f"({x[0]:.2f}).{x[1]}" for x in self.superposition]))

    def apply_operators(self, c = [], a = []):
        # c is a sequence of mode indices for the creation operators
        # a is a sequence of mode indices for the annihilation operators
        # The total sequence is normal ordered. For other sequences, apply this method multiple times.

        # We apply the operators right to left, naturally.
        for a_i in range(len(a) - 1, -1, -1):
            new_superposition = []
            for i in range(len(self.superposition)):
                if a[a_i] in self.superposition[i][1].occupied_modes:
                    # Doesn't vanish
                    new_coef = sign(sum(occ_m < a[a_i] for occ_m in self.superposition[i][1].occupied_modes)) * self.superposition[i][0]
                    self.superposition[i][1].occupied_modes.remove(a[a_i])
                    new_superposition.append([new_coef, self.superposition[i][1]])
            self.superposition = new_superposition
        for c_i in range(len(c) - 1, -1, -1):
            new_superposition = []
            for i in range(len(self.superposition)):
                if c[c_i] not in self.superposition[i][1].occupied_modes:
                    # Doesn't vanish
                    new_coef = sign(sum(occ_m < c[c_i] for occ_m in self.superposition[i][1].occupied_modes)) * self.superposition[i][0]
                    self.superposition[i][1].occupied_modes.add(c[c_i])
                    new_superposition.append([new_coef, self.superposition[i][1]])
            self.superposition = new_superposition

    def overlap(self, other):
        res = 0.0
        for i in range(len(self.superposition)):
            for j in range(len(other.superposition)):
                res += np.conjugate(self.superposition[i][0]) * other.superposition[j][0] * self.superposition[i][1].overlap(other.superposition[j][1])
        return(res)

    def scale(self, k):
        for i in range(len(self.superposition)):
            self.superposition[i][0] *= k





class CS():

    def __init__(self, M, S, Z):
        self.M = M
        self.S = S
        self.Z = Z

        self.pi = [np.arange(self.S, self.M, 1), np.arange(0, self.S, 1)]

        self.decompose_into_occupancy_basis()

    def decompose_into_occupancy_basis(self):
        new_occupancy_basis_decomposition = [] # each element is [coef, occupancy_basis_state]
        for r in range(min(self.S, self.M - self.S) + 1):
            cur_sign = cumsign(r)
            cur_a_subsets = subset_indices(self.pi[0], r)
            cur_b_subsets = subset_indices(self.pi[1], r)
            for a_subset in cur_a_subsets:
                for b_subset in cur_b_subsets:
                    cur_occupancy_set = set(self.pi[1])-set(b_subset)
                    cur_occupancy_set.update(set(a_subset))
                    scaled_a_subset = []
                    for i in range(len(a_subset)):
                        scaled_a_subset.append(a_subset[i]-self.S)
                    cur_det = np.linalg.det(self.Z[np.ix_(scaled_a_subset, b_subset)])
                    if cur_det != 0:
                        new_occupancy_basis_decomposition.append([cur_sign * cur_det, occupancy_basis_state(cur_occupancy_set)])
        self.occupancy_basis_decomposition = occupancy_basis_superposition(new_occupancy_basis_decomposition)

    def slow_overlap(self, other):
        """res = 0.0
        for occ_self in self.occupancy_basis_decomposition:
            for occ_other in other.occupancy_basis_decomposition:
                res += np.conjugate(occ_self[0]) * occ_other[0] * occ_self[1].overlap(occ_other[1])
        return(res)"""
        return(self.occupancy_basis_decomposition.overlap(other.occupancy_basis_decomposition))

    def OLD_overlap_with_transition(self, other, i, j):
        # for now: both i and j are in pi_1
        self_signed = flip_lower_signs(self.Z, [], [i])
        other_signed = flip_lower_signs(other.Z, [], [j])
        #sus = reduced_matrix(np.matmul(np.conjugate(self_signed.T), other_signed), [j], [i])
        sus = np.matmul(reduced_matrix(other_signed, [], [i,j]), np.conjugate(reduced_matrix(self_signed, [], [i,j]).T))
        return(sign(i+j)*np.dot(other_signed[:,i], np.conjugate(self_signed[:,j]))*np.linalg.det(np.identity(self.M - self.S) + sus))
        # https://arxiv.org/pdf/1704.04405

    def overlap_with_transition(self, other, i, j):
        # for now: both i and j are in pi_1
        self_signed = flip_lower_signs(self.Z, [], i+j)
        other_signed = flip_lower_signs(other.Z, [], i+j)
        fast_M = np.zeros((self.M - self.S + len(i), self.M - self.S + len(i)), dtype=complex)
        fast_M[len(i):,:len(i)] = np.take(other_signed, i, axis = 1)
        fast_M[:len(i),len(i):] = np.conjugate(np.take(self_signed, j, axis = 1).T)
        fast_M[len(i):,len(i):] = np.identity(self.M - self.S) + np.matmul(reduced_matrix(other.Z, [], i+j), np.conjugate(reduced_matrix(self.Z, [], i+j).T))
        return(np.linalg.det(fast_M))

    def overlap(self, other, c = [], a = []):
        reduction_summants = []
        # As long as c and a are in pi_1, we treat them the same, as they just end up being "applied" to the bra maybe
        self_signed = flip_lower_signs(self.Z, [], c)
        other_signed = flip_lower_signs(other.Z, [], a)
        """for col in set(c).union(set(a)):
            reduction_summants.append(np.outer(other_signed[:,col], np.conjugate(self_signed[:,col])))"""
        #reduction_summants.append(np.outer(other.Z[:,0], np.conjugate(self.Z[:,0])))
        #reduction_summants.append(np.outer(other.Z[:,1], np.conjugate(self.Z[:,1])))
        #for col in a:
        #    reduction_summants.append(np.outer(other.Z[:,col], np.conjugate(self.Z[:,col])))
        sus = np.matmul(other_signed, np.conjugate(self_signed.T))
        #return(np.linalg.det(np.identity(self.M - self.S) + sus - np.sum(reduction_summants, axis = 0)))
        return(np.linalg.det(np.identity(self.M - self.S) + np.matmul(reduced_matrix(other_signed, [], a), np.conjugate(reduced_matrix(self_signed, [], c).T))))



"""
lol = occupancy_basis_superposition([[2, occupancy_basis_state({0, 1})], [-5, occupancy_basis_state({0})], [10, occupancy_basis_state({1})], [4, occupancy_basis_state(set())]])
print(lol)
print("Square norm =", lol.overlap(lol))
lol.apply_operators([2], [0])
print(lol)"""



M = 8
S = 5
N = 5

global_c_sequence = [1, 0]
global_a_sequence = [2, 3]

print(f"Calculating < Z_a | creation({global_c_sequence}) annihilation({global_a_sequence}) | Z_b >")
is_all_equal = True
for i in range(N):
    print("---------- test no.", i)
    A = CS(M, S, random_complex_matrix(M-S, S, (-1.0, 1.0)))
    B = CS(M, S, random_complex_matrix(M-S, S, (-1.0, 1.0)))

    #print("  Origin. A:", A.occupancy_basis_decomposition)
    #print("  Origin. B:", B.occupancy_basis_decomposition)

    A.occupancy_basis_decomposition.apply_operators([], global_c_sequence)
    B.occupancy_basis_decomposition.apply_operators([], global_a_sequence)
    #print("  Dir. r. A:", A.occupancy_basis_decomposition)
    #print("  Dir. r. B:", B.occupancy_basis_decomposition)

    A_var = CS(M, S, np.matmul(A.Z, np.matmul(Q(global_c_sequence[0], 1), R(global_c_sequence[0], 1))))
    B_var = CS(M, S, np.matmul(B.Z, np.matmul(Q(global_a_sequence[0], 1), R(global_a_sequence[0], 1))))
    A_var.occupancy_basis_decomposition.scale(sign(global_c_sequence[0]))
    B_var.occupancy_basis_decomposition.scale(sign(global_a_sequence[0]))
    #print("  Equiv. A:", A_var.occupancy_basis_decomposition)
    #print("  Equiv. B:", B_var.occupancy_basis_decomposition)


    slow = A.occupancy_basis_decomposition.overlap(B.occupancy_basis_decomposition)
    #quick = A.overlap(B, global_c_sequence, global_a_sequence)
    quick = A.overlap_with_transition(B, global_c_sequence, global_a_sequence)

    print(f"  Slow, trustworthy overlap: {slow:.4f}")
    print(f"  Fast overlap:              {quick:.4f}")

    if(np.round(slow, 5) != np.round(quick, 5)):
        is_all_equal = False
if is_all_equal:
    print("Both methods are equivalent")

# TODO: the issue is that just constructing ZQR states doesn't correctly reduce occupancies, and the way we take overlap of two differently reduced kets was treated badly on paper. We need to fix this.

"""
CS_list = []
for i in range(N):
    CS_list.append(CS(M, S, random_complex_matrix(M-S, S, (-1.0, 1.0))))
    CS_list[i].decompose_into_occupancy_basis()
    #print(CS_list[i].occupancy_basis_decomposition)

pairings = subset_indices(np.arange(0, N, 1), 2)

is_all_equal = True
for pair in pairings:
    print(f"Pair {pair[0]+1} : {pair[1] + 1}")
    slow = CS_list[pair[0]].slow_overlap(CS_list[pair[1]])
    quick = CS_list[pair[0]].quick_overlap(CS_list[pair[1]])
    print(f"  Slow overlap: {slow}")
    print(f"  Quick overlap: {quick}")
    if(np.round(slow, 5) != np.round(quick, 5)):
        is_all_equal = False
if is_all_equal:
    print("Both methods are equivalent")"""
