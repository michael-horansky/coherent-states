
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

def tail_sublist_sign(l, sub):

    res = 0
    mutable_l = l.copy()
    for i in range(len(sub)):
        j = mutable_l.index(sub[i]) # this has to be brought to index 0
        res += j
        del mutable_l[j]
    return(sign(res), mutable_l)


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
        #self_signed = flip_lower_signs(self.Z, [], i+j)
        #other_signed = flip_lower_signs(other.Z, [], i+j)
        fast_M = np.zeros((self.M - self.S + len(i), self.M - self.S + len(i)), dtype=complex)
        fast_M[len(i):,:len(i)] = np.take(other.Z, i, axis = 1)
        fast_M[:len(i),len(i):] = np.conjugate(np.take(self.Z, j, axis = 1).T)
        fast_M[len(i):,len(i):] = np.identity(self.M - self.S) + np.matmul(reduced_matrix(other.Z, [], i+j), np.conjugate(reduced_matrix(self.Z, [], i+j).T))
        return(sign(sum(i + j)) * np.linalg.det(fast_M))

    def OLD_overlap(self, other, c = [], a = []):
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
        return(np.linalg.det(np.identity(self.M - self.S) + np.matmul(reduced_matrix(other.Z, [], c), np.conjugate(reduced_matrix(self.Z, [], a).T))))


    def overlap(self, other, c = [], a = []):
        # for now: both i and j are in pi_1
        # c and a are both in the order in which they act on the ket, right to left
        tau = list(set(c).intersection(set(a)))
        rho_sign, rho = tail_sublist_sign(c, tau)
        sigma_sign, sigma = tail_sublist_sign(a, tau)

        tau_sorting_prefactor = rho_sign * sigma_sign

        fast_M = np.zeros((self.M - self.S + len(rho), self.M - self.S + len(rho)), dtype=complex)
        fast_M[len(rho):,:len(rho)] = np.take(other.Z, rho, axis = 1)
        fast_M[:len(rho),len(rho):] = np.conjugate(np.take(self.Z, sigma, axis = 1).T)
        fast_M[len(rho):,len(rho):] = np.identity(self.M - self.S) + np.matmul(reduced_matrix(other.Z, [], rho+sigma+tau), np.conjugate(reduced_matrix(self.Z, [], rho+sigma+tau).T))
        return(sign(sum(rho + sigma)) * tau_sorting_prefactor * np.linalg.det(fast_M))

    # pi_0-reductions
    def disjoint_pi_zero_overlap(self, other, c = [], a = []):
        c = np.array(c, dtype=int)-self.S
        a = np.array(a, dtype=int)-self.S
        fast_M = np.zeros((self.S + len(c), self.S + len(c)), dtype=complex)
        fast_M[len(c):,:len(c)] = np.conjugate(np.take(self.Z, c, axis = 0).T)
        fast_M[:len(c),len(c):] = np.take(other.Z, a, axis = 0)
        fast_M[len(c):,len(c):] = np.identity(self.S) + np.matmul(np.conjugate(reduced_matrix(self.Z, np.concatenate((c, a)), []).T), reduced_matrix(other.Z, np.concatenate((c, a)), []))
        return(sign(len(c)) * np.linalg.det(fast_M))
    def repeated_pi_zero_overlap(self, other, c = [], a = []):
        c = np.array(c, dtype=int)-self.S
        a = np.array(a, dtype=int)-self.S
        fast_M = np.zeros((self.S + len(c), self.S + len(c)), dtype=complex)
        fast_M[len(c):,:len(c)] = np.conjugate(np.take(self.Z, c, axis = 0).T)
        fast_M[:len(c),len(c):] = np.take(other.Z, a, axis = 0)
        fast_M[len(c):,len(c):] = np.identity(self.S) + np.matmul(np.conjugate(reduced_matrix(self.Z, c, []).T), reduced_matrix(other.Z, c, []))
        return(sign(len(c)) * np.linalg.det(fast_M))
    def pi_zero_overlap(self, other, c = [], a = []):
        tau = list(set(c).intersection(set(a)))
        rho_sign, rho = tail_sublist_sign(c, tau)
        sigma_sign, sigma = tail_sublist_sign(a, tau)

        tau_sorting_prefactor = rho_sign * sigma_sign
        print(f"Sublist sign of {tau} in {c} = {tail_sublist_sign(c, tau)}")
        print(f"Sublist sign of {tau} in {a} = {tail_sublist_sign(a, tau)}")

        tau = np.array(tau, dtype=int)-self.S
        rho = np.array(rho, dtype=int)-self.S
        sigma = np.array(sigma, dtype=int)-self.S

        print(f"tau = {tau}, rho = {rho}, sigma = {sigma}")

        X = len(tau) + len(rho)
        fast_M = np.zeros((self.S + X, self.S + X), dtype=complex)
        fast_M[X:,:X] = np.conjugate(np.take(self.Z, np.concatenate((tau, rho)), axis = 0).T)
        fast_M[:X,X:] = np.take(other.Z, np.concatenate((tau, sigma)), axis = 0)
        fast_M[X:,X:] = np.identity(self.S) + np.matmul(np.conjugate(reduced_matrix(self.Z, np.concatenate((tau, rho, sigma)), []).T), reduced_matrix(other.Z, np.concatenate((tau, rho, sigma)), []))
        #print(fast_M)
        return(sign(len(tau) + len(sigma)) * tau_sorting_prefactor * np.linalg.det(fast_M))

    def mixed_reduction_overlap(self, other, i, j):
        # i is in pi_0, j is in pi_1
        i = np.array(i, dtype=int) - self.S

        X = len(i)
        fast_M = np.zeros((self.M - self.S, self.M - self.S), dtype=complex)
        fast_M[:X,:X] = np.conjugate(np.take(np.take(self.Z, i, axis = 0), j, axis = 1).T)
        fast_M[:X,X:] = np.conjugate(np.take(reduced_matrix(self.Z, i, []), j, axis = 1).T)
        fast_M[X:,:X] = np.matmul(reduced_matrix(other.Z, i, j), np.conjugate(np.take(reduced_matrix(self.Z, [], j), i, axis = 0).T ))
        fast_M[X:,X:] = np.identity(self.M - self.S - X) + np.matmul(reduced_matrix(other.Z, i, j), np.conjugate(reduced_matrix(self.Z, i, j).T ))
        return(sign(X * self.S + sum(j)) * np.linalg.det(fast_M))
        # Note that when indexing modes from 0, the sign flip (j+1) becomes (j), since a mode at index x has x lower-index modes, rather than x-1
    def mixed_reduction_overlap_alt(self, other, i, j):
        # i is in pi_0, j is in pi_1
        i = np.array(i, dtype=int) - self.S

        X = len(i)
        fast_M = np.zeros((self.S, self.S), dtype=complex)
        fast_M[:X,:X] = np.conjugate(np.take(np.take(self.Z, i, axis = 0), j, axis = 1).T)
        fast_M[X:,:X] = np.conjugate(np.take(reduced_matrix(self.Z, [], j), i, axis = 0).T)
        fast_M[:X,X:] = np.matmul(np.conjugate(np.take(reduced_matrix(self.Z, i, []), j, axis = 1).T ), reduced_matrix(other.Z, i, j))
        fast_M[X:,X:] = np.identity(self.S - X) + np.matmul(np.conjugate(reduced_matrix(self.Z, i, j).T ), reduced_matrix(other.Z, i, j))
        return(sign(X * self.S + sum(j)) * np.linalg.det(fast_M))
        # Note that when indexing modes from 0, the sign flip (j+1) becomes (j), since a mode at index x has x lower-index modes, rather than x-1

    def general_overlap(self, other, c, a):
        # assumes c and a are both ascending. That's the only assumption. I'll add the perm sign soon, then no assumptions will be made (except for non-repeated indices i guess? easy to test for tbh)
        varsigma_a = []
        tau_a = []
        varsigma_b = []
        tau_b = []
        sigma_intersection = []
        len_sigma_a = 0
        len_sigma_b = 0
        len_tau_a = 0
        len_tau_b = 0
        for i in range(len(c)):
            # using len(c) = len(a)
            if c[i] < S:
                # pi_1
                len_sigma_a += 1
                if c[i] not in a:
                    varsigma_a.append(c[i])
                else:
                    sigma_intersection.append(c[i])
            else:
                # pi_0
                len_tau_a += 1
                tau_a.append(c[i])
            if a[i] < S:
                # pi_1
                len_sigma_b += 1
                if a[i] not in c:
                    varsigma_b.append(a[i])
            else:
                # pi_0
                len_tau_b += 1
                tau_b.append(a[i])
        if len_tau_a <= len_tau_b:
            fast_M = np.zeros((len(tau_b) + S - len(sigma_intersection), len(tau_b) + M - S - len(sigma_intersection)), dtype=complex)
            fast_M[]???




"""
lol = occupancy_basis_superposition([[2, occupancy_basis_state({0, 1})], [-5, occupancy_basis_state({0})], [10, occupancy_basis_state({1})], [4, occupancy_basis_state(set())]])
print(lol)
print("Square norm =", lol.overlap(lol))
lol.apply_operators([2], [0])
print(lol)"""


M = 9
S = 5
N = 5

# The order of application is c_last ... c_first a_first ... a_last
global_c_sequence = [2, 3, 1]
global_a_sequence = [1, 2, 0]
#global_c_sequence = [7, 9, 10]
#global_a_sequence = [9, 11, 8]
mixed_i = [6, 7]
mixed_j = [2, 4]

print(f"Calculating < Z_a | creation({list(reversed(global_c_sequence))}) annihilation({global_a_sequence}) | Z_b >")
is_all_equal = True
for i in range(N):
    print("---------- test no.", i)
    A = CS(M, S, random_complex_matrix(M-S, S, (-1.0, 1.0)))
    B = CS(M, S, random_complex_matrix(M-S, S, (-1.0, 1.0)))

    #print("  Origin. A:", A.occupancy_basis_decomposition)
    #print("  Origin. B:", B.occupancy_basis_decomposition)

    """A.occupancy_basis_decomposition.apply_operators([], global_c_sequence)
    B.occupancy_basis_decomposition.apply_operators([], global_a_sequence)
    #print("  Dir. r. A:", A.occupancy_basis_decomposition)
    #print("  Dir. r. B:", B.occupancy_basis_decomposition)

    #A_var = CS(M, S, np.matmul(A.Z, np.matmul(Q(global_c_sequence[0], 1), R(global_c_sequence[0], 1))))
    #B_var = CS(M, S, np.matmul(B.Z, np.matmul(Q(global_a_sequence[0], 1), R(global_a_sequence[0], 1))))
    #A_var.occupancy_basis_decomposition.scale(sign(global_c_sequence[0]))
    #B_var.occupancy_basis_decomposition.scale(sign(global_a_sequence[0]))
    #print("  Equiv. A:", A_var.occupancy_basis_decomposition)
    #print("  Equiv. B:", B_var.occupancy_basis_decomposition)


    slow = A.occupancy_basis_decomposition.overlap(B.occupancy_basis_decomposition)
    quick = A.overlap(B, global_c_sequence, global_a_sequence)
    #quick = A.overlap_with_transition(B, global_c_sequence, global_a_sequence)
    #quick = A.pi_zero_overlap(B, global_c_sequence, global_a_sequence) """

    A.occupancy_basis_decomposition.apply_operators([], mixed_i)
    B.occupancy_basis_decomposition.apply_operators([], mixed_j)

    slow = A.occupancy_basis_decomposition.overlap(B.occupancy_basis_decomposition)
    quick = A.mixed_reduction_overlap(B, mixed_i, mixed_j)
    quick_alt = A.mixed_reduction_overlap_alt(B, mixed_i, mixed_j)

    print(f"  Slow, trustworthy overlap: {slow:.4f}")
    print(f"  Fast overlap:              {quick:.4f}")
    print(f"  Fast alt. overlap:         {quick_alt:.4f}")

    if(np.round(slow, 4) != np.round(quick, 4)):
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
