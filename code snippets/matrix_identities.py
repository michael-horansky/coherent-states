import numpy as np

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

def take(m, row_indices, column_indices):
    return(np.take(np.take(m, row_indices, axis = 0), column_indices, axis = 1))


"""
# First: UAV for inverse of principally-reduced equals principally-reduced of inverse

m = 5
n = 7

U = random_complex_matrix(m, n, (-1.0, 1.0))
V = random_complex_matrix(n, m, (-1.0, 1.0))
A = random_complex_matrix(n, n, (-1.0, 1.0))

A_inv = np.linalg.inv(A)

mu = [2, 3, 5]

A_reduced = reduced_matrix(A, mu, mu)
A_reduced_inv = np.linalg.inv(A_reduced)

print("Direct method:", np.matmul(reduced_matrix(U, [], mu), np.matmul(A_reduced_inv, reduced_matrix(V, mu, []))))
#print("Indirect method:", np.matmul(reduced_matrix(U, [], mu), np.matmul(reduced_matrix(A_inv, mu, mu), reduced_matrix(V, mu, []))))
#print("Indirect method:", np.matmul(U, np.matmul(A_inv, V)) - np.matmul(np.take(U, mu, axis = 1), np.matmul(np.take(np.take(A_inv, mu, axis = 0), mu, axis = 1), np.take(V, mu, axis = 0))))

# The "set to zero" method
U_copy = np.array(U)
V_copy = np.array(V)

U_copy[:,mu] = np.zeros((m, len(mu)))
V_copy[mu,:] = np.zeros((len(mu), m))
#print("Indirect method:", np.matmul(U_copy, np.matmul(A_inv, V_copy)))

# the gepetto method
reduced_inverse = reduced_matrix(A_inv, mu, mu) - np.matmul(np.take(reduced_matrix(A_inv, mu, []), mu, axis = 1), np.matmul(np.take(np.take(A_inv, mu, axis = 0), mu, axis = 1), np.take(reduced_matrix(A_inv, [], mu), mu, axis = 0)))
#print("Indirect method:", np.matmul(reduced_matrix(U, [], mu), np.matmul(reduced_inverse, reduced_matrix(V, mu, []))))

# https://epubs.siam.org/doi/epdf/10.1137/S0895479892227761
print("Directly calculating the inverse of reduced matrix")
print("(A_reduced)^-1 =", A_reduced_inv)
sussily_reduced_A = reduced_matrix(A_inv, mu, mu) - np.matmul(np.take(reduced_matrix(A_inv, mu, []), mu, axis = 1), np.matmul(np.linalg.inv(A_inv[mu,:][:,mu]), np.take(reduced_matrix(A_inv, [], mu), mu, axis = 0)))
print("Indirectly:", sussily_reduced_A)
if (np.round(A_reduced_inv, 2) == np.round(sussily_reduced_A, 2)).all():
    print(" They're equal!")"""


# Anti-compound matrix:
# We have a tensor V_i,i', where i = 1 ... n and i' = 1 ... n, i neq i', and V_i,i' = V_i',i
# and we wish to find an n x n square matrix W such that det W_<i,i'>,<i,i'> = V_i,i'

n = 5
V = np.zeros((n, n), dtype=complex)
for i in range(n):
    for j in range(i + 1, n):
        V[i][j] = np.random.random() * 1
        V[j][i] = V[i][j]

W = np.zeros((n, n), dtype=complex)

# diagonal terms
for i in range(n):
    silly_sum = 0.0
    for j in range(n):
        silly_sum += np.sqrt(V[i][j])
    W[i][i] = silly_sum
# off-diagonal terms
for i in range(n):
    for j in range(i + 1, n):
        W[i][j] = np.sqrt(W[i][i] * W[j][j] - V[i][j])
        W[j][i] = W[i][j]

#print("----------- W -------------")
#print(W)
#print("----------- V -------------")
#print(V)

# Now for the test:
passes = True
for i in range(n):
    for j in range(i + 1, n):
        subdet = np.linalg.det(take(W, [i,j],[i,j]))
        if np.round(subdet, 2) != np.round(V[i][j], 2):
            passes = False
            print(np.round(subdet, 2), "should be equal to", np.round(V[i][j], 2))
if passes:
    print("It works!")


