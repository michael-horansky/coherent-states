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

def test_anti_compound_matrix():

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

def test_weighted_minor_sum():

    n = 5

    M = random_complex_matrix(n, n, (-1, 1))

    r = 2
    V = np.zeros(tuple([n] * (2 * r)), dtype = complex)

    """for i in range(n):
        for j in range(i + 1, n):
            for i_prime in range(n):
                for j_prime in range(i_prime + 1, n):
                    V[i][j][i_prime][j_prime] = np.random.rand() * 2 - 1"""
                    #V[j][i][i_prime][j_prime] = - V[i][j][i_prime][j_prime]
                    #V[i][j][j_prime][i_prime] = - V[i][j][i_prime][j_prime]
                    #V[j][i][j_prime][i_prime] = V[i][j][i_prime][j_prime]
                    #V[i_prime][j_prime][i][j] = np.conjugate(V[i][j][i_prime][j_prime])
                    #V[j_prime][i_prime][i][j] = - V[i_prime][j_prime][i][j]
                    #V[i_prime][j_prime][j][i] = - V[i_prime][j_prime][i][j]
                    #V[j_prime][i_prime][j][i] = V[i_prime][j_prime][i][j]
    """
    # diagonal V
    for i in range(n):
        for j in range(i + 1, n):
                V[i][j][i][j] = np.random.rand() * 2 - 1
                V[j][i][j][i] = V[i][j][i][j]
    """
    a = 2
    b = 3
    V[a][b][a][b] = 1.0
    #V[b][a][b][a] = V[a][b][a][b]

    slow_sum = 0.0
    diag_slow_sum = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            for i_prime in range(n):
                for j_prime in range(i_prime + 1, n):
                    if i == i_prime and j == j_prime:
                        diag_slow_sum += np.linalg.det(take(M, [i, j], [i_prime, j_prime]))
                    slow_sum += V[i][j][i_prime][j_prime] * np.linalg.det(take(M, [i, j], [i_prime, j_prime]))
    print(f"  Diagonal slow sum = {diag_slow_sum:.4f}")
    print(f"  Slow sum = {slow_sum:.4f}")

    diag_quick_sum = 0.5 * (np.trace(M) * np.trace(M) - np.trace(np.matmul(M, M)))
    print(f"  Diagonal quick sum = {diag_quick_sum:.4f}")

    """reduced_V_a = np.zeros((n, n), dtype=complex)
    reduced_V_b = np.zeros((n, n), dtype=complex)
    #reduced_V_c = np.zeros((n, n), dtype=complex)
    #reduced_V_d = np.zeros((n, n), dtype=complex)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for l in range(n):
                    reduced_V_a[i][j] += V[i][k][j][l]
                    reduced_V_b[i][j] += V[k][i][l][j]
                #reduced_V_c[i][j] += V[i][k][k][j]
                #reduced_V_d[i][j] += V[k][i][j][k]"""
    reduced_V_a =  np.sum(V, axis = (1, 3)) + np.sum(V, axis = (0, 2))
    reduced_V_b =  np.sum(V, axis = (0, 2)) + np.sum(V, axis = (1, 3))
    print(reduced_V_a)
    print(reduced_V_b)


    N = np.matmul(np.matmul(M, reduced_V_b.T), np.matmul(M, reduced_V_a.T))
    quick_sum = 0.5 * (np.trace(np.matmul(M, reduced_V_a.T)) * np.trace(np.matmul(M, reduced_V_b.T)) - np.trace(N)) / (np.sum(V))
    print(f"  Quick sum = {quick_sum:.4f}")

    """V_var = np.zeros(tuple([n] * (2 * r)), dtype = complex)
    for i in range(n):
        for j in range(n):
                V_var[i][j][i][j] = 1.0
                #V[j][i][j][i] = 1.0

    slow_sum_var = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            for i_prime in range(n):
                for j_prime in range(i_prime + 1, n):
                    slow_sum_var += V_var[i][j][i_prime][j_prime] * np.linalg.det(take(M, [i, j], [i_prime, j_prime]))
    print(f"  Slow sum (var) = {slow_sum_var:.4f}")
    reduced_V_a_var = np.sum(V_var, axis = (1, 3))
    reduced_V_b_var = np.sum(V_var, axis = (0, 2))
    print(reduced_V_a_var)
    print(reduced_V_b_var)


    N_var = np.matmul(np.matmul(M, reduced_V_b.T), np.matmul(M, reduced_V_a.T))
    quick_sum_var = 0.5 * (np.trace(np.matmul(M, reduced_V_a_var.T)) * np.trace(np.matmul(M, reduced_V_b_var.T)) - np.trace(N_var)) / np.sum(V_var)
    print(f"  Quick sum = {quick_sum_var:.4f}")"""

def test_theorem_F():

    m = 5
    n = 7

    rho = [0, 2, 3]
    sigma = [4, 6]

    A = random_complex_matrix(m, n, (-1, 1))
    B = random_complex_matrix(n, m, (-1, 1))

    X_inv = np.linalg.inv(np.delete(np.delete(np.identity(m)+A@B, rho, axis = 0), rho, axis = 1))

    LHS = np.linalg.det(np.identity(m)+A@B) / np.linalg.det(np.identity(m - len(rho)) + np.delete(np.delete(A, rho, axis = 0), sigma, axis = 1) @ np.delete(np.delete(B, sigma, axis = 0), rho, axis = 1))

    RHS = np.linalg.det(np.identity(len(rho)) + np.take(A, rho, axis = 0) @ (np.identity(n) - np.delete(B, rho, axis = 1) @ X_inv @ np.delete(A, rho, axis = 0)) @ np.take(B, rho, axis = 1)) / np.linalg.det(np.identity(len(sigma)) - np.delete(np.take(B, sigma, axis = 0), rho, axis = 1) @ X_inv @ np.take(np.delete(A, rho, axis = 0), sigma, axis = 1))

    if np.round(LHS, 4) == np.round(RHS, 4):
        print("Theorem F holds!")
    else:
        print("It doesn't work.")

#test_weighted_minor_sum()
test_theorem_F()
