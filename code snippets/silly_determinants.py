import numpy as np

M = 10
S = 6

def print_matrix(m):
    number_of_rows = len(m)
    number_of_columns = len(m[0])


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

def hc(m):
    return(np.conjugate(m.T))

def overlap(A, B, reduced_columns = []):
    reduction_summants = []
    for col in reduced_columns:
        reduction_summants.append(np.outer(B[:,col], np.conjugate(A[:,col])))
    return np.linalg.det(np.identity(M-S) + np.matmul(B, hc(A)) - np.sum(reduction_summants, axis = 0))



A = random_complex_matrix((M-S), S, (-1, 1))
B = random_complex_matrix((M-S), S, (-1, 1))
print("A matrix:")
print(np.array_str(A, precision=2))
print("B matrix:")
print(np.array_str(B, precision=2))

print("Reducing the second column for A and third column for B:")
print("---------------------- using R-matrices ----------------------")
print(f"  overlap value: {overlap(np.matmul(A, R(1, 1)), np.matmul(B, R(2, 1)))}")
print("  A.R_2")
print(np.array_str(np.matmul(A, R(1, 1)), precision=2))
print("  B.R_3")
print(np.array_str(np.matmul(B, R(2, 1)), precision=2))

print("---------------------- using R-matrices ----------------------")
print(f"  overlap value: {overlap(A, B, [1, 2])}")

"""print("---------------------- using reduction ----------------------")
print(f"  using reduction: {overlap(reduced_matrix(A, [], [1]), reduced_matrix(B, [], [2]))}")
print("  reduced A matrix:")
print(np.array_str(reduced_matrix(A, [], [1, 2]), precision=2))
print("  reduced B matrix:")
print(np.array_str(reduced_matrix(B, [], [1, 2]), precision=2))"""
