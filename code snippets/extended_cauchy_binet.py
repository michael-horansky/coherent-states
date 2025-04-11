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

def sign(i):
    if i % 2 == 0:
        return(1.0)
    else:
        return(-1.0)

def recast_choice(choice, excluded_indices, include_excluded_indices = False):
    recasted = []
    reduced_index = 0
    for i in range(max(max(choice) + len(excluded_indices) + 1, max(excluded_indices) + len(choice) + 1)):
        if i not in excluded_indices:
            if reduced_index in choice:
                recasted.append(i)
            reduced_index += 1
        elif include_excluded_indices:
            recasted.append(i)
    return(recasted)

def R(i):
    res = np.identity(M)
    res[i][i] = 0.0
    return(res)

def Q(i):
    res = np.identity(M)
    for j in range(i):
        res[j][j] = -1.0
    return(res)


N = 7
M = 9
S_u = [2, 4]
S_v = [4, 7] # len must match S_v
u = random_complex_matrix(N, M, (-1, 1))
v = random_complex_matrix(M, N, (-1, 1))

r = 3 # we're looking at all r-rank minors

row_choices = subset_indices(np.arange(0, N, 1), r)
reduced_column_choices = subset_indices(np.arange(0, M - len(S_u), 1), r - len(S_u))

# First, direct summation calculation for a specific row choice
row_choice = np.array([1, 4, 6])
slow = 0.0
for c_choice in reduced_column_choices:
    u_sub = u[np.ix_(row_choice, recast_choice(c_choice, S_u, True))]
    v_sub = v[np.ix_(recast_choice(c_choice, S_v, True), row_choice)]
    slow += np.linalg.det(u_sub) * np.linalg.det(v_sub)

print(f"slow: {slow:.2f}")



fast_M = np.zeros((len(S_u) + len(row_choice), len(S_u) + len(row_choice)), dtype=complex)
fast_M[len(S_u):,:len(S_u)] = u[np.ix_(row_choice, S_u)]
fast_M[:len(S_u),len(S_u):] = v[np.ix_(S_v, row_choice)]
fast_M[len(S_u):,len(S_u):] = np.matmul(np.take(reduced_matrix(np.matmul(u, Q(S_u[0])), [], S_u), row_choice, axis = 0), np.take(reduced_matrix(np.matmul(Q(S_v[0]), v), S_v, []), row_choice, axis = 1) )

fast = sign(len(S_u)) * np.linalg.det(fast_M)

print(f"fast: {fast:.2f}")

ultrafast = np.linalg.det(np.matmul(np.matmul(fast_M[:len(S_u),len(S_u):], np.linalg.inv(fast_M[len(S_u):,len(S_u):])), fast_M[len(S_u):,:len(S_u)])) * np.linalg.det(fast_M[len(S_u):,len(S_u):])

print(f"ultrafast: {ultrafast:.2f}")

"""all_choices = subset_indices(np.arange(0, M, 1), len(S_u))
choices_u = []
choices_v = []
for choice in all_choices:
    if set(S_u).issubset(set(choice)):
        choices_u.append(choice)
    if set(S_v).issubset(set(choice)):
        choices_v.append(choice)"""



