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


# Here we test Theorem E.3, the most general statement of the problem

def take_constrained(mat, const, seq):
    # both const and seq are tuples
    # seq_i contains 0 to dim(mat)_i - const_i
    const_rows, const_cols = const
    seq_rows, seq_cols = seq

    shifted_seq_rows = []
    shifted_seq_cols = []
    for i in range(const_rows):
        shifted_seq_rows.append(i)
    for i in range(const_cols):
        shifted_seq_cols.append(i)
    for i in range(len(seq_rows)):
        shifted_seq_rows.append(seq_rows[i] + const_rows)
    for i in range(len(seq_cols)):
        shifted_seq_cols.append(seq_cols[i] + const_cols)
    return(np.take(np.take(mat, shifted_seq_rows, axis = 0), shifted_seq_cols, axis = 1))

dim_a = 5
dim_b = 10

u_x = 0
v_x = 3
u_y = 3
v_y = v_x + u_y - u_x
print(f"u_x = {u_x}, v_x = {v_x}; u_y = {u_y}, v_y = {v_y}")
if v_y < v_x:
    print("ERROR: take v_y \\geq v_x")

X_rows = u_x + dim_a
X_cols = v_x + dim_b
Y_rows = v_y + dim_b
Y_cols = u_y + dim_a

print(f"X is ({X_rows}, {X_cols}); Y is ({Y_rows}, {Y_cols})")

X = random_complex_matrix(X_rows, X_cols, (-1, 1))
Y = random_complex_matrix(Y_rows, Y_cols, (-1, 1))

# r represents the size of the smaller minor
# For the minors to be square, we demand
#   1. u_x + r_a = v_x + r_b = r >= 0
#   2. u_y + r_a = v_y + r_b = r >= 0
# Satisfying 1. satisfies 2. given the constraint u_x + v_y = v_x + u_y
# We have u_x + r_a >= 0, u_y + r_a >= 0, similar for r_b
# Hence r_a = r - min(u_x, u_y), r_b = r - min(v_x, v_y)
# and r ranges from max(min(u_x, u_y), min(v_x, v_y)) to min(dim_a - min(u_x, u_y), dim_b - min(v_x, v_y)) inclusive

direct_sum = 0.0
for r in range(max(min(u_x, u_y), min(v_x, v_y)), min(dim_a + min(u_x, u_y), dim_b + min(v_x, v_y)) + 1):
    r_a = r - min(u_x, u_y)
    r_b = r - min(v_x, v_y)
    print(f"r = {r}; r_a = {r_a}, r_b = {r_b}")
    print(f"  minor in X = ({u_x + r_a}, {v_x + r_b})")
    print(f"  minor in Y = ({u_y + r_a}, {v_y + r_b})")
    a_set = subset_indices(np.arange(0, dim_a, 1, dtype=int), r_a)
    for a in a_set:
        b_set = subset_indices(np.arange(0, dim_b, 1, dtype=int), r_b)
        for b in b_set:
            X_minor = np.linalg.det(take_constrained(X, (u_x, v_x), (a, b)))
            Y_minor = np.linalg.det(take_constrained(Y, (v_y, u_y), (b, a)))
            direct_sum += X_minor * Y_minor

print(f"Direct approach: S = {direct_sum}")

v_x_seq = np.arange(0, v_x, 1, dtype=int)
v_y_seq = np.arange(0, v_y, 1, dtype=int)

fast_M = np.zeros((v_y + u_x + dim_a, v_x + u_y + dim_a), dtype=complex)
# fast_M[:v_y, :v_x] = 0
fast_M[v_y:, :v_x] = np.take(X, v_x_seq, axis = 1)
fast_M[:v_y, v_x:] = np.take(Y, v_y_seq, axis = 0)
fast_M[v_y:, v_x:] = np.matmul(reduced_matrix(X, [], v_x_seq), reduced_matrix(Y, v_y_seq, []))
fast_M[u_x+v_y:, u_x+v_y:] += np.identity(dim_a)
print(f"Indirect app.: S = {sign(v_y * (1 + v_y - v_x)) * np.linalg.det(fast_M)}")





"""rows = 9
cols = 5

X = random_complex_matrix(rows, cols, (-1, 1))
Y = random_complex_matrix(cols, rows, (-1, 1))

m = 1 # constraint on rows of X
n = 1 # constrint on cols of X
m_i = list(np.arange(0, m, 1, dtype = int))
n_i = list(np.arange(0, n, 1, dtype = int))

# ------------------- Direct calculation
direct_sum = 0.0

for r in range(min(rows, cols) + 1):
    a_set = subset_indices(np.arange(m, rows, 1, dtype=int), r)
    for a in a_set:
        b_set = subset_indices(np.arange(n, cols, 1, dtype=int), r)
        for b in b_set:
            direct_sum += np.linalg.det(np.take(np.take(X, m_i + a, axis = 0), n_i + b, axis = 1)) * np.linalg.det(np.take(np.take(Y, n_i + b, axis = 0), m_i + a, axis = 1))

# ------------------ Indirect calculation
fast_M = np.zeros((rows + n, rows + n), dtype=complex)
fast_M[n:,:n] = np.take(X, n_i, axis = 1)
fast_M[:n,n:] = np.take(Y, n_i, axis = 0)
fast_M[n:,n:] = np.matmul(reduced_matrix(X, [], n_i), reduced_matrix(Y, n_i, []))
fast_M[n+m:,n+m:] += np.identity(rows - m)
indirect_a = sign(n) * np.linalg.det(fast_M)

fast_N = np.zeros((cols + m, cols + m), dtype=complex)
fast_N[m:,:m] = np.take(Y, m_i, axis = 1)
fast_N[:m,m:] = np.take(X, m_i, axis = 0)
fast_N[m:,m:] = np.matmul(reduced_matrix(Y, [], m_i), reduced_matrix(X, m_i, []))
fast_N[n+m:,n+m:] += np.identity(cols - n)
indirect_b = sign(n) * np.linalg.det(fast_N)

print(f"Direct approach: {direct_sum}")
print(f"Indirect app. a: {indirect_a}")
print(f"Indirect app. b: {indirect_b}")"""




