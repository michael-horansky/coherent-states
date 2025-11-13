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

def take(m, row_indices = None, column_indices = None):
    if row_indices is None:
        return(np.take(m, column_indices, axis = 1))
    if column_indices is None:
        return(np.take(m, row_indices, axis = 0))
    return(np.take(np.take(m, row_indices, axis = 0), column_indices, axis = 1))


# dimensions
x = 5
y = 7

# removed indices
m = [1, 3]
n = [0, 2, 5]

m_comp = []
n_comp = []
for i in range(x):
    if i not in m:
        m_comp.append(i)
for i in range(y):
    if i not in n:
        n_comp.append(i)


A = np.random.random((x, y))
B = np.random.random((y, x))

master_matrix = np.identity(x) + np.matmul(A, B)

# reordering
reordered_matrix = np.zeros((x, x))
reordered_matrix[:len(m),:len(m)] = np.identity(len(m)) + np.matmul(take(A, m, None), take(B, None, m))
reordered_matrix[len(m):,:len(m)] = np.matmul(take(A, m_comp, None), take(B, None, m))
reordered_matrix[:len(m),len(m):] = np.matmul(take(A, m, None), take(B, None, m_comp))
reordered_matrix[len(m):,len(m):] = np.identity(x - len(m)) + np.matmul(take(A, m_comp, None), take(B, None, m_comp))

print("----------- master matrix ----------------")
print(master_matrix)

print("----------- reordered matrix ----------------")
print(reordered_matrix)



print("Directly:", np.linalg.det(master_matrix))
print("Reordered:", np.linalg.det(reordered_matrix))



"""

direct_comp = np.linalg.det(np.identity(x - len(m)) + np.matmul(reduced_matrix(A, m, n), reduced_matrix(B, n, m)))

print("Directly:", direct_comp)

quick_comp = np.linalg.det(reduced_matrix(master_matrix, m, m  ))
print("Quickly:", quick_comp)"""
