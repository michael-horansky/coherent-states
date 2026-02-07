import numpy as np

def overlap(z, m):
    """
    z : array of length 5
    m : integer
    """
    e = np.zeros((5, 5), dtype=complex)

    e[0, 0] = z[0]
    for i in range(1, 5):
        e[i, 0] = e[i-1, 0] + z[i]

    for n in range(1, 5):
        for k in range(1, n+1):
            e[n, k] = e[n-1, k] + z[n] * e[n-1, k-1]

    return e[4, m-1]   # Matlab is 1-based

