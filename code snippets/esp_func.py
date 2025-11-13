import numpy as np

def esp(roots, order):
    # roots is a list
    # order is an integer <= len(roots)

    if order == 0:
        return(1.0)
    if order == len(roots):
        return(np.prod(roots))
    partial_esp = np.zeros(order + 1, dtype = complex)
    partial_esp[0] = 1.0
    for i in range(len(roots)):
        for j in range(min(i + 1, order), 0, -1):
            partial_esp[j] += roots[i] * partial_esp[j - 1]
    return(partial_esp[order])


lol = [1, 5, 3]

print(esp(lol, 1))

old_z_a = np.arange(0, 10, 1)
old_z_b = np.arange(0, 10, 1)
M = len(old_z_a)


def trup(c, a):
    z_a = []
    z_b = []

    prefactor = 1.0
    cur_sign = 1.0
    cur_pos_a = M - 1
    cur_pos_b = M - 1
    for i in range(len(c) - 1, -1, -1):
        prefactor *= np.conjugate(old_z_a[c[i]]) * old_z_b[a[i]]

        for j in range(cur_pos_a, c[i], -1):
            print(j)
            # note we omit c[i] itself, as it is set to zero
            z_a.insert(0, cur_sign * old_z_a[j])
        print("sikogj")
        cur_pos_a = c[i] - 1
        for j in range(cur_pos_b, a[i], -1):
            # note we omit c[i] itself, as it is set to zero
            z_b.insert(0, cur_sign * old_z_b[j])
        cur_pos_b = a[i] - 1
        cur_sign *= -1
    for j in range(cur_pos_a, -1, -1):
        z_a.insert(0, cur_sign * old_z_a[j])
    for j in range(cur_pos_b, -1, -1):
        z_b.insert(0, cur_sign * old_z_b[j])
    print(f"z_a = {z_a}")
    print(f"z_b = {z_b}")

trup([2, 3], [2, 5])

