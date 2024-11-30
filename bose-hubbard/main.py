from two_mode import *

N = 15
S = 3
M = 2

BS2 = BH(1, 0.5, 2.0 * np.pi, 0.1, 0, 0, S, M)
BS2.sample_CS(N, np.array([-np.sqrt(0.7), np.sqrt(0.3)], dtype=complex))

t_space, A_evol, basis_evol, E_evol = BS2.iterate(0.08, 0.00002)

psi_mag = np.zeros(len(t_space))
avg_n_1 = np.zeros(len(t_space), dtype=complex)

for t_i in range(len(t_space)):
    for a in range(N):
        for b in range(N):
            psi_mag[t_i] += np.conjugate(A_evol[t_i][a]) * A_evol[t_i][b] * CS(M, basis_evol[t_i][b]).overlap(CS(M, basis_evol[t_i][a]), S)
            avg_n_1[t_i] += np.conjugate(A_evol[t_i][a]) * A_evol[t_i][b] * np.conjugate(basis_evol[t_i][a][0]) * basis_evol[t_i][b][0] * CS(M, basis_evol[t_i][b]).overlap(CS(M, basis_evol[t_i][a]), S, reduction = 1)
plt.plot(t_space, psi_mag, label="$\\langle \\Psi | \\Psi \\rangle}$")
plt.plot(t_space, avg_n_1, label="$\\frac{\\langle N_1 \\rangle}{S}$")
plt.legend()
plt.show()

"""
# we plot the z1 and z2 of the basis set for fun

plt.subplot(1, 2, 1)
plt.title("xi_1")
plt.xlabel("Re(xi_1)")
plt.ylabel("Im(xi_1)")
plt.xlim((-1, 1))
plt.ylim((-1, 1))
cur_x = []
cur_y = []
for i in range(N):
    cur_x.append(BS2.basis[i].xi[0].real)
    cur_y.append(BS2.basis[i].xi[0].imag)
plt.scatter(cur_x, cur_y)
plt.gca().set_aspect("equal")

plt.subplot(1, 2, 2)
plt.title("xi_2")
plt.xlabel("Re(xi_1)")
plt.ylabel("Im(xi_1)")
plt.xlim((-1, 1))
plt.ylim((-1, 1))
cur_x = []
cur_y = []
for i in range(N):
    cur_x.append(BS2.basis[i].xi[1].real)
    cur_y.append(BS2.basis[i].xi[1].imag)
plt.scatter(cur_x, cur_y)
plt.gca().set_aspect("equal")

plt.tight_layout()
plt.show()
"""
