from two_mode import *

N = 15

BS2 = BH(1, 0.5, 2.0 * np.pi, 0.1, 0, 0, 3, 2)
BS2.sample_CS(N, np.array([-np.sqrt(0.7), np.sqrt(0.3)]))

A_evol, basis_evol, E_evol = BS2.iterate(8.0, 0.01)




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
