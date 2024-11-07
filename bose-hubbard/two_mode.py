import numpy as np
import matplotlib.pyplot as plt

import time

class CS():
    # object Coherent state, namely the SU(2) one

    def __init__(self, M, xi = np.array([])):
        # M is the number of modes, equivalent to the number of complex parameters

        # here we force an SU(2) state
        self.M = 2 #M

        if len(xi) == 0:
            xi = np.zeros(self.M, dtype=complex)
        self.xi = xi


    def spherical_param(self, theta):
        # initializes self.xi from spherical parametrization
        if self.M == 2:
            # theta[0] in (0, pi); theta[1] in (0, 2.pi)
            self.xi[0] = np.cos(theta[0]/2)
            self.xi[1] = np.sin(theta[0]/2) * np.exp(1j * theta[1])

    def overlap(self, other, S, reduction = 0):
        # calculates the overlap integral < other (r) | self (r) >, where r(eduction) is the number of apostrophes
        coef_product = np.sum(np.conjugate(other.xi) * self.xi)
        return(np.power(coef_product, S - reduction))


class BH():
    #driven Bose-Hubbard model

    def __init__(self, J_0, J_1, omega, U, K, j_zero, S, M):
        # Hamiltonian parameters
        self.J_0 = J_0
        self.J_1 = J_1
        self.omega = omega
        self.U = U
        self.K = K
        self.j_zero = j_zero


        # Fock parameters
        self.S = S
        self.M = M

        print("A driven Bose-Hubbard model with %i bosons over %i modes has been initialized." % (self.S, self.M))
        print(f"    | J(t) = {self.J_0} + {self.J_1} . cos({self.omega:.2f}t)   (hopping interaction)")
        print(f"    | U = {self.U}                       (on-site interaction)")
        print(f"    | K = {self.K}, j_0 = {self.j_zero}                (harmonic trapping potential)")

    def sample_CS(self, N, xi_1 = np.array([])):
        # samples N-1 CSs with the "unnormalized, uncoupled z" method, plus the initial pure CS state
        if len(xi_1) == 0:
            xi_1 = np.zeros(self.M, dtype=complex)
            xi_1[0] = 1.0
        self.basis = [CS(self.M, xi_1)]
        for i in range(N-1):
            z = np.zeros(self.M, dtype=complex)
            z_mag_sq = 0.0
            for j in range(self.M):
                z[j] = (2 * np.random.rand() - 1) + 1j * (2 * np.random.rand() - 1)
                z_mag_sq += z[j].real * z[j].real + z[j].imag * z[j].imag
            z_mag_sq = np.sqrt(z_mag_sq)
            for j in range(self.M):
                z[j] /= z_mag_sq
            self.basis.append(CS(self.M, z))

        print("A basis set of size %i has been initialized." % N)

    def J(self, t):
        return(self.J_0 + self.J_1 * np.cos(self.omega * t))

    def H(self, t, cur_A, cur_basis):
        # this evaluates < Psi | H | Psi >
        N = len(cur_basis)
        M = len(cur_basis[0].xi)
        H = 0.0
        for k in range(N):
            for j in range(N):
                sum1 = 0.0
                for i in range(M-1):
                    sum1 += (np.conjugate(cur_basis[k].xi[i]) * cur_basis[j].xi[i+1] + np.conjugate(cur_basis[k].xi[i+1]) * cur_basis[j].xi[i]) * cur_basis[j].overlap(cur_basis[k], self.S, reduction = 1)
                sum1 *= (- self.J(t) * self.S)

                sum2 = 0.0
                for i in range(M):
                    sum2 += (np.conjugate(cur_basis[k].xi[i]) * np.conjugate(cur_basis[k].xi[i]) * cur_basis[j].xi[i] * cur_basis[j].xi[i]) * cur_basis[j].overlap(cur_basis[k], self.S, reduction = 2)
                sum2 *= (self.U * self.S * (self.S - 1) / 2)

                sum3 = 0.0
                for i in range(M):
                    sum3 += (1 + i - self.j_zero) * (1 + i - self.j_zero) * (np.conjugate(cur_basis[k].xi[i]) * cur_basis[j].xi[i]) * cur_basis[j].overlap(cur_basis[k], self.S, reduction = 1)
                sum3 *= (self.S * self.K / 2)

                H += np.conjugate(cur_A[k]) * cur_A[j] * (sum1 + sum2 + sum3)

    def Theta(self, t, cur_A, cur_basis):
        # cur_A is an ndarray of complex decomposition coefficients A(t) of length N
        # cur_basis is a list of N instances of CS, each possessing an ndarray of complex parameters of length M
        N = len(cur_basis)
        M = len(cur_basis[0].xi)

        m_Theta = np.zeros(((M+1)*N, (M+1)*N), dtype=complex)

        # First, we fill in X
        for i in range(N):
            for j in range(N):
                m_Theta[i][j] = cur_basis[j].overlap(cur_basis[i], self.S)
        # Then, we fill in Y and Y^h.c.
        for a in range(N):
            for b in range(M):
                for d in range(N):
                    m_Theta[a][N + b * N + d] = self.S * np.conjugate(cur_basis[a].xi[b]) * cur_A[d] * cur_basis[d].overlap(cur_basis[a], self.S, reduction = 1)
                    m_Theta[N + b * N + d][a] = np.conjugate(m_Theta[a][N + b * N + d])
        # Then, we fill in Z
        for i in range(M):
            for j in range(M):
                for a in range(N):
                    for b in range(N):
                        # first, we evaluate (F_ij)_ab
                        m_Theta[N + i * N + a][N + j * N + b] = self.S * (self.S - 1) * cur_basis[b].overlap(cur_basis[a], self.S, reduction = 2) * np.conjugate(cur_basis[a].xi[j]) * cur_basis[b].xi[i]
                        if i == j:
                            m_Theta[N + i * N + a][N + j * N + b] += self.S * cur_basis[b].overlap(cur_basis[a], self.S, reduction = 1)
                        m_Theta[N + i * N + a][N + j * N + b] *= np.conjugate(cur_A[a]) * cur_A[b]
        return(m_Theta)

    def R(self, t, cur_A, cur_basis):
        N = len(cur_basis)
        M = len(cur_basis[0].xi)

        R = np.zeros((M+1)*N, dtype=complex)
        # First, we fill in R_1
        for k in range(N):
            for j in range(N):
                sum1 = 0.0
                for i in range(M-1):
                    sum1 += (np.conjugate(cur_basis[k].xi[i]) * cur_basis[j].xi[i+1] + np.conjugate(cur_basis[k].xi[i+1]) * cur_basis[j].xi[i]) * cur_basis[j].overlap(cur_basis[k], self.S, reduction = 1)
                sum1 *= (- self.J(t) * self.S)

                sum2 = 0.0
                for i in range(M):
                    sum2 += (np.conjugate(cur_basis[k].xi[i]) * np.conjugate(cur_basis[k].xi[i]) * cur_basis[j].xi[i] * cur_basis[j].xi[i]) * cur_basis[j].overlap(cur_basis[k], self.S, reduction = 2)
                sum2 *= (self.U * self.S * (self.S - 1) / 2)

                sum3 = 0.0
                for i in range(M):
                    sum3 += (1 + i - self.j_zero) * (1 + i - self.j_zero) * (np.conjugate(cur_basis[k].xi[i]) * cur_basis[j].xi[i]) * cur_basis[j].overlap(cur_basis[k], self.S, reduction = 1)
                sum3 *= (self.S * self.K / 2)

                R[k] += cur_A[j] * (sum1 + sum2 + sum3)

        # Then, we fill in R_2
        for m in range(M):
            for k in range(N):
                for j in range(N):
                    if m == 0:
                        term1 = - self.J(t) * self.S * cur_basis[j].xi[m+1] * cur_basis[j].overlap(cur_basis[k], self.S, reduction = 1)
                    elif m == M-1:
                        term1 = - self.J(t) * self.S * cur_basis[j].xi[m-1] * cur_basis[j].overlap(cur_basis[k], self.S, reduction = 1)
                    else:
                        term1 = - self.J(t) * self.S * (cur_basis[j].xi[m+1] + cur_basis[j].xi[m-1]) * cur_basis[j].overlap(cur_basis[k], self.S, reduction = 1)

                    term2 = 0.0
                    for i in range(M-1):
                        term2 += (np.conjugate(cur_basis[k].xi[i]) * cur_basis[j].xi[i+1] + np.conjugate(cur_basis[k].xi[i+1]) * cur_basis[j].xi[i]) * cur_basis[j].overlap(cur_basis[k], self.S, reduction=2)
                    term2 *= (- self.J(t) * self.S * (self.S - 1) * cur_basis[j].xi[m])

                    term3 = self.U * self.S * (self.S - 1) * (np.conjugate(cur_basis[k].xi[m]) * cur_basis[j].xi[m] * cur_basis[j].xi[m]) * cur_basis[j].overlap(cur_basis[k], self.S, reduction = 2)

                    term4 = 0.0
                    for i in range(M):
                        term4 += (np.conjugate(cur_basis[k].xi[i]) * np.conjugate(cur_basis[k].xi[i]) * cur_basis[j].xi[i] * cur_basis[j].xi[i]) * cur_basis[j].overlap(cur_basis[k], self.S, reduction = 3)
                    term4 *= (self.U * self.S * (self.S - 1) * (self.S - 2) * cur_basis[j].xi[m] / 2)

                    term5 = (self.K / 2) * self.S * (1 + m - self.j_zero) * (1 + m - self.j_zero) * cur_basis[j].xi[m] * cur_basis[j].overlap(cur_basis[k], self.S, reduction = 1)

                    term6 = 0.0
                    for i in range(M):
                        term6 += (1 + i - self.j_zero) * (1 + i - self.j_zero) * (np.conjugate(cur_basis[k].xi[i]) * cur_basis[j].xi[i]) * cur_basis[j].overlap(cur_basis[k], self.S, reduction = 2)
                    term6 *= self.K * self.S * (self.S - 1) * cur_basis[j].xi[m] / 2

                    R[N + m * N + k] += np.conjugate(cur_A[k]) * cur_A[j] * (term1 + term2 + term3 + term4 + term5 + term6)

        return(R)



    def iterate(self, max_t, dt, xi_1 = np.array([])):
        # everything is in natural units (hbar=1)
        # maximum time will actually be J_0 * max_t, which will also be the units we display it in
        N = len(self.basis)

        if len(xi_1) == 0:
            xi_1 = self.basis[0]
        else:
            xi_1 = CS(self.M, xi_1)

        t_space = np.arange(0.0, self.J_0 * max_t + dt, dt)

        step_N = len(t_space)

        A_evol = np.zeros((step_N, N), dtype=complex)
        basis_evol = np.zeros((step_N, N, self.M), dtype=complex)
        E_evol = np.zeros(step_N, dtype=complex)

        def record_state(t_i, cur_it_A, cur_it_basis):
            for i in range(N):
                A_evol[t_i][i] = cur_it_A[i]
                for j in range(self.M):
                    basis_evol[t_i][i][j] = cur_it_basis[i].xi[j]
            E_evol[t_i] = self.H(t_i * dt, cur_it_A, cur_it_basis)

        # we set up the iterated variables

        # Here we initialize A(0) with the naive algorithm
        it_A = np.zeros(N, dtype=complex)
        for i in range(N):
            it_A[i] = xi_1.overlap(self.basis[i], self.S)
        Psi_0_mag = 0.0
        for a in range(N):
            for b in range(N):
                Psi_0_mag += xi_1.overlap(self.basis[a], self.S) * self.basis[b].overlap(xi_1, self.S) * self.basis[a].overlap(self.basis[b], self.S)
        it_A /= Psi_0_mag

        it_basis = []
        for i in range(N):
            it_basis.append(CS(self.M, self.basis[i].xi.copy()))

        record_state(0, it_A, it_basis)

        start_time = time.time()
        progress = 0

        print(f"Iterative simulation of the Bose-Hubbard model on a timescale of t_max = {self.J_0 * max_t}, dt = {dt} ({step_N} steps) at {time.strftime("%H:%M:%S", time.localtime( start_time))}")

        for t_i in range(1, step_N):
            cur_t = (t_i-1) * dt
            # RK4

            #k1
            it_A_copy = it_A.copy()
            it_basis_copy = []
            for i in range(N):
                it_basis_copy.append(CS(self.M, it_basis[i].xi.copy()))
            cur_R = self.R(cur_t, it_A_copy, it_basis_copy)

            """based = self.Theta(cur_t, it_A_copy, it_basis_copy)
            print(based)
            plt.imshow(np.abs(based), cmap='hot', interpolation='nearest')
            plt.show()"""
            #print(it_A)

            cur_Theta_inv = np.linalg.inv(self.Theta(cur_t, it_A_copy, it_basis_copy))
            k1 = - 1j * cur_Theta_inv.dot(cur_R)

            #k2
            it_A_copy += (dt / 2) * k1[:N]
            for n in range(N):
                it_basis_copy[n].xi += (dt / 2) * k1[N + self.M * n:N + self.M * (n+1)]
            cur_R = self.R(cur_t + dt / 2, it_A_copy, it_basis_copy)
            cur_Theta_inv = np.linalg.inv(self.Theta(cur_t + dt / 2, it_A_copy, it_basis_copy))
            k2 = - 1j * cur_Theta_inv.dot(cur_R)

            #k3
            it_A_copy += (dt / 2) * (k2[:N]-k1[:N])
            for n in range(N):
                it_basis_copy[n].xi += (dt / 2) * (k2[N + self.M * n:N + self.M * (n+1)]-k1[N + self.M * n:N + self.M * (n+1)])
            cur_R = self.R(cur_t + dt / 2, it_A_copy, it_basis_copy)
            cur_Theta_inv = np.linalg.inv(self.Theta(cur_t + dt / 2, it_A_copy, it_basis_copy))
            k3 = - 1j * cur_Theta_inv.dot(cur_R)

            #k4
            it_A_copy += (dt / 2) * (2 * k3[:N]-k2[:N])
            for n in range(N):
                it_basis_copy[n].xi += (dt / 2) * (2 * k3[N + self.M * n:N + self.M * (n+1)]-k2[N + self.M * n:N + self.M * (n+1)])
            cur_R = self.R(cur_t + dt, it_A_copy, it_basis_copy)
            cur_Theta_inv = np.linalg.inv(self.Theta(cur_t + dt, it_A_copy, it_basis_copy))
            k4 = - 1j * cur_Theta_inv.dot(cur_R)

            it_A += (dt / 6) * (k1[:N] + 2 * k2[:N] + 2 * k3[:N] + k4[:N])
            for n in range(N):
                it_basis[n].xi += (dt / 6) * (k1[N + self.M * n:N + self.M * (n+1)] + 2 * k2[N + self.M * n:N + self.M * (n+1)] + 2 * k3[N + self.M * n:N + self.M * (n+1)] + k4[N + self.M * n:N + self.M * (n+1)])

            record_state(t_i, it_A, it_basis)
            if np.floor(t_i / step_N * 100) > progress:
                progress = np.floor(t_i / step_N * 100)
                print("  " + str(progress) + "% done; est. time of finish: " + time.strftime("%H:%M:%S", time.localtime( (time.time()-start_time) * 100 / progress + start_time )), end='\r')


        # This function outputs the following arrays:
        #    1. [t][n] = A_n(t)
        #    2. [t] = <Psi(t) | H | Psi(t)> = <E>(t), for checking whether energy is conserved
        #    3. [t][n] = sum_m |xi_nm(t)|^2 (for checking if CSs stay SU(N)-normalized during their dynamical evolution)
        return(t_space, A_evol, basis_evol, E_evol)


