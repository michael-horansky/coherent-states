# -----------------------------------------------------------------------------
# ------------ UnNormalised de Aguiar Semi-Decoupled Basis method -------------
# -----------------------------------------------------------------------------

import math
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

import time
from class_Semaphor import Semaphor

import csv


from unnormalized_aguiar_solver import bosonic_su_n
import functions

class UNDASDB(bosonic_su_n):

    def __init__(self, ID):

        super().__init__(ID)

        self.decoupling_tolerance = [] #[b_i] = user-set tolerance for partitioning the b_i-th basis


    # Firstly: We shall use dense output

    def simulate_uncoupled_basis(self, max_t = -1, N_dtp = -1, rtol = 1e-3, reg_timescale = -1, N_semaphor = 100):
        # rtol/reg_timescale may be a tuple, in which case the first element is the basis timescale and the second element the wavef timescale
        if isinstance(rtol, tuple):
            rtol_basis, rtol_wavef = rtol
        else:
            rtol_basis = rtol
            rtol_wavef = rtol

        if isinstance(reg_timescale, tuple):
            reg_timescale_basis, reg_timescale_wavef = reg_timescale
        else:
            reg_timescale_basis = reg_timescale
            reg_timescale_wavef = reg_timescale

        print("Simulating state evolution by propagating uncoupled basis vectors one by one and then evolving the decomposition.")

        if not self.is_phys_init:
            print("ERROR: You have attempted to simulate state evolution before specifying the system hamiltonian. You can do this by calling set_hamiltonian_tensors(A, B).")
            return(-1)

        if not self.is_basis_init:
            print("ERROR: You have attempted to simulate state evolution before sampling a basis set. You can do this by calling one of the many basis-sampling methods listed below.")
            for m, desc in bosonic_su_n.basis_sampling_methods.items():
                print(f"{m} [{desc}]")
            return(-1)
        if not self.is_wavef_init:
            print("ERROR: You have attempted to simulate state evolution before specifying the initial wavefunction state. You can do this by calling set_initial_wavefunction(initial_wavefunction, message).")
            return(-1)

        # We propagate each basis vector one by one TODO

        # Set up semaphor for basis propagation
        # TODO

        if not self.is_t_space_init:
            if max_t == -1 or N_dtp == -1:
                print("  ERROR: If t_space has not been previously initialized, max_t and N_dtp are required arguments!")
                return(-1)
            simulation_start_time = 0.0 # TODO is data loaded, this should be the last timestamp
            self.t_space = np.linspace(simulation_start_time, max_t, N_dtp+1)
            self.is_t_space_init = True


        self.basis_evol = []
        self.wavef_evol = []

        self.evol_benchmarks = []

        # From this point on, the new method shines! We shall use a customisable RK45 iterator with exit conditions etc
        for b_i in range(len(self.basis)):

            print(f"# Analyzing basis no. {b_i + 1}")
            cur_start_time = time.time()

            cur_basis_evol = [np.zeros((self.N[b_i], self.M-1), dtype=complex)]
            for i in range(self.N[b_i]):
                for j in range(self.M-1):
                    cur_basis_evol[0][i][j] = self.basis[b_i][i][j]
            for t_i in range(1, N_dtp+1):
                cur_basis_evol.append(np.zeros((self.N[b_i], self.M-1), dtype=complex))

            print(f"  Uncoupled basis propagation on a timescale of t = ({self.t_space[0]} - {self.t_space[-1]}), rtol = {rtol_basis} at {time.strftime("%H:%M:%S", time.localtime(time.time()))}")

            basis_solutions = []
            for n in range(self.N[b_i]):
                A_0 = 1 / np.sqrt(functions.square_norm(self.basis[b_i][n], self.S))
                y_0 = np.array([A_0] + list(self.basis[b_i][n].copy()))
                #basis_solvers.append(sp.integrate.RK45(self.variational_y_dot, t0 = self.t_space[0], y0 = y_0, t_bound = self.t_space[-1]))
                basis_solutions.append(functions.solve_ivp_with_condition(self.variational_y_dot, (self.t_space[0], self.t_space[-1]), y0=y_0, t_eval=self.t_space, args=(reg_timescale_basis,), exit_condition=None, rtol = rtol_basis, max_step = 0.01))

            # NOTE: When dynamically re-partitioning the basis set, we would always only do the solution between two t_eval (or t_partition) vals, and then re-initialised the solvers
            for n in range(self.N[b_i]):
                for t_i in range(1, N_dtp+1):
                    for m in range(self.M - 1):
                        cur_basis_evol[t_i][n][m] = basis_solutions[n].y[t_i][m + 1]

            self.basis_evol.append(cur_basis_evol)

            x_data = []
            y_data = []
            y_dot_data = []
            E_data = []

            def basis_dependent_wavef_y_dot(t, y, reg_timescale, semaphor_event_ID = None):

                # y[a] = A_a(t)

                x_data.append(t)
                #y_data.append([])
                #y_dot_data.append([])

                cur_basis = np.zeros((self.N[b_i], self.M - 1), dtype=complex)
                cur_basis_dot = np.zeros((self.N[b_i], self.M - 1), dtype=complex)
                cur_N = np.zeros(self.N[b_i], dtype=complex)
                cur_N_dot = np.zeros(self.N[b_i], dtype=complex)
                for n in range(self.N[b_i]):
                    cur_weighted_basis_state = basis_solutions[n].sol(t)
                    cur_weighted_basis_state_dot = self.variational_y_dot(t, cur_weighted_basis_state)
                    cur_basis[n] = cur_weighted_basis_state[1:]
                    cur_basis_dot[n] = cur_weighted_basis_state_dot[1:]
                    cur_N[n] = cur_weighted_basis_state[0]
                    cur_N_dot[n] = cur_weighted_basis_state_dot[0]

                    #y_data[-1].append(np.sqrt(functions.square_mag(cur_weighted_basis_state[0])) * np.sqrt(functions.square_norm(cur_basis[n], self.S).real))
                    #y_dot_data[-1].append(np.sqrt(functions.square_mag(cur_weighted_basis_state_dot[0])) * np.sqrt(functions.square_norm(cur_basis_dot[n], self.S).real))
                y_data.append([basis_solutions[0].sol(t)[0].real, basis_solutions[0].sol(t)[1].real, basis_solutions[0].sol(t)[1].imag])
                y_dot_data.append([self.variational_y_dot(t, basis_solutions[0].sol(t))[0].real, self.variational_y_dot(t, basis_solutions[0].sol(t))[1].real, self.variational_y_dot(t, basis_solutions[0].sol(t))[1].imag])

                """ # OLD
                # Overlaps
                X = np.zeros((self.N[b_i], self.N[b_i], 3), dtype=complex) #[i][j][r] = { z^(r) | z^(r) }
                for i in range(self.N[b_i]):
                    for j in range(self.N[b_i]):
                        base_product = 1 + np.sum(np.conjugate(cur_basis[i]) * cur_basis[j])
                        X[i][j][2] = np.power(base_product, self.S - 2)
                        X[i][j][1] = X[i][j][2] * base_product
                        X[i][j][0] = X[i][j][1] * base_product
                Y = np.zeros((self.N[b_i], self.N[b_i]), dtype=complex)
                for i in range(self.N[b_i]):
                    for j in range(self.N[b_i]):
                        Y[i][j] = np.sum(np.conjugate(cur_basis[i]) * cur_basis_dot[j])

                # Regularisation
                if reg_timescale != -1:
                    X_inv = np.linalg.inv(X[:,:,0] + reg_timescale * np.identity(self.N[b_i], dtype=complex)) # inverse of the non-reduced overlap matrix
                else:
                    X_inv = np.linalg.inv(X[:,:,0]) # inverse of the non-reduced overlap matrix

                def ebe(i, k): # extended basis element
                    if k != self.M - 1:
                        return(cur_basis[i][k])
                    else:
                        return(1.0)

                # Hamiltonian tensors and differential vector
                cur_H_A, cur_H_B = self.calculate_hamiltonian_tensors(t)
                H_matrix = np.zeros((self.N[b_i], self.N[b_i]), dtype=complex)
                for a in range(self.N[b_i]):
                    for b in range(self.N[b_i]):
                        one_body_interaction = 0.0
                        two_body_interaction = 0.0
                        for alpha in range(self.M):
                            for beta in range(self.M):
                                one_body_interaction += cur_H_A[alpha][beta] * np.conjugate(ebe(a, alpha)) * ebe(b, beta)
                                for gamma in range(self.M):
                                    for delta in range(self.M):
                                        two_body_interaction += cur_H_B[alpha][beta][gamma][delta] * np.conjugate(ebe(a, alpha)) * np.conjugate(ebe(a, beta)) * ebe(b, gamma) * ebe(b, delta)
                        H_matrix[a][b] = self.S * X[a][b][1] * one_body_interaction + 0.5 * self.S * (self.S - 1) * X[a][b][2] * two_body_interaction

                E_data.append(0.0)
                for a in range(self.N[b_i]):
                    for b in range(self.N[b_i]):
                        E_data[-1] += np.conjugate(y[a]) * y[b] * H_matrix[a][b]

                # M matrix
                #M = -1j * np.matmul(X_inv, H_matrix) - self.S * np.matmul(X_inv * (Y.T), X[:,:,1])
                M = np.matmul(X_inv, -1j * H_matrix - self.S * X[:,:,1] * Y)"""

                def ebe(i, k): # extended basis element
                    if k != self.M - 1:
                        return(cur_basis[i][k])
                    else:
                        return(1.0)

                cur_H_A, cur_H_B = self.calculate_hamiltonian_tensors(t)

                Xi = np.zeros((3, self.N[b_i], self.N[b_i]), dtype=complex)
                zeta = np.zeros((self.N[b_i], self.N[b_i]), dtype=complex)
                epsilon = np.zeros((self.N[b_i], self.N[b_i]), dtype=complex)
                eta = np.zeros((self.N[b_i], self.N[b_i]), dtype=complex)
                for a in range(self.N[b_i]):
                    for b in range(self.N[b_i]):
                        cur_base_overlap = 1 + np.sum(np.conjugate(cur_basis[a]) * cur_basis[b])
                        cur_xi = np.power(np.conjugate(cur_N[a]) * cur_N[b], 1.0 / self.S) * cur_base_overlap
                        Xi[0][a][b] = np.power(cur_xi, self.S)
                        Xi[1][a][b] = Xi[0][a][b] / cur_base_overlap
                        Xi[2][a][b] = Xi[1][a][b] / cur_base_overlap

                        zeta[a][b] = Xi[0][a][b] * cur_N_dot[b] / cur_N[b]

                        epsilon[a][b] = Xi[1][a][b] * (np.sum(np.conjugate(cur_basis[a]) * cur_basis_dot[b]))

                        one_body_interaction = 0.0
                        two_body_interaction = 0.0
                        for alpha in range(self.M):
                            for beta in range(self.M):
                                one_body_interaction += cur_H_A[alpha][beta] * np.conjugate(ebe(a, alpha)) * ebe(b, beta)
                                for gamma in range(self.M):
                                    for delta in range(self.M):
                                        two_body_interaction += cur_H_B[alpha][beta][gamma][delta] * np.conjugate(ebe(a, alpha)) * np.conjugate(ebe(a, beta)) * ebe(b, gamma) * ebe(b, delta)
                        eta[a][b] = self.S * Xi[1][a][b] * one_body_interaction + 0.5 * self.S * (self.S - 1) * Xi[2][a][b] * two_body_interaction

                E_data.append(0.0)
                for a in range(self.N[b_i]):
                    for b in range(self.N[b_i]):
                        E_data[-1] += np.conjugate(y[a]) * y[b] * eta[a][b]


                # Regularisation
                if reg_timescale != -1:
                    Xi_inv = np.linalg.inv(Xi[0] + reg_timescale * np.identity(self.N[b_i], dtype=complex)) # inverse of the non-reduced overlap matrix
                else:
                    Xi_inv = np.linalg.inv(Xi[0]) # inverse of the non-reduced overlap matrix

                M = np.matmul(Xi_inv, -1j * eta - zeta - self.S * epsilon)


                # Semaphor
                self.semaphor.update(semaphor_event_ID, t)

                return(M.dot(y))

            # ---------------- wavef propagation ------------------
            # We propagate the wavefunction
            print("  Propagating wavefunction decomposition...")
            cur_wavef_evol = [np.zeros(self.N[b_i], dtype=complex)]
            for i in range(self.N[b_i]):
                cur_wavef_evol[0][i] = self.wavef[b_i][i]
            for t_i in range(1, N_dtp+1):
                cur_wavef_evol.append(np.zeros(self.N[b_i], dtype=complex))

            # Set up semaphor event
            msg = f"  Wavefunction propagation over the evolved uncoupled basis on a timescale of t = ({self.t_space[0]} - {self.t_space[-1]}), rtol = {rtol_wavef}"
            new_sem_ID = self.semaphor.create_event(np.linspace(self.t_space[0], self.t_space[-1], N_semaphor + 1), msg)

            y_0 = self.wavef[b_i].copy()
            iterated_solution = sp.integrate.solve_ivp(basis_dependent_wavef_y_dot, [self.t_space[0], self.t_space[-1]], y_0, method = 'RK45', t_eval = self.t_space, args = (reg_timescale_wavef, new_sem_ID), rtol = rtol_wavef)

            plt.title("Interpolated basis values")
            plt.ylim((-1e2, 1e2))
            plt.plot(x_data, y_data, label = "z")
            plt.plot(x_data, y_dot_data, label = "dz/dt")
            plt.plot(x_data, np.gradient(np.array(y_data), np.array(x_data), axis = 0), label = "interpolated dz/dt", linestyle="dashed")
            plt.plot(x_data, E_data, label = "energy")
            plt.legend()
            plt.show()

            # The issue is that after this method concludes, the basis norms are discarded, but they are now a vital part of the solution!
            # Two ways to resolve this:
            #     1. Store an extra set of values for basis norms
            #     2. Absorb N into A
            # Let's firstly do method 2 because it's easier, and if the floating point error persists, we shall try method 1


            for t_i in range(1, N_dtp+1):
                for n in range(self.N[b_i]):
                    cur_wavef_evol[t_i][n] = iterated_solution.y[n][t_i] * basis_solutions[n].sol(self.t_space[t_i])[0]

            self.semaphor.finish_event(new_sem_ID, "    Simulation")

            self.wavef_evol.append(cur_wavef_evol)

            self.evol_benchmarks.append(time.time() - cur_start_time)
            print("    Total (basis & wavefunction) benchmark: " + functions.dtstr(self.evol_benchmarks[b_i]))

        self.is_basis_evol = True
        self.is_wavef_evol = True

