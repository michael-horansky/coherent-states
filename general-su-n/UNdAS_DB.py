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
                basis_solutions.append(functions.solve_ivp_with_condition(self.variational_y_dot, (self.t_space[0], self.t_space[-1]), y0=y_0, t_eval=self.t_space, args=(reg_timescale_basis,), exit_condition=None, rtol = rtol_basis))

            # NOTE: When dynamically re-partitioning the basis set, we would always only do the solution between two t_eval (or t_partition) vals, and then re-initialised the solvers
            for n in range(self.N[b_i]):
                for t_i in range(1, N_dtp+1):
                    for m in range(self.M - 1):
                        cur_basis_evol[t_i][n][m] = basis_solutions[n].y[t_i][m + 1]

            self.basis_evol.append(cur_basis_evol)

            def basis_dependent_wavef_y_dot(t, y, reg_timescale, semaphor_event_ID = None):
                cur_basis = np.zeros((self.N[b_i], self.M - 1), dtype=complex)
                cur_basis_dot = np.zeros((self.N[b_i], self.M - 1), dtype=complex)
                for n in range(self.N[b_i]):
                    cur_weighted_basis_state = basis_solutions[n].sol(t)
                    cur_basis[n] = cur_weighted_basis_state[1:]
                    cur_basis_dot[n] = self.variational_y_dot(t, cur_weighted_basis_state)[1:]

                overlap_matrix = np.zeros((self.N[b_i], self.N[b_i], 3), dtype=complex) #[i][j][r] = { z^(r) | z^(r) }
                for i in range(self.N[b_i]):
                    for j in range(self.N[b_i]):
                        base_product = 1 + np.sum(np.conjugate(cur_basis[i]) * cur_basis[j])
                        overlap_matrix[i][j][2] = np.power(base_product, self.S - 2)
                        overlap_matrix[i][j][1] = overlap_matrix[i][j][2] * base_product
                        overlap_matrix[i][j][0] = overlap_matrix[i][j][1] * base_product

                def ebe(i, k): # extended basis element
                    if k != self.M - 1:
                        return(cur_basis[i][k])
                    else:
                        return(1.0)

                # We now initialize M_ij
                M = np.zeros((self.N[b_i], self.N[b_i]), dtype=complex)
                for i in range(self.N[b_i]):
                    for j in range(self.N[b_i]):
                        M[i][j] = 1j * overlap_matrix[i][j][0]

                # Now we initialize R_i
                R = np.zeros(self.N[b_i], dtype=complex)
                for i in range(self.N[b_i]):
                    sum1 = 0.0
                    for j in range(self.N[b_i]):
                        sum1 += y[j] * overlap_matrix[i][j][1] * np.sum(np.conjugate(cur_basis[i]) * cur_basis_dot[j])
                    sum1 *= -1j * self.S

                    sum2 = 0.0
                    for j in range(self.N[b_i]):
                        # We calculate the matrix element
                        first_order_element_sum = 0.0
                        for a in range(self.M):
                            for b in range(self.M):
                                first_order_element_sum += self.H_A(t, a, b) * np.conjugate(ebe(i, a)) * ebe(j, b)
                        first_order_element_sum *= self.S * overlap_matrix[i][j][1]

                        second_order_element_sum = 0.0
                        for a in range(self.M):
                            for b in range(self.M):
                                for c in range(self.M):
                                    for d in range(self.M):
                                        second_order_element_sum += self.H_B(t,a,b,c,d) * np.conjugate(ebe(i, a)) * np.conjugate(ebe(i, b)) * ebe(j, c) * ebe(j, d)
                        second_order_element_sum *= 0.5 * self.S * (self.S - 1) * overlap_matrix[i][j][2]

                        sum2 += y[j] * (first_order_element_sum + second_order_element_sum)
                    R[i] = sum1 + sum2
                # y_dot = M^(-1) . R
                # Regularisation
                if reg_timescale != -1:
                    for i in range(self.N[b_i]):
                        M[i][i] += reg_timescale
                M_inv = np.linalg.inv(M)

                # Semaphor
                self.semaphor.update(semaphor_event_ID, t)

                return(M_inv.dot(R))

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

            for t_i in range(1, N_dtp+1):
                for n in range(self.N[b_i]):
                    cur_wavef_evol[t_i][n] = iterated_solution.y[n][t_i]

            self.semaphor.finish_event(new_sem_ID, "    Simulation")

            self.wavef_evol.append(cur_wavef_evol)

            self.evol_benchmarks.append(time.time() - cur_start_time)
            print("    Total (basis & wavefunction) benchmark: " + functions.dtstr(self.evol_benchmarks[b_i]))

        self.is_basis_evol = True
        self.is_wavef_evol = True

