
def flatten_basis(M, S):
    # Creates a list of all fermionic states with M modes and S particles,
    # where each element is a distinct ascending list of S non-equal integers, each between 0 and M-1 (inclusive)

    if S>M:
        print("Cannot fit", S, "fermions into", M, "states, error.")
        return([])

    # We fix the first mode occupancy and then find the answer for M - 1 occupancies
    if S == 0:
        return([[]]) # the only solution is the empty occupancy solution
    answer = []
    for last_particle_index in range(S-1, M):
        remainder = flatten_basis(last_particle_index, S - 1)
        for row in remainder:
            answer.append(row + [last_particle_index])
    return(answer)

class dmitry_states_solver(ground_state_solver):

    def __init__(self, ID):
        super().__init__(ID)


    def H_element(self, occupancy_left, occupancy_right):
        # both occupancy_left/right are elements of fock_basis (i.e. lists of occupied mode indices)
        pass

    def fock_solution(self, t_space, N_semaphor = 100):
        # General solution to be plotted

        print("Solving the Hamiltonian on the Fock basis...")


        # We use the time-dependent Schrodinger Equation over the full occupancy number basis
        # d/dt | Psi > = -i H | Psi >
        # | Psi > = c_i | u_i >, where | u_i > is the i-th element of the full occpuancy number basis
        # Hence dc_j/dt = -i c_i < u_j | H | u_i > = -i c_i H_ji
        # In vector form: dc/dt = -i H . c

        print("  Finding the basis...")
        fock_basis = flatten_basis(self.M, self.S)
        fock_N = len(fock_basis)

        def c_dot(t, y, semaphor_event_ID = None):
            # We find the H_ij matrix
            H_matrix = np.zeros((fock_N, fock_N), dtype=complex)
            for i in range(fock_N):
                for j in range(fock_N):
                    # Calculating H_ij

                    # First order
                    for a in range(self.M):
                        for b in range(self.M):
                            bra_occupancy = fock_basis[i].copy()
                            ket_occupancy = fock_basis[j].copy()
                            coef = 1.0

                            # a^h.c._a acts on the bra
                            if bra_occupancy[a] == 0:
                                continue
                            coef *= np.sqrt(bra_occupancy[a])
                            bra_occupancy[a] -= 1
                            # a_b acts on the ket
                            if ket_occupancy[b] == 0:
                                continue
                            coef *= np.sqrt(ket_occupancy[b])
                            ket_occupancy[b] -= 1

                            if bra_occupancy == ket_occupancy:
                                H_matrix[i][j] += self.H_A(t, a, b) * coef
                    # Second order
                    for a in range(self.M):
                        for b in range(self.M):
                            for c in range(self.M):
                                for d in range(self.M):
                                    bra_occupancy = fock_basis[i].copy()
                                    ket_occupancy = fock_basis[j].copy()
                                    coef = 1.0

                                    # a^h.c._a acts on the bra
                                    if bra_occupancy[a] == 0:
                                        continue
                                    coef *= np.sqrt(bra_occupancy[a])
                                    bra_occupancy[a] -= 1
                                    # a^h.c._b acts on the bra
                                    if bra_occupancy[b] == 0:
                                        continue
                                    coef *= np.sqrt(bra_occupancy[b])
                                    bra_occupancy[b] -= 1
                                    # a_c acts on the ket
                                    if ket_occupancy[c] == 0:
                                        continue
                                    coef *= np.sqrt(ket_occupancy[c])
                                    ket_occupancy[c] -= 1
                                    # a_d acts on the ket
                                    if ket_occupancy[d] == 0:
                                        continue
                                    coef *= np.sqrt(ket_occupancy[d])
                                    ket_occupancy[d] -= 1

                                    if bra_occupancy == ket_occupancy:
                                        H_matrix[i][j] += 0.5 * self.H_B(t, a, b, c, d) * coef
            self.semaphor.update(semaphor_event_ID, t)
            return( - 1j * H_matrix.dot(y))


        if not self.is_t_space_init:
            if len(t_range) == 2:
                self.t_space = np.linspace(0, t_range[0], t_range[1] + 1)
            elif len(t_range) == 3:
                self.t_space = np.linspace(t_range[0], t_range[1], t_range[2] + 1)
            else:
                print("  ERROR: t_space is a required argument when t_space has not been initialized, and must be a list of form [(t_start,) t_stop, N_dtps]")
                return(-1)
            self.is_t_space_init = True

        N_dtp = len(self.t_space)

        # Decomposition of initial state from self.wavef_initial_wavefunction, self.wavef_message
        # c_i(t = 0) = < u_i | z_0 }

        """if self.wavef_message in ["aguiar"]:
            # initial_wavefunction describes an Aguiar unnormalized coherent state.
            c_0 = self.decompose_aguiar_into_fock(np.array(self.wavef_initial_wavefunction), fock_basis)
        elif self.wavef_message in ["grossmann", "frank"]:
            # initial_wavefunction describes a Grossmann normalized coherent state.
            c_0 = self.decompose_grossmann_into_fock(np.array(self.wavef_initial_wavefunction), fock_basis)
        elif self.wavef_message in ["", "NONE"]:
            c_0 = self.decompose_aguiar_into_fock(self.basis[0][0], fock_basis)
        else:
            print(f"  ERROR: self.wavef_message {self.wavef_message} not recognized.")
            return(-1)"""
        c_0 = self.decompose_init_wavef_into_fock(fock_basis)

        msg = f"  Solving the Schrodinger equation discretised on the full Fock basis on a timescale of t = ({self.t_space[0]} - {self.t_space[-1]})..."
        new_sem_ID = self.semaphor.create_event(np.linspace(self.t_space[0], self.t_space[-1], N_semaphor + 1), msg)

        sol = sp.integrate.solve_ivp(c_dot, [self.t_space[0], self.t_space[-1]], c_0, method = 'RK45', t_eval = self.t_space, args = (new_sem_ID,))

        self.solution_benchmark = self.semaphor.finish_event(new_sem_ID, "    Simulation")

        print("  Translating Fock basis coefficients into mode occupancies...")

        self.solution = []
        for t_i in range(len(self.t_space)):
            cur_c = np.zeros(fock_N, dtype=complex)
            for i in range(fock_N):
                cur_c[i] = sol.y[i][t_i]

            self.solution.append(np.zeros(self.M))
            for m in range(self.M):
                for i in range(fock_N):
                    self.solution[t_i][m] += fock_basis[i][m] * (cur_c[i].real * cur_c[i].real + cur_c[i].imag * cur_c[i].imag) / self.S

        print(f"  Done! Solution on the full occupancy basis found in {functions.dtstr(self.solution_benchmark)}")


        self.is_solved = True




