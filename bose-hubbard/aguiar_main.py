from aguiar_states_unnormalized import *


double_mode = BH("pure_initially_tiny_dt")

#double_mode.load_recent_data()

z_0 = 0.0+ 1j * 0.0

double_mode.set_global_parameters(S = 5, M = 2, J_0 = 1, J_1 = 0.5, omega = 2 * np.pi, U = 0.1, K = 0, j_zero = 0)
double_mode.sample_gridlike(4, np.array([z_0], dtype=complex), 0.35)
N = len(double_mode.basis)

double_mode.iterate(max_t = 0.2, dt = 0.000002, N_dtp = 200)
double_mode.save_recent_data()

double_mode.plot_recent_data(graph_list = ["expected_mode_occupancy", "initial_basis_heatmap"])


