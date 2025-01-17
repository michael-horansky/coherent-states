from aguiar_states_unnormalized import *





"""N_space_t_space = np.linspace(0, 0.5, 200)
res_N_space = two_mode_solution(N_space_t_space, 5, 0.1, 1)

plt.plot(N_space_t_space, res_N_space)
plt.show()"""



double_mode = BH("nonzero_z")
#double_mode.load_recent_data()

z_0 = 0.00+ 1j * 1.00
double_mode.set_global_parameters(S = 10, M = 2, J_0 = 1, J_1 = 0.5, omega = 2 * np.pi, U = 0.1, K = 0, j_zero = 0)
double_mode.sample_gridlike(4, np.array([z_0], dtype=complex), 0.35, "aguiar")

double_mode.iterate(max_t = 2, N_dtp = 200, rtol = 2e-3, reg_timescale = 1e-4)
double_mode.save_recent_data()

double_mode.plot_recent_data(graph_list = ["expected_mode_occupancy", "initial_basis_heatmap"])


"""grossmann_1 = BH("grossmann_1_S=10")

grossmann_1_xi_0 = np.array([ -np.sqrt(0.7) + 0*1j, np.sqrt(0.3) + 0*1j ])
grossmann_1.set_global_parameters(S = 10, M = 2, J_0 = 1, J_1 = 0.5, omega = 2 * np.pi, U = 0.1, K = 0, j_zero = 0)
grossmann_1.sample_gridlike(3, grossmann_1_xi_0, 0.25, "grossmann")

grossmann_1.iterate(max_t = 2, N_dtp = 200, rtol = 2e-3, reg_timescale = 1e-4)
grossmann_1.save_recent_data()

grossmann_1.plot_recent_data(graph_list = ["expected_mode_occupancy", "initial_basis_heatmap"])"""


#triple_mode = BH("M=3,S=5")

#triple_mode.load_recent_data()

#z_0 = 0.00+ 1j * 0.00
#triple_mode.set_global_parameters(S = 5, M = 3, J_0 = 1, J_1 = 0.5, omega = 2 * np.pi, U = 0.1, K = 0, j_zero = 0)
#triple_mode.sample_gridlike(3, np.array([z_0, z_0], dtype=complex), 0.5)

#triple_mode.iterate(max_t = 0.000002, N_dtp = 100, rtol = 1e-4)
#triple_mode.iterate(max_t = 0.2, N_dtp = 200, rtol = 2e-3, reg_timescale = 1e-4)
#triple_mode.iterate(max_t = 0.2, dt = 0.000002, N_dtp = 200)
#triple_mode.save_recent_data()

#triple_mode.plot_recent_data(graph_list = ["expected_mode_occupancy"])


