from unnormalized_aguiar_solver import *

from functools import partial


def A_Frank(t, a, b, J_0, J_1, omega, K, j_zero):
    if a == b:
        return(K * (a - j_zero) * (a - j_zero) / 2)
    if a == b + 1 or a == b - 1:
        return(J_0 + J_1 * np.cos(omega * t))
    return(0.0)

def B_Frank(t, a, b, c, d, U):
    if a == b and a == c and a == d:
        return(U)
    return(0.0)


Frank_params = {
        "J_0" : 1,
        "J_1" : 0.5,
        "omega" : 2 * np.pi,
        "U" : 0.1,
        "K" : 0,
        "j_zero" : 0
    }

A_Frank_wrapper = partial(A_Frank, J_0 = Frank_params["J_0"], J_1 = Frank_params["J_1"], omega = Frank_params["omega"], K = Frank_params["K"], j_zero = Frank_params["j_zero"])
B_Frank_wrapper = partial(B_Frank, U = Frank_params["U"])

def A_BH(t, a, b):
    BH_params = {
        "J_0" : 1,
        "J_1" : 0.5,
        "omega" : 2 * np.pi,
        "K" : 0,
        "j_zero" : 0
    }
    if a == b:
        return(BH_params["K"] * (a - BH_params["j_zero"]) * (a - BH_params["j_zero"]) / 2)
    if a == b + 1 or a == b - 1:
        return(BH_params["J_0"] + BH_params["J_1"] * np.cos(BH_params["omega"] * t))
    return(0.0)

def B_BH(t, a, b, c, d):
    BH_params_B = {
        "U" : 0.1
    }
    if a == b and a == c and a == d:
        return(BH_params_B["U"])
    return(0.0)





#AF_sig = inspect.signature(A_Frank_wrapper)
#print(AF_sig.parameters)
"""
bose_hubbard = bosonic_su_n("bose_hubbard_M=2_max_t=2")
#bose_hubbard.load_data()

bose_hubbard.set_global_parameters(M = 2, S = 15)
bose_hubbard.set_hamiltonian_tensors(A_BH, B_BH)

# Note: to get more basis vectors in a sample, increase particle number! It makes saturation less probable :)
bose_hubbard.sample_gaussian(z_0 = np.array([0.00+ 1j * 0.50], dtype=complex), width = 1.0, conditioning_limit = 10e10, N_max = 30, max_saturation_steps = 500)
bose_hubbard.set_initial_wavefunction()

bose_hubbard.simulate_uncoupled_basis(max_t = 2.0, N_dtp = 200, rtol = (2e-3, 2e-3), reg_timescale = (-1, 1e-6))
bose_hubbard.save_data()

bose_hubbard.plot_data()"""


"""bose_hubbard = bosonic_su_n("bose_hubbard_M=2_S=10_mixed_init")
bose_hubbard.load_data()

bose_hubbard.set_global_parameters(M = 2, S = 10)
bose_hubbard.set_hamiltonian_tensors(A_BH, B_BH)

z_0 = np.array([1.00+ 1j * 0.00], dtype=complex)
z_1 = np.array([0.00+ 1j * 0.00], dtype=complex)
#bose_hubbard.set_initial_wavefunction(z_0, "aguiar_pure")
bose_hubbard.set_initial_wavefunction([[z_0, z_1], [1, 1]], "aguiar_disc")
# Note: to get more basis vectors in a sample, increase particle number! It makes saturation less probable :)
#bose_hubbard.sample_gaussian(z_0 = z_0, width = 1.2, conditioning_limit = 10e10, N_max = 30, max_saturation_steps = 5000)
#bose_hubbard.sample_gaussian(width = 1.0, conditioning_limit = 10e4, N_max = 25, max_saturation_steps = 5000)
bose_hubbard.sample_gaussian(width = 1.0, conditioning_limit = 10e4, N_max = 10, max_saturation_steps = 5000)
bose_hubbard.sample_gaussian(width = 1.0, conditioning_limit = 10e4, N_max = 1, max_saturation_steps = 5000)

bose_hubbard.simulate_uncoupled_basis(max_t = 2.0, N_dtp = 200, rtol = (1e-10, 1e-10), reg_timescale = (1e-2, 1e-3))

bose_hubbard.fock_solution()
bose_hubbard.save_data()

bose_hubbard.plot_data(graph_list = ["expected_mode_occupancy"])"""





"""bose_hubbard = bosonic_su_n("bose_hubbard_M=3_S=20")
bose_hubbard.load_data(["system"])

#bose_hubbard.set_global_parameters(M = 3, S = 20)
#bose_hubbard.set_hamiltonian_tensors(A_BH, B_BH)

z_0 = np.array([0.50+ 1j * 0.50, 0.50 + -1j * 0.50], dtype=complex)
bose_hubbard.set_initial_wavefunction(z_0, "aguiar_pure")
# Note: to get more basis vectors in a sample, increase particle number! It makes saturation less probable :)
#bose_hubbard.sample_gaussian(z_0 = z_0, width = 1.0, conditioning_limit = 10e11, N_max = 50, max_saturation_steps = 5000)
#bose_hubbard.sample_gaussian(z_0 = z_0, width = 1.0, conditioning_limit = 10e11, N_max = 25, max_saturation_steps = 5000)
bose_hubbard.sample_gaussian(z_0 = z_0, width = 1.0, conditioning_limit = 10e11, N_max = 10, max_saturation_steps = 5000)
bose_hubbard.sample_gaussian(z_0 = z_0, width = 1.0, conditioning_limit = 10e11, N_max = 1, max_saturation_steps = 5000)


bose_hubbard.simulate_uncoupled_basis(max_t = 2.0, N_dtp = 200, rtol = (1e-10, 1e-10), reg_timescale = (1e-4, 1e-4))
#bose_hubbard.fock_solution(t_range = [2.0, 200])
#bose_hubbard.save_data(["system", "setup"])

bose_hubbard.plot_data( graph_list = ["expected_mode_occupancy"])"""





#bose_hubbard = bosonic_su_n("bose_hubbard_M=3_S=20")
bose_hubbard = bosonic_su_n("bose_hubbard_M=2_S=10_mixed_init")
bose_hubbard.load_data()

bose_hubbard.plot_data(graph_list = ["expected_mode_occupancy"])
