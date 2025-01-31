from unnormalized_aguiar_uncoupled_basis import *

import math
import numpy as np
import scipy as sp

from functools import partial

import warnings

# Note that epsilon^alpha = omega * (n + 1/2). In the paper, omega = mass = 1 (unity)
def A_DHT(t, a, b):
    DHT_params = {
        "xi" : 2.1,
        "lambda_0" : 0.01,
    }
    if a == b:
        return(3/2 + DHT_params["xi"] * DHT_params["xi"] / 2)
    if a == b + 1:
        return(-DHT_params["xi"] * np.sqrt(a/2))
    if b == a + 1:
        return(-DHT_params["xi"] * np.sqrt(b/2))
    return(0.0)

def B_DHT(t, a, b, c, d):
    DHT_params = {
        "xi" : 2.1,
        "lambda_0" : 0.01,
    }
    if (a + b + c + d) % 2 == 1:
        return(0.0)

    prefix = 1 / (np.pi * np.sqrt(np.power(2, a+b+c+d) * math.factorial(a) * math.factorial(b) * math.factorial(c) * math.factorial(d)))

    H_a = sp.special.hermite(a)
    H_b = sp.special.hermite(b)
    H_c = sp.special.hermite(c)
    H_d = sp.special.hermite(d)

    with warnings.catch_warnings(action="ignore"):
        coefs = (H_a * H_b * H_c * H_d).coeffs # descending powers
    even_coefs = coefs[::-2] # now ascending powers

    coef_sum = even_coefs[0] * np.sqrt(np.pi / 2)
    for tau in range(1, int((a+b+c+d)/2)+1):
        coef_sum += even_coefs[tau] * np.sqrt(np.pi / 2) * (1 / np.power(4, tau)) * sp.special.factorial2(2 * tau - 1)

    result = 0.5 * DHT_params["lambda_0"] * prefix * coef_sum

    if a == b and a == c and a == d:
        result += 1
    return(result)




DHT = bosonic_su_n("DHT_M=3")
#DHT.load_data()

DHT.set_global_parameters(M = 3, S = 10)
DHT.set_hamiltonian_tensors(A_DHT, B_DHT)

z_0 = np.array([0.50+ 1j * 0.50, 0.50 + -1j * 0.50], dtype=complex)
# Note: to get more basis vectors in a sample, increase particle number! It makes saturation less probable :)
DHT.sample_gaussian(z_0 = z_0, width = 1.0, conditioning_limit = 10e8, N_max = 20, max_saturation_steps = 5000)
DHT.sample_gaussian(z_0 = z_0, width = 1.0, conditioning_limit = 10e8, N_max = 5, max_saturation_steps = 5000)
DHT.sample_gaussian(z_0 = z_0, width = 1.0, conditioning_limit = 10e8, N_max = 1, max_saturation_steps = 5000)
DHT.set_initial_wavefunction()

DHT.simulate_variational(max_t = 2.0, N_dtp = 200, rtol = 1e-3, reg_timescale = 1e-2)
DHT.fock_solution()
DHT.save_data()

DHT.plot_data(graph_list = ["expected_mode_occupancy"])


"""DHT = bosonic_su_n("DHT_M=2")
DHT.load_data()

#DHT.set_global_parameters(M = 2, S = 10)
#DHT.set_hamiltonian_tensors(A_DHT, B_DHT)

#z_0 = np.array([0.10+ 1j * 0.10], dtype=complex)
# Note: to get more basis vectors in a sample, increase particle number! It makes saturation less probable :)
#DHT.sample_gaussian(z_0 = z_0, width = 1.0, conditioning_limit = 10e11, N_max = 5, max_saturation_steps = 5000)
#DHT.sample_gaussian(z_0 = z_0, width = 1.0, conditioning_limit = 10e11, N_max = 5, max_saturation_steps = 5000)
#DHT.sample_gaussian(z_0 = z_0, width = 1.0, conditioning_limit = 10e11, N_max = 1, max_saturation_steps = 5000)
#DHT.set_initial_wavefunction()

#DHT.simulate_variational(max_t = 2.0, N_dtp = 200, rtol = 1e-3, reg_timescale = 1e-4)
#DHT.fock_solution()
#DHT.save_data()

DHT.plot_data(graph_list = ["expected_mode_occupancy"])"""

