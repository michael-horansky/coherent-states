import numpy as np



def overlap(t_p, p_p, t, p):
    return(np.cos(t_p/2) * np.cos(t/2) + np.sin(t_p/2) * np.sin(t/2) * np.exp(1j * (p - p_p)))

def o(t_p, p_p, t, p):
    return(np.log(overlap(t_p, p_p, t, p)))


def g(t_p, p_p, t, p, delta = 0.01, debug = False):

    g = [[0, 0], [0, 0]]
    i_val = overlap(t_p, p_p, t, p)

    g[0][0] = (o(t_p + delta, p_p, t + delta, p) - o(t_p + delta, p_p, t - delta, p) - o(t_p - delta, p_p, t + delta, p) + o(t_p - delta, p_p, t - delta, p)) / (4 * delta * delta)
    g[0][1] = (o(t_p, p_p + delta, t + delta, p) - o(t_p, p_p + delta, t - delta, p) - o(t_p, p_p - delta, t + delta, p) + o(t_p, p_p - delta, t - delta, p)) / (4 * delta * delta)
    g[1][0] = (o(t_p + delta, p_p, t, p + delta) - o(t_p + delta, p_p, t, p - delta) - o(t_p - delta, p_p, t, p + delta) + o(t_p - delta, p_p, t, p - delta)) / (4 * delta * delta)
    g[1][1] = (o(t_p, p_p + delta, t, p + delta) - o(t_p, p_p + delta, t, p - delta) - o(t_p, p_p - delta, t, p + delta) + o(t_p, p_p - delta, t, p - delta)) / (4 * delta * delta)

    c = np.exp(1j * (p - p_p)) / (4 * i_val * i_val)
    print(f"g_11 = {g[0][0]} (predicted value {c})")
    print(f"g_12 = {g[0][1]} (predicted value {-1j * c * np.sin(t_p)})")
    print(f"g_21 = {g[1][0]} (predicted value {1j * c * np.sin(t)})")
    print(f"g_22 = {g[1][1]} (predicted value {c * np.sin(t_p) * np.sin(t)})")

    return(g)


def sample_angles():
    return([np.random.rand() * np.pi, np.random.rand() * 2.0 * np.pi, np.random.rand() * np.pi, np.random.rand() * 2.0 * np.pi])


my_samp = sample_angles()
my_g = g(my_samp[0],my_samp[1],my_samp[2],my_samp[3], debug = True)
print(my_g)
print("and the determinant is equal to", my_g[0][0] * my_g[1][1] - my_g[1][0] * my_g[0][1])


