import numpy as np


class NumericalEvolution():
    # A class encompassing the properties of a function recorded as a set of datapoints
    def __init__(self, t, values, errors = False):
        if len(t) != len(values):
            # invalid
            return(False)
        if type(errors) != bool:
            if len(t) != len(errors):
                # invalid
                return(False)
        self.t = np.array(t)
        self.values = np.array(values)
        if type(errors) != bool:
            self.errors = np.abs(np.array(errors))
        else:
            self.errors = False

    def __str__(self):
        str_result = f"A representation of a function's time evolution on {len(self.errors)} datapoints.\n"
        str_result += f"    t      = [{self.t[0]     }, {self.t[1]     } ... {self.t[-1]}     ]\n"
        str_result += f"    values = [{self.values[0]}, {self.values[1]} ... {self.values[-1]}]\n"
        if type(self.errors) == bool: #no errors provided
            str_result += f"    errors = [not provided]\n"
        else:
            str_result += f"    errors = [{self.errors[0]}, {self.errors[1]} ... {self.errors[-1]}]\n"
        return(str_result)


def explicit_runge_kutta_delta_y(derivative_fun, t, dt, y_n, butcher_tableau):
    # Takes y(t) = y_n and calculates y(t+dt) = y_(n+1)

    S = len(butcher_tableau[1])
    y_dim = len(y_n)
    k = np.zeros((S, y_dim), dtype=complex) # k is a list of complex vectors!


    # We calculate the coefficients
    k[0] = derivative_fun(t, y_n)
    for i in range(1, S):
        delta_y = np.zeros(y_dim, dtype=complex)
        for j in range(i):
            delta_y += butcher_tableau[0][i-1][j] * k[j]
        k[i] = derivative_fun(t + butcher_tableau[2][i-1] * dt, y_n + delta_y * dt)

    # We calculate the result
    if len(butcher_tableau) == 3:
        return([y_n + butcher_tableau[1].dot(k) * dt, False]) # no error
    return([y_n + butcher_tableau[1].dot(k) * dt, (butcher_tableau[1] - butcher_tableau[3]).dot(k) * dt])


def explicit_runge_kutta(derivative_fun, t_param, y_0, butcher_tableau, N_dtp = 100):
    # butcher tableau is a list [a, b, c(, d)], where
    #   a is the Runge-Kutta table: an (S-1,S-1) ndarray where all elements above the diagonal are ignored (e.g. treated as zero)
    #   b is the list of weights: an (S) ndarray = [b_1, b_2 ... b_S]
    #   c is the list of nodes: an (S-1) ndarray = [c_2, c_3 ... c_S]
    #   d is the list of error-evaluating weights: an (S) ndarray = [d_1, d_2 ... d_S]

    full_t_space = np.arange(t_param[0], t_param[1], t_param[2])
    step_N = len(full_t_space)

    # If the user asks for more datapoints than there are steps, we will limit the datapoint number
    if N_dtp > step_N:
        N_dtp = step_N

    if len(butcher_tableau) == 3:
        y_evol = [[], []] # t, values
    else:
        y_evol = [[], [], []] # t, values, errors
    y_iterated = y_0.copy()
    y_evol[0].append(t_param[0])
    y_evol[1].append(y_iterated)
    if len(butcher_tableau) == 4:
        y_evol[2].append(np.zeros(len(y_iterated)))

    for t_i in range(1, step_N):
        previous_time = t_param[0] + (t_i-1) * t_param[2]
        new_y = explicit_runge_kutta_delta_y(derivative_fun, previous_time, t_param[2], y_iterated, butcher_tableau)
        y_iterated = new_y[0] # we rewrite the previous cycle's data
        # Check if state should be recorded and progress updated
        if np.floor(t_i / (step_N-1) * (N_dtp-1)) > (len(y_evol[0]) - 1):
            y_evol[0].append(t_param[0] + t_i * t_param[2])
            y_evol[1].append(new_y[0])
            if len(butcher_tableau) == 4:
                y_evol[2].append(new_y[1])
        """if np.floor(t_i / (step_N-1) * 100) > progress:
            progress = int(np.floor(t_i / (step_N-1) * 100))
            ETA = time.strftime("%H:%M:%S", time.localtime( (time.time()-start_time) * 100 / progress + start_time ))
        print("  " + str(progress).zfill(2) + "% done (" + str(t_i) + "/" + str(step_N-1) + "); est. time of finish: " + ETA, end='\r')"""
    if len(butcher_tableau) == 3:
        return(NumericalEvolution(y_evol[0], y_evol[1]))
    else:
        return(NumericalEvolution(y_evol[0], y_evol[1], y_evol[2]))


# Standard Butcher tableaux

dormand_prince_tableau_a = np.array([[1/5       , 0          , 0         , 0       , 0          , 0    ],
                                     [3/40      , 9/40       , 0         , 0       , 0          , 0    ],
                                     [44/45     , -56/15     , 32/9      , 0       , 0          , 0    ],
                                     [19372/6561, -25360/2187, 64448/6561, -212/729, 0          , 0    ],
                                     [9017/3168 , -355/33    , 46732/5247, 49/176  , -5103/18656, 0    ],
                                     [35/384    , 0          , 500/1113  , 125/192 , -2187/6784 , 11/84]])
dormand_prince_tableau_b = np.array( [5179/57600, 0          , 7571/16695, 393/640 , -92097/339200, 187/2100, 1/40])
dormand_prince_tableau_c = np.array( [1/5       , 3/10       , 4/5       , 8/9     , 1          , 1    ])
dormand_prince_tableau_d = np.array( [35/384    , 0          , 500/1113  , 125/192 , -2187/6784 , 11/84, 0])

dormand_prince_tableau = [dormand_prince_tableau_a, dormand_prince_tableau_b, dormand_prince_tableau_c, dormand_prince_tableau_d]


rk4_a = np.array([[1/2, 0  , 0],
                  [0  , 1/2, 0],
                  [0  ,   0, 1]])
rk4_b = np.array([1/6, 1/3, 1/3, 1/6])
rk4_c = np.array([1/2, 1/2, 1])

rk4_tableau = [rk4_a, rk4_b, rk4_c]




