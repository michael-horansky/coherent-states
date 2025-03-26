import numpy as np
import scipy as sp
from functools import partial

import functions

import matplotlib.pyplot as plt

def f(t, y, x):
    return(np.power(y * (y - x) / (x / 2 - y), 2))

def exit_condition(y):
    if max(y) > 10:
        return True # force exit
    return False


t_space = np.linspace(0, 10, 100)
t_eval = np.linspace(0, 5, 500)

y_0 =np.linspace(-1, 1, 10)

ode_result = functions.solve_ivp_with_condition(f, (t_space[0], t_space[-1]), y0=y_0, t_eval=t_eval, args=(0.5,), exit_condition=exit_condition)

plt.plot(ode_result.t, ode_result.y)
for i in range(len(y_0)):
    plt.plot(t_eval, ode_result.sol(t_eval)[i], "--")
plt.xlim((0, 1.5))
plt.ylim((-2, 10))
plt.show()


