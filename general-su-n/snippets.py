import numpy as np
import scipy as sp
from functools import partial

import matplotlib.pyplot as plt

def f(t, y, x):
    return(y * (y - x) * (x / 2 - y))



t_space = np.linspace(0, 10, 100)
t_eval = np.linspace(0, 5, 5)

y_0 =np.linspace(-1, 1, 10)
f_partial = partial(f, x = 0.5)

solver = sp.integrate.RK45(f_partial, t0 = t_space[0], y0 = y_0, t_bound = t_space[-1])


t_res = np.zeros(len(t_eval))
y_res = np.zeros(len(t_eval))

y_res[0] = y_0
#t_dense = [t_space[0]]
#y_dense = [y_0]
t_eval_i = 0
while(solver.status == "running"):
    solver.step()
    #t_dense.append(solver.t)
    #y_dense.append(solver.y)

    t_eval_i_new = np.searchsorted(t_eval, solver.t, side='right')
    t_eval_step = t_eval[t_eval_i:t_eval_i_new]




"""t_interpolated = []
y_interpolated = []
resul_interpolator = sp.interpolate.interp1d(x = t_dense, y = y_dense, axis = 0, assume_sorted = True, bounds_error = False, fill_value = 0.0)
for t in np.linspace(0, 20, 100):
    print(t, resul_interpolator(t))
    t_interpolated.append(t)
    y_interpolated.append(resul_interpolator(t))


plt.plot(t_interpolated, y_interpolated)
plt.show()"""
