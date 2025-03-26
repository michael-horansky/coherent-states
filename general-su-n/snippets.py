import numpy as np
import scipy as sp
from functools import partial

import matplotlib.pyplot as plt

def f(t, y, x):
    return(np.power(y * (y - x) / (x / 2 - y), 2))



t_space = np.linspace(0, 10, 100)
t_eval = np.linspace(0, 5, 500)

y_0 =np.linspace(-1, 1, 10)
f_partial = partial(f, x = 0.5)



# Prepare the commanders
CUSTOM_MESSAGES = sp.integrate._ivp.ivp.MESSAGES.copy()
CUSTOM_MESSAGES[-2] = "Premature exit condition was satisfied, partial solution saved."
#t_eval = lalala, already done
solver = sp.integrate.RK45(f_partial, t0 = t_space[0], y0 = y_0, t_bound = t_space[-1])

def exit_condition(y):
    if max(y) > 10:
        return True # force exit
    return False

# Prepare the grab-bag
interpolants = [] # Stores the dense output
y_eval = [] # Stores value at t_eval
t_interpolants = [t_space[0]] # Endpoints of all interpolants

# Prepare dynamic trackers
status = None
t_eval_i = 0 # We assume direction > 0

while status is None:
    # Make a step
    message = solver.step()
    # Check if finished/failed
    if solver.status == 'finished':
        status = 0
    elif solver.status == 'failed':
        status = -1
        break

    # Check if solver.y is still well-formed (i.e. not diverged etc)
    if exit_condition(solver.y):
        status = -2
        break

    # Grab the dense output interpolant
    sol = solver.dense_output()
    interpolants.append(sol)
    t_interpolants.append(solver.t)

    # Add the values at t_eval to grab-bag
    t_eval_i_new = np.searchsorted(t_eval, solver.t, side='right')
    t_eval_step = t_eval[t_eval_i:t_eval_i_new]
    if t_eval_step.size > 0:
        for t_eval_val in t_eval_step:
            y_eval.append(sol(t_eval_val))
        t_eval_i = t_eval_i_new

message = CUSTOM_MESSAGES.get(status, message)

sol = sp.integrate._ivp.common.OdeSolution(t_interpolants, interpolants)
ode_result = sp.integrate._ivp.ivp.OdeResult(t = t_eval[:t_eval_i], y = y_eval, sol = sol, nfev=solver.nfev, njev=solver.njev, nlu=solver.nlu, status=status, message=message, success=status >= 0)

print(ode_result)
print(ode_result.sol(5))

plt.plot(ode_result.t, ode_result.y)
for i in range(len(y_0)):
    plt.plot(t_eval, ode_result.sol(t_eval)[i], "--")
plt.xlim((0, 1.5))
plt.ylim((-2, 10))
plt.show()



"""
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
    t_eval_step = t_eval[t_eval_i:t_eval_i_new]"""




"""t_interpolated = []
y_interpolated = []
resul_interpolator = sp.interpolate.interp1d(x = t_dense, y = y_dense, axis = 0, assume_sorted = True, bounds_error = False, fill_value = 0.0)
for t in np.linspace(0, 20, 100):
    print(t, resul_interpolator(t))
    t_interpolated.append(t)
    y_interpolated.append(resul_interpolator(t))


plt.plot(t_interpolated, y_interpolated)
plt.show()"""
