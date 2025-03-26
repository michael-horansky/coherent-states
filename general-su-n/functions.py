import numpy as np
import scipy as sp

# -------------------------------- Integrators --------------------------------

# Prepare the commanders
CUSTOM_MESSAGES = sp.integrate._ivp.ivp.MESSAGES.copy()
CUSTOM_MESSAGES[-2] = "Premature exit condition was satisfied, partial solution saved."
#t_eval = lalala, already done

def solve_ivp_with_condition(fun, t_span, y0, t_eval, args = None, exit_condition = None):
    t_start, t_end = map(float, t_span)

    if args is not None:
        try:
            _ = [*(args)]
        except TypeError as exp:
            suggestion_tuple = (
                "Supplied 'args' cannot be unpacked. Please supply `args`"
                f" as a tuple (e.g. `args=({args},)`)"
            )
            raise TypeError(suggestion_tuple) from exp

        def partial_fun(t, x, fun=fun):
            return fun(t, x, *args)

    solver = sp.integrate.RK45(partial_fun, t0 = t_start, y0 = y0, t_bound = t_end)

    # Prepare the grab-bag
    interpolants = [] # Stores the dense output
    y_eval = [] # Stores value at t_eval
    t_interpolants = [t_start] # Endpoints of all interpolants

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
        if exit_condition is not None:
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
    return(sp.integrate._ivp.ivp.OdeResult(t = t_eval[:t_eval_i], y = y_eval, sol = sol, nfev=solver.nfev, njev=solver.njev, nlu=solver.nlu, status=status, message=message, success=status >= 0))

