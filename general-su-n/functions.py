import numpy as np
import scipy as sp

# -------------------------------- Integrators --------------------------------

# Prepare the commanders
CUSTOM_MESSAGES = sp.integrate._ivp.ivp.MESSAGES.copy()
CUSTOM_MESSAGES[-2] = "Premature exit condition was satisfied, partial solution saved."
#t_eval = lalala, already done

def solve_ivp_with_condition(fun, t_span, y0, t_eval, args = None, exit_condition = None, **options):
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

    solver = sp.integrate.RK45(partial_fun, t0 = t_start, y0 = y0, t_bound = t_end, **options)

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



def square_mag(c):
    return(c.real * c.real + c.imag * c.imag)

# -------------------- Coherent state functions --------------------
# We _deliberately_ do not implement instances of coherent states as
# class instances, as the dynamical manipulation is fast only when
# working with ndarrays. Hence these functions provide physical
# methods for overlap integrals etc.

def square_norm(z, S):
    # Returns a real number = { z | z }
    return(np.power(1 + np.sum(z.real * z.real + z.imag * z.imag), S))

def overlap(z1, z2, S, r = 0):
    # Returns a complex number = { z1^(r) | z2^(r) }, where r is the reduction.
    return(np.power(1 + np.sum(np.conjugate(z1) * z2), S - r))







def dtstr(seconds, max_depth = 2):
    # Dynamically chooses the right format
    # max_depth is the number of different measurements (e.g. max_depth = 2: "2 days 5 hours")
    if seconds >= 60 * 60 * 24:
        # Days
        if max_depth == 1:
            return(f"{int(np.round(seconds / (60 * 60 * 24)))} days")
        remainder = seconds % (60 * 60 * 24)
        days = int((seconds - remainder) / (60 * 60 * 24))
        return(f"{days} days {dtstr(remainder, max_depth - 1)}")
    if seconds >= 60 * 60:
        # Hours
        if max_depth == 1:
            return(f"{int(np.round(seconds / (60 * 60)))} hours")
        remainder = seconds % (60 * 60)
        hours = int((seconds - remainder) / (60 * 60))
        return(f"{hours} hours {dtstr(remainder, max_depth - 1)}")
    if seconds >= 60:
        # Minutes
        if max_depth == 1:
            return(f"{int(np.round(seconds / 60))} min")
        remainder = seconds % (60)
        minutes = int((seconds - remainder) / (60))
        return(f"{minutes} min {dtstr(remainder, max_depth - 1)}")
    if seconds >= 1:
        # Seconds
        if max_depth == 1:
            return(f"{int(np.round(seconds))} sec")
        remainder = seconds % (1)
        secs = int((seconds - remainder))
        return(f"{secs} sec {dtstr(remainder, max_depth - 1)}")
    # Milliseconds
    return(f"{int(np.round(seconds / 0.001))} ms")

