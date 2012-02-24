#!/usr/bin/python
from numpy import dot, array, sqrt, arctan, sin, cos, pi, zeros
from copy import deepcopy
from pts.bfgs import LBFGS, BFGS, SR1
from scipy.linalg import eigh
from pts.func import NumDiff
from pts.metric import Default, Metric_reduced, Metric
from pts.dimer_rotate import rotate_dimer, rotate_dimer_mem
from numpy import arccos
from sys import stdout
from pts.memoize import Memoize
from pts.trajectories import empty_traj, empty_log

def qn(pes, start_geo, metric, max_iteration = 100000000, \
       converged = 0.00016, max_step = 0.1, pickle_log = empty_log, \
       update_method = "SR1", trajectory = empty_traj, logfile = None, **params):
    """ A simple quasi Newton method
    pes :  potential surface to calculate on, needs f and fprime function
    start_geo : one geometry on pes, as start for search
    start_mode : first mdoe_vector suggestion
    metric     : might affect everything, as defines distances and angles, from pts.metric module
    """
    hess = eval(update_method)
    hess = hess()

    # do not change them
    geo = deepcopy(start_geo) # of dimer middle point
    error_old = 0
    old_grad = None

    if logfile == None or logfile == "-":
        selflogfile = stdout
    else:
        selflogfile = open(logfile, "a")

    selflogfile.write("Values are in eV, Angstrom, degrees or combinations of them\n")
    selflogfile.write("Iteration:       Energy            ABS. Force          Max. Force        Step\n")
    selflogfile.flush()

    conv = False
    res = {}

    i = 0
    # main loop:
    while i < max_iteration:
         pickle_log("Center", geo)
         energy, grad = pes.taylor(geo)

         # Test for convergence, converged if saddle point is reached
         abs_force = metric.norm_down(grad, geo)
         error = max(abs(grad.flatten()))
         if error < converged:
             # Breakpoint 1: calculation has converged
             conv = True
             selflogfile.write("Calculation is converged with max(abs(force)) %8.5f < %8.5f \n" % \
              (error, converged))
             break

         if not old_grad == None:
             dg = grad - old_grad
             hess.update(step, dg)

         # calculate one step of the qausi_newton, also update dimer direction
         step = - hess.inv(grad)

         step_len = metric.norm_up(step, start_geo)
         if step_len > max_step:
             assert step_len > 0
             step *= max_step / step_len
             step_len = max_step


         # Give some report during dimer optimization
         selflogfile.write("%i:   %12.5f       %12.5f       %12.5f       %9.5f\n" % \
               (i, energy, abs_force, error, step_len))
         selflogfile.flush()

         old_grad = grad
         geo = geo + step
         i += 1
         error_old = error
         trajectory(geo, i, [( grad, "grads", "Gradients")])

    if conv:
        print "Calculation is converged"
    else:
        selflogfile.write("Calculation is not converged\n")
        print "No convergence reached"

    # add some more results to give back
    res["convergence"] = conv
    res["abs_force"] = abs_force
    res["steps"] = i
    res["conv_criteria"] = error

    return geo, res

def main(args):
    from pts.io.read_inp_dimer import read_dimer_input
    from pts.trajectories import empty_traj, traj_every, traj_long, traj_last
    from ase.io import write
    pes, start_geo, __, params, atoms, funcart = read_dimer_input(args[1:], args[0] )
    metric = Default()

    if "trajectory" in params:
        if params["trajectory"] in ["empty", "None", "False"]:
            params["trajectory"] = empty_traj
        elif params["trajectory"] == "every":
            params["trajectory"] = traj_every(atoms, funcart)
        elif params["trajectory"] == "one_file":
            params["trajectory"] = traj_long(atoms, funcart, ["grads"])
        else:
            params["trajectory"] = traj_last(atoms, funcart)
    else:
            params["trajectory"] = traj_last(atoms, funcart)

    geo, res = qn(pes, start_geo, metric, **params)

    print ""
    print ""
    print "Results as given back from Quasi Newton routine"
    print geo
    print res
    print ""
    print "Results Quasi Newton calculation:"
    print "Quasi Newton converged: ", res["convergence"]
    print "Forces left (RMS):", res["abs_force"]
    print "Final geometry:"
    atoms.set_positions(funcart(geo))
    write("-", atoms)

# python dimer.py [-v]:
if __name__ == "__main__":
    import doctest
    doctest.testmod()
