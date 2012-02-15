from numpy import sqrt, dot, array, zeros, vstack
from pts.metric import Default
from copy import deepcopy
from pts.steepest_decent import steepest_decent_path, start_values
#from ase.optimize.oldqn import GoodOldQuasiNewton as QN
from scipy.optimize import fmin_l_bfgs_b as minimize
from scipy.optimize import fmin as minimize2
from numpy import savetxt, loadtxt

def write_down_path(path, file, tolerance, name):
    savetxt(file, path)

def read_path(file):
    path = loadtxt(file)

    return path

def chain_for_fun(raw, fun):
    """
    generates the chain corresponding to raw
    which holds the same coordinates but in
    internal coordinates so that raw = fun(result)
    """
    result = zeros(raw.shape)
    for i, ral in enumerate(raw):
        result[i] = fun.pinv(ral)
        assert dot(fun(result[i]) -ral, fun(result[i]) -ral) < 1e-16
    return result

def chain_to_points(chain, fun):
    """
    chain in internal coordinates to chain in
    cartesian coordinates
    """
    result = zeros(chain.shape)
    for i, chi in enumerate(chain):
        result[i] = fun(chi)
    return result

def relax_points(fun, points, tolerance):
    """
    Given a set of points relax every one of them to the nearest
    stationary point

    This is supposed to be an optimizing the stationary points before
    feeding them into find_connections. It is tried to converge the points
    to tolerance * 0.1 as it is supposed that tolerance is used for the
    find_connections also and then it might be favourable if the points are
    better than the trial for the next step. But it is only checked if tolerance
    is reached, as this is the minimal requirement lateron. Some warnings are given
    back if the minimizer returned with something strange but only the gradients
    too high give reason for stop.
    """
    results = deepcopy(points)

    # optimize all of them
    for i, point in enumerate(points):

         # see docstring of mueller_brown module for the strategies
         # we expect something with MIN1, TS1, MIN2, TS2, MIN3, ..
         # optimize mins and ts differently
         if i % 2 == 0:
             p, val, d = minimize(fun, point, fun.fprime, pgtol = tolerance * 0.001 )

             # check for error
             if d['warnflag'] != 0:
                 print "Minimization return with error", d['warnflag']

             # energy should not go up!
             if val > fun(point):
                 print "This should be a minimization?"
                 print "Got value", val, "for optimized calculation against", fun(point)

             gr = d['grad']
         else:
             def g2(x):
                  """
                  |g|^2 should always be minimized, also for
                  transititon states, but here w do not really
                  have the derivatives (or want to have them
                  """
                  g = fun.fprime(x)
                  return dot(g, g)

             #Thus use a minimizer without need of derivatives
             p = minimize2(g2, point, ftol = tolerance * 0.01, disp = False )
             gr = fun.fprime(p)
             # we do not have any warnings and other informations
             d = None

         if dot(gr,gr) > tolerance:
             print "This is not converged enough, thus stop"
             print "Still have forces", gr , " for point", p
             print "Calculation", d
            # exit()

         results[i] = p

    return results

def steepest_decent(force, x0, fixedlength = True, alpha = 0.001):
    """
    Simple steepest decent step,
    forces should be belonging to the paramter x
    x0 is the old place
    fixedlength = True will norm the step of the object
    alhpa is the scaling factor for the step, together with
    fixedlenght = True it is the steplength
    This version does not consider metric
    """
    step = force
    if fixedlength:
        step = step / sqrt(dot(step, step))
    return x0 + alpha * step

def steepest_decent_met(metric, force, x0, fixedlength = True, alpha = 0.01):
    """
    Simple steepest decent step,
    function fun should give the forces belonging to the paramter x
    x0 is the old place
    fixedlength = True will norm the step of the object
    alhpa is the scaling factor for the step, together with
    fixedlenght = True it is the steplength
    This version does consider metric
    """
    step = metric.raises(force, x0)
    if fixedlength:
        step = step / sqrt(dot(step, force))
    return x0 + alpha * step


def steepest_decent_path_simple(fun, x0, metric, store_every = 1,
                               max_iter = 10000000, revert_dir = False, alpha = 0.01, **params):
    """
    Steepest decent path, with simple steepest decent step.
    """
    path = [x0]

    start_step, __ = start_values(fun, metric, x0)
    if revert_dir:
        start_step = - start_step

    xval = x0 + start_step * alpha
    force = None
    force_old = None

    for it in range(max_iter):
        if conv_border(force, force_old, metric, xval):
        # loop will break here or if there are too many steps, don't want to try for infinity
            break
        if it % store_every == 0:
            path.append(xval)

        force_old = force
        force = -fun.fprime(xval)
        xval = steepest_decent_met(metric, force, xval, alpha = alpha, **params)

    path.append(xval)

    return True, array(path)


def conv_border(force, forceold, metric, x0):
    """
    Test if the convergence was reached
    """
    if forceold is None:
        return False

    force_up = metric.raises(force, x0)
    if dot(force_up, forceold) < 0:
        return True
    else:
        return False

def find_connections(fun, points, metric = Default(None), **params):
    """
    Find the minimal energy path connecting the given points (minima
    and transition points). points should contain all stationary points
    the path will cross

    This function belives fun to be a rather cheap function so that a
    few thousand steps should not make too much computationally costs
    """
    num_points = len(points)
    # at least one transition state
    assert num_points >= 3
    # num_points = n_minima + n_ts with n_ts = n_minima - 1
    assert num_points % 2 == 1

    # pall is complete path
    pall = None

    # separate minima from transition states
    minima = []
    ts = []
    for i, point in enumerate(points):
        if (i % 2 == 0):
            minima.append(point)
        else:
            ts.append(point)

    # try from each transition state to connect it
    for i in range(len(ts)):
         start = array(ts[i])
         # there are two minima we want to reach:
         goal1 = array(minima[i])
         goal2 = array(minima[i+1])

         all_reached = 3

         # first direction
         pones, min = path_l_to_r(metric, fun, start, goal1, goal2, True, **params)
         all_reached = all_reached - min

         # second direction
         p2, min = path_l_to_r(metric, fun, start, goal1, goal2, False, **params)
         all_reached = all_reached - min

         # we want the path min1-ts-min2, but there is no way of telling
         # if direction = True is min1-ts or ts-min2 beforehand, thus now
         # we have to decide if the second path piece comes before or after
         # the first one
         if min == 1:
            pones = vstack((p2, pones))
         else:
            pones = vstack((pones, p2))

         # add up all pathes, thus pall = min1-ts1-min2-ts2-min3-...
         if pall == None:
             pall = deepcopy(pones)
         else:
             pall = vstack((pall, pones))

    return pall


def path_l_to_r(metric, fun, start, reactant, product, direction, simple = False, tol_points = 1e-1, **params):
    """
    Generates a path going in steepest decent direction from start.
    As the reaction path is supposed to go from reactants over transition state to products
    the path piece start-reactants is switched.
    The convergence obtained by the steepest decent path is checked.
    Path is forwared together with the result on which path piece could be deteceted in
    the returned path.
    direction decides if the starting vector is negated, allowing easily to check both paths.

    This version considers metric
    """
    # the actual steepest decent path calculation
    if simple:
        conv, pone = steepest_decent_path_simple(fun, start, metric, revert_dir = direction, **params)
    else:
        conv, pone = steepest_decent_path(fun, start, metric, revert_dir = direction, **params)

    if not conv:
        print "This path did not converge within the given number of steps"
        print "Started", start," invert start direction", direction

    # ckeck if the results from the path are somewhat near the reactants or products
    diff1 = pone[-1] - reactant
    diff2 = pone[-1] - product
    diff1_down = metric.lower(diff1, 0.5*(pone[-1] + reactant))
    diff2_down = metric.lower(diff2, 0.5*(pone[-1] + product))

    if dot(diff1, diff1_down) < tol_points  and dot(diff1, diff1_down) < dot(diff2, diff2_down):
        # we got as path: start-reactants, switch it to reactants-start
        pone = revert_mat(pone)
        min = 1
    elif dot(diff2, diff2_down) < tol_points:
        # as path: start -products, don't change anything
        min = 2
    else:
        # None of the minima found, thus we cannot tell which of them we got
        # and wheter it is desired to swith the path
        print "Path not converged to one of the expected minima"
        min = 0

    return pone, min

def revert_mat(m1):
    """
    Reverts the context of the matrix along the first axis
    """
    m2 = zeros(m1.shape)
    lm1_1 = len(m1) - 1
    for i, m in enumerate(m1):
       m2[lm1_1 - i] = m
    return m2

