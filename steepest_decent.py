from pts.ode import rk45
from numpy import dot, array, sqrt
from math import pow
from pts.func import NumDiff
from scipy.linalg import eig
from numpy import finfo

# This is how exact we can have floating point values
tol_exact = finfo(float).eps

def steepest_decent_path(fun, x0, metric, store_steps = 0.0,
                        tolerance = 1e-10, max_iter = 10000000, revert_dir = False, **params):
    """
    Calculates a steepest decent path starting from a point (minima or
    transition state). Needs a starting step as at the stationary points
    forces should be zero.

    The integration will be done by a

    See: Matthias Bollhoefer and. Vokler Mehrmann, Numerische Mathematik
    """

    path = [x0]
    xval, h = make_first_step(x0, fun, metric, revert_dir, tolerance)
    print "Starting", xval, h, x0

    def phi(t, x):
        """
        Function for putting it in the Runge-Kutta step function
        We don't have any dependency of t, but Runge-Kutta expect
        t to be needed in the function
        Considers the metric. and normalises the vector.
        """
        s = - fun.fprime(x)
        s_up = metric.raises(s,x)
        return  s_up

    # t for runge-kutta integration
    t = 0
    t_old = 0


    #FIXME: we need some borders for h, but how to specify them best?
    h_old = h

    # converged means reached a minimum
    converged = False

    # needed if we overstep
    xold = None
    for it in xrange(max_iter):

         # don't having fixed end like in normal integration, integrate
         # till convergence reached, scale step down if we stepped to far
         # this is a bit more complicated than for normal integration with fixed border
         converged, xval = convergence_and_border(fun, metric, xval, xold, tolerance)
         if converged:
             # loop will break here or if there are too many steps, don't want to try for infinity
             break

         # runge-kutta steps with Runge-Kutta-Feldberg - 4(5)
         # ATTENTION: this code returns step * h in principle
         step1, step2 = rk45(t, xval, phi, h)
         stepg1 = metric.lower(step1, xval)
         stepg2 = metric.lower(step2, xval)

         # error between 4 and 5, as we want difference between real steps have to
         # divide by h's here
         tau = sqrt(dot(step1 - step2, stepg1 - stepg2)) / h**2

         # for relative error one needs |x|^2 (+1 for not having this to become too small)
         nu = metric.norm_up(xval, xval) + 1.

         if tau  <= tolerance * nu:
             # accept this step, update xval and t
             t = t + h
             xold = xval
             xval = xval + step1
             h_old = h

             # store path (but don't take the steps here too small)
             if t - t_old >= store_steps:
                 # we will need the path only for plotting mostly
                 # here we need them only in some bigger interval
                 path.append(xval)
                 t_old = t

         # FIXME: testing for tau==NaN, but if try which value to use
         if tau != tau:
             if h <= h_old:
                 h = h * 0.01
             else:
                 h = h_old
             #print "Reseted h", h

         # scale the step length:
         # enlarge if we are at least 2 times better than we need
         # or shrinken it if we don't fullfil our requirements
         # tau==Nan will just skip the trial
         if tau <= tolerance * nu / 2. or tau > tolerance * nu:
             # step scaling, for hoping for tau = tolerance in next step
             if tau < tol_exact:
                 tau = tol_exact
             factor = pow((tolerance / tau) * nu, 0.25)
             h = h * factor

             # runge -kutta will need also some smaller fractions of h
             # and the differences in the results should be meaningfull
             # but give warning as it might cause calculation to stop
             if h < tol_exact * 100 or h != h:
                 print "Warning doing minimal step"
                 h = tol_exact * 100

    # keep the last step in any case
    path.append(xval)

    # just debugging output
    force = - fun.fprime(xval)
    print "converged", dot(force, force),tau, h

    return converged, array(path)

def convergence_and_border(fun, metric, x, xold, tol):
    """
    Test for convergence (end of iteration of path)
    and rescale the step if it goes too far:
    """
    force = - fun.fprime(x)

    if xold is None:
        # First step return
        return False, x

    # forces from previous step, better store for expensive calcualtions
    force_old = - fun.fprime(xold)


    conv = False
    force_old_up = metric.raises(force_old, xold)
    step = x - xold

    # just ensure this is not going on for ever:
    for i in xrange(15):

        # this means forces are both in the same direction or too near to each other to
        # do useful things with their differences
        if dot(force, force_old_up) >= 0. or abs(dot(force, force_old_up)) < finfo(float).eps:
            break
        print "Inverted direction, rescale", i
        # else: dot(force, force_old_up) < 0., means new forces mostly in opposite direction than
        # the ones from the last iteration, thus we overstepped, thus we should better
        # try a smaller step

        # Choose a better step
        # consider: - d_force/dS ~~ (force_old - force)/ step
        # 0 = d_force * dS -force -> dS = force /d_force
        scale = force / (force_old - force)
        step = step  * scale
        force = - fun.fprime(xold + step)


    # new x value
    x = xold + step

    # Now check how large the forces at the newest point are at all:
    force_up = metric.raises(force, x)

    # We have also to check if forces are going down, otherwise we might be too near the
    # TS still
    # FIXME: can we expect the same tolerance in this direction than in the other?
    if dot(force, force_up) < tol * 100 and dot(force, force_up) < dot(force_old, force_old_up):
        #print "Converged to minima"
        conv = True

    return conv, x

def make_first_step(x0, fun, metric, revert_dir, tolerance):
    """
    at TS no real steepest decent direction
    thus get first step differently

    direction can be gotten by second derivatives
    steplength has to be calculated differently: try not to
    have it too big, but the forces there should already be larger
    than our tolerance as we expect to have exactness (of the TS)
    only to this tolerance

    Check further if the direction of the forces there are going
    in direction of the TS
    """
    # direction, curvature for step length approximation
    start_dir, curvature = start_values(fun, metric, x0)

    # both signs are valid, as both paths might be wanted
    # decide here the direction
    if revert_dir:
        start_dir = - start_dir

    # FIXME: how to best specify the lenght of the first step
    h =  tolerance / abs(curvature) * 1.00001
    h_keep = h
    if h < tol_exact * 100:
        print "Enlarging step, attention might be below tolerance"
        h = tol_exact * 100

    for i in range(15):
        # enlarge step till the tolerance is reached
        xval = x0 + h * start_dir
        h_keep = h
        force = - fun.fprime(xval)
        for_length = metric.norm_down(force, xval)
        f_dir = force / for_length
        if abs(dot(f_dir, start_dir) -1.) < tolerance and for_length > tolerance:
            break
        h = h * 2.

    return xval, h_keep


def start_values(fun, metric, x0):
    """
    At start the gradient should vanish, thus
    use second derivatives for deciding on the first
    direction

    Example: use mueller brown potential
    >>> from pts.mueller_brown import MB, CHAIN_OF_STATES
    >>> from pts.metric import Default, Metric
    >>> from pts.func import compose
    >>> from pts.test.testfuns import mb2
    >>> from numpy import zeros

    Around first transition state
    >>> ts1 = CHAIN_OF_STATES[1]

    This direction should go in steepest decent direction
    >>> sdir, __ = start_values(MB, Default(), ts1)

    Is normed:
    >>> dot(sdir, sdir) - 1. < 1e-12
    True

    Some energy values:
    First at transtition state:
    >>> MB(ts1)
    -40.66484351147777

    Energy should go down in direction of steepest decent
    >>> MB(ts1 + sdir * 0.0001)
    -40.664847693656483
    >>> MB(ts1 + sdir * 0.01)
    -40.701726989599678
    >>> MB(ts1 - sdir * 0.01)
    -40.703002363574996

    On mueller brown its easy to get the perpendicular direction:
    >>> max_dir = zeros(2)
    >>> max_dir[0] = sdir[1]
    >>> max_dir[1] = -sdir[0]

    Here energy should go up
    >>> MB(ts1 + max_dir * 0.01)
    -40.640087066488903
    >>> MB(ts1 - max_dir * 0.01)
    -40.64058812903157

    Mixed directions:
    >>> MB(ts1 + (sdir * 0.9**2 + max_dir * 0.1**2) * 0.01) > MB(ts1 + sdir * 0.01)
    True

    Test some other functions and directions:
    >>> fun = mb2()
    >>> mb = compose(MB, fun)
    >>> ts1_2 = fun.pinv(ts1)
    >>> met = Metric(fun)
    >>> max(abs(ts1 - fun(ts1_2))) < 1e-12
    True

    >>> sdir2, k = start_values(mb, Default(), ts1_2)

    # perpendicular direction:
    >>> max_dir = zeros(2)
    >>> max_dir[0] = sdir2[1]
    >>> max_dir[1] = -sdir2[0]

    # energy goes down in sdir and up in max_dir
    >>> mb(ts1_2 + sdir2 * 0.01)
    -40.679755453436535
    >>> mb(ts1_2 + max_dir * 0.01)
    -40.649340351181749

    Mixed directions:
    >>> mb(ts1_2 + (sdir2 * 0.9**2 + max_dir * 0.1**2) * 0.01) > mb(ts1_2 + sdir2 * 0.01)
    True

    We got different directions than in "Cartesian" coordinates:
    >>> max(abs(ts1 + sdir * 0.01 - fun(ts1_2 + sdir2 * 0.01))) > 1e-3
    True
    >>> max(abs(ts1 + sdir * 0.01 - fun(ts1_2 - sdir2 * 0.01))) > 1e-2
    True

    >>> sdir2, k = start_values(mb, met, ts1_2)

    # perpendicular direction:
    >>> max_dir = zeros(2)
    >>> max_dir[0] = sdir2[1]
    >>> max_dir[1] = -sdir2[0]

    # energy goes down in sdir and up in max_dir
    >>> mb(ts1_2 + sdir2 * 0.01)
    -40.70300236357501
    >>> mb(ts1_2 + max_dir * 0.01)
    -40.625299777152236

    Here we got the same direction
    >>> max(abs(ts1 + sdir * 0.01 - fun(ts1_2 - sdir2 * 0.01))) < 1e-14
    True
    """
    f2 = NumDiff(fun.fprime)
    # second derivatives
    hess = f2.fprime(x0)
    # Metric norm matrix
    g = metric.g_mat(x0)

    # search for smallest eigenvalue and its vector
    # of H v = w g v
    w, v = eig(hess,b = g)
    a = w.argmin()
    start_dir = v[:,a]

    # direction should be normed
    start_dir = start_dir / metric.norm_up(start_dir, x0)
    return start_dir, w[a]

# python steepest_decent.py [-v]:
if __name__ == "__main__":
    import doctest
    doctest.testmod()