#!/usr/bin/python
"""
Adapted from cosopt/quadratic_string.py
"""

__all__ = ["odeint1", "rk45", "rk4", "rk5"]

from numpy import asarray, max, abs
from scipy.integrate import odeint

def odeint1(t0, y0, f, T=None, args=(), tol=1.0e-7, maxiter=3):
    """Integrate

        dy / dt = f(t, y)

    from (t0, y0) to t = T (or infinity)

    Example:

        >>> def f(t, y):
        ...     yp = - (y - 100.0)
        ...     yp[0] *= 0.01
        ...     yp[1] *= 100.
        ...     return yp

        >>> t0 = 0.0
        >>> y0 = [80., 120]

        >>> odeint1(t0, y0, f)
        array([ 100.,  100.])
    """

    y = asarray(y0).copy()

    yshape = y.shape
    ysize = y.size

    # odeint() from scipy expects f(y, t) and flat array y:
    def f1(y, t):

        # restore shape:
        y.shape = yshape

        # call original function:
        yp = f(t, y, *args)

        # flatten:
        y.shape = ysize
        yp.shape = ysize

        return yp

    # initial values:
    t = t0
    y.shape = ysize

    if T is not None:
        #
        # compute y(t + T):
        #
        ys = odeint(f1, y, [t, t + T])

        y = ys[1]
    else:
        # guess for the upper integration limit:
        T = 1.0

        iteration = -1
        while iteration < maxiter:
            iteration += 1
            #
            # compute y(t + T) and y(t + 2 * T):
            #
            ys = odeint(f1, y, [t, t + T, t + 2 * T])

            if max(abs(ys[2] - ys[1])) < tol:
#               print "odeint1: T=", t + 2 * T
                break

            # advance and scale the upper limit:
            t += 2 * T
            T *= 4
            y = ys[2]

        if iteration >= maxiter:
            print "odeint1: WARNING: maxiter=", maxiter, "exceeded"

        y = ys[2]

    # return in original shape:
    y.shape = yshape

    return y

def rk45(t1, x1, f, h, args=()):
    """Returns RK4 and RK5 steps as a tuple.

        >>> from numpy import exp

    For x(t) = x0 * exp(-t) the derivative is given by:

        >>> def f(t, x): return -x

        >>> x0 = 100.

        >>> rk45(0.0, x0, f, 1.0)
        (-63.461538461538453, -63.285256410256402)

    Actual solution changes by

        >>> x0 * exp(-1.0) - x0
        -63.212055882855765

    Euler step would have been:
        >>> 1.0 * (- x0)
        -100.0
    """

    k1 = h * f(t1, x1, *args)

    t2 = t1 + 1.0/4.0 * h
    x2 = x1 + 1.0/4.0 * k1
    k2 = h * f(t2, x2, *args)

    t3 = t1 + 3.0/8.0 * h
    x3 = x1 + 3.0/32.0 * k1 + 9.0/32.0 * k2
    k3 = h * f(t3, x3, *args)

    t4 = t1 + 12.0/13.0 * h
    x4 = x1 + 1932./2197. * k1 - 7200./2197. * k2 + 7296./2197. * k3
    k4 = h * f(t4, x4, *args)

    t5 = t1 + 1.0 * h
    x5 = x1 + 439./216. * k1 - 8. * k2 + 3680./513. * k3 - 845./4104. * k4
    k5 = h * f(t5, x5, *args)

    t6 = t1 + 0.5 * h
    x6 = x1 - 8./27.*k1 + 2.*k2 - 3544./2565. * k3 + 1859./4104.*k4 - 11./40. * k5
    k6 = h * f(t6, x6, *args)

    step4 = 25./216.*k1 + 1408./2565.*k3 + 2197./4104.*k4 - 1./5.*k5
    step5 = 16./135.*k1 + 6656./12825.*k3 + 28561./56430.*k4 - 9./50.*k5 + 2./55.*k6

    return step4, step5

def rk4(t1, x1, f, h, args=()):
    """Returns RK4 step.
    """

    k1 = h * f(t1, x1, *args)

    t2 = t1 + 1.0/4.0 * h
    x2 = x1 + 1.0/4.0 * k1
    k2 = h * f(t2, x2, *args)

    t3 = t1 + 3.0/8.0 * h
    x3 = x1 + 3.0/32.0 * k1 + 9.0/32.0 * k2
    k3 = h * f(t3, x3, *args)

    t4 = t1 + 12.0/13.0 * h
    x4 = x1 + 1932./2197. * k1 - 7200./2197. * k2 + 7296./2197. * k3
    k4 = h * f(t4, x4, *args)

    t5 = t1 + 1.0 * h
    x5 = x1 + 439./216. * k1 - 8. * k2 + 3680./513. * k3 - 845./4104. * k4
    k5 = h * f(t5, x5, *args)

    step4 = 25./216.*k1 + 1408./2565.*k3 + 2197./4104.*k4 - 1./5.*k5

    return step4

def rk5(t1, x1, f, h, args=()):
    """Returns RK5 step.
    """

    k1 = h * f(t1, x1, *args)

    t2 = t1 + 1.0/4.0 * h
    x2 = x1 + 1.0/4.0 * k1
    k2 = h * f(t2, x2, *args)

    t3 = t1 + 3.0/8.0 * h
    x3 = x1 + 3.0/32.0 * k1 + 9.0/32.0 * k2
    k3 = h * f(t3, x3, *args)

    t4 = t1 + 12.0/13.0 * h
    x4 = x1 + 1932./2197. * k1 - 7200./2197. * k2 + 7296./2197. * k3
    k4 = h * f(t4, x4, *args)

    t5 = t1 + 1.0 * h
    x5 = x1 + 439./216. * k1 - 8. * k2 + 3680./513. * k3 - 845./4104. * k4
    k5 = h * f(t5, x5, *args)

    t6 = t1 + 0.5 * h
    x6 = x1 - 8./27.*k1 + 2.*k2 - 3544./2565. * k3 + 1859./4104.*k4 - 11./40. * k5
    k6 = h * f(t6, x6, *args)

    step5 = 16./135.*k1 + 6656./12825.*k3 + 28561./56430.*k4 - 9./50.*k5 + 2./55.*k6

    return step5

# python ode.py [-v]:
if __name__ == "__main__":
    import doctest
    doctest.testmod()

# Default options for vim:sw=4:expandtab:smarttab:autoindent:syntax
