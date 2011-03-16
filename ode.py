#!/usr/bin/env python
"""
Runge-Kutta procedures adapted from cosopt/quadratic_string.py

FIXME: The "exact" integration using scipy.integrate.odeint without
tolerance settings may require unnecessary much time. Is there a better
way?
"""

__all__ = ["odeint1", "rk45", "rk4", "rk5"]

from numpy import array, max, abs, searchsorted
from scipy.integrate import odeint
from func import Func

VERBOSE = False

def odeint1(t0, y0, f, T=None, tol=1.0e-7, maxiter=12):
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

    y = ODE(t0, y0, f)

    if T is not None:
        return y(T)
    else:
        # integrate to infinity, for that guess
        # the upper integration limit:
        T = 1.0

        # will be comparing these two:
        y1, y2 = y(T), y(2*T)

        iteration = -1
        while max(abs(y2 - y1)) > tol and iteration < maxiter:
            iteration += 1

            # scale the upper limit and advance:
            T *= 2
            y1, y2 = y2, y(2*T)

        if iteration >= maxiter:
            print "odeint1: WARNING: maxiter=", maxiter, "exceeded"

        if VERBOSE:
            print "odeint1: T=", 2 * T, "guessed", iteration, "times"

        return y2

class ODE(Func):
    def __init__(self, t0, y0, f, args=()):
        """Build a Func() y(t) by integrating

            dy / dt = f(t, y, *args)

        from t0 to t.

        It is supposed to work for any shape of y.

        Example:

            >>> def f(t, y):
            ...     yp = - (y - 100.0)
            ...     yp[0, 0] *= 0.01
            ...     yp[0, 1] *= 100.
            ...     yp[1, 0] *= 0.1
            ...     yp[1, 1] *= 1.0
            ...     return yp

            >>> t0 = 0.0
            >>> y0 = [[80., 120], [20.0, 200.]]

            >>> y = ODE(t0, y0, f)

            >>> y(0.0)
            array([[  80.,  120.],
                   [  20.,  200.]])

            >>> y(1.0)
            array([[  80.19900333,   99.99999999],
                   [  27.61300655,  136.78795028]])

        At large t all elements approach 100.0:

            >>> max(abs(y(10000.0) - 100.0)) < 1.0e-9
            True

            >>> max(abs(y.fprime(2.0) - f(2.0, y(2.0)))) == 0.0
            True
        """

        # make a copy of the input (paranoya):
        y0 = array(y0)

        # table of know results:
        self.__ts = [t0]
        self.__ys = {t0 : y0}

        self.__yshape = y0.shape
        self.__ysize = y0.size

        # odeint() from scipy expects f(y, t) and flat array y:
        def f1(y, t):

            # restore shape:
            y.shape = self.__yshape

            # call original function:
            yp = f(t, y, *args)

            # flatten:
            y.shape = self.__ysize
            yp.shape = self.__ysize

            return yp

        # this 1D-array valued function will be integrated:
        self.__f1 = f1

    def f(self, t):

        # aliases:
        ts = self.__ts
        ys = self.__ys
        f1 = self.__f1

        if t in ts:
            return ys[t]

        i = searchsorted(ts, t)

        # FIXME: t >= t0:
        assert i > 0

        # integrate from t0 < t:
        t0 = ts[i-1]
        y0 = ys[t0]

        assert t > t0

        # reshape temporarily:
        y0.shape = self.__ysize

        #
        # compute y(t):
        #
        _y0, y = odeint(f1, y0, [t0, t])

        # restore original shape:
        y0.shape = self.__yshape
        y.shape = self.__yshape

        # insert new result into table:
        ts.insert(i, t)
        ys[t] = y

        if VERBOSE:
            print "ODE: ts=", ts
            print "ODE: ys=", ys

        return y.copy()

    def fprime(self, t):

        y = self.f(t)

        y.shape = self.__ysize

        yp = self.__f1(y, t)

        yp.shape = self.__yshape

        return yp

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
