"""
Import minimization funciton:

    >>> from scipy.optimize import fmin_l_bfgs_b as minimize

Find the three minima:

    >>> b, _, _ = minimize(energy, [0,0], gradient)
    >>> b
    array([ 0.62349942,  0.02803776])

    >>> a, _, _ = minimize(energy, [-1,1], gradient)
    >>> a
    array([-0.55822362,  1.44172583])

    >>> c, _, _ = minimize(energy, [0,1], gradient)
    >>> c
    array([-0.05001084,  0.46669421])

Import the path representtion:

    >>> from path import Path

Construct a path connecting two minima b and c:
    >>> p = Path([b,c])

    >>> p(0)
    array([ 0.62349942,  0.02803776])

    >>> p(1)
    array([-0.05001084,  0.46669421])

    >>> energy(p(0))
    -108.16672411685231

    >>> energy(b)
    -108.16672411685231

Simplex minimizer that does not require gradients:

    >>> from scipy.optimize import fmin
    >>> bc = fmin(lambda x: -energy(p(x)), 0.5)
    Optimization terminated successfully.
             Current function value: 72.246872
             Iterations: 12
             Function evaluations: 24

    >>> bc
    array([ 0.60537109])

    >>> p(bc)
    array([ 0.21577578,  0.29358769])

Define a function that retunrs the square of the gradient,
it has its minima at stationary points, both at PES minima
and PES saddle points:

    >>> from numpy import dot
    >>> def g2(x):
    ...   g = gradient(x)
    ...   return dot(g, g)
    ...
    >>> ts2 = fmin(g2, [0.2, 0.3])
    Optimization terminated successfully.
             Current function value: 0.000022
             Iterations: 24
             Function evaluations: 46
    >>> ts2
    array([ 0.21248201,  0.2929813 ])
    >>> ts2 - p(bc)
    array([-0.00329377, -0.00060639])

... very close! Compare the energies:

    >>> energy(ts2)
    -72.248940103363921
    >>> energy(p(bc))
    -72.246871930853558

The true TS between b and c is at (0.212, 0.293)
with the energy -72.249 according to MB79.

But for another TS approximation between minima a and b
the guess is much worse:

    >>> p = Path((a,b))
    >>> ab = fmin(lambda x: -energy(p(x)), 0.5)
    Optimization terminated successfully.
             Current function value: -12.676227
             Iterations: 13
             Function evaluations: 26

    >>> ab
    array([ 0.30664062])

    >>> p(ab)
    array([-0.19585933,  1.00823164])

    >>> energy(p(ab))
    12.676227327570253

The true TS between a and b is at  (-0.822, 0.624)
with the energy -40.665 according to MB79 that is significantly
further away than the maximum on the linear path between a and b.

    >>> ts1 = fmin(g2, [-0.8, 0.6])
    Optimization terminated successfully.
             Current function value: 0.000028
             Iterations: 28
             Function evaluations: 53
    >>> ts1
    array([-0.82200123,  0.62430438])
    >>> energy(ts1)
    -40.664843511462038

So indeed tha path maximum p(ab) is not even close to a saddle point,
and the gradient minimization would even fail if starting from p(ab).

To use a minimizer with gradients we need to compose functions AND the gradients
consistent with the chain differentiation rule for this:

    q(x) = q(p(x))

Remember that p(x) is the path funciton defined above.

    >>> q = MuellerBrown()

Input is assumed to be a vector by l_bfgs_b minimizer:

    >>> def f(x): return -q.f( p.f(x[0]) )

And this is no more than the chain differentiation rule, with
type wrappers to make l_bfgs_b optimizer happy:

    >>> def fprime(x): return -dot( q.fprime( p.f(x[0]) ), p.fprime(x[0]) ).flatten()

flatten() here has an effect of turning a scalar into length 1 array.

    >>> ab, _, _ = minimize(f, [0.5], fprime, bounds=[(0., 1.)])
    >>> ab
    array([ 0.30661623])

(this converges after 10 func calls, compare with 26 in simplex method)

The composition may be automated by a type Func() operating
with .f and .fprime methods.

    >>> from func import compose
    >>> e = compose(q, p)
    >>> e(0.30661623), e.fprime(0.30661623)
    (12.676228284381487, 8.8724010719257174e-06)

Gives the same energy of the maximum on the linear path between a and b
and almost zero gradient. To use it with minimizer one needs
to invert the sign and wrap scalars/vectors into arrays:

    >>> def f(x): return -e.f(x[0])
    >>> def fprime(x): return -e.fprime(x[0]).flatten()

    >>> ab, _, _ = minimize(f, [0.5], fprime, bounds=[(0., 1.)])
    >>> ab
    array([ 0.30661623])

Build the approximate energy profile along the A-B path,
use only seven points to evaluate MB79, both values
and gradients:

    >>> from numpy import linspace
    >>> xs      = linspace(0., 1., 7)
    >>> ys      = [ e(x)        for x in xs ]
    >>> yprimes = [ e.fprime(x) for x in xs ]

    >>> from func import CubicSpline
    >>> e1 = CubicSpline(xs, ys, yprimes)

Maximal difference between the real energy profile and
cubic spline approximaton:

    >>> xx = linspace(0., 1., 71)
    >>> errs = [abs(e(x) - e1(x)) for x in xx]
    >>> xm, err = max(zip(xx, errs), key=lambda s: s[1])

    >>> xm, err
    (0.58571428571428574, 0.73161131588862816)

    >>> e(xm), e1(xm)
    (-57.002611182814157, -56.270999866925528)

Find an energy minimum at approximated energy profile:

    >>> def f(x): return -e1.f(x[0])
    >>> def fprime(x): return -e1.fprime(x[0]).flatten()

    >>> ab1, _, _ = minimize(f, [0.5], fprime, bounds=[(0., 1.)])
    >>> ab1
    array([ 0.30776737])

This is not much different from the previous result, 0.30661623.

    >>> -e(0.30776737), -e1(0.30776737)
    (-12.674103478706414, -12.634608278360837)

    >>> -e(0.30661623), -e1(0.30661623)
    (-12.676228284381487, -12.632359891279457)
"""
__all__ = ["energy", "gradient"] # "MuellerBrown"]

from numpy import exp, array

# FIXME: how to mve them into class definiton out of global namespace?
AA = (-200., -100., -170.,  15.)
aa = (  -1.,   -1.,   -6.5,  0.7)
bb = (   0.,    0.,   11.,   0.6)
cc = ( -10.,  -10.,   -6.5,  0.7)
xx = (   1.,    0.,   -0.5, -1.)
yy = (   0.,    0.5,   1.5,  1.)

class MuellerBrown():
    """
    Mueller, Brown, Theoret. Chim. Acta (Berl.) 53, 75-93 (1979)
    Potential used in string method publications

    Initialze the calculator:

        >>> mb = MuellerBrown()

    Compute energy:
        >>> mb((0,0))
        -48.401274173183893

        >>> mb((-0.558, 1.442))
        -146.69948920058778

        >>> mb((0.623, 0.028))
        -108.16665005353302

        >>> mb((-0.050, 0.467))
        -80.767749248757724

    "Calling" |mb| is the same as invoking its value method |f|:

        >>> mb.f((-0.050, 0.467))
        -80.767749248757724

    or gradients:

        >>> mb.fprime((-0.558, 1.442))
        array([ -1.87353829e-04,   2.04493895e-01])

        >>> mb.fprime((0.623, 0.028))
        array([-0.28214372, -0.19043391])

        >>> mb.fprime((-0.050, 0.467))
        array([ 0.04894393,  0.44871342])

    
    (WARNING: OCRed data below)

    Table 1. Energy minima and saddle points of the analytical, two-parametric model potential v
    Energy minima a
               E        x     y
    Minimum A -146.700 -0.558 1.442
    Minimum B -108.167  0.623 0.028
    Minimum C  -80.768 -0.050 0.467
    
    Saddle points b
     E       x     y      E      x     y
    -40.665 -0.822 0.624 -72.249 0.212 0.293
    
    Saddle points c
    Run (P1, P2) E      x    y    E      x    y    N d
    1 (A, B)    -40.67 -0.82 0.62 --               124
    2 (B, A)    -40.68 -0.82 0.62 --               155
    3 (B, C)    --                -72.26 0.22 0.29 74
    4 (C, B)    --                -72.27 0.21 0.30 73
    5 (A, C)    -40.68 -0.81 0.62 --               112
    6 (C, A)    -40.67 -0.82 0.62 --               113

    a Determined by unconstrained energy minimization using the simplex method [25].
    b Determined by minimization of |grad E|^2
    c Determined by the constrained simplex optimization procedure 8.
    d Total number of function evaluations.
    """

    def __str__(self):
        return "MuellerBrown"

    def f(self, v):

        x = v[0]
        y = v[1]

        f = 0.0
        for A, a, b, c, x0, y0 in zip(AA, aa, bb, cc, xx, yy):
            f += A * exp( a * (x - x0)**2 + b * (x - x0) * (y - y0) + c * (y - y0)**2 )

        return f

    # make MB PES callable:
    __call__ = f

    def fprime(self, v):

        x = v[0]
        y = v[1]

        dfdx = 0.0
        dfdy = 0.0
        for A, a, b, c, x0, y0 in zip(AA, aa, bb, cc, xx, yy):
            expt = A * exp( a * (x - x0)**2 + b * (x - x0) * (y - y0) + c * (y - y0)**2 )
            dfdx += ( 2 * a * (x - x0) + b * (y - y0) ) * expt
            dfdy += ( 2 * c * (y - y0) + b * (x - x0) ) * expt

        return array((dfdx, dfdy))


# define MB constant, define two functions without
# another level of indirection:
MB = MuellerBrown()
energy = MB.f
gradient = MB.fprime

# python mueller_brown.py [-v]:
if __name__ == "__main__":
    import doctest
    doctest.testmod()

# Default options for vim:sw=4:expandtab:smarttab:autoindent:syntax

