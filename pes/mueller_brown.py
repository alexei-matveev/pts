"""
Import minimization funciton:

    >>> MB([ 0.62349942,  0.02803776])
    -108.16672411685232

    >>> MB.fprime([ 0.62349942,  0.02803776])
    array([  8.57060655e-06,   6.74209871e-06])

    >>> MB([-0.55822362,  1.44172583])
    -146.69951720995329

    >>> MB([-0.05001084,  0.46669421])
    -80.767818129651189

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

    >>> from pts.path import Path

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

Test also dimer:
    >>> from pts.dimer import dimer
    >>> from pts.metric import Default

    >>> res, dict = dimer(MB, array([0.2, 0.3]), array([0.,1.]),
    ... Default(), dimer_distance = 0.001, trial_step = 0.0003,
    ... start_step_length = 0.01, max_step = 0.001 ) #doctest:+ELLIPSIS
    Intermediate steps ...
    ...
    Calculation is converged

    We got the positions:
    >>> print "%5.4f %5.4f" % (res[0], res[1])
    0.2125 0.2930

    But it has reached convergence
    >>> dict["trans_convergence"]
    True
    >>> print round(energy(res), 6)
    -72.24894

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

    >>> from pts.func import compose
    >>> e = compose(q, p)
    >>> e(0.30661623), e.fprime(0.30661623)
    (12.676228284381487, array(8.8724010719257174e-06))

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

    >>> from pts.func import CubicSpline
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

Test also dimer:
    >>> res, dict = dimer(MB, array([-0.7, 0.5]), array([0.,1.]),
    ... Default(), dimer_distance = 0.001,
    ... start_step_length = 0.0003, max_step = 0.005 ) #doctest:+ELLIPSIS
    Intermediate steps ...
    ...
    Calculation is converged

    We got the positions:
    >>> print "%5.4f %5.4f" % (res[0], res[1])
    -0.8220 0.6243

    But it has reached convergence
    >>> dict["trans_convergence"]
    True
    >>> print round(energy(res), 6)
    -40.664844
"""
__all__ = ["energy", "gradient"] # "MuellerBrown"]

from numpy import exp, array, asarray, shape, zeros
from pts.func import Func

# special points on PES:
CHAIN_OF_STATES = [[-0.55822362,  1.44172583], # A
                   [-0.82200123,  0.62430438], # AC
                   [-0.05001084,  0.46669421], # C
                   [ 0.21248201,  0.2929813 ], # BC
                   [ 0.62349942,  0.02803776]] # B

# FIXME: how to mve them into class definiton out of global namespace?
AA = (-200., -100., -170.,  15.)
aa = (  -1.,   -1.,   -6.5,  0.7)
bb = (   0.,    0.,   11.,   0.6)
cc = ( -10.,  -10.,   -6.5,  0.7)
xx = (   1.,    0.,   -0.5, -1.)
yy = (   0.,    0.5,   1.5,  1.)

class MuellerBrown(Func):
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

    def taylor(self, v):

        v = asarray(v)

        x = v[0]
        y = v[1]

        f = zeros(shape(v)[1:])
        df = zeros(shape(v))

        for A, a, b, c, x0, y0 in zip(AA, aa, bb, cc, xx, yy):
            expt = A * exp( a * (x - x0)**2 + b * (x - x0) * (y - y0) + c * (y - y0)**2 )
            f += expt
            df[0] += ( 2 * a * (x - x0) + b * (y - y0) ) * expt
            df[1] += ( 2 * c * (y - y0) + b * (x - x0) ) * expt

        # FIXME: for doctests only:
        if shape(f) == ():
            f = f.item()

        return f, df


# define MB constant, define two functions without
# another level of indirection:
MB = MuellerBrown()
energy = MB.f
gradient = MB.fprime

#
# This is a more or less accurate path representation,
# obtained with sopt.py and tangent definition based
# on Path interpolation:
#
_text = \
"""
-5.582236200000002535e-01 1.441725830000000652e+00
-5.889095044792225231e-01 1.411000926817213452e+00
-6.195840790635999129e-01 1.380260174772572235e+00
-6.502178138647621841e-01 1.349517065675220007e+00
-6.807960786422688759e-01 1.318761934720610318e+00
-7.113781251764491609e-01 1.287900034772518909e+00
-7.417972785381243828e-01 1.257049544795482809e+00
-7.721089144294280837e-01 1.226085276927466605e+00
-8.022815661772801921e-01 1.194937418538846385e+00
-8.322188395145893036e-01 1.163549321887947796e+00
-8.618370189843385099e-01 1.131747292812276040e+00
-8.908200979277289822e-01 1.099393077974350597e+00
-9.187750116566468783e-01 1.066022609569609036e+00
-9.446427478075644091e-01 1.031124282074677323e+00
-9.668226256844321131e-01 9.935161114957989925e-01
-9.822542702428508354e-01 9.524609978127120957e-01
-9.877368538593902514e-01 9.091620239809889981e-01
-9.830185180917577759e-01 8.665624677200165493e-01
-9.705123209529852302e-01 8.246428276183876971e-01
-9.518133037561034016e-01 7.842201403789466063e-01
-9.295594917283934722e-01 7.464204400465751510e-01
-9.045355555439946738e-01 7.114179413788591644e-01
-8.767345112996879353e-01 6.777792651868104201e-01
-8.458895525998182396e-01 6.460285417357766180e-01
-8.130110079762739028e-01 6.168652176893446581e-01
-7.780960887525804059e-01 5.908184285598830909e-01
-7.410591936468293728e-01 5.680609188154156497e-01
-7.019697816912031652e-01 5.488095669531681819e-01
-6.613762725693010935e-01 5.332307386202752575e-01
-6.196783322923072879e-01 5.210879737184220817e-01
-5.772249060035197221e-01 5.118930514367940887e-01
-5.343073936497869658e-01 5.050462819012045967e-01
-4.911583990497283558e-01 4.999436413706714877e-01
-4.478980613638710029e-01 4.960452408155087145e-01
-4.045713676980098872e-01 4.929120450424777644e-01
-3.612142873000621091e-01 4.902085203335573294e-01
-3.178453553535365850e-01 4.876842196360078319e-01
-2.744803136584368364e-01 4.851532098905791401e-01
-2.311298149267339841e-01 4.824713942083047602e-01
-1.877847457094780348e-01 4.795152977331787469e-01
-1.444510577286106379e-01 4.761623819293467896e-01
-1.011799826162962102e-01 4.722733614748899633e-01
-5.802510986391933717e-02 4.676555980275141633e-01
-1.507675361880467940e-02 4.620038980345776980e-01
2.786966365893274802e-02 4.544608670703070841e-01
6.988145914151772109e-02 4.425891832332836673e-01
1.086897438504894758e-01 4.227551032824524713e-01
1.418785931315222903e-01 3.943589002685154998e-01
1.694674024145219060e-01 3.609023676867177954e-01
1.935778478694709093e-01 3.247261960482917820e-01
2.158362727789536750e-01 2.871598259909757811e-01
2.373198864607767022e-01 2.493859314543764860e-01
2.590351419082075335e-01 2.118196657674288608e-01
2.821093628166766454e-01 1.750765232080689071e-01
3.076928109224640884e-01 1.396477684013741616e-01
3.370299605906669771e-01 1.072505461149791256e-01
3.705794177775519338e-01 8.021574384713055228e-02
4.090877220815290483e-01 6.003404085915361316e-02
4.510063844868033023e-01 4.711301040021879244e-02
4.937934212273279444e-01 3.948027429901658503e-02
5.369003309458981743e-01 3.462090618581825652e-02
5.802029433071246300e-01 3.100631855274589566e-02
6.234994199999999154e-01 2.803775999999991164e-02
"""

from numpy import loadtxt
from StringIO import StringIO

#
# Path(_nodes) gives a path that has the component
# of the gradient orthogonal to the tangent below sqrt(0.08)
# (significantly smaller at the path nodes):
#
_nodes = loadtxt(StringIO(_text))

def show_path(p, n=10):
    from numpy import linspace

    ch = array([ p(t) for t in linspace(0., 1., n)])
    print "ch=", ch

    show_chain(ch, style="g-")

def show_chain(p=None, style="ro-", save=None, clear=False):
    from pylab import hold, contour, plot, xlim, ylim, show, savefig, clf #, imshow
    from numpy import linspace, empty, transpose

    # intervals:
    x_range = (-1.2, 0.8)
    y_range = (-0.2, 1.8)

    # create grid:
    n = 100
    xs = linspace(x_range[0], x_range[1], n)
    ys = linspace(y_range[0], y_range[1], n)

    zs = empty((n, n))
    for i in range(n):
        for j in range(n):
            zs[i, j] = MB([xs[i], ys[j]])

    # when displayed, x and y are transposed:
    zs = transpose(zs)

    # dont know what it does:
    # hold(True)
    if clear: clf()

#   # Plotting color map:
#   imshow(zs, origin='lower', extent=[-1, 1, -1, 2])

    # Plotting contour lines:
    contour(zs, 100, origin='lower', extent=(x_range + y_range))

    # ideal path:
    plot(_nodes[:, 0], _nodes[:, 1], "k--")

    # three minima, and two TSs:
    points = array(CHAIN_OF_STATES)

    # overlay positions of minima and stationary points
    # onto coutour plot:
    plot(points[:, 0], points[:, 1], "ko")

    # overlay a path onto coutour plot:
    if p is not None:
        p = asarray(p)
        plot(p[:, 0], p[:, 1], style)

    ylim(*y_range)
    xlim(*x_range)

    if save is None:
        show()
    else:
        savefig(save)

# python mueller_brown.py [-v]:
if __name__ == "__main__":
    import doctest
    doctest.testmod()

# Default options for vim:sw=4:expandtab:smarttab:autoindent:syntax

