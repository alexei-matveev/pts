#!/usr/bin/env python

"""
Example use: define a few nodes, here four nodes in 2D, the same as in
path.py:

    >>> from numpy import asarray, arctan2, pi
    >>> nodes = asarray(((-100., -100.), (0., -50.), (0., 50.), (100., 100.)))

                                           + + B
                                       + +     (100, 100)
                                   + +
                               + +
                           + +
                         +
                        X
                       +(0, 50)
                       +
                       +
                        +
                        (0, 0)
                          +
                          +
                          +
                        X
                      + (0, -50)
                  + +
              + +
          + +
    A + +
    (-100, -100)

This  illustrates "arbitrarness"  in the  definition of  the tangents,
basically there is  more than one way to plot  a "reasonable" path and
the corresponding tangents:

    >>> asarray(tangent1(nodes))
    array([[ 0.89442719,  1.4472136 ],
           [ 0.89442719,  1.4472136 ]])

This is the angle in grads:

    >>> arctan2(1.4472136, 0.89442719) / pi * 180
    58.282525696859388

    >>> asarray(tangent2(nodes))
    array([[ 100.,  150.],
           [ 100.,  150.]])

    >>> arctan2(150., 100.) / pi * 180
    56.309932474020215

    >>> asarray(tangent3(nodes))
    array([[  50.,  275.],
           [  50.,  275.]])

    >>> arctan2(275., 50.) / pi * 180
    79.69515353123397

    >>> asarray(tangent4(nodes))
    array([[  42.2291236 ,  297.50776405],
           [  42.2291236 ,  297.50776405]])

    >>> arctan2(297.50776405, 42.2291236) / pi * 180
    81.921237154537607

* Optimization of a descrete MEP with vertices selected by fixing
  reaction coordiante:

    >>> n = 5
    >>> from numpy import array

One equilibrium of Ar4  LJ cluster (in coordinates of c2v_tetrahedron1
Func:

    >>> w = 0.39685026

    >>> from test.testfuns import c2v_tetrahedron1
    >>> A = array([w, w, +w])

Another equilibrium:

    >>> B = array([w, w, -w])

Halfway between A and B:

    >>> C = (A + B) / 2.0

    >>> xs = array([A, C, B])

    >>> from path import MetricPath
    >>> from metric import Metric
    >>> from numpy import linspace

Here "z" is a 3 -> (4 x 3) function generating tetrahedron:

    >>> z = c2v_tetrahedron1()

This  buils  the PES  as  a  composition of  QFunc  as  a funciton  of
cartesian coordiantes and "z"-matrix:

    >>> from ase import Atoms
    >>> from qfunc import QFunc
    >>> from func import compose

    >>> pes = compose(QFunc(Atoms("Ar4")), z)

    >>> from rc import Volume
    >>> vol = compose(Volume(), z)

We will abuse MetricPath to  distribute "n" points along the path with
equal separations as measured by cartesian metric:

    >>> p = MetricPath(xs, Metric(z).norm_up)
    >>> x0 = map(p, linspace(0., 1., n))

Inital energies and values of reaction coordinate:

    >>> from numpy import round

    >>> round(map(pes, x0), 4)
    array([  -6.    ,   32.3409,  190.    ,   32.3409,   -6.    ])

    >>> round(map(vol, x0), 4)
    array([-1. , -0.5, -0. ,  0.5,  1. ])

    >>> x1, info = soptimize(pes, x0, tangent1, rc=vol)
    >>> info["iterations"]
    15

Optimized   energies   and  values   of   reaction  coordinate   after
optimization:

    >>> round(map(pes, x1), 4)
    array([-6.    , -4.5326, -4.4806, -4.5326, -6.    ])

    >>> round(map(vol, x1), 4)
    array([-1. , -0.5, -0. ,  0.5,  1. ])

Note  that  due to  symmetry  that  happened  to be  preserved  during
optimization  the  square TS  state  is too  high.   This  would be  a
symmetry distorted midpoint:

    >>> C = array([w + 0.1, w - 0.1, 0.0])

And the rombus TS is by about half a unit lower in energy:

    >>> x2, info = soptimize(pes, [A, C, B], tangent1, rc=vol)
    >>> pes(x2[1])
    -5.073420858462792

Resutls  of an  optimization  without constraints,  or  rather with  a
"dynamic" constraint  where a motion of  a vertex is  restricted to be
orthogonal  to  the  tangent,  will  depend on  details  of  the  path
relaxation algorithm:

    >>> x1, info = soptimize(pes, x0, tangent1)
    >>> info["iterations"]
    14

    >>> round(map(pes, x1), 4)
    array([-6.    , -4.8521, -4.4806, -4.8521, -6.    ])

    >>> round(map(vol, x1), 4)
    array([-1.   , -0.958, -0.   ,  0.958,  1.   ])

Note  that the  funciton soptimize()  does not  do any  "respacing" by
default.
"""

__all__ = []

from numpy import array, asarray, empty, dot, max, abs, sqrt, shape, linspace
from numpy import vstack, sign
from bfgs import BFGS, SR1, Array
from common import cumm_sum, pythag_seps
from metric import cartesian_norm

VERBOSE = 0

TOL = 1.e-6

XTOL = TOL   # step size tolerance
FTOL = 1.e-5 # gradient tolerance
CTOL = TOL   # constrain tolerance

MAXIT = 50
MAXSTEP = 0.05
SPRING = 100.0

from chain import Spacing, Norm2
# spacing = Spacing(Norm())
spacing = Spacing(Norm2())

def tangent1(X, norm=cartesian_norm):
    """For n geometries X[:] return n-2 tangents computed
    as averages of forward and backward tangents.
    """

    # T = []
    # for i in range(1, len(X) - 1):
    #     a = X[i] - X[i-1]
    #     b = X[i+1] - X[i]
    #     a /= norm(a, X[i]) # FIXME: (X[i] + X[i-1]) / 2.0 ?
    #     b /= norm(b, X[i]) # FIXME: (X[i] + X[i+1]) / 2.0 ?
    #     T.append(a + b)

    X = asarray(X)

    # pairwise differences along the chain:
    deltas = X[1:] - X[:-1]

    # middle points of each interval (used only if norm() makes use of
    # them):
    centers = (X[1:] + X[:-1]) / 2.0

    # normalize  tangents  before  averaging forward-  and  backfoward
    # tangent:
    T = array([ d / norm(d, x) for d, x in zip(deltas, centers)])

    # sum of a forward and backward normalized tangents:
    return T[:-1] + T[1:]

def tangent2(X):
    """For n geometries X[:] return n-2 tangents computed
    as central differences:

        T[i] = X[i+1] - X[i-1]
    """

    X = asarray(X)

    return X[2:] - X[:-2]

from path import Path

def tangent3(X):
    """For n geometries X[:] return n-2 tangents computed from a fresh
    spline interpolation.
    """

    s = linspace(0., 1., len(X))

    p = Path(X, s)

    return map(p.fprime, s[1:-1])

def tangent4(X):
    """For  n geometries  X[:]  return n-2  tangents  computed from  a
    different spline interpolation.
    """

    # abscissas of the nodes:
    s = cumm_sum(pythag_seps(X))
    s = s / s[-1]

    assert s[0] == 0.0
    assert s[-1] == 1.0

    p = Path(X, s)

    return map(p.fprime, s[1:-1])

def test(A, B, trafo=None):

    print "A=", A
    print "B=", B

    from pts.pes.mueller_brown import MB
    from pts.pes.mueller_brown import show_chain

    x = [A, B]

    # change coordinates:
    if trafo is not None:
        from func import compose
        MB = compose(MB, trafo)
        x = array(map(trafo.pinv, x))

    def show(x):
        if trafo is not None:
            show_chain(map(trafo, x))
        else:
            show_chain(x)

    from numpy import savetxt

    def callback(x, e, g):
        # show(x)
        # savetxt("path.txt", x)
        # print "chain spacing=", spacing(x)
        pass

    from path import MetricPath # for respacing
    from rc import Linear # as reaction coordinate
    rcoord = Linear([1., -1.])

    from metric import Metric
    mt = Metric(rcoord)

    n = 3
    n_max = 30
    while True:
        #
        # Respace vertices based on custom metric built from the
        # definition of reaction coordinate:
        #
        p = MetricPath(x, mt.norm_up)
        x = array(map(p, linspace(0., 1., n)))

        print "BEFORE, rc(x)=", map(rcoord, x)
        show(x)

        # x = respace(x, tangent4, spacing)

        # print "RESPACE, x=", x
        # print "spacing(x)=", spacing(x)
        # show(x)

#       x, stats = soptimize(MB, x, tangent1, spacing, maxit=20, maxstep=0.1, callback=callback)
#       x, stats = soptimize(MB, x, tangent4, maxit=20, maxstep=0.1, callback=callback)
        x, stats = soptimize(MB, x, tangent4, rc=rcoord, maxit=20, maxstep=0.1, callback=callback)
        savetxt("mb-path.txt-" + str(len(x)), x)

        print "AFTER, rc(x)=", map(rcoord, x)
        show(x)

        if n < n_max:
            # double the number of beads:
            n = 2 * n + 1
        else:
            print "========================================================="
            print "Conveged for the maximal tested number of beads: ", n
            print "========================================================="
            break

from func import Reshape, Elemental
from memoize import Memoize, DirStore

def soptimize(pes, x0, tangent=tangent1, rc=None, constraints=None, pmap=map, callback=None, **kwargs):
    """
    Several choices for pmap argument to allow for parallelizm, e.g.:

        from paramap import pmap

    for parallelizm in general, or

        from qfunc import qmap

    for parallelizm and chdir-isolation of QM calculations.

    """

    if rc is not None and constraints is not None:
        assert False, "Provide either reaction coordinate, local, or collective constraints."

    n = len(x0)
    assert n >= 3

    x0 = asarray(x0)

    # string (long) vector shape:
    xshape = x0.shape

    # position (short) vector shape:
    vshape = x0[0].shape

    # position (short) vector dimension:
    vsize  = x0[0].size

    #
    # sopt() driver deals only with the collections of 1D vectors, so
    #
    # (1) reshape the input ...
    x0 = x0.reshape(n, -1)

    # (2) "reshape" provided PES Func to make it accept 1D arrays:
    pes = Reshape(pes, xshape=vshape)

    #
    # For restarts and post-analysis. FIXME:  we keep it here for some
    # time for  accessing cached  result of the  PES as a  function of
    # internal coordinates. Otherwise it must die in favor of cache.d:
    #
    pes = Memoize(pes, DirStore("soptimize.d"))

    # (3) if present, "reshape" the constraint Func:
    if constraints is not None:
        # here we assume n-2 constraints, one per moving image,
        # the terminal images are fix anyway:
        constraints = Reshape(constraints, xshape=xshape, fshape=(n-2,))

    # (4) make PES function elemental, allow for parallelizm:
    pes = Elemental(pes, pmap)
    # FIXME: should we request the caller to provide "elemental" PES?

    # FIXME: available tangent definitions already expect/accept
    # (a group of) 1D vectors. Reshape them too, if needed.

    tangents = wrap1(tangent, x0[0], x0[-1])

    #
    # The function "lambdas" is used to compute lagrange multipliers and is
    # expected to adhere to specific interface.
    #
    # By default Lagrange multipliers ensure displacements are orthogonal to
    # tangents (lambda1):
    #
    lambdas = lambda1

    #
    # If global  (collective) constraints  are provided, use  those to
    # define "lambdas".  So far Collective constraints is a length N-2
    # vector valued function of all N vertices, including the terminal
    # ones.   This  makes   them  suitable   for  solving   a  typical
    # optimization problem when the terminal beads are frozen.
    #
    if constraints is not None:
        # real constraints require also the terminal beads:
        constraints = wrap2(constraints, x0[0], x0[-1])

        # prepare function that will compute the largangian
        # factors for this particular constraint:
        lambdas = mklambda3(constraints)

    #
    # If local constraints are provided, use them to define lambdas. A
    # local  constraint is  a  (differentiable) function  of a  single
    # vertex. We need at least one such function per vertex. Currently
    # it is  assumed that  the terminal vertices  are frozen,  so that
    # their local constraints are ignored.
    #
    if rc is not None:
        # (2) "reshape" provided PES Func to make it accept 1D arrays:
        rc = Reshape(rc, xshape=vshape)

        local = [rc.taylor] * len(x0)

        lambdas = mklambda0(local[1:-1])

    def cb(x, e, g, t, lam):
        if callback is not None:
            #
            # We were obliged to report every iteration, unfortunately
            # we  have  to  report  the  info  about  terminals  beads
            # too. This massaging is supposed to do that:
            #
            assert len(vshape) == 1 # FIXME: generalize!
            x2 = vstack((x0[0], x, x0[-1]))

            # add fake tangents for terminal vertices:
            t2 = vstack((x[0] - x0[0], t, x0[-1] - x[-1]))

            #
            # Hopefully the function is cheap or uses a result cache:
            #
            e2, g2 = pes.taylor(x2)
            e2 = asarray(e2)
            g2 = asarray(g2)
            callback(x2, e2, g2, t2)

    xm, info = sopt(pes.taylor, x0[1:-1], tangents, lambdas, callback=cb, **kwargs)

    # put the terminal images back:
    xm = vstack((x0[0], xm, x0[-1]))

    #
    # In  this scope  PES is  memoized,  compute the  energies of  the
    # (hopefully)  converged chain  including terminal  vertices. Note
    # that  terminals  are not  treated  by  sopt()  but the  info  is
    # expected  by the  caller.  Also  note that  sopt()  delivers the
    # corresponding  info from  the last  iteration and  only  for the
    # moving vertices:
    #
    energies, gradients = pes.taylor(xm)

    energies = asarray(energies)
    gradients = asarray(gradients)

    # info = {"iterations": iteration + 1,
    #         "converged": converged,
    #         "energies": E,
    #         "gradients": G,
    #         "tangents": T,
    #         "lambdas": LAM,
    #         "step": dR}

    xm.shape = xshape

    info["energies"] = energies
    info["gradients"] = gradients

    assert len(vshape) == 1 # FIXME: generalize!

    info["step"] = vstack((zeros(vshape), info["step"], zeros(vshape)))

    # FIXME:  need a  convention for  the terminal  tangents, lambdas,
    # until then delete incomplete info:
    del info["tangents"]
    del info["lambdas"]

    return xm, info

def sopt(fg, X, tangents, lambdas=None, xtol=XTOL, ftol=FTOL,
         maxit=MAXIT, maxstep=MAXSTEP, alpha=70., callback=None,
         **kwargs):
    """
    |fg| is supposed to be an elemental function that returns a tuple

        fg(X) == (values, derivatives)

    with values and derivatives  being the arrays/lists of results for
    all x in  X. Note, that only the  derivatives (gradients) are used
    for optimization.

    Kwargs accomodates all arguments  that are not interpreted by this
    sub but are otherwise used in other optimizers.
    """

    if VERBOSE:
        print "sopt: xtol=", xtol, "ftol=", ftol, "maxit=", maxit, "maxstep=", maxstep
        if len(kwargs) > 0:
            print "sopt: ignored kwargs=", kwargs

    def norm(x):
        # "L-infinity norm"
        # return max(abs(x))
        "L2 norm"
        return cartesian_norm(x, None)

    # init array of hessians:
    H = Array([ BFGS(alpha) for _ in X ])
    B = Array([ SR1(alpha) for _ in X ])

    # geometry, energy and the gradient from previous iteration:
    R0, G0 = None, None

    # initial value for the variable:
    R = asarray(X)

    iteration = -1 # prefer to increment at the top of the loop
    converged = False

    # fixed trust radius:
    TR = maxstep

    while not converged and iteration < maxit:
        iteration += 1

        # To be  able to distinguish  convergence by step, or  by gradient
        # count the number of satisfied criteria:
        criteria = 0

        if VERBOSE:
            print "sopt: =============== Iteration ", iteration, " ==============="
            if VERBOSE > 1:
                print "sopt: scheduling gradients for R="
                print R

        # compute energy and gradients at all R[i]:
        E, G = fg(R)

        # FIXME: better make sure gradients are returned as arrays:
        G = asarray(G)

        #
        # Need tangents just for convergency check and RPH:
        #
        T = tangents(R)

        #
        # Update the hessian representation:
        #
        if iteration == 0: # only later R0 and G0 become meaningfull!
            pass
        else:
            H.update(R - R0, G - G0)
            B.update(R - R0, G - G0)

        #
        # Convergency checking, based on gradients (ftol) ... {{{
        #

        # lagrange multipliers (in simple case components of gradients parallel
        # to the tangents):
        LAM = lambdas(R, G, H, T)

        # remaining gradient for CURRENT geometry just for convergency check:
        G2 = empty(shape(G)) # move out of the loop
        for i in xrange(len(G)):
            G2[i] = G[i] - LAM[i] * T[i]

        if norm(G2) < ftol:
            # FIXME: this may change after update step!
            criteria += 1
            if VERBOSE:
                print "sopt: converged by force", norm(G2), '<', ftol, "(", norm.__doc__, ")"

        if VERBOSE:
            print "sopt: obtained energies E=", asarray(E)
            if VERBOSE > 1:
                print "sopt: obtained gradients G="
                print G
            print "sopt: g(para)=", LAM, "(lambdas)"
            print "sopt: g(ortho norms)=", asarray([sqrt(dot(g, g)) for g in G2])
            print "sopt: g(ORTHO NORM)=", norm(G2)
            if VERBOSE > 1:
                print "sopt: g(ortho)="
                print G2

        # These were  used for convergency  check, used below  only to
        # report  additional info  upon  convergence:
        del G2 # T, LAM
        # ... done convergency check }}}

        #
        # step(h) with h in [0, 1]  is an estimate of the step towards
        # the stationary  point.  Note  that there might  be numerical
        # problems computing the  value and (especialy) the derivative
        # of that funciton at h close to 1.0:
        #
        step = Step(R, G, B, H, tangents, lambdas)
        # step = Step1(R, G, B, H, tangents, lambdas)

        #
        # This either returns h  = 1.0 if a step is not  too long or a
        # smaller value such that norm(step(h)) <= TR:
        #
        h = scale(step, norm, TR)

        #
        # This actually computes the step:
        #
        dR = step(h)

        if VERBOSE:
            print "sopt: ODE step, propose h=", h, "norm(dR)=", norm(dR), "(", norm.__doc__, ")"
            print "sopt: dR=\n", dR

        #
        # Convergency check by step size (xtol) ...
        #
        if norm(dR) < xtol:
            criteria += 1
            if VERBOSE:
                print "sopt: converged by step norm(dR)", norm(dR), '<', xtol, "(", norm.__doc__, ")"

        #
        # Verify the  step length  one last time,  in case all  of the
        # magic above breaks:
        #
        length = norm(dR)
        if length > TR:
            print "sopt: WARNING: step too long by factor", length/TR, ", scale down !!!"

        #
        # The info  we pass  to the caller,  should be  consistent, so
        # invoke  callback before  updating the  state vector,  as the
        # energies, gradients, tangents and lambdas correspond to this
        # state:
        #
        if callback is not None:
            callback(R, E, G, T, LAM)

        # Save for later  comparison. E, G = fg(R)  will re-bind E and
        # G, not modify them:
        R0, G0 = R, G

        # actually update position vector:
        R = R + dR

        # See while  loop condition. We  are only converged  when both
        # criteria are satisfied:
        if criteria >= 2:
            converged = True

    #
    # We are outside of the loop again:
    #
    if VERBOSE:
        if iteration >= maxit and not converged:
            print "sopt: exceeded number of iterations", maxit

    #
    # Also return number of  interations, convergence status, and last
    # values of the energies, gradients  and step. Note again that the
    # energies,  gradients, tangents,  and lambdas  correspond  to the
    # last geometry used for convergency check.  The returned geometry
    # differs   from   that   by    dR.    You   need   to   recompute
    # energies/gradients  and such  if  you want  them  for the  final
    # geometry.  This is left to the caller.
    #
    info = {"iterations": iteration + 1,
            "converged": converged,
            "energies": E,
            "gradients": G,
            "tangents": T,
            "lambdas": LAM,
            "step": dR}

    return R, info

def scale(step, norm, TR):
    """
    For a "curve" step(h) with h in [0, 1] find a suitable h such that

        norm(step(h)) <= TR

    Was a piece of sopt() code, extracted here for brevity.
    """

    # trust(h) > 0 means we would accept the step(h):
    def trust(h):
        nrm = norm(step(h))

        if VERBOSE:
            if nrm > TR:
                word = "too long"
            else:
                word = "looks OK"
            print "sopt: step(", h, ")", word, "by factor", nrm/TR, "of trust radius", TR

        return TR - nrm

    #
    # We are looking  for a point h in [0,  1] such that |step(h)|
    # == TR, if there is no such point then h should be 1. This is
    # the initial interval:
    #
    ha, hb = 0.0, 1.0

    #
    # As  an optimization  the first  thing we  do is  proposing a
    # point  hoping that  in the  "criminal" cases  of  very large
    # steps  (TR  << dmax)  we  can  avoid  integrating ODE  to  h
    # significantly  larger than  TR /  dmax.  So  here we  try to
    # guess a more conservative upper bound of the interval.
    #
    # A rough  estimate of the step  follows, this is  cheap as it
    # does not involve ODE integration:
    #
    dmax = norm(1.0 * step.fprime(0.0))
    if dmax > TR:
        h = TR / dmax
    else:
        h = 1.0
    #
    # Of course if it is not  an upper bound, then it is new lower
    # bound. In any case we restric the search the interval to [0,
    # h] or [h, 1] in the first iteration of the following loop:
    #
    while True:
        if trust(h) <= 0.0:
            ha, hb = ha, h
        else:
            ha, hb = h, hb

        if hb - ha > TOL:
            # FIXME: since here  we cannot exclude that h  == 1, we should
            # better not use the derivatves, just use bisection:
            h = (ha + hb) / 2.0
        else:
            break

    assert hb == 1.0 or trust(ha) > 0.0 > trust(hb)

    # prefer the left side so  that |step(h)| < TR, unless we can
    # also trust(hb). In this case hb == 1.
    if trust(hb) > 0:
        h = hb
    else:
        h = ha

    assert h != 0.0, "must be rare, see how often"

    return h

def respace(x0, tangents, spacing):

    assert len(x0) >= 3

    x0 = asarray(x0)

    # tangents need terminal beads:
    tangents = wrap1(tangents, x0[0], x0[-1])

    # spacing also requires terminal beads:
    lambdas = wrap1(spacing, x0[0], x0[-1])

    xm = resp(x0[1:-1], tangents, lambdas)

    # put the terminal images back:
    xm = vstack((x0[0], xm, x0[-1]))

    return xm

from ode import ODE, limit

def resp(x, t, s):
    """
        dx / dh = - t(x) * s(x)
    """

    def f(h, y):
        T = t(y)
        S = s(y)
        yprime = empty(shape(y))
        for i in range(len(y)):
            # FIXME: yes, with PLUS sign:
            yprime[i] = T[i] * S[i]
        return yprime

    print "resp: s(x0)=", s(x)

    #
    # This solves for X(t) such that dX / dt = f(t, X(t)), with X(0) =
    # x:
    #
    X = ODE(0.0, x, f)

    #
    # This computes X(inf):
    #
    x8 = limit(X)

    print "resp: s(x8)=", s(x8)
    return x8

from numpy import log, zeros
from func import Func

class Step(Func):
    """
    A function of a scalar scale  parameter "h" in the interval [0, 1]
    that implements a "step towards the stationary solution". Think of
    a scaled Newton step dx = - h * H * g towards the stationary point
    but keep  in mind that due  to constraints and  path specifics the
    way  towards the stationary  point is  not necessarily  a straight
    line like is implied by a newton formula.

    To get a rough estimate of the (remaining) step one could use

        step = Step(X, G, H, H, tangents, lambdas)
        dX = (1.0 - h) * step.fprime(h)

    which  for a  special  case  of h  =  0 does  not  involve no  ODE
    integration.
    """
    def __init__(self, X, G, B, H, tangents, lambdas):
        #
        # Function to integrate (t is "time", not "tangent"):
        #
        #   dg / dt = f(t, g)
        #
        def f(t, g):
            return gprime(t, g, H, G, X, tangents, lambdas)
        #
        # This Func can  integrate to T for any T  in the interval [0,
        # infinity).   To get  an  idea, very  approximately for  step
        # scale h = 1.0  the change of a gradients is: G1  = G + 1.0 *
        # f(0.0, G). Also we will need these H and G later:
        #
        self._slots = ODE(0.0, G, f), H, G

    def taylor(self, h):

        ode, H, G = self._slots

        #
        # Upper integration limit T (again "time", not "tangent)":
        #
        if h < 1.0:
            #
            # Assymptotically the gradients decay as exp[-t]
            #
            T = - log(1.0 - h)
            G8, G8prime = ode.taylor(T)

            # Use one-to-one relation between dx and dg:
            dX = H.inv(G8 - G)
            dXprime = H.inv(G8prime) / (1.0 - h)
        else:
            G8 = limit(ode)

            # Use  one-to-one  relation  between  dx  and  dg.  FIXME:
            # how to approach limit(self.ode.fprime(T) * exp(T))?
            dX = H.inv(G8 - G)
            dXprime = None

        return dX, dXprime

class Step1(Func):
    """
    A function of a scalar scale  parameter "h" in the interval [0, 1]
    that implements a "step towards the stationary solution". Think of
    a scaled Newton step dx = - h * H * g towards the stationary point
    but keep  in mind that due  to constraints and  path specifics the
    way  towards the stationary  point is  not necessarily  a straight
    line like is implied by a newton formula.

    To get a rough estimate of the (remaining) step one could use

        step = Step(X, G, B, H, tangents, lambdas)
        dX = (1.0 - h) * step.fprime(h)

    which  for a  special  case  of h  =  0 does  not  involve no  ODE
    integration.
    """

    def __init__(self, X0, G0, B, H, tangents, lambdas):
        #
        # Function to integrate (h is "evolution time"):
        #
        #   dx / dh = f(h, x)
        #
        def xprime(h, x):
            """For the descent procedure return

              dx / dh = - H * ( g(x0 + x)  - lambda * t(x0 + x) )

            The Lagrange contribution parallel to the tangent t(x) is computed
            by the function lambdas(X, G, H, T).

            The gradient g(x) has the following relation to the coordinats:

                g - g0 = B * x

            with B being the inverse of H.

            Note that  this is equivalent  to the descent procedure

              dy / dh = - ( y  - lambda(y) * t(y) )

            for local coordinates  y (aka gradient g) defined  by a one-to-one
            relation:

              (x - x0) = H * (y - y0)

            By  now y  is the  same  as the  gradient g.  Though the  positive
            definite  hessian  H may  distrub  this  equivalence  at some  PES
            regions.

            The  current  form  of  xprime()  may be  used  to  either  ensure
            orthogonality of  dx /  dh to the  tangents or preserve  the image
            spacing  depending on  the definition  of function  lambdas() that
            delivers lagrangian factors.  I am afraid one cannot satisfy both.

            NOTE: imaginary time variable "h" is not used anywhere.
            """

            # G = G(X), here B should represent PES well:
            G = G0 + B.app(x)

            # T = T(X(G)):
            T = tangents(X0 + x)

            # Compute Lagrange factors, here H should be non-negative:
            LAM = lambdas(X0 + x, G, H, T)

            #
            # Add lagrange forces, only  component parallel to the tangents is
            # affected:
            #
            G2 = empty(shape(G))
            for i in xrange(len(G)):
                G2[i, ...] = G[i] - LAM[i] * T[i]

            # here H should be non-negative:
            return -H.inv(G2)
            # return dX1

        #
        # This Func can  integrate to T for any T  in the interval [0,
        # infinity).   To get  an  idea, very  approximately for  step
        # scale h = 1.0  the change of a gradients is: G1  = G + 1.0 *
        # f(0.0, G). Also we will need these H and G later:
        #
        self.ode = ODE(0.0, zeros(shape(X0)), xprime)

    def taylor(self, h):
        #
        # Upper integration limit T (again "time", not "tangent)":
        #
        if h < 1.0:
            #
            # Assymptotically the gradients decay as exp[-t]
            #
            T = - log(1.0 - h)
            X, Xprime = self.ode.taylor(T)

            return X, Xprime / (1.0 - h)
        else:
            X = limit(self.ode)

            # FIXME:   how  to  approach   limit(self.ode.fprime(T)  *
            # exp(T))?
            return X, None

from numpy.linalg import solve

def gprime(h, G, H, G0, X0, tangents, lambdas):
    """For the descent procedure return

      dy / dh = - ( y  - lambda(y) * t(y) )

    The Lagrange contribution parallel to the tangent t(y) is added if
    the function "lambdas()" is provided.

    Procedure uses the one-to-one relation between coordinates

      (x - x0) = H * (y - y0)

    to compute the tangents at x:

      t(x) = t(x(y))

    By  now y  is the  same  as the  gradient G.  Though the  positive
    definite  hessian  H may  distrub  this  equivalence  at some  PES
    regions.

    The current form of gprime() translated to real space variables

        dx / dh = H * dg / dh

    may  be used  to either  ensure orthogonality  of dx  / dh  to the
    tangents or preserve the image spacing depending on the definition
    of function  "lambdas()" that  delivers lagrangian factors.   I am
    afraid one cannot satisfy both.

    NOTE: imaginary time variable "h" is not used anywhere.
    """

    # X = X(G):
    X = X0 + H.inv(G - G0)

    # T = T(X(G)):
    T = tangents(X)

    # compute lagrange factors:
    LAM = lambdas(X, G, H, T)

    # add lagrange forces, only components parallel
    # to the tangents is affected:
    dG = empty(shape(G))
    for i in xrange(len(G)):
        dG[i] = G[i] - LAM[i] * T[i]

    return -dG

def vectorize(f):
    """Array-version of f.  Used  to build LOCAL lagrange factors. For
    any  function   f()  opearting  on  a  few   arguments  return  an
    "elemental" function F, e.g.:

    def F(X, G, H, T):
        return map(f, X, G, H, T)

    FIXME: see if numpy.vectorize can be used
    """

    def _f(*args):
        return map(f, *args)

    return _f

def mklambda0(C):
    """
    Returns a  function that generates lagrangian "forces"  due to the
    differentiable constraints C from

        (X, G, H, T) == (geometry, gradient, hessian, tangents)

    Here constraints C is a list of local constraints, each a function
    of the corresponding vertex.   This type of constraints is usefull
    to enforce local properties of  vertices, such as a bond length or
    an  angle. Within  this  file it  is  assumed that  differentiable
    funcitons  are implemented  as functions  returning a  tuple  of a
    value and first derivatives.
    """

    def _lambda0(X, G, H, T):

        #
        # Evaluate  constraints  and  their derivatives.   Values  are
        # neglected   as  we  assume   the  vertices   satisfy  target
        # constraints  already.    This  is  more   like  "preserving"
        # properties  rather than  "seeking" target  values  (cp.  the
        # NEB-style springs in mklambda4):
        #
        A = [a for _, a in [c(x) for c, x in zip(C, X)]]

        # for historical reasons the main code is here:
        return map(lambda0, X, G, H, T, A)

    return _lambda0

def lambda0(x, g, h, t, a):
    """
    Compute  Lagrange  multiplier  to  compensate  for  the  constrain
    violation that would occur if the motion would proceed along

        dx = - h * g.

    In other words, find the lagrange factor lam such that

        dx = - h * (g - lam * t)

    is "orthogonal" to a

        a' * dx = 0.

    That is solve for lam in

        lam * (a' * h * t) = (a' * h * g).

    Input:
        x -- geometry
        g -- gradient
        h -- hessian
        t -- tangent
        a -- constraint derivatives
    """

    #
    # This would appliy the (inverse) hessian twice:
    #
    # lam = dot(a, h.inv(g)) / dot(a, h.inv(t))
    #

    # This applies hessian once:
    ha = h.inv(a)

    #
    # FIXME: division by zero and  small numbers occur here when it is
    #        becoming "increasingly difficult" to "fix" the constraint
    #        by applying tangential forces.
    #
    lam = dot(ha, g) / dot(ha, t)

    return lam

@vectorize
def lambda1(x, g, h, t):
    """Find the lagrange factor lam such that

        dx = - h * (g - lam * t)

    is orthogonal to t

        t' * dx = 0

    That is solve for lam in

        lam * (t' * h * t) = (t' * h * g)

    Input:
        x -- geometry
        g -- gradient
        h -- hessian
        t -- tangent

    FIXME: needs more general definition of "orthogonality"
    """

    #
    # The "constraint" to be satisfied is "no displacement along the
    # tangent". FIXME: with a general metric the second "t" would need
    # to be converted to covariant coordinates (aka lower index):
    #
    return lambda0(x, g, h, t, t)

@vectorize
def lambda2(x, g, h, t):
    """Find the lagrange factor lam such that

        g   =  g - lam * t
         bot

    is orthogonal to t

        t' * g   =   0
              bot

    That is solve for lam in

        lam * (t' * t) = (t' * g)

    Input:
        x -- geometry
        g -- gradient
        h -- hessian
        t -- tangent

    FIXME: needs more general definition of "orthogonality"
    """

    lam = dot(t, g) / dot(t, t)

    return lam

def mklambda3(constraints):
    """
    Returns a  function that generates lagrangian "forces"  due to the
    differentiable constraints from

        (X, G, H, T) == (geometry, gradient, hessian, tangents)

    Here "constraints" are global constraints, that is a vector valued
    differentiable function of all  vertices. This type of constraints
    is  usefull to  enforce  "collective" constraints,  such as  equal
    pairwise distances between neightboring vertices.
    """

    def lambda3(X, G, H, T):

        #
        # Evaluate  constraints  and  their derivatives.   Values  are
        # neglected   as  we  assume   the  vertices   satisfy  target
        # constraints  already.    This  is  more   like  "preserving"
        # properties  rather than  "seeking" target  values  (cp.  the
        # NEB-style springs in mklambda4):
        #
        c, A = constraints(X)

        # for historical reasons the main code is here:
        return glambda(G, H, T, A)

    return lambda3

def mklambda4(springs, k=SPRING):
    """Returns a function that generates tangential "forces" due
    to the springs from provided arguments:

        (X, G, H, T) == (geometry, gradient, hessian, tangents)
    """

    def lambda4(X, G, H, T):

        # these are parallel components of gradients, to be removed:
        lam2 = lambda2(X, G, H, T)

        #
        # Evaluate deviations from  equilibrium. Here, the derivatives
        # are  ignored, rather  the values  of the  "deformations", if
        # different from zero, are  used to add tangential "forces":
        #
        c, _ = springs(X)

        # instead add spring forces:
        lam4 = asarray(c) * k

#       print "LAM (2) =", lam2
#       print "LAM (4) =", lam4

        return lam2 + lam4

    return lambda4

def glambda(G, H, T, A):
    """
    Compute  Lagrange  multipliers  to  compensate for  the  constrain
    violation that would occur if the motion would proceed along

        dX = - H * G.

    Lagrange  multipliers   LAM  are  supposed  to  be   used  to  add
    contributions  PARALLEL to  the  tangents and  thus, redefine  the
    default direction:

        G := G - LAM * T   (no sum over path point index i)
         i    i     i   i

    This amounts to  solving the system of N equations  for LAM with N
    being the number of constraints (equal to the number of points)

        A * dX = 0,  k = 1..N
         k

    or, more explicitly

        A * H * ( G - LAM * T ) = 0
         k
    """

    # number of constraints:
    n = len(A)

    # FIXME: number of degrees of freedom same as number of
    #        constraints:
    assert len(T) == n

    # dx / dh without constraints would be this:
    xh = H.inv(-G)

    # dc / dh without constraints would be this:
    ch = zeros(n)
    for i in xrange(n):
        for j in xrange(n):
            ch[i] += dot(A[i, j], xh[j])
            # FIXME: place to use npz.matmul() here!

    # unit lagrangian force along the tangent j would change
    # the constraint i that fast:
    xt = H.inv(T)
    ct = zeros((n, n))
    for i in xrange(n):
        for j in xrange(n):
            ct[i, j] = dot(A[i, j], xt[j])

    # Lagrange factors to fullfill constains:
    lam = - solve(ct, ch) # linear equations
    # FIXME: we cannot compensate constraints if the matrix is singular!

    return lam

def wrap1(tangents, A, B):
    """A decorator for the tangent function tangents(X) that appends
    terminal points A and B before calling it.
    """
    def _tangents(X):
        Y = vstack((A, X, B))
        return tangents(Y)
    return _tangents

def wrap2(spacing, A, B):
    """Constrains on the chain of states based on state spacing.
    """

    def _constr(X):
        Y = vstack((A, X, B))

        # NOTE: spacing for N points returns N-2 results and its
        # N derivatives:
        c, cprime = spacing.taylor(Y)

        # return derivatives wrt moving beads:
        return c, cprime[:, 1:-1]

    return _constr

def test1(n):
    from numpy import array

    # One   equilibrium  of   Ar4  LJ   cluster  (in   coordinates  of
    # c2v_tetrahedron1 Func):
    w = 0.39685026
    A = array([w, w, +w])

    # Another equilibrium:
    B = array([w, w, -w])

    # Halfway between A and B:
    C = (A + B) / 2.0
    C = array([w + 0.01, w - 0.01, 0.0])

    xs = array([A, C, B])

    from test.testfuns import c2v_tetrahedron1, diagsandhight
    from path import MetricPath
    from metric import Metric
    from numpy import linspace

    z = c2v_tetrahedron1()

    # z = diagsandhight()
    # r = 1.12246195815
    # A = array([r, r,  r / sqrt(2.)])
    # B = array([r, r, -r / sqrt(2.)])
    # C = array([r, r * sqrt(2.), 0.])
    # xs = array([A, C, B])

    p = MetricPath(xs, Metric(z).norm_up)

    x0 = map(p, linspace(0., 1., n))

    from ase import Atoms
    from qfunc import QFunc
    from func import compose

    pes = compose(QFunc(Atoms("Ar4")), z)

    from rc import Volume

    vol = compose(Volume(), z)

    def callback(x, e, g):
        # from pts.tools.jmol import jmol_view_path
        print "energies=", e # map(pes, x)
        print "volume=", map(vol, x)
        # jmol_view_path(map(z, x), syms=["Ar"]*4, refine=1)
        pass

    print "BEFORE:"
    callback(x0, map(pes, x0), map(pes.fprime, x0))

    x1, info = soptimize(pes, x0, tangent1, rc=vol, callback=callback)
    # print "info=", info

    print "AFTER:"
    callback(x1, map(pes, x1), map(pes.fprime, x1))

    # from pts.tools.jmol import jmol_view_path
    # jmol_view_path(map(z, x1), syms=["Ar"]*4)

    # from scipy.linalg import eigh
    # from func import NumDiff
    # hess = NumDiff(pes.fprime, h=1.0e-5).fprime(x1[2])
    # e, V = eigh(hess)
    # from numpy import dot, diag
    # print "diff=", dot(hess, V) - dot(V, diag(e))
    # print "eigenvalues and eigenvectors:", e
    # print V

# python fopt.py [-v]:
if __name__ == "__main__":
    import doctest
    doctest.testmod()
#   import profile
#   profile.run("test1(5)")
#   exit()
    from pts.pes.mueller_brown import CHAIN_OF_STATES as P
    # from pts.test.testfuns import Affine
    # trafo = Affine([[2.0, 0], [0.0, 0.5]])
    # test(P[0], P[4]) #, trafo)

# Default options for vim:sw=4:expandtab:smarttab:autoindent:syntax
