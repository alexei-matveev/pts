#!/usr/bin/env python
"""
To run these tests, execute "python path.py".

Example use: define a few nodes, here four nodes in 2D:

    >>> path = ((-100., -100.), (0., -50.), (0., 50.), (100., 100.))

and  construct a path  connecting them  by default  with 4-equidistant
nodes

    >>> p = Path(path)

This builds  linear, quadratic or  spline representation of  the path,
depending on the number of nodes.

Now you can evaluate the path function at any point (between 0 and 1),
the equidistant values 0, 1/3, 2/3  and 1 correspond to the four nodes
we provided:

    >>> p(0)
    array([-100., -100.])

    >>> p(1)
    array([ 100.,  100.])

Rounding is only for tests to succeed despite the finite precision:

    >>> from numpy import round

    >>> round(p(1./3.), 10)
    array([  0., -50.])

    >>> round(p(2./3.), 10)
    array([  0.,  50.])

    >>> round(p(0.5), 10)
    array([ 0.,  0.])

"Calling"  a path  function as  above  is equivalent  to invoking  the
"value" method |f|:

    >>> p.f(0)
    array([-100., -100.])

In addition  the spline parametrization is used  to efficently compute
the  tangential of  the path.  Get  this by  calling the  "derivative"
method |fprime|:

    >>> p.fprime(0)
    array([ 650.,  -25.])

Using the length  of the tangential one can compute  the arc length of
the path section:

    >>> arc = Arc(p)
    >>> arc(0.0)
    0.0

    >>> arc(1.0)
    347.535794972545

    >>> arc(0.5)
    173.7678974862725

Using the reciprocal function |arg|  one may reparametrize the path in
units of arc-length:

    >>> arg = Inverse(arc)
    >>> arg(arc(0.5))
    0.5
    >>> arc(arg(173.76789748627249))
    173.7678974862725

    >>> round(p(arg(173.76789748627249)), 10)
    array([ 0.,  0.])

To properly compose the  path function p(x) and parametrization arg(s)
so that the derivatives of  the resulting function are also considered
one may do:

    >>> from func import compose
    >>> P = compose(p, arg)

    >>> round(P(173.76789748627249), 10)
    array([ 0.,  0.])

    >>> P.fprime(173.76789748627249)
    array([-0.07974522,  0.99681528])

The length of the resulting  tangent is identically one because of the
chosen  parametrization. Note, that  P is  not a  Path() but  rather a
Func() so that, for example, P.nodes will fail.

A similar functionality is provided by the MetricPath class:

    >>> p = MetricPath(path)

Generate equidistant points on the path

    >>> round(array(map(p, linspace(0.0, 1.0, 5))))
    array([[-100., -100.],
           [ -16.,  -84.],
           [  -0.,    0.],
           [  16.,   84.],
           [ 100.,  100.]])

    >>> from numpy import max, abs

Note the normalized path length parmeters of the original vertices are
not equally spaced:

    >>> ss, ys = p.nodes
    >>> ss
    array([ 0.        ,  0.35861009,  0.64138991,  1.        ])

    >>> max(abs(ys - array(path)))
    0.0

    >>> max(abs(ys - array(map(p, ss)))) < 2.0e-14
    True

Below we test the "nonpublic" part of the MetricPath() interface:

    >>> p.arc(0.0)
    0.0

    >>> p.arc(1.0)
    354.2485910296407

This number is slightly different from the one obtained earlier with a
Path(), 347.53579497254498, as  MetricPath() is different, though also
passes  through  the same  vertices.   The  difference  arises from  a
different choice of "abscissas". Same  holds for the arc length of the
half the path:

    >>> p.arc(0.5)
    177.12429551482035

The inverse of the |arc| method is the |arg| method that gives you the
spline-coordinate of  the point separated  from the origin by  any arc
length:

    >>> p.arg(p.arc(0.5))
    0.5

    >>> p.arc(p.arg(177.12429551482035))
    177.12429551482035

Derivative  with  respect to  the  normalized  path  parameter is  the
(unnormalized) tangent vector:

    >>> round(array(map(p.fprime, ss)))
    array([[ 352.,  -40.],
           [  50.,  351.],
           [  50.,  351.],
           [ 352.,  -40.]])

Verify correctness of derivatives at the original vertices:

    >>> from func import NumDiff
    >>> ts1 = array(map(p.fprime, ss))
    >>> ts2 = array(map(NumDiff(p).fprime, ss))
    >>> max(abs(ts1 - ts2)) < 1.0e-9
    True

There  is   a  limitation  in  the  library   that  implements  spline
fitting. The abscissas  need to be increasing. If  not, the Path() has
to work around that:

    >>> ys = array ([1, 2, 3, 4])
    >>> xs = array ([1, 0, -1, -2])

Here abscissas are not monotnically increasing, rather decreasing:

    >>> pp = Path (ys, xs)
    >>> max (abs (ys - map (pp, xs))) < 1.0e-14
    True

The slope was deliberately chosen constant:

    >>> max (abs ([pp.fprime (x) - (-1) for x in  xs])) < 1.0e-14
    True

Make sure we dont leak the modified abscissas:

    >>> x1, y1 = pp.nodes
    >>> (xs == x1).all()
    True
    >>> (ys == y1).all()
    True
"""

__all__ = ["Path", "MetricPath"]

from scipy.optimize import brentq as root

from numpy import array, arange
from numpy import asarray, empty, zeros, linspace

from func import LinFunc, QuadFunc, SplineFunc, Func
from func import Integral, Inverse

from common import pythag_seps, cumm_sum
from metric import cartesian_norm


def monotonic (xs):
    """
    >>> monotonic ([])
    True
    >>> monotonic ([1.0])
    True
    >>> monotonic ([1, 0])
    False
    >>> monotonic ([1, 2, 3, 3, 4])
    False
    """
    x0 = None
    for x in xs:
        if x0 is not None and x <= x0:
            return False
        x0 = x
    return True


class Path(Func):
    """
    Supports operations on a path  represented by a line, parabola, or
    a spline, depending on whether it has 2, 3 or > 3 points.

    Do not  abuse the fact that a  path can be mutated  by setting its
    nodes  to different  values,  it requires  the same  computational
    effort  as   constructing  a  new  Path  but   may  easily  become
    confusing. You  will probably not  like a Path  parametrization to
    suddenly  change just  because  a function  you  called with  this
    parametrization  choses  to  mutate  the  its  state  rather  than
    building a fresh one for personal use.

    Instead of

      p.nodes = xs, ys

    consider

      p = Path (ys, xs)

    """

    def __init__(self, ys, xs=None):

        if xs is None:
            # Generate initial paramaterisation density.  TODO: Linear
            # at present, perhaps change eventually
            xs = linspace (0.0, 1.0, len (ys))
        # Else take predefined node  abscissas.  They should be better
        # monotonically increasing.   The spline library  will bark if
        # not.

        # Node is a tuple of (x, y(x)). We use assignment here to test
        # the property handler (get/set_nodes). FIXME: we pass a tuple
        # of arrays, not an array of tuples:
        self.nodes = xs, ys

        # This generates linear/quadratic/spline representation of the
        # path  using the  normalized positions  chosen above  for the
        # respective points from the state vector.

        #
        # We   delegate  the   control  over   details  of   the  path
        # parametrization to function composition.  All of
        #
        #          Path(x), Path(x(s)), Path(x(s(w)))
        #
        # run along the same path albeit the "distance" along the path
        # is measured differently every time: in x, in s, or in w.
        #

    # The next two implement the interface of Func(),
    # however the path function is vector valued!
    def f(self, x):
        """
        Evaluates the path point  from the path parametrization.  Here
        the x is the *spline* argument, NOT the (normalized) length or
        weighted  length along  the path.   FIXME: To  use  other path
        parametrization   look  into   MetricPath   class.   Same   as
        __call__(self, x)
        """

        # Sign flip, eventually:
        s = self.__s

        # Evaluate   each   vector   component   by   calling   stored
        # parametrization:
        fs = array([f.f(s(x)) for f in self.__fs])

        # restore original shape, say (NA x 3):
        fs.shape = self.__yshape

        return fs

    def fprime(self, x):
        """
        Evaluates  the derivative  (unnormalized  tangential) wrt  the
        "spline" argument x
        """

        # Sign flip, eventually:
        s = self.__s

        # Evaluate   each   vector   component   by   calling   stored
        # parametrization of derivative:
        fprimes = array([s(f.fprime(s(x))) for f in self.__fs])

        # Restore original shape, say (NA x 3):
        fprimes.shape = self.__yshape

        return fprimes

    #
    #   self.__call__() equivalent to self.f() is inherited form Func()
    #

    def get_nodes(self):
        """
        Property handler, returns a tuple of arrays (xs, ys)
        """

        # Sign flip, eventually:
        s = self.__s

        # Abscissas:
        xs = array (map (s, self.__xs))

        # In the original shape:
        ys = self.__ys.reshape((self.__node_count,) + self.__yshape)

        return xs, ys

    def set_nodes(self, nodes):
        """
        Property handler, expects a tuple of arrays (xs, ys)
        """

        # Nodes is a tuple of abscisas and vectors:
        xs, ys = nodes

        assert len(xs) == len(ys)

        # number of path nodes that define a path:
        self.__node_count = len(ys)

        # |ys| is vector of arrays defining the path
        ys = array(ys)

        # Sign flip, eventually:
        if monotonic (xs):
            s = lambda x: x
        else:
            s = lambda x: -x

        xs = map (s, xs)
        assert monotonic (xs)

        self.__xs = array(xs)
        self.__s = s

        # save original shape of the input arrays:
        self.__yshape = ys[0].shape

        # internally we treat them flat, so care only about total size:
        self.__dimension = ys[0].size

        # treat internally each of |ys| as a flat array:
        self.__ys = ys.reshape(self.__node_count, self.__dimension)

        # Generate linear/quadratic/spline  representation of the path
        # using  the   normalized  positions  chosen   above  for  the
        # respective points from the state vector:
        self.__regen_path_func()

    nodes = property(get_nodes, set_nodes)

    def __regen_path_func(self):
        """
        Rebuild a  new path  function and the  derivative of  the path
        based on the contents of state_vec.
        """

        assert self.__node_count == len(self.__ys)
        assert self.__node_count == len(self.__xs)
        assert self.__node_count > 1

        self.__fs = []

        for i in range(self.__dimension):

            ys = self.__ys[:,i]
            assert self.__node_count == len(ys)

            # linear path
            if self.__node_count == 2:
                self.__fs.append(LinFunc(self.__xs, ys))

            # parabolic path
            elif self.__node_count == 3:

                self.__fs.append(QuadFunc(self.__xs, ys))

            else:
                # spline path
                self.__fs.append(SplineFunc(self.__xs, ys))


def scatter(rho, weights):
    """
    For  each  weight |w|  in  |weights| find  an  |s|  such that  the
    integral equation holds:

        W(s) / W(1) = w

    with
                  S
                 /
        W(S) =  | rho(s) ds
               /
               0

    This amount to inversion W(S) -> S(W).

        >>> wts = linspace(0.0, 1.0, 5)

    Usual density:

        >>> rho = lambda s: 1.0

        >>> scatter(rho, wts)
        array([ 0.  ,  0.25,  0.5 ,  0.75,  1.  ])

    Alternative implementations, see below:

        >>> scatter1(rho, wts)
        array([ 0.  ,  0.25,  0.5 ,  0.75,  1.  ])

        >>> scatter2(rho, wts)
        array([ 0.  ,  0.25,  0.5 ,  0.75,  1.  ])

    Unnormalized density:

        >>> rho = lambda s: 10.0

        >>> scatter(rho, wts)
        array([ 0.  ,  0.25,  0.5 ,  0.75,  1.  ])

        >>> scatter1(rho, wts)
        array([ 0.  ,  0.25,  0.5 ,  0.75,  1.  ])

        >>> scatter2(rho, wts)
        array([ 0.  ,  0.25,  0.5 ,  0.75,  1.  ])

    This is a piecewise defined density:

        >>> def rho(s):
        ...     if abs(s - 0.5) <= 0.25:
        ...         return 1.0
        ...     else:
        ...         return 0.0

    Solutions for W(S) = 0 or W(S) = W(1.0) are not unique,
    avoid asking for ambiguous solutions:

        >>> scatter(rho, wts[1:-1])
        array([ 0.375,  0.5  ,  0.625])

    Neither scatter1() nor scatter2() appear to work for
    non strictly positive densities rho.
    """

    # Weight of a string:
    weight = Integral(rho)

    # Scale  up  weights  so  that  they have  the  dimension  of  the
    # integral:
    weights = asarray(weights) * weight(1.0)

    arcs = empty(len(weights))

    for i, w in enumerate(weights):
        #
        # For each weight w solve the equation
        #
        #   weight(s) - w = 0
        #
        # without using  derivatives. Note that  the derivative, which
        # is  the  density rho(s)  may  not  be  continuous. Think  of
        # piecewise non-zero rho.
        #
        arcs[i] = root(lambda s: weight(s) - w, 0.0, 1.0)

    return arcs

def scatter1(rho, weights):
    """
    See scatter ...

    This one uses relies on Inverse of an Integral.  Implementation of
    Inverse uses the  newton method, and will have  problems if rho is
    not strictly positive.
    """

    # Weight of a string, W(S):
    weight = Integral(rho)

    # Arc length, S(W):
    arc = Inverse(weight)

    # Scale up weights so that they have the dimension of the integral:
    weights = asarray(weights) * weight(1.0)

    return array(map(arc, weights))

def scatter2(rho, weights):
    """
    See scatter ...

    Caluculate the function to integrate  for the path over at several
    points,  consider  this  function  to  be linear  in  between  and
    calculate with  trapez rule the  path-length then make  an inverse
    function of it and find the wanted points
    """
    from scipy.interpolate import splrep, splev

    # arbitrary number of points:
    NUM_POINTS = 100

    # equally spaced grid:
    x = linspace(0.0, 1.0, NUM_POINTS)

    # density at the grid:
    y = map(rho, x)

    # integral:
    w = zeros(len(y))
    for i in range(len(w)-1):
        w[i+1] = w[i] + 0.5 * (y[i+1] + y[i]) * (x[i+1] - x[i])

    # linear spline, x = x(w)
    spline = splrep(w, x, k = 1)

    # scale up weights so that they have the dimension of the integral:
    weights = asarray(weights) * w[-1]

    # evaluate spline:
    arcs = [splev(p, spline, der=0) for p in weights]

    return array(arcs)

class Arc(Integral):
    """
    Line integral of over path x(t):

                  t=T
                  /
        arc(T) = | dt * | dx/dt |
                /
                t=0

    with  the  length of  the  tangent  dx/dt  defined by  some  norm.
    Defaults to cartesian norm.
    """

    def __init__(self, x, norm=cartesian_norm):

        def sprime(t):
            X, Xprime = x.taylor(t)
            return norm(Xprime, X)

        Integral.__init__(self, sprime)

class MetricPath(Func):
    """
    This version uses normalized path length as path parameter. Though
    it accepts  arbitrary metric as  a definiton of length  the actual
    path,  due to  use  of  splines, is  not  invariant to  coordinate
    transformaiton.

    The  MetricPath is  immutable,  the nodes  of  the MetricPath  are
    read-only:

        p = MetricPath(ys, norm)
        xs, ys = p.nodes
    """

    def __init__(self, ys, norm=cartesian_norm):

        #
        # These numbers  will serve as a primary  path abscissas, they
        # correlate but  do not equate with the  path length. Consider
        # the  piecewise  linear zig-zag  path  and the  corresponding
        # lengths of its sections:
        #
        xs = cumm_sum(pythag_seps(ys, norm))

        xs = xs / xs[-1]
        assert xs[0] == 0.0
        assert xs[-1] == 1.0

        #
        # This  is  the  primary  path  parmatrization:  it  generates
        # linear/quadratic/spline representation of the path using the
        # abscissas chosen  above for  the respective points  from the
        # state vector:
        #
        p = Path(ys, xs)

        #
        # We   delegate  the   control  over   details  of   the  path
        # parametrization to function composition.  Both
        #
        #          Path(x) and  Path(x(s))
        #
        # run along the same path albeit the "distance" along the path
        # is measured  differently every time: in units  of abscissa x
        # or in path length s.

        #
        # Next, prepare the two transforamtions:
        #
        #       s = arc(x): spline argument -> arc length of the path
        #
        # and the reciprocal
        #
        #       x = arg(s): arc length of the path -> spline argument
        #
        # by integrating the length of the path tangent.
        #
        # The  derivative of  s(x) must  be  positive for  s(x) to  be
        # invertible.   This is the  case for  tangent length.   It is
        # assumed that s(0) = 0.
        #

        # Pass  integration  criteria  to  scipy.integrate.quad()  via
        # **kwargs if desired:
        arc = Arc(p, norm)
        arg = Inverse(arc)

        #
        # arc(x,a) evaluates the (cartesian) path length from the path
        # point  parametrized  by  spline-argument  |a| to  the  point
        # parametrized  by |x|.  NOTE:  this is  NOT the  general path
        # weight, just its (cartesian) length.
        #
        # The length of the tangential is by definition the derivative
        # of the path length wrt spline argument:
        #
        #  ds / dx = |dp / dx| > 0  =>  s(x)
        #
        # is a monotonic  (invertible) function.  arg(s) evaluates the
        # path coordinate (spline argument) from the path arc distance
        # from the  origin.  Note: x  = arg(s) is  inverse of the  s =
        # arc(x).  Similarly
        #
        #  dx  /  ds =  |dp  /  dx|^-1 >  0  =>  x(s)
        #
        # is a  monotonic (invertible) function.   But we are  using a
        # different, more straightforward, strategy.
        #
        # FIXME:  anything  better  than  Newton  solver  for  inverse
        # function?
        #

        #
        # (Private) instance slots:  primary path parmatrization, norm
        # definition, funciton converting length to abscissa and total
        # path length:
        #
        self.p = p
        self.norm = norm
        self.arc = arc
        self.arg = arg

    # The next method implements the interface of Func(), however the
    # path function is vector valued!
    def taylor(self, s):
        """Returns the path derivative with respect to normalized path
        length s from the interval [0, 1].
        """

        # full path length:
        L = self.arc(1.0)

        # abscissa corresponding to normalized path length:
        x = self.arg(s * L)

        # value and derivative of the primary path parametrization:
        y, yprime = self.p.taylor(x)

        return y, yprime * L / self.norm(yprime, y)

    def get_nodes(self):
        """Property handler, returns a tuple of arrays (xs, ys)
        """

        # primary path nodes:
        xs, ys = self.p.nodes

        # full path length:
        L = self.arc(1.0)

        return array(map(self.arc, xs)) / L, ys

    nodes = property(get_nodes)

def write_gplfile(path, file):
    f = open(file, 'w')
    ss = arange(0., 1., 1e-3)

#   if False:
#       fs = array([self.f(s) for s in ss])
#       plt.plot(fs[:,0], fs[:,1], '-')

    for s in ss:
        f.write('\t'.join(['%f' % num for num in path(s)]))
        f.write('\n')
    f.write('\n\n')

    xs, ys = path.nodes
    for i in ys:
        f.write('\t'.join(['%f' % num for num in i]))
        f.write('\n')
    f.close()

# python path.py [-v]:
if __name__ == "__main__":
    import doctest
    doctest.testmod()

# Default options for vim:sw=4:expandtab:smarttab:autoindent:syntax
