#!/usr/bin/python
"""
To run these tests, execute "python path_representation.py".

Example use: define a few nodes, here four nodes in 2D:

    >>> path = ((-100., -100.), (0., -50.), (0., 50.), (100., 100.))

and construct a path connecting them by default with 4-equidistant nodes

    >>> p = Path(path)

This builds linear, quadratic or spline representation of the path,
depending on the number of nodes.

Now you can evaluate the path function at any point (between 0 and 1),
the equidistant values 0, 1/3, 2/3 and 1 correspond to
the four nodes we provided:

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

"Calling" a path function as above is equivalent to invoking
the "value" method |f|:

    >>> p.f(0)
    array([-100., -100.])

In addition the spline parametrization is used to efficently compute
the tangential of the path. Get this by calling the "derivative" method |fprime|:

    >>> p.fprime(0)
    array([ 650.,  -25.])

Using the length of the tangential one can compute the arc length
of the path section:

    >>> arc = Integral(p.tangent_length)
    >>> arc(0.0)
    0.0

    >>> arc(1.0)
    347.53579497254498

    >>> arc(0.5)
    173.76789748627249

Using the reciprocal function |arg| one may reparametrize the
path in units of arc-length:

    >>> arg = Inverse(arc)
    >>> arg(arc(0.5))
    0.5
    >>> arc(arg(173.76789748627249))
    173.76789748627249

    >>> round(p(arg(173.76789748627249)), 10)
    array([ 0.,  0.])

To properly compose the path function p(x) and parametrization
arg(s) so that the derivatives of the resulting function are
also considered one may do:

    >>> from func import compose
    >>> P = compose(p, arg)

    >>> round(P(173.76789748627249), 10)
    array([ 0.,  0.])

    >>> P.fprime(173.76789748627249)
    array([-0.07974522,  0.99681528])

The length of the resulting tangent is identically one because of
the chosen parametrization. Note, that P is not a Path() but rather
a Func() so that, for example, P.tangent_length() will fail.

A similar functionality is provided by the PathRepresentation class:

    >>> p = PathRepresentation(path)
    >>> p.arc(0.0)
    0.0

    >>> p.arc(1.0)
    347.53579497254498

    >>> p.arc(0.5)
    173.76789748627249

    >>> p.arc(0.5, 0.0) + p.arc(1.0, 0.5) - p.arc(1.0, 0.0)
    0.0

The inverse of the |arc| method is the |arg| method that gives you
the spline-coordinate of the point separated from the origin by
any arc length:

    >>> p.arg(p.arc(0.5))
    0.5

    >>> p.arc(p.arg(173.76789748627249))
    173.76789748627249

Generate equidistant points on the path
    >>> round(p.generate_beads(update=True))
    array([[-100., -100.],
           [  -2.,  -58.],
           [   2.,   58.],
           [ 100.,  100.]])

without update=True the tangents would not have been updated:

    >>> round(p.path_tangents * 100)
    array([[ 100.,   -4.],
           [  28.,   96.],
           [  28.,   96.],
           [ 100.,   -4.]])

You can use more points in the path if you decide so:

    >>> p.beads_count = 5
    >>> round(p.generate_beads())
    array([[-100., -100.],
           [ -17.,  -82.],
           [   0.,    0.],
           [  17.,   82.],
           [ 100.,  100.]])
"""

__all__ = ["Path", "PathRepresentation"]

import scipy.optimize
import scipy.integrate

from numpy import linalg, array, dot, sqrt, ones, arange, column_stack
from numpy import ndarray
from numpy import asarray

from func import LinFunc, QuadFunc, SplineFunc, Func
from func import Integral, Inverse

# simplified logger:
import sys
def debug(msg): sys.stderr.write(msg + "\n")

class Path(Func):
    """Supports operations on a path represented by a line, parabola, or a 
    spline, depending on whether it has 2, 3 or > 3 points.
    """

    def __init__(self, ys, xs=None):

        # number of path nodes that define a path:
        self.__node_count = len(ys)

        if xs is not None:
            # take predefined node spacing:
            assert len(ys) == len(xs)
            self.__xs = asarray(xs)
        else:
            # generate initial paramaterisation density
            # TODO: Linear at present, perhaps change eventually
            n = self.__node_count
            # evenly spaced array of length n from 0.0 to 1.0:
            self.__xs = array([ i / (n - 1.0) for i in range(n) ])

        # here the spline functions will be stored, one for each dimension:
        self.__fs = [] # redundant

        # so far this only accepts "ys" not "xs":
        self.set_nodes(ys)

        # generate linear/quadratic/spline representation of the path
        # using the normalized positions chosen above for the respective
        # points from the state vector:

        self.__regen_path_func()

        #
        # This function appeared idempotent, so calling it more than one time
        # doesnt hurt the numbers, only performance. However the legacy
        # behavior of the child class PathRepresentation changes if we
        # call __regen_path_func() from inside the set_nodes() property
        # handler. PathRepresentation expects to be able to update
        # the nodes of the parent Path without the path spline parametrization
        # to be regenerated. This leaves the parent Path class in a somewhat
        # inconsistent state.
        #

        #
        # FIXME: ultimately, we should delegate the control over details
        # of the path parametrization to function composition.
        # All of
        #          Path(x), Path(x(s)), Path(x(s(w)))
        #
        # run along the same path albeit the "distance" along the path
        # is measured differently every time: in x, in s, or in w.
        #

        # TODO check all beads have same dimensionality

    # The next two implement the interface of Func(),
    # however the path function is vector valued!
    def f(self, x):
        """Evaluates the path point from the path parametrization.
        Here the x is the *spline* argument, NOT the (normalized) length or
        weighted length along the path.
        FIXME: To use other path parametrization look into PathRepresentation class.
        Same as __call__(self, x)
        """

        # evaluate each vector component by calling stored parametrization:
        fs = array([f.f(x) for f in self.__fs])

        # restore original shape, say (NA x 3):
        fs.shape = self.__yshape

        return fs

    def fprime(self, x):
        """Evaluates the derivative (unnormalized tangential) wrt the "spline" argument x
        """

        # evaluate each vector component by calling stored parametrization of derivative:
        fprimes = array([f.fprime(x) for f in self.__fs])

        # restore original shape, say (NA x 3):
        fprimes.shape = self.__yshape

        return fprimes

#   self.__call__() equivalent to self.f() is inherited form Func()

    def get_nodes(self):
        # in the original shape:
        return self.__ys.reshape((self.__node_count,) + self.__yshape)

    def set_nodes(self, ys):
        """So far does not accepts spacing of nodes in primary (spline) coordinate,
        only the node positions |ys|. Because it is a property handler.
        """

        # |ys| is vector of arrays defining the path
        # FXIME: what if each array is of say Nx3 shape?

        # will reference original array if the type matches:
        ys = asarray(ys)

        # first dimension is the node count:
        assert self.__node_count == len(ys)

        # save original shape of the input arrays:
        self.__yshape = ys[0].shape
#       print "as is shape=", (self.__node_count,) + self.__yshape

        # internally we treat them flat, so care only about total size:
        self.__dimension = ys[0].size
#       print "internal shape=", (self.__node_count, self.__dimension)

        # treat internally each of |ys| as a flat array:
        self.__ys = ys.reshape(self.__node_count, self.__dimension)
#       print "ys=", self.__ys

        # generate linear/quadratic/spline representation of the path
        # using the normalized positions chosen above for the respective
        # points from the state vector:

        # FIXME: uncommenting this will FAIL the last doctest:
        # self.__regen_path_func()

        #
        # This function appeared idempotent, so calling it more than one time
        # doesnt hurt the numbers, only performance. However the legacy
        # behavior of the child class PathRepresentation changes if we
        # call __regen_path_func() from inside of the set_nodes() property
        # handler. PathRepresentation expects to be able to update
        # the nodes of the parent Path without the path spline parametrization
        # to be regenerated. This leaves the parent Path class in a somewhat
        # inconsistent state.
        #

    nodes = property(get_nodes, set_nodes)

    @property
    def dimension(self):
        return self.__dimension

    def __regen_path_func(self):
        """Rebuild a new path function and the derivative of the path based on 
        the contents of state_vec.
        FXIME: This function appears to be idempotent, so one might want to ensure
        the execution of the body only after state vector changes.
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
                # FIXME: here the self.__xs seems to be ignored!

                # TODO: at present, transition state assumed to be half way ebtween reacts and prods
                ps = array((0.0, 0.5, 1.0))
                ps_x_pow_2 = ps**2
                ps_x_pow_1 = ps
                ps_x_pow_0 = ones(len(ps_x_pow_1))

                A = column_stack((ps_x_pow_2, ps_x_pow_1, ps_x_pow_0))

                quadratic_coeffs = linalg.solve(A,ys)

                self.__fs.append(QuadFunc(quadratic_coeffs))

            else:
                # spline path
                self.__fs.append(SplineFunc(self.__xs, ys))


    def tangent_length(self, x):
        "Returns the 2-norm of the path tangential wrt spline argument"

        df = self.fprime(x)
        ds = sqrt(dot(df, df))
        return ds

    def tangent(self, x):
        """Returns the (normalized) tangent to the path at point x <- [0,1]."""

        # to avoid things like array([[1],[2]]) flatten:
        t = self.fprime(x).flatten()
        t = t / linalg.norm(t)
        return t

    @property
    def tangents(self):
        """Returns the unit tangents at path nodes."""

        return array([ self.tangent(x) for x in self.__xs ])

class PathRepresentation(Path):
    """Supports operations on a path represented by a line, parabola, or a 
    spline, depending on whether it has 2, 3 or > 3 points.
    """

    def __init__(self, state_vec, beads_count = None, rho = lambda x: 1):

        # construct the spline paramtrization of the path through the nodes
        # in the |state_vec|:
        Path.__init__(self, state_vec)

        # number of vectors defining the path
        if beads_count is None:
            # default to the number of nodes:
            self.beads_count = len(state_vec)
        else:
            self.beads_count = beads_count

        self.__path_tangents = []

        # Next, prepare the two transforamtions:
        #
        #       s = arc(x): spline argument -> arc length of the path
        #
        # and the reciprocal
        #
        #       x = arg(s): arc length of the path -> spline argument
        #
        # by integrating the length of the path tangential tangent_length(x)
        #
        # The derivative of s(x) must be positive for s(x) to be invertible.
        # This is the case for tangent_length() that is the norm of the tangential.
        # It is assumed that s(0) = 0.
        #

        # Pass integration criteria to scipy.integrate.quad() via **kwargs if desired:
        self.arc = Integral(self.tangent_length)
        self.arg = Inverse(self.arc)

        #
        # arc(x,a) evaluates the (cartesian) path length from the path point
        # parametrized by spline-argument |a| to the point parametrized by |x|.
        # NOTE: this is NOT the general path weight, just its (cartesian) length.
        #
        # The cartesian length of the tangential returned by tangent_length()
        # is by definition the derivative of the path length wrt spline argument:
        #
        #  ds / dx = |dp / dx| > 0  =>  s(x) is a monotonic (invertible) function
        #
        # arg(s) evaluates the path coordinate (spline argument) from the path arc distance
        # from the origin.
        # Note: x = arg(s) is inverse of the s = arc(x).
        #
        # Similarly
        #
        #  dx / ds = |dp / dx|^-1 > 0  =>  x(s) is a monotonic (invertible) function
        #
        # but we are using a different, more straightforward, strategy.
        #
        # FIXME: anything better than Newton solver for inverse function?
        #

        self.__rho = self.set_rho(rho)

        # Next, prepare the two transforamtions:
        #
        #       w = s2w(s): arc length -> arc weight
        #
        # and the reciprocal
        #
        #       s = w2s(w): arc weight -> arc length
        #
        # by integrating the weight function __rho
        #

        # Pass integration criteria to scipy.integrate.quad() via **kwargs if desired:
        self.s2w = Integral(self.__rho)
        self.w2s = Inverse(self.s2w)

        #
        # FIXME: ultimately, we should delegate the control over details
        # of the path parametrization to function composition.
        # All of
        #          Path(x), Path(x(s)), Path(x(s(w)))
        #
        # run along the same path albeit the "distance" along the path
        # is measured differently every time: in x, in s, or in w.
        #


        # TODO check all beads have same dimensionality

    def get_fs(self):
        return self.__fs

    @property
    def path_tangents(self):
        return self.__path_tangents

    def generate_beads(self, update = False):
        """Returns an array of the self.__beads_count vectors of the coordinates 
        of beads along a reaction path, according to the established path 
        (line, parabola or spline) and the parameterisation density."""

        # For the desired distances along the string, find the values of the
        # normalised coordinate that achive those distances.
        normd_positions = self.__generate_normd_positions()

        bead_vectors = []
        bead_tangents = []
        for str_pos in normd_positions:
            bead_vectors.append(self.f(str_pos))
            bead_tangents.append(self.tangent(str_pos))


        # path nodes that are usually consistent with the path parametrization:
        nodes = self.nodes
        (reactants, products) = (nodes[0], nodes[-1])
        bead_vectors = array([reactants] + bead_vectors + [products])
        bead_tangents = array([self.tangent(0)] + bead_tangents + [self.tangent(1)])

        if update:
            #
            # I think there is something tricky here: bead positions and
            # bead tangents here were generated using the "old" path prametrization
            #
            #   p = p(x), t(x) = dp/dx / |dp/dx|.
            #
            # However now that we are about to change the state vector, the new
            # path parametrization has to be generated (by regen_path_fun()).
            # The new path is still passing through the points in the state array,
            # but the tangents of the new path do not need to be preserved by
            # a new parametrization. Similarly, but maybe less relevant, is that
            # the beads are not "equidistant" on the new reparametrized path.
            #

            # currently this assumes to update the nodes of the Path but not
            # to generate a new Path parametrization. Are we misusing parent class
            # for bead storage here?
            self.set_nodes(bead_vectors)
            #
            # So, when is appropriate moment to reparametrize the path (regen_path_func())?
            # Should it be initiated from the |state_vec| handler?
            #

            self.__path_tangents = bead_tangents
            #
            # And what tangents should be stored/calculated here?
            # Consistent with the old or the new parametrization?
            #

        return bead_vectors

    def dump_rho(self):
        res = 0.02
        print "rho: ",
        for x in arange(0.0, 1.0 + res, res):
            if x < 1.0:
                print self.__rho(x),
        print
        raw_input("that was rho...")

    def set_rho(self, new_rho, normalise=True):
        """Set new bead density function, ensuring that it is normalised."""
        if normalise:
            (int, err) = scipy.integrate.quad(new_rho, 0.0, 1.0)
        else:
            int = 1.0
        self.__rho = lambda x: new_rho(x) / int
        return self.__rho

    def __generate_normd_positions(self):

        # full path length:
        arc = self.arc(1.0)

        # full path weight:
        weight = self.s2w(arc)

        # shortcut for number of beads:
        n = self.beads_count

        # put n-2 points with these path weights from path origin:
        weights = [ i * weight / (n - 1) for i in range(1, n - 1) ]

        # NOTE: these are unnormalized path arcs:
        arcs = [ self.w2s(w) for w in weights ]

        #  desired fractional positions along the string
        return [ self.arg(s) for s in arcs ]
            
# python path_representation.py [-v]:
if __name__ == "__main__":
    import doctest
    doctest.testmod()

# Default options for vim:sw=4:expandtab:smarttab:autoindent:syntax
