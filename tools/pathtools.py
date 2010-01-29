from aof.path import Path
import numpy as np
from aof.common import vector_angle
import aof.func as func
import scipy as sp

class PathTools:
    """
    Implements operations on reaction pathways, such as estimation of 
    transition states using gradient/energy information.

    >>> pt = PathTools([0,1,2,3], [1,2,3,2])
    >>> pt.steps
    array([ 0.,  1.,  2.,  3.])

    >>> pt.ts_highest()
    [(3, array([2]))]

    >>> pt = PathTools([0,1,2,3], [1,2,3,2], [0,1,-0.1,0])
    >>> res1 = pt.ts_splcub()

    >>> pt = PathTools([[0,0],[1,0],[2,0],[3.,0]], [1,2,3,2.], [[0,0],[1,0],[-0.1,0],[0,0]])
    >>> res2 = pt.ts_splcub()
    >>> res1[0][0] == res2[0][0]
    True

    >>> np.round(res1[0][0], 0)
    3.0
    >>> res1[0][0] > 3
    True

    """
    def __init__(self, state, energies, gradients=None):

        self.n = len(energies)
        self.state = np.array(state).reshape(self.n, -1)
        self.energies = np.array(energies)

        if gradients != None:
            self.gradients = np.array(gradients).reshape(self.n, -1)
            assert self.state.shape == self.gradients.shape

        assert len(state) == len(energies)

        self.steps = np.zeros(self.n)

        for i in range(self.n)[1:]:
            x = self.state[i]
            x_ = self.state[i-1]
            self.steps[i] = np.linalg.norm(x -x_) + self.steps[i-1]
        self.steps = self.steps / self.steps[-1]
        print "self.steps", self.steps

    def ts_spl(self, tol=1e-10):
        """Returns list of all transition state(s) that appear to exist along
        the reaction pathway."""

        assert False, "Function ts_spl, untested after module change / reorganisation."

        lg.info("Estimating the transition states along the pathway, mode = %s" % mode)
        n = self.n
        Es = self.energies.reshape(n,-1)
        dofs = self.state
        assert len(dofs) == len(Es)

        """Uses a spline representation of the energy/coordinates of the entire path."""
        ys = np.hstack([dofs, Es])

        step = 1. / n
        xs = np.arange(0., 1., step)

        p = Path(ys, xs)

        E_estim_neg = lambda s: -p(s)[-1]
        E_prime_estim = lambda s: p.fprime(s)[-1]

        ts_list = []
        for x in xs[2:]:#-1]:
            # For each pair of points along the path, find the minimum
            # energy and check that the gradient is also zero.
            E_0 = -E_estim_neg(x - step)
            E_1 = -E_estim_neg(x)
            x_min = sp.optimize.fminbound(E_estim_neg, x - step, x, xtol=tol)
            E_x = -E_estim_neg(x_min)
#            print x_min, np.abs(E_prime_estim(x_min)), E_0, E_x, E_1

            # Use a looser tollerance when minimising the gradient than for 
            # the energy function. FIXME: can this be done better?
            E_prime_tol = tol * 1E4
            if np.abs(E_prime_estim(x_min)) < E_prime_tol and (E_0 <= E_x >= E_1):
                p_ts = p(x_min)
                ts_list.append((p_ts[-1], p_ts[:-1]))

    def ts_splcub(self, numerical=False, tol=1e-10):
        """
        Uses a spline representation of the molecular coordinates and 
        a cubic polynomial defined from the slope / value of the energy 
        for pairs of points along the path.
        """

        ys = self.state.copy()

        step = 1. / self.n
        ss = self.steps
        Es = self.energies
        print Es

        # build fresh functional representation of optimisation 
        # coordinates as a function of a path parameter s

        xs = Path(ys, ss)
        
        ts_list = []

        for i in range(self.n)[1:]:#-1]:
            # For each pair of points along the path, find the minimum
            # energy and check that the gradient is also zero.
            E_0 = Es[i-1]
            E_1 = Es[i]
            dEdx_0 = self.gradients[i-1]
            dEdx_1 = self.gradients[i]
            dxds_0 = xs.fprime(ss[i-1])
            dxds_1 = xs.fprime(ss[i])

            #energy gradient at "left/right" bead along path
            dEds_0 = np.dot(dEdx_0, dxds_0)
            dEds_1 = np.dot(dEdx_1, dxds_1)

            dEdss = np.array([dEds_0, dEds_1])

            if (E_1 >= E_0 and dEds_1 < 0) or (E_1 <= E_0 and dEds_0 > 0):
                print "Found: i = %d E_1 = %f E_0 = %f dEds_1 = %f dEds_0 = %f" % (i, E_1, E_0, dEds_1, dEds_0)
                cub = func.CubicFunc(ss[i-1:i+1], Es[i-1:i+1], dEdss)

                """    # TODO: delete?
                if numerical:
#                    print "cub",cub
                    # Some versions of SciPy seem to give the function called 
                    # by fminbound a 1d vector, and others give it a scalar.
                    # The workaround here is to use the atleast_1d function.
                    E_estim_neg = lambda s: -cub(np.atleast_1d(s)[0])
                    E_prime_estim = lambda s: cub.fprime(np.atleast_1d(s)[0])

  #                  print ss[i-1], ss[i]
                    s_min = sp.optimize.fminbound(E_estim_neg, ss[i-1], ss[i], xtol=tol, disp=3)
                    E_s = -E_estim_neg(s_min)
 #                   print "E_0, E_s, E_1, s_min", E_0, E_s, E_1, s_min
    #                print np.abs(E_prime_estim(s_min)), tol
                    E_prime_estim_sqr = lambda x: np.abs(E_prime_estim(x))
                    s_min2 = sp.optimize.fmin(E_prime_estim_sqr, s_min, disp=0, xtol=tol)
                    
#                    print "%.15e %.15e" % (s_min, s_min2)
#                    print (E_0 - E_s)

                    if np.abs(s_min[0] - s_min2[0]) < 0.00001:
                        #print "Found", i
                        xs_ts = xs(s_min)
                        print "s_min", s_min
                        raw_input()
                        ts_list.append((E_s, xs_ts))"""

                # find the stationary points of the cubic
                statpts = cub.stat_points()
                assert statpts != []
                found = 0
                for p in statpts:
                    # test curvature
                    if cub.fprimeprime(p) < 0:
                        ts_list.append((cub(p), xs(p)))
                        found += 1

                assert found == 1, "Must be exactly 1 stationary points in cubic path segment but there were %d" % found

        return ts_list

    def ts_highest(self):
        """
        Just picks the highest energy from along the path.
        """
        i = self.energies.argmax()
        ts_list = [(self.energies[i], self.state[i])]

        return ts_list

# Testing the examples in __doc__strings, execute
# "python gxmatrix.py", eventualy with "-v" option appended:
if __name__ == "__main__":
    import doctest
    doctest.testmod()

# You need to add "set modeline" and eventually "set modelines=5"
# to your ~/.vimrc for this to take effect.
# Dont (accidentally) delete these lines! Unless you do it intentionally ...
# Default options for vim:sw=4:expandtab:smarttab:autoindent:syntax


