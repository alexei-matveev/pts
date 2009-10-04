import unittest

import sys
import aof.test

from aof.molinterface import *
from aof.common import *

class TestCommon(aof.test.MyTestCase):
    def test_numdiff_small(self):
        from numpy import sin, cos, exp, array, linalg

        f = lambda x: array((sin(x[0]),))
        res = array([ numdiff(f, array((i,))) for i in range(10) ]).flatten()
        correct = array([cos(i) for i in range(10)])

        self.assert_(numpy.linalg.norm(res - correct) < NUM_DIFF_ERR)

    def test_nummdiff_big(self):
        from numpy import sin, cos, exp, array, linalg

        def fbig(X):
            x = X[0]
            y = X[1]
            z = X[2]
            return array([exp(x*y), cos(z), sin(x)])

        def gbig(X):
            x = X[0]
            y = X[1]
            z = X[2]
            dgdx = array([y*exp(x*y), 0., cos(x)])
            dgdy = array([x*exp(x*y), 0., 0.])
            dgdz = array([0, -sin(z), 0.])
            return array([dgdx, dgdy, dgdz])

        X1 = array([1,2,3])
        estim = numdiff(fbig, X1).flatten()
        exact = gbig(X1).flatten()
        diff = linalg.norm(estim - exact)
        self.assert_(diff < NUM_DIFF_ERR)

def suite():
    return unittest.TestLoader().loadTestsFromTestCase(TestCommon)

if __name__ == "__main__":
    unittest.TextTestRunner(verbosity=2).run(suite())


