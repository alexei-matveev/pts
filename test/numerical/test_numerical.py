import unittest

import sys
import aof.test

import aof.numerical as num

class TestNumerical(aof.test.MyTestCase):
    def setUp(self):
        self.nd = num.NumDiff()
        self.NUM_DIFF_ERR = 1e-6

    def test_numdiff_small(self):
        from numpy import sin, cos, exp, array, linalg, arange

        f = lambda x: array((sin(x[0]),))
        print self.nd.numdiff(f, array((0.,)))
        l = [ self.nd.numdiff(f, array((i,))) for i in arange(10.) ]
        l2 = array([x for (x,y) in l]).flatten()
        print l2
        res = array(l2)
        print res
        correct = array([cos(i) for i in range(10)])

        self.assert_(linalg.norm(res - correct) < self.NUM_DIFF_ERR)

    def test_nummdiff_big(self):
        from numpy import sin, cos, exp, array, linalg, arange

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

        X1 = arange(3.)
        estim, err = self.nd.numdiff(fbig, X1)
        exact = gbig(X1).flatten()
        diff = linalg.norm(estim.flatten() - exact)
        self.assert_(diff < self.NUM_DIFF_ERR)

def suite():
    return unittest.TestLoader().loadTestsFromTestCase(TestNumerical)

if __name__ == "__main__":
    unittest.TextTestRunner(verbosity=2).run(suite())


