import unittest

import sys
import pts.test
import numpy as np


import pts.common as com

from pts.searcher import NEB
import pts.searcher

class TestNEB(pts.test.MyTestCase):
    def setUp(self):
        pass

    def test_project_out(self):
        v1 = np.array([1,0])
        v2 = np.array([1,0])
        v3 = pts.searcher.project_out(v1,v2)
        self.assertAlmostEqualVec(v3, np.zeros(2))

        v4 = np.array([3,1])
        v5 = np.array([1,0])
        v6 = pts.searcher.project_out(v5,v4)
        v7 = com.normalise(v6)
        self.assertAlmostEqualVec(v7, np.array([0,1]))

        v4 = np.array([3,1])
        v5 = np.array([-1,0])
        v6 = pts.searcher.project_out(v5,v4)
        v7 = com.normalise(v6)
        self.assertAlmostEqualVec(v7, np.array([0,1]))


def suite_neb():
    return unittest.TestLoader().loadTestsFromTestCase(TestNEB)



