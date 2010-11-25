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

    def test_NEB1(self):
        assert False, "Moved to doctests"
        default_spr_const = 1.
        reactants = np.array([0,0])
        products = np.array([3,3])

        neb = NEB([reactants, products], lambda x: True, pts.qcdrivers.GaussianPES(), default_spr_const, beads_count = 10)

        angs = neb.get_angles()
        self.assertAlmostEqualVec(angs, 180. * np.ones(neb.beads_count - 2))

        # get energy
        print neb.obj_func()
        # get gradients
        # get tangents
        neb.update_tangents()
        ts = neb.tangents
        self.assert_(len(ts) == neb.beads_count)
        for t in ts[1:-1]:
            self.assertAlmostEqual(np.linalg.norm(t,2), 1)

    def test_NEB2(self):
        assert False, "Moved to doctests"

        default_spr_const = 1.
        reactants = np.array([0,0])
        products = np.array([1,1])

        neb = NEB([reactants, products], lambda x: True, pts.qcdrivers.GaussianPES(), default_spr_const, beads_count = 3)
        neb.obj_func(np.array([[0,0],[0,1],[1,1]]))
        print neb.bead_pes_energies
        neb.update_tangents()
        self.assertAlmostEqualVec(neb.tangents, np.array([[0,0],[1,0],[0,0]]))

        neb.bead_pes_energies = np.array([0,1,0])
        print neb.bead_pes_energies
        neb.update_tangents()
        self.assertAlmostEqualVec(neb.tangents, np.array([[0,0],[1./np.sqrt(2.),1./np.sqrt(2.)],[0,0]]))

        neb.bead_pes_energies = np.array([0,1,0.9])
        print neb.bead_pes_energies
        neb.update_tangents()
        print "In the following, the tangent in the middle should be mostly (1,0)"
        print neb.tangents

        neb.bead_pes_energies = np.array([1,0,-1])
        print neb.bead_pes_energies
        neb.update_tangents()
        self.assertAlmostEqualVec(neb.tangents, np.array([[0,0],[0,1],[0,0]]))



def suite_neb():
    return unittest.TestLoader().loadTestsFromTestCase(TestNEB)



