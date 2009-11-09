import sys
import unittest
import os
import pickle

import numpy
import ase

import aof
import aof.asemolinterface as ami
import aof.coord_sys as cs
import aof.common as common
from aof.common import Job
from aof.common import file2str

# whether to perform tests quickly
quick = True

# whether to perform visual tests, requiring user interaction
visual = False

def test_calc(**kwargs):
    if quick:
        return ase.EMT()
    else:
        return aof.aof.ase_gau.Gaussian(**kwargs)

class TestAseMolInterface(aof.test.MyTestCase):

    def setUp(self):
        self.original_dir = os.getcwd()
        new_dir = os.path.dirname(__file__)
        if new_dir != '':
            os.chdir(new_dir)

    
    def tearDown(self):
        os.chdir(self.original_dir)

    def test_constructor_XYZ(self):

        print "Testing: MolecularInterface constructor with XYZ"

        params = dict()
        params["calculator"] = ase.EMT, [], []

        m1 = file2str("benzyl.xyz")
        m2 = m1

        mi = ami.MolInterface([m1, m2], params)
        print "Testing: MolecularInterface"
        print mi

    def test_constructor_run_calc(self):

        print "Testing: MolecularInterface constructor with XYZ"

        params = dict()
        d = dict()
        params["calculator"] = ase.EMT, [], d

        m1 = file2str("benzyl.xyz")
        m2 = m1

        mi = ami.MolInterface([m1, m2], params)

        c = mi.reagent_coords[0]

        j = Job(c, Job.G())
        print mi.run(j)

    def test_constructor_ZMatrix(self):

        print "Testing: MolecularInterface constructor with ZMatrices"
        params = dict()
        params["calculator"] = ase.EMT, [], dict()

        m1 = file2str("CH4.zmt")
        m2 = m1

        mi = ami.MolInterface([m1, m2], params)
        print mi

        # XYZ with ZMatrix
        m2 = file2str("benzyl.xyz")
        self.assertRaises(cs.ZMatrixException, ami.MolInterface, [m1, m2], params)

        # non-matching atoms
        m2 = file2str("NH4.zmt")
        self.assertRaises(ami.MolInterfaceException, ami.MolInterface, [m1, m2], params)

        # mixed dihedrals
        m2 = file2str("CH4-mixeddih.zmt")
        self.assertRaises(ami.MolInterfaceException, ami.MolInterface, [m1, m2], params)

    def test_constructor_ComplexCoordSys(self):
        pass

def suite():
    return unittest.TestLoader().loadTestsFromTestCase(TestAseMolInterface)

if __name__ == "__main__":
    visual = True
    quick = False
    unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite([TestAseMolInterface("test_constructor_run_calc")]))


