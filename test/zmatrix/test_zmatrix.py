import sys
import unittest
import os

import aof
from aof.zmatrix import *

def file2str(f):
    f = open(f, "r")
    mystr = f.read()
    f.close()
    return mystr

print "__file__", __file__

class TestZMatrixAndAtom(aof.test.MyTestCase):

    def setUp(self):
        self.original_dir = os.getcwd()
        os.chdir(os.path.dirname(__file__))

    
    def tearDown(self):
        os.chdir(self.original_dir)

    def test_ZMatrix(self):
        input = """N
H 1 hn
H 1 hn 2 hnh
H 1 hn 2 hnh 3 120.1

hn 1.2
hnh 109.5
"""
        
        z = ZMatrix(input)

        hexane_zmt = file2str("hexane.zmt")

        z = ZMatrix(hexane_zmt)
        print "Testing hexane.zmt -> xyz"
        self.assertEqual(z.xyz_str().lower(), file2str("hexane.xyz").lower())

        print "Testing hexane2.zmt -> zmt"
        self.assertEqual(z.zmt_str().lower(), file2str("hexane2.zmt").lower())

        z = ZMatrix(file2str("benzyl.zmt"))
        print "Testing benzyl.zmt -> xyz"
        self.assertMatches(file2str("benzyl.xyz"), z.xyz_str())

    def test_ZMatrix_BigComplex(self):
        print "Testing bigComplex.zmt -> xyz, view in molden"
        z = ZMatrix(file2str("bigComplex.zmt"))
        f = open("bigComplex.xyz", "w")
        f.write("49\n\n" + z.xyz_str())
        f.close()
        import os
        os.system("molden bigComplex.xyz")

    def test_ZMatrixiExceptions(self):
        #  no space between
        input1 = """N
H 1 hn
H 1 hn 2 hnh
H 1 hn 2 hnh 3 120.1
hn 1.2
hnh 109.5
"""

        # no variables
        input2 = """N
H 1 hn
H 1 hn 2 hnh
H 1 hn 2 hnh 3 120.1
"""
        input3 = ""

        # missing variable
        input4 = """N
H 1 hn
H 1 hn 2 hnh
H 1 hn 2 hnh 3 120.1

hn 1.2
"""
 
        self.assertRaises(Exception, ZMatrix, input1)
        self.assertRaises(Exception, ZMatrix, input2)
        self.assertRaises(Exception, ZMatrix, input3)
        self.assertRaises(Exception, ZMatrix, input4)


    def test_Atom(self):
        a = Atom("H")
        b = Atom("He 1 HHe")
        c = Atom("C 2 CHe 1 CHeH")
        d = Atom("F 1 abc 2 sdf 3 dih")
        e = Atom("F 1 abc 2 sdf 3 -123.1")

        list = [a, b, c, d, e]
        list = str([str(i) for i in list])
        self.assertEqual(list, "['H', 'He 1 HHe', 'C 2 CHe 1 CHeH', 'F 1 abc 2 sdf 3 dih', 'F 1 abc 2 sdf 3 -123.1']")

    def test_dcart_on_dint(self):
        ANGSTROMS_TO_BOHRS = 1.8897

        DEG_TO_RAD = numpy.pi / 180.
        RAD_TO_DEG = 180. / numpy.pi

        m1 = file2str("CH4.zmt")
        z = ZMatrix(m1)
        m, e = z.dcart_on_dint(z.get_internals())

        print "Testing that the numerical diff errors are small"
        self.assert_(numpy.linalg.norm(e) < 1e-8)

        
        z = ZMatrix(file2str("benzyl.zmt"))
        m, e = z.dcart_on_dint(z.get_internals())

        print "Testing that the numerical diff errors are small"
        self.assert_(numpy.linalg.norm(e) < 1e-8)

        print "Testing generation of forces coordinate system transform matrix"

        zmt_grads_from_benzyl_log = numpy.array([-0.02391, -0.03394, -0.08960, -0.03412, -0.12382, -0.15768, -0.08658, -0.01934, 0.00099, 0.00000, -0.00541, 0.00006, -0.00067, 0.00000, -0.00556, 0.00159, 0.00000, -0.00482, -0.00208])
        xyz_grads_from_benzyl_xyz_log = numpy.array([0.023909846, 0.0, 0.034244932,  0.053971884, 0.0, -0.124058188, -0.004990116, 0.000000000, 0.000806757, -0.005402561, 0.000000000, -0.006533931,  0.008734562, 0.000000000, -0.006763414, -0.002889556, 0.000000000, -0.013862257, -0.072130600, 0.000000000, 0.125686058, -0.005409690, 0.000000000, 0.000029026, -0.002717236, 0.000000000, -0.005359264, 0.002107675, 0.000000000, -0.005198587, 0.004815793, 0., 0.001008869])

        calculated_zmt_grads = numpy.dot(m, xyz_grads_from_benzyl_xyz_log)

        """Gradients in Gaussian are in Hartree/Bohr or radian, but the transform 
        matrix generated by the ZMatrix class has units of angstroms/bohr or 
        degree. For this reason, when comparing Gaussian's gradients in terms of
        z-matrix coordinates (in benzyl.log) with those computed by transforming 
        Gaussian's forces in terms of cartesians (in benzyl_zmt.log), one needs to 
        multiply the angular forces from benzyl.log by the following factor:
        (ANGSTROMS_TO_BOHRS * RAD_TO_DEG). That's what the following two lines 
        are for."""
        for i in [2,4,6,7,8,9,11,12,13,15,16,18]:
            calculated_zmt_grads[i] *= (ANGSTROMS_TO_BOHRS * RAD_TO_DEG)

        self.assertAlmostEqualVec(calculated_zmt_grads, zmt_grads_from_benzyl_log, 1e-3)
#        print "xyz_grads_from_benzyl_log:", xyz_grads_from_benzyl_xyz_log
#        print "zmt_grads_from_benzyl_log:", zmt_grads_from_benzyl_log
#        print "calculated_zmt_grads:", calculated_zmt_grads



def suite():
    return unittest.TestLoader().loadTestsFromTestCase(TestZMatrixAndAtom)

if __name__ == "__main__":
    unittest.TextTestRunner(verbosity=2).run(suite())


