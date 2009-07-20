import sys
sys.path.append("../../")
sys.path.append("../")

import testing

from molinterface import *

class TestMolInterface(testing.MyTestCase):
    def test_MolInterface2(self):
        return
        m1 = file2str("CH4.zmt")
        m2 = m1

        mi = MolInterface([m1, m2])
        print "Testing: MolecularInterface"
        print mi
        self.assertEqual(mi.format, "zmt")
        self.assertEqual(mi.atoms, ['C', 'H', 'H', 'H', 'H'])
        self.assertEqual(mi.var_names, ['ch', 'hch', 'hchh'])
        print mi.reagent_coords[0]
        self.assert_((numpy.array(mi.reagent_coords[0]) == numpy.array([1.09, 109.5, 120.])).all())


        X = numpy.array([1.0900000000000001, 109.5, 120.0])
        print mi.opt_coords2cart_coords(X)

        print "Testing: coordsys_trans_matrix()"
        print mi.coordsys_trans_matrix(X)

        print "Testing: run_qc()"
        logfilename = mi.run_qc(X)
        print "file", logfilename, "created"
        (e, g) = mi.logfile2eg(logfilename, X)
        self.assertAlmostEqual(e, -1087.8230597121312)
        for g1, g2 in zip(array([  3.85224119e-02,   8.29947922e-06,  -2.61589395e-07]), g):
            self.assertAlmostEqual(g1,g2)

    def test_MolInterface3(self):
        return
        f = open("H2O.zmt", "r")
        m1 = f.read()
        m2 = m1
        f.close()

        mi = MolInterface(m1, m2)
        print "Testing: MolecularInterface"
        print mi

        X = numpy.array([1.5, 100.1])
        print mi.opt_coords2cart_coords(X)

        print "Testing: coordsys_trans_matrix()"
        print mi.coordsys_trans_matrix(X)

        print "Testing: run_job()"
        logfilename = mi.run_job(X)
        print "file", logfilename, "created"
        print mi.logfile2eg(logfilename, X)

        import cclib
        file = cclib.parser.ccopen("H2O.log")
        data = file.parse()
        print "SCF Energy and Gradients from direct calc on z-matrix input:"
        print data.scfenergies[-1]
        print data.grads
        print data.gradvars

    def test_MolInterface(self):
        print "test_MolInterface"
        print line()
        m1 = file2str("NH3.zmt")
        m2 = m1

        mi = MolInterface([m1, m2])
        output = """N     0.9800000000   0.0000000000   0.0000000000
H     1.3429960463   0.0000000000   1.0267204441
H     1.3429960463  -0.8891659872  -0.5133602221
H     1.3429960463   0.8891659872  -0.5133602221
"""
        self.assertMatches(mi.coords2moltext(numpy.array([0.97999999999999998, 1.089, 109.471, 120.0])), output)

        print "Testing: coords2qcinput()"
        print mi.coords2qcinput(numpy.array([0.97999999999999998, 1.089, 109.471, 120.0]))
        print mi.coords2qcinput(numpy.array([0.97999999999999998, 1.089, 129.471, 0.0]))

        str_zmt = """H
        F  1  r2
        Variables:
        r2= 0.9000
        """

        print "Testing: zmt2xyz()"
        print mi.zmt2xyz(str_zmt)

        print "Testing: coords2xyz()"
        print mi.coords2xyz([0.97999999999999998, 1.089, 109.471, 120.0])

        print "Testing: opt_coords2cart_coords()"
        cart_coords = numpy.array([0.98, 0., 0., 1.34299605, 0., 1.02672044, 1.34299605, -0.88916599, -0.51336022,  1.34299605,  0.88916599, -0.51336022])
        X = numpy.array([0.97999999999999998, 1.089, 109.471, 120.0])
        self.assertAlmostEqualVec(mi.opt_coords2cart_coords(X), cart_coords)

        print "Testing: coordsys_trans_matrix()"
        matrix_correct = numpy.array([ 1., 0., 0., 1., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0.3333297, 0., 0.94281032, 0.3333297, -0.81649769, -0.47140516, 0.3333297, 0.81649769, -0.47140516, 0., 0., 0., 0.01791965, 0., -0.00633548, 0.01791965, 0.00548668, 0.00316774, 0.01791965, -0.00548668, 0.00316774, 0., 0., 0., 0., 0., 0., 0., 0.00895983, -0.01551887, 0., -0.00895983, -0.01551887])
        matrix = mi.coordsys_trans_matrix(X).flatten()

        print matrix
        print matrix_correct
        self.assertAlmostEqualVec(matrix, matrix_correct)

    """def assertAlmostEqualVec(self, v1, v2):
        print v1
        print v2
        diff = numpy.linalg.norm(v1 - v2)
        self.assertAlmostEqual(0.0, diff)"""

    def test_MolInterface(self):
        m1 = file2str("almost_straight.zmt")
        m2 = m1

        mi = MolInterface([m1, m2])
        zmt_coords = numpy.array([180.])
        str, coords = mi.coords2xyz(zmt_coords)
        f = open("almost_straight.xyz", "w")
        f.write("3\n\n" + str)
        f.close()

        print mi.coordsys_trans_matrix(zmt_coords)


if __name__ == "__main__":
    testing.main()


