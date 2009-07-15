import unittest

import sys
sys.path.append("../")
from zmatrix import *

def file2str(f):
    f = open(f, "r")
    mystr = f.read()
    f.close()
    return mystr


class TestZMatrixAndAtom(unittest.TestCase):

    def setUp(self):
        pass

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
        self.assertEqual(z.xyz_str(), file2str("hexane.xyz"))

        self.assertEqual(z.zmt_str(), file2str("hexane2.zmt"))

        z = ZMatrix(file2str("benzyl.zmt"))
        self.assertEqual(file2str("benzyl.xyz"), z.xyz_str())

    def test_ZMatrix_BigComplex(self):
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


if __name__ == "__main__":
#    test_Atom()
#    test_ZMatrix()
    unittest.main()


