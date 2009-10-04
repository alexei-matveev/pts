import unittest
import numpy

import sys
sys.path.append("../../")

main = unittest.main

class MyTestCase(unittest.TestCase):
    def __init__(self, x):
        unittest.TestCase.__init__(self, x)

    def assertMatches(self, s1, s2):
        """Tests that two strings are token-wise identical, word tokens are
        compared case insensitively and numbers are rounded to the 4 decimal
        places."""
        import re
        s1_toks = s1.lower().split()
        s2_toks = s2.lower().split()
        def proc(s):
            if re.match(r"[+-]?\d+(\.\d*)?", s):
                return round(float(s), 4)
            else:
                return s
        s1_h = [proc(i) for i in s1_toks]
        s2_h = [proc(i) for i in s2_toks]
        self.assert_(s2_h == s1_h)
        

    def assertAlmostEqualVec(self, v1, v2, max_diff=1e-5):
        """Tests that the norm of two vectors are within diff of each other."""
        diff = numpy.linalg.norm(v1 - v2)
        self.assert_(diff < max_diff)

