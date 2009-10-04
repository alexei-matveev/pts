import unittest

import aof.test.common as common
import aof.test.zmatrix as zmatrix

alltests = unittest.TestSuite([common.suite(), zmatrix.suite()])

if __name__ == "__main__":
    unittest.TextTestRunner(verbosity=4).run(alltests)


