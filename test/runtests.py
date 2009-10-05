#!/usr/bin/env python

import unittest
import sys
print "tsdf"
import aof.test.common as common
import aof.test.zmatrix as zmatrix
import aof.test.searcher as searcher

print "here"
test_suites = {
    'common':   common.suite(),
    'zmatrix':  zmatrix.suite(),
    'neb':      searcher.suite_neb(),
    }


print "here"
print test_suites.values()
alltests = unittest.TestSuite(test_suites.values())
test_suites['all'] = alltests

tests = sys.argv[1:]

if len(tests) == 0:
    tests.append("all")

if __name__ == "__main__":
    for test in tests:
        if not test in test_suites:
            print "Test", test, "not found"
        else:
            print "Running test suite:", test
            unittest.TextTestRunner(verbosity=4).run(test_suites[test])


