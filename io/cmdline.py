#!/usr/bin/env python
"""
Shared code for parsing command line here
"""

import getopt

# for get calculator
from ase.calculators import *
from pts.gaussian import Gaussian
from pts.common import file2str
from pts.defaults import default_params, default_calcs, default_lj, default_vasp

LONG_OPTIONS = ["calculator="]

def get_options(argv, options="", long_options=LONG_OPTIONS):

    opts, args = getopt.getopt(argv, options, long_options)

    return opts, args

def get_defaults():
    """
    Returns a copy of the parameter dictionary with default settings
    """
    return default_params.copy()

def get_calculator(file_name):

    calculator = None
    if file_name in default_calcs:
        calculator = eval("%s" % (file_name))
    else:
        str1 = file2str(file_name) # file file_name has to
        # contain line calculator = ...
        exec(str1)

    return calculator

def get_mask(strmask):
    tr = ["True", "T", "t", "true"]
    fl = ["False", "F", "f", "false"]
    mask = strmask.split()
    # test that all values are valid:
    true_or_false = tr + fl
    for element_of_mask in mask:
        assert( element_of_mask in true_or_false)
    # Transform mask in logicals ([bool(m)] would
    # be only false for m = ""
    mask = [m in tr for m in mask]
    return mask

# Default options for vim:sw=4:expandtab:smarttab:autoindent:syntax
