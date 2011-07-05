#!/usr/bin/env python
"""
Shared code for parsing command line here
"""

import getopt

# for get calculator
from ase.calculators import *
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
    mask = strmask.split()
    mask = [m == "True" for m in mask]
    return mask

# Default options for vim:sw=4:expandtab:smarttab:autoindent:syntax
