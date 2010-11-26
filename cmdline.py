#!/usr/bin/python
"""
Shared code for parsing command line here
"""

import getopt

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
    from ase.calculators import *
    from pts.common import file2str
   #scope = {}
   #execfile(file_name, scope)
    # print "scope=", scope
    str1 = file2str(file_name) # file file_name has to
    # contain line calculator = ...
    exec(str1)
    return calculator
    #return scope["calculator"]

def get_mask(strmask):
    mask = eval("%s" % (strmask))
    print mask
    return mask

# Default options for vim:sw=4:expandtab:smarttab:autoindent:syntax
