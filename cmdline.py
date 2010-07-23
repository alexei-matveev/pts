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

    scope = {}
    execfile(file_name, scope)
    # print "scope=", scope

    return scope["calculator"]

# Default options for vim:sw=4:expandtab:smarttab:autoindent:syntax
