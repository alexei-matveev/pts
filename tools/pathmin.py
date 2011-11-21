#!/usr/bin/env python
"""
Tool to find the iteration with minimal sum for a given argument.
The default argument is RMS Perp Forces
Also it needes that all the single beads are below some barier, say 1.0.

usage:
  paratools min-iter <logfile>

<logfile> should be file in the pathsearcher output file format. *.log

also possible:
  paratools min-iter --limit <min> --argument <argument> <logfile>

Here --limit sets the barrier for the single bead values to <min>
and --argument changes the argument to check from RMS Perp Forces
to <argument>.

give the arguments --method max to get the iteration with the lowest
  maximal value instead.

Can process several logfiles at once
"""
from sys import argv, exit
from pydoc import help
from copy import copy
from pts.io.cmdline import get_options

def main(argv):
    barrier = 10000000000000000.0
    argument = "RMS Perp Forces"
    method = "sum"
    output = "full"

    if '--help' in argv:
        print __doc__
        exit()

    opts, args = get_options(argv, long_options=["limit=", "argument=", "method=", "output="])

    if len(args) < 1:
        print "ERROR, there is need for at least one input file!"
        print __doc__
        exit()

    for opt, val in opts:
        if opt == "--limit":
            barrier = float(val)
        if opt == "--argument":
            argument = val
        if opt == "--method":
            method = val
        if opt == "--output":
            output = val

    for file in args:
        searchmin(file, barrier, argument, method, output)


def searchmin(file, bar, arg, meth, out):
    """
    Expect to get files containing lines with arg: v1 | v2 | v3 | ...
    where vi refers to the value of arg vor the i'th bead. as v1 and v-1 should
    be not changing, they are not considered
    """
    fields = arg.split()
    len_start = len(fields)
    mini = -1
    min_line = None
    min_value = None
    j = 1

    # Make it a function
    fun = eval(meth)

    filein = open(file, "r")
    # Filter the lines in the file
    for line in filein:
        if line.startswith('%s' % arg):
            fields = line.split()
            dataline = []
            # Line holds first len_start strings with parts of arg
            # Then followes :, then the values and separators are
            # following alternately
            datapoints = (len(fields) - len_start) / 2
            for i in range(datapoints):
                dataline.append(float(fields[len_start + 1 +2 * i]))

            dataline = dataline[1:-1] # Do not use termination beads

            if max(dataline) < bar:
                if min_value == None:
                    # First iteration is always minimal
                    mini = j
                    min_line = dataline
                    min_value = fun(dataline)
                else:
                    if min_value > fun(dataline):
                        mini = j
                        min_line = dataline
                        min_value = fun(dataline)
            j += 1

    if out == "quick":
        print  mini, min_value, max(min_line), sum(min_line)
        return None

    print "The line with minimum", meth, " was", mini, "with value", min_value
    print "sum ", sum(min_line), "max", max(min_line), "average", sum(min_line)/len(min_line)
    print "Here the values could be found:"
    print min_line

if __name__ == "__main__":
    main(argv[1:])

