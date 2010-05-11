#!/usr/bin/env python
"""
This tool takes a tabular of some data (for example internal coordinates)
and prints them

Even if i is desinged for the output of the xyz2tabint file it should read
every other input also, but the naming in the figures may go wrong then

It is designed for a file with two tabulars, the first one of a path,
the second one giving the dat afor the beads

The data could be used directly as the variables x or y (for a x-y plot)
or a function could work on them. So far the only function available takes
the difference of two of the values.
"""
from sys import argv as arg
from sys import exit
import numpy as np
from matplotlib.pyplot import plot, show, legend, rcParams

def read_tab(filename):
    """
    Reads in the table(s) from a file and gives them back.
    The file is supposed to be the output of a xyz2tabint calculation
    Thus the name of the file should be in a line starting with:

       observed in file: beadstringinit.xyz  the following values were calculated,

    And there are other lines starting with loop or chart, which could be omitted.
    If a line does not contain only numbers, it is not used, else it is stored in
    path or beads, beliving that the file starts with a path part, followed by a bead
    path
    """

    f_1 = open(filename, "r")

    path = None
    beads = None
    rest = []
    # this way the beads will not be listed in the legend
    pname = "_nolegend_"
    bname = "_nolegend_"
    rname = "_nolegend_"
    # Now_path decides if the input line is put into path or beads
    # as in the format used normaly the table is preceeded by three lines
    # of input (used here to extract the name), it is set to false in the
    # beginning, even if things should start with path
    now_path = 0

    for line in f_1:
        fields = line.split()
        if fields[0] == "#observed":
            # this line holds the name, which we only want to use for the
            # first table if its a two table file, as otherwise there would be a to
            # big legend
            if now_path is 1:
               now_path = 2
               beads = []
            elif now_path is 0:
               now_path = 1
               path = []
               # there is the name of the file situatedm which has been read in by
               # xyz2tabint;
               pname = fields[3]
            elif now_path is 2:
               now_path = 3
               rname = pname + " rest"
            continue
        # this two lines should also contain only text
        elif fields[0] == "#loop;":
            continue
        elif fields[0] == "#chart":
            continue

        try:
            # only the lines which contain only numbers are used here
            numbers = [float(f) for f in fields]
            if now_path == 1:
                path.append(numbers)
            elif now_path == 2:
                beads.append(numbers)
            else:
                rest.append(numbers)
        except:
            pass

    # path and bead should have (if used) elements
    # P(i,j), where i is the iteration and j the
    # number of the variable, here its changed to P(j,i)
    if path is not None:
        path = np.asarray(path)
        path = path.T

    if beads is not None:
        beads = np.asarray(beads)
        beads = beads.T

    try:
        rest1 = rest[0]
    except:
        rest = None

    if rest is not None:
        rest = np.asarray(rest)
        rest = rest.T

    return path, pname, beads, bname, rest, rname


def main(argv = None):

    files = []
    option = None

    # Variables coulb be given directly or via system
    if argv is None:
        argv = arg[1:]

    # there may be some options inside
    # otherwise it should be a file used
    # as untill now all options are which and how
    # to use the lines of the table, they are set for once
    for ar in argv:
        if ar.startswith("--"):
            option = ar[2:]
            if option == "help":
                print __doc__
                exit()
        else:
            files.append(ar)

    def makeplotter(tab, funx ,funcs, name, option = None):
         # A wrapper aoround the plot function, which uses
         # the table tab from which its extract via funx the
         # x values, and then giving name to distinguish between
         # different input files (only for the pathes, beads are
         # without name)
         # option may telling that instead of a line, points should
         # be used (for beads)
         x = funx(tab)
         for i, fun in enumerate(funcs):
             y = fun(tab)
             lab = '%i' % (i+1)
             if name is not None:
                 lab = name + ' ' + lab
             if option is None:
                 plot(x, y, label= lab)
             else:
                 plot(x, y, option, label= lab)

    if files is []:
        print "ERROR: table needed to get the values to print from!"
        exit()

    for file in files:
       # make the plots ready for all files given
       # first get the data from there
       tablep, pname, tableb, bname, tabelr, rname = read_tab(file)

       # one of them should at least be there
       if tablep is not None:
           tlen = len(tablep)
       elif tableb is not None:
           tlen = len(tableb)
       else:
           tlen = len(tabelr)

       # the functions which should operate on the table
       # have been given in the option
       xfun, yfuns = decide_which_values(option, tlen)

       # for each table which is there, make it run
       if tablep is not None:
           makeplotter(tablep, xfun, yfuns, pname)

       if tableb is not None:
           makeplotter(tableb, xfun, yfuns, bname, 'o')

       if tabelr is not None:
           makeplotter(tabelr, xfun, yfuns, rname, 'o') 

    # set some parameters for the legend, as it would be
    # to big else
    rcParams['font.size'] = 8
    # the lines may be a bit wider than the default
    rcParams['lines.linewidth'] = 2

    # prepare the legend, should be placed at the best position
    # available
    legend( loc = "best")
    # Now to the actual plots (legend and plots)
    show()

def decide_which_values(option, tlen):
    """
    This function decides which values are wanted to be plotted
    There are three different ways of deciding:
    no option had been set,
    option is a filename (set as --f filename) in which the
    data is listed
    option is a string of variables
    In any case the first is supposed to be the x-value
    The rest are y values
    """
    if option is None:
        return usedefault(tlen)
    else:
        if option.startswith("f"):
            return optionsfile(option[1:])
        else:
            option = option.split()
            return optionsstring(option)

def usedefault(tlen):
    """
    As a default take the first column of the tables as x-values
    and all the rest as y -values, needs to know how many columns
    are there in the table (tlen)
    Note that the table is stored transposed, so they are actually
    lines
    """
    xfun = takei(0).give
    yfuncs = []
    for i in range(1,tlen):
        yfuncs.append(takei(i).give)
    return xfun, yfuncs

def optionsfile(file):
    """
    The options are given in a file
    so read them in, seperate the
    variables
    """
    fo = open(file, "r")
    opts = fo.read()
    opts = opts.split()
    return optionsstring(opts)

def optionsstring(args):
    """
    Having the options as arguments
    decides on xfun (first arguments)
    and yfuncs (all the last)
    """
    # even in the smallest possible case there
    # need at least four arguments (two for x and two
    # for y)
    assert(len(args) > 3)
    # the first ones are for xfun, shar says how many
    # arguments are needed to decided on the table sings
    # thus how many arguments to omit to have the next
    # y-func with the first arguments starting
    xfun, shar = singleopt(args)
    yfuncs = []
    # the actual yfunc always starts at args[0]
    args = args[shar:]

    # cycle over all the arguments left, there may
    # be some more yfuncs, else the same as for xfun
    for i in range(len(args)):
         yfunc, shar = singleopt(args)
         if yfunc == None:
             break
         yfuncs.append(yfunc)
         args = args[shar:]
    return xfun, yfuncs

def singleopt(args):
    """
    Gets the actual function, first argument tells which kind it is,
    Then there are some more arguments needed to define the function fully
    """
    if args == []:
        # if args has already been exceeded (should not be stored)
        return None, None
    else:
        # This is tha case just take the i'th column of the table
        # One argument is additional wanted giving i
        if args[0] == "t":
            try:
                fun = takei(int(args[1])).give
            except:
                print "ERROR: something is wrong with the functions options"
                print "option take the i'th argument should be proceed by integer number i"
                exit()
            # return created function, two arguments used
            return fun, 2
        if args[0] == "d":
        # this takes the difference between the i'th and the j'th argument
        # should be precceded by i and j
            try:
                fun = difference(int(args[1]), int(args[2])).give
            except:
                print "ERROR: something is wrong with the functions options"
                print "option difference between the i'th and j'th  argument"
                print "should be proceed by the integer numbers i and j"
                exit()
            # return created function, three arguments used
            return fun, 3



class takei():
    """
    Just give back the i'th element
    """
    def __init__(self, i):
        self.i = i

    def give(self, line):
        return line[self.i]

class difference():
    """
    Just give back the difference between the
    a'th and b'th argument (e.g l[i]-l[j])
    """
    def __init__(self, a,b):
        self.a = a
        self.b = b

    def give(self, line):
        return line[self.a] - line[self.b]


if __name__ == "__main__":
    import doctest
    doctest.testmod()
    main()



