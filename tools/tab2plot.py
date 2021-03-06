#!/usr/bin/env python
"""
This tool takes a tabular of some data (for example internal coordinates)
and prints them

Even if i is desinged for the output of the xyz2tabint file it should read
every other input also, but the naming in the figures may go wrong then

It is designed for a file with two tabulars, the first one of a path,
the second one giving the dat afor the beads

The data to plot is taken from the tabulars, as a default the x value
is the first row of the tabular, the other rows are all y values. But there
is also the possibility to choose the data to plot y hand:
The option --"k1 n1_1 .. k2 n2_1 .." can set the values.
Here k can be choosen from the following table, the n's following are the number
of the columns in the tabular to use for the option, there must be exactly the number
needed by the special option:
 k  number of n's needed description
 t   1                  : just take the column given by n
 d   2                  : difference n1 - n2
 s   0                  : gives the difference in the symmetry
                          works on the next two functions given
                          and uses them to calculate 0.5 * (k2(k1)-k2(-k1))
                          when given a (float) number after s the k1 function
                          values are shifted to this number

The data could be used directly as the variables x or y (for a x-y plot)
or a function could work on them. So far the functions available are one (t)
which just takes the value, another (d) which takes the differences of two of the values
and a las

There is also the possibility of setting an option --title"string" --xlable"string"
and --ylabel"string" which then will be put in the picture

By setting something like --log" n1 n2", here the ni can be a number or x, y
The axes announced by these numbers will be set to logarithmic scale
"""
from sys import argv as arg
from sys import exit
import numpy as np
from matplotlib.pyplot import plot, show, legend, rcParams, xlabel, ylabel, xscale, yscale
from matplotlib.pyplot import gca,figure
from matplotlib.pyplot import title as set_title
from matplotlib.pyplot import savefig
from scipy.interpolate import splrep, splev
from copy import copy
from matplotlib import pyplot as plt
from matplotlib import cm

global plot_style
global color_num
color_num = 0
plot_style = "-"

def colormap(i, n):
    """
    Returns a color understood by color keyword of plt.plot().

    To choose most appropriate color map see

        http://www.scipy.org/Cookbook/Matplotlib/Show_colormaps
    """
    from matplotlib import cm # color management

    # we want to map i=0 to 0.0 and i=imax to 1.0
    imax = n - 1
    if imax == 0:
        imax = 1

    #
    # Color maps map the interval [0, 1] ontoto a color palette:
    #
    # return cm.hsv(float(i) / imax)
    return cm.jet(float(i) / imax)


def increase_color(color, style, repeat):
    """
    Changes the color (and if the list of them is finished
    also the style) of the output line
    """
    global color_num

    color_num = color_num + 1 % 20
    new_color = colormap(color_num, 20)

    return new_color, copy(style)

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
    rname = ""
    # Now_path decides if the input line is put into path or beads
    # as in the format used normaly the table is preceeded by three lines
    # of input (used here to extract the name), it is set to false in the
    # beginning, even if things should start with path
    now_path = 0

    for line in f_1:
        fields = line.split()
        if fields[0] in ["#observed", "observed"]:
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
        rest = np.asarray(rest)
        rest = rest.T
    except:
        rest = None

    return path, pname, beads, bname, rest, rname


def main(argv):

    files = []
    option = None
    xtil = None
    ytil = None
    til = None
    log = []
    outputfile = None
    option2 = None
    xrange = None
    yrange = None



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
            elif option.startswith("xlabel"):
                xtil =  option[6:]
            elif option.startswith("ylabel"):
                ytil =  option[6:]
            elif option.startswith("title"):
                til =  option[5:]
            elif option.startswith("log"):
                opts = option.split()
                log.append(opts[1])
            elif option.startswith("output"):
                outputfile = option[6:]
            elif option.startswith("xrange"):
                opts = option.split()
                xrange = [float(op) for op in opts[1:3]]
            elif option.startswith("yrange"):
                opts = option.split()
                yrange = [float(op) for op in opts[1:3]]
            else:
                option2 = option
        else:
            files.append(ar)

    if files is []:
        print "ERROR: table needed to get the values to print from!"
        exit()

    setup_plot(x_label = xtil, y_label = ytil, title = til, log = log)

    n = len(files)
    for i, file in enumerate(files):
       # make the plots ready for all files given
       # first get the data from there
       tablep, pname, tableb, bname, tabelr, rname = read_tab(file)
       prepare_plot(tablep, pname, tableb, bname, tabelr, rname, option2, colormap(i,n))

    plot_data(xrange = xrange, yrange = yrange, savefile = outputfile)


def setup_plot( title = None, x_label = None, y_label = None, log = []):
    """
    A function which prepares everything for a plot: sets title and labels
    and changes to logarithmic scale if required.
    """
    # set some parameters for the legend, as it would be
    # to big else
    rcParams['font.size'] = 8
    # the lines may be a bit wider than the default
    rcParams['lines.linewidth'] = 2

    if title is not None:
         set_title(str(title), fontsize = 'large')

    if x_label is not None:
         xlabel(x_label)

    if y_label is not None:
         ylabel(y_label)

    if log != []:
         for lg in log:
             if lg in ['x', 1, '1']:
                 xscale('log')
             if lg in ['y', 'z', 2, '2']:
                 yscale('log')

def makeplotter(tab, funx, funcs, name, colors, option = None):
     """
     A wrapper aoround the plot function, which uses
     the table tab from which its extract via funx the
     x values, and then giving name to distinguish between
     different input files (only for the pathes, beads are
     without name)
     option may telling that instead of a line, points should
     be used (for beads)
     """
     x = funx(tab)

     if len(colors) < len(funcs):
         colors = [colors[0] for i in range(len(funcs))]

     for i, fun, color in zip( range(0,len(funcs)),funcs, colors):
         y = fun(tab)
         lab = '%i' % (i)
         if name is not None:
             if i > 0:
                 lab = name + ' ' + lab
             else:
                 lab = name

         # use style for the line or the one given directly
         if option is None:
            opt = plot_style
         else:
            opt = option

         plot(x, y, opt, color = color, label = lab)

def prepare_plot( tablep, pname, tableb, bname, tabelr, rname, option, colors):
    """
    Generates plot for a path plotting object. Dependent on what
    is not None of the given parameters different things are plottet.
    Different availabilities of the path interpreting tools can thus
    be handled by the same function.

    tablep: Table of points on a path, will be given as a line.
    tableb: Table of bead points, thus given as points.
    tabler: Table of random points, when given alone connect them with
            a dotted line, else give them as points but a different kind
            than the beads.
    """
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
        makeplotter(tablep, xfun, yfuns, pname, colors)

    if tableb is not None:
        makeplotter(tableb, xfun, yfuns, bname, colors, 'o')

    if tabelr is not None:
        if tableb is not None:
            r_opt = 'D'
        else:
            r_opt = 'o:'
        makeplotter(tabelr, xfun, yfuns, rname, colors, r_opt)


def plot_data( hold = False, xrange = None, yrange = None, savefile = None):
    """
    Make the last settings before the actual plot. Then
    plot (or put into a file).
    """
    a = gca()
    if xrange != None:
        a.set_xlim(xrange)
    if yrange != None:
        a.set_ylim(yrange)


    # prepare the legend, should be placed at the best position
    # available
    legend( loc = "best")
    # Now to the actual plots (legend and plots), show on screen
    # or save as file
    if hold:
        pass
    elif savefile == None:
        show()
    else:
        savefig(savefile)


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
        if args[0] == "s":
        # finds out if symmetry of path is given (in a special coordinate)
        # the symmetry t(s') = t(-s') is checked, where s and t are functions
        # s may be shifted a constant m, thus s' = m - s
            offall = 1
            m = None
            try:
               m = float(args[1])
               offall +=1
            except:
               # m does not need to be explicity there, else it's 0
               #thus there need not to be any special care if it's not present
               pass

            # First funs to select the s values from the table
            # funs can be any of the other functions given above
            funs, offset = singleopt(args[offall:])
            offall += offset
            # Second fun (used for the t's)
            funt, offset2 = singleopt(args[offall:])
            offall += offset2
            if m == None:
                fun = testsymmetric(funs, funt).give
            else:
                fun = testsymmetric(funs, funt, m).give
            return fun, offall


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
    #in case line[self.a] is a list, which cannot do substract directly
        diff=[]
        for i in range(len(line[self.a])):
            diff.append(line[self.a][i]-line[self.b][i])
        return diff


class testsymmetric():
    """
    Finds out about symmetry of one system
    uses a function as s values, which should
    show the symmetry around s = m
    and another function t, which will be
    tested on its symmetry
    the average value of t(m-s') und t(m+s')
    is calculated and substracted from the current
    value t(s=m-s'), this value is used as output

    other than the difference and the takei function it
    works on the whole table all at once and cannot used
    line after line
    """
    def __init__(self, funs, funt, m = 0.0):
        self.funs = funs
        self.funt = funt
        self.m = m

    def give(self, table):
         s = np.asarray(self.funs(table) - self.m)
         end = min( abs(s[0]), abs(s[-1]))
         ii = np.nonzero( abs(s) <= end )[0]
         ii = ii.tolist()
         t = np.asarray(self.funt(table))
         tout = t.copy()
         spl = splrep(s, t)
         for i in ii:
              tout[i] = 0.5 * (t[i] - splev(-s[i], spl))
         for i in range(len(t)):
              if i < ii[0]:
                  tout[i] = tout[ii[0]]
              elif i > ii[-1]:
                  tout[i] = tout[ii[-1]]
         return tout

if __name__ == "__main__":
    import doctest
    doctest.testmod()
    main(arg[1:])



