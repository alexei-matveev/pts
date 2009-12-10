"""Module to plot Potential Energy Surfaces and strings/bands."""

import aof
from copy import deepcopy
import numpy as np
from numpy import array, arange, vstack
import tempfile, os

try:
    import Gnuplot, Gnuplot.PlotItems, Gnuplot.funcutils
except:
    print "Warning, couldn't import Python GNU Plot interface"
    exit(1)


class SurfPlot():
    def __init__(self, pes):
        self.__pes = pes

    def plot(self, path = None, write_contour_file=False, maxx=3.0, minx=0.0, maxy=3.0, miny=0.0):
        import os
        opt = deepcopy(path)

        # Points on grid to draw PES
        ps = 20.0
        xrange = arange(ps)*((maxx-minx)/ps) + minx
        yrange = arange(ps)*((maxy-miny)/ps) + miny

        # tmp data file
        (fd, tmpPESDataFile,) = tempfile.mkstemp(text=1)
        Gnuplot.funcutils.compute_GridData(xrange, yrange, 
            lambda x,y: self.__pes.energy([x,y]), filename=tmpPESDataFile, binary=0)

        g = Gnuplot.Gnuplot(debug=1)
        g('set contour')
        g('set cntrparam levels 100')

        # write out file containing 2D data representing contour lines
        if write_contour_file:
            (fd1, tmp_contour_file,) = tempfile.mkstemp(text=1)

            g('unset surface')
            str = "set table \"%s\"" % tmp_contour_file
            g(str)
            g.splot(Gnuplot.File(tmpPESDataFile, binary=0)) 
            g.close()

            print tmpPESDataFile
            os.unlink(tmpPESDataFile)
            os.close(fd)
            return tmp_contour_file

        # Make a 2-d array containing a function of x and y.  First create
        # xm and ym which contain the x and y values in a matrix form that
        # can be `broadcast' into a matrix of the appropriate shape:
        g('set data style lines')
        g('set hidden')
        g.xlabel('x')
        g.ylabel('y')

        # Get some tmp filenames
        (fd, tmpPathDataFile,) = tempfile.mkstemp(text=1)
        Gnuplot.funcutils.compute_GridData(xrange, yrange, 
            lambda x,y: self.__pes.energy([x,y]), filename=tmpPESDataFile, binary=0)
        if opt != None:
            opt.shape = (-1,2)
            pathEnergies = array (map (self.__pes.energy, opt.tolist()))
            pathEnergies += 0.05
            xs = array(opt[:,0])
            ys = array(opt[:,1])
            data = transpose((xs, ys, pathEnergies))
            Gnuplot.Data(data, filename=tmpPathDataFile, inline=0, binary=0)
            import os
            #wt()


            # PLOT SURFACE AND PATH
            g.splot(Gnuplot.File(tmpPESDataFile, binary=0), 
                Gnuplot.File(tmpPathDataFile, binary=0, with_="linespoints"))
        else:

            # PLOT SURFACE ONLY
            g.splot(Gnuplot.File(tmpPESDataFile, binary=0))

        #wt()

        os.unlink(tmpPathDataFile)
        os.unlink(tmpPESDataFile)


class Plot2D:

    plot_count = 0

    def ___init__(self):
        """Given a path object cos, displays the a 2D depiction of it's 
        first two dimensions as a graph.
        
        >>> neb = aof.searcher.NEB(array([[0,0.],[3.,3]]), aof.pes.GaussianPES(), 1)
        >>> p = Plot2D()
        >>> p.plot(aof.pes.GaussianPES(), neb) 
        Press to continue...
        """

    def plot(self, pes, cos, path_res = 0.002):
        g = Gnuplot.Gnuplot(debug=1)

        state = cos.state_vec.copy()
        xs = state[:,0]
        ys = state[:,1]
        x0 = xs.min()
        x1 = xs.max()
        xframe = (x1 - x0) * 0.2 # add 20% frame
        y0 = ys.min()
        y1 = ys.max()
        yframe = (y1 - y0) * 0.2 # add 20% frame

        x0 = x0 - xframe
        x1 = x1 + xframe
        y0 = y0 - yframe
        y1 = y1 + yframe




        g.xlabel('x')
        g.ylabel('y')

        g('set xrange [' + str(x0) + ':' + str(x1) + ']')
        g('set yrange [' + str(y0) + ':' + str(y1) + ']')
        g('set key right bottom')
        g('set key box')

        # Get some tmp filenames
        (fd, tmp_file1,) = tempfile.mkstemp(text=1)
        (fd, tmp_file2,) = tempfile.mkstemp(text=1)
        (fd, tmp_file3,) = tempfile.mkstemp(text=1)

        plots = []
        if cos.pathfs() != None:
            params = arange(0, 1 + path_res, path_res)
            f_x = cos.pathfs()[0].f
            f_y = cos.pathfs()[1].f
            xs = array ([f_x(p) for p in params])
            ys = array ([f_y(p) for p in params])

            # smooth path
            smooth_path = vstack((xs,ys)).transpose()
         

        sp = SurfPlot(pes)
        contour_file = sp.plot(None, write_contour_file=True, minx=x0, maxx=x1, miny=y0, maxy=y1)

        # state vector
        data2 = cos.state_vec
        Gnuplot.Data(data2, filename=tmp_file2, inline=0, binary=0)

        # PLOT THE VARIOUS PATHS
        g('set output "gp_test' + str(self.plot_count) + '.ps"')
        self.plot_count += 1

        if cos.pathfs() != None:
            g.plot(Gnuplot.File(tmp_file2, binary=0, with_ = "points", title = "get_state_vec()"), 
                Gnuplot.File(contour_file, binary=0, title="contours", with_="lines"),Gnuplot.Data(smooth_path, filename=tmp_file1, inline=0, binary=0, with_='lines'))
        else:
            g.plot(Gnuplot.File(tmp_file2, binary=0, with_ = "points", title = "get_state_vec()"), 
                Gnuplot.File(contour_file, binary=0, title="contours", with_="lines"))

        #g.hardcopy('gp_test.ps', enhanced=1, color=1)

        raw_input('Press to continue...\n')

        os.unlink(tmp_file1)
        os.unlink(tmp_file2)
        os.unlink(tmp_file3)
        os.unlink(contour_file)

# python path_representation.py [-v]:
if __name__ == "__main__":
    import doctest
    doctest.testmod()

# Default options for vim:sw=4:expandtab:smarttab:autoindent:syntax
