#!/usr/bin/env python


"""Displays a couple of ASE GUIs to facilitate the ordering of atoms in pairs of molecules."""

import os
import sys
import getopt

class Usage(Exception):
    def __init__(self, msg):
        self.msg = msg

plot_str = """
set term postscript size 8,6 enhanced color
set output "plots.ps"

set multiplot

set size 1, 0.5
#set style point 1 linewidth 5


set style data linespoints

set ylabel "Hartree"
set origin 0,0.5
plot [1:%(maxit)d] %(energyplots)s
set origin 0, 0
set logscale y
set nokey

set ylabel "Hartree / Angstrom"
set xlabel "Band String Gradient Evaluations"
plot [1:%(maxit)d] %(gradientplots)s

"""
#aseneb.txt.arch" using 2:4 title "ASE NEB", "myneb_aselbfgs-newe.txt.arch" using 2:4 title "My NEB (ASE-LBFGS)", "myneb_numpy

def run(args, maxit=50):
    for fn in args:
        f = open(fn)
        f_out = open(fn + '.out', 'w')
        while True:
            line = f.readline()
            if not line:
                break

            if line[0:7] == 'Archive':
                line = line.split(':')[1]
                """a = line.split()
                bc, bgc, gc, ec, rmsf, e, maxe = a
                bc, bgc, gc, ec = [int(i) for i in bc, bgc, gc, ec]
                rmsf, e, maxe = [float(i) for i in rmsf, e, maxe]"""

                f_out.write(line)
        f.close()
        f_out.close()

    plot_files = ['"' + fn + '.out"' for fn in args]
    energy_plots = [fn + ' using 3:6' for fn in plot_files]
    energy_plots = ','.join(energy_plots)

    gradient_plots = [fn + ' using 3:5' for fn in plot_files]
    gradient_plots = ','.join(gradient_plots)

    values = {'maxit': maxit, 'energyplots': energy_plots, 'gradientplots': gradient_plots}
    gnuplot_str = plot_str % values

    gpfile = '_'.join(args) + '.gp'
    out = open(gpfile, 'w')
    out.write(gnuplot_str)
    out.close()

    os.system('gnuplot ' + gpfile)
    os.system('gv plots.ps')


                




def main(argv=None):
    if argv is None:
        argv = sys.argv
    try:
        try:
            opts, args = getopt.getopt(argv[1:], "h", ["help"])
        except getopt.error, msg:
             raise Usage(msg)
        
        if len(args) < 1:
            raise Usage("Must specify at least one file")
        run(args)

    except Usage, err:
        print >>sys.stderr, err.msg
        print >>sys.stderr, "for help use --help"
        return 2

if __name__ == "__main__":
    sys.exit(main())



