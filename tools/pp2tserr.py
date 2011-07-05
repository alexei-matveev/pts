#!/usr/bin/env python

"""
Path Pickle to Transition State Error

Converts a pickle of (state_vec, energies, gradients, MolInterface) to the 
error between various estimates of the transition state and the real transition state.

"""

import sys
import getopt
from pts.tools.pathtools import unpickle_path

import numpy as np
from ase.io import read

import pts
import pts.tools.rotate as rot
from pts.common import file2str, rms

def usage():
    print "Usage: paratools pp2ts_err [options] file.pickle [ts-actual.xyz]"
    print "Options:"
    print " -t: find TS approximation from path"
    print " -g: gnuplot output"
    print " -m: display modes"

class Usage(Exception):
    pass

def disp_forces(pathtools):
    print "para\tperp"
    for f_perp, f_para in pathtools.projections():
        f_perp = np.linalg.norm(f_perp)
        f_para = rms(f_para)

        print "%.3f\t%.3f" % (f_perp, f_para)

def energy_from_file(f):
    energy = None
    f_in = open(f, "r")
    n= f_in.readline()
    energystr = f_in.readline()
    f_in.close()
    if "E" in energystr:
       fields = energystr.split()
       energy = float(fields[-1])
    return energy

def main(argv):
    try:
        try:
            opts, args = getopt.getopt(argv, "htmgf", ["help"])
        except getopt.error, msg:
             usage()
             return
        
        ts_from_path = False
        gnuplot_out = False
        forces = False
        modes = False
        for o, a in opts:
            if o in ("-h", "--help"):
                usage()
                return 
            elif o in ("-t"):
                ts_from_path = True
            elif o in ("-f"):
                forces = True
            elif o in ("-g"):
                gnuplot_out = True
            elif o in ("-m"):
                modes = True
            else:
                usage()
                return -1

        if len(args) < 1 or len(args) > 2:
            raise Usage("Requires either one or two arguments.")

        fn_pickle = args[0]
        state, ss, es, gs, ns, cs = unpickle_path(fn_pickle)
        max_coord_change = np.abs(state[0] - state[-1]).max()
        print "Max change in any one coordinate was %.2f" % max_coord_change
        print "                            Per bead %.2f" % (max_coord_change / len(state))
        print "Energy of all beads    %s" % es
        print "Energy of highest bead %.2f" % es.max()

        ts_known = len(args) == 2
        if ts_known:
             fn_ts = args[1]
             ts = read(fn_ts)
             ts_carts = ts.get_positions()
             ts_energy = energy_from_file(fn_ts)

        if gnuplot_out:
            s = '\n'.join(['%.2f' % e for e in es])
            print s

        pt = pts.tools.PathTools(state, es, gradients=gs)

        if gnuplot_out:
            pts.tools.gnuplot_path(pt, fn_pickle)

        methods = {'Spling and cubic': pt.ts_splcub,
                   'Highest': pt.ts_highest,
                   'Spline only': pt.ts_spl,
                   'Spline and average': pt.ts_splavg,
                   'Bell Method': pt.ts_bell
                  }

        if forces:
            disp_forces(pt)

        if ss != None:
            pt2 = pts.tools.PathTools(state, es, gs, ss)
            methods2 = {'Spline only (with abscissa)': pt2.ts_spl,
                        'Spline and average (with abscissa)': pt2.ts_splavg,
                        'Spline and cubic (with abscissa)': pt2.ts_splcub
                       }
            methods.update(methods2)

            if forces:
                disp_forces(pt2)


        def estims():
            ks = methods.keys()
            ks.sort()
            for k in ks:
                res = methods[k]()
                if res != []:
                    yield (k, res[-1])

        for name, est in estims():
            energy, coords, s0, s1, s_ts, l, r = est
            if ts_from_path:
                print name
                carts_c = cs(coords)
                for n, cor in zip(ns, carts_c):
                     print "%3s  %12.8f %12.8f %12.8f" %(n, cor[0], cor[1], cor[2])
                print
            if modes:
                modes =  pt.modeandcurvature(s_ts, l, r, cs)
                for namemd, modevec in modes:
                     print "Approximation of modes using method:", namemd
                     for line in modevec:
                         print "   %12.8f  %12.8f  %12.8f" % (line[0], line[1], line[2])
                print

            carts = cs(coords)
            if ts_known:
                energy_err = energy - ts_energy
                error = rot.cart_diff(carts, ts_carts)[0]

                print "%s: (E_est - E_corr) = %.3f ;Geom err = %.3f ;bracket = %.1f-%.1f" % (name.ljust(20), energy_err, error, s0, s1)

    except Usage, err:
        print >>sys.stderr, err
        print >>sys.stderr, "for help use --help"
        usage()
        return 2
    except IOError, err:
        print >>sys.stderr, err
        return -1

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

