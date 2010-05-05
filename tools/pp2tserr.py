#!/usr/bin/env python

"""
Path Pickle to Transition State Error

Converts a pickle of (state_vec, energies, gradients, MolInterface) to the 
error between various estimates of the transition state and the real transition state.

"""

import sys
import getopt
import pickle

import numpy as np

import aof
import aof.tools.rotate as rot
from aof.common import file2str, rms

def usage():
    print "Usage: " + sys.argv[0] + " [options] file.pickle [ts-actual.xyz]"
    print "Options:"
    print " -d: dump"
    print " -g: gnuplot output"

class Usage(Exception):
    pass

def disp_forces(pathtools):
    print "para\tperp"
    for f_perp, f_para in pathtools.projections():
        f_perp = np.linalg.norm(f_perp)
        f_para = rms(f_para)

        print "%.3f\t%.3f" % (f_perp, f_para)

def main(argv=None):
    if argv is None:
        argv = sys.argv
    try:
        try:
            opts, args = getopt.getopt(argv[1:], "hdgf", ["help"])
        except getopt.error, msg:
             usage()
             return
        
        dump = False
        gnuplot_out = False
        forces = False
        for o, a in opts:
            if o in ("-h", "--help"):
                usage()
                return 
            elif o in ("-d"):
                dump = True
            elif o in ("-f"):
                forces = True
            elif o in ("-g"):
                gnuplot_out = True
            else:
                usage()
                return -1

        if len(args) < 1 or len(args) > 2:
            raise Usage("Requires either one or two arguments.")

        fn_pickle = args[0]
        f_ts = open(fn_pickle)
        try:
            state, es, gs, ss, ssold, cs = pickle.load(f_ts)
            f_ts.close()
        except ValueError:
            ssold = None
            try:
                f_ts.close()
                f_ts = open(fn_pickle)
                state, es, gs, ss, cs = pickle.load(f_ts)
                f_ts.close()
            except ValueError:
                f_ts.close()
                f_ts = open(fn_pickle)
                state, es, gs, cs = pickle.load(f_ts)
                f_ts.close()
                ss = None
        max_coord_change = np.abs(state[0] - state[-1]).max()
        print "Max change in any one coordinate was %.2f" % max_coord_change
        print "                            Per bead %.2f" % (max_coord_change / len(state))
        print "Energy of all beads    %s" % es
        print "Energy of highest bead %.2f" % es.max()

        ts_known = len(args) == 2
        if ts_known:
             fn_ts = args[1]
             ts = aof.coord_sys.XYZ(file2str(fn_ts))
             ts_carts = ts.get_cartesians()
             ts_energy = ts.get_potential_energy()

        if gnuplot_out:
            s = '\n'.join(['%.2f' % e for e in es])
            print s

        pt = aof.tools.PathTools(state, es, gs)

        if gnuplot_out:
            aof.tools.gnuplot_path(state, es, fn_pickle)

        methods = {'Spling and cubic': pt.ts_splcub,
                   'Highest': pt.ts_highest,
                   'Spline only': pt.ts_spl,
                   'Spline and average': pt.ts_splavg,
                   'Bell Method': pt.ts_bell
                  }

        if forces:
            disp_forces(pt)

        if ss != None:
            pt2 = aof.tools.PathTools(state, es, gs, ss)
            methods2 = {'Spline only (2)': pt2.ts_spl,
                        'Spline and average (2)': pt2.ts_splavg,
                        'Spline and cubic (2)': pt2.ts_splcub
                       }
            methods.update(methods2)

            if forces:
                disp_forces(pt2)

        def estims():
            for k in methods.keys():
                res = methods[k]()
                if res != []:
                    yield (k, res[-1])

        for name, est in estims():
            energy, coords, s0, s1, s_ts, l, r = est
            cs.set_internals(coords)
            if dump:
                print name
                print cs.xyz_str()
                print cs.get_internals()
                print
                modes =  pt.modeandcurvature(s_ts, l, r, cs)
                for namemd, modevec in modes:
                     print "Approximation of modes using method:", namemd
                     for line in modevec:
                         print "   %12.8f  %12.8f  %12.8f" % (line[0], line[1], line[2])
                print

            carts = cs.get_cartesians()
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
    sys.exit(main())

