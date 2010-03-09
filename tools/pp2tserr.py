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
from aof.common import file2str

def usage():
    print "Usage: " + sys.argv[0] + " [options] file.pickle [ts-actual.xyz]"

class Usage(Exception):
    pass

def main(argv=None):
    if argv is None:
        argv = sys.argv
    try:
        try:
            opts, args = getopt.getopt(argv[1:], "hd", ["help"])
        except getopt.error, msg:
             usage()
             return
        
        dump = False
        for o, a in opts:
            if o in ("-h", "--help"):
                usage()
                return 
            if o in ("-d"):
                dump = True
            else:
                usage()
                return -1

        if len(args) < 1 or len(args) > 2:
            raise Usage("Requires either one or two arguments.")

        fn_pickle = args[0]
        f_ts = open(fn_pickle)
        state, es, gs, cs = pickle.load(f_ts)
        max_coord_change = np.abs(state[0] - state[-1]).max()
        print "Max change in any one coordinate was %.2f" % max_coord_change
        print "                            Per bead %.2f" % (max_coord_change / len(state))
        print "Energy of all beads    %s" % es
        print "Energy of highest bead %.2f" % es.max()


        if len(args) == 2:
            fn_ts = args[1]
            ts = aof.coord_sys.XYZ(file2str(fn_ts))
            ts_carts = ts.get_cartesians()
            ts_energy = ts.get_potential_energy()

        pt = aof.tools.PathTools(state, es, gs)

        estims = []
        pt.ts_splcub()

        #print pt
        print pt.para_forces
        print pt.para_forces_fd
        exit()
        estims.append(('Spling and cubic', pt.ts_splcub()[-1]))
        estims.append(('Highest', pt.ts_highest()[-1]))
        estims.append(('Spline only', pt.ts_spl()[-1]))
        estims.append(('Spline and average', pt.ts_splavg()[-1]))

        for name, est in estims:
            energy, coords, s0, s1, _, _ = est
            energy_err = energy - ts_energy
            cs.set_internals(coords)
            if dump:
                print name
                print cs.xyz_str()
                print cs.get_internals()
                print

            carts = cs.get_cartesians()
            error = rot.cart_diff(carts, ts_carts)[0]

            print "%s: (E_est - E_corr) = %.3f Geom err = %.3f bracket = %.1f-%.1f" % (name.ljust(20), energy_err, error, s0, s1)

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


