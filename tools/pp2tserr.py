#!/usr/bin/env python

"""
Path Pickle to Transition State Error

Converts a pickle of (state_vec, energies, gradients, MolInterface) to the 
error between various estimates of the transition state and the real transition state.

"""

import sys
import getopt
import pickle

import aof
import aof.tools.rotate as rot
from aof.common import file2str

def usage():
    print "Usage: " + sys.argv[0] + " [options] file.pickle ts-actual.xyz"

class Usage(Exception):
    pass

def main(argv=None):
    if argv is None:
        argv = sys.argv
    try:
        try:
            opts, args = getopt.getopt(argv[1:], "h", ["help"])
        except getopt.error, msg:
             usage()
        
        for o, a in opts:
            if o in ("-h", "--help"):
                usage()
                return 
            else:
                usage()
                return -1

        if len(args) != 2:
            raise Usage("Requires two arguments.")

        fn_pickle = args[0]
        fn_ts = args[1]
        f_ts = open(fn_pickle)
        state, es, gs, cs = pickle.load(f_ts)

        ts = aof.coord_sys.XYZ(file2str(fn_ts))
        ts_carts = ts.get_cartesians()

        pt = aof.tools.PathTools(state, es, gs)

        estims = []
        estims.append(pt.ts_splcub()[-1])
        estims.append(pt.ts_highest()[-1])

        for e in estims:
            cs.set_internals(e[1])
            carts = cs.get_cartesians()
            error = rot.cart_diff(carts, ts_carts)[0]

            print error

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


