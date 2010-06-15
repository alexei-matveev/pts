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
    print " -t          : find TS aprpoximation from path"
    print " -g          : gnuplot output"
    print " -m          : display modes"
    print " -q          : query records"
    print " -v          : verbose"
    print " -j <header> : generate QC job for local search using <header>."
    print " -r <range>  : specify range of iterations to process"
    print " -l <label>  : label"

class Usage(Exception):
    pass

def disp_forces(pathtools):
    print "para\tperp"
    for f_perp, f_para in pathtools.projections():
        f_perp = np.linalg.norm(f_perp)
        f_para = rms(f_para)

        print "%.3f\t%.3f" % (f_perp, f_para)

def range2ixs(s):
    """
    >>> range2ixs('1')
    [1]

    >>> range2ixs('1-5')
    [1, 2, 3, 4, 5]

    >>> range2ixs('1,5,6-10')
    [1, 5, 6, 7, 8, 9, 10]

    """
    if s == '-':
        return range(1,10000)
    ixs = []
    subranges = s.split(',')
    for sr in subranges:
        nums = [int(n) for n in sr.split('-')]
        assert len(nums) in (1,2)
        if len(nums) == 1:
            nums.append(nums[0])
        ixs += range(nums[0], nums[1]+1)
    return ixs

def main(argv=None):
    if argv is None:
        argv = sys.argv
    try:
        try:
            opts, args = getopt.getopt(argv[1:], "j:htmgfq:vr:l:", ["help"])
        except getopt.error, msg:
             usage()
             return
        
        # defaults
        verbose = False
        ts_from_path = False
        gnuplot_out = False
        forces = False
        modes = False
        query = False
        write_jobs = False
        label = 'label'
        path_rec_ixs = [-1]
        ts_rec_ixs = [-1]
        for o, a in opts:
            if o in ("-h", "--help"):
                usage()
                return 
            elif o == "-r":
                rec_ixs = range2ixs(a)
            elif o == "-t":
                ts_from_path = True
#                ts_rec_ixs = range2ixs(a)
            elif o in ("-f"):
                forces = True
            elif o in ("-g"):
                gnuplot_out = True
#                path_rec_ixs = range2ixs(a)
            elif o in ("-m"):
                modes = True
            elif o in ('-q'):
                query = a
            elif o == '-v':
                verbose = True
            elif o == '-l':
                label = a


            elif o == '-j':
                write_jobs = True
                qcjob_info = a
            else:
                usage("Unrecognised option %s" % o)
                return -1

        if not len(args) in (1,2):
            raise Usage("Requires either one or two arguments.")

        fn_pickle = args[0]
        f_arc = open(fn_pickle)
        entries = []

        cs = pickle.load(f_arc)
        #TODO: something like: assert issubclass(cs, aof.coord_sys.CoordSys)

        e_prev = None
        while True:
            try:
                e = pickle.load(f_arc)
                if e_prev is None or e['N'] != e_prev['N']:
                    entries.append(e)
                e_prev = e
            except EOFError, err:
                break

        if query:
            if not query in entries[0]:
                print "Possibly arguments to -q:"
                keys = entries[0].keys()
                keys.sort()
                print '\n'.join(keys)
                exit(1)
            print '#', query
            for i, record in enumerate(entries):
                print i, record[query]
            exit()

        path_list = []
        ts_list = []
        for i in rec_ixs:
            try:
                record = entries[i]
            except IndexError, err:
                break
            state = record['state_vec']
            es = record['energies']
            gs = record['gradients']
            ss = record['pathps']

            path_list.append((i, state, es, gs, ss))
            

        max_coord_change = np.abs(state[0] - state[-1]).max()
        print "Last archive entry"
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

        if forces:
            disp_forces(pt)

        if ts_from_path:

            for path in path_list:
                i, state, es, gs, ss = path

                print "Analysing path %d of %d beads..." % (i, len(es))
                pt = aof.tools.PathTools(state, es, gs, ss)

                methods = {'SplCubic': pt.ts_splcub,
#                           'High': pt.ts_highest,
#                           'Spl': pt.ts_spl
#                           'SplAvg': pt.ts_splavg,
#                           'Bell': pt.ts_bell
                          }


                def estims():
                    ks = methods.keys()
                    ks.sort()
                    for k in ks:
                        res = methods[k]()
                        if res != []:
                            yield (k, res[-1])

                for name, est in estims():
                    energy, coords, s0, s1, s_ts, l, r = est
                    ts_list.append((i, s_ts, energy)) 
                    cs.set_internals(coords)
                    if write_jobs:
                        fn = '%s-loc_search-iter%d-%s.com' % (label, i, name)
                        print "Writing QC job to", fn
                        f = open(fn, 'w')
                        f.write(make_job_gau(qcjob_info))
                        f.write(cs.xyz_str())
                        f.write('\n\n')

                        f.close()

                    if verbose:
                        print name
                        print cs.xyz_str()
                        print

                    carts = cs.get_cartesians()
                    if ts_known:
                        energy_err = energy - ts_energy
                        error = rot.cart_diff(carts, ts_carts)[0]

                        print "%s: (E_est - E_corr) = %.3f ;Geom err = %.3f ;bracket = %.1f-%.1f" % (name.ljust(20), energy_err, error, s0, s1)

#        print len(ts_list)
#        exit()
        if gnuplot_out:
            aof.tools.gnuplot_path3D(path_list, ts_list, fn_pickle)

    except Usage, err:
        print >>sys.stderr, err
        print >>sys.stderr, "for help use --help"
        usage()
        return 2
    except IOError, err:
        print >>sys.stderr, err
        return -1

def make_job_gau(data):
    """
    Constructs a Gaussian job.
    """
    s = """# %(lot)s opt=(calcfc,noeigentest,loose,ts) freq

blah

%(spin)d %(mult)d
"""

    lot, spin, mult = data.split(',')

    return s % {'lot': lot, 'spin': int(spin), 'mult': int(mult)}


if __name__ == "__main__":
    sys.exit(main())

