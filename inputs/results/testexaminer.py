#!/usr/bin/env python

import os
from os.path import join as pjoin
import sys
from glob import glob
import pickle
import re
from subprocess import Popen, PIPE

import numpy as np

import aof

args = sys.argv[1:]

test_dirs = glob("test[1-9]*")

print test_dirs

start_dir = os.getcwd()

def name(s):
    """
    >>> name(test1_something)
    'something'
    """
    return '_'.join(s.split('_')[1:])

simdata = {'ethane-hydroxide_reactants-products': 'HF/3-21G,-1,1',
    'alanine-dipeptide_adp-c5-adp-c7ax': 'HF/3-21G,0,1',
    'dielsalder_re-pr': 'HF/3-21G,0,1',
    'dielsalder_re-stack-pr-stack': 'HF/3-21G,0,1',
    'Mn_H_ib_tbut_re-pr': 'HF/3-21G,1,6'
    }

def vsys(s, dry=False):
    print s
    if dry:
        os.system('touch DRY_RUN')
    else:
        os.system(s)

def examine_log(f):
    grad_calls = 0
    e_calls = 0
    status = 'Unfinished'
    nimag = -1
    ls = open(f).read().splitlines()
    for l in ls:
        if l.find('Cartesian Forces:') == 1:
            grad_calls += 1
        elif l.find('SCF Done') != -1:
            e_calls += 1
        elif l.find('Normal termination of G') != -1:
            status = 'Finished'
            s = ''.join(open(f).read().splitlines())
            s = ''.join(s.split())
            arc = re.findall(r"1\\1\\.+?\\\\@", s)
            assert len(arc) == 2
            arc = arc[-1]
            m = re.search(r"NImag=(\d)", arc)
            assert m != None, arc
            nimag = int(m.group(1))

    return status, nimag, grad_calls, e_calls

def examine_geom(carts, carts_correct, ixs):
    def coord(carts, ix):
        carts = carts.reshape(-1,3)
        try:
            if len(ix) == 2:
                d = rc.distance(carts[ix])
                assert d > 0
                return d
            if len(ix) == 3:
                return rc.angle(carts[ix])
            if len(ix) == 4:
                return rc.dihedral(carts[ix])
        except:
            print carts
            print ixs
            print "ERROR"
            exit()
        assert False

    l1 = np.array([coord(carts, i) for i in ixs])
    l1.sort()
    l2 = np.array([coord(carts_correct, i) for i in ixs])
    l2.sort()
    l = np.abs((l1 - l2) / l2)

    return l.sum() / len(l)
 
def examine_job(logfile, tsfile, rcfile):
    """
        Returns the number of gradient calls to converge, or the convergence status.
        logfile: e.g. Gaussian output
        tsfile: contains known ts
        rcfile: coordinates of bonds associated with reaction coordinate

    """

    rc_ixs = eval(open(rcfile).read())
    status, nimag, grad_calls, e_calls = examine_log(logfile)
    
    # nimag should really be 1, but the ethane system had one very low mode.
    if status != 'Finished' or not nimag in (1,2):
        return 'NotConverged'

    found_ts_str = Popen(["babel", "-ig03", logfile, '-oxyz'], stdout=PIPE).communicate()[0]
    found_ts = aof.coord_sys.XYZ(found_ts_str).get_cartesians()
    known_ts = aof.coord_sys.XYZ(open(tsfile).read()).get_cartesians()
    found_ts_error = get_rc_error(found_ts, known_ts, rc_ixs)

    if found_ts_error < 0.01:
        return grad_calls
    else:
        return 'WronglyConverged'

def ts_rc_files(simname):

    path = '/home/hugh/src/sem-global-ts/inputs/knownts/'
    tsfile = path + simname + '.ts.xyz'
    rcfile = tsfile + '.rc'

    return tsfile, rcfile

def examine_cos(arc, oldstylearc=True):
    arc = open(arc)
    entries = []
    converged = None

    # Read in archive
    while True:
        try:
            entries.append(pickle.load(arc))
        except EOFError, err:
            break

    if not oldstylearc:
        final = entries[-1]
        assert type(final) == str

        converged = final[1:] == 'Event: Optimisation Converged'
    
    for i in range(1,100):
        if type(entries[-i]) == str:
            continue
        arc = entries[-i]
        break

    gradcalls = arc['bead_gs']
    return converged, gradcalls

def dump_convergence(fn):
    lines = open(fn).readlines()
    for l in lines:
        if l[:-1] == '|Optimisation Converged|':
            last = l
        elif l[:-1] == "|Optimisation STOPPED (maximum iterations)|":
            last = l
    print "Convergence:", last

def get_rc_error(carts, carts_correct, ixs):
    def coord(carts, ix):
        carts = carts.reshape(-1,3)
        try:
            if len(ix) == 2:
                d = aof.rc.distance(carts[ix])
                assert d > 0
                return d
            if len(ix) == 3:
                return aof.rc.angle(carts[ix])
            if len(ix) == 4:
                return aof.rc.dihedral(carts[ix])
        except:
            print carts
            print ixs
            print "ERROR"
            exit()
        assert False

    l1 = np.array([coord(carts, i) for i in ixs])
    l1.sort()
    l2 = np.array([coord(carts_correct, i) for i in ixs])
    l2.sort()
    l = np.abs((l1 - l2) / l2)

#    print ixs
#    print "l2",l2
#    print "l1",l1

    return l.sum() / len(l)


mode = args[0]
assert mode in ('make_local_jobs', 'submit', 'analysed_local_jobs')

gcdicts = {}
estims = ['SplCubic', 'High', 'Spl', 'SplAvg', 'Bell']

for d in test_dirs:
    os.chdir(d)

    n = name(d)
    qcparams = simdata[n]

    cos_logfile = d + '-stdout.log'
    if mode == 'analyse_band':
        pass
    elif mode == 'submit':
        for j in glob('*.com'):
            s = 'subhere %s' % j
            print s
            os.system(s)

    # Create QC inputs
    elif mode == 'make_local_jobs':

        # slurp output file and test convergence
        dump_convergence(glob("test*-stdout.log")[0])


        arc_files = glob("*archive.pickle")
        assert len(arc_files) == 1
        converged, gradcalls = examine_cos(arc_files[0])
        print "gradcalls", gradcalls
        converged = False # just for testing
        if converged:
            os.system('arcproc.py -l %s -t -j %s -e SplCubic,High,Spl,SplAvg,Bell %s' % (n, qcparams, arc_files[0]))

    # Analyse Local TS Searches
    elif mode == 'analysed_local_jobs':

        tsfile, rcfile = ts_rc_files(n)
        gcdicts[d] = {}
        for e in estims:
            local_jobs = glob('%s*%s.log' % (n, e))
            assert len(local_jobs) == 1
            j = local_jobs[0]

            grad_calls = examine_job(j, tsfile, rcfile)
            gcdicts[d][e] = grad_calls

    else:
        assert False

    os.chdir(start_dir)

# output results
print gcdicts
