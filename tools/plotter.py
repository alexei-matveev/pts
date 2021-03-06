#!/usr/bin/env python

"""Generates plots (via GNUPlot) to examine convergence.

Usage:
    
    $ python plotter.py file.log

    Warning:
     - In the process of being deprecated. Not sure how much longer this tool will be usable.
     - See: tools/arcproc.py
"""

import os
import sys
import getopt
from os.path import basename
import pts
import numpy as np
from numpy import array # needed for eval to work

def sub(s, old, new):
    return new.join(s.split(old))

class Usage(Exception):
    def __init__(self, msg):
        self.msg = msg

plot_str = """
#set term postscript portrait color
#set output "plots.ps"

set multiplot

set size 1, 0.333
#set style point 1 linewidth 5

set style data linespoints

set yrange [:0.1]
set ylabel "Energy per bead"
set origin 0,0.67
%(pre_energy)s
plot [1:] %(energyplots)s, %(growplots)s, %(resplots)s, %(cbplots)s
%(post_energy)s

set yrange [*:*]
set nokey
set origin 0,0.333 
set logscale y

set ylabel "Max Stepsize"
plot [1:] %(stepplots)s

set origin 0, 0.166666
set size 1, 0.166666
set logscale y
set ylabel "RMS Force"
plot [1:] %(gradientplots)s

set origin 0, 0
set ylabel "Max Force"
set xlabel "Bead Gradient Evaluations"
plot [1:] %(maxfplots)s


"""
#aseneb.txt.arch" using 2:4 title "ASE NEB", "myneb_aselbfgs-newe.txt.arch" using 2:4 title "My NEB (ASE-LBFGS)", "myneb_numpy

def run(args, extra, maxit=500, known_ts_aa_dists = 0):

    archive_tag = 'Archive'
    ts_tag = 'TS ESTIM CARTS:'
    max_tag = 'HIGHEST CARTS:'

    titles = []
    for fn in args:
        f = open(fn)
        f_out = open(fn + '.out', 'w')

        fn_grow = fn + '.grow.out'
        f_grow = open(fn_grow, 'w')

        fn_res = fn + '.res.out'
        f_res = open(fn_res, 'w')

        fn_cb = fn + '.cb.out'
        f_cb = open(fn_cb, 'w')

        title = fn.split('/')[-1].split('.')[0]
        titles.append(title)

        highestN = 0
        prev = 0., 0., 0.
        prev_bc = 0
        prev_res = 0
        prev_cb = 0

        ts_err_ts_estim = 0
        ts_err_max = 0
        ts_count = 0

        count = 0
        while True:
            line = f.readline()
            if not line:
#                maxit = min(maxit, highestN)
                break

            if line.startswith(archive_tag):
                d = ' '.join(line.split(' ')[1:])
                d = eval(d)
                # bc:   bead count
                # N:    no iterations
                # res:  no respaces
                # cb:   no callbacks
                # rmsf: rms forces
                # e:    energy
                # maxe: max energy
                # s:    step size
                # Nb:   number of bead gradient calls
                bc = d['bc']
                N = d['N']
                res = d['resp']
                cb = d['cb']
                rmsf = d['rmsf']
                maxf = d.get('maxf', rmsf)
                e = d['e']
                maxe = d['maxe']
                s = d['s']
                s_ts_cumm = d['s_ts_cumm']
                s_max_cumm = d['s_max_cumm']
                ixhigh = d['ixhigh']
                Nb = d.get('bead_gs', N)

                if e == 0:
                    continue

                if prev_bc != bc:
                    div = max(np.abs(e), 1)
                #print div
                e = e + div


                # get errors between estimated TSs and known TS
                ts_estim_err = 1
                ts_max_err = 1
                if 'ts_estim_carts' in d:
                    a_ts_estim = eval(d['ts_estim_carts'][1])
                    aa_dists_ts_estim = pts.common.atom_atom_dists(a_ts_estim)
                    ts_estim_err = np.linalg.norm(known_ts_aa_dists - aa_dists_ts_estim)
                    #print "ts_estim_err", ts_estim_err

                if 'bead_carts' in d:
                    a_max = eval(d['bead_carts'])
                    a_max.sort()
                    a_max = a_max[-1][1]
                    aa_dists_max = pts.common.atom_atom_dists(a_max)
                    ts_max_err = np.linalg.norm(known_ts_aa_dists - aa_dists_max)
                    #print "ts_max_err", ts_max_err

                tuple = (bc,   N,   Nb,   res,   cb,   maxf,   rmsf,   e,   maxe,   s,   e/bc, ts_estim_err, ts_max_err, s_max_cumm)

                # Set up dictionary of indices of fields so that strings of 
                # gnuplot syntax can access them.
                ids =  ['bc', 'N', 'Nb', 'res', 'cb', 'maxf', 'rmsf', 'e', 'maxe', 's', 'e_bc', 'ts_estim_err', 'ts_max_err', 's_max_cumm']
                ixs = range(len(ids)+1)[1:] # [1,2,3...]
                d = dict(zip(ids, ixs))


                if (rmsf, e, maxe) != prev:
                    outline = "%d\t%d\t%d\t%d\t%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n" % tuple
                    f_out.write(outline)
                    prev = rmsf, e, maxe
                    highestN = max(N, highestN)
                else:
                    pass#assert False, "repeated data in output file, this must not occur if getting of cartesian geom is to work correctly."

                # file of growth events
                if bc > prev_bc and N > 1:
                    f_grow.write(outline)

                # file of respace events
                if res > prev_res and N > 1:
                    f_res.write(outline)

                # file of callback events
                if cb > prev_cb and N > 1:
                    f_cb.write(outline)

                prev_bc = bc
                prev_res = res
                prev_cb = cb

        f.close()
        f_res.close()
        f_out.close()
        f_cb.close()
        f_grow.close()

    plot_files = ['"' + fn + '.out"' for fn in args]
    energy_plots = [fn + ' using %(Nb)d:%(e_bc)d with lines' % d for fn in plot_files]
    energy_plots = [p + ' title "' + t + '"' for (p,t) in zip(energy_plots, titles)]
    energy_plots = ','.join(energy_plots)

    grow_plots = ['"' + fn + '.grow.out"' for fn in args]
    titles = ['title "Growth"'] + ['notitle' for i in grow_plots[1:]]
    energy = '%(Nb)d:%(e_bc)d' % d
    grow_plots = ['%s using %s %s with points lw 5 lt 7' % (fn,energy,t) for (fn,t) in zip(grow_plots, titles)]
    grow_plots = ','.join(grow_plots)

    res_plots = ['"' + fn + '.res.out"' for fn in args]
    titles = ['title "Respace"'] + ['notitle' for i in res_plots[1:]]
    res_plots = ['%s using %s %s with points lw 3 lt 9' % (fn,energy,t) for (fn,t) in zip(res_plots, titles)]
    res_plots = ','.join(res_plots)

    cb_plots = ['"' + fn + '.cb.out"' for fn in args]
    titles = ['title "Callback"'] + ['notitle' for i in cb_plots[1:]]
    cb_plots = ['%s using %s %s with points lw 1 lt 11' % (fn,energy,t) for (fn, t) in zip(cb_plots, titles)]
    cb_plots = ','.join(cb_plots)


    gradient_plots = [fn + ' using %(Nb)d:%(rmsf)d' % d for fn in plot_files]
    gradient_plots = ','.join(gradient_plots)

    step_plots = [fn + ' using %(Nb)d:%(s_max_cumm)d' % d for fn in plot_files]
    step_plots = ','.join(step_plots)

    maxf_plots = [fn + ' using %(Nb)d:%(maxf)d' % d for fn in plot_files]
    maxf_plots = ','.join(maxf_plots)

    ts_max_err_plots = [fn + ' using %(Nb)d:%(ts_max_err)d' % d for fn in plot_files]
    ts_max_err_plots = ','.join(ts_max_err_plots)

    values = {'maxit': maxit, 
              'energyplots': energy_plots, 
              'gradientplots': gradient_plots, 
              'stepplots': step_plots, 
              'growplots': grow_plots,
              'resplots': res_plots,
              'cbplots': cb_plots,
              'ts_estim_err': ts_max_err_plots,
              'ts_max_err': ts_max_err_plots,
              'maxfplots': maxf_plots}

    values.update(extra)
    gnuplot_str = plot_str % values

    gpfile = '_'.join([basename(a) for a in args]) + '.gp'
    out = open(gpfile, 'w')
    out.write(gnuplot_str)
    out.close()
    os.system('gnuplot -persist ' + gpfile)
#    os.system('gv plots.ps')

def file2carts(f):
    s = pts.common.file2str(f)
    if pts.coord_sys.XYZ.matches(s):
        return pts.coord_sys.XYZ(s).get_cartesians()
    elif pts.coord_sys.ZMatrix.matches(s):
        return pts.coord_sys.ZMatrix(s).get_cartesians()
    else:
        raise False, "Unrecognised file format in:\n" + s

def main(argv=None):
    known_ts_aa_dists = 0

    if argv is None:
        argv = sys.argv
    try:
        try:
            opts, args = getopt.getopt(argv[1:], "hp:", ["help", "ts="])
        except getopt.error, msg:
             raise Usage(msg)
        
        if len(args) < 1:
            raise Usage(__doc__)

        extra = {'pre_energy': '', 'post_energy': ''}
        for o, a in opts:
            if o == '-p':
                section, code = a.split('=')
                extra[section] = code
            elif o == '--ts':
                known_ts_file = a
                known_ts_aa_dists = pts.common.atom_atom_dists(pts.common.file2carts(known_ts_file))
                #print "known_ts_aa_dists", known_ts_aa_dists
                
        run(args, extra, known_ts_aa_dists=known_ts_aa_dists)

    except Usage, err:
        print >>sys.stderr, err.msg
        print >>sys.stderr, "for help use --help"
        return 2

if __name__ == "__main__":
    sys.exit(main())



