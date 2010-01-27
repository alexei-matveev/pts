#!/usr/bin/env python


"""Displays a couple of ASE GUIs to facilitate the ordering of atoms in pairs of molecules."""

import os
import sys
import getopt
from os.path import basename

def sub(s, old, new):
    return new.join(s.split(old))

class Usage(Exception):
    def __init__(self, msg):
        self.msg = msg

plot_str = """
set term postscript portrait color
set output "plots.ps"

set multiplot

set size 1, 0.333
#set style point 1 linewidth 5


set style data linespoints

set ylabel "Energy per bead"
set origin 0,0.67
%(pre_energy)s
plot [1:%(maxit)d] %(energyplots)s, %(growplots)s, %(resplots)s, %(cbplots)s
%(post_energy)s

set nokey
set origin 0,0.333 
set logscale y

set ylabel "RMS Stepsize
plot [1:%(maxit)d] %(stepplots)s

set origin 0, 0
set logscale y

set ylabel "RMS Forces"
set xlabel "Chain Gradient Evaluations"
plot [1:%(maxit)d] %(gradientplots)s


"""
#aseneb.txt.arch" using 2:4 title "ASE NEB", "myneb_aselbfgs-newe.txt.arch" using 2:4 title "My NEB (ASE-LBFGS)", "myneb_numpy

def run(args, extra, maxit=50):
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

        while True:
            line = f.readline()
            if not line:
#                maxit = min(maxit, highestN)
                break

            if line[0:7] == 'Archive':
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
                bc = d['bc']
                N = d['N']
                res = d['resp']
                cb = d['cb']
                rmsf = d['rmsf']
                e = d['e']
                maxe = d['maxe']
                s = d['s']
                s_ts_cumm = d['s_ts_cumm']
                ixhigh = d['ixhigh']

                tuple = (bc,   N,   res,   cb,   rmsf,   e,   maxe,   s,   e/bc)
                ids =  ['bc', 'N', 'res', 'cb', 'rmsf', 'e', 'maxe', 's', 'e/bc']
                ixs = range(len(ids)+1)[1:]
                d = dict(zip(ids, ixs))

                if (rmsf, e, maxe) != prev:
                    line = "%d\t%d\t%d\t%d\t%f\t%f\t%f\t%f\t%f\n" % tuple
                    f_out.write(line)
                    prev = rmsf, e, maxe
                    highestN = max(N, highestN)

                # file of growth events
                if bc > prev_bc and N > 1:
                    f_grow.write(line)

                # file of respace events
                if res > prev_res and N > 1:
                    f_res.write(line)

                # file of callback events
                if cb > prev_cb and N > 1:
                    f_cb.write(line)


                prev_bc = bc
                prev_res = res
                prev_cb = cb

        f.close()
        f_res.close()
        f_out.close()
        f_cb.close()
        f_grow.close()

    plot_files = ['"' + fn + '.out"' for fn in args]
    energy_plots = [fn + ' using %(N)d:%(e)d with lines' % d for fn in plot_files]
    energy_plots = [p + ' title "' + t + '"' for (p,t) in zip(energy_plots, titles)]
    energy_plots = ','.join(energy_plots)

    grow_plots = ['"' + fn + '.grow.out"' for fn in args]
    titles = ['title "Growth"'] + ['notitle' for i in grow_plots[1:]]
    energy = '%(N)d:%(e)d' % d
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


    gradient_plots = [fn + ' using %(N)d:%(rmsf)d' % d for fn in plot_files]
    gradient_plots = ','.join(gradient_plots)

    step_plots = [fn + ' using %(N)d:%(s)d' % d for fn in plot_files]
    step_plots = ','.join(step_plots)

    values = {'maxit': maxit, 
              'energyplots': energy_plots, 
              'gradientplots': gradient_plots, 
              'stepplots': step_plots, 
              'growplots': grow_plots,
              'resplots': res_plots,
              'cbplots': cb_plots}

    values.update(extra)
    gnuplot_str = plot_str % values

    gpfile = '_'.join([basename(a) for a in args]) + '.gp'
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
            opts, args = getopt.getopt(argv[1:], "hp:", ["help"])
        except getopt.error, msg:
             raise Usage(msg)
        
        if len(args) < 1:
            raise Usage("Must specify at least one file")

        extra = {'pre_energy': '', 'post_energy': ''}
        for o, a in opts:
            if o == '-p':
                section, code = a.split('=')
                extra[section] = code
                
        run(args, extra)

    except Usage, err:
        print >>sys.stderr, err.msg
        print >>sys.stderr, "for help use --help"
        return 2

if __name__ == "__main__":
    sys.exit(main())



