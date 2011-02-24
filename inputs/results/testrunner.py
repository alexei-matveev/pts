#!/usr/bin/env python

import os
from os.path import join as pjoin
import sys
from glob import glob

args = sys.argv[1:]

methods = ['s','ss','gs','neb', 's_lbfgs']
coordss = ['xyz', 'zmt']
assert len(args) == 2 and args[0] in methods and args[1] in coordss, "Usage: %s %s %s" % (sys.argv[0], '|'.join(methods), '|'.join(coordss))
method = args[0]
coords = args[1]

def rmext(s):
    return ''.join(s.split('.')[:-1])

results_dir = method + '-' + coords
os.system('mkdir %s' % results_dir)

inputs_dir = '/home/hugh/src/sem-global-ts/inputs'
params_template = '/home/hugh/src/sem-global-ts/inputs/results/params.py'

def vsys(s, dry=False):
    print s
    if dry:
        os.system('touch DRY_RUN')
    else:
        os.system(s)

class Test:
    
    def __init__(self, test, reactprod, params_generator, dry_run=False):

        self.input_dir = pjoin(inputs_dir, test)
        print self.input_dir
        self.re, self.pr = reactprod

        self.name = os.path.basename(test) + '_' + rmext(self.re) + '-' + rmext(self.pr)

        self.re = pjoin(self.input_dir, self.re)
        self.pr = pjoin(self.input_dir, self.pr)

        self.command = os.path.abspath('/home/hugh/src/sem-global-ts/inputs/gensearch.py')
        self.params_gen = params_generator
        self.dry_run = dry_run
        self.initdir = os.getcwd()

        self.exe = lambda s: vsys(s, dry_run)

    def run(self, N):
        self.name = 'test' + str(N) + '_' + self.name
        print '=========================================================='
        print self.name
        print '=========================================================='

        # create / change into dir
        self.outdir = pjoin(results_dir, self.name)
        os.system("mkdir %s" % self.outdir)
        print self.outdir
        os.chdir(self.outdir)

        # generate parameters file
        self.params = self.params_gen(params_template)

        """output_files = ['*.log', '*.pickle', 'cpu_occupation_timing.txt']

        # backup old files if they exist
        old_files = [i for s in [glob(pat) for pat in output_files] for i in s]
        if old_files == []:
            self.exe('mkdir old')
            self.exe('mv *.log *.pickle cpu_occupation_timing.txt old/')"""

        cmd = "python %(command)s --params %(params)s %(re)s %(pr)s > %(name)s-stdout.log" % self.__dict__
        print 'Executing', cmd, 'in', os.getcwd()
        self.exe(cmd)

        # create results dir

        #cmd = "mv *.log *.pickle cpu_occupation_timing.txt %(outdir)s" % self.__dict__

        #self.exe(cmd)
        self.exe("rm -r tmp")

        os.chdir(self.initdir)


def gen_params_func(*args):
    """Returns a function that takes a filename of the params template, generates a new params file, and eturns the name of it."""

    methods_count = sum([m in args for m in methods])
    assert methods_count == 1, "NO!    %d methods selected." % methods_count

    have_method = False

    overrides=['# Begin per-simulation params']
    for a in args:
        if a == 'xyz':
            overrides.append('force_cart_opt=True')
            overrides.append("extra_opt_params = {'alpha': 200}")

        elif a in ('s', 'gs', 'ss'):
            overrides.append("opt_type = 'multiopt'")
            have_method = True
            if a == 's':
                overrides.append("method = 'string'")
            elif a == 'ss':
                overrides.append("method = 'searchingstring'")
            elif a == 'gs':
                overrides.append("method = 'growingstring'")

        elif a == 's_lbfgs':
            overrides.append("method = 'string'")
            overrides.append("opt_type = 'ase_lbfgs'")
            overrides.append("extra_opt_params = {'backtracking': 3, 'alpha': 200}")
            have_method = True

        elif a == 'neb':
            overrides.append("opt_type = 'ase_lbfgs'")
            overrides.append("method = 'neb'")
            overrides.append("extra_opt_params = {'backtracking': 3, 'alpha': 200}")
            have_method = True
        elif '=' in a:
            overrides.append(a)
        else:
            assert False

    assert have_method
    overrides.append('# End per-simulation params')

    def write_file(name):
        f_in = open(name)
        f_out_name = name + '.tmp'
        f_out = open(f_out_name, 'w')
        f_in_contents = f_in.read()
        f_out.write(f_in_contents)
        f_out.write('\n'.join(overrides))
        f_in.close()
        f_out.close()
        return f_out_name
        
    return write_file

print "Running preliminary test..."
test0 = Test('LJ', ('1.t', '2.t'), gen_params_func('maxit=4', 'beads_count=7', 'xyz', method))
test0.run(0)

defaults = ['maxit=100', 'beads_count=7', coords, method]

tests = [

         (Test('ethane-hydroxide', ('reactants.zmt', 'products.zmt'), gen_params_func(*defaults)),  True),

         (Test('alanine-dipeptide', ('adp-c5.zmt', 'adp-c7ax.zmt'), gen_params_func(*defaults)),    False),

         (Test('dielsalder', ('re.zmt', 'pr.zmt'), gen_params_func(*defaults)),                     False),

         (Test('dielsalder', ('re-stack.zmt', 'pr-stack.zmt'), gen_params_func(*defaults)),         False),

         (Test('Mn_H_ib_tbut', ('re.xyz', 'pr.xyz'), gen_params_func(*defaults)),                   False)

         ]

for i, (t, b) in enumerate(tests):
    if b:
        t.run(i+1)
    else:
        print "Skipping", (i+1)


