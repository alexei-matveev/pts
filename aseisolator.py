#!/usr/bin/env python

import sys
import getopt
import os
import common
import pickle

import ase


def usage():
    print "Usage: " + sys.argv[0] + " ase_settings.py atoms.xyz"

class ASEIsolatorException(Exception):
    def __init__(self, msg):
        self.msg = msg

def main(argv=None):
    if argv is None:
        argv = sys.argv
    try:
        try:
            opts, args = getopt.getopt(argv[1:], "h", ["help"])
        except getopt.error, msg:
             raise ASEIsolatorException(msg)
        
        for o, a in opts:
            if o in ("-h", "--help"):
                usage()
                return 0
            else:
                usage()
                return -1

        if len(args) != 2:
            raise ASEIsolatorException("Exactly two input files must be given.")

        ase_job_settings = os.path.abspath(args[0])
        molecule = args[1]


        # create atoms object based on molecular geometry in file
        atoms = ase.read(molecule)

        jobname =  os.path.splitext(molecule)[0]

        # setup directories, filenames
        isolation_dir = "iso_" + jobname

        old_dir = os.getcwd()


        # if a tmp directory is specified, then use it
        if common.TMP_DIR_ENV_VAR in os.environ:
            tmp_dir = os.environ['common.TMP_DIR_ENV_VAR']
            abs_tmp_dir = os.path.abspath(tmp_dir)
            if not os.path.exists(abs_tmp_dir):
                os.mkdir(abs_tmp_dir)
            os.chdir(abs_tmp_dir)
        else:
            abs_tmp_dir = os.path.abspath(old_dir)

        if not os.path.exists(isolation_dir):
            os.mkdir(isolation_dir)
        os.chdir(isolation_dir)

        # set up calculators, etc.
        exec open(ase_job_settings).read()

        # Based on what was found in ase_job_settings, perform further
        # setup for 'atoms'
        atoms.set_cell(mycell)
        atoms.set_pbc(mypbc)
        atoms.set_calculator(mycalc)

        # run job using ASE calculator
        g = atoms.get_forces().flatten()
        e = atoms.get_potential_energy()

        os.chdir(old_dir)

        result = (e, g)
        result_file = os.path.join(abs_tmp_dir, jobname + common.LOGFILE_EXT)

        pickle.dump(result, open(result_file, "w"))

        # just for testing...
        print pickle.load(open(result_file, "r"))


    except ASEIsolatorException, err:
        print >>sys.stderr, err.msg
        print >>sys.stderr, "for help use --help"
        return 2
    except IOError, err:
        print >>sys.stderr, err
        return -1

if __name__ == "__main__":
    sys.exit(main())


