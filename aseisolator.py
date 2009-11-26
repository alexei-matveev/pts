#!/usr/bin/env python

import sys
import getopt
import os
import common
import pickle

import ase

def usage():
    print "Usage: " + sys.argv[0] + " [options] ase_settings.py atoms.xyz"
    print "       -o: optimise"

class ASEIsolatorException(Exception):
    def __init__(self, msg):
        self.msg = msg

def main(argv=None):
    if argv is None:
        argv = sys.argv
    try:
        try:
            opts, args = getopt.getopt(argv[1:], "ho", ["help", "optimise"])
        except getopt.error, msg:
             raise ASEIsolatorException(msg)
        
        mode = "calc_eg"
        for o, a in opts:
            if o in ("-h", "--help"):
                usage()
                return 0
            if o in ("-o", "--optimise"):
                mode = "optimise"
            else:
                usage()
                return -1

        if len(args) != 2:
            raise ASEIsolatorException("Exactly two input files must be given.")

        ase_job_settings = os.path.abspath(args[0])
        molecule = os.path.abspath(args[1])


        # create atoms object based on molecular geometry in file
        atoms = ase.read(molecule)

        jobname =  os.path.splitext(molecule)[0]

        # setup directories, filenames
        isolation_dir = os.path.join("isolation_" + os.path.basename(jobname))
        print isolation_dir

        old_dir = os.getcwd()


        # if a tmp directory is specified, then use it
        tmp_dir = common.get_tmp_dir()
        os.chdir(tmp_dir)

        # Create/change into isolation directory. This directory holds temporary files
        # specific to a computation, not including input and output files.
        if not os.path.exists(isolation_dir):
            os.mkdir(isolation_dir)
        os.chdir(isolation_dir)

        # set up calculators, etc.
        exec open(ase_job_settings).read()

        # Based on what was found in ase_job_settings, perform further
        # setup for 'atoms'
        if 'mycell' in locals():
            atoms.set_cell(mycell)
        if 'mypbc' in locals():
            atoms.set_pbc(mypbc)

        if not 'mycalc' in locals():
            raise ASEIsolatorException("'mycalc' not defined in " + ase_job_settings)

        atoms.set_calculator(mycalc)

        result_file = os.path.join(tmp_dir, jobname + common.LOGFILE_EXT)

        if mode == "calc_eg":
            # run job using ASE calculator
            g = atoms.get_forces().flatten()
            e = atoms.get_potential_energy()

            os.chdir(old_dir)

            result = (e, g)

            pickle.dump(result, open(result_file, "w"))

            # just for testing...
            #print pickle.load(open(result_file, "r"))

        elif mode == "optimise":
            optim = ase.LBFGS(atoms, trajectory='opt.traj')
            optim.run()
            os.chdir(old_dir)

            ase.write(result_file, atoms, format="traj")
        else:
            raise ASEIsolatorException("Unrecognised mode: " + mode)

    except ASEIsolatorException, err:
        print >>sys.stderr, err.msg
        print >>sys.stderr, "for help use --help"
        return 2
    except IOError, err:
        print >>sys.stderr, err
        return -1

if __name__ == "__main__":
    sys.exit(main())


