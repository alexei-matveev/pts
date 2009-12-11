#!/usr/bin/env python

import sys
import getopt
import os
import aof.common as common
import aof.coord_sys as coord_sys
import pickle
#from aof.sched import Item
#from common import Job.G

import ase

def usage():
    print "Usage: " + sys.argv[0] + " [options] calculator.pickle molecule.pickle"
    print "       -o: optimise"

class PickleRunnerException(Exception):
    def __init__(self, msg):
        self.msg = msg

def main(argv=None):
    if argv is None:
        argv = sys.argv
    try:
        try:
            opts, args = getopt.getopt(argv[1:], "ho", ["help", "optimise"])
        except getopt.error, msg:
             raise PickleRunnerException(msg)
        
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

        if len(args) != 1:
            raise PickleRunnerException("Exactly one input file must be given.")

#        calc_filename = os.path.abspath(args[0])
        mol_filename = os.path.abspath(args[0])

#        calc_pickled = open(calc_filename, "rb")
        mol_pickled = open(mol_filename, "rb")

        print "About to unpickle"
        # create atoms object based on pickled inputs
        mol, f, data = pickle.load(mol_pickled)

        print "mol", str(mol)

        if not isinstance(mol, coord_sys.CoordSys):
            raise PickleRunnerException("De-pickled molecule was not an instance of aof.coord_sys.CoordSys: " + str(type(mol)))

        if not mol.get_calculator():
            raise PickleRunnerException("Molecule object had no calculator.")

        jobname =  mol_filename.split(".")[0]

        # setup directories, filenames
        isolation_dir = os.path.join("isolation_" + os.path.basename(jobname))
        print "isolation_dir", isolation_dir

        old_dir = os.getcwd()

        # if a tmp directory is specified, then use it
        tmp_dir = common.get_tmp_dir()
        os.chdir(tmp_dir)

        # Create/change into isolation directory. This directory holds temporary files
        # specific to a computation, not including input and output files.
        if not os.path.exists(isolation_dir):
            os.mkdir(isolation_dir)
        os.chdir(isolation_dir)


        # Perform final tasks, e.g. copy the 
        # WAVECAR or blah.chk file here.
        if f != None:
            if not callable(f):
                raise PickleRunnerException("Supplied function was neither callable or None.")
            function(mol.get_calculator(), data)


        result_file = os.path.join(tmp_dir, jobname + common.OUTPICKLE_EXT)

        if mode == "calc_eg":
            # run job using ASE calculator
            print "Running pickle job in", os.getcwd()

            print "isolation_dir", isolation_dir
            print "type(mol._atoms.calc)", type(mol._atoms.calc)
            if 'jobname' in mol._atoms.calc.__dict__:
                mol._atoms.calc.jobname = isolation_dir
            g = -mol.get_forces(flat=True)
            assert len(g.shape) == 1
            e = mol.get_potential_energy()

            os.chdir(old_dir)

            result = (e, g)

            pickle.dump(result, open(result_file, "w"))

            # just for testing...
            print pickle.load(open(result_file, "r"))

        elif mode == "optimise":
            optim = ase.LBFGS(mol, trajectory='opt.traj')
            optim.run(steps=10)
            os.chdir(old_dir)

            ase.write(result_file, mol._atoms, format="traj")
        else:
            raise PickleRunnerException("Unrecognised mode: " + mode)

    except PickleRunnerException, err:
        print >>sys.stderr, err.msg
        print >>sys.stderr, "for help use --help"
        return 2
    except IOError, err:
        print >>sys.stderr, err
        return -1

if __name__ == "__main__":
    sys.exit(main())


