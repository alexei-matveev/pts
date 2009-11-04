import scipy
import re
import thread
import logging
import string
import os
import sys
from copy import deepcopy
from subprocess import Popen
import pickle

import numpy

import common
import coord_sys as csys

lg = logging.getLogger('mylogger') #common.PROGNAME)

numpy.set_printoptions(linewidth=180)

class MolInterfaceException(Exception):
    def __init__(self, msg):
        self.msg = msg
    def __str__(self):
        return self.msg
 
class MolInterface:
    """Converts between molecule representations (i.e. internal xyz/zmat 
    representation), optimisation coordinates, Quantum Chemistry logfile format
    and QC program input format.
    """

    def __init__(self, mol_strings, params = dict()):
        """mol_strings: list of strings, each of which describes a molecule, 
        format can be z-matrix or xyz format, but formats must be consistent."""

        assert len(mol_strings) > 1

        first = mol_strings[0]
        if csys.ZMatrix.matches(first):
            mols = [csys.ZMatrix(s) for s in mol_strings]
        elif csys.XYZ.matches(first):
            mols = [csys.XYZ(s) for s in mol_strings]
        elif c_sys.ComplexCoordSys(first):
            mols = [csys.ComplexCoordSys(s) for s in mol_strings]
        else:
            raise MolInterfaceException("Unrecognised geometry string:\n" + first)


        # used to number input files as they are created and run
        self.job_counter = 0
        self.job_counter_lock = thread.allocate_lock()
        self.logfile2eg_lock = thread.allocate_lock()

        # lists of various properties for input reagents
        atoms_lists    = [m.get_chemical_symbols() for m in mols]
        coord_vec_lens = [len(m.get_internals()) for m in molreps]

        var_names = [m.var_names for m in mols]

        if not common.all_equal(atoms_lists):
            raise MolInterfaceException("Input molecules do not have consistent atoms.")

        elif not common.all_equal(coord_vec_lens):
            raise MolInterfaceException("Input molecules did not have a consistent number of variables.")

        if not common.all_equal(var_names):
            raise MolInterfaceException("Input molecules did not have the same variable names.")

#        self.format = formats[0]
#        self.atoms = atoms_lists[0]
#        self.natoms = len(atoms_lists[0])
        """if self.format == "zmt":
            self.var_names = var_names[0]
            self.nvariables = len(self.var_names)

            if OLD_PYBEL_CODE:
                self.zmt_spec = molreps[0].zmt_spec
            else:
                self.zmatrix = zmatrix.ZMatrix(mol_strings[0])"""

        self.reagent_coords = [m.coords for m in molreps]

        # Make sure that when interpolating between the dihedral angles of reactants 
        # and reagents, that this is done using the shortest possible arc length
        # around a circle. This only needs to be done for dihedrals, but this is
        # implicitely asserted by the inequality tested for below (I think).

        # it's not done if a transition state is specified
        if len(self.reagent_coords) == 2:
            react = self.reagent_coords[0]
            prod  = self.reagent_coords[1]
            for i in range(len(react)):
                if self.var_names[i] in self.mols.dih_vars:
                    if abs(react[i] - prod[i]) > 180.0 * common.DEG_TO_RAD:
                        assert self.var_names[i] in self.mols.dih_vars
                        if react[i] > prod[i]:
                            prod[i] += 360.0 * common.DEG_TO_RAD
                        else:
                            react[i] += 360.0 * common.DEG_TO_RAD


        # setup process placement command
        # TODO: suport for placement commands other than dplace
        self.gen_placement_command = None
        if "placement_command" in params:
            if params["placement_command"] == "dplace":
                self.gen_placement_command = self.gen_placement_command_dplace
            else:
                raise Exception("Use of " + params["placement_command"] + " not implemented")

    def __str__(self):
        mystr = "format = " + self.format
        mystr += "\natoms = " + str(self.atoms)
        if self.format == "zmt":
            mystr += "\nvar_names = " + str(self.var_names)
        mystr += "\nreactant coords = " + str(self.reagent_coords[0])
        mystr += "\nproduct coords = " + str(self.reagent_coords[1])
        if OLD_PYBEL_CODE:
            mystr += "\nzmt_spec:\n" + self.zmt_spec
        return mystr

    def __repr__(self):
        return "MolInterface: Writeme: __repr__()"
    def geom_checker(self, coords):
        """Not Yet Implemented.
        
        Checks that coords will generate a chemically reasonable 
        molecule, i.e. no overlap."""
        assert False, "Not yet implemented"
        return True

   def run(self, job):
        """Assigned by constructor to one of run_internal() or run_external(), 
        depending on input parameters."""
        assert False, "This should never run directly."

    def run_ase(self, job):
        (mystr, _) = self.coords2xyz(job.v)

        tmp_dir = common.get_tmp_dir()

        job_base_name = "asejob" + str(self.__get_job_counter())
        mol_pickled = os.path.join(tmp_dir, job_base_name + ".pickle")
        ase_stdout_file = os.path.join(tmp_dir, job_base_name + ".stdout")

        # write input file as xyz format
        coord_sys_obj = self.build_coord_sys(job.v)
        f = open(mol_pickled, "wb")
        pickle.dump(coord_sys_obj, f)
        f.close()

        cmd = ["python", "-m", "pickle_runner", calc_pickle, mol_pickled]
        p = Popen(cmd, stdout=open(ase_stdout_file, "w"))

        (pid, ret_val) = os.waitpid(p.pid, 0)
        if ret_val != 0:
            raise MolInterfaceException("pickle_runner returned with " + str(ret_val)
                + "\nwhen attempting to run " + ' '.join(cmd)
                + "\nMake sure $PYTHONPATH contains " + sys.path[0] )

        # load results from file
        (e, g) = pickle.load(open(results_file, "r"))

        return common.Result(job.v, e, g)

    def run_internal(self, job):
        """Used to return results from analytical potentials."""

        coords = job.v
        e1 = self.analytical_pes.energy(coords)

        g1 = self.analytical_pes.gradient(coords)
        r = common.Result(coords, e1, gradient=g1)
        return r

    def gen_placement_command_dplace(self, p_low, p_high):
        """Generates a placement command (including arguments) for placement on
        processors p_low to p_high."""
        return "dplace -c %d-%d" % (p_low, p_high)

    def __get_job_counter(self):
        """Get unique numeric id for a job. Must be threadsafe."""

        self.job_counter_lock.acquire()
        counter = self.job_counter
        self.job_counter += 1
        self.job_counter_lock.release()

        return counter


