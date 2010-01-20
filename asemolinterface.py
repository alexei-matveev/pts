from __future__ import with_statement

import scipy
import re
import thread
import logging
import threading
import string
import os
import sys
from copy import deepcopy
from subprocess import Popen, STDOUT
import pickle

import numpy

import aof.common as common
import aof.coord_sys as csys

lg = logging.getLogger('aof.asemolinterface') #common.PROGNAME)

numpy.set_printoptions(linewidth=180)

class MolInterfaceException(Exception):
    def __init__(self, msg):
        self.msg = msg
    def __str__(self):
        return self.msg
 
class MolInterface:
    """Interface between optimisation coordinates and CoordSys object, 
    providing also functionality to run a energy/gradient calculation of a
    particular vector under a separate python interpreter instance."""

    def __init__(self, mol_strings, params = dict()):
        """mol_strings: list of strings, each of which describes a molecule, 
        format can be z-matrix or xyz format, but formats must be consistent."""

        assert len(mol_strings) > 1
        assert "calculator" in params

        first = mol_strings[0]
        if csys.ZMatrix.matches(first):
            mols = [csys.ZMatrix(s) for s in mol_strings]
        elif csys.XYZ.matches(first):
            mols = [csys.XYZ(s) for s in mol_strings]
        elif csys.ComplexCoordSys.matches(first):
            mols = [csys.parse_and_return(s) for s in mol_strings]
        else:
            raise MolInterfaceException("Unrecognised geometry string:\n" + first)

        # used to number input files as they are created and run
        self.job_counter = 0
        self.job_counter_lock = thread.allocate_lock()
        self.build_coord_sys_lock = threading.RLock()

        # lists of various properties for input reagents
        atoms_lists    = [m.get_chemical_symbols() for m in mols]
        coord_vec_lens = [len(m.get_internals()) for m in mols]

        all_var_names = [m.var_names for m in mols]
        all_dih_vars = [m.dih_vars for m in mols]

        if not common.all_equal(atoms_lists):
            raise MolInterfaceException("Input molecules do not have consistent atoms.")

        elif not common.all_equal(coord_vec_lens):
            raise MolInterfaceException("Input molecules did not have a consistent number of variables.")

        if not common.all_equal(all_var_names):
            raise MolInterfaceException("Input molecules did not have the same variable names.")
        
        if not common.all_equal(all_dih_vars):
            raise MolInterfaceException("Input molecules did not have the same dihedral variables.")

        self.var_names = all_var_names[0]

        [m.set_var_mask(params['mask']) for m in mols]
        self.reagent_coords = [m.get_internals() for m in mols]
        print self.reagent_coords[0]

        # Make sure that when interpolating between the dihedral angles of reactants 
        # and reagents, that this is done using the shortest possible arc length
        # around a circle. This only needs to be done for dihedrals, but this is
        # implicitely asserted by the inequality tested for below (I think).

        # It's not done if a transition state is specified, since in this case,
        # we don't want to interpolate via the shortest arc on the circle but 
        # rather via the given TS.
        if len(self.reagent_coords) == 2:
            react = self.reagent_coords[0]
            prod  = self.reagent_coords[1]
            for i in range(len(react)):
                if self.var_names[i] in mols[0].dih_vars:
                    if abs(react[i] - prod[i]) > 180.0 * common.DEG_TO_RAD:
#                        assert self.var_names[i] in self.mols[0].dih_vars
                        if react[i] > prod[i]:
                            prod[i] += 360.0 * common.DEG_TO_RAD
                        else:
                            react[i] += 360.0 * common.DEG_TO_RAD

        # setup function that generates
        self.place_str = None
        if 'placement' in params and params['placement'] != None:
            f = params['placement']
            assert callable(f), "Function to generate placement command was not callable."

            # perform test to make sure command is in path
            command = f(None)
            assert common.exec_in_path(command), "Generated placement command was not in path."
            self.place_str = f


        a,b,c,d = params["calculator"]
        self.calc_tuple = a,b,c
        self.pre_calc_function = d

        self.mol = mols[0]

        if 'cell' in params and params['cell'] != None:
            self.mol.set_cell(params['cell'])
        if 'pbc' in params and params['pbc'] != None:
            self.mol.set_pbc(params['pbc'])


    def __str__(self):
        mystr = "format = " + self.mol.__class__.__name__
        mystr += "\natoms = " + str(self.mol.get_chemical_symbols())
        mystr += "\nvar_names = " + str(self.var_names)
        mystr += "\nreactant coords = " + str(self.reagent_coords[0])
        mystr += "\nproduct coords = " + str(self.reagent_coords[1])
        return mystr

    def __repr__(self):
        return "MolInterface: Writeme: __repr__()"

    def geom_checker(self, coords):
        """Not Yet Implemented.
        
        Checks that coords will generate a chemically reasonable 
        molecule, i.e. no overlap."""
        assert False, "Not yet implemented"
        return True

    def build_coord_sys(self, v, calc_kwargs=None):
        """Builds a coord sys object with internal coordinates given by 'v' 
        and returns it."""

        with self.build_coord_sys_lock:
            m = self.mol.copy()
            m.set_internals(v)
            tuple = deepcopy(self.calc_tuple)

            if calc_kwargs:
                assert type(tuple[2]) == dict
                assert type(calc_kwargs) == dict
                tuple[2].update(calc_kwargs)

            m.set_calculator(tuple)
            return m

    def run(self, item):

        job = item.job

        tmp_dir = common.get_tmp_dir()

        # job_name will be related to the bead number if given
        ix = self.__get_job_counter()
        if job.num_bead != None:
            ix = job.num_bead
        job_name = "beadjob%2.2i" % ix
        item.job_name = job_name

        mol_pickled = os.path.join(tmp_dir, job_name + common.INPICKLE_EXT)
        ase_stdout_file = os.path.join(tmp_dir, job_name + ".stdout")
        results_file = job_name + common.OUTPICKLE_EXT
        results_file = os.path.join(tmp_dir, results_file)

        # compile package of extra data
        extra_data = dict()
        extra_data['item'] = item
        function = self.pre_calc_function

        # write input file as pickled object
        coord_sys_obj = self.build_coord_sys(job.v)
        f = open(mol_pickled, "wb")
        packet = coord_sys_obj, function, extra_data
        pickle.dump(packet, f)
        f.close()

        cmd = ["python", "-m", "aof.pickle_runner", mol_pickled]

        # Generate placement command, e.g. for dplace
        if callable(self.place_str):
            placement = self.place_str(item.tag)
            cmd = placement.split() + cmd
            lg.info("Running with placement command %s" % placement)
        print "Final command", ' '.join(cmd)
        p = Popen(cmd, stdout=open(ase_stdout_file, "w"), stderr=STDOUT)

        (pid, ret_val) = os.waitpid(p.pid, 0)
        if ret_val != 0:
            raise MolInterfaceException("pickle_runner.py returned with " + str(ret_val)
                + "\nwhen attempting to run " + ' '.join(cmd)
                + "\nMake sure $PYTHONPATH contains " + sys.path[0] 
                + "\n" + common.file2str(ase_stdout_file))

        # load results from file
        (e, g, dir) = pickle.load(open(results_file, "r"))

        return common.Result(job.v, e, g, dir=dir)

    def run_internal(self, job):
        """Used to return results from analytical potentials."""

        coords = job.v
        e1 = self.analytical_pes.energy(coords)

        g1 = self.analytical_pes.gradient(coords)
        r = common.Result(coords, e1, gradient=g1)
        return r

    def __get_job_counter(self):
        """Get unique numeric id for a job. Must be threadsafe."""

        self.job_counter_lock.acquire()
        counter = self.job_counter
        self.job_counter += 1
        self.job_counter_lock.release()

        return counter

