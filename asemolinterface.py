from __future__ import with_statement

import scipy
import re
import thread
import logging
import threading
import string
import time
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

zmt1 = """H
O 1 oh
O 2 oo 1 ooh
H 3 ho 2 hoo 1 hooh

oh 1.
ho 1.
oo 1.2
ooh 109.
hoo 109.
hooh 60.
"""

zmt2 = """H
O 1 oh
O 2 oo 1 ooh
H 3 ho 2 hoo 1 hooh

oh  1.
ho 1.
oo 1.2
ooh 109.
hoo 109.
hooh -60.
"""

zmt3 = """H
O 1 oh
O 2 oo 1 ooh
H 3 ho 2 hoo 1 -hooh

oh  1.
ho 1.
oo 1.2
ooh 109.
hoo 109.
hooh 60.
"""

zmt4 = """H
O 1 oh
O 2 oo 1 ooh
H 3 ho 2 hoo 1 hooh

oh  1.
ho 1.
oo 1.2
ooh -109.
hoo 109.
hooh -121.
"""

zmt_template = """H
O 1 oh
O 2 oo 1 ooh
H 3 ho 2 hoo 1 hooh

oh  1.
ho 1.
oo 1.2
ooh -109.
hoo 109.
hooh %f
"""

ccs1 = r"""
xyz = "C 0. 0. 0.\n"
x = XYZ(xyz)
zmt = "H\nO 1 oh\nO 2 oo 1 ooh\nH 3 ho 2 hoo 1 hooh\n\noh 1.\nho 1.\noo 1.2\nooh 109.\nhoo 109.\nhooh 60.\n"
z = ZMatrix2(zmt, RotAndTrans())
ccs = ccsspec([x,z])
"""

ccs2 = r"""
xyz = "C 0. 0. 0.\n"
x = XYZ(xyz)
zmt = "H\nO 1 oh\nO 2 oo 1 ooh\nH 3 ho 2 hoo 1 hooh\n\noh 1.\nho 1.\noo 1.2\nooh 109.\nhoo 109.\nhooh -121.\n"
z = ZMatrix2(zmt, RotAndTrans())
ccs = ccsspec([x,z])
"""

ccs3 = r"""
xyz = "C 0. 0. 0.\n"
x = XYZ(xyz)
zmt1 = "H\nO 1 oh\nO 2 oo 1 ooh\nH 3 ho 2 hoo 1 hooh\n\noh 1.\nho 1.\noo 1.2\nooh 109.\nhoo 109.\nhooh 60.\n"
zmt2 = "H\nO 1 oh\nO 2 oo 1 ooh\nH 3 ho 2 hoo 1 hooh\n\noh 1.\nho 1.\noo 1.2\nooh 109.\nhoo 109.\nhooh 60.\n"

z1 = ZMatrix2(zmt1, RotAndTrans())
z2 = ZMatrix2(zmt2, RotAndTrans())

ccs = ccsspec([x,z1,z2])
"""

ccs4 = r"""
xyz = "C 0. 0. 0.\n"
x = XYZ(xyz)
zmt1 = "H\nO 1 oh\nO 2 oo 1 ooh\nH 3 ho 2 hoo 1 hooh\n\noh 1.\nho 1.\noo 1.2\nooh 109.\nhoo 109.\nhooh -120.1\n"
zmt2 = "H\nO 1 oh\nO 2 oo 1 ooh\nH 3 ho 2 hoo 1 hooh\n\noh 1.\nho 1.\noo 1.2\nooh 109.\nhoo 109.\nhooh 160.\n"

z1 = ZMatrix2(zmt1, RotAndTrans())
z2 = ZMatrix2(zmt2, RotAndTrans())

ccs = ccsspec([x,z1,z2])
"""


class MolInterfaceException(Exception):
    def __init__(self, msg):
        self.msg = msg
    def __str__(self):
        return self.msg
 
class MolInterface:
    """Interface between optimisation coordinates and CoordSys object, 
    providing also functionality to run a energy/gradient calculation of a
    particular vector under a separate python interpreter instance."""

    def __init__(self, 
            mol_strings, 
            calculator=None, 
            mask=None, 
            placement=None,
            pbc=None, 
            cell=None,
            name=None):

        """mol_strings: list of strings, each of which describes a molecule, 
        format can be z-matrix or xyz format, but formats must be consistent.
        
        Note: zmt3 has negated variable, and is technically a different zmatrix to 
        zmt1; this should raise an error at some point.
        (test not yet implemented however)
        >>> mi = MolInterface([zmt1, zmt3])
        TODO: need to add an extra test so that this fails.

        >>> mi = MolInterface([zmt1, zmt2])
        
        >>> mi = MolInterface([zmt1, zmt4])
        >>> rc = mi.reagent_coords
        >>> dih1, dih2 = rc[0][-1], rc[1][-1]
        >>> numpy.abs(dih1 - dih2) * common.RAD_TO_DEG < 180
        True

        >>> many = [zmt_template % dih for dih in range(-360,360,20)]
        >>> from random import choice
        >>> randoms = [[choice(many), choice(many)] for i in range(20)]
        >>> mis = [MolInterface(l) for l in randoms]
        >>> dih_pairs = [(m.reagent_coords[0][-1], m.reagent_coords[1][-1]) for m in mis]
        >>> diffs = [numpy.abs(a-b) for a,b in dih_pairs]
        >>> max(diffs) < 180
        True

        >>> def too_bigs(m):
        ...     r = m.reagent_coords
        ...     dih_ixs = [i for i in range(len(r[0])) if m.mol.kinds[i] == 'dih']
        ...     l = [numpy.abs(r[0][i] - r[1][i]) for i in dih_ixs]
        ...     return [i for i in l if i * common.RAD_TO_DEG >= 180]

        >>> mi = MolInterface([ccs1, ccs2])
        >>> too_bigs(mi)
        []

        >>> mi = MolInterface([ccs1, ccs3])
        Traceback (most recent call last):
           ...
        MolInterfaceException: Input molecules do not have consistent atoms.

        >>> mi = MolInterface([ccs3, ccs4])
        >>> too_bigs(mi)
        []

        """

        assert len(mol_strings) > 1

        first = mol_strings[0]
        if csys.ZMatrix2.matches(first):
            mols = [csys.ZMatrix2(s) for s in mol_strings]
        elif csys.XYZ.matches(first):
            mols = [csys.XYZ(s) for s in mol_strings]
        elif csys.ComplexCoordSys.matches(first):
            mols = [csys.ComplexCoordSys(s) for s in mol_strings]
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
        all_kinds = [m.kinds for m in mols]

        if not common.all_equal(atoms_lists):
            raise MolInterfaceException("Input molecules do not have consistent atoms.")

        elif not common.all_equal(coord_vec_lens):
            raise MolInterfaceException("Input molecules did not have a consistent number of variables.")

        if not common.all_equal(all_var_names):
            raise MolInterfaceException("Input molecules did not have the same variable names.")
        
        if not common.all_equal(all_kinds):
            raise MolInterfaceException("Input molecules did not have the same variable types.")

        self.var_names = all_var_names[0]

        [m.set_var_mask(mask) for m in mols]

        self.reagent_coords = [m.get_internals() for m in mols]

        N = len(m.get_internals())

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
            for i in range(N):
                if mols[0].kinds[i] == 'dih':
                    if abs(react[i] - prod[i]) > 180.0 * common.DEG_TO_RAD:
                        if react[i] > prod[i]:
                            prod[i] += 360.0 * common.DEG_TO_RAD
                        else:
                            react[i] += 360.0 * common.DEG_TO_RAD

        # setup function that generates
        self.place_str = None
        if placement != None:
            f = placement
            assert callable(f), "Function to generate placement command was not callable."

            # perform test to make sure command is in path
            command = f(None)
            assert common.exec_in_path(command), "Generated placement command was not in path."
            self.place_str = f


        if calculator != None:
            a,b,c,d = calculator
            self.calc_tuple = a,b,c
            self.pre_calc_function = d

        self.mol = mols[0]

        if cell != None:
            self.mol.set_cell(cell)
        if pbc != None:
            self.mol.set_pbc(pbc)


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

        print "HERE", tmp_dir, job_name
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
        t0 = time.time()
        p = Popen(cmd, stdout=open(ase_stdout_file, "w"), stderr=STDOUT)

        (pid, ret_val) = os.waitpid(p.pid, 0)

        t1 = time.time()
        print "Time taken to run job %s was %.1f" % (job_name, (t1 - t0))
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

# Testing the examples in __doc__strings, execute
# "python gxmatrix.py", eventualy with "-v" option appended:
if __name__ == "__main__":
    import doctest
    doctest.testmod()

# You need to add "set modeline" and eventually "set modelines=5"
# to your ~/.vimrc for this to take effect.
# Dont (accidentally) delete these lines! Unless you do it intentionally ...
# Default options for vim:sw=4:expandtab:smarttab:autoindent:syntax

