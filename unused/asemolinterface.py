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
import path

import numpy
from pts.metric import setup_metric

import pts.common as common
from pts.coord_sys import CoordSys

lg = logging.getLogger('pts.asemolinterface') #common.PROGNAME)

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

    def __init__(self, atoms, fun, mol_strings, **kwargs):

        """mol_strings: list of strings, each of which describes a molecule, 
        format can be z-matrix or xyz format, but formats must be consistent.

        >>> atoms = None
        >>> from pts.zmat import ZMat
        >>> fun = ZMat([(), (0,), (1,0), (2, 1, 0)])
        >>> zmt1 = [1., 1.2, 109. * common.DEG_TO_RAD,  1,
        ...   109. * common.DEG_TO_RAD, 60. * common.DEG_TO_RAD]
        >>> zmt2 = [1., 1.2, 109. * common.DEG_TO_RAD,  1,
        ...   109. * common.DEG_TO_RAD, -60. * common.DEG_TO_RAD]
        >>> zmt3 = zmt1
        >>> zmt4 = [1., 1.2, -109. * common.DEG_TO_RAD,  1,
        ...   109. * common.DEG_TO_RAD, -121. * common.DEG_TO_RAD]

        FIXME: This description is outdated:
        Note: zmt3 has negated variable, and is technically a different zmatrix to 
        zmt1; this should raise an error at some point.
        (test not yet implemented however)

        >>> mi = MolInterface(atoms, fun, [zmt1, zmt3])

        FIXME: outdated
        TODO: need to add an extra test so that this fails.

        >>> mi = MolInterface(atoms, fun, [zmt1, zmt2])
        
        >>> mi = MolInterface(atoms, fun, [zmt1, zmt4])

        There is no more checking if for a special coordinate transformation the
        dihedral angles are below 180 (as there is no way to know if there are
        dihedral angles at all)
        >>> rc = mi.reagent_coords
        >>> dih1, dih2 = rc[0][-1], rc[1][-1]
        >>> numpy.abs(dih1 - dih2) * common.RAD_TO_DEG < 180
        False

        >>> def too_bigs(m):
        ...     r = m.reagent_coords
        ...     l = [numpy.abs(r[0][-1] - r[1][-1])]
        ...     return [i for i in l if i * common.RAD_TO_DEG >= 180]

        >>> from cfunc import Mergefuncs, Cartesian
        >>> from numpy import array
        >>> fun2 = Mergefuncs([Cartesian(), fun], [3,6])
        >>> ccs1 =  array([0. ,0., 0., 1., 1.2, 109. * common.DEG_TO_RAD,  1,
        ...            109. * common.DEG_TO_RAD, 60. * common.DEG_TO_RAD])
        >>> ccs2 =  array([0. ,0., 0., 1., 1.2, 109. * common.DEG_TO_RAD,  1,
        ...            109. * common.DEG_TO_RAD, -60. * common.DEG_TO_RAD])
        >>> mi = MolInterface(atoms, fun2, [ccs1, ccs2])
        >>> too_bigs(mi)
        []

        # MolInterface does not know anything about the used function, thus all
        # kind of checking for consistency is disabled and has to be done outside
        # this module
       #>>> mi = MolInterface([ccs1, ccs3])
       #Traceback (most recent call last):
       #   ...
       #MolInterfaceException: Input molecules do not have consistent atoms.

       #>>> mi = MolInterface([ccs3, ccs4])
       #>>> too_bigs(mi)
       #[]

        """

        assert len(mol_strings) > 1

        mols = [CoordSys(atoms, fun, s) for s in mol_strings]

        # Make sure that when interpolating between the dihedral angles
        #  that this is done using the shortest possible arc length
        # around a circle. This is also done for the rotation part (quaternions)
        # of the rotation and translation objects.
        # the shortest way is always enforced between two succeding geometries,
        # thus the long way can be gotten with more geometries in between
        # this is done before the first path is generated by changing the internal
        # coordinates ot the mols if needed. Changes in the geometries for generating
        # a new path are allowed to overstep this.


        self._kwargs = kwargs
        calculator = kwargs.get('calculator', None)
        pre_calc_function = kwargs.get('pre_calc_function', None)
        name       = kwargs.get('name', None)
        self.output_path = kwargs.get('output_path', None)

        # used to number input files as they are created and run
        self.job_counter = 0
        self.job_counter_lock = thread.allocate_lock()
        self.build_coord_sys_lock = threading.RLock()


        # setup function that generates
        self.place_str = None

        self.reagent_coords = [m.get_internals() for m in mols]

        self.mol = mols[0]

        #We need an object for changing the metric. Lateron we want to distinguish between
        # contra- and covariant vectors, therefore we need some way to change them
        # the function given as argument is used to transform from whatever coordinates
        # we are dealing with to Cartesian ones, where changing between contra- and covaraint
        # vectors is straight forward
        # Here we setup the metric class in the metric module.
        # Lateron it can then be used by any module.
        setup_metric(self.mol.int2cart)

    def __str__(self):
        mystr = "format = " + self.mol.__class__.__name__
        mystr += "\natoms = " + str(self.mol._atoms.get_chemical_symbols())
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
            m = deepcopy(self.mol)
            m._atoms = deepcopy(self.mol._atoms)
            m.set_internals(v)

           #if calc_kwargs:
           #    assert type(tuple[2]) == dict
           #    assert type(calc_kwargs) == dict
           #    tuple[2].update(calc_kwargs)

            return m

    def run(self, item):

        job = item.job

        tmp_dir = common.get_tmp_dir()

        # job_name will be related to the bead number if given
        ix = self.__get_job_counter()
        if job.num_bead != None:
            ix = job.num_bead
        if self.output_path != None:
            job_name = self.output_path + "/" + "beadjob%2.2i" % ix
        else:
            job_name = "beadjob%2.2i" % ix
        item.job_name = job_name

        mol_pickled = os.path.join(tmp_dir, job_name + common.INPICKLE_EXT)
        ase_stdout_file = os.path.join(tmp_dir, job_name + ".stdout")
        results_file = job_name + common.OUTPICKLE_EXT
        results_file = os.path.join(tmp_dir, results_file)

#       print "HERE", tmp_dir, job_name
        # compile package of extra data
        extra_data = dict()
        extra_data['item'] = item

        # write input file as pickled object
        coord_sys_obj = self.build_coord_sys(job.v)
        f = open(mol_pickled, "wb")
        packet = coord_sys_obj, extra_data
        pickle.dump(packet, f, protocol=2)
        f.close()

        cmd = ["python", "-m", "pts.pickle_runner", mol_pickled]

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

