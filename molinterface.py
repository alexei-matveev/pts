#import numpy
import scipy
import re
import cclib
import thread
import logging
import zmatrix
import string
from copy import deepcopy

from common import *

lg = logging.getLogger(PROGNAME)

numpy.set_printoptions(linewidth=180)

DEFAULT_GAUSSIAN03_HEADER = "# HF/3-21G force\n\ncomment\n\n0 1\n"
DEFAULT_GAUSSIAN03_FOOTER = ""
DEFAULT_GAUSSIAN03_QCINPUT_EXT = ".com"
DEFAULT_GAUSSIAN03_QCOUTPUT_EXT = ".log"
DEFAULT_GAUSSIAN03_PROGNAME = "g03"

OLD_PYBEL_CODE = False

class MolRep:
    """Object which exposes information from a molecule in a string specified 
    in z-matrix/cartesian format. Information extracted is: 
        format (xyz or zmt)
        zmt_spec (atom connectivity)
        coords (xyz or z-matrix coordinate values)
        var_names (for z-matrix)
        atoms (list of atom names)
    """
    def __init__(self, mol_text):

        xyz = re.compile(r"""(\w\w?(\s+[+-]?\d+\.\d*){3}\s*)+""")

        if zmatrix.ZMatrix.matches(mol_text):
            self.format = "zmt"

            if OLD_PYBEL_CODE:
                parts = re.search(r"(.+?\n)\s*\n(.+)", mol_text, re.S)
                self.zmt_spec = parts.group(1) # z-matrix text, specifies connection of atoms
                variables_text = parts.group(2)
                self.var_names = re.findall(r"(\w+).*?\n", variables_text)
                self.coords = re.findall(r"\w+\s+([+-]?\d+\.\d*)\n", variables_text)

                self.atoms = re.findall(r"(\w\w?).*?\n", self.zmt_spec)
                self.coords = numpy.array([float(c) for c in self.coords])

            else:
                self.zmatrix = zmatrix.ZMatrix(mol_text)
                self.var_names = self.zmatrix.var_names
                self.coords = self.zmatrix.coords
                self.dih_vars = self.zmatrix.dih_vars
                self.atoms = [a.name for a in self.zmatrix.atoms]

        elif xyz.match(mol_text) != None:
            self.format = "xyz"
            self.coords = re.findall(r"([+-]?\d+\.\d*)", mol_text)

            self.atoms = re.findall(r"(\w\w?).+?\n", mol_text)
            self.coords = numpy.array([float(c) for c in self.coords])
            self.var_names = "no variable names"
        else:
            raise MolRepException("can't understand input file:\n" + mol_text)

class MolRepException(Exception):
    def __init__(self, msg):
        self.msg = msg
    def __str__(self, msg):
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

        molreps = [MolRep(mol_str) for mol_str in mol_strings]

        # used to number input files as they are created and run
        self.job_counter = 0
        self.job_counter_lock = thread.allocate_lock()
        self.logfile2eg_lock = thread.allocate_lock()

        # lists of various properties for input reagents
        atoms_lists    = [m.atoms for m in molreps]
        formats        = [m.format for m in molreps]
        coord_vec_lens = [len(m.coords) for m in molreps]

        if not all_equal(atoms_lists):
            raise Exception("Input molecules do not have consistent atoms.")

        elif not all_equal(formats):
            raise Exception("Input molecules do not have consistent formats.")

        elif not all_equal(coord_vec_lens):
            raise Exception("Input molecules did not have a consistent number of variables.")

        elif formats[0] == "zmt":
            var_names = [m.var_names for m in molreps]
            if not all_equal(var_names):
                raise Exception("Input molecules did not have the same variable names.")

        self.format = formats[0]
        self.atoms = atoms_lists[0]
        self.natoms = len(atoms_lists[0])
        if self.format == "zmt":
            self.var_names = var_names[0]
            self.nvariables = len(self.var_names)

            if OLD_PYBEL_CODE:
                self.zmt_spec = molreps[0].zmt_spec
            else:
                self.zmatrix = zmatrix.ZMatrix(mol_strings[0])

        self.reagent_coords = [m.coords for m in molreps]

        if self.format == "zmt":
            # Make sure that when interpolating between the dihedral angles of reactants 
            # and reagents, that this is done using the shortest possible arc length
            # around a circle. This only needs to be done for dihedrals, but this is
            # implicitely asserted by the inequality tested for below (I think).

            # it's not done if a transition state is specified
            if len(self.reagent_coords) == 2:
                react = self.reagent_coords[0]
                prod  = self.reagent_coords[1]
                for i in range(len(react)):
                    if abs(react[i] - prod[i]) > 180.0:
                        assert self.var_names[i] in self.zmatrix.dih_vars
                        if react[i] > prod[i]:
                            prod[i] += 360.0
                        else:
                            react[i] += 360.0

        if "qcinput_head" in params:
            self.qcinput_head = params["qcinput_head"]
            self.qcinput_head = expand_newline(self.qcinput_head)
        else:
            self.qcinput_head = DEFAULT_GAUSSIAN03_HEADER

        if "qcinput_foot" in params:
            self.qcinput_foot = params["qcinput_foot"]
        else:
            self.qcinput_foot = DEFAULT_GAUSSIAN03_FOOTER

        if "qcinput_ext" in params:
            self.qcinput_ext = params["qcinput_ext"]
        else:
            self.qcinput_ext = DEFAULT_GAUSSIAN03_QCINPUT_EXT

        if "qcoutput_ext" in params:
            self.qcoutput_ext = params["qcoutput_ext"]
        else:
            self.qcoutput_ext = DEFAULT_GAUSSIAN03_QCOUTPUT_EXT

        # setup Quantum Chemistry package info
        self.qc_command = DEFAULT_GAUSSIAN03_PROGNAME
        self.run = self.run_external
        self.coords2moltext = self.coords2moltext_Gaussian
        self.logfile2eg = self.logfile2eg_Gaussian

        if "qc_program" in params:
            if params["qc_program"] == "g03":
                self.qc_command = DEFAULT_GAUSSIAN03_PROGNAME
                self.coords2moltext = self.coords2moltext_Gaussian
                self.logfile2eg = self.logfile2eg_Gaussian

            elif params["qc_program"] == "paragauss":
                self.qc_command = "pgwrap" # temporary
                self.coords2moltext = self.coords2moltext_ParaGauss
                self.logfile2eg = self.logfile2eg_ParaGauss

            elif params["qc_program"] == "analytical_GaussianPES":
                self.run = self.run_internal
                self.analytical_pes = GaussianPES()
            else:
                raise Exception("Use of " + params["qc_program"] + " not implemented")

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
        mystr += "\nvar_names = " + str(self.var_names)
        mystr += "\nreactant coords = " + str(self.reagent_coords[0])
        mystr += "\nproduct coords = " + str(self.reagent_coords[1])
        if OLD_PYBEL_CODE:
            mystr += "\nzmt_spec:\n" + self.zmt_spec
        return mystr

    def geom_checker(self, coords):
        """Not Yet Implemented.
        
        Checks that coords will generate a chemically reasonable 
        molecule, i.e. no overlap."""
        assert False, "Not yet implemented"
        return True

    def coords2qcinput(self, coords, path = None):
        """Generates an input file for a quantum chemistry program, returning
        the contents of the file as a string. If path is given, the file is
        written to that path."""

        mol_text = self.coords2moltext(coords)

        str = self.qcinput_head + mol_text + self.qcinput_foot + "\n"
        if path != None:
            f = open(path, 'w')
            f.write(str)
            f.close()

        return str

    def get_reagent_coords(self):
        assert False, "Not yet implemented"

    def coords2moltext(self, coords):
        assert False, "Should never be called"

    def coords2moltext_ParaGauss(self, coords):
        import gxfile

        assert False, "Not Yet Implemented"

        (s, c) = self.coords2xyz(coords)

        atnums = get_atnums(s)
        positions = get_positions(s) # or maybe use c?
        isyms = get_positions(s)
        inums = get_inums(s)
        iconns = get_iconns(s)
        ivars = get_ivars(s)

        #gxfile.gxwrite() atnums, positions, isyms, inums, iconns, ivars = 1

    def coords2opt_mol_text(self, coords):
        """Returns a string describing the molecule in terms of the coordinate
        system that the optimisation is being done in, e.g. in xyz-format
        (cartesians) or z-matrix format."""

        if self.format == "zmt":
            self.zmatrix.set_internals(coords)
            return self.zmatrix.zmt_str()
        elif self.format == "xyz":
            s, c = self.coords2xyz(coords)
            return s
        else:
            assert False, "Should never happen"
            
    def coords2moltext_Gaussian(self, coords):
        """For a set of internal coordinates, returns the string describing the 
        molecule in xyz format."""
        #TODO: remove this function and one below?
        (s, c) = self.coords2xyz(coords)
        return s

    def __opt_coords2cart_coords(self, coords):
        """Returns the cartesian coordinates based on the given set of internal 
        coordinates.
        
        Only used by OLD_PYBEL_CODE"""
        (s, c) = self.coords2xyz(coords)
        return c
    
    def coordsys_trans_matrix(self, coords):
        """Generates the n*m matrix [dXi/dCj] which represents the change of
        the cartesian coordinates Xi with respect to the optimisation 
        coordinates Cj, i <- 1..n, j <- 1..m."""

        if self.format == "zmt":
            if OLD_PYBEL_CODE:
                nd = NumDiff()
                dX_on_dC, err = nd.numdiff(self.__opt_coords2cart_coords, coords)
                lg.debug("Errors in numerical differentiation were " + str(err))
            else:
                dX_on_dC, err = self.zmatrix.dcart_on_dint(coords)

        elif self.format == "xyz":

            # transforming cartesians to cartesians => no transformation at all
            dX_on_dC = numpy.eye(len(coords))

        else:
            assert False, "Should never happen."

        return dX_on_dC

    def coords2molstr(self, coords):
        """Generates a string containing a molecule specification in Gaussian 
        Z-matrix or Gaussian XYZ format (without method keywords, i.e. only 
        the molecule specification) based on coords.
        
        Only use by OLD_PYBEL_CODE?"""

        str = ""
        if self.format == "xyz":
            for i in range(self.natoms):
                str += "%s\t%s\t%s\t%s" % (self.atom_symbols[i], coords[3*i], coords[3*i+1], coords[3*i+2])
        elif self.format == "zmt":
            if OLD_PYBEL_CODE:
                str += self.zmt_spec + "\n"
                for i in range(self.nvariables):
                    str += "%s=%s\n" % (self.var_names[i], coords[i])
            else:
                str = self.zmatrix.zmt_str()
        else:
            raise Exception("Unrecognised self.mol_rep_type")

        return str

    def coords2xyz(self, coords):
        """Generates the xyz coordinates (both as a Gaussian format molecule
        string and as a numpy array) based on a set of input coordinates 
        given in the optimisation coordinate system."""

        if OLD_PYBEL_CODE:
            str = self.coords2molstr(coords)

            if self.format == "zmt":
                (str, coords) = self.__zmt2xyz(str)
            else:
                assert False, "OLD_PYBEL_CODE and not zmt"
        else:
            if self.format == "zmt":
#                coords = self.zmatrix.int2cart(coords)
                self.zmatrix.state_mod_lock.acquire()

                self.zmatrix.set_internals(coords)
                self.zmatrix.gen_cartesian()
                str = self.zmatrix.xyz_str()

                self.zmatrix.state_mod_lock.release()

            elif self.format == "xyz":
                assert len(coords) % 3 == 0

                coord_tuples = deepcopy(coords)
                coord_tuples.shape = (-1,3)
                coord_strs = ["%f\t%f\t%f" % (x,y,z) for (x,y,z) in coord_tuples]
                str = string.join([a + "\t" + c for (a,c) in zip(self.atoms, coord_strs)], "\n")
                str += "\n"

            else:
                assert False, "Unsuported format, should never happen"

        return (str, coords)

    def __zmt2xyz(self, str):
        """Converts a string describing a molecule using the z-matrix format
        to it's cartesian coordinates representation. Returns a duple of the
        xyz format representation and an vector of floats."""

        import pybel
        header = "# hf/3-21g\n\ncomment\n\n0 1\n" # dummy gaussian header
        mol = pybel.readstring("gzmat", header + str)
        str = mol.write("xyz")
        str = "\n".join(re.split(r"\n", str)[2:])

        xyz_coords = re.findall(r"[+-]?\d+\.\d*", str)
        xyz_coords = [float(c) for c in xyz_coords]

        return (str, numpy.array(xyz_coords))

    def run(self, job):
        """Assigned by constructor to one of run_internal() or run_external(), 
        depending on input parameters."""
        assert False, "This should never run directly."

    def run_external(self, job):
        """Runs an external program to generate gradient and energy."""
        coords = job.v

        # parameters specific to the execution of the current job
        local_params = dict()
        p1 = job.processor_ix_start
        p2 = job.processor_ix_end

        # e.g. "dplace -c 0-4"
        if self.gen_placement_command != None:
            local_params["placement_command"] = self.gen_placement_command(p1, p2)

        # call qchem program
        try:
            outputfile = self.run_qc(coords, local_params)
        except Exception, e:
            lg.error("Exception thrown when calling self.run_qc: " + str(e))
            return

        # parse output file in thread-safe manner
        self.logfile2eg_lock.acquire()
        e, g = self.logfile2eg(outputfile, coords)
        self.logfile2eg_lock.release()

        return Result(coords, e, g)

    def run_internal(self, job):
        """Used to return results from analytical potentials."""

        coords = job.v
        e1 = self.analytical_pes.energy(coords)

        g1 = self.analytical_pes.gradient(coords)
        r = Result(coords, e1, gradient=g1)
        return r

    def gen_placement_command_dplace(self, p_low, p_high):
        """Generates a placement command (including arguments) for placement on
        processors p_low to p_high."""
        return "dplace -c %d-%d" % (p_low, p_high)

    def run_qc(self, coords, local_params = dict()):
        """Launches a quantum chemistry code, blocks until it has finished, 
        then returns output filename."""
        import os
        import subprocess

        # Generate id for job in thread-safe manner
        self.job_counter_lock.acquire()
        inputfile  = __name__ + "-" + str(self.job_counter) + self.qcinput_ext
        outputfile = __name__ + "-" + str(self.job_counter) + self.qcoutput_ext
        self.job_counter += 1
        self.job_counter_lock.release()

        self.coords2qcinput(coords, inputfile)

        command = self.qc_command + " " + inputfile
        
        if "placement_command" in local_params:
            command = params["placement_command"] + " " + command

        p = subprocess.Popen(command, shell=True)
        sts = os.waitpid(p.pid, 0)

        # TODO: check whether outputfile exists before returning name
        return outputfile

    def logfile2eg(self, logfilename, coords):
        """Extracts the energy and gradient from logfilename. Converts the
        gradient from cartesian coordinates to the coordinate system that the
        optimisation is performed in (which might be either internals or 
        cartesians)."""

        raise False, "Should never be called."

    def logfile2eg_ParaGauss(self, logfilename, coords):
        """ParaGauss logfile parser. See comment for logfile2eg() for general 
        information."""

        import gxfile
        atnums, xyz, isyms, inums, iconns, ivars, grads, energy = gxfile.gxread(logfilename)

        grads_opt = self.__transform(self, grads, coords)

        return (energy, grads_opt)

    def logfile2eg_Gaussian(self, logfilename, coords):
        """Gaussian logfile parser. See comment for logfile2eg() for general 
        information."""

        file = cclib.parser.ccopen(logfilename, loglevel=logging.ERROR)
        data = file.parse()

        # energy gradients in cartesian coordinates
        if hasattr(data, "grads"):
            grads_cart = data.grads[-1].flatten()
        else:
            raise Exception("No gradients found in file " + logfilename)

        # Gaussian gives gradients in Hartrees per Bohr Radius
        grads_cart *= ANGSTROMS_TO_BOHRS # conversion is temporary, will eventually change when cclib has gradients code

        if hasattr(data, "scfenergies"):
            energy = data.scfenergies[-1] / HARTREE_TO_ELECTRON_VOLTS # conversion is temporary, will eventually change when cclib has gradients code
        elif hasattr(data, "mmenergies"):
            energy = data.mmenergies[-1] / HARTREE_TO_ELECTRON_VOLTS # conversion is temporary, will eventually change when cclib has gradients code
        else:
            raise Exception("No energies found in file " + logfilename)

        lg.debug(logfilename + ": Raw gradients in cartesian coordinates: " + str(grads_cart))

        # Transform the gradients from cartesian coordinates to the 
        # optimisation (e.g. z-matrix) coordinate system.
        grads_opt = self.__transform(grads_cart, coords, logfilename)

        return (energy, grads_opt)

    def __transform(self, grads_cart, coords, logfilename):
        """Transforms the gradients from the cartesian coordinate system to the
        coordinate system of the optimisation."""

        transform_matrix = self.coordsys_trans_matrix(coords)
        if numpy.linalg.norm(transform_matrix) > 10:
            lg.warning(line() + "Enormous coordinate system derivatives" + line())
            lg.warning(logfilename + ": largest elts of trans matrix: " + str(vecmaxs(abs(transform_matrix))))

        # energy gradients in optimisation coordinates
        # Gaussian returns forces, not gradients
        """print "transform_matrix"
        print transform_matrix
        print transform_matrix.shape
        print "grads_cart"
        print grads_cart
        print grads_cart.shape"""

        grads_opt = -numpy.dot(transform_matrix, grads_cart)
        if numpy.linalg.norm(grads_opt) > 10 or numpy.linalg.norm(grads_cart) > 10:
            lg.warning(line() + "Enormous gradients" + line())
            lg.warning(logfilename + ": Largest ZMat gradients: " + str(vecmaxs(abs(grads_opt))))
            lg.warning(logfilename + ": Largest Cartesian gradients: " + str(vecmaxs(abs(grads_cart))))
            lg.warning(logfilename + ": Largest elts of trans matrix: " + str(vecmaxs(abs(transform_matrix))))

        return grads_opt

