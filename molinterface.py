import numpy
import scipy
import re
import cclib
import thread
import logging
import zmatrix

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
                self.atoms = [a.name for a in self.zmatrix.atoms]

        elif xyz.match(mol_text) != None:
            self.format = "xyz"
            self.coords = re.findall(r"([+-]?\d+\.\d*)", mol_text)
            self.var_names = "no variable names"
        else:
            raise Exception("can't understand input file:\n" + mol_text)


class MolInterface:
    """Converts between molecule representations (i.e. internal xyz/zmat 
    representation), optimisation coordinates, Quantum Chemistry logfile format
    and QC program input format.
    """

    def __init__(self, mol_strings, params = dict()):

        assert len(mol_strings) > 1

        molreps = []
        for mol_str in mol_strings:
            molreps.append(MolRep(mol_str))

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
        self.logfile2eg = self.logfile2eg_Gaussian
        # TODO: setup alternate logfile2eg function
        self.qc_command = DEFAULT_GAUSSIAN03_PROGNAME
        self.run = self.run_external
        if "qc_program" in params:
            if params["qc_program"] == "g03":
                self.qc_command = DEFAULT_GAUSSIAN03_PROGNAME
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
        """For a set of internal coordinates, returns the string describing the 
        molecule in xyz format."""
        (s, c) = self.coords2xyz(coords)
        return s

    def opt_coords2cart_coords(self, coords):
        """Returns the cartesian coordinates based on the given set of internal 
        coordinates."""
        (s, c) = self.coords2xyz(coords)
        return c
    
    def coordsys_trans_matrix(self, coords):
        """Generates the n*m matrix [dXi/dCj] which represents the change of
        the cartesian coordinates Xi with respect to the optimisation 
        coordinates Cj, i <- 1..n, j <- 1..m."""

        dX_on_dC = numdiff(self.opt_coords2cart_coords, coords)
        return dX_on_dC

    def coords2molstr(self, coords):
        """Generates a string containing a molecule specification in Gaussian 
        Z-matrix or Gaussian XYZ format (without method keywords, i.e. only 
        the molecule specification) based on coords."""

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
        """Generates the xyz coordinates (both as a Gaussian z-matrix format 
        string and as an array of floats) based on a set of input coordinates 
        given in the optimisation coordinate system."""

        if OLD_PYBEL_CODE:
            str = self.coords2molstr(coords)
    
            if self.format == "zmt":
                (str, coords) = self.zmt2xyz(str)
        else:
            coords = self.zmatrix.int2cart(coords)
            str = self.zmatrix.xyz_str()

        return (str, coords)

    def zmt2xyz(self, str):
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
        inputfile = __name__ + "-" + str(self.job_counter) + self.qcinput_ext
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

    def logfile2eg_Gaussian(self, logfilename, coords):
        """Extracts the energy and gradient from logfilename. Converts the
        gradient from cartesian coordinates to the coordinate system that the
        optimisation is performed in (which might be either internals or 
        cartesians)."""

        import cclib
        file = cclib.parser.ccopen(logfilename, loglevel=logging.ERROR)
        data = file.parse()

        # energy gradients in cartesian coordinates
        if hasattr(data, "grads"):
            grads_cart = data.grads
        else:
            raise Exception("No gradients found in file " + logfilename)

        # Gaussian gives gradients in Hartrees per Bohr Radius
        grads_cart *= ANGSTROMS_TO_BOHRS

        if hasattr(data, "scfenergies"):
            energy = data.scfenergies[-1]
        elif hasattr(data, "mmenergies"):
            energy = data.mmenergies[-1]
        else:
            raise Exception("No energies found in file " + logfilename)

        lg.debug(logfilename + ": Raw gradients in cartesian coordinates: " + str(grads_cart))

        transform_matrix = self.coordsys_trans_matrix(coords)
        if numpy.linalg.norm(transform_matrix) > 10:
            lg.error(line() + "Enormous coordinate system derivatives" + line())
            lg.debug(logfilename + ": transform matrix: " + str(transform_matrix))


        # energy gradients in optimisation coordinates
        # Gaussian returns forces, not gradients
        grads_opt = -numpy.dot(transform_matrix, grads_cart)
        if numpy.linalg.norm(grads_opt) > 10:
            lg.error(line() + "Enormous gradients" + line())
            lg.debug(logfilename + ": ZMat gradients: " + str(grads_opt))
            lg.debug(logfilename + ": transform matrix: " + str(transform_matrix))

        return (energy, grads_opt)



