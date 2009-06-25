import numpy
import scipy
import re
import copy
import cclib
import thread
import logging

from common import *

lg = logging.getLogger(PROGNAME)

numpy.set_printoptions(linewidth=180)

DEFAULT_GAUSSIAN03_HEADER = "# HF/3-21G force\n\ncomment\n\n0 1\n"
DEFAULT_GAUSSIAN03_FOOTER = ""
DEFAULT_GAUSSIAN03_QCINPUT_EXT = ".com"
DEFAULT_GAUSSIAN03_QCOUTPUT_EXT = ".log"
DEFAULT_GAUSSIAN03_PROGNAME = "g03"

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
        zmt = re.compile(r"""(\w\w?\s*
                             (\w\w?\s+\d+\s+\w+\s*
                             (\w\w?\s+\d+\s+\w+\s+\d+\s+\w+\s*
                             (\w\w?\s+\d+\s+\w+\s+\d+\s+\w+\s+\d+\s+\S+\s*)*)?)?
                             (\w+\s+[+-]?\d+\.\d*\s*)+)$""", re.X)

        xyz = re.compile(r"""(\w\w?(\s+[+-]?\d+\.\d*){3}\s*)+""")

        if zmt.match(mol_text) != None:
            self.format = "zmt"
            parts = re.search(r"(.+?\n)\s*\n(.+)", mol_text, re.S)

            self.zmt_spec = parts.group(1) # z-matrix text, specifies connection of atoms
            variables_text = parts.group(2)
            self.var_names = re.findall(r"(\w+).*?\n", variables_text)
            self.coords = re.findall(r"\w+\s+([+-]?\d+\.\d*)\n", variables_text)
            
        elif xyz.match(mol_text) != None:
            self.format = "xyz"
            self.coords = re.findall(r"([+-]?\d+\.\d*)", mol_text)
            self.var_names = "no variable names"
        else:
            raise Exception("can't understand input file")

        self.atoms = re.findall(r"(\w\w?).*?\n", self.zmt_spec)
        self.coords = numpy.array([float(c) for c in self.coords])

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
            self.zmt_spec = molreps[0].zmt_spec
            self.nvariables = len(self.var_names)

        self.reagent_coords = [m.coords for m in molreps]

        if "qcinput_head" in params:
            self.qcinput_head = params["qcinput_head"]
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
        if "qc_program" in params:
            if params["qc_program"] == "analytical_GaussianPES":
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

        self.ANGSTROMS_TO_BOHRS = 1.8897

    def __str__(self):
        mystr = "format = " + self.format
        mystr += "\natoms = " + str(self.atoms)
        mystr += "\nvar_names = " + str(self.var_names)
        mystr += "\nreactant coords = " + str(self.reagent_coords[0])
        mystr += "\nproduct coords = " + str(self.reagent_coords[1])
        mystr += "\nzmt_spec:\n" + self.zmt_spec
        return mystr

    def geom_checker(self, coords):
        """Not Yet Implemented.
        
        Checks that coords will generate a chemically reasonable 
        molecule, i.e. no overlap."""
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
        return 
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

        dX_on_dC = self.numdiff(self.opt_coords2cart_coords, coords)
        return dX_on_dC

    def coords2xyz(self, coords):
        """Generates the xyz coordinates (both as a Gaussian z-matrix format 
        string and as an array of floats) based on a set of input coordinates 
        given in the optimisation coordinate system."""

        str = ""
        if self.format == "xyz":
            for i in range(self.natoms):
                str += "%s\t%s\t%s\t%s" % (self.atom_symbols[i], coords[3*i], coords[3*i+1], coords[3*i+2])
        elif self.format == "zmt":
            str += self.zmt_spec + "\n"
            for i in range(self.nvariables):
                str += "%s=%s\n" % (self.var_names[i], coords[i])

            (str, coords) = self.zmt2xyz(str)
        else:
            raise Exception("Unrecognised self.mol_rep_type")

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
#        print str

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
            lg.error("Exception thrown when calling self.run_qc")
            return

        # parse output file
        e, g = self.logfile2eg(outputfile, coords)
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
        grads_cart = data.grads

        # Gaussian gives gradients in Hartrees per Bohr Radius
        grads_cart *= self.ANGSTROMS_TO_BOHRS
        energy = data.scfenergies[-1]
        lg.debug("Raw gradients in cartesian coordinates: " + str(grads_cart))

        transform_matrix = self.coordsys_trans_matrix(coords)

        # energy gradients in optimisation coordinates
        # Gaussian returns forces, not gradients
        grads_opt = -numpy.dot(transform_matrix, grads_cart)
        
        return (energy, grads_opt)

    def numdiff(self, f, X):
        """For function f, computes f'(X) numerically based on a finite difference approach."""

        N = len(X)
        df_on_dX = []
        num_diff_err = 1e-3 # as fraction of derivative being measured

        def update_estim(dx, ix):
            X1 = copy.deepcopy(X)
            X2 = copy.deepcopy(X)
            X1[ix] += dx
            X2[ix] -= dx
            f1, f2 = f(X1), f(X2)
            return (f1 - f2)/ (2*dx)

        for i in range(N):
            dx = 1e-1
            df_on_dx = update_estim(dx, i)
            while True:
                #break # at the moment, no error control, just single quick+dirty finite diff measurement
                lg.debug("df_on_dx = " + str(df_on_dx))
                prev_df_on_dx = df_on_dx
                dx /= 2.0
                df_on_dx = update_estim(dx, i)
                norm = numpy.linalg.norm(prev_df_on_dx)

                #  Hmm, problems with this when used for z-mat conversion
                err = self.calc_err(df_on_dx, prev_df_on_dx)
                if numpy.isnan(err):
                    raise Exception("NaN encountered in numerical differentiation")
                if err < num_diff_err:
                    break

            df_on_dX.append(df_on_dx)

        return numpy.array(df_on_dX)

    def calc_err(self, estim1, estim2):
        max_err = -1e200
        diff = estim1 - estim2
        for i in range(len(estim1)):
            err = diff[i] / estim1[i]
            if not numpy.isnan(err):
                if max_err < abs(err):
                    max_err = abs(err)
#                    print "updating err ", err

        return max_err

##################### Start Test Functions ######################
def test_MolInterface2():
    f = open("CH4.zmt", "r")
    m1 = f.read()
    m2 = m1
    f.close()

    mi = MolInterface(m1, m2)
    print "Testing: MolecularInterface"
    print mi

    X = numpy.array([1.0900000000000001, 109.5, 120.0])
    print mi.opt_coords2cart_coords(X)

    print "Testing: coordsys_trans_matrix()"
    print mi.coordsys_trans_matrix(X)

    print "Testing: run_job()"
    logfilename = mi.run_job(X)
    print "file", logfilename, "created"
    print mi.logfile2eg(logfilename, X)

    import cclib
    file = cclib.parser.ccopen("CH4.log", loglevel=logging.ERROR)
    data = file.parse()
    print "SCF Energy and Gradients from direct calc on z-matrix input:"
    print data.scfenergies[-1]
    print data.grads
    print data.gradvars

def test_MolInterface3():
    f = open("H2O.zmt", "r")
    m1 = f.read()
    m2 = m1
    f.close()

    mi = MolInterface(m1, m2)
    print "Testing: MolecularInterface"
    print mi

    X = numpy.array([1.5, 100.1])
    print mi.opt_coords2cart_coords(X)

    print "Testing: coordsys_trans_matrix()"
    print mi.coordsys_trans_matrix(X)

    print "Testing: run_job()"
    logfilename = mi.run_job(X)
    print "file", logfilename, "created"
    print mi.logfile2eg(logfilename, X)

    import cclib
    file = cclib.parser.ccopen("H2O.log")
    data = file.parse()
    print "SCF Energy and Gradients from direct calc on z-matrix input:"
    print data.scfenergies[-1]
    print data.grads
    print data.gradvars

def test_MolInterface():
    f = open("NH3.zmt", "r")
    m1 = f.read()
    m2 = m1
    f.close()

    mi = MolInterface(m1, m2)
    print "Testing: MolecularInterface"
    print mi

    print "Testing: coords2moltext()"
    print mi.coords2moltext([0.97999999999999998, 1.089, 109.471, 120.0])

    print "Testing: coords2qcinput()"
    print mi.coords2qcinput([0.97999999999999998, 1.089, 109.471, 120.0])
    print mi.coords2qcinput([0.97999999999999998, 1.089, 129.471, 0.0])

    str_zmt = """H
    F  1  r2
    Variables:
    r2= 0.9000
    """
    
    print "Testing: zmt2xyz()"
    print mi.zmt2xyz(str_zmt)

    X = numpy.arange(4.0)*3.14159/4
    print mi.numdiff(numpy.sin, X)

    print "Testing: coords2xyz()"
    print mi.coords2xyz([0.97999999999999998, 1.089, 109.471, 120.0])

    print "Testing: opt_coords2cart_coords()"

    X = numpy.array([0.97999999999999998, 1.089, 109.471, 120.0])
    print mi.opt_coords2cart_coords(X)

    print "Testing: coordsys_trans_matrix()"
    print mi.coordsys_trans_matrix(X)

    print "Testing: run_job()"
    logfilename = mi.run_job(X)
    print "file", logfilename, "created"
    print mi.logfile2eg(logfilename, X)

    import cclib
    file = cclib.parser.ccopen("NH3.log")
    data = file.parse()
    print "SCF Energy and Gradients from direct calc on z-matrix input:"
    print data.scfenergies[-1]
    print data.grads
    print data.gradvars

############## END TEST FUNCTIONS ###############


