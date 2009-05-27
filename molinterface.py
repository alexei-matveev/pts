import numpy
import scipy
import re
import copy

DEFAULT_GAUSSIAN03_HEADER = "# HF/3-21G\ncomment\n0 1\n"

DEFAULT_GAUSSIAN03_FOOTER = ""


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
                             (\w\w?\s+\w+\s*
                             (\w\w?\s+\w+\s+\w+\s*
                             (\w\w?\s+\w+\s+\w+\s+\S+\s*)*)?)?
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
        self.coords = [float(c) for c in self.coords]

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




class MolInterface:
    """Converts between molecule representations (i.e. internal xyz/zmat 
    representation), optimisation coordinates, Quantum Chemistry logfile format
    and QC program input format.
    """

    def __init__(self, mol1, mol2, params = dict()):
        m1 = MolRep(mol1)
        m2 = MolRep(mol2)

        if m1.atoms != m2.atoms:
            raise Exception("input molecules have different atoms")
        elif m1.format != m2.format:
            raise Exception("input molecules in different formats")
        elif m1.format == "zmt" and (m1.var_names != m2.var_names):
            raise Exception("input molecules have different variable names in z-matrix")

        
        self.format = m1.format
        self.atoms = m1.atoms
        self.natoms = len(m1.atoms)
        if self.format == "zmt":
            self.var_names = m1.var_names
            self.zmt_spec = m1.zmt_spec
            self.nvariables = len(self.var_names)

        self.mol1_coords = m1.coords
        self.mol2_coords = m2.coords

        if "qcinput_head" in params:
            self.qcinput_head = params["qcinput_head"]
        else:
            self.qcinput_head = DEFAULT_GAUSSIAN03_HEADER

        if "qcinput_foot" in params:
            self.qcinput_foot = params["qcinput_foot"]
        else:
            self.qcinput_foot = DEFAULT_GAUSSIAN03_FOOTER


    def __str__(self):
        mystr = "format = " + self.format
        mystr += "\natoms = " + str(self.atoms)
        mystr += "\nvar_names = " + str(self.var_names)
        mystr += "\nmol1_coords = " + str(self.mol1_coords)
        mystr += "\nmol2_coords = " + str(self.mol2_coords)

        mystr += "\nzmt_spec:\n" + self.zmt_spec
        return mystr

    def coords2qcinput(self, coords, path = None):
        mol_text = self.coords2moltext(coords)

        str = self.qcinput_head + mol_text + self.qcinput_foot
        if path != None:
            f = open(path, 'w')
            f.write(str)
            f.close()

        return str

    def coords2moltext(self, coords):
        str = ""
        if self.format == "xyz":
            for i in range(self.natoms):
                str += "%s\t%s\t%s\t%s" % (self.atom_symbols[i], coords[3*i], coords[3*i+1], coords[3*i+2])
        elif self.format == "zmt":
            str += self.zmt_spec + "\n"
            for i in range(self.nvariables):
                str += "%s=%s\n" % (self.var_names[i], coords[i])


            print "str = ", str
            (str, coords) = self.zmt2xyz(str)
        else:
            raise Exception("Unrecognised self.mol_rep_type")

        return str

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

        return (str, xyz_coords)

    def logfile2eg(self, filename):
        pass

    def numdiff(self, f, X):
        """For function f, computes f'(X) numerically based on a finite difference approach."""

        N = len(X)
        df_on_dX = []
        num_diff_err = 1e-3

        def update_estim(dx, ix):
            X1 = copy.deepcopy(X)
            X2 = copy.deepcopy(X)
            X1[ix] += dx
            X2[ix] -= dx
            return (f(X1) - f(X2))/ (2*dx)

        for i in range(N):
            dx = 1e-5
            df_on_dx = update_estim(dx, i)
            while True:
                prev_df_on_dx = df_on_dx
                dx /= 2.0
                df_on_dx = update_estim(dx, i)
                err = numpy.linalg.norm(df_on_dx - prev_df_on_dx) / numpy.linalg.norm(prev_df_on_dx)
                if numpy.isnan(err):
                    raise Exception("NaN encountered in numerical differentiation")
                if err < num_diff_err:
                    break

            df_on_dX.append(df_on_dx)

        return numpy.array(df_on_dX)

