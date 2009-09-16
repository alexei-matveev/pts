import re
import numpy
from common import *
import numerical
import sys
import getopt
import thread


class Atom():
    def __init__(self, astr=None, ix=None):

        # define patterns to match various atoms
        # first atom
        a1 = re.compile(r"\s*(\w\w?)\s*")

        # second atom
        a2 = re.compile(r"\s*(\w\w?)\s+(\d+)\s+(\S+)\s*")

        # 3rd atom
        a3 = re.compile(r"\s*(\w\w?)\s+(\d+)\s+(\S+)\s+(\d+)\s+(\S+)\s*")

        # remaining atoms
        aRest = re.compile(r"\s*(\w\w?)\s+(\d+)\s+(\S+)\s+(\d+)\s+(\S+)\s+(\d+)\s+(\S+)\s*")

        self.a = self.b = self.c = self.dst = self.ang = self.dih = self.name = None
        patterns = [aRest, a3, a2, a1]
        for pat in patterns:
            match = pat.match(astr)
            if match != None:
                groups = pat.search(astr).groups()
                groups_count = len(groups)
                if groups_count >= 1:
                    self.name = groups[0]
                    self.name = self.name[0].upper() + self.name[1:]
                if groups_count >= 3:
                    self.a = int(groups[1])
                    self.dst = self.__process(groups[2])
                if groups_count >= 5:
                    self.b = int(groups[3])
                    self.ang = self.__process(groups[4])
                if groups_count == 7:
                    self.c = int(groups[5])
                    self.dih = self.__process(groups[6])

                break
        if self.name == None:
            raise Exception("None of the patterns for an atom spec matched: " + astr)

        self.ix = ix
    
    def not_dummy(self):
        """Returns true if and only if the atom is not a dummy atom."""
        return self.name.lower() != "x" and self.name.lower() != "xx"

    def dih_var(self):
        """Returns the name of the dihedral variable."""

        if isinstance(self.dih, str):
            if self.dih[0] == "-":
                return self.dih[1:]
            else:
                return self.dih
        else:
            return None

    def all_vars(self):
        """Return a list of all variables associated with this atom."""
        potentials_list = [self.dst, self.ang, self.dih]
        vars_list = []
        for var in potentials_list:
            if isinstance(var, str):
                if var[0] == "-":
                    vars_list.append(var[1:])
                else:
                    vars_list.append(var)
        return vars_list

    def __process(self, varstr):
        """Converts str to float if it matches, otherwise returns str, since it must
        therefore be a variable name."""

        if re.match(r"[+-]?\d+(\.\d+)?", varstr) != None:
            return float(varstr)
        return varstr

    def __str__(self):
        mystr = self.name
        if self.a != None:
            mystr += " " + str(self.a) + " " + str(self.dst)
            if self.b != None:
                mystr += " " + str(self.b) + " " + str(self.ang)
                if self.c != None:
                    mystr += " " + str(self.c) + " " + str(self.dih)
        return mystr

def normalise(x):
    x = x / numpy.linalg.norm(x)
    return x

class ZMatrix():
    @staticmethod
    def matches(mol_text):
        """Returns True if and only if mol_text matches a z-matrix. There must be at least one
        variable in the variable list."""
        zmt = re.compile(r"""\s*(\w\w?\s*
                             \s*(\w\w?\s+\d+\s+\S+\s*
                             \s*(\w\w?\s+\d+\s+\S+\s+\d+\s+\S+\s*
                             ([ ]*\w\w?\s+\d+\s+\S+\s+\d+\s+\S+\s+\d+\s+\S+[ ]*\n)*)?)?)[ \t]*\n
                             (([ ]*\w+\s+[+-]?\d+\.\d*[ \t\r\f\v]*\n)+)\s*$""", re.X)
        return (zmt.match(mol_text) != None)

    def __init__(self, mol_text):

        self.atoms = []
        self.vars = dict()
        self.atoms_dict = dict()

        if self.matches(mol_text):
            parts = re.search(r"(?P<zmt>.+?)\n\s*\n(?P<vars>.+)", mol_text, re.S)

            # z-matrix text, specifies connection of atoms
            zmt_spec = parts.group("zmt")
            variables_text = parts.group("vars")
            self.var_names = re.findall(r"(\w+).*?\n", variables_text)
            self.no_vars = len(self.var_names)
            self.coords = re.findall(r"\w+\s+([+-]?\d+\.\d*)\n", variables_text)
            self.coords = numpy.array([float(c) for c in self.coords])
        else:
            raise Exception("Z-matrix not found in string:\n" + mol_text)
        
        # Create data structure of atoms. There is both an ordered list and an 
        # unordered dictionary with atom index as the key.
        lines = zmt_spec.split("\n")
        ixs = range(1, len(lines) + 1)
        for line, ix in zip(lines, ixs):
            a = Atom(line, ix)
            self.atoms.append(a)
            self.atoms_dict[ix] = a
        
        # Dictionary of dihedral angles
        self.dih_vars = dict()
        for atom in self.atoms:
            if atom.dih_var() != None:
                self.dih_vars[atom.dih_var()] = 1

        # flags = True/False indicating whether variables are dihedrals or not
        self.dih_flags = numpy.array([(var in self.dih_vars) for var in self.var_names])

        # TODO: check that z-matrix is ok, e.g. A, B 1 ab, etc...

        # Create dictionary of variable values (unordered) and an 
        # ordered list of variable names.

        print "Molecule"
        for i in range(len(self.var_names)):
            key = self.var_names[i]
            val = float(self.coords[i])

            # move all dihedrals into domain [0,360)
            """if key in self.dih_vars:
                if val >= 0.0:
                    print "not changing", key, "from", val
                else: #if val < 0.0:
                    print "changing", key, "from", val,
                    val += 360.0
                    print "to",val"""

            self.vars[key] = val

        # check that z-matrix is fully specified
        self.zmt_ordered_vars = []
        for atom in self.atoms:
            self.zmt_ordered_vars += atom.all_vars()
        for var in self.zmt_ordered_vars:
            if not var in self.vars:
                raise Exception("Variable '" + var + "' not given in z-matrix")

        self.state_mod_lock = thread.allocate_lock()


    def get_var(self, var):
        """If var is numeric, return it, otherwise look it's value up 
        in the dictionary of variable values."""

        if type(var) == str:
            if var[0] == "-":
                return -1 * self.vars[var[1:]]
            else:
                return self.vars[var]
        else:
            return var

    def zmt_str(self):
        """Returns a z-matrix format molecular representation in a string."""
        mystr = ""
        for atom in self.atoms:
            mystr += str(atom) + "\n"
        mystr += "\n"
        for var in self.var_names:
            mystr += var + "\t" + str(self.vars[var]) + "\n"
        return mystr

    def xyz_str(self):
        """Returns an xyz format molecular representation in a string."""
        if not "vector" in self.atoms[0].__dict__:
            self.gen_cartesian()

        mystr = ""
        for atom in self.atoms:
            if atom.name[0].lower() != "x":
                mystr += atom.name + " " + self.__pretty_vec(atom.vector) + "\n"
        return mystr

    def __pretty_vec(self, x):
        """Returns a pretty string rep of a (3D) vector."""
        return "%f\t%f\t%f" % (x[0], x[1], x[2])

    def set_internals(self, internals):
        """Update stored list of variable values."""
        for i, var in zip( internals, self.var_names ):
            self.vars[var] = i

    def get_internals(self):
        """Return the current set of internals."""

        curr = []
        for var in self.var_names:
            curr.append(self.vars[var])

        return numpy.array(curr)

    def int2cart(self, x):
        """Based on a vector x of new internal coordinates, returns a 
        vector of cartesian coordinates. The internal dictionary of coordinates 
        is updated."""

        self.state_mod_lock.acquire()

        self.set_internals(x)
        y = self.gen_cartesian()

        self.state_mod_lock.release()

        return y.flatten()

    def dcart_on_dint(self, x):
        """Returns the matrix of derivatives dCi/dIj where Ci is the ith cartesian coordinate
        and Ij is the jth internal coordinate."""

        nd = numerical.NumDiff()
        mat = nd.numdiff(self.int2cart, x)
        #print "mat", mat
        return mat

    def gen_cartesian(self):
        """Generates cartesian coordinates from z-matrix and the current set of 
        internal coordinates. Based on code in OpenBabel."""
        
        r = numpy.float64(0)
        sum = numpy.float64(0)

        xyz_coords = []
        for atom in self.atoms:
            if atom.a == None:
                atom.vector = numpy.zeros(3)
                if atom.not_dummy():
                    xyz_coords.append(atom.vector)
                continue
            else:
                avec = self.atoms_dict[atom.a].vector
                dst = self.get_var(atom.dst)

            if atom.b == None:
                atom.vector = numpy.array((dst, 0.0, 0.0))
                if atom.not_dummy():
                    xyz_coords.append(atom.vector)
                continue
            else:
                bvec = self.atoms_dict[atom.b].vector
                ang = self.get_var(atom.ang) * DEG_TO_RAD

            if atom.c == None:
                cvec = VY
                dih = 90. * DEG_TO_RAD
            else:
                cvec = self.atoms_dict[atom.c].vector
                dih = self.get_var(atom.dih) * DEG_TO_RAD

            v1 = avec - bvec
            v2 = avec - cvec

            n = numpy.cross(v1,v2)
            nn = numpy.cross(v1,n)
            n = normalise(n)
            nn = normalise(nn)

            n *= -numpy.sin(dih)
            nn *= numpy.cos(dih)
            v3 = n + nn
            v3 = normalise(v3)
            v3 *= dst * numpy.sin(ang)
            v1 = normalise(v1)
            v1 *= dst * numpy.cos(ang)
            v2 = avec + v3 - v1

            atom.vector = v2

            if atom.not_dummy():
                xyz_coords.append(atom.vector)
        
        xyz_coords = numpy.array(xyz_coords)
        return xyz_coords

class Usage(Exception):
    def __init__(self, msg):
        self.msg = msg
    def __str__(self, msg):
        return self.msg


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    try:
        try:
            opts, args = getopt.getopt(argv[1:], "h", ["help"])
        except getopt.error, msg:
             raise Usage(msg)

        print "argv =", argv
        if len(argv) != 1:
            raise Usage("Exactly 1 input file must be specified.")
        inputfile = argv[0]

    except Usage, err:
        print >>sys.stderr, err.msg
        print >>sys.stderr, "for help use --help"
        return 2
    
    f = open(inputfile, "r")
    zmt_txt = f.read()
    f.close()
    zmt = ZMatrix(zmt_txt)
    for var in zmt.zmt_ordered_vars:
        print zmt.get_var(var)

    

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt, e:
        print e
    sys.exit()


