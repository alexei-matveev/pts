import re
import numpy
from common import *


class Atom():
    def __init__(self, astr=None, ix=None):

        # first atom
        a1 = re.compile(r"\s*(\w\w?)\s*")
        # second atom
        a2 = re.compile(r"\s*(\w\w?)\s+(\d+)\s+(\S+)\s*")
        a3 = re.compile(r"\s*(\w\w?)\s+(\d+)\s+(\S+)\s+(\d+)\s+(\S+)\s*")
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

def chomp(str):
    pass

class ZMatrix():
    def __init__(self, mol_text):

        # TODO: clean the following up
        zmt1 = re.compile(r"""[ \t]*(\w\w?[ \t\r\f\v]*\n
                             [ \t]*(\w\w?\s+\d+\s+\w+[ \t\r\f\v]*\n
                             [ \t]*(\w\w?\s+\d+\s+\w+\s+\d+\s+\w+[ \t\r\f\v]*\n
                             [ \t]*(\w\w?\s+\d+\s+\w+\s+\d+\s+\w+\s+\d+\s+\S+[ \t]*\n)*)?)?)
                             [ \t]*\n
                             ((\w+\s+[+-]?\d+\.\d*[ \t\r\f\v]*\n)+)\s*$""", re.X)

        zmt2 = re.compile(r"""(\w\w?\s*
                             (\w\w?\s+\d+\s+\w+\s*
                             (\w\w?\s+\d+\s+\w+\s+\d+\s+\w+\s*
                             (\w\w?\s+\d+\s+\w+\s+\d+\s+\w+\s+\d+\s+\S+\s*)*)?)?
                             (\w+\s+[+-]?\d+\.\d*\s*)+)$""", re.X)

        zmt = re.compile(r"""\s*(\w\w?\s*
                             \s*(\w\w?\s+\d+\s+\S+\s*
                             \s*(\w\w?\s+\d+\s+\S+\s+\d+\s+\S+\s*
                             ([ ]*\w\w?\s+\d+\s+\S+\s+\d+\s+\S+\s+\d+\s+\S+[ ]*\n)*)?)?)\n
                             (([ ]*\w+\s+[+-]?\d+\.\d*[ \t\r\f\v]*\n)+)\s*$""", re.X)


        self.atoms = []
        self.vars = dict()
        self.atoms_dict = dict()

        if zmt.match(mol_text) != None:
            parts = re.search(r"(?P<zmt>.+?)\n\s*\n(?P<vars>.+)", mol_text, re.S)

            zmt_spec = parts.group("zmt") # z-matrix text, specifies connection of atoms
            variables_text = parts.group("vars")
            var_names = re.findall(r"(\w+).*?\n", variables_text)
            coords = re.findall(r"\w+\s+([+-]?\d+\.\d*)\n", variables_text)
        else:
            raise Exception("Z-matrix not found in string:\n" + mol_text)
        
        # Create data structure of atoms. There is both an ordered list and an 
        # unordered dictionary.
        lines = zmt_spec.split("\n")
        ixs = range(len(lines) + 1)[1:]
        for line, ix in zip(lines, ixs):
            a = Atom(line, ix)
            self.atoms.append(a)
            self.atoms_dict[ix] = a
        
        # Create dictionary of variable values (unordered) and an 
        # ordered list of variable names.
        self.var_names = []
        for i in range(len(var_names)):
            key = var_names[i]
            val = coords[i]
            self.vars[key] = float(val)
            self.var_names.append(key)

        # check that z-matrix is fully specified
        required_vars = []
        for atom in self.atoms:
            required_vars += atom.all_vars()
        for var in required_vars:
            if not var in self.vars:
                raise Exception("Variable '" + var + "' not given in z-matrix")

    def __get_var(self, var):
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
        """Returns an zmatrix format molecular representation in a string."""
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
            mystr += atom.name + " " + self.__pretty_vec(atom.vector) + "\n"
        return mystr

    def __pretty_vec(self, x):
        """Returns a pretty string rep of a (3D) vector."""
        return "%f\t%f\t%f" % (x[0], x[1], x[2])

    def set_internals(self, internals):
        ilist = internals.tolist()
        for i, var in zip( ilist, self.var_names ):
            self.vars[var] = i

    def int2cart(self, x):
        """Based on a vector x of new internal coordinates, returns a 
        vector of cartesian coordinates. The internal dictionary of coordinates 
        is updated."""
        self.set_internals(x)
        y = self.gen_cartesian()
        return y.flatten()

    def gen_cartesian(self):
        """Generates cartesian coordinates from z-matrix and the current set of internal coordinates."""
        
        r = numpy.float64(0)
        sum = numpy.float64(0)

        xyz_coords = []
        for atom in self.atoms:
            if atom.a == None:
                atom.vector = numpy.zeros(3)
                xyz_coords.append(atom.vector)
                continue
            else:
                avec = self.atoms_dict[atom.a].vector
                dst = self.__get_var(atom.dst)

            if atom.b == None:
                atom.vector = numpy.array((dst, 0.0, 0.0))
                xyz_coords.append(atom.vector)
                continue
            else:
                bvec = self.atoms_dict[atom.b].vector
                ang = self.__get_var(atom.ang) * DEG_TO_RAD

            if atom.c == None:
                cvec = VY
                dih = 90. * DEG_TO_RAD
            else:
                cvec = self.atoms_dict[atom.c].vector
                dih = self.__get_var(atom.dih) * DEG_TO_RAD

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
            xyz_coords.append(atom.vector)
        
        xyz_coords = numpy.array(xyz_coords)
        return xyz_coords


