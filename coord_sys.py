from __future__ import with_statement
import re
import ase
from ase import Atoms
import numpy
import threading
import numerical
import os


import common

class Anchor(object):
    def __init__(self, initial):
        if self.dims != len(initial):
            raise ComplexCoordSysException("initial value did not match required number of dimensions: " + str(initial))
    def reposition(self, *args):
        assert False, "Abstract function"

    def set(self, t):
        assert len(t) == self.dims
        self._coords = t

    def get(self):
        return self._coords

    coords = property(get, set)

class Dummy(Anchor):
    dims = 0
    def __init__(self, initial=numpy.array([])):
        Anchor.__init__(self, initial)
        self._coords = numpy.array([])

    def reposition(self, x):
        return x

class RotAndTrans(Anchor):
    dims = 7

    def __init__(self, initial, parent = None):
        Anchor.__init__(self, initial)
        self._parent = parent

        quaternion = initial[:4]
        n = numpy.linalg.norm(quaternion)
        if abs(n - 1.0) > 0.001:
            msg = "Initial value for quaternion was " + str(quaternion) + ", norm was " + str(n)
            raise ComplexCoordSysException()

        self._coords = initial

    @property
    def qnorm(self):
        return numpy.linalg.norm(self.coords[0:4])

    def quaternion2rot_mat(self, quaternion):
        """See http://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation"""
        a,b,c,d = quaternion

        m = numpy.array([[ a*a + b*b - c*c - d*d , 2*b*c + 2*a*d,         2*b*d - 2*a*c  ],
                         [ 2*b*c - 2*a*d         , a*a - b*b + c*c - d*d, 2*c*d + 2*a*b  ],
                         [ 2*b*d + 2*a*c         , 2*c*d - 2*a*b        , a*a - b*b - c*c + d*d  ]])

        return m

    def reposition(self, carts):
        """Based on a quaternion and a translation, transforms a set of 
        cartesion positions x."""

        quaternion = self.coords[0:4]
        trans_vec  = self.coords[4:]

        # TODO: need to normalise?
        rot_mat = self.quaternion2rot_mat(quaternion)

        transform = lambda vec3d: numpy.dot(rot_mat, vec3d) + trans_vec
        res = numpy.array(map(transform, carts))

        if self._parent != None:
            res += self._parent.get_centroid()

        return res

class CoordSys(object):
    """Abstract coordinate system. Sits on top of an ASE Atoms object."""

    def __init__(self, atom_symbols, atom_xyzs, abstract_coords, anchor=Dummy(), cell=None, pbc=None):

        self._dims = len(abstract_coords)
        self._atoms = Atoms(symbols=atom_symbols, positions=atom_xyzs)
        if cell:
            self._atoms.set_cell(cell)
        if pbc:
            self._atoms.set_pbc(pbc)

        self._state_lock = threading.RLock()

        self._anchor=anchor

        # TODO: recently added, watch that this is ok
        self._coords = abstract_coords.copy()

        self._var_mask = None
        self._exclusions_count = 0

    def get_dims(self):
        return self._dims + self._anchor.dims - self._exclusions_count

    dims = property(get_dims)
    def __getstate__(self):
        odict = self.__dict__.copy()
        del odict["_atoms"]
        del odict["_state_lock"]

        # create a string representing the atoms object using the ASE io functionality
        tmp = os.tmpnam()
        ase.write(tmp, self._atoms, format="traj")
        f = open(tmp, "rb")
        odict["pickled_atoms"] = f.read()
        f.close()

        return odict

    def __setstate__(self, idict):
        pickled_atoms = idict.pop("pickled_atoms")

        self.__dict__.update(idict)

        tmp = os.tmpnam()
        f = open(tmp, mode="wb")
        f.write(pickled_atoms)
        f.close()
        self._atoms = ase.read(tmp, format="traj")
        self._state_lock = threading.RLock()

    def __getattr__(self, a):
        """Properties not available in this object are brought through from the Atoms object."""
        """if a in self.__dict__:
            return self.__dict__[a]
        else:"""
        return self._atoms.__getattribute__(a)

    def get_positions(self):
        """ASE style interface function"""
        return common.make_like_atoms(self._coords)

    def get_centroid(self):
        c = numpy.zeros(3)
        carts = self.get_cartesians()
        for v in carts:
            c += v
        c = c / len(carts)
        return c

    def set_calculator(self, calc):
        return self._atoms.set_calculator(calc)

    def set_positions(self, x):
        assert x.shape[1] == 3

        self.set_internals(x.flatten()[0:self.dims])
        assert len(self._coords) >= self._dims

    def set_var_mask(self, mask):
        """Sets a variable exclusion mask. Only include variables that are True."""

        assert type(mask) == numpy.ndarray
        assert mask.dtype == bool
        assert len(mask) == len(self._coords)

        self._exclusions_count = len(mask) - sum(mask)
        self._var_mask = mask.copy()


    def _demask(self, x):
        """Builds a vector to replace the current internal state vector, that 
        will only update those elements which specified by the mask."""
        if not self._var_mask:
            return x

        newx = self._coords.copy()
        j = 0
        for m, xi in zip (self._var_mask, x):
            if m:
                newx[j] = xi
                j += 1

        assert j == len(x)
        return newx
        
    def _mask(self, x):
        """Builds a vector of internal variables by only including those that 
        are specified."""
        if not self._var_mask:
            return x

        j = 0
        output = numpy.zeros(self._dims - self._exclusions_count)
        for m, xi in zip (self._var_mask, x):
            if m:
                output[j] = xi
                j += 1

        assert j == self._dims - self._exclusions_count

        return output

    def set_internals(self, x):

        """print len(x)
        print self._exclusions_count
        print self.dims"""
            
        assert len(x) == self.dims
        x = self._demask(x)

        self._coords = x[:self._dims]

        anchor_coords = x[self._dims:]
        self._anchor.coords = anchor_coords

    def needs_anchor(self):
        """Returns True if object cannot be placed in cartesian space without an anchor. """
        assert False, "Abstract function"

    def get_cartesians(self):
        """Returns Cartesians as a flat array."""
        assert False, "Abstract function"

    def get_internals(self):
        raw = numpy.hstack([self._coords.copy(), self._anchor.coords])
        masked = self._mask(raw)
        return masked

    def apply_constraints(self, vec):
        return vec

    def get_forces(self, flat=False):
        cart_pos = self.get_cartesians()
        self._atoms.set_positions(cart_pos)

        forces_cartesian = self._atoms.get_forces().flatten()
        transform_matrix, errors = self.get_transform_matrix(self._coords)
        #TODO: test magnitude of errors
        forces_coord_sys = numpy.dot(transform_matrix, forces_cartesian)
        
        forces_coord_sys = self.apply_constraints(forces_coord_sys)

        if flat:
            return forces_coord_sys
        else:
            return common.make_like_atoms(forces_coord_sys)
    
    def get_potential_energy(self):

        cart_pos = self.get_cartesians()
        self._atoms.set_positions(cart_pos)

        return self._atoms.get_potential_energy()
       

    def copy(self, new_coords=None):
        assert False, "Abstract function"

    @property
    def atoms(self):
        self._atoms.positions = self.get_cartesians()
        return self._atoms

    #def set_atoms(self, atoms):
    #    self._atoms = atoms

    def xyz_str(self):
        """Returns an xyz format molecular representation in a string."""

        cart_coords = self.get_cartesians()
        self._atoms.set_positions(cart_coords)
        list = ['%-2s %22.15f %22.15f %22.15f' % (s, x, y, z) for s, (x, y, z) in zip(self._atoms.get_chemical_symbols(), self._atoms.get_positions())]
        geom_str = '\n'.join(list) + '\n\n'

        return geom_str

    def native_str(self):
        pass

    def get_transform_matrix(self, x):
        """Returns the matrix of derivatives dCi/dIj where Ci is the ith cartesian coordinate
        and Ij is the jth internal coordinate, and the error."""

        nd = numerical.NumDiff()
        mat = nd.numdiff(self.int2cart, x)
        return mat

    def int2cart(self, x):
        """Based on a vector x of new internal coordinates, returns a 
        vector of cartesian coordinates. The internal dictionary of coordinates 
        is updated."""

        with self._state_lock:
            old_x = self._coords.copy()

            self.set_internals(x)
            y = self.get_cartesians()

            self.set_internals(old_x)

            return y.flatten()

    def __pretty_vec(self, x):
        """Returns a pretty string rep of a (3D) vector."""
        return "%f\t%f\t%f" % (x[0], x[1], x[2])


class ZMTAtom():
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
                    self.ang = self.__process(groups[4], isangle=True)
                if groups_count == 7:
                    self.c = int(groups[5])
                    self.dih = self.__process(groups[6], isangle=True)

                break
        if self.name == None:
            raise Exception("None of the patterns for an atom spec matched: " + astr)

        self.ix = ix
    
    def not_dummy(self):
        """Returns true if and only if the atom is not a dummy atom."""
        return self.name.lower() != "x" and self.name.lower() != "xx"

    def dih_var(self):
        """Returns the name of the dihedral variable."""

        if isinstance(self.dih, basestring):
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

    def __process(self, varstr, isangle=False):
        """Converts str to float if it matches, otherwise returns str, since it must
        therefore be a variable name."""

        if re.match(r"[+-]?\d+(\.\d+)?", varstr) != None:
            if isangle:
                return float(varstr) * common.DEG_TO_RAD
            else:
                return float(varstr)
        return varstr

    def __str__(self):
        mystr = self.name
        if self.a != None:
            mystr += " " + str(self.a) + " " + str(self.dst)
            if self.b != None:
                ang = self.ang
                if not isinstance(ang, basestring):
                    ang *= common.RAD_TO_DEG
                mystr += " " + str(self.b) + " " + str(ang)
                if self.c != None:
                    dih = self.dih
                    if not isinstance(dih, basestring):
                        dih *= common.RAD_TO_DEG
                    mystr += " " + str(self.c) + " " + str(dih)

        return mystr

def myenumerate(list, start=0):
    ixs = range(start, len(list) + start)
    return zip (ixs, list)

class ZMatrix(CoordSys):
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

    def __init__(self, mol_text, anchor=Dummy()):

        self.zmtatoms = []
        self.vars = dict()
        self.zmtatoms_dict = dict()
        self._anchor = anchor

        if not self.matches(mol_text):
            raise ZMatrixException("Z-matrix not found in string:\n" + mol_text)

        parts = re.search(r"(?P<zmt>.+?)\n\s*\n(?P<vars>.+)", mol_text, re.S)

        # z-matrix text, specifies connection of atoms
        zmt_spec = parts.group("zmt")

        # variables text, specifies values of variables
        variables_text = parts.group("vars")
        self.var_names = re.findall(r"(\w+).*?\n", variables_text)
        coords = re.findall(r"\w+\s+([+-]?\d+\.\d*)\n", variables_text)
        self._coords = numpy.array([float(c) for c in coords])
        #print "self._coords", self._coords
    
        # Create data structure of atoms. There is both an ordered list and an 
        # unordered dictionary with atom index as the key.
        lines = zmt_spec.split("\n")
        for ix, line in myenumerate(lines, start=1):
            a = ZMTAtom(line, ix)
            self.zmtatoms.append(a)
            self.zmtatoms_dict[ix] = a

        # Dictionaries of (a) dihedral angles and (b) angles
        self.dih_vars = dict()
        self.angles = dict()
        for atom in self.zmtatoms:
            if atom.dih_var() != None:
                self.dih_vars[atom.dih_var()] = 1
                self.angles[atom.dih_var()] = 1
            if atom.ang != None:
                self.angles[atom.ang] = 1

        #print "self.dih_vars", self.dih_vars
        # flags = True/False indicating whether variables are dihedrals or not
        # DO I NEED THIS?
        #self.dih_flags = numpy.array([(var in self.dih_vars) for var in self.var_names])

        # TODO: check that z-matrix is ok, e.g. A, B 1 ab, etc...

        # Create dictionary of variable values (unordered) and an 
        # ordered list of variable names.
        #print "Molecule"
        for i in range(len(self.var_names)):
            key = self.var_names[i]
            if key in self.angles:
                self._coords[i] *= common.DEG_TO_RAD
            val = float(self._coords[i])

            self.vars[key] = val

        # check that z-matrix is fully specified
        self.zmt_ordered_vars = []
        for atom in self.zmtatoms:
            self.zmt_ordered_vars += atom.all_vars()
        for var in self.zmt_ordered_vars:
            if not var in self.vars:
                raise ZMatrixException("Variable '" + var + "' not given in z-matrix")

        #self.state_mod_lock = thread.allocate_lock()

        #print self.zmtatoms
        symbols = [a.name for a in self.zmtatoms]
        CoordSys.__init__(self, symbols, 
            self.get_cartesians(), 
            self._coords,
            anchor)

    def wants_anchor(self):
        return True

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
        for atom in self.zmtatoms:
            mystr += str(atom) + "\n"
        mystr += "\n"
        for var in self.var_names:
            if var in self.angles:
                mystr += var + "\t" + str(self.vars[var] * common.RAD_TO_DEG) + "\n"
            else:
                mystr += var + "\t" + str(self.vars[var]) + "\n"
        return mystr

    def set_internals(self, internals):
        """Update stored list of variable values."""

        #internals = numpy.array(internals[0:self._dims])
        CoordSys.set_internals(self, internals)

        for i, var in zip( internals[0:self._dims], self.var_names ):
            self.vars[var] = i

    def get_cartesians(self):
        """Generates cartesian coordinates from z-matrix and the current set of 
        internal coordinates. Based on code in OpenBabel."""
        
        r = numpy.float64(0)
        sum = numpy.float64(0)

        xyz_coords = []
        for atom in self.zmtatoms:
            if atom.a == None:
                atom.vector = numpy.zeros(3)
                if atom.not_dummy():
                    xyz_coords.append(atom.vector)
                continue
            else:
                avec = self.zmtatoms_dict[atom.a].vector
                dst = self.get_var(atom.dst)

            if atom.b == None:
                atom.vector = numpy.array((dst, 0.0, 0.0))
                if atom.not_dummy():
                    xyz_coords.append(atom.vector)
                continue
            else:
                bvec = self.zmtatoms_dict[atom.b].vector
                ang = self.get_var(atom.ang) # * DEG_TO_RAD

            if atom.c == None:
                cvec = common.VY
                dih = 90. * common.DEG_TO_RAD
            else:
                cvec = self.zmtatoms_dict[atom.c].vector
                dih = self.get_var(atom.dih) # * DEG_TO_RAD

            v1 = avec - bvec
            v2 = avec - cvec

            n = numpy.cross(v1,v2)
            nn = numpy.cross(v1,n)
            n = common.normalise(n)
            nn = common.normalise(nn)

            n *= -numpy.sin(dih)
            nn *= numpy.cos(dih)
            v3 = n + nn
            v3 = common.normalise(v3)
            v3 *= dst * numpy.sin(ang)
            v1 = common.normalise(v1)
            v1 *= dst * numpy.cos(ang)
            v2 = avec + v3 - v1

            atom.vector = v2

            if atom.not_dummy():
                xyz_coords.append(atom.vector)
        
        xyz_coords = numpy.array(xyz_coords)
        xyz_coords.shape = (-1,3)

        if self._anchor != None:
            xyz_coords = self._anchor.reposition(xyz_coords)

        return xyz_coords

class ComplexCoordSys(CoordSys):
    """Object to support the combination of multiple CoordSys objects into one."""

    def __init__(self, sub_parts):
        self._parts = sub_parts

        parents = numpy.array([p._anchor != None for p in self._parts])
        anchors = numpy.array([p.wants_anchor != None for p in self._parts])
        if not (parents & anchors).all():
            raise ComplexCoordSysException("Not all objects that need one have an anchor, and/or some that don't need one have one.")

        l_join = lambda a, b: a + b
        atom_symbols = reduce(l_join, [p.get_chemical_symbols() for p in self._parts])

        cart_coords = self.get_cartesians()
        abstract_coords = numpy.hstack([p.get_internals() for p in self._parts])

        CoordSys.__init__(self, 
            atom_symbols, 
            cart_coords, 
            abstract_coords)

    def get_internals(self):
        ilist = [p.get_internals() for p in self._parts]
        return numpy.hstack(ilist)

    def set_internals(self, x):

        CoordSys.set_internals(self, x)

        i = 0
        for p in self._parts:
            p.set_internals(x[i:i + p.dims])
            i += p.dims

    def get_cartesians(self):
        carts = [p.get_cartesians() for p in self._parts]
        return numpy.vstack(carts)
 

class XYZ(CoordSys):

    __pattern = re.compile(r'(\d+\s+)?(\s*\w\w?(\s+[+-]?\d+\.\d*){3}\s*)+')

    def __init__(self, molstr):
        if molstr[-1] != '\n':
            molstr += '\n'

        if not self.matches(molstr):
            raise CoordSysException("String did not match pattern for XYZ:\n" + molstr)

        coords = re.findall(r"([+-]?\d+\.\d*)", molstr)
        atom_symbols = re.findall(r"(\w\w?).+?\n", molstr)
        self._coords = numpy.array([float(c) for c in coords])

        CoordSys.__init__(self, atom_symbols, 
            self._coords.reshape(-1,3), 
            self._coords)

    def wants_anchor(self):
        return False
    def get_transform_matrix(self, x):
        return numpy.eye(self._dims), 0

    def get_cartesians(self):
        return self._coords.reshape(-1,3)

    def copy(self, new_coords=None):
        new = deepcopy(self)
        new.set_atoms(self._atoms.copy())
        if new_coords != None:
            new.set_internals(new_coords)

        return new

    @staticmethod
    def matches(molstr):
        return XYZ.__pattern.match(molstr) != None

class CoordSysException(Exception):
    def __init__(self, msg):
        self.msg = msg
    def __str__(self):
        return self.msg

class ComplexCoordSysException(Exception):
    def __init__(self, msg):
        self.msg = msg
    def __str__(self):
        return self.msg
 

class ZMatrixException(Exception):
    def __init__(self, msg):
        self.msg = msg
    def __str__(self):
        return self.msg


# OO bhaviour testing
class A():
    x = 3

class B(A):
    def b(self):
        return self.c

class C(B):
    def z(self):
        A.x = 1
        self.c=3
    def y(self):
        return A.x
