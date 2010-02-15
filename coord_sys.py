from __future__ import with_statement
import re
import ase
from ase import Atoms


import numpy # FIXME: unify
from numpy import array, arange, abs

import threading
import numerical
import os
from copy import deepcopy
import operator
from scipy.optimize import fmin_bfgs, fmin

import common
import zmat

# Some strings for testing
testccs1 = """
H2 =  "H  0.0 0.0 0.0\\n"
H2 += "H  0.0 0.0 0.908\\n"

H2O =  "H\\n"
H2O += "O 1 ho1\\n"
H2O += "H 2 ho2 1 hoh\\n"
H2O += "\\n"
H2O += "ho1   1.09\\n"
H2O += "ho2   1.09\\n"
H2O += "hoh  120.1\\n"

xyz = XYZ(H2)

anchor = RotAndTrans([1.,0.,0.,3.,1.,1.], parent=xyz)
zmt = ZMatrix2(H2O, anchor=anchor)

ccs = [xyz, zmt]

"""

def vec_to_mat(v):
    """Generates rotation matrix based on vector v, whose length specifies 
    the rotation angle and whose direction specifies an axis about which to
    rotate."""

    v = numpy.array(v)
    assert len(v) == 3
    phi = numpy.linalg.norm(v)
    a = numpy.cos(phi/2)
    if phi < 0.02:
        #print "Used Taylor series approximation, phi =", phi
        """
        q2 = sin(phi/2) * v / phi
           = sin(phi/2) / (phi/2) * v/2
           = sin(x)/x + v/2 for x = phi/2

        Using a taylor series for the first term...

        (%i1) taylor(sin(x)/x, x, 0, 8);
        >                           2    4      6       8
        >                          x    x      x       x
        > (%o1)/T/             1 - -- + --- - ---- + ------ + . . .
        >                          6    120   5040   362880

        Below phi/2 < 0.01, terms greater than x**8 contribute less than 
        1e-16 and so are unimportant for double precision arithmetic.

        """
        x = phi / 2
        taylor_approx = 1 - x**2/6. + x**4/120. - x**6/5040. + x**8/362880.
        q2 = v/2 * taylor_approx

    else:
        q2 = numpy.sin(phi/2) * v / phi

    b,c,d = q2

    m = numpy.array([[ a*a + b*b - c*c - d*d , 2*b*c + 2*a*d,         2*b*d - 2*a*c  ],
               [ 2*b*c - 2*a*d         , a*a - b*b + c*c - d*d, 2*c*d + 2*a*b  ],
               [ 2*b*d + 2*a*c         , 2*c*d - 2*a*b        , a*a - b*b - c*c + d*d  ]])

    return m


class Anchor(object):
    """Abstract object to support positioning of a Internal Coordinates object 
    in Cartesian Space."""

    def __init__(self, initial):
        if self.dims != len(initial):
            raise ComplexCoordSysException("initial value did not match required number of dimensions: " + str(initial))
    def reposition(self, *args):
        assert False, "Abstract function"

    def set_cartesians(self, *args, **kwargs):
        pass

    def set(self, t):
        assert len(t) == self.dims
        self._coords = t

    def get(self):
        return self._coords

    def get_dims(self):
        return self._dims
    dims = property(get_dims)

    coords = property(get, set)

class Dummy(Anchor):
    _dims = 0
    def __init__(self, initial=numpy.array([])):
        Anchor.__init__(self, initial)
        self._coords = numpy.array([])

    def __eq__(self, d):
        return d.__class__ == self.__class__
    
    def __ne__(self, d):
        return not self.__eq__(d)

    def reposition(self, x):
        return x

class RotAndTrans(Anchor):
    _dims = 6

    def __init__(self, initial, parent = None):
        """
        initial:
            initial value

        parent:
            Other anchored object that this anchor gives an orientation 
            relative to.
        """
        Anchor.__init__(self, initial)
        self._parent = parent

        initial = numpy.array(initial)
        self._coords = initial

    def quaternion2rot_mat(self, quaternion):
        """See http://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation"""
        a,b,c,d = quaternion

        m = numpy.array([[ a*a + b*b - c*c - d*d , 2*b*c + 2*a*d,         2*b*d - 2*a*c  ],
                         [ 2*b*c - 2*a*d         , a*a - b*b + c*c - d*d, 2*c*d + 2*a*b  ],
                         [ 2*b*d + 2*a*c         , 2*c*d - 2*a*b        , a*a - b*b - c*c + d*d  ]])

        return m

    def set_cartesians(self, new, orig, ftol=1e-8):
        """
        Sets value of internal quaternion / translation data using transformed 
        and non-transformed cartesian coordinates.

        >>> r = RotAndTrans(arange(6)*1.0)
        >>> orig = array([[0.,0,0],[0,0,1],[0,1,0]])
        >>> new =  array([[0.,0,0],[0,1,0],[1,0,0]])
        >>> r.set_cartesians(new, orig)
        >>> new2 = r.reposition(orig)
        >>> abs((new2 - new).round(4))
        array([[ 0.,  0.,  0.],
               [ 0.,  0.,  0.],
               [ 0.,  0.,  0.]])

        >>> parent = XYZ("C 1.0 -1.0 1.0\\n")
        >>> parent.get_centroid()
        array([ 1., -1.,  1.])

        >>> r = RotAndTrans(arange(6)*1.0, parent=parent)
        >>> orig = array([[0.,0,0],[0,0,1],[0,1,0]])
        >>> new  = array([[0.,0,0],[0,0,1],[0,1,0]])
        >>> r.set_cartesians(new, orig)
        >>> new3 = r.reposition(orig)
        >>> abs((new3 - new).round(4))
        array([[ 0.,  0.,  0.],
               [ 0.,  0.,  0.],
               [ 0.,  0.,  0.]])
        """
        assert (orig[0] == numpy.zeros(3)).all()

        Anchor.set_cartesians(self)

        # only three points are required to determine rotation/translation
        new = new[:3].copy()
        orig = orig[:3].copy()

        if self._parent != None:
            new -= self._parent.get_centroid()
        self._coords[3:] = new[0]

        # coords as they would be after rotation but not translation
        rotated = new - self._coords[3:]

        def f(i):
            mat = vec_to_mat(i)
            transformed = array([numpy.dot(mat, i) for i in orig])
            v =  (transformed - rotated).flatten()
            tmp = numpy.sqrt(numpy.dot(v, v))
            return tmp

        old_rot = self._coords[:3].copy()
        best, err, _, _, _  = fmin(f, old_rot, ftol=ftol*0.1, full_output=1, disp=0, maxiter=2000)
        if err > ftol:
            raise CoordSysException("Didn't converge in anchor parameterisation, %.20f > %.20f" %(err, ftol))
        self._coords[0:3] = best
        #return self._coords


    def reposition(self, carts):
        """Based on a quaternion and a translation, transforms a set of 
        cartesion positions x."""

        rot_vec = self.coords[0:3]
        trans_vec  = self.coords[3:]

        # TODO: need to normalise?
        rot_mat = vec_to_mat(rot_vec)

        transform = lambda vec3d: numpy.dot(rot_mat, vec3d) + trans_vec
        res = numpy.array(map(transform, carts))

        if self._parent != None:
            res += self._parent.get_centroid()

        return res

class CoordSys(object):
    """Abstract coordinate system. Sits on top of an ASE Atoms object."""

    dih_vars = dict()

    def __init__(self, atom_symbols, atom_xyzs, abstract_coords, anchor=Dummy(), cell=None, pbc=None):

        # enforce correct capitalisation
        atom_symbols = [s.title() for s in atom_symbols]

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

        # hack to provide extra functionality with ASE
        self.pass_through = False

        # stores (constructor, args, kwargs) for calculator generation
        self.calc_tuple = None

    def _get_mol_str(self, s):
        if os.path.exists(s):
            return common.file2str(s)
        else:
            return s

    def set_calculator(self, calc_tuple):
        if calc_tuple == None:
            return

        con, args, kwargs = calc_tuple
        assert callable(con)
        assert type(args) == list
        assert type(kwargs) == dict

        self.calc_tuple = calc_tuple
        try:
            calc = con(*args, **kwargs)
        except TypeError, e:
           raise CoordSysException(str(e) + " (Are you supplying the wrong keyword arguments to this calculator?)")
        self._atoms.set_calculator(calc)

    """def copy(self):
        cs = deepcopy(self)
        cs._atoms = self._atoms.copy()

        calc = deepcopy(self._atoms.get_calculator())
        cs.set_calculator(calc)

        return cs"""

    def test_matches(self, s):
        if not self.matches(s):
            raise CoordSysException("String:\n %s\n doesn't specify an object of type %s" % (s, self.__class__.__name))

    def copy(self, new_coords=None):
        new = deepcopy(self)
        new._atoms = self._atoms.copy()
        if new_coords != None:
            new.set_internals(new_coords)

        return new


    def __str__(self):
        s = '\n'.join([self.__class__.__name__, str(self._coords), str(self._var_mask)])
        return s

    def __len__(self):
        return len(self._atoms)
    def get_dims(self):
        return self._dims + self._anchor.dims - self._exclusions_count

    dims = property(get_dims)
    def __getstate__(self):
        odict = self.__dict__.copy()
        del odict["_atoms"]
        del odict["_state_lock"]

        # create a string representing the atoms object using the ASE io functionality
        tmp = os.tmpnam()
        old_calc = self._atoms.calc
        self._atoms.calc = None
        ase.write(tmp, self._atoms, format="traj")
        f = open(tmp, "rb")
        odict["pickled_atoms"] = f.read()
#        odict["pickled_calc"] = self._atoms.calc
        f.close()
        os.unlink(tmp)

        self._atoms.calc = old_calc

        return odict

    def __setstate__(self, idict):
        pickled_atoms = idict.pop("pickled_atoms")
        # temp
        """kwargs = [('ismear', '1'),
             ('sigma', '0.15')
             , ('xc'     , 'VWN')
             , ('isif'   , '2')
             , ('enmax'  , '300')
             , ('idipol' , '3')
             , ('enaug'  , '300')
             , ('ediffg' , '-0.02')
             , ('voskown', '1')
             , ('istart' , '1')
             , ('icharg' , '1')
             , ('nelmdl' , '0')
             , ('kpts'   , [1,1,1])]"""

        calc_tuple = idict.pop("calc_tuple")

        # temp
        #pickled_calc = ase.Vasp(**dict(kwargs)) #dict.pop("pickled_calc")
        self.__dict__.update(idict)

        tmp = os.tmpnam()
        f = open(tmp, mode="wb")
        f.write(pickled_atoms)
        f.close()
        self._atoms = ase.read(tmp, format="traj")
        os.unlink(tmp)

        self.set_calculator(calc_tuple)
        self._state_lock = threading.RLock()

    def __getattr__(self, a):
        """Properties not available in this object are brought through from the Atoms object."""
        """if a in self.__dict__:
            return self.__dict__[a]
        else:"""
        return self._atoms.__getattribute__(a)

    def get_positions(self):
        """ASE style interface function"""

        # hack to provide extra compatibility with ASE
        if self.pass_through:
            return self._atoms.get_positions()

        return common.make_like_atoms(self._mask(self._coords))

    def get_centroid(self):
        c = numpy.zeros(3)
        carts = self.get_cartesians()
        for v in carts:
            c += v
        c = c / len(carts)
        return c

    """def set_calculator(self, calc):
        return self._atoms.set_calculator(calc)"""

    def set_positions(self, x):
        """ASE style interface function"""
        assert x.shape[1] == 3

        #demasked = self._demask(x.flatten()[0:)

        self.set_internals(x.flatten()[0:self.dims])
        assert len(self._coords) >= self._dims

    def set_var_mask(self, mask):
        """Sets a variable exclusion mask. Only include variables that are True."""

        if mask == None:
            return

        mask = numpy.array(mask)
        assert mask.dtype == bool
        assert len(mask) == self.dims

        self._exclusions_count = len(mask) - sum(mask)
        self._var_mask = mask.copy()

    def _demask(self, x):
        """Builds a vector to replace the current internal state vector, that 
        will only update those elements which specified by the mask."""
        if self._var_mask == None:
            return x

        newx = numpy.hstack([self._coords.copy(), self._anchor.coords])
        j = 0
        assert len(self._var_mask,) == len(newx)
        for i, m in enumerate(self._var_mask):
            if m:
                newx[i] = x[j]
                j += 1

        assert j == len(x)
        return newx
        
    def _mask(self, x):
        """Builds a vector of internal variables by only including those that 
        are specified."""
        if self._var_mask == None:
            return x

        j = 0
        output = numpy.zeros(self.dims) * numpy.nan
        assert len(self._var_mask) == len(x)
        for m, xi in zip (self._var_mask, x):
            if m:
                output[j] = xi
                j += 1

        assert j == self.dims
        assert not numpy.isnan(output).any()

        return output

    def set_internals(self, x):

        assert not numpy.isnan(x).any()

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

    def set_cartesians(self, *args):
        """Sets internal coordinates (including those of the Anchor) based on 
        the given set of cartesians and the pure, non-rotated cartesians."""
        if self._anchor != None:
            self._anchor.set_cartesians(*args)

    def get_internals(self):
        raw = numpy.hstack([self._coords.copy(), self._anchor.coords])
        masked = self._mask(raw)
        return masked.copy()

    def apply_constraints(self, vec):
        return vec

    def get_forces(self, flat=False, **kwargs):
        cart_pos = self.get_cartesians()
        self._atoms.set_positions(cart_pos)

        forces_cartesian = self._atoms.get_forces().flatten()
        transform_matrix, errors = self.get_transform_matrix(self._mask(self._coords))
        
        print "numdiff errors", errors.max()

        forces_coord_sys = numpy.dot(transform_matrix, forces_cartesian)
        print "forces_coord_sys", forces_coord_sys
        
        forces_coord_sys = self.apply_constraints(forces_coord_sys)

        #forces_masked = self._mask(forces_coord_sys)
        forces_masked = forces_coord_sys

        if flat:
            return forces_masked
        else:
            return common.make_like_atoms(forces_masked)
    
    def get_potential_energy(self):

        cart_pos = self.get_cartesians()
        self._atoms.set_positions(cart_pos)

        return self._atoms.get_potential_energy()
       

    """def copy(self, new_coords=None):
        assert False, "Abstract function" """

    @property
    def atoms(self):
        self._atoms.positions = self.get_cartesians()
        return self._atoms

    @property
    def atoms_count(self):
        return len(self._atoms)

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
            old_x = self._mask(self._coords)

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

    def __repr__(self):
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
    """Supports optimisations in terms of z-matrices.
    
    TODO: test angles given as constants. IS there a problem with radians/degrees?
    """
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

    def __init__(self, mol, anchor=Dummy()):

        molstr = self._get_mol_str(mol)

        self.zmtatoms = []
        self.vars = dict()
        self.zmtatoms_dict = dict()
        self._anchor = anchor

        if not self.matches(molstr):
            raise ZMatrixException("Z-matrix not found in string:\n" + molstr)

        parts = re.search(r"(?P<zmt>.+?)\n\s*\n(?P<vars>.+)", molstr, re.S)

        # z-matrix text, specifies connection of atoms
        zmt_spec = parts.group("zmt")

        # variables text, specifies values of variables
        variables_text = parts.group("vars")
        self.var_names = re.findall(r"(\w+).*?\n", variables_text)
        coords = re.findall(r"\w+\s+([+-]?\d+\.\d*)\n", variables_text)
        self._coords = numpy.array([float(c) for c in coords])
    
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

    def __repr__(self):
        return self.zmt_str()

    @property
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
    """Object to support the combination of multiple CoordSys objects into one.
    
    >>> l = [ZMatrix2("H\\nH 1 hh\\n\\nhh  1.0\\n")]
    >>> css = ComplexCoordSys(l)
    Traceback (most recent call last):
        ...
    ComplexCoordSysException: Not all objects that need an anchor have one, and/or some that don't need one have one.

    Test of Internals <-> Cartesians interconversion
    ================================================

    Using testccs1...
    >>> ccs = ComplexCoordSys(testccs1)
    >>> ints = ccs.get_internals().round(3)

    Get it's carts and transform them...
    >>> xyz = ccs.get_cartesians() + 10.
    >>> mat = vec_to_mat([3,2,1])
    >>> xyz = numpy.array([numpy.dot(mat, c) for c in xyz])

    Re-generate internals from those carts
    >>> ccs.set_cartesians(xyz)
    >>> ints2 = ccs.get_internals().round(3)

    Make sure the internal coords of the water have changed.
    >>> (ints == ints2)[6:9].all()
    True

    """

    @staticmethod
    def matches(s):
        """Text format to create this object is any valid Python syntax that 
        will result in a variable sys_list pointing to an instance.

        >>> s = "xyz1 = 'H 0. 0. 0.'\\nxyz2 = 'H 0. 0. 1.08'\\nccs = [XYZ(xyz1), XYZ(xyz2)]"
        >>> ComplexCoordSys.matches(s)
        True

        >>> s = "xyz1 = 'H 0. 0. 0.'\\nxyz2 = 'H 0. 0. 1.08'\\nwrong_name = [XYZ(xyz1), XYZ(xyz2)]"
        >>> ComplexCoordSys.matches(s)
        False
        """
        ccs = None
        try:
            exec(s)
        except SyntaxError, err:
            pass

        return isinstance(ccs, list) and reduce(operator.and_, [isinstance(i, CoordSys) for i in ccs])

    def __init__(self, s):
        """
        >>> s = "xyz1 = 'H 0. 0. 0.'\\nxyz2 = 'H 0. 0. 1.08'\\nccs = [XYZ(xyz1), XYZ(xyz2)]"
        >>> ccs = ComplexCoordSys(s)
        >>> ccs.get_cartesians()
        array([[ 0.  ,  0.  ,  0.  ],
               [ 0.  ,  0.  ,  1.08]])
        
        >>> exec(s)
        >>> ccs = ComplexCoordSys(ccs)
        >>> ccs.get_cartesians()
        array([[ 0.  ,  0.  ,  0.  ],
               [ 0.  ,  0.  ,  1.08]])

        >>> new_carts = [[3,2,1],[4,5,6]]
        >>> ccs.set_cartesians(new_carts)
        >>> (new_carts == ccs.get_cartesians()).all()
        True

        """
        if isinstance(s, str):
            self.test_matches(s)
            exec(s)
            self._parts = ccs
        else:
            self._parts = s

        has_no_anc = numpy.array([p._anchor == Dummy() for p in self._parts])
        wants_anc = numpy.array([p.wants_anchor for p in self._parts])
        if not (has_no_anc ^ wants_anc).all():
            raise ComplexCoordSysException("Not all objects that need an anchor have one, and/or some that don't need one have one.")

        l_join = lambda a, b: a + b
        atom_symbols = reduce(l_join, [p.get_chemical_symbols() for p in self._parts])

        cart_coords = self.get_cartesians()
        abstract_coords = numpy.hstack([p.get_internals() for p in self._parts])

        CoordSys.__init__(self, 
            atom_symbols, 
            cart_coords, 
            abstract_coords)

        # list of names of all constituent vars
        list = [p.var_names for p in self._parts]
        self.var_names = [n for ns in list for n in ns]

        # list of all dihedral vars
        def addicts(x,y):
            x.update(y)
            return x
        list = [p.dih_vars for p in self._parts]
        self.dih_vars = reduce(addicts, list)

    def get_internals(self):
        ilist = [p.get_internals() for p in self._parts]
        iarray = numpy.hstack(ilist)

        return self._mask(iarray).copy()

    """def set_var_mask(self, m):
        CoordSys.set_var_mask(self, m)

        i = 0
        for p in self._parts:
            #print "p.dims:",p.dims
            p_old_dims = p.dims
            p.set_var_mask(m[i:i + p_old_dims])
            i += p_old_dims

        assert i == len(m)"""

    def set_internals(self, x):
        

        CoordSys.set_internals(self, x)
        x = self._demask(x)

        i = 0
        for p in self._parts:
            p.set_internals(x[i:i + p.dims])
            i += p.dims

        assert i == self.dims + self._exclusions_count

    def get_cartesians(self):
        carts = [p.get_cartesians() for p in self._parts]
        return numpy.vstack(carts)
    
    def set_cartesians(self, x):
        CoordSys.set_cartesians(self)
        x = numpy.array(x).reshape(-1,3)
        asum = 0
        for p in self._parts:
            p.set_cartesians(x[asum:asum + p.atoms_count])
            asum += p.atoms_count
 

class XYZ(CoordSys):

    __pattern = re.compile(r'(\d+\s+)?(\s*\w\w?(\s+[+-]?\d+\.\d*){3}\s*)+')

    def __init__(self, mol):

        molstr = self._get_mol_str(mol)
        if molstr[-1] != '\n':
            molstr += '\n'

        if not self.matches(molstr):
            raise CoordSysException("String did not match pattern for XYZ:\n" + molstr)

        coords = re.findall(r"([+-]?\d+\.\d*)", molstr)
        atom_symbols = re.findall(r"([a-zA-Z][a-zA-Z]?).+?\n", molstr)
        self._coords = numpy.array([float(c) for c in coords])

        CoordSys.__init__(self, atom_symbols, 
            self._coords.reshape(-1,3), 
            self._coords)

        self.dih_vars = dict()

    def __repr__(self):
        return self.xyz_str()

    @property
    def wants_anchor(self):
        return False
    def get_transform_matrix(self, x):
        m = numpy.eye(self._dims)

        j = []
        if self._var_mask != None:
            for a,b in zip(m, self._var_mask):
                if b:
                    j.append(a)
            m = numpy.array(j)

        assert m.shape[0] == self.dims
        return m, numpy.array([0.0])

    def get_var_names(self):
        return ["<cart>" for i in range(self.dims)]
    var_names = property(get_var_names)

    def get_cartesians(self):
        return self._coords.reshape(-1,3)

    def set_cartesians(self, x):
        CoordSys.set_cartesians(self)

        tmp = x.reshape(-1)
        assert len(tmp) == len(self._coords)
        self._coords = tmp

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

class ZMatrix2(CoordSys):
    """
    Supports optimisations in terms of z-matrices. This version uses zmat.py. 
    This second version exists because zmat.py has xyz -> zmt support.

        >>> s = "C\\nH 1 ch1\\nH 1 ch2 2 hch1\\nH 1 ch3 2 hch2 3 hchh1\\nH 1 ch4 2 hch3 3 -hchh2\\n\\nch1    1.09\\nch2    1.09\\nch3    1.09\\nch4    1.09\\nhch1 109.5\\nhch2 109.5\\nhch3 109.5\\nhchh1  120.\\nhchh2  120.\\n"

        >>> z = ZMatrix2(s)
        >>> z.get_internals()
        array([ 1.09      ,  1.09      ,  1.09      ,  1.09      ,  1.91113553,
                1.91113553,  1.91113553,  2.0943951 ,  2.0943951 ])

        >>> ints = z.get_internals()

        >>> from numpy import round
        >>> print round(z.get_cartesians(), 7)
        [[ 0.         0.         0.       ]
         [ 1.09       0.         0.       ]
         [ 0.504109   0.        -0.9664233]
         [-0.3638495 -0.9685445  0.3429796]
         [-0.9555678 -1.4333516  0.827546 ]]
        >>> cs = z.get_cartesians() + 1000
        >>> z.set_cartesians(cs)
        >>> from numpy import abs
        >>> abs(z.get_internals() - ints).round()
        array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.])

        >>> r = repr(z)
        
        Tests of anchored Z-Matrix
        ==========================

        >>> ac = [0.,0.,0.,3.,1.,1.]
        >>> a = RotAndTrans(ac)
        >>> z = ZMatrix2(s, anchor=a)
        >>> z.get_internals()
        array([ 1.09      ,  1.09      ,  1.09      ,  1.09      ,  1.91113553,
                1.91113553,  1.91113553,  2.0943951 ,  2.0943951 ,  0.        ,
                0.        ,  0.        ,  3.        ,  1.        ,  1.        ])

        >>> ints = z.get_internals().copy()

        >>> (z.get_cartesians() - array([3.,1.,1.])).round(3)
        array([[ 0.   ,  0.   ,  0.   ],
               [ 1.09 ,  0.   ,  0.   ],
               [ 0.504,  0.   , -0.966],
               [-0.364, -0.969,  0.343],
               [-0.956, -1.433,  0.828]])

        >>> ints[-6:-3] = [1,2,3]
        >>> ints_old = ints.copy()
        >>> z.set_internals(ints)
        >>> cs = z.get_cartesians() + 1000
        >>> mat = vec_to_mat([3,2,1])
        >>> cs = numpy.array([numpy.dot(mat, i) for i in cs])
        >>> z.set_cartesians(cs)
        >>> res = abs(z.get_internals() - ints_old).round(2)
        >>> res[:-6]
        array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.])
        >>> (res[-6:] != ints_old[-6:]).all()
        True

        >>> r = repr(z)
        
   
    """
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

    def __init__(self, mol, anchor=Dummy()):

        molstr = self._get_mol_str(mol)

        self.zmtatoms = []
        self.vars = dict()
        self.zmtatoms_dict = dict()
        self._anchor = anchor

        if not self.matches(molstr):
            raise ZMatrixException("Z-matrix not found in string:\n" + molstr)

        parts = re.search(r"(?P<zmt>.+?)\n\s*\n(?P<vars>.+)", molstr, re.S)

        # z-matrix text, specifies connection of atoms
        zmt_spec = parts.group("zmt")

        # variables text, specifies values of variables
        variables_text = parts.group("vars")
        self.var_names = re.findall(r"(\w+).*?\n", variables_text)
        coords = re.findall(r"\w+\s+([+-]?\d+\.\d*)\n", variables_text)
        self._coords = numpy.array([float(c) for c in coords])
    
        # Create data structure of atoms. There is both an ordered list and an 
        # unordered dictionary with atom index as the key.
        lines = zmt_spec.split("\n")
        for ix, line in myenumerate(lines, start=1):
            a = ZMTAtom(line, ix)
            self.zmtatoms.append(a)

        # Dictionaries of (a) dihedral angles and (b) angles
        self.dih_vars = dict()
        self.angles = dict()
        for atom in self.zmtatoms:
            if atom.dih_var() != None:
                self.dih_vars[atom.dih_var()] = 1
                self.angles[atom.dih_var()] = 1
            if atom.ang != None:
                self.angles[atom.ang] = 1

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

        symbols = [a.name for a in self.zmtatoms]

        spec = self.make_spec(self.zmtatoms)
        self._zmt = zmat.ZMat(spec)
        CoordSys.__init__(self, symbols, 
            self.get_cartesians(), 
            self._coords,
            anchor)
        

    def make_spec(self, zmtatoms):
        l = []
        for a in zmtatoms:
            if a.a == None:
                con = ()
            elif a.b == None:
                con = (a.a - 1,)
            elif a.c == None:
                con = (a.a - 1, a.b - 1,)
            else:
                con = (a.a - 1, a.b - 1, a.c - 1)

            l.append(con)
        return l

    def __repr__(self):
        return self.zmt_str()

    @property
    def wants_anchor(self):
        return True

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

        CoordSys.set_internals(self, internals)

        for i, var in zip( internals[0:self._dims], self.var_names ):
            self.vars[var] = i

    def set_cartesians(self, carts):
        """Calculates internal coordinates based on given cartesians."""

        internals_zmt = self._zmt.pinv(carts)
        pure_carts = self._zmt.f(internals_zmt)

        CoordSys.set_cartesians(self, carts, pure_carts)

        self.set_internals(numpy.hstack([internals_zmt, self._anchor.get()]))

    def get_cartesians(self, anchor=True):
        """Generates cartesian coordinates from z-matrix and the current set of 
        internal coordinates."""
        
        xyz_coords = self._zmt.f(self._coords)

        if self._anchor != None and anchor:
            xyz_coords = self._anchor.reposition(xyz_coords)

        return xyz_coords

# Testing the examples in __doc__strings, execute
# "python gxmatrix.py", eventualy with "-v" option appended:
if __name__ == "__main__":
    import doctest
    doctest.testmod()

# You need to add "set modeline" and eventually "set modelines=5"
# to your ~/.vimrc for this to take effect.
# Dont (accidentally) delete these lines! Unless you do it intentionally ...
# Default options for vim:sw=4:expandtab:smarttab:autoindent:syntax

