from __future__ import with_statement
import re
import ase

import numpy # FIXME: unify
from numpy import array, arange, abs, ones, zeros
from numpy import sin, cos, sqrt
from numpy import arccos as acos

import threading
import numerical
import os
from copy import deepcopy
import operator
from scipy.optimize import fmin_bfgs, fmin

from ase import Atoms
from ase.calculators import SinglePointCalculator

import common
import zmat

class ccsspec():
    def __init__(self, parts, carts=None, mask=None):
        self.parts = parts
        self.carts = carts
        self.mask = mask

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

ccs = ccsspec([xyz, zmt])

"""

def vec_to_mat(v):
    """Generates rotation matrix based on vector v, whose length specifies 
    the rotation angle and whose direction specifies an axis about which to
    rotate."""
    from quat import rotmat
    return rotmat(v)
   #v = numpy.array(v)
   #assert len(v) == 3
   #phi = numpy.linalg.norm(v)
   #a = numpy.cos(phi/2)
   #if phi < 0.02:
   #    #print "Used Taylor series approximation, phi =", phi
   #    """
   #    q2 = sin(phi/2) * v / phi
   #       = sin(phi/2) / (phi/2) * v/2
   #       = sin(x)/x + v/2 for x = phi/2

   #    Using a taylor series for the first term...

   #    (%i1) taylor(sin(x)/x, x, 0, 8);
   #    >                           2    4      6       8
   #    >                          x    x      x       x
   #    > (%o1)/T/             1 - -- + --- - ---- + ------ + . . .
   #    >                          6    120   5040   362880

   #    Below phi/2 < 0.01, terms greater than x**8 contribute less than
   #    1e-16 and so are unimportant for double precision arithmetic.

   #    """
   #    x = phi / 2
   #    taylor_approx = 1 - x**2/6. + x**4/120. - x**6/5040. + x**8/362880.
   #    q2 = v/2 * taylor_approx

   #else:
   #    q2 = numpy.sin(phi/2) * v / phi

   #b,c,d = q2

   #m = numpy.array([[ a*a + b*b - c*c - d*d , 2*b*c + 2*a*d,         2*b*d - 2*a*c  ],
   #                 [ 2*b*c - 2*a*d         , a*a - b*b + c*c - d*d, 2*c*d + 2*a*b  ],
   #                 [ 2*b*d + 2*a*c         , 2*c*d - 2*a*b        , a*a - b*b - c*c + d*d  ]])

   #return m


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


    def need_for_completation(self):
        # What is needed to get complete quaternion from
        # the stored variables, default case there is not even
        # an  really anchor (thus None) (for example for Dummy Anchor)
        return None, numpy.zeros(3)


class Dummy(Anchor):
    _dims = 0
    kinds = []
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

    def __init__(self, initial=ones(6), parent = None):
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
        self.kinds = ['anc_q' for i in 1,2,3] + ['anc_c' for i in 1,2,3]

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
        from aof.quat import cart2vec
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
        # calculates directly the rotation quaternion
        best = cart2vec(orig, rotated)

        # verify that there has been the right result given back
        # may be omitted after code is supposed to be save
        m1 = vec_to_mat(best)
        transform = lambda vec3d: numpy.dot(m1, vec3d)
        assert (abs(rotated - array(map(transform, orig))) < 1e-15).all()

        self._coords[0:3] = best
        #return self._coords


    def reposition(self, carts):
        """Based on a quaternion and a translation, transforms a set of 
        cartesion positions x."""
        return self.transformer(carts, self._coords )

    def transformer(self, carts, vec):
        """Based on a quaternion and a translation, transforms a set of
        cartesion positions x."""
       #rot_vec = self.coords[0:3]
       #trans_vec  = self.coords[3:]
        rot_vec = vec[0:3]
        trans_vec  = vec[3:]

        # TODO: need to normalise?
        rot_mat = vec_to_mat(rot_vec)

        transform = lambda vec3d: numpy.dot(rot_mat, vec3d) + trans_vec
        res = numpy.array(map(transform, carts))

        if self._parent != None:
            res += self._parent.get_centroid()

        return res

class RotAndTransLin(RotAndTrans):
    _dims = 5

    def __init__(self, initial=ones(5), parent = None):
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
        self.kinds = ['anc_ql' for i in 1,2] + ['anc_c' for i in 1,2,3]
        self.ac = None
        self.axis = numpy.zeros(3)

    def set_cartesians(self, new, orig, ftol=1e-8):
        """
        Sets value of internal quaternion / translation data using transformed
        and non-transformed cartesian coordinates.

        >>> r = RotAndTransLin(arange(5)*1.0)
        >>> orig = array([[0.,0,0],[0,0,1]])
        >>> new =  array([[0.,0,0],[0,1,0]])
        >>> r.set_cartesians(new, orig)
        >>> new2 = r.reposition(orig)
        >>> abs((new2 - new).round(4))
        array([[ 0.,  0.,  0.],
               [ 0.,  0.,  0.]])

        >>> r2 = RotAndTransLin(arange(5)*1.0)
        >>> orig2 = array([[0.,0,0],[0,0,1]])
        >>> new4  = array([[0.,0,0],[0,0,1]])
        >>> r2.set_cartesians(new4, orig2)
        WARNING: two objects are alike
        >>> new3 = r2.reposition(orig2)
        >>> abs((new3 - new4).round(4))
        array([[ 0.,  0.,  0.],
               [ 0.,  0.,  0.]])
        """
        from quat import cart2veclin
        assert (orig[0] == numpy.zeros(3)).all()

        Anchor.set_cartesians(self)

        # we have only two points for sure
        # they are required to determine rotation/translation
        new = new[:2].copy()
        orig = orig[:2].copy()

        if self._parent != None:
            new -= self._parent.get_centroid()
        self._coords[2:] = new[0]

        # coords as they would be after rotation but not translation
        rotated = new - self._coords[2:]
        free_axis = orig[1] - orig[0]
        free_axis = free_axis / numpy.linalg.norm(free_axis)

        # calculates directly the quaternion in vector description
        best = cart2veclin(orig, rotated)
        best = self.remove_component( best, free_axis)
        j = self.complete_vec(best,free_axis)
        mat = vec_to_mat(j)
        transform = array([numpy.dot(mat,o) for o in orig])
        # test if code is correct
        assert (abs(transform - rotated) < 1e-12).all()

        self._coords[0:2] = best
        #return self._coords

    def remove_component(self, v, axis):
        """
        make a two variable from a three variable vector
        """
        if self.ac == None:
           laxis = list(axis)
           self.ac = laxis.index(max(laxis))

        if not (self.axis == axis).all():
          self.axis = axis

        return remove_third_component(self.ac, v)


    def complete_vec(self, v, axis):
        """the vector for the rotation matrix, expandation of the two
           compontents still there, as the third can be calculated together
           with the free axis of rotation (for the linear object) axis with
           v*axis = 0"""

        # this should be stored, to be the same during the whole calculation
        # hopefully the not transformed cartesian coordinates should start with
        # the same ones, self.ac is the indice which will not be used by the vector
        if self.ac == None:
           laxis = list(axis)
           self.ac = laxis.index(max(laxis))

        if not (self.axis == axis).all():
          self.axis = axis

        return vector_completation(self.ac, v, axis)

    def reposition(self, carts):
        """Based on a quaternion and a translation, transforms a set of
        cartesion positions x."""
        return self.transformer(carts, self.coords)

    def transformer(self, carts, vec):
        """Based on a quaternion and a translation, transforms a set of
        cartesion positions x."""
        rot_vec1 = vec[0:2]
        free_axis = carts[1] - carts[0]
        free_axis = free_axis / numpy.linalg.norm(free_axis)

        rot_vec = self.complete_vec(rot_vec1, free_axis)
        trans_vec  = vec[2:]

        # TODO: need to normalise?
        rot_mat = vec_to_mat(rot_vec)

        transform = lambda vec3d: numpy.dot(rot_mat, vec3d) + trans_vec
        res = numpy.array(map(transform, carts))

        if self._parent != None:
            res += self._parent.get_centroid()

        return res

    def need_for_completation(self):
        # This is the case, where it was originally designed for: there is
        # an anchor but only two coordinates are given as variables
        return self.ac, self.axis

def remove_third_component(ac, v):
    """
    For the linear case the third compontent does not have to be
    given directly, as it can be recalculated from the other two
    """
    w = numpy.zeros((2,))
    j = 0
    for i in range(3):
      if not i==ac:
        w[j] = v[i]
        j += 1
    return w

def vector_completation(ac, v, axis):
    """
    What to do to get a complete vector, which defines a quaternion
    from a two variable case, ac is the variable in the three vector
    case, which has been skipped, v is the 2 dimensional vector, axis
    defines the axis, along which the product w * axis vanishes
    """
    w = numpy.zeros((3,))
    nv = []
    for i in range(3):
      if not i==ac:
        nv.append(i)

    for j, n in enumerate(nv):
      w[n] = v[j]

    # w[self.ac] ==0 on the right hand side, thus need not be omited from the dot
    # product
    w[ac] = numpy.dot(w,axis) /axis[ac]

    return w

class CoordSys(object):
    """Abstract coordinate system. Sits on top of an ASE Atoms object."""

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

    def completion_anchor(self):
        # returns informations of included anchor
        return [self._anchor.need_for_completation()]

    def set_calculator(self, calc_tuple):
#        print "set_calc",type(self), calc_tuple

        self.calc_tuple = calc_tuple
        if calc_tuple == None:
            return

       #con, args, kwargs = calc_tuple
       #assert callable(con)
       #assert type(args) == list
       #assert type(kwargs) == dict
        calc = calc_tuple

       #try:
       #    calc = con(*args, **kwargs)
       #except TypeError, e:
       #   raise CoordSysException(str(e) + " (Are you supplying the wrong keyword arguments to this calculator?)")
        self._atoms.set_calculator(calc)

    """def copy(self):
        cs = deepcopy(self)
        cs._atoms = self._atoms.copy()

        calc = deepcopy(self._atoms.get_calculator())
        cs.set_calculator(calc)

        return cs"""

    def test_matches(self, s):
        if not self.matches(s):
            raise CoordSysException("String:\n %s\n doesn't specify an object of type %s" % (s, self.__class__.__name__))

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
#        print self.__dict__
#        assert False
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
        #print type(self)
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
        assert len(mask) == self.dims, "%d != %d" % (len(mask), self.dims)

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

        assert j == len(x), "%d %d" % (j, len(x))
        return newx
        
    def _mask(self, x):
        if self._var_mask == None:
            return x

        assert len(x) == len(self._var_mask)
        l = [x[i] for i in range(len(x)) if self._var_mask[i]]
        if isinstance(x, list):
            return l
        else:
            return numpy.array(l)

    def _mask_old(self, x):
        """Builds a vector of internal variables by only including those that 
        are specified."""
        if self._var_mask == None:
            return x

        j = 0
        output = numpy.zeros(self.dims) * numpy.nan
        assert len(self._var_mask) == len(x), "%d != %d" % (len(self._var_mask), len(x))
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

    @property
    def kinds(self):
        return self._mask(self._kinds + self._anchor.kinds)

    def get_internals(self):
        raw = numpy.hstack([self._coords.copy(), self._anchor.coords])
        masked = self._mask(raw)
        return masked.copy()

    def apply_constraints(self, vec):
        return vec

    def get_cartforces(self):
        cart_pos = self.get_cartesians()
        self._atoms.set_positions(cart_pos)

        return self._atoms.get_forces()


    def get_forces(self, flat=False, **kwargs):
        cart_pos = self.get_cartesians()
        self._atoms.set_positions(cart_pos)

        forces_cartesian = self._atoms.get_forces().flatten()
        transform_matrix, errors = self.get_transform_matrix(self._mask(self._coords))

        
        print "Numerical Differentiation max error", errors.max()

        forces_coord_sys = numpy.dot(transform_matrix, forces_cartesian)
        
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
        geom_str = '\n'.join(list) # + '\n\n'

        return geom_str

    def native_str(self):
        pass

    def int2cartprime(self, x):
        """"
        returns the matrix o fderivatives dCi/DIj from the ith cartesian coordinate
        after jth internal coordinate.
        FIXME: This should be possible without need of numerical differences of the
        get_transform_matrix algorithm
        """
        mat,__ = self.get_transform_matrix(x)
        return mat


    def get_transform_matrix(self, x):
        """Returns the matrix of derivatives dCi/dIj where Ci is the ith cartesian coordinate
        and Ij is the jth internal coordinate, and the error."""

        def int2cartf( x):
            return self.int2cart(x).flatten()

        nd = numerical.NumDiff(method='simple')
        mat, err = nd.numdiff(int2cartf, x)
        """nd2 = numerical.NumDiff()
        mat2,err2 = nd2.numdiff(self.int2cart, x)
        print "err2",err2.max()
        m = abs(mat - mat2).max()
        import func
        alexei_nd = func.NumDiff(f=self.int2cart)
        mat3 = alexei_nd.fprime(x)
        m2 = abs(mat - mat3).max()
        print "x",x
        print "m",m
        print "m2",m"""
        return mat, err

    def int2cart2(self, x):
        """Based on a vector x of new internal coordinates, returns a 
        vector of cartesian coordinates. The internal dictionary of coordinates 
        is updated."""

        with self._state_lock:
            old_x = self._mask(self._coords)

            self.set_internals(x)
            y = self.get_cartesians()

            self.set_internals(old_x)

            return y.flatten()

    def transform_contra_to_co(self, vec, place):
        return contoco(self.int2cartprime, place, vec)

    def transform_co_to_contra(self, vec, place):
        return cotocon(self.int2cartprime, place, vec)

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
        """ Tests whether string |s| can correctly specify a ComplexCoordSys object.
        Text format to create this object is any valid Python syntax that 
        will result in a variable sys_list pointing to an instance.

        >>> s = "xyz1 = 'H 0. 0. 0.'\\nxyz2 = 'H 0. 0. 1.08'\\nccs = ccsspec([XYZ(xyz1), XYZ(xyz2)])"
        >>> ComplexCoordSys.matches(s)
        True

        >>> s = "xyz1 = 'H 0. 0. 0.'\\nxyz2 = 'H 0. 0. 1.08'\\nwrong_name = ccsspec([XYZ(xyz1), XYZ(xyz2)])"
        >>> ComplexCoordSys.matches(s)
        False
        """

        #TODO: even better would be if this function returned Boolean, "reason"

        ccs = None

        if type(s) is str:
            try:
                exec(s)
            except SyntaxError, err:
                return False
        else:
            ccs = s

        if ccs == None:
            return False

        # list of conditions
        if not hasattr(ccs, 'parts'):
            print "There was no 'ccs' object defined in molecule input, or it had no field 'parts'."
            return False

        if not reduce(operator.and_, [isinstance(i, CoordSys) for i in ccs.parts]):
            print "Not every member of 'ccs.parts' was a CoordSys object."
            return False

        if ccs.carts != None and not isinstance(ccs.carts, XYZ):
            print "The specified object containing the Cartesian coordinates had type %s but it must be either an XYZ object or None." % type(ccs.carts)
            return False
            
        if ccs.mask != None and not isinstance(ccs.mask, list) and \
            not reduce(operator.and_, [isinstance(i,bool) for i in ccs.mask]):
            
            print "A variable mask was given as 'ccs.mask' but it was not a list of booleans"
            return False

        return True

    def __init__(self, s):
        """
        >>> s = "xyz1 = 'H 0. 0. 0.'\\nxyz2 = 'H 0. 0. 1.08'\\nccs = ccsspec([XYZ(xyz1), XYZ(xyz2)])"
        >>> ccs2 = ComplexCoordSys(s)
        >>> ccs2.get_cartesians()
        array([[ 0.  ,  0.  ,  0.  ],
               [ 0.  ,  0.  ,  1.08]])
        
        >>> exec(s)
        >>> ccs3 = ComplexCoordSys(ccs.parts)
        There was no 'ccs' object defined in molecule input, or it had no field 'parts'.
        >>> ccs3.get_cartesians()
        array([[ 0.  ,  0.  ,  0.  ],
               [ 0.  ,  0.  ,  1.08]])

        >>> new_carts = [[3,2,1],[4,5,6]]
        >>> ccs3.set_cartesians(new_carts)
        >>> (new_carts == ccs3.get_cartesians()).all()
        True

        """
        from_str = isinstance(s, str)
        carts = None
        if from_str:
            self.test_matches(s)
            exec(s)
            self._parts = ccs.parts
            if ccs.carts != None:
                carts = ccs.carts.get_cartesians()
            mask = ccs.mask
        elif self.matches(s):
            ccs = None
            ccs = s
            self._parts = ccs.parts
            if ccs.carts != None:
                carts = ccs.carts.get_cartesians()
            mask = ccs.mask
        else:
            self._parts = s
            mask = None

        has_no_anc = numpy.array([p._anchor == Dummy() for p in self._parts])
        wants_anc = numpy.array([p.wants_anchor for p in self._parts])
        if not (has_no_anc ^ wants_anc).all():
            raise ComplexCoordSysException("Not all objects that need an anchor have one, and/or some that don't need one have one.")

        l_join = lambda a, b: a + b
        atom_symbols = reduce(l_join, [p.get_chemical_symbols() for p in self._parts])
        if carts != None and atom_symbols != ccs.carts.get_chemical_symbols():
            s = "CCS: %s\nand\nCARTS: %s" % (str(atom_symbols), str(ccs.carts.get_chemical_symbols()))
            raise ComplexCoordSysException("Atomic symbols of given Cartesian geometry do not match those specified for construction of the ComplexCoordSystem:\n" + s)

        if carts == None:
            carts = self.get_cartesians()
        abstract_coords = numpy.hstack([p.get_internals() for p in self._parts])

        CoordSys.__init__(self, 
            atom_symbols, 
            carts, 
            abstract_coords)

        # list of names of all constituent vars
        list = [p.var_names for p in self._parts]
        self.var_names = [n for ns in list for n in ns]

        self.set_cartesians(carts)
        self.set_var_mask(mask)

        # list of variables types
        list = [p.kinds for p in self._parts]
        self._kinds = reduce(operator.add, list)

    def completion_anchor(self):
        # returns for each part the values for completing the anchor objects
        # starts with None for object without real anchor (e.g. Cartesian part)
        # -1 for complete anchor and else the number of the variable to add
        # the second part of the tuple is the axis vector
        return [p._anchor.need_for_completation() for p in self._parts]

    def get_internals(self):
        ilist = [p.get_internals() for p in self._parts]
        iarray = numpy.hstack(ilist)

        assert (iarray == self._coords).all()
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

        assert (self.get_cartesians()  - x).sum().round(4) == 0, \
            "Convergence not reached when numerically setting internals from \
            Cartesians, errors were: %s" % (self.get_cartesians() - x)
        self._coords = numpy.hstack([p.get_internals() for p in self._parts])

    def int2cart(self, new_internals):
        """
        Transforms internal coordinates to Cartesian ones without changing the internal
        state of this class
        """
        i = 0
        new_xyz = []
        x = self._demask(new_internals)
        for p in self._parts:
            new_xyz.append(p.int2cart(x[i:i + p.dims]))
            i += p.dims
        assert i == self.dims + self._exclusions_count
        return numpy.vstack(new_xyz)

 
class XYZ(CoordSys):

    __pattern = re.compile(r'(\d+[^\n]*\n[^\n]*\n)?(\s*\w\w?(\s+[+-]?\d+\.\d*){3}\s*)+')

    def __init__(self, mol):

        assert isinstance(mol, str), "'mol' had type %s" % type(mol)
        molstr = self._get_mol_str(mol)
        if molstr[-1] != '\n':
            molstr += '\n'

        # Check if there is an energy specification, if so create a 
        # SinglePointCalculator to return this energy (or zero otherwise)
        molstr_lines = molstr.splitlines()
        if len(molstr_lines) > 1:
            line2 = molstr_lines[1]
        else:
            line2 = ''
        fp_nums = re.findall(r"-?\d+\.\d*", line2)
        energy = None
        if len(fp_nums) == 1:
            energy = float(fp_nums[0])
            molstr = '\n'.join(molstr_lines[2:]) + '\n'
        else:
            assert len(fp_nums) in [0,3], "Bad line 2 of XYZ file: '%s'. Should be either blank, or an energy spec or cartesian coords of an atom." % line2

        if not self.matches(molstr):
            raise CoordSysException("String did not match pattern for XYZ:\n" + molstr)

        coords = re.findall(r"([+-]?\d+\.\d*)", molstr)
        atom_symbols = re.findall(r"([a-zA-Z][a-zA-Z]?).+?\n", molstr)
        self._coords = numpy.array([float(c) for c in coords])

        CoordSys.__init__(self, atom_symbols, 
            self._coords.reshape(-1,3), 
            self._coords)

        if energy != None:
            calc_tuple = SinglePointCalculator, [energy, None, None, None, self._atoms], {}
            self.set_calculator(calc_tuple)

        assert len(self._coords) == self._coords.size
        self._kinds = ['cart' for i in self._coords]


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

    def int2cart(self, new_internals):
        """
        Transforms internal coordinates to Cartesian ones without changing the internal
        state of this class
        This routine is for the case, where the internal coordinates are also Cartesian
        """
        all_internals = self._demask(new_internals)
        return all_internals


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

        >>> z = ZMatrix2(s,RotAndTrans(initial=zeros(6)))
        >>> z.get_internals().round(3)[:9]
        array([ 1.09 ,  1.09 ,  1.09 ,  1.09 ,  1.911,  1.911,  1.911,  2.094,  2.094])

        >>> from aof.zmat import ZMatrix3, ZMat

        >>> ints = z.get_internals()

        >>> carts = z.get_cartesians().flatten()
        >>> max(abs(carts - z.int2cart(ints).flatten()))
        0.0
        >>> ints2 = None
        >>> ints2 = ints + array([1e-4, 0.01, 0.002, 0.0003, 2e-3, 1e-5, 0.1, 0.002, 0.01, 0.01, 0.2, 1e-6, 0.001, 0.3 , 0.1])
        >>> z.set_internals(ints2)
        >>> carts2 = z.get_cartesians().flatten()

        >>> ints3 = (ints + ints2) / 2
        >>> int_co_3 = contoco( z.int2cartprime, ints3, (ints2 - ints))
        Average iterations per variable 7.4
        >>> (numpy.dot(int_co_3, ints3 - ints) - numpy.dot(carts2 - carts, carts2-carts)).round(3)
        -0.318

        >>> ints4 = (ints + ints3) / 2
        >>> z.set_internals(ints3)
        >>> carts3 = z.get_cartesians().flatten()
        >>> int_co_3_2 = contoco( z.int2cartprime, ints4, (ints3 - ints))
        Average iterations per variable 7.4
        >>> (numpy.dot(int_co_3_2, ints3 - ints) - numpy.dot(carts3 - carts, carts3-carts)).round(7)
        4.07e-05

        >>> z.set_internals(ints)


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
        >>> abs(z.get_internals() - ints).round()[:9]
        array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.])

        >>> r = repr(z)
        
        Tests of anchored Z-Matrix
        ==========================

        >>> ac = [0.,0.,0.,3.,1.,1.]
        >>> a = RotAndTrans(ac)
        >>> z = ZMatrix2(s, anchor=a)
        >>> z.get_internals().round(3)
        array([ 1.09 ,  1.09 ,  1.09 ,  1.09 ,  1.911,  1.911,  1.911,  2.094,  2.094,  0.   ,  0.   ,  0.   ,  3.   ,  1.   ,  1.   ])

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
                             (([ ]*\w+(\s*|\s*[=]\s*)[+-]?(\d+\.\d*[ \t\r\f\v]*\n|\n))+)\s*$""", re.X)
        return (zmt.match(mol_text) != None)

    def __init__(self, mol, anchor=Dummy()):

        assert isinstance(anchor, Anchor)
        molstr = self._get_mol_str(mol)

        self.zmtatoms = []
        #self.vars = dict()
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
        coordsall = re.findall(r"\w+(\s+|\s*[=]\s*)([+-]?\d+\.\d*)\n", variables_text)
        coords = [ci[1] for ci in coordsall]
        if coords == []:
            self._coords = numpy.array([1. for var in self.var_names])
        else:
            self._coords = numpy.array([float(c) for c in coords])
        N = len(self._coords)
    
        # Create data structure of atoms. There is both an ordered list and an 
        # unordered dictionary with atom index as the key.
        lines = zmt_spec.split("\n")
        for ix, line in myenumerate(lines, start=1):
            a = ZMTAtom(line, ix)
            self.zmtatoms.append(a)

        # Dictionaries of (a) dihedral angles and (b) angles
        kinds = list(self.var_names)
        isvar = lambda v: isinstance(v, str)
        d = dict()
        def f(s):
            if s[0] == '-':
                s = s[1:]
            return s

        for atom in self.zmtatoms:
            if isvar(atom.dst):
                d[f(atom.dst)] = 'dst'
            if isvar(atom.ang):
                d[f(atom.ang)] = 'ang'
            if isvar(atom.dih):
                d[f(atom.dih)] = 'dih'
        kinds = [d[k] for k in kinds]

        # Create dictionary of variable values (unordered) and an 
        # ordered list of variable names.
        for i in range(N):
            #key = self.var_names[i]
            if kinds[i] in ('ang', 'dih'):
                self._coords[i] = common.DEG_TO_RAD * self._coords[i]
            #val = float(self._coords[i])
            #self.vars[key] = val

        self._kinds = kinds
        # check that z-matrix is fully specified
        self.zmt_ordered_vars = []
        for atom in self.zmtatoms:
            self.zmt_ordered_vars += atom.all_vars()
        for var in self.zmt_ordered_vars:
            if not var in self.var_names:
                raise ZMatrixException("Variable '" + var + "' not given in z-matrix")

        assert len(kinds) == len(self.var_names) == len(self._coords)

        symbols = [a.name for a in self.zmtatoms]

        # setup object which actually does zmt <-> cartesian conversion
        spec = self.make_spec(self.zmtatoms)
        self._zmt = zmat.ZMat(spec)
        self.spec = spec
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
        for i, var in enumerate(self.var_names):
            if self._kinds[i] in ('dih', 'ang'):
                mystr += var + "\t" + str(self._coords[i] * common.RAD_TO_DEG) + "\n"
            else:
                mystr += var + "\t" + str(self._coords[i]) + "\n"
        return mystr

    def set_internals(self, internals):
        """Update stored list of variable values."""

        CoordSys.set_internals(self, internals)

        #for i, var in zip( internals[0:self._dims], self.var_names ):
        #    self.vars[var] = i

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

    def int2cart(self, new_internals):
        """
        Transforms internal coordinates to Cartesian ones without changing the internal
        state of this class
        """
        all_internals = self._demask(new_internals)
        xyz_coords =  self._zmt.f(all_internals[:-self._anchor.dims])
        #xyz_coords =  self._zmt.f(all_internals)
        if self._anchor != None:
            xyz_coords = self._anchor.transformer(xyz_coords, all_internals[-self._anchor.dims:])
        return xyz_coords

def contoco(F, pos, vec):
    """
    assuming that F is a function to get the derivatives, transforming vectors
    of kind vec into cartesians, and that all takes place at position pos,
    this function transforms contra in covariant vectors
    """
    B = F(pos)
    return btb(B, vec)

def cotocon(F, pos, vec):
    """
    assuming that F is a function to get the derivatives, transforming vectors
    of kind vec into cartesians, and that all takes place at position pos,
    this function transforms co in contravariant vectors
    """
    B = numpy.matrix(F(pos))
    B = numpy.asarray(B.I).T
    return btb(B, vec)

def btb(B, vec):
    """
    Returns the product B^T B vec
    """
    return numpy.dot(B, numpy.dot(B.T, vec))
    #return numpy.dot(B.T, numpy.dot(B, vec))

def fake_xyz_string(ase_atoms, start = None ):
    """
    Like creating an xyz-file but let it go to a string
    The string can be read in and understod by MolInterface
    """
    symbols = ase_atoms.get_chemical_symbols()
    if start == None:
        xyz_str = '%d\n\n' % len(symbols)
    else:
        xyz_str = '%d\n\n' % (len(symbols) - start)
    for i, (s, (x, y, z)) in enumerate(zip(symbols, ase_atoms.get_positions())):
        if start == None or (i - start > -1):
            xyz_str += '%-2s %22.15f %22.15f %22.15f\n' % (s, x, y, z)
    return xyz_str


# The next three routines take a plain ase atoms object and transform it:
# to one of the possible coordinate systems given in this module

def ase2xyz(c_ase):
    c_xyz = XYZ( fake_xyz_string(c_ase))
    c_xyz.set_cartesians(c_ase.get_positions())
    return c_xyz

def ase2int(c_ase, zmts, nums):
    if nums > 1:
      c_int = ZMatrix2(zmts, RotAndTrans())
    else:
      c_int = ZMatrix2(zmts, RotAndTransLin())
    c_int.set_cartesians(c_ase.get_positions())
    return c_int

def ase2ccs(c_ase, zmts, el_nums, elem_num):
    zmt1s = []
    for i, zmt1 in enumerate(zmts):
      if el_nums[i] > 1:
        zmt1s_1 = ZMatrix2(zmt1, RotAndTrans())
      else:
        zmt1s_1 = ZMatrix2(zmt1, RotAndTransLin())
      zmt1s.append(zmt1s_1)

    num_atoms = len(c_ase.get_atomic_numbers())
    symb_atoms = c_ase.get_chemical_symbols()
    carts = XYZ(fake_xyz_string(c_ase))
    diffhere =  (elem_num) / 3
    if diffhere < num_atoms:
      co_all = XYZ(fake_xyz_string(c_ase, start = diffhere))
      zmt1s.append(co_all)
    ccs1 =  ccsspec(zmt1s, carts=carts)
    return ccs1

def xyz2ccs(xyz_str, zmts, el_nums, elem_num):
    zmt1s = []
    for i, zmt1 in enumerate(zmts):
      if el_nums[i] > 1:
        zmt1s_1 = ZMatrix2(zmt1, RotAndTrans())
      else:
        zmt1s_1 = ZMatrix2(zmt1, RotAndTransLin())
      zmt1s.append(zmt1s_1)

    symb_atoms = findall(r"([a-zA-Z][a-zA-Z]?).+?\n", xyz_str)
    num_atoms = len(symb_atoms)
    carts = XYZ(fake_xyz_string(c_ase))
    diffhere =  (elem_num) / 3
    if diffhere < num_atoms:
      lines = xyz_str.splitlines(True)
      num_carts_ats = num_atoms - (elem_num/ 3)
      co_cart = "%d\n\n" % num_carts_ats
      for i in range(1, num_carts_ats + 1):
          co_cart += lines[-i]
      zmt1s.append(co_all)
    ccs1 =  ccsspec(zmt1s, carts=xyz_str)
    return zmt1

def enforce_short_way(zmti):
  from numpy import pi, zeros, dot
  from numpy.linalg import norm
  """
  This routine takes a list of coordinate system objects and
  transforms them in a way, which ensures that by an interpolation
  the shortest way is taken. This is especially important for dihedral
  angles, where each angle can be given in a multiple way, differing by
  2 * PI but describing exactly the same system. And there are quaternions
  which may be defined different ways. As the interpolation is later on just
  linear in all coordinates, these special coordiantes are identified and
  adapted if needed
  """
  for i_n1, zm in enumerate(zmti):
    m1 =  zm.get_internals()
    can1 = zm.completion_anchor()
    i_n2 = i_n1 + 1
    # just ckeck two succeding geometry objects, for the last
    # make dummy compare with isself (needs FIXME?)
    # competion anchor is needed if there are internal parts in the geometries
    # which have only two atoms
    if i_n2 > len(zmti) - 1:
       i_n2 = len(zmti) - 1
    m2 = zmti[i_n2].get_internals()
    can2 = zmti[i_n2].completion_anchor()
    ancs = []
    anc = []
    anc_l = []
    ancs_l = []

    for i, k in enumerate(zm.kinds):
       if k == "dih":
           # dihedrals can just be handled by themselves
           delta = m2[i] - m1[i]
           while delta >  pi: delta -= 2.0 * pi
           while delta < -pi: delta += 2.0 * pi
           m2[i] = m1[i] + delta
       elif k == "anc_q":
           # the quaternions need some more logic, one has to deal
           # with the three variables defining one in one go, thus
           # only collect them here.
           if ancs == []:
             last = i
           else:
             last = ancs[-1]
           # the quaternion kinds have to follow directly after each other
           # if there is a gap, the variable belongs to the next one
           if (i-last) > 1:
              # store this quaternion already
              anc.append(ancs)
              ancs = []
           ancs.append(i) # collect for one quaternion
       elif k == "anc_ql":
           if ancs_l == []:
             last_l = i
           else:
             last_l = ancs_l[-1]
           # the quaternion kinds have to follow directly after each other
           # if there is a gap, the variable belongs to the next one
           if (i-last_l) > 1:
              # store this quaternion already
              anc_l.append(ancs_l)
              ancs_l = []
           ancs.append(i) # collect for one quaternion
    if ancs != []: # store the last one
       anc.append(ancs)
    if ancs_l != []: # store the last one
       anc_l.append(ancs_l)
    # now loop over the quaternions rather than all variables

    for anchor in anc:
       v1 = m1[anchor]
       v2 = m2[anchor]
       # v1 and v2 are the quaternion (there are two to compare)
       lv1 = norm(v1)
       lv2 = norm(v2)
       # norm is value for angle
       if dot(v1, v2) < 0:
          # find out if they are lying in the same direction
          lv2 *= -1.0
       # make norm(vi) = 1
       v2 /= lv2
       v1 /= lv1
       delta = lv2 - lv1
       # normalize the interval between two angles:
       while delta >  pi: delta -=  2.0 * pi
       while delta < -pi: delta +=  2.0 * pi
       lv2 = lv1 + delta
       # give back changed norm
       v2 *= lv2
       v1 *= lv1
       # reduce result if necessary
       m1[anchor] = v1
       m2[anchor] = v2
    # return changed values
    zmti[i_n1].set_internals(m1)
    zmti[i_n2].set_internals(m2)

    # Linear Rot and Trans need special handling:
    icm = -1
    rat = RotAndTrans()
    for i, anchor in enumerate(anc_l):
       ic1 = (None, zeros(3))
       b_back = [0, 1, 2]
       while ic1[0] == None:
         icm += 1
         # get information to get complete quaternion, if only two
         # variables are stored, None means non-internal object, if it
         # is not none, one is reached, where the anchor gives variables
         # to the complete system
         ic1 = can1[icm]
       v1 = m1[anchor]
       v2 = m2[anchor]
       # v1 and v2 are the quaternion (there are two to compare)
       # first enlarge them, if the reduced (linear) kind has been given
       assert(ic1[0] > -1)
       assert(ic1[0] == can2[icm][0])
       assert(ic1[1] == can2[icm][1]).all()
       b_back.remove(ic1[0])
       mat1 = vec_to_mat(vector_completation(ic1[0], v1, ic1[1]))
       mat2 = vec_to_mat(vector_completation(can2[icm][0], v2, can2[icm][1]))
       rat.set_cartesians(numpy.dot(mat1, ic1[1]))
       or1 = rat._coords[0:3]
       rat.set_cartesians(numpy.dot(mat2, ic1[1]), numpy.dot(mat1, ic1[1]))
       rot = rat._coords[0:3]
       delta = norm(rot)
       rot /= delta
       # normalize the interval between two angles:
       while delta >  pi: delta -=  2.0 * pi
       while delta < -pi: delta +=  2.0 * pi
       rot *= delta
       v2 = mult_vec_of_quad(or1, rot)
       # give back changed norm
       # reduce result if necessary
       m2[anchor] = v2[b_back]
    # return changed values
    zmti[i_n2].set_internals(m2)

def mult_vec_of_quad(v1, v2):
    """
    Two rotations performed one after another are packed to one single of them
    This is done in quaternion space of Quat (the unitary quaternion) where it is
    a simple multiplication, the result is transformed back in the vector description
    """
    from aof.quat import Quat
    v1_ang = sqrt(numpy.dot(v1, v1))
    v2_ang = sqrt(numpy.dot(v2, v2))
    if v1_ang == 0:
      v1r = v1
    else:
      v1r = v1 / v1_ang
    if v2_ang == 0:
      v2r = v2
    else:
      v2r = v2 / v2_ang
    qp = Quat([cos(v1_ang/2)] + [vi*sin(v1_ang/2) for vi in v1r])
    ql = Quat([cos(v2_ang/2)] + [vi*sin(v2_ang/2) for vi in v2r])
    qa = qp * ql
    qaa = numpy.asarray(qa)[1:]
    ang = acos(qa[0]) * 2
    if abs(numpy.dot(qaa, qaa)) == 0:
      lqaa = 1
    else:
      lqaa = sqrt(numpy.dot(qaa, qaa))
    # give back as vector
    vall = numpy.array([qai/lqaa * ang for qai in qaa])
    return vall



# Testing the examples in __doc__strings, execute
# "python gxmatrix.py", eventualy with "-v" option appended:
if __name__ == "__main__":
    import doctest
    doctest.testmod()

# You need to add "set modeline" and eventually "set modelines=5"
# to your ~/.vimrc for this to take effect.
# Dont (accidentally) delete these lines! Unless you do it intentionally ...
# Default options for vim:sw=4:expandtab:smarttab:autoindent:syntax

