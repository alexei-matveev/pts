from pts.func import NumDiff
from copy import deepcopy
import operator
from scipy.optimize import fmin_bfgs, fmin

from ase import Atoms
from ase.calculators import SinglePointCalculator

import common
import zmat

from quat import rotmat, _rotmat

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
        from pts.quat import cart2vec
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
        m1 = rotmat(best)
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
        rot_mat = rotmat(rot_vec)

        transform = lambda vec3d: numpy.dot(rot_mat, vec3d) + trans_vec
        res = numpy.array(map(transform, carts))

class CoordSys(object):
    """
    This class mediates between a function which converts internal to
    cartesian coordinates and a molinterface object.

    It is the minimum interface needed for running a path searcher calculation.
    The function should be of form Func from pts.func.
    If the function provides a first derivative function fprime and this is told
    to the CoordSys this function is used for calculating the said derivatives.
    Else a numerical differencial is used for it, as provided by wrapper NumDiff.

    There is also a atoms object stored in order to calculate energies and forces.
    The current internal coordinates are stored also in internal state.
    """
    def __init__(self, atoms, f, init_coords):
        """
        CoordSys initalization, storing all needed things

        atoms :  ase.Atoms object, is used for getting forces or energies
                           belonging to a (Cartesian) geometry
        f : Function to transform internal coordinates in Cartesians (may have also
               Func structure with analitical fprime routine.
        init_coords : first internal coordinates for the state

        transform_back: it may be wanted to have also a routine which does Cartesian to
                        internal calculation. Only request it, if you have set it
        """
        self._atoms = atoms

        # FIXME: always provide a Func to this constructor, this
        # is a tempoarary workaround relying on numerical
        # differentiation:
        try:
            _ = f.fprime
        except AttributeError:
            f = NumDiff(f)

        self._fun = f

        self.set_internals(init_coords)

    def int2cart(self, y):
        """
        Just transform internals y to Cartesians
        """
        return self._fun(y)

    def get_transform_matrix(self, y):
        """
        First derivative, is put in Matrix shape.
        Usually one would expect to get if from fprime
        differently, as there are always three of the
        Cartesians (x, y, z) bound together
        """
        m = self._fun.fprime(y)
        m.shape = (-1, len(y))
        return m.T


    def set_internals(self, y):
        """
        Stores a new internal state for the internal coordinates
        """
        self._coords = y

    def get_internals(self):
        """
        Returns the internal coordinates
        """
        return self._coords

    def get_cartesians(self):
        """
        i = 0
        new_xyz = []
        x = self._demask(new_internals)
        for p in self._parts:
            new_xyz.append(p.int2cart(x[i:i + p.dims]))
            i += p.dims
        assert i == self.dims + self._exclusions_count
        return numpy.vstack(new_xyz)

    def get_transform_matrix(self, x):
        """returns the matrix of derivatives dci/dij where ci is the ith cartesian coordinate
        and ij is the jth internal coordinate, and the error.
        here taking the submatrices of the parts and building up a complete transformation
        matrix. here the parts do not interact with each other
        """
        i = 0
        j = 0
        carts = self.int2cart(x).flatten()
        m = numpy.zeros((self._dims,len(carts)))
        x = self._demask(x)
        err_all = numpy.array([0.0])


        for p in self._parts:
            p_mat, error = p.get_transform_matrix(x[i:i + p.dims])
            sh = p_mat.shape
            a, b = sh
            assert a == p.dims
            m[i:i+p.dims,j:j+b] = p_mat
            err_all += error
            i += p.dims
            j += len(new_xyz)

        assert i == self.dims + self._exclusions_count
        assert j == len(carts)

        if self._var_mask != None:
            m_red = []
            for m_i, b in zip(m, self._var_mask):
                if b:
                   m_red.append(m_i)
            m = numpy.asarray(m_red)

        return m, err_all

 
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
        """returns the matrix of derivatives dci/dij where ci is the ith cartesian coordinate
        and ij is the jth internal coordinate, and the error.
        In the Cartesian case one has only to consider that several coordinates may be fixed
        """
        return self.int2cart(self._coords)

    def get_forces(self):
        """
        Calculates the forces belonging to the current internal coordinate state.
        Transforms them with help of the first derivative matrix in internal
        space (as atoms handles only Cartesian).
        """
        x = self.int2cart(self._coords)
        self._atoms.set_positions(x)
        force_x = self._atoms.get_forces().flatten()
        trans_m = self.get_transform_matrix(self._coords)
        force_y = dot(trans_m, force_x)
        return force_y

    def get_potential_energy(self):
        """
        Calculates the energy to the current internal coordinate state.
        """
        y = deepcopy(self._coords)
        x = self.int2cart(y)
        self._atoms.set_positions(x)
        return self._atoms.get_potential_energy()

