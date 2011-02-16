from pts.func import NumDiff
from copy import deepcopy
from numpy import dot

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
    def __init__(self, atoms, f, init_coords, fprime_exist = False):
        """
        CoordSys initalization, storing all needed things

        atoms :  ase.Atoms object, is used for getting forces or energies
                           belonging to a (Cartesian) geometry
        f : Function to transform internal coordinates in Cartesians (may have also
               Func structure with analitical fprime routine.
        init_coords : first internal coordinates for the state

        fprime_exist: as default the fprime will be generated with NumDiff functionalities
                      by setting this variable to True the fprime routine of f will be used
                      instead, in this case it is assumed that f is a function of type Func
        transform_back: it may be wanted to have also a routine which does Cartesian to
                        internal calculation. Only request it, if you have set it
        """
        self._atoms = atoms
        if fprime_exist:
            self._fun = f
        else:
            self._fun = NumDiff(f)
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
        Returns the Cartesian coordinates, which belong
        to the current state of internal coordinates
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

