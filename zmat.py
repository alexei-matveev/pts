"""
This module provides conversion from z-matrix coordiantes to cartesian.

For the moment the |atoms| provided as an input to |z2c| should be
subscriptable objects with following entries:

    atom[0] = index of a in 4-atom tuple x-a-b-c or None
    atom[2] = index of b in 4-atom tuple x-a-b-c or None
    atom[4] = index of c in 4-atom tuple x-a-b-c or None

    atom[1] = x-a distance
    atom[3] = x-a-b angle (currently in degrees)
    atom[5] = x-a-b-c angle (currently in degrees)

Change the definitons of |fst|, |snd|, |trd|, |dst|, |ang|, |dih|
in the body of |z2c| if the structure of the atom object is going to
change.

An example use with tuples:

    >>> z2c([(None,), (0, 10.0, None), (1, 3.0, 0, 90.0, None)])
    Vector([Vector([0.0, 0.0, 0.0]), Vector([10.0, 0.0, 0.0]), Vector([10.0, 1.8369701987210297e-16, 3.0])])

The entries may come in any order, but the cross-referencing should
be correct:

    >>> z2c([(2, 10.0, None), (0, 3.0, 2, 90.0, None), (None,)])
    Vector([Vector([10.0, 0.0, 0.0]), Vector([10.0, 1.8369701987210297e-16, 3.0]), Vector([0.0, 0.0, 0.0])])

The entries may also be lists instead of tuples:

    >>> z2c([[2, 10.0, None], [0, 3.0, 2, 90.0, None], [None]])
    Vector([Vector([10.0, 0.0, 0.0]), Vector([10.0, 1.8369701987210297e-16, 3.0]), Vector([0.0, 0.0, 0.0])])

"""

from math import pi
# from numpy import array as V, sin, cos, cross, dot, sqrt
from vector import Vector as V, dot, cross
from bmath import sin, cos, sqrt

class ZMError(Exception):
    pass

def z2c(atoms):
    """Generates cartesian coordinates from z-matrix and the current set of 
    internal coordinates. Based on code in OpenBabel."""
    
    # how to get the atoms involved in bonding:
    def fst(x):
        a = x[0] # a in x-a-b-c
        if a is None: return None
        return atoms[a]

    def snd(x):
        b = x[2] # b in x-a-b-c
        if b is None: return None
        return atoms[b]

    def trd(x):
        c = x[4] # c in x-a-b-c
        if c is None: return None
        return atoms[c]

    DEG_TO_RAD = pi / 180.0

    # how to get values for bond length/angles:
    def dst(x): return x[1]              # x-a length
    def ang(x): return x[3] * DEG_TO_RAD # x-a-b angle in radians
    def dih(x): return x[5] * DEG_TO_RAD # x-a-b-c dihedral angle in radians

    #
    # nothing below depends on the type and structure of an atom in "atoms" ...
    #

    # cache atomic positions, keys are the element IDs of |atoms|:
    cache = dict()

    def pos(atom):
        "Return atomic position, compute if necessary and memoize"

        aid = id(atom)
        # we use id() to cache results, that means we may formally
        # unnecessarily recompute positions for atoms with the
        # same set of internal coordiantes, but having different IDs!
        # Not the usual case anyway.
        if aid in cache:
            # catch infinite recursion:
            if cache[aid] is None:
                raise ZMError("cycle")
            else:
                return cache[aid]
        else:
            # prevent infinite recursion, put some nonsense:
            cache[aid] = None
            # for actual computation see "pos1" below:
            try:
                p = pos1(atom)
            except ZMError, e:
                raise ZMError("pos1 of", atom, e.args)
            cache[aid] = p
            return p

    def pos1(x):
        "Compute atomic position, using memoized funciton pos()"

        # a in x-a-b-c:
        a = fst(x)

        # default origin:
        if a is None: return V((0.0, 0.0, 0.0))

        # sanity:
        if a is x: raise ZMError("same x&a")

        # position of a, and distance x-a:
        avec = pos(a)
        distance = dst(x)

        # b in x-a-b-c:
        b = snd(x)

        # default X-axis:
        if b is None: return V((dst(x), 0.0, 0.0)) # FXIME: X-axis

        # sanity:
        if b is a: raise ZMError("same x&b")
        if b is x: raise ZMError("same x&b")

        # position of b and angle x-a-b:
        bvec = pos(b)
        angle = ang(x)

        # c in x-a-b-c:
        c = trd(x)

        # position of c and dihedral angle x-a-b-c:
        if c is None:
            # default plane here:
            cvec = V((0.0, 1.0, 0.0))
            dihedral = 90.0 * DEG_TO_RAD
        else:
            cvec = pos(c)
            dihedral = dih(x)

        # sanity:
        if c is b: raise ZMError("same b&c")
        if c is a: raise ZMError("same a&c")
        if c is x: raise ZMError("same x&c")

        # normalize vector:
        def normalise(v):
            n = sqrt(dot(v, v))
            # numpy will just return NaNs:
            if n == 0.0: raise ZMError("divide by zero")
            return v / n

        v1 = avec - bvec
        v2 = avec - cvec

        n = cross(v1,v2)
        nn = cross(v1,n)

        n = normalise(n)
        nn = normalise(nn)

        n *= -sin(dihedral)
        nn *= cos(dihedral)
        v3 = n + nn
        v3 = normalise(v3)
        v3 *= distance * sin(angle)
        v1 = normalise(v1)
        v1 *= distance * cos(angle)
        p = avec + v3 - v1

        return p

    # compute all positions:
    xyz = []
    for atom in atoms:
        xyz.append(pos(atom))

    # convert to vector/numpy:
    xyz = V(xyz)
    return xyz

# Testing the examples in __doc__strings, execute
# "python gxmatrix.py", eventualy with "-v" option appended:
if __name__ == "__main__":
    import doctest
    doctest.testmod()

# You need to add "set modeline" and eventually "set modelines=5"
# to your ~/.vimrc for this to take effect.
# Dont (accidentally) delete these lines! Unless you do it intentionally ...
# Default options for vim:sw=4:expandtab:smarttab:autoindent:syntax
