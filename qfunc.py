#!/usr/bin/python
"""
    >>> from ase import Atoms
    >>> ar4 = Atoms("Ar4")

This uses LennardJones() as default calculator:

    >>> pes = QFunc(ar4)

Provide a different one by specifying the second
argument, e.g: pes = QFunc(ar4, gaussian)

    >>> from numpy import array
    >>> x = array([[  1.,  1.,  1. ],
    ...            [ -1., -1.,  1. ],
    ...            [  1., -1., -1. ],
    ...            [ -1.,  1., -1. ]])

    >>> pes(x)
    -0.046783447265625

    >>> pes.fprime(x)
    array([[ 0.02334595,  0.02334595,  0.02334595],
           [-0.02334595, -0.02334595,  0.02334595],
           [ 0.02334595, -0.02334595, -0.02334595],
           [-0.02334595,  0.02334595, -0.02334595]])

    >>> from numpy import linspace
    >>> [ (scale, pes(scale * x)) for scale in linspace(0.38, 0.42, 3) ]
    [(0.38, -5.469484020549146), (0.40000000000000002, -5.9871235862374306), (0.41999999999999998, -5.5011134098626151)]

Find the minimum (first scale the cluster by 0.4 which is close
to the minimum):

    >>> from fopt import minimize
    >>> x = x * 0.4
    >>> xm, fm, _ = minimize(pes, x)
    >>> round(fm, 7)
    -6.0
    >>> xm
    array([[ 0.39685026,  0.39685026,  0.39685026],
           [-0.39685026, -0.39685026,  0.39685026],
           [ 0.39685026, -0.39685026, -0.39685026],
           [-0.39685026,  0.39685026, -0.39685026]])
"""

__all__ = ["QFunc"]

from func import Func
from ase import LennardJones

class QFunc(Func):
    def __init__(self, atoms, calc=LennardJones()):

        # we are going to repeatedly set_positions() for this instance,
        # do we make a copy to avoid effects visible outside?
        self.atoms = atoms # .copy() ?
        self.calc = calc
        self.atoms.set_calculator(calc)

    def f(self, positions):

        # update positions:
        self.atoms.set_positions(positions)

        # request the energy:
        return self.atoms.get_potential_energy()

    def fprime(self, positions):

        # update positions:
        self.atoms.set_positions(positions)

        # request the forces. NOTE: forces are negative of the gradients:
        return  - self.atoms.get_forces()

# python qfunc.py [-v]:
if __name__ == "__main__":
    import doctest
    doctest.testmod()

# Default options for vim:sw=4:expandtab:smarttab:autoindent:syntax
