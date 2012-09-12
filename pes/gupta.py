#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
See  "Theoretical   Studies  of  Palladium-Gold   Nanoclusters:  Pd-Au
Clusters with up to 50 Atoms", Faye Pittaway, Lauro Oliver Paz-Borbon,
Roy L.  Johnston, Haydar  Arslan, Riccardo Ferrando, Christine Mottet,
Giovanni Barcaro, and Alessandro  Fortunelli J.  Phys.  Chem.  C 2009,
113, 9141–9152, http://dx.doi.org/10.1021/jp9006075

A potential energy surface is a function of coordinates only. The rest
of  the info,  here atomic  composition, is  provided  at construction
time:

    >>> f = Gupta(["Au", "Au", "Pd", "Pd", "Pd", "Au"])

Now  f(x) is  a Func(tion)  of 6x3  coordinates organized  into  a 6x3
array:

    >>> x = array([[1.60705688238, -0.00283613710287, 0.00283613710287],
    ...            [-0.00283613710287, 1.60705688238, 0.00283613710287],
    ...            [-0.00525477215725, -1.59087506386, 0.00525477215725],
    ...            [-1.59087506386, -0.00525477215725, 0.00525477215725],
    ...            [-0.00525477215725, -0.00525477215725, 1.59087506386],
    ...            [-0.00283613710287, -0.00283613710287, -1.60705688238]])

This is the value at this point:

    >>> f(x)
    -10.320392207689487

And this is the magnitude of its gradient:

    >>> max(abs(f.fprime(x)))
    17.423253922379555

Those coordinates do  not look like a stationary  point.  Minimize the
function:

    >>> from pts.fopt import minimize
    >>> xm, info = minimize(f, x)

    >>> info["converged"], info["iterations"]
    (True, 17)

The energy gets lower:

    >>> f(xm)
    -18.776211979362952

This is still  somewhat higher than, -18.893036 eV,  the number quoted
for  a C2v  structure in  the supplementary  material of  the original
publication.

The corresponding gradients vanish, though:

    >>> max(abs(f.fprime(xm)))
    6.8573842848529409e-07

This is the gradient function:

    >>> g = NumDiff(f.fprime)

This is the hessian at the stationary point:

    >>> H = g.fprime(xm)

Its shape is 6 x 3 x 6 x 3, change that

    >>> H.shape = (18, 18)

in order to be able to use in the eigensolver:

    >>> from scipy.linalg import eigh
    >>> e, V = eigh(H)
    >>> min(e)
    -6.0878979649077398e-12

Virtually zero. That is strange ... Exchange two atoms:

    >>> f = Gupta(["Au", "Au", "Au", "Pd", "Pd", "Pd"])
    >>> xm, info = minimize(f, xm)
    >>> f(xm)
    -18.893036306423806

Now it  compares well to -18.893036  eV --- the  earlier structure was
not C2v but rather C3v.
"""

from numpy import array, dot, sqrt, exp, max, abs, zeros, shape
from pts.func import NumDiff # Func

#
# TABLE 2: Gupta Potential Parameters Used in This Study
#
# (a) average parameters (average)
#
#             pair         A/eV    ζ/eV    p        q       ro/Å
#          -----------     ---------------------------------------
PARAMSA = {("Pd", "Pd") : (0.1746, 1.7180, 10.867,  3.7420, 2.7485),
           ("Au", "Au") : (0.2061, 1.7900, 10.229,  4.0360, 2.8840),
           ("Pd", "Au") : (0.1900, 1.7500, 10.540,  3.8900, 2.8160)}
#
# (b) experimental-fitted parameters (exp-fit)
#
PARAMSB = {("Pd", "Pd") : (0.1715, 1.7019, 11.000,  3.7940, 2.7485),
           ("Au", "Au") : (0.2096, 1.8153, 10.139,  4.0330, 2.8840),
           ("Pd", "Au") : (0.2764, 2.0820, 10.569,  3.9130, 2.8160)}
#
# (c) DFT-fitted parameters (DFT-fit)
#
PARAMSC =  {("Pd", "Pd") : (0.1653, 1.6805, 10.8535, 3.7516, 2.7485),
            ("Au", "Au") : (0.2091, 1.8097, 10.2437, 4.0445, 2.8840),
            ("Pd", "Au") : (0.1843, 1.7867, 10.5420, 3.8826, 2.8160)}

class Gupta(NumDiff):
    """
    FIXME: ...
    """
    def __init__(self, symbols, params=PARAMSC):

        # Save atomic symbols:
        self.symbols = symbols

        # Save force field parameters, optional, defaults are DFT-fit,
        # see above:
        self.params = params

    def f(self, x):
        n = len(x)
        assert n == len(self.symbols)

        e = 0.0
        for i in range(n):
            vm = 0.0
            for j in range(n):
                if i == j: continue

                v = x[j] - x[i]
                r = sqrt(dot(v, v))

                # get pair interaction params:
                A, zeta, p, q, ro = self.pair(i, j)

                e += A * exp(- p * (r / ro - 1.0))
                vm += zeta**2 * exp(- 2.0 * q * (r / ro - 1.0))

            e -= sqrt(vm)

        return e

    def fprime(self, x):
        n = len(x)
        assert n == len(self.symbols)

        g = zeros(shape(x))
        gm = zeros(shape(x))
        for i in range(n):
            vm = 0.0
            gm[...] = 0.0
            for j in range(n):
                if i == j: continue

                v = x[j] - x[i]
                r = sqrt(dot(v, v))

                # get pair interaction params:
                A, zeta, p, q, ro = self.pair(i, j)

                gvr = - (p / ro) * A * exp(- p * (r / ro - 1.0)) * (v / r)

                g[j] += gvr
                g[i] -= gvr

                vm += zeta**2 * exp(- 2.0 * q * (r / ro - 1.0))

                gvm = - (2.0 * q / ro) * zeta**2 * exp(- 2.0 * q * (r / ro - 1.0)) * (v / r)

                gm[j] += gvm
                gm[i] -= gvm

            g -= gm / (2.0 * sqrt(vm))

        return g

    def pair(self, i, j):
        "Derive pair interaction parameters from atom indices."
        a = self.symbols[i]
        b = self.symbols[j]

        if (a, b) in self.params:
            return self.params[(a, b)]
        elif (b, a) in self.params:
            return self.params[(b, a)]
        else:
            raise ValueError("No such pairs", [(a, b), (b, a)])

def main(argv):
    from ase.io import read, write
    from pts.fopt import minimize
    from sys import stdout, stderr

    for path in argv[1:]:
        atoms = read(path)
        symbols = atoms.get_chemical_symbols()
        x = atoms.get_positions()

        f = Gupta(symbols)
        xm, info = minimize(f, x)
        # print >> stderr, "e=", f(xm)

        if not info["converged"]:
            print >> stderr, "Not converged:", path

        atoms.set_positions(xm)
        write(stdout, atoms, format="xyz")

# main(["x", "a1.xyz"])
# exit(0)

# python gupta.py [-v]:
if __name__ == "__main__":
    import doctest
    doctest.testmod()
    # import sys
    # main(sys.argv)

# Default options for vim:sw=4:expandtab:smarttab:autoindent:syntax
