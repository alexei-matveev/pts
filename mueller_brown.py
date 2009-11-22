__all__ = ["energy", "gradient"] # "MuellerBrown"]

from numpy import exp, array

AA = (-200., -100., -170.,  15.)
aa = (  -1.,   -1.,   -6.5,  0.7)
bb = (   0.,    0.,   11.,   0.6)
cc = ( -10.,  -10.,   -6.5,  0.7)
xx = (   1.,    0.,   -0.5, -1.)
yy = (   0.,    0.5,   1.5,  1.)

# find MB constant below:
def energy(v): return MB.energy(v)
def gradient(v): return MB.gradient(v)

class MuellerBrown():
    """
    Mueller, Brown, Theoret. Chim. Acta (Berl.) 53, 75-93 (1979)
    Potential used in string method publications

    Initialze the calculator:

        >>> f = MuellerBrown()

    Compute energy:
        >>> f.energy((0,0))
        -48.401274173183893

        >>> f.energy((-0.558, 1.442))
        -146.69948920058778

        >>> f.energy((0.623, 0.028))
        -108.16665005353302

        >>> f.energy((-0.050, 0.467))
        -80.767749248757724

    or gradients:

        >>> f.gradient((-0.558, 1.442))
        array([ -1.87353829e-04,   2.04493895e-01])

        >>> f.gradient((0.623, 0.028))
        array([-0.28214372, -0.19043391])

        >>> f.gradient((-0.050, 0.467))
        array([ 0.04894393,  0.44871342])

    
    (WARNING: OCRed data below)

    Table 1. Energy minima and saddle points of the analytical, two-parametric model potential v
    Energy minima a
               E        x     y
    Minimum A -146.700 -0.558 1.442
    Minimum B -108.167  0.623 0.028
    Minimum C  -80.768 -0.050 0.467
    
    Saddle points b
     E       x     y      E      x     y
    -40.665 -0.822 0.624 -72.249 0.212 0.293
    
    Saddle points c
    Run (P1, P2) E      x    y    E      x    y    N d
    1 (A, B)    -40.67 -0.82 0.62 --               124
    2 (B, A)    -40.68 -0.82 0.62 --               155
    3 (B, C)    --                -72.26 0.22 0.29 74
    4 (C, B)    --                -72.27 0.21 0.30 73
    5 (A, C)    -40.68 -0.81 0.62 --               112
    6 (C, A)    -40.67 -0.82 0.62 --               113

    a Determined by unconstrained energy minimization using the simplex method [25].
    b Determined by minimization of |grad E|^2
    c Determined by the constrained simplex optimization procedure 8.
    d Total number of function evaluations.
    """

    def __str__(self):
        return "MuellerBrown"

    def energy(self, v):

        x = v[0]
        y = v[1]

        f = 0.0
        for A, a, b, c, x0, y0 in zip(AA, aa, bb, cc, xx, yy):
            f += A * exp( a * (x - x0)**2 + b * (x - x0) * (y - y0) + c * (y - y0)**2 )

        return f

    def gradient(self, v):

        x = v[0]
        y = v[1]

        dfdx = 0.0
        dfdy = 0.0
        for A, a, b, c, x0, y0 in zip(AA, aa, bb, cc, xx, yy):
            expt = A * exp( a * (x - x0)**2 + b * (x - x0) * (y - y0) + c * (y - y0)**2 )
            dfdx += ( 2 * a * (x - x0) + b * (y - y0) ) * expt
            dfdy += ( 2 * c * (y - y0) + b * (x - x0) ) * expt

        return array((dfdx, dfdy))

MB = MuellerBrown()

# python mueller_brown.py [-v]:
if __name__ == "__main__":
    import doctest
    doctest.testmod()

# Default options for vim:sw=4:expandtab:smarttab:autoindent:syntax

