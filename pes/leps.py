#
# Credit: http://userpage.chemie.fu-berlin.de/~naundorf/MD/leps.f90
#

from numpy import array, empty, sqrt, exp

def pot(X, Y):

#  m1, m2, m3 = 1.0079, 19., 19.

#  m12 = m1 * m2 / (m1 + m2)
#  m23 = m2 * m3 / (m2 + m3)
#  m123 = m1 * (m2 + m3) / (m1 + m2 + m3)
#
#  d23 = Y
#  d123 = sqrt(m12 / m123) * X
#  d12 = d123 - m3 / (m2 + m3) * d23
#  d13 = d12 + d23
#
#  return leps(d12, d23, d13)
   return leps(X, Y, X + Y)

def leps(ab, bc, ac):
    """
    leps

      IN      :     DAB          : A-B distance (in Angstrom)
                    DBC          : B-C distance (in Angstrom)
                    DAC          : A-C distance (in Angstrom)

      RET: LEPS potential value (in kJ/mol)

    Purpose:
      Calculate three-body A-B-C LEPS potential for the given
      distances (DAB, DBC, DAC)

    References:

      [1]    :   Jonathan et al., Mol. Phys. vol. 24 (1972) 1143
      [2]    :   Jonathan et al., Mol. Phys. vol. 43 (1981) 215

    Distances (in angstrom):

      >>> hf, ff, inf = 0.917, 1.418, 1000.
      >>> ehf = leps(hf, inf, hf + inf)
      >>> eff = leps(inf, ff, inf + ff)

    Energy in kJ/mol:

      >>> leps(1.900, 1.440, 1.900 + 1.440) - eff
      9.8528712996229899

    FIXME: original paper seems to have used different units (kcals).
    """

    # set potential parameters for H-F-F
    # 0 = HF, 1 = FF, 2 = FH

    # dissociation energies
    D = array([ 590.5, 157.3, 590.5 ])

    # beta parameters
    beta = array([ 2.219, 2.920, 2.219 ])

    # equilibrium positions
    REQ = array([ 0.917, 1.418, 0.917 ])

    # Sato parameter
    Delta = array([ 0.0, -0.35, 0.0 ])

#   # set potential parameters for F-D-Br
#   # 0 = DF, 1 = DBr, 2 = FBr

#   # dissociation energies
#   D = array([ 590.7, 378.2, 249.1 ])

#   # beta parameters
#   beta = array([ 2.203, 1.797, 2.294 ])

#   # equilibrium positions
#   REQ = array([ 0.917, 1.414, 1.759 ])

#   # Sato parameter
#   Delta = array([ 0.17, 0.05, 0.05 ])

    # calculate integrals
    Q = empty(3)
    J = empty(3)
    Q[0], J[0] = integrals( D[0], beta[0], Delta[0], ab - REQ[0] )
    Q[1], J[1] = integrals( D[1], beta[1], Delta[1], bc - REQ[1] )
    Q[2], J[2] = integrals( D[2], beta[2], Delta[2], ac - REQ[2] )

    # put it all together
    S = 1.0 + Delta
    res = Q[0] / S[0] + Q[1] / S[1] + Q[2] / S[2] \
         - sqrt( J[0]**2 / S[0]**2 + J[1]**2 / S[1]**2 \
                 + J[2]**2 / S[2]**2 \
                 - J[0] * J[1] / S[0] / S[1] \
                 - J[0] * J[2] / S[0] / S[2] \
                 - J[1] * J[2] / S[1] / S[2] )
    return res

def integrals(D, beta, Delta, R):
    """
    integrals

      IN      :     D            : energy of dissociation
                    beta         : beta parameter
                    Delta        : Sato parameter
                    R            : position (rel. to equilibrium)

      OUT     :     Q            : Coulomb integral
                    J            : Exchange integral

    Purpose:
      Calculate coulomb- and exchange integrals.
      The dissociation energy of the tripplet state is set 
      to: D^3 = 0.5 * D^1.
    """

    # morse potentials
    A = exp( -beta * R )

    # binding
    E1 = D * ( A*A - 2.0 * A )

    # anti-binding
    E3 = 0.5 * D * ( A*A + 2.0 * A )

    # integrals
    Q = 0.5 * ( E1 * (1.0 + Delta) + E3 * (1.0 - Delta) )
    J = 0.5 * ( E1 * (1.0 + Delta) - E3 * (1.0 - Delta) )

    return Q, J


def test_leps(A1=0.2, A2=3.0, B1=0.2, B2=3.0, N1=50, N2=50):
    "Test the LEPS potential."

    for j in range(1, N1+1):

        A = (j - 1) * (A2 - A1) / (N1 - 1) + A1

        for k in range(1, N2+1):

            B = (k - 1) * (B2 - B1) / (N2 - 1) + B1

            U = leps( A, B, A + B )

            # FIXME: why?
            if U > 100.0: U = 100.0

            print "%15.5f%15.5f%15.5f" % (B, A, U)

        print

def show_chain(p=None, style="ro-", save=None, clear=False):
    from pylab import contour, plot, xlim, ylim, show, savefig, clf #, imshow, hold
    from numpy import linspace, empty, transpose, asarray

    # intervals:
    x_range = (1.5, 2.5)
    y_range = (1.3, 1.6)

    # create grid:
    n = 100
    xs = linspace(x_range[0], x_range[1], n)
    ys = linspace(y_range[0], y_range[1], n)

    zs = empty((n, n))
    for i in range(n):
        for j in range(n):
            zs[i, j] = pot(xs[i], ys[j])

    # when displayed, x and y are transposed:
    zs = transpose(zs)

    # dont know what it does:
    # hold(True)
    if clear: clf()

#   # Plotting color map:
#   imshow(zs, origin='lower', extent=[-1, 1, -1, 2])

    # Plotting contour lines:
    contour(zs, 100, origin='lower', extent=(x_range + y_range))

    # ideal path:
#   plot(_nodes[:, 0], _nodes[:, 1], "k--")

    # three minima, and two TSs:
#   points = array(CHAIN_OF_STATES)

    # overlay positions of minima and stationary points
    # onto coutour plot:
#   plot(points[:, 0], points[:, 1], "ko")

    # overlay a path onto coutour plot:
    if p is not None:
        p = asarray(p)
        plot(p[:, 0], p[:, 1], style)

    ylim(*y_range)
    xlim(*x_range)

    if save is None:
        show()
    else:
        savefig(save)

# python leps.py [-v]:
if __name__ == "__main__":
    import doctest
    doctest.testmod()
    show_chain()

# Default options for vim:sw=4:expandtab:smarttab:autoindent:syntax
