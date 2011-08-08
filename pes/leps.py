#
# Credit: http://userpage.chemie.fu-berlin.de/~naundorf/MD/leps.f90
#

from numpy import array, empty, sqrt, exp

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
    """

    # set potential parameters for F-D-Br
    # 0 = DF, 1 = DBr, 2 = FBr

    # dissociation energies
    D = array([ 590.7, 378.2, 249.1 ])

    # beta parameters
    beta = array([ 2.203, 1.797, 2.294 ])

    # equilibrium positions
    REQ = array([ 0.917, 1.414, 1.759 ])

    # Sato parameter
    Delta = array([ 0.17, 0.05, 0.05 ])

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

test_leps()
