#!/usr/bin/python
"""
  Module for calculating the frequencies of a system. The main routine is
  vibmodes(atoms, func, delta = 0.01, p_map = ps_map  , direction = 'central', alsovec = False)

  atoms: atomic system to look at, the position of it has to be set (it will be
  the place where the frequencies are calculated)

  func is the function which gives the gradients of the atom object at an given
  position x

  delta and direction are values of the derivatef function used to build up
  the hessian matrix direction = 'central' is highly recommended, delta gives
  the size of the steps taken to approximate the curvature

  p_map says how the different calculations needed for the numerical approximating
  the hessian are calculated, if it is a parallel map function this actions will be
  calculated in parallel, as default it is set to a function of the paramap module, which
  runs every calculation on its own process

  There is also inlcuded  the calculating of a
  numerical derivative (a hessian if the target function is the gradient)
  of a function, by using one of the map functions of paramap.py
  for the running of all the needed jobs.

  Test of the Module:

  First test the derivates of a function

       >>> def g(x):
       ...     return [ 2 * x[0] * x[1] * x[2]  , x[1]**2 , x[2] ]


       >>> hessian = derivatef(g, [1.0, 2.0, 1.0], p_map = ps_map,direction = 'forward' )
       >>> print hessian
       [[ 4.    0.    0.  ]
        [ 2.    4.01  0.  ]
        [ 4.    0.    1.  ]]

       >>> hessian = derivatef(g, [1.0, 2.0, 1.0], p_map = ps_map )
       >>> print hessian
       [[ 4.  0.  0.]
        [ 2.  4.  0.]
        [ 4.  0.  1.]]

       >>> hessian = derivatef(g, [1.0, 2.0, 1.0],p_map = ps_map,direction = 'backward' )
       >>> print hessian
       [[ 4.   -0.   -0.  ]
        [ 2.    3.99 -0.  ]
        [ 4.   -0.    1.  ]]


Ar4 Cluster as first simple atomic/molecule test system with
  LennardJones-potential.

    >>> from ase import Atoms

  One equilibrium:

    >>> w=0.39685026
    >>> A = ([[ w,  w,  w],
    ...       [-w, -w,  w],
    ...       [ w, -w, -w],
    ...       [-w,  w, -w]])

    >>> ar4 = Atoms("Ar4", A)

  Define LJ-PES:

    >>> from ase.calculators.lj import LennardJones

    >>> ar4.set_calculator(LennardJones())

  Calculate the vibration modes

    >>> vibmodes(ar4, workhere = True, storehessian = False)
    ====================================================
     Number  imag.   Energy in eV      Energy in cm^-1
    ----------------------------------------------------
      1       no          0.1548          1248.65
      2       no          0.1095           882.95
      3       no          0.1095           882.95
      4       no          0.1095           882.95
      5       no          0.0774           623.92
      6       no          0.0774           623.92
      7       yes         0.0000             0.00
      8       yes         0.0000             0.00
      9       yes         0.0000             0.00
     10       yes         0.0022            17.58
     11       yes         0.0022            17.58
     12       yes         0.0022            17.58
    ----------------------------------------------------
    >>> from ase.constraints import FixAtoms

    >>> c = FixAtoms([1, 2])
    >>> ar4.set_constraint(c)

    >>> vibmodes(ar4, workhere = True, storehessian = False)
    FWRAPPER: The following mask has been obtained from the constraints of the atoms
    [True, True, True, False, False, False, False, False, False, True, True, True]
    ====================================================
     Number  imag.   Energy in eV      Energy in cm^-1
    ----------------------------------------------------
      1       no          0.1291          1041.10
      2       no          0.0948           764.64
      3       no          0.0656           529.25
      4       no          0.0547           441.39
      5       no          0.0547           441.00
      6       yes         0.0018            14.35
    ----------------------------------------------------

  second test System: N-N with EMT calculator

    >>> from ase.calculators.emt import EMT

    >>> n2 = Atoms('N2', [(0, 0, 0), (0, 0, 1.1)])

    >>> n2.set_calculator( EMT())
    >>> vibmodes(n2, workhere = True, storehessian = False)
    ====================================================
     Number  imag.   Energy in eV      Energy in cm^-1
    ----------------------------------------------------
      1       no          0.2540          2048.94
      2       no          0.0000             0.00
      3       no          0.0000             0.00
      4       yes         0.0000             0.00
      5       yes         0.0399           321.63
      6       yes         0.0399           321.63
    ----------------------------------------------------


    >>> n2.set_positions([[  0.0, 0.0, 0.000],
    ...                   [  0.0, 0.0, 1.130]])

    >>> vibmodes(n2, workhere = True, storehessian = False)
    ====================================================
     Number  imag.   Energy in eV      Energy in cm^-1
    ----------------------------------------------------
      1       no          0.2325          1874.96
      2       no          0.0047            37.51
      3       no          0.0047            37.51
      4       no          0.0000             0.00
      5       no          0.0000             0.00
      6       no          0.0000             0.00
    ----------------------------------------------------

"""
import numpy as np
from scipy.linalg import eigh as eigensolver2
from math import sqrt
import ase.atoms
import ase.units as units
from paramap import pa_map, ps_map, td_map, pmap, pool_map
from sys import stdout
from qfunc import fwrapper

MOREOUT = False

def derivatef( g0, x0, delta = 0.01, p_map = pool_map  , direction = 'central' ):
    '''
    Derivates another function numerically,

    g is the function or a vector of them
    x0 is the geometry (as a list) on which the minimum
    should be found
    delta is the step size
    direction: central uses formula (f(r+d) - f(r-d)) / 2d
               forward            ( f(r+d) - f(r)) / d
               backward is forward with -d

    gives back the derivatives as a matrix
    nabla gi/ nabla x0j

    The gradient/derivative given back can also be an array
    '''
    assert direction in ['central', 'forward', 'backward']

    if direction =='backward':
    # change sign of delta if other direction is wanted
         delta = -delta

    xs = []
    try:
        geolen = len(x0)
    except TypeError:
        geolen = 1
        x0 = [x0]

    # building up the list of wanted geometries
    # consider the different directions
    if direction == 'central':
        # two inputs per geometry values
        # one in each direction
        for i in range(geolen):
            xinter = []
            xinter = x0[:]
            xinter[i] += delta
            xs.append(xinter)
            xinter = []
            xinter = x0[:]
            xinter[i] -= delta
            xs.append(xinter)
    else:
        # first for the middle geometry the value is
        # needed
        xs.append(x0[:])
        # for the rest only one per geometry
        for  i in range(geolen):
            xinter = []
            xinter = x0[:]
            xinter[i] += delta
            xs.append(xinter)

    # calculation of the functionvalues for all the geometries
    # at the same time
    g1 = p_map(g0, xs)
    g1 = np.asarray(g1)
    # now it is possible to find out, how big g1 is
    # g1 may be an array (then we want the total length
    # not only in one direction
    derlen = len(g1[0].flatten())

    # deriv is the matrix with the derivatives
    # for g = deriv * geo
    # again the direction makes a difference
    # compare the formulas given above
    if direction == 'central':
        geolen = len(g1)/2
        deriv = np.zeros([geolen, derlen])
        for i in range(0, geolen):
        # alternate the values for plus and minus are stored
        # if the g elements are arrays they have to be converged
            gplus = g1[2*i].flatten()
            gminus = g1[2*i+1].flatten()
            deriv[i,:] = (gplus - gminus) / ( 2 * delta)
    else:
        geolen = len(g1)-1
        deriv = np.zeros([geolen, derlen])
        gmiddle = g1[0]
        gmiddle = gmiddle.flatten()
        for i, gval  in enumerate(g1[1:]):
            gval = gval.flatten()
            deriv[i,:] = (gval - gmiddle) / delta
    return deriv

def vibmodes(atoms, startdir = None, mask = None, workhere = False, storehessian = True, alsovec = False, **kwargs ):
     """
     Wrapper around vibmode, which used the atoms objects
     The hessian is calculated by derivatef
     qfunc.fwrapper is used as a wrapper to calulate the gradients
     """
     func = fwrapper(atoms, startdir = startdir, mask = mask, workhere = workhere)
     xcenter = func.getpositionsfromatoms()
     mass = func.getmassfromatoms()
     # the derivatives are needed
     hessian = derivatef( func.perform, xcenter,**kwargs )
     if storehessian:
         store_hessian(hessian)
     freqs, modes = vibmod( mass, hessian)
     output(freqs, modes, mass, alsovec)


def store_hessian(hessian):
     print "Vibration module: printing numerical calculated hessian:"
     print hessian
     wr = open("Hessian.dat","w").write
     dim1 = len(hessian)
     for i in range(dim1):
         for j in range(dim1):
             wr("%23.12f " % hessian[i,j])
         wr("\n")

def vibmod(mass, hessian):
     """
     calculates the vibration modes in harmonic approximation

     atoms is the atom system (ase.atoms) with positions set to the geometry, on which the
     frequencies are wanted

     func should do the gradient call on a given geometry

     if p_map is choosen as a parallel variant, all gradient calculations will be performed
     in parallel
     direction = 'central' should be the most accurate one but 'forward'/'backward' needs 
     fewer calculations all together (but the results may be really much worse, compare the results
     from the derivatef tests in the doctests.
     delta is the defaultvalue for delta in the derivatef function

     alsovec says that not only the frequencies of the mode but also the eigenvectors are 
     wanted
     """
     mass = np.asarray(mass)

     # make sure that the hessian is hermitian:
     hessian = 0.5 * (hessian + hessian.T)

     # and also the massvec
     mass = 0.5 * (mass + mass.T)

     # solves the eigenvalue problem w, vr = eig(a,b)
     #   a * vr[:,i] = w[i] * b * vr[:,i]
     eigvalues, eigvectors = geigs2(hessian, mass)
     freqs = eigvalues.astype(complex)**0.5
     modes = eigvectors.T

     return freqs, modes

def output(freqs, eigvectors, mass, alsovec = False):
     # scale eigenvalues in different units:
     # E = hbar * omega [eV] = hvar * [1/s]
     # omega = sqrt(H /m), [H] = [kJ/Ang^2] , [m] = [amu], [omega] = [1/s] = [ J/m^2 /kg]
     scalfact = units._hbar * 1e10 / units.Ang * sqrt( units.kJ /( units._amu * 1000 ) )
     modeenerg =  scalfact * freqs
     modeincm  = modeenerg * units._e / units._c / units._hplanck * 0.01
     print "===================================================="
     print " Number  imag.   Energy in eV      Energy in cm^-1"
     print "----------------------------------------------------"
     for i, mode_e  in enumerate(modeenerg):
           if mode_e.imag != 0 and abs(mode_e.imag) > abs(mode_e.real):
               print "%3d       yes     %10.4f       %10.2f" % (i+1,  mode_e.imag, modeincm[i].imag)
           else:
               print "%3d       no      %10.4f       %10.2f" % (i+1,  mode_e.real, modeincm[i].real)
     print "----------------------------------------------------"

     if (alsovec):
          write = stdout.write

          # we don't know if they are cartesian
          print "The corresponding eigenvectors  are:"
          print "Number   Vector"
          for i, ev  in enumerate(eigvectors):
               write("%3d :    " % (i+1)  )
               for j in range(int(len(ev)/3)):
                    for k in [0,1,2]:
                        write("  %10.7f" % (ev[j * 3 + k]))
                    write("\n         " )
               for k in range(int(len(ev)/3)*3,len(ev)):
                    write("  %10.7f" % (ev[k]))
               write("\n")
          print "----------------------------------------------------"
          print "kinetic energy distribution"
          print "Mode        %Ekin distribution on the atoms"
          for i, ev  in enumerate(eigvectors):
              write("%3d :    " % (i+1)  )
              ek =  ev *  np.dot(mass, ev.T)
              ek = np.asarray(ek)
              ek.shape = (-1, 3)
              for ek_1 in ek:
                  eks = sum(ek_1)
                  write("  %7.2f" % (eks * 100))
              write("\n" )


def check_eigensolver(a, V1, A, B):
     if MOREOUT:
         print "Check the results for the eigensolver:"
         #print np.dot(V, np.dot(A, V.T)) - a * np.eye(len(a))
         print "V^TAV -a, maximum value:", (abs(np.dot(V1.T, np.dot(A, V1)) - a * np.eye(len(a)))).max()
         print "V^TBV -1, maximum value:", (abs(np.dot(np.dot(V1.T,B),V1) - np.eye(len(a)))).max()

         print "a=", a
         print "V[:, 0]=", V1[:, 0]
         print "V[:, -1]=", V1[:, -1]

         print (abs(np.dot(A, V1) - np.dot(np.dot(B, V1), np.diag(a)))).max()
         x = np.dot(np.dot(np.transpose(V1), B), V1)
         print (abs(x - np.diag(np.diag(x)))).max()


     assert((abs(np.dot(V1.T, np.dot(A, V1)) - a * np.eye(len(a)))).max() < 1e-8)
     assert((abs(np.dot(np.dot(V1.T,B),V1) - np.eye(len(a)))).max() < 1e-8)



def geigs2(A, B):
    """Wrapper around eigensolver from scipy,
       which gives the output to our special need"""

    fun = lambda x: 1.0 / sqrt(x)
    mhalf = matfun(B, fun)

    mam = np.dot(mhalf, np.dot(A, mhalf.T))

    mam = 0.5 * (mam + mam.T)

    a, V = eigensolver2(mam)

    # changing V back to the original problem
    # (A*V = lamda * B * V)
    V = np.dot(mhalf, V)

    check_eigensolver(a, V, A, B)
    # In this case V should be normed AND orthogonal
    # so there is nothing else to do here
    # a should be also sorted, but as we want
    # the reversed order, and thus have to change there
    # anyhow something, we can as well sort it again

    # Bring the results in descending order:
    sorter = list(np.argsort(a, kind='mergesort'))
    sorter.reverse()
    a = a[sorter]
    V1 = V[:, sorter]

    return a, V

def matfun(M, fun):
    aval, Avec = eigensolver2(M)
    anew = np.asarray([fun(av) for av in aval])
    # the vector is given in the format Av[:,i] for the aval[i]
    O = np.dot(Avec, np.dot(np.diag(anew), Avec.T))
    return O

# python vib.py [-v]:
if __name__ == "__main__":
    import doctest
    doctest.testmod()

# Default options for vim:sw=4:expandtab:smarttab:autoindent:syntax
