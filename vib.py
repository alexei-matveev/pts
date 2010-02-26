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

    >>> vibmodes(ar4, workhere = True)
    ====================================================
     Number  imag.   Energy in eV      Energy in cm^-1
    ----------------------------------------------------
      1       no       0.1548129       1248.6494855
      2       no       0.1094714        882.9459913
      3       no       0.1094714        882.9459913
      4       no       0.1094714        882.9459913
      5       no       0.0773558        623.9162798
      6       no       0.0773558        623.9162798
      7       no       0.0000000          0.0000076
      8       no       0.0000000          0.0000064
      9       yes      0.0000000          0.0000056
     10       yes      0.0021796         17.5798776
     11       yes      0.0021796         17.5798776
     12       yes      0.0021796         17.5798776
    ----------------------------------------------------

  second test System: N-N with EMT calculator

    >>> from ase.calculators.emt import EMT

    >>> n2 = Atoms('N2', [(0, 0, 0), (0, 0, 1.1)])

    >>> n2.set_calculator( EMT())
    >>> vibmodes(n2, workhere = True )
    ====================================================
     Number  imag.   Energy in eV      Energy in cm^-1
    ----------------------------------------------------
      1       no       0.2540363       2048.9398454
      2       no       0.0000000          0.0000000
      3       no       0.0000000          0.0000000
      4       no       0.0000000          0.0000000
      5       yes      0.0398776        321.6345510
      6       yes      0.0398776        321.6345510
    ----------------------------------------------------


    >>> n2.set_positions([[  0.0, 0.0, 0.000],
    ...                   [  0.0, 0.0, 1.130]])

    >>> vibmodes(n2, workhere = True)
    ====================================================
     Number  imag.   Energy in eV      Energy in cm^-1
    ----------------------------------------------------
      1       no       0.2324660       1874.9643134
      2       no       0.0046502         37.5060049
      3       no       0.0046502         37.5060049
      4       no       0.0000000          0.0000000
      5       no       0.0000000          0.0000000
      6       no       0.0000000          0.0000000
    ----------------------------------------------------

"""
import numpy as np
from scipy.linalg import eig
from math import sqrt
import ase.atoms
import ase.units as units
from aof.paramap import pa_map, ps_map, td_map, pmap, pool_map
from sys import stdout
from aof.qfunc import fwrapper

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

def vibmodes(atoms, startdir = None, mask = None, workhere = False, alsovec = False, **kwargs ):
     """
     Wrapper around vibmode, which used the atoms objects
     The hessian is calculated by derivatef
     qfunc.fwrapper is used as a wrapper to calulate the gradients
     """
     func = fwrapper(atoms, startdir = startdir, mask = mask, workhere = workhere )
     xcenter = func.getpositionsfromatoms()
     mass = func.getmassfromatoms()
     # the derivatives are needed
     hessian = derivatef( func.perform, xcenter,**kwargs )
     vibmod( mass, hessian, alsovec)

def vibmod(mass, hessian, alsovec = False):
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
     eigvalues, eigvectors = eig(hessian, mass)
     eigvalues, eigvectors = normandsort(eigvalues, eigvectors, mass)

     # scale eigenvalues in different units:
     # E = hbar * omega [eV] = hvar * [1/s]
     # omega = sqrt(H /m), [H] = [kJ/Ang^2] , [m] = [amu], [omega] = [1/s] = [ J/m^2 /kg]
     scalfact = units._hbar * 1e10 / units.Ang * sqrt( units.kJ /( units._amu * 1000 ) )
     modeenerg =  scalfact *  eigvalues.astype(complex)**0.5
     modeincm  = modeenerg * units._e / units._c / units._hplanck * 0.01
     print "===================================================="
     print " Number  imag.   Energy in eV      Energy in cm^-1"
     print "----------------------------------------------------"
     for i, mode_e  in enumerate(modeenerg):
           if mode_e.imag != 0:
               print "%3d       yes     %10.7f       %12.7f" % (i+1,  mode_e.imag, modeincm[i].imag)
           else:
               print "%3d       no      %10.7f       %12.7f" % (i+1,  mode_e.real, modeincm[i].real)
     print "----------------------------------------------------"

     if (alsovec):
          writevec = stdout.write

          # we want the eigenvector as EV[eigenvalue,:]:
          eigvectors = eigvectors.T

          # the eigenvectors are not normed, therefore:

          # we don't know if they are cartesian
          print "The corresponding eigenvectors  are:"
          print "Number   Vector"
          for i, ev  in enumerate(eigvectors):
               writevec("%3d :    " % i)
               for j in range(int(len(ev)/3)):
                    for k in [0,1,2]:
                        writevec("  %10.7f" % (ev[j * 3 + k]))
                    writevec("\n         " )
               for k in range(int(len(ev)/3)*3,len(ev)):
                    writevec("  %10.7f" % (ev[k]))
               writevec("\n")
          print "----------------------------------------------------"
          print "kinetic energy distribution"
          ekin_ev = np.diag(np.dot(eigvectors, np.dot( mass, eigvectors.T)))
          print "Mode        %Ekin"
          for i, ekin_e in enumerate(ekin_ev):
              print "%3d :     %12.10f" % (i+1, ekin_e)
          print "----------------------------------------------------"

def normandsort(eigval, eigvec, metric):
    # normingconstants for  the eigvec:
    nor = np.diag(np.dot(eigvec.T, np.dot(metric, eigvec)))
    # storage of eigenvalues and eigenvec
    eigdata = np.zeros((len(eigval), len(eigval)+1))
    # order after the eigvalues:
    sorter = np.argsort(eigval.real, kind='mergesort')
    # revert order (start with the biggest one)
    # and order the eigenvectors with the same cheme
    for i in range(len(eigval)):
        eigdata[len(eigval)-1-i,0] = eigval[sorter[i]]
        eigdata[len(eigval)-1-i,1:] =  eigvec[sorter[i]] / nor[sorter[i]]

    return eigdata[:,0], eigdata[:,1:]


# python vib.py [-v]:
if __name__ == "__main__":
    import doctest
    doctest.testmod()

