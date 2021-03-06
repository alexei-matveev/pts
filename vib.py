#!/usr/bin/env python
from __future__ import print_function
__doc__ = \
"""
Module for calculating the frequencies  of a system.  The main routine
is

    vibmodes(atoms)

    atoms: atomic system to look at,  the position of it has to be set
    (it will be the place where the frequencies are calculated)

    delta and direction  are values of the derivatef  function used to
    build  up  the hessian  matrix  direction  =  'central' is  highly
    recommended,  delta   gives  the  size  of  the   steps  taken  to
    approximate the curvature

    pmap says how the  different calculations needed for the numerical
    approximating the hessian are calculated,  if it is a parallel map
    function this  actions will be calculated in  parallel, as default
    it is  set to a function  of the paramap module,  which runs every
    calculation on its own process

The  computation  proceeds   by  calculating  numerically  the  second
derivative of the energy (the hessian, by num. diff. of the forces).

Parallelization is achieved by using one of the par-map functions from
paramap.py for the running of all the jobs.

First test the derivates of a function

       >>> def g(x):
       ...     return [ 2 * x[0] * x[1] * x[2]  , x[1]**2 , x[2] ]

FIXME: for some reason the default pool-based pmap fails the doctests,
use a different implementation here:

       >>> from paramap import pmap
       >>> hessian = derivatef(g, [1.0, 2.0, 1.0], pmap=pmap, direction='forward')
       >>> print (hessian)
       [[ 4.    0.    0.  ]
        [ 2.    4.01  0.  ]
        [ 4.    0.    1.  ]]

       >>> hessian = derivatef(g, [1.0, 2.0, 1.0], pmap=pmap)
       >>> print (hessian)
       [[ 4.  0.  0.]
        [ 2.  4.  0.]
        [ 4.  0.  1.]]

       >>> hessian = derivatef(g, [1.0, 2.0, 1.0], pmap=pmap, direction='backward')
       >>> print (hessian)
       [[ 4.   -0.   -0.  ]
        [ 2.    3.99 -0.  ]
        [ 4.   -0.    1.  ]]

Ar4  Cluster   as  first  simple  atomic/molecule   test  system  with
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

    >>> freqs, modes = vibmodes(ar4, workhere=True)

Frequencies in cm-1, note that some are truly imaginary:

    >>> from numpy import round, array

Do not quarrel about line breaks, just compare the results

    >>> r_freqs = array([ 1248.65 +0.j ,    882.95 +0.j ,    882.95 +0.j ,    882.95 +0.j,
    ...   623.92 +0.j,     623.92 +0.j,       0.00 +0.j ,      0.00 +0.j,
    ...     0.00 +0.j,       0.00+17.58j,     0.00+17.58j,   0.00+17.58j])
    >>> max(abs(cm(freqs) - r_freqs)) < 1e-2
    True

    >>> from ase.constraints import FixAtoms

    >>> c = FixAtoms([1, 2])
    >>> ar4.set_constraint(c)

    >>> freqs, modes = vibmodes(ar4, workhere=True)

    >>> r_freqs = array([ 1041.10 +0.j,     764.64 +0.j ,    529.25 +0.j ,    441.39 +0.j,
    ...   441.00 +0.j,       0.00+14.35j])
    >>> max(abs(cm(freqs) - r_freqs)) < 1e-2
    True

Second test System: N-N with EMT calculator

    >>> from ase.calculators.emt import EMT

    >>> n2 = Atoms('N2', [(0, 0, 0), (0, 0, 1.1)])

    >>> n2.set_calculator( EMT())
    >>> freqs, modes = vibmodes(n2, workhere=True)

    >>> r_freqs = array([  9.93941862e+02 +0.00000000e+00j,
    ...      3.38851377e+02 +0.00000000e+00j,
    ...      3.38851377e+02 +0.00000000e+00j,
    ...      7.50802183e-06 +0.00000000e+00j,
    ...      0.00000000e+00 +0.00000000e+00j,
    ...      0.00000000e+00 +1.88069678e-06j])

    >>> max(abs(cm(freqs) - r_freqs)) < 1e-2
    True

    >>> n2.set_positions([[  0.0, 0.0, 0.000],
    ...                   [  0.0, 0.0, 1.130]])

    >>> freqs, modes = vibmodes(n2, workhere=True)
    >>> r_freqs = array([  9.28113298e+02+0.j,   3.69163533e+02+0.j,   3.69163533e+02+0.j,
    ...       2.74729403e-06+0.j,   0.00000000e+00+0.j,   0.00000000e+00+0.j])

    >>> max(abs(cm(freqs) - r_freqs)) < 1e-2
    True

For pretty-printing the frequencies use:

    >>> output(freqs)
    ====================================================
     Number  imag.   Energy in eV      Energy in cm^-1
    ----------------------------------------------------
      1       no          0.1151           928.11
      2       no          0.0458           369.16
      3       no          0.0458           369.16
      4       no          0.0000             0.00
      5       no          0.0000             0.00
      6       no          0.0000             0.00
    ----------------------------------------------------
"""
from numpy import asarray, dot, zeros, abs, eye, diag, sqrt
from numpy import argsort, savetxt, empty
from scipy.linalg import eigh
from numpy import repeat
import ase.atoms
import ase.units as units
from paramap import pmap3
import sys
from pts.func import compose
from pts.qfunc import QFunc
from pts.qfunc import fwrapper, constraints2mask, pwrapper
from pts.cfunc import Masked, Cartesian
from copy import deepcopy

VERBOSE = False

def harmonic (q, U, T):
    """
    Does harmonic analysis of U(q).  Here U(q) is a differentiable PES
    Func.  T  is  a  kinetic  energy  form  represented  as  a  Metric
    object. Returns square frequencies w^2 and the corresponding modes
    as row vectors.

        >>> from numpy import pi, abs, sign
        >>> from pts.pes.ab2 import AB2
        >>> from pts.zmat import ZMat, MassWeighted
        >>> from pts.func import compose
        >>> from pts.metric import Metric

    UO22+ PES as a function of cartesian coordinates. Soft bending:

        >>> U = AB2 ((1.76, 64.50), (pi, 2.05))
        >>> s = [1.76, 1.76, pi]

    Hard bending (commented):

        XXX U = AB2 ((1.80, 43.364), (pi, 13.009))
        XXX s = [1.80, 1.80, pi]

    To get translations and rotations right, use cartesian geometry:

        >>> z = ZMat ([(None, None, None), (1, None, None), (1, 2, None)], base=1)
        >>> q = z(s)

    T is a quadratic form (mass matrix). The masses are those of UO2:

        >>> T = Metric (MassWeighted ([238.029, 15.9994, 15.9994]))

        >>> w2, modes = harmonic (q, U, T)

    Report  imaginary  frequencies  as  negative numbers,  rounded  to
    integers in cm^-1:

        >>> w = abs (cm (sqrt (w2 + 0j))) * sign (w2)
        >>> w.astype (int)
        array([1115, 1047,  159,  159,    0,    0,    0,    0,    0])

    The other, hard bending PES, would give these at the corresponding
    minimum:

        array([914, 858, 393, 393,   0,   0,   0,   0,   0])

    Note the softer stretching modes and hard bending modes.
    """
    from pts.func import NumDiff
    from numpy import size, shape

    # FIXME:  use module  functionality here  for more  expensive PESs
    # when numerically computing derivatives:
    G = NumDiff (U.fprime)
    H = G.fprime (q)

    # Mass matrix at this location, tensor shaped:
    M = T.g (q)

    # Hessian and mass matrix as square n x n matrices:
    n = size (q)
    H.shape = (n, n)
    M.shape = (n, n)

    eigvalues, eigvectors = geigs (H, M)

    # Eigensolver returns  modes as column  vectors, eigvectors[:, i],
    # we will return modes as  row vectors (for ease of iteration) and
    # in the same shape as the input coordinate array:
    modes = eigvectors.T
    modes.shape = (n,) + shape (q)

    return eigvalues, modes


def derivatef( g0, x0, delta = 0.01, pmap = pmap3, direction = 'central' ):
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
            xinter = deepcopy(x0)
            xinter[i] += delta
            xs.append(xinter)
            xinter = deepcopy(x0)
            xinter[i] -= delta
            xs.append(xinter)
    else:
        # first for the middle geometry the value is
        # needed
        xs.append(x0[:])
        # for the rest only one per geometry
        for  i in range(geolen):
            xinter = deepcopy(x0)
            xinter[i] += delta
            xs.append(xinter)

    # calculation of the functionvalues for all the geometries
    # at the same time
    g1 = pmap(g0, xs)
    g1 = asarray(g1)
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
        deriv = zeros([geolen, derlen])
        for i in range(0, geolen):
        # alternate the values for plus and minus are stored
        # if the g elements are arrays they have to be converged
            gplus = g1[2*i].flatten()
            gminus = g1[2*i+1].flatten()
            deriv[i,:] = (gplus - gminus) / ( 2 * delta)
    else:
        geolen = len(g1)-1
        deriv = zeros([geolen, derlen])
        gmiddle = g1[0]
        gmiddle = gmiddle.flatten()
        for i, gval  in enumerate(g1[1:]):
            gval = gval.flatten()
            deriv[i,:] = (gval - gmiddle) / delta
    return deriv

def mass_matrix (masses, mask=None):
    """
    Returns mass matrix  form the vector of N  atomic masses. The mask
    which should be  a vector of 3N booleans, if not  None, is used to
    reduce the dimension of the mass matrix.
    """
    mm = eye (len (masses) * 3) * repeat (masses, 3)
    if mask is not None:
        mm = reduce_mass_matrix (mm, mask)
    return mm

def reduce_mass_matrix (mass, mask):
    """
    Returns mass  matrix, containing only the elements  which are True
    in  mask  *  mask,  therefore  giving only  back  the  mass-matrix
    relevant for the active elements
    """

    imax, jmax = mass.shape
    assert imax == jmax # both dimensions share the mask[:]

    # list of indices of variable coordinates as given by mask:
    vars = [ i for i, var in enumerate(mask) if var ]

    # number of variables
    nvars = len(vars)

    # output smaller matrix:
    mass1 = empty((nvars, nvars))

    for i, ii in enumerate(vars):
        for j, jj in enumerate(vars):
            mass1[i, j] = mass[ii, jj]

    return mass1

def vibmodes(atoms, startdir=None, mask=None, workhere=False, save=None, give_output = 0, pmap = pmap3, **kwargs):
    """
    Wrapper around vibmode, which used the atoms objects
    The hessian is calculated by derivatef
    qfunc.fwrapper is used as a wrapper to calulate the gradients
    """
    from pts.memoize import Memoize, DirStore
    coord = atoms.get_positions()

    if mask == None:
        mask = constraints2mask(atoms)

    if mask == None: # (if still None)
        fun = Cartesian()
    else:
        fun = Masked(Cartesian(), mask, coord.flatten())

    xcenter = fun.pinv(coord)

    myfunc = QFunc(atoms, atoms.get_calculator())

    myfunc = Memoize(myfunc, DirStore("cache.d"))

    myfunc = compose( myfunc, fun)

    pmapc = pwrapper(pmap)
    func = fwrapper(myfunc, startdir = startdir, mask = mask, workhere = workhere)

    # the derivatives are needed
    hessian = derivatef( func, xcenter, pmap = pmapc, **kwargs)

    # save is assumend to be a filename, so far only the hessian is saved:
    if save is not None:
        savetxt(save, hessian)

    mass = mass_matrix (atoms.get_masses(), mask)

    freqs, modes = vibmod(mass, hessian)

    # the output will be printed on demand, with modes if wanted
    # the output for the freqs can easily be recreated later on, but
    # for the mode vectors the mass (not included in the direct output_
    # is needed
    if VERBOSE or give_output == 2:
        output(freqs, modes, mass, mask)
    elif give_output == 1:
        output(freqs)


    return freqs, modes

def vibmod (mass, hessian):
    """
    Calculates  the  vibration   frequencies  and  modes  in  harmonic
    approximation. The mode vectors as  return from this sub are "row"
    vectors: freqs[i] corresponds to modes[i].
    """
    mass = asarray(mass)

    # make sure that the hessian is hermitian:
    hessian = 0.5 * (hessian + hessian.T)

    # and also the massvec
    mass = 0.5 * (mass + mass.T)

    # solve the generalized eigenvalue problem
    #
    #   w, V = geigs(H, M)
    #
    #   A * V[:, i] = w[i] * M * V[:, i]
    #
    # For symmetric H and symmetric M > 0 the eigenvalues are real and
    # eigenvectors can be (and are) chosen orthogonal:
    #
    eigvalues, eigvectors = geigs (hessian, mass)

    #
    # Negative  eigenvalues correspond  to imaginary  frequencies, for
    # square root to work lift the real numbers to complex domain:
    #
    freqs = sqrt (eigvalues + 0j)
    modes = eigvectors.T

    return freqs, modes

def eV(w):
    "Convert frequencies to eV"

    # E = hbar * omega [eV] = hvar * [1/s]
    # omega = sqrt(H / m),
    # [H] = [kJ/Ang^2], [m] = [amu], [omega] = [1/s] = [ J/m^2 /kg]
    ev = units._hbar * 1e10 / units.Ang * sqrt( units.kJ /( units._amu * 1000 ) )

    return w * ev

def cm(w):
    "Convert frequencies to cm^-1"

    cm1 = units._e / units._c / units._hplanck * 0.01

    return eV(w) * cm1

def output(freqs, eigvectors=None, mass=None, mask=None):
    # scale eigenvalues to different units:
    modeenerg = eV(freqs)
    modeincm = cm(freqs)
    print ("====================================================")
    print (" Number  imag.   Energy in eV      Energy in cm^-1")
    print ("----------------------------------------------------")
    for i, mode_e  in enumerate(modeenerg):
          if mode_e.imag != 0 and abs(mode_e.imag) > abs(mode_e.real):
              print ("%3d       yes     %10.4f       %10.2f" % (i+1,  mode_e.imag, modeincm[i].imag))
          else:
              print ("%3d       no      %10.4f       %10.2f" % (i+1,  mode_e.real, modeincm[i].real))
    print ("----------------------------------------------------")

    if eigvectors is not None:
         assert mass is not None

         write = sys.stdout.write

         # we don't know if they are cartesian
         print ("The corresponding eigenvectors  are:")
         print ("Number   Vector")
         for i, ev  in enumerate(eigvectors):
              write("%3d :    " % (i+1)  )
              for j in range(int(len(ev)/3)):
                   for k in [0,1,2]:
                       write("  %10.7f" % (ev[j * 3 + k]))
                   write("\n         " )
              for k in range(int(len(ev)/3)*3,len(ev)):
                   write("  %10.7f" % (ev[k]))
              write("\n")
         print ("----------------------------------------------------")
         print ("kinetic energy distribution")
         print ("Mode        %Ekin distribution on the atoms")
         for i, ev  in enumerate(eigvectors):
             write("%3d :    " % (i+1)  )
             ek =  ev *  dot(mass, ev.T)
             ek = asarray(ek)
             if mask != None:
                ek2 = zeros(len(mask))
                j = 0
                for i, flag in enumerate(mask):
                    if flag:
                       ek2[i] = ek[j]
                       j += 1
                assert(j == len(ek))
                ek = ek2
             ek.shape = (-1, 3)
             for ek_1 in ek:
                 eks = sum(ek_1)
                 write("  %7.2f" % (eks * 100))
             write("\n" )


def check_eigensolver(a, V1, A, B):
    if VERBOSE:
        from numpy import transpose
        print ("Check the results for the eigensolver:")
        #print dot(V, dot(A, V.T)) - a * eye(len(a))
        print ("V^TAV -a, maximum value:", (abs(dot(V1.T, dot(A, V1)) - a * eye(len(a)))).max())
        print ("V^TBV -1, maximum value:", (abs(dot(dot(V1.T,B),V1) - eye(len(a)))).max())

        print ("a=", a)
        print ("V[:, 0]=", V1[:, 0])
        print ("V[:, -1]=", V1[:, -1])

        print ((abs(dot(A, V1) - dot(dot(B, V1), diag(a)))).max())
        x = dot(dot(transpose(V1), B), V1)
        print ((abs(x - diag(diag(x)))).max())


    assert((abs(dot(V1.T, dot(A, V1)) - a * eye(len(a)))).max() < 1e-8)
    assert((abs(dot(dot(V1.T,B),V1) - eye(len(a)))).max() < 1e-8)

def geigs(A, B):
    """Wrapper around eigensolver from scipy,
       which gives the output to our special need"""

    fun = lambda x: 1.0 / sqrt(x)
    mhalf = funm(B, fun)

    mam = dot(mhalf, dot(A, mhalf.T))

    # FIXME: why doing it? A is assumed to be symmetric here.
    # mam = 0.5 * (mam + mam.T)

    a, V = eigh(mam)

    # changing V back to the original problem
    # (A*V = lamda * B * V)
    V = dot(mhalf, V)

    # FIXME: for debug only:
    check_eigensolver(a, V, A, B)
    # In  this case  V should  be normed  AND orthogonal  so  there is
    # nothing else to do here a  should be also sorted, but as we want
    # the  reversed  order,  and  thus  have to  change  there  anyhow
    # something, we can as well sort it again

    # Bring the results in descending order:
    sorter = list(argsort(a, kind='mergesort'))
    sorter.reverse()

    return a[sorter], V[:, sorter]

def funm(M, fun):

    # eigenvalues m[i] and the corresponding eigenvectors V[:, i] of M:
    m, V = eigh(M)

    # fm = fun(m) for diagonal m:
    fm = [fun(ev) for ev in m]

    return dot(V, dot(diag(fm), V.T))

def main(argv):
    """
    Usage:

        frequencies --calculator <calculator file> <geometry file>

    accepts also the options:

    --num-procs <n>

        number of processors available

    --alsovec <True|False>

        the eigenvector is also given back as output

    --mask string

        string  should contain the  mask, which  Cartesian coordinates
        should be fixed

    The numerically build second derivative of the energy (the Hessian
    approximation) is build with the following settings, instead the defaults:

    --direction <forward|central|backwards>

      In which direction the step is taken for the approximation. The default
      value is "central".

    --delta d

      How far away the step is done from the current geometry. In case of central
      differences, there is one step of +delta and one of -delta for each
      of the coordinates.

    --save-hessian filename

      This will save the approximated Hessian to a file named filename.
    """
    from pts.ui.cmdline import get_options, get_calculator, get_mask
    from pts.sched import Strategy
    from pts.paramap import PMap3

    if argv[0] == "--help":
         print (main.__doc__)
         sys.exit()

    # vibration module  options (not all that the vibmodes function has):
    opts, args = get_options(argv, long_options=["calculator=","mask=", "num-procs=", "alsovec=",
                                                 "delta=", "direction=", "save-hessian="])

    # and one geometry:
    if len(args) != 1:
        print ("Error: Need exactly one geometry file to process!", file=sys.stderr)
        print ("Error: found instead:", args, file=sys.stderr)
        print (main.__doc__, file=sys.stderr)
        sys.exit(1)

    atoms = ase.io.read(args[0])

    # default values for the options
    calculator = None
    go = 1
    # for parallel calculations
    num_procs = None
    mask = None
    save_h = None
    direction = "central"
    delta = 0.01

    for opt, value in opts:
         if opt == "--calculator":
             calculator = get_calculator(value)
         elif opt =="--alsovec":
             # set output level (only eigvalues or also eigvectors)
             if str(value) in ["True", "true","t"]:
                 go = 2
         elif opt == "--num-procs":
             num_procs = int(value)
         elif opt == "--direction":
             direction = value
         elif opt == "--delta":
             delta = float(value)
         elif opt == "--save-hessian":
             save_h = value
         elif opt == "--mask":
             mask = get_mask(value)

    # this option has to be given!!
    assert calculator != None

    atoms.set_calculator(calculator)
    # calculate the vibration modes, gives also already the output (to
    # stdout)
    if num_procs == None:
        # default values for paramap (pool_map does not need topology)
        vibmodes(atoms, mask = mask, give_output = go, direction = direction, delta = delta,\
                    startdir="start_files", save = save_h)
    else:
        # use PMap with HCM strategy: num_procs should hold the number
        # of  procces available  for  the complete  job  they will  be
        # distributed as good as possible
        sched = Strategy(topology = [num_procs], pmin = 1, pmax = num_procs)
        # new paramap  version, using this Strategy  (the pmap topolgy
        # specifications are put in environment variables, which could
        # be used by the qm-program
        pmap = PMap3(strat = sched)
        vibmodes(atoms, pmap = pmap, mask = mask, give_output = go, direction = direction, delta = delta,\
                    startdir="start_files", save = save_h)

# python vib.py [-v]:
if __name__ == "__main__":
    import doctest
    from sys import argv as sargs
    # make a message (the complete docstring available)
    if len(sargs) > 1:
        print (__doc__)
        sys.exit()
    doctest.testmod()

# Default options for vim:sw=4:expandtab:smarttab:autoindent:syntax
