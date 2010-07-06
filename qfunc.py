#!/usr/bin/python
from __future__ import with_statement # need to be at the beginning
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
from os import path, mkdir, chdir, getcwd, system
try:
    from multiprocessing import current_process
    def current_process_name(): return current_process().name
except:
    from processing import currentProcess
    def current_process_name(): return currentProcess().getName()

from shutil import copy2 as cp
import numpy as np

class QFunc(Func):
    def __init__(self, atoms, calc=LennardJones(), moving=None):

        # we are going to repeatedly set_positions() for this instance,
        # So we make a copy to avoid effects visible outside:
        self.atoms = atoms.copy()
        self.calc = calc
        self.atoms.set_calculator(calc)

        # list of moving atoms whose coordinates are considered as variables:
        self.moving = moving

    # (f, fprime) methods inherited from abstract Func and use this by default:
    def taylor(self, x):
        "Energy and gradients"

        # aliases:
        atoms = self.atoms
        moving = self.moving

        if moving is not None:

            # it is assumed that the initial positions are meaningful:
            y = atoms.get_positions()

            assert len(moving) <= len(y)

            # update positions of moving atoms:
            y[moving] = x

            # rebind x:
            x = y

        # update positions:
        atoms.set_positions(x)

        # request energy:
        e = atoms.get_potential_energy()

        # request forces. NOTE: forces are negative of the gradients:
        g = - atoms.get_forces()

        if moving is not None:

            # rebind g:
            g = g[moving]

        # return both:
        return e, g

from paramap import pmap

def qmap(f, xs, map=pmap, format="%02d"):
    """Apply (parallel) map with chdir() isolation for each evaluation.
    Directories are created, if needed, following the format:

        format % i

    The default format leads to directory names: 00, 01, 02, ...
    """

    def _f(args):
        i, x = args

        if format is not None:
            dir = format % i
        else:
            dir ="."

        context = QContext(wd=dir)
        with context:
            fx = f(x)

        return fx

    return pmap(_f, enumerate(xs))

# a list of restartfiles that might be usefull to copy-in
# for a warm-start of a calculation, if it is complete, no
# modifications to ASE are necessary:
RESTARTFILES = ["WAVECAR", "CHG", "CHGCAR" , "saved_scfstate.dat", "*.testme"]

def constraints2mask(atoms):
    """
    Given an atomic object (ase/aof) there is
    tried to get a mask, using some of the fixing methods
    of the constraints
    """
    mask0 = None
    fix = atoms._get_constraints()
    if not fix == []:
        mask0 = [True for i in range(len(atoms.get_positions().flatten()))]
        for fix1 in fix:
                fixstr = str(fix1)
                if fixstr.startswith("FixScaled"):
                     num = fix1.a
                     mask1 = fix1.mask
                     mask1 -= True
                     mask0[num * 3: num * 3 + 3] = mask1
                elif fixstr.startswith("FixAtoms"):
                     nums = fix1.index
                     for num in nums:
                         mask1 = [False for i in range(3)]
                         mask0[num * 3: num * 3 + 3] = mask1
    return mask0

class fwrapper(object):
    """
    Wrapper around a function which changes in
    a workingdirectory special for the processes
    it works in and handles startdata for the qm-solver
    if self.start is not None
    """
    def __init__(self, atoms, startdir = None, mask = None, workhere = True):
        self.start = startdir
        self.mask = mask
        self.workhere = workhere
        self.atoms = atoms

        # the working directory before changed, thus where
        # to return lateron
        self.wopl = getcwd()
        qf = QFunc(self.atoms, calc = self.atoms.get_calculator())
        self.fun = qf.fprime

        if self.atoms == None:
             print "FWRAPPER_WARNING: Your function needs an atom object!"
             print "Make sure it's there, when calculation starts!"
             print "This way you have to set any constraints by hand!"
        if startdir is not None:
             if startdir.startswith('/'):
                  pass
             elif startdir.startswith('~'):
                  pass
             else:
                 self.start = self.wopl + '/' + startdir
        if mask == None and not self.atoms == None:
             mask0 = constraints2mask(atoms)
             if not mask0 == None:
                 print "FWRAPPER: The following mask has been obtained from the constraints of the atoms"
                 print mask0
                 self.mask = mask0


    def getmassfromatoms(self):
        """
        gives back the masses form the atoms as a vector
        filtered through the mask
        """
        mass1 = self.atoms.get_masses()
        massvec = np.eye(len(mass1) * 3) *  np.repeat(mass1, 3)
        if self.mask is not None:
            massvec = self.reducevecm(massvec, self.mask)
        return massvec

    def reducevecm(self, mass, mask):
        """
        gives back a mass-matrix, containing only the elements
        which are True in mask * mask, therefore giving
        only back the mass-matrix relevant for the active
        elements
        """

        imax, jmax = mass.shape
        assert imax == jmax # both dimensions share the mask[:]

        # list of indices of variable coordinates as given by mask:
        vars = [ i for i, var in enumerate(mask) if var ]

        # number of variables
        nvars = len(vars)

        # output smaller matrix:
        mass1 = np.empty((nvars, nvars))

        for i, ii in enumerate(vars):
            for j, jj in enumerate(vars):
                mass1[i, j] = mass[ii, jj]

        return mass1

    def getpositionsfromatoms(self):
        """
        Gives all the coordinates of the atom object
        for which a vibration is wanted
        they are given back as a list
        """
        self.xcore = self.atoms.get_positions()
        if self.mask is not None:
            xcen = self.reducevecx(self.xcore, self.mask)
        else:
            xcen = self.xcore
            xcen = (xcen.flatten()).tolist()
        return xcen

    def reducevecx(self, x, mask):
        """
        From a vector (like the coordinates ore gradients)
        the coordinates are extracted (and given back as a
        list) which are True in mask
        """
        imax, jmax = x.shape
        cnt_act_elem = mask.count(True)
        xout = []
        m_count = 0
        for i in range(imax):
             for j in range(jmax):
                 if mask[m_count] == True:
                     xout.append(x[i,j])
                 m_count += 1
        return xout

    def enlargex(self, x, mask):
        """
        The list, containing the geometry coordinates
        are extenced to the full coordinates, by using
        self.xcore values for all elementes which are
        False in the mask, the others are set by x
        gives back a vector if the shape of xcore
        """
        mar = mask
        if mar == None:
            mar = [True for i in range(len(x))]
        imax, jmax = self.xcore.shape
        xout = self.xcore
        l_count = 0
        m_count = 0
        for i in range(imax):
                 for j in range(jmax):
                     if mar[m_count] == True:
                         xout[i, j] = x[l_count]
                         l_count += 1
                     m_count += 1
        return xout

    def perform(self,x, num = None):
        """
        This is the function which should be called by the
        processes for mapping and so on
        If wanted it changes the current working directory/back
        and copies starting files, when called, the result
        is given back as a list and contain only derivatives
        for the elements which are True in self.mask
        """
        # if called instead of the fun itself via name.perform
        # the name of the working directory is just the name
        # of the current Process
        if not self.workhere:
            if num == None:
                wx = current_process_name()
            else:
                wx = "Geometrycalculation%s" % (str(num))

            if not path.exists(wx):
                mkdir(wx)
            chdir(wx)
            print "FWRAPPER: Entering working directory:", wx
        # if there is a startingdirectory named, now is the time
        # to have a look, if there is something useful inside
        if self.start is not None and path.exists(self.start):
             wh = getcwd()
             # in RESTARTFILES are the names of all the files for all
             # the quantum chemistry calculators useful for a restart
             for singfile in RESTARTFILES:
                # copy every one of them, which is available to the
                # current directory
                try:
                   filename = self.start + "/" + singfile
                   cp(filename,wh)
                   print "FWRAPPER: Copying start file",filename
                except IOError:
                   pass
        xvec = self.enlargex(x, self.mask)
        # the actual call of the original function
        res = self.fun(xvec)
        if self.mask is not None:
            res = self.reducevecx(res, self.mask)
        else:
            res = (res.flatten()).tolist()
        # if the startingdirectory does not exist yet but has a
        # resonable name, it now can be created
        if self.start is not None and not path.exists(self.start):
             try:
                 # Hopefully the next line is thread save
                 mkdir(self.start)
                 # if this process was alowed to create the starting
                 # directory it can also put its restartfiles in,
                 # the same thing than above but in the other direction
                 wh = getcwd()
                 for singfile in RESTARTFILES:
                     try:
                          filename = wh + "/" + singfile
                          cp(filename, self.start)
                          print "FWRAPPER: Storing file",filename,"in start directory"
                          print self.start, "for further use"
                     except IOError:
                          pass
             except OSError:
                 print "FWRAPPER: not make new path", wx
        # it is safer to return to the last working directory, so
        # the code does not affect too many things
        if not self.workhere:
            chdir(self.wopl)
        # the result of the function call:
        return res


class QContext(object):
    """For the option of working in a workingdirectory different to
    the one used, and the possibility to use old stored datafiles
    for restart some extra parameters are needed

    WARNING: this will create and remove a directory |dir|:

    This does not need to be an absolute path, of course:

        >>> dir = "/tmp/666777888999000111"
        >>> preparations = QContext(wd=dir)

    The commands in the block will be executed with |dir| as $CWD:

        >>> with preparations:
        ...     print getcwd()
        /tmp/666777888999000111

        >>> system("rmdir " + dir)
        0
    """
    def __init__(self, wd = None, restartdir = None):

        # working directory (may not yet exist):
        self.wd = wd

        # restart files copied from or put there:
        self.restartdir = restartdir

    def __enter__(self):
        # As a default the working directory is not changed:
        if self.wd is None: return

        # save for __exit__ to chdir back:
        self.__cwd = getcwd()

        # first the working directory is changed to
        # If it does not exist, it is
        # created, make sure that if it exists, there is
        # no grabage inside

        # print "QContext: chdir to", self.wd
        if not path.exists(self.wd):
            mkdir(self.wd)
        chdir(self.wd)

        # there may exist some files stored self.restartdir
        # which might be useful
        if self.restartdir is not None and path.exists(self.restartdir):
            # print "QContext: using data for restarting from directory", self.restartdir

            # files being in self.restartdir are copied in
            # the current working directory
            cmd = "echo cp " + self.restartdir + '/*  .'
            system(cmd)
            # FIXME: we should copy only RESTARTFILES


    def __exit__(self, exc_type, exc_val, exc_tb):
        # FIXME: do we ignore exceptions passed to us here?
        if exc_type is not None:
            return False

        # As a default the working directory is not changed:
        if self.wd is None: return True

        # the files stored in the restartdir may be of course be there before
        # the calculation and be stored by hand or another code, but if they
        # do not exist yet and there are several calculations going on, then
        # they can be done now
        if self.restartdir is not None:
            # The calculations we are considering are all very near each other
            # so the starting values may be from any other finished calculation
            # if any has stored one, there is no need for anymore storage
            # and the code uses the threadsavenes of the path.exists function

            if not path.exists(self.restartdir):
                # so here build the function
                mkdir(self.restartdir)
                # print "QContext: store data for restarting"

                # make sure RESTARTFILES lists files essential for restart:
                for file in RESTARTFILES:
                    # copy the interesting files of in to working directory
                    cmd = "echo cp " + file + " " + self.restartdir
                    system(cmd)
        # it is safer to return to the last working directory, so
        # the code does not affect too many things
        chdir(self.__cwd)

        return True

# python qfunc.py [-v]:
if __name__ == "__main__":
    import doctest
    doctest.testmod()

# Default options for vim:sw=4:expandtab:smarttab:autoindent:syntax
