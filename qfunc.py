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


class QFunc(Func):
    def __init__(self, atoms, calc=LennardJones(), startdir = 'restarthelp', restart = False):

        # we are going to repeatedly set_positions() for this instance,
        # So we make a copy to avoid effects visible outside:
        self.atoms = atoms.copy()
        self.calc = calc
        self.atoms.set_calculator(calc)
        # for the option of working in a workingdirectory different to
        # the one used, and the possibility to use old stored datafiles
        # for restart some extra parameters are needed
        self.basdir = getcwd()
        self.startdir = self.basdir + '/' + startdir
        self.restart = restart

    # (f, fprime) methods inherited from abstract Func and use this by default:
    # the number is used to get a unique working directory, if several tasks are
    # performed in parallel, this might be useful. As a default the working directory
    # is not changed,
    # it is changed if the function is called with positions and a number
    def taylor(self, positions, number = None ):
        "Energy and gradients"

        # only if number exists, there is done something
        if not number == None:
             # first the working directory is changed to
             # to the number, as a subdirectory of the current
             # one; if it does not exist, it is
             # created, make sure that if it exists, there is
             # no grabage inside
             print "Qfunc: change directory"
             wodir = self.basdir + '/' + str("%03d" % number)
             print "from", self.basdir, "change to", wodir
             if not path.exists(wodir):
                 mkdir(wodir)
             chdir(wodir)
             # if the restart flag is set, there may exist some
             # files stored elsewere (in self.startdir) which might be
             # useful, startdir is supposed to be a subdirectory from
             # the main working directory
             if path.exists(self.startdir) and self.restart:
                 print "Qfunc: using data for restarting from directory"
                 print self.startdir
                 # all the files being in self.startdir are copied in
                 # the current working directory
                 cmd = "cp " + self.startdir + '/*  .'
                 system(cmd)


        # update positions:
        self.atoms.set_positions(positions)

        # request energy:
        e = self.atoms.get_potential_energy()

        # request forces. NOTE: forces are negative of the gradients:
        g = - self.atoms.get_forces()

        # if number exists, there are things going on with the subdirectories
        if not number == None:
            # the files stored in the startdir may be of course be there before
            # the calculation and be stored by hand or another code, but if they
            # do not exist yet and there are several calculations going on, then
            # they can be done now
            if self.restart:
                # The calculations we are considering are all very near each other
                # so the starting values may be from any other finished calculation
                # if any has stored one, there is no need for anymore storage
                # and the code uses the threadsavenes of the path.exists function
                if not path.exists(self.startdir):
                    # so here build the function
                    mkdir(self.startdir)
                    print "Qfunc: store data for restarting"
                    # the altered ASE framework includes a function
                    # for the calculators, which gives back the names
                    # of interesting files for a restart as a string
                    # this function is not included in standard ASE
                    # it is not implemented for all calculators yet
                    for element in self.calc.restartfiles():
                        # copy the interesting files of interest in the
                        # working directory
                        cmd2 = "cp " + element + " " + self.startdir
                        system(cmd2)
            # it is saver to return to the last working directory, so
            # the code does not affect too many things
            chdir(self.basdir)

        # return both:
        return e, g

# a list of restartfiles that might be usefull to copy-in
# for a warm-start of a calculation, if it is complete, no
# modifications to ASE are necessary:
RESTARTFILES = ["WAVECAR", "saved_scfstate.dat", "*.testme"]

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
        # the number is used to get a unique working directory, if several tasks are
        # performed in parallel, this might be useful. As a default the working directory
        # is not changed:
        if self.wd is None: return

        # save for __exit__ to chdir back:
        self.__cwd = getcwd()

        # first the working directory is changed to
        # If it does not exist, it is
        # created, make sure that if it exists, there is
        # no grabage inside

        # print "QContext: change directory"
        # print "QContext: chdir to", self.wd
        if not path.exists(self.wd):
            mkdir(self.wd)
        chdir(self.wd)

        # if the restart flag is set, there may exist some
        # files stored elsewere (in self.restartdir) which might be
        # useful, restartdir is supposed to be a subdirectory from
        # the main working directory
        if self.restartdir is not None and path.exists(self.restartdir):
            # print "QContext: using data for restarting from directory", self.restartdir

            # files being in self.restartdir are copied in
            # the current working directory
            cmd = "echo cp " + self.restartdir + '/*  .'
            system(cmd)
            # FIXME: we should copy only RESTARTFILES


    def __exit__(self, exc_type, exc_val, exc_tb):
        # FIXME: do we ignore exceptions passed to us here?

        # if number exists, there are things going on with the subdirectories
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
                print "QContext: store data for restarting"
                # the altered ASE framework includes a function
                # for the calculators, which gives back the names
                # of interesting files for a restart as a string
                # this function is not included in standard ASE
                # it is not implemented for all calculators yet
                for element in RESTARTFILES:
                    # copy the interesting files of interest in the
                    # working directory
                    cmd = "echo cp " + element + " " + self.restartdir
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
