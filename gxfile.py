#!/usr/bin/python

__all__ = ["gxread", "gxwrite", "is_dummy"]

from sys import stdout
from warnings import warn

# to wrap the 3D vectors as V([x,y,z]):
# from vector import Vector as V
import numpy as np
from numpy import array as V
# from derivatives import DerivVector as V

# Dummy atoms have to be distinguished from real  ones.
# In PG dummy atoms historically have atomic number 99.
# Within the python sources in this repo, let the dummy
# atoms to be identified by this:
DUMMY = 0

# tiny helper function:
def is_dummy(atnum):
    return atnum == DUMMY

# for private use only:
class EOF(Exception): pass

def fromgx( file = "gxfile" ):
    """Return a list of tuples (atnum, position) suitable for
    feeding into Atom() class constructor of ASE:
    E.g.:

            atoms = [ Atom(*a) for a in fromgx() ]

    Note tuple expansion *a in argument to the Atom() constructor.
    """

    atnums, positions, _, _, _, _, _, _ = gxread( file )

    return zip( atnums, positions)
#enddef

def gxread( file='gxfile' ):
    """Read gxfile in following format, return a list of lists (columns)
    that is a table in the row-major order.
    So far only the geometry section is returned.

    (first and last lines are not parts of the file, rather for field width extimation):
    ^12345 1234567890123456789012 1234567890123456789012 1234567890123456789012 123 123   123 123 123   123 123 123
    ^92.00         0.000000000000         0.000000000000         0.000000000000   1   1     0   0   0     0   0   0
    ^99.00         1.000000000000         0.000000000000         0.000000000000   0   2     1   0   0     0   0   0
    ^ 8.00         0.000000000000         0.000000000000         3.354241540616   2   3     1   2   0     1   0   0
    ^ 8.00         0.000000000000         0.000000000000        -3.354241540616   2   4     1   2   3     1   0   0
    ^ ...
    ^ 1.00        -5.184294677816         2.993153927796         0.427911418660   5  18     6   1   3     4   7   0
    ^ 1.00         0.000000000000        -5.986307855591         0.427911418660   5  19     7   1   3     4   7   0
    ^ 1.00         5.184294677816        -2.993153927796        -0.427911418660   5  20     8   1   4     4   7   0
    ^ 1.00        -5.184294677816        -2.993153927796        -0.427911418660   5  21     9   1   4     4   7   0
    ^ 1.00         0.000000000000         5.986307855591        -0.427911418660   5  22    10   1   4     4   7   0
    ^  -1.0         0.000000000000         0.000000000000         0.000000000000   0   0     0   0   0     0   0   0    0
    ^     -28546.827681881085     -28546.827681881085
    ^    1        0.000000000000   0.000000000000   0.000000000000
    ^    2        0.000000000000   0.000000000000  -0.000493202624
    ^   ...
    ^   20       -0.000000596917  -0.000000344630   0.000002806462
    ^   21        0.000000000000   0.000000689261   0.000002806462
    ^12345      1234567890123456 1234567890123456 1234567890123456
     """

    # a generator function:
    def parse(lines):
        # iterator over lines in a file:
        for line in lines:
            fields = _parse_geom(line)
            # stop iterations on negative atomic numbers:
            if fields[0] < 0:  return
            yield fields

    # a generator for lines of the file:
    lines = open(file)

    # force generator to list:
    rows = list( parse(lines) )
    # print "rows=", rows
    
    # rearrange rows to columns:
    cols = zip( *rows )

    # unpack, we will change the type of "xyz":
    atnums, xyz, isyms, inums, iconns, ivars = cols

    # helper funciton to convert a list of 3-vectors to Nx3 numpy array:
    def nx3(xyz):
        # FIXME: how to convert a list of numpy arrays to numpy array?:
        n = len(xyz)
        arr = np.zeros((n, 3))
        for i in xrange(n):
            arr[i,:] = xyz[i]
        return arr
    #enddef

    # convert to 2D array:
    xyz = nx3(xyz)
    # print "xyz=", xyz

    #
    # Second, read in energy:
    #
    try: # if the energy/forces section is present ...
        try: # if the energy line is present ...
            fields = lines.next().split()
            energy = float( fields[0] )
        except StopIteration: # is raised by .next() on EOF
            warn("gxread: gxfile does not contain energy")
            raise EOF, "gxread: no energies"
            # or maybe "return atoms, loop, None" right away?
        #end try

        #
        # Third, read in forces:
        #

         # parse the rest of the file:
        grads = [ _parse_grad(line) for line in lines ]

        # convert to 2D array:
        grads = nx3(grads)
        # print "grads=", grads
    except EOF:
        energy = None
        grads = None
    #end try

    # return all columns and scalar energy:
    return atnums, xyz, isyms, inums, iconns, ivars, grads, energy 
#end def

def gxwrite(atnums, positions, isyms, inums, iconns, ivars, grads=None, energy=None, loop=1, file='-'):
    """Write gxfile in following format,
    (first and last lines are not parts of the file, rather for field width extimation):
    #12345 1234567890123456789012 1234567890123456789012 1234567890123456789012 123 123   123 123 123   123 123 123
    ^92.00         0.000000000000         0.000000000000         0.000000000000   1   1     0   0   0     0   0   0
    ^99.00         1.000000000000         0.000000000000         0.000000000000   0   2     1   0   0     0   0   0
    ^ 8.00         0.000000000000         0.000000000000         3.354241540616   2   3     1   2   0     1   0   0
    ^ 8.00         0.000000000000         0.000000000000        -3.354241540616   2   4     1   2   3     1   0   0
    ^ ...
    ^ 1.00        -5.184294677816         2.993153927796         0.427911418660   5  18     6   1   3     4   7   0
    ^ 1.00         0.000000000000        -5.986307855591         0.427911418660   5  19     7   1   3     4   7   0
    ^ 1.00         5.184294677816        -2.993153927796        -0.427911418660   5  20     8   1   4     4   7   0
    ^ 1.00        -5.184294677816        -2.993153927796        -0.427911418660   5  21     9   1   4     4   7   0
    ^ 1.00         0.000000000000         5.986307855591        -0.427911418660   5  22    10   1   4     4   7   0
    ^  -1.0         0.000000000000         0.000000000000         0.000000000000   0   0     0   0   0     0   0   0    0
    ^     -28546.827681881085     -28546.827681881085
    # 12345678901234567890123 12345678901234567890123
    ^    1        0.000000000000   0.000000000000   0.000000000000
    ^    2        0.000000000000   0.000000000000  -0.000493202624
    ^   ...
    ^   20       -0.000000596917  -0.000000344630   0.000002806462
    ^   21        0.000000000000   0.000000689261   0.000002806462
    #12345      1234567890123456 1234567890123456 1234567890123456
     """
    # special case for file=="-":
    if file == "-":
        write = stdout.write
    else:
        write = open(file,"w").write
    #end if

    #
    # First, the geometry section ...
    #
    cols = (atnums, positions, isyms, inums, iconns, ivars)
    rows = zip( *cols )

    for row in rows: # row = (atnum, pos, isym, inum, iconn, ivar)
        write( _tostr_geom( *row ) )
    #end for

    #
    # ... closed by a line starting with negative number:
    #
    write( "%6.1f %22.12f %22.12f %22.12f   0   0     0   0   0     0   0   0    0\n" \
             % ( -loop, 0.0,0.0,0.0 ) )

    if energy == None: return
    #
    # Second, the energy and gradients (not needed if this is a new geometry):
    #

    # energy:
    write( " %23.12f %23.12f\n"  % (energy,energy) )

    # gradients:
    for inum, grad in enumerate(grads):
        # enumerate starts from zero:
        write( _tostr_grad(inum+1, grad) )
#end def

def _tostr_geom(atnum, pos, isym, inum, iconn, ivar):
    """Return a line of gxfile in following format:
    ^ 1.00        -5.184294677816         2.993153927796         0.427911418660   5  18     6   1   3     4   7   0
    """

    # extract the fields from records:

    if is_dummy(atnum):
        # gxfile convention for dummy atoms:
        atnum = 99.0
    else:
        atnum = float(atnum)

    # concatenate tuples, the first one is a singleton:
    fields = (atnum,) + tuple(pos) + (isym, inum) + tuple(iconn) + tuple(ivar)

    return ( "%5.2f %22.12f %22.12f %22.12f %3i %3i   %3i %3i %3i   %3i %3i %3i\n" \
                 % fields )
#end def

def _tostr_grad(inum, grad):
    """Return a line of gxfile containg cartesian gradients:
    ^   20       -0.000000596917  -0.000000344630   0.000002806462
    """
    #
    # WARNING: all the numerical ids are rebased to 1,
    #          so that those (invalid) -1s become 0.
    #

    # conversion to floats is for the case when vector components are DerivVars:
    grad = tuple( map(float, grad) )

    # concatenate tuples, the first one is a singleton:
    fields = (inum,) + grad
    return ( "%5i      %16.9e %16.9e %16.9e\n" % fields )
#end def

def _parse_geom(line):
    """Parse a line of gxfile in following format:
    ^ 1.00        -5.184294677816         2.993153927796         0.427911418660   5  18     6   1   3     4   7   0
    """
    fields = line.split()
    # print(fields)

    # nuclear charge, may even be fractional, for dummy atoms Z=99:
    atnum = int( float( fields[0] ) )

    # gxfile convention for dummy atoms:
    if atnum == 99.0: atnum = DUMMY

    # exit on negative charge:
    if atnum < 0:
        # this is the only usefull field in this line,
        # but there is no use for it here ...
        return ( atnum, None, None, None, None, None )

    # 3D vector of atomic position:
    pos    = [ float(i) for i in fields[1:4] ] # yes, three fields 1 <= xyz < 4

    # Vrap 3D vectors in vector type, see "import ... as V" at the top:
    pos = V(pos)

    # uniqie atom id, indexing groups of symmetry equivalent atoms:
    isym = int( fields[4] )

    # numeric ID used to define connectiviites:
    inum   = int( fields[5] )

    # connectivities:
    iconn  = [ int(i) for i in fields[6:9] ] # three fields

    # integer lables for "variable" internal coordinates,
    # zero for "constrained" internal coordinates:
    ivar  = [ int(i) for i in fields[9:12] ] # three fields

    return ( atnum, pos, isym, inum, iconn, ivar )
#enddef

def _parse_grad(line):
    """Parse a line of gxfile containg cartesian gradients:
    ^   20       -0.000000596917  -0.000000344630   0.000002806462
    """
    fields = line.split()
    grad = map( float, fields[1:4] )

    # Vrap 3D vectors in vector type, see "import ... as V" at the top:
    return V(grad)
#enddef

# You need to add "set modeline" and eventually "set modelines=5"
# to your ~/.vimrc for this to take effect.
# Dont (accidentally) delete these lines! Unless you do it intentionally ...
# Default options for vim:sw=4:expandtab:smarttab:autoindent:syntax
