#!/usr/bin/python

import sys
import pickle

from pts.tools.pathtools import PathTools
from pts.searcher import PathRepresentation
from pts.path import Path
from pts.common import make_like_atoms
import numpy as np
from pydoc import help
from os import path, mkdir, chdir, getcwd, system

def main(argv=None):
    """
    Takes a path.pickle file and estimates the transition states from it

    gives back the transition states (and the modevectors) for different
    transition state estimates

    first argument is the path.pickle file to read in
    (or --help to get this help text)

    One can choose which transtition state estimates are to be generated
    by giving the numbers related to the wanted ones (default is all of them)
    1  : highest
    2  : Spline
    3  : Spline and average
    4  : Spline and cubic
    5  : Three points
    6  : Bell method

    Other possible arguments:
    --p : print the resulting transition state estimates (is also the default
          if no other argument is set
    --m : print also available mode vector estimates
          (can also be given by any combination of p and m)
    --f : calculates the forces on each transition state estimate and
          gives also their projection on the mode vectors
          (the calculator, which is used, is pickled in the path.pickle)
          by default the --p option is set false
          the following argument can be the name of a file to write the
          forces and energies to, which may be reused for later calculations
          the file's name should not be a number or start with --
    --ff: like f but the output of the comparision is printed  to an external file
          energyandforce.dat
    --r : the next argument has to be the name of a file to read the forces from
          it is only useful together with the --f option
          if it is set, the program will look in the forces file if there is 
          already a calculation for the current approximation (note it will not
          check anything else) and then use it instead of a new calculation
          missing forces will be calculated as before
          The file should be look like the energyandforce.dat generated with the 
          --ff option
    --c : for comparing the strings with the previous ones
          the output are relevant data from the approximated string
          the --p option is set to false by default
          if the next argument is a file name (for a logfile)
          the program search in it for data to compare the results
          from

    Except with the --ff option all output goes by default to the standard output
    """

    printwithmodes = False
    printvectors = True
    calculateforces = False
    reloadfile = None
    wanted = []
    fileout = "-"
    fileout2 = None
    comparepath = False
    dump = False

    # structure of inputs, has to be improved later
    if argv is None:
         argv = sys.argv[1:]

    # The --help option is handled seperatly
    # The first input argument has to be the path.pickle object
    if argv[0] == '--help':
        print main.__doc__
        sys.exit()
    else:
        try:
            filename = argv[0]
            f_ts = open(filename,"r")
        except:
            print "ERROR: No path file found to read input from"
            print "First argument of call must be a path.pickle object"
            print "Usage of this function:"
            print main.__doc__
            sys.exit()

    if len(argv)>1:
        argv = argv[1:]
        # cycle over all the remaining arguments, they
        # are moved to the beginning, so argv[0] is always the
        # first argument not yet interpreted
        for i in range(len(argv)):
            if argv[0].startswith('--'):
                # the options start with --
                # sometimes the following argument(s) belong
                # also to the option and are read in also
                arg = argv[0][2:]
                if arg in ['m','pm', 'mp'] :
                     printwithmodes = True
                     printvectors = True
                elif arg == 'p':
                     printvectors = True
                elif arg == 'd':
                     print "Only special output"
                     dump = True
                elif arg in ['ff', 'ffile', 'foutput']:
                     fileout2 = 'energyandforce.dat'
                     calculateforces = True
                     printvectors = False
                     try:
                         if not argv[1].startswith('--'):
                           try:
                             int(argv[1])
                           except:
                             fileout = argv[1]
                             argv = argv[1:]
                     except:
                       pass
                elif arg.startswith('f'):
                     calculateforces = True
                     printvectors = False
                     if not argv[1].startswith('--'):
                        try:
                            int(argv[1])
                        except:
                            fileout = argv[1]
                            argv = argv[1:]
                elif arg in ['reload', 'reloadforces', 'reloadfile', 'r']:
                     reloadfile = argv[1]
                     argv = argv[1:]
                elif arg == 'c':
                     comparepath = True
                     printvectors = False
                     filecomp = None
                     try:
                         filecomp =  open(argv[1], "r")
                         argv = argv[1:]
                         numinfile = int(argv[1])
                         argv = argv[1:]
                     except:
                         numinfile = -1
                argv = argv[1:]
            else:
                try:
                   # the arguments not starting with --, may be the number
                   # of the transition state approximation wanted
                   for a in argv:
                       wanted.append(int(a))
                except:
                    #FIXME: is it okay to ignore everything else?
                    pass
                argv = argv[len(wanted):]
            if argv == []:
                break

    # if none is choosen, all are selected
    if wanted == []:
        wanted = [1, 2, 3, 4, 6]

    posonstring = None
    posonstring2 = None
    # load pickle object as seen by Hugh
    try:
        coord_b, energy_b, gradients_b, posonstring, posonstring2, at_object =  pickle.load(f_ts)
        f_ts.close()
    except :
        try:
            f_ts.close()
            f_ts = open(filename,"r")
            coord_b, energy_b, gradients_b, posonstring, at_object =  pickle.load(f_ts)
            f_ts.close()
        except ValueError:
            f_ts.close()
            f_ts = open(filename,"r")
            coord_b, energy_b, gradients_b, at_object = pickle.load(f_ts)
            f_ts.close()
    # calculate the (wanted) estimates
    estms, stx1, stx2, stx3 = esttsandmd(coord_b, energy_b, gradients_b, at_object, posonstring, posonstring2, wanted)
    # show the result
    if printvectors:
        if dump:
            print_estimatesdump(estms, at_object)
        else:
            print_estimates(estms, at_object, printwithmodes)
    if calculateforces:
        getforces(estms, at_object, fileout, reloadfile, fileout2, dump)
    if comparepath:
        newpath = Path(coord_b, stx2)
        oldpath = Path(coord_b, stx1)
        thirdpath = Path(coord_b, stx3)
        comparepathes(oldpath, newpath, thirdpath, gradients_b, numinfile, filecomp)

def esttsandmd(coord_b, energy_b, gradients_b, at_object, states = None,\
              statesold = None, ts_wanted = [1, 2, 3, 4, 5, 6] ):
    """
    calculating of wanted TS-estimates and their modes
    This is done in two different ways of parametrizing the string:
    First as it is done in the pathway tools
    Second with the spacing gotten from the PathRepresentation object
    (should be the same as opptimized)
    """

    # in this variable the estimates will be stored as
    # ( name, ts-estimate object, (modename, modevec) * number of modeapprox
    ts_all = []

    # First the simple path
    path = PathTools(coord_b, energy_b, gradients_b)
    # get the wanted objects
    estfrompathfirst(path, ts_all, at_object, " with coordinate distance of beads", ts_wanted)

    statex1 = path.steps
    # Build up a PathRepresentation, initialize it, generate new bead (spacing)
    # and get back the spacings on the string
    numbeads = len(energy_b)
    ptr = PathRepresentation(coord_b, numbeads)
    ptr.regen_path_func()
    #ptr.generate_beads(True)
    startx =  ptr.positions_on_string()
    if not states == None:
        startx = states
    # with the additional startvalue startx the same as for the other path
    path2 = PathTools(coord_b, energy_b, gradients_b, startx)
    statex2 = path2.steps
    if states == None:
        estfrompath(path2, ts_all, at_object, " with initial distance by string", ts_wanted )
    else:
        estfrompath(path2, ts_all, at_object, " with given distance by string", ts_wanted )
    statex3 = startx
    if not statesold == None:
        startx = statesold
        #path = PathTools(coord_b, energy_b, gradients_b, startx)
        #estfrompath(path, ts_all, at_object, " with given distance by previous string", ts_wanted )
        statex3 = path.steps

    return (ts_all, statex1, statex2, statex3)


def estfrompathfirst(pt, ts_sum, cs, addtoname, which):
    """
    Some approximations are independent of the path,
    thus this wrapper calculates all, while
    estfrompath only calculates the one depending on a path
    """
    ts_est = []
    #cs_c = cs.copy()
    if 1 in which:
        ts_est.append(('Highest', pt.ts_highest()[-1]))
    if 5 in which:
        ts_int = pt.ts_threepoints()
        ts_est.append(('Three points', ts_int[-1]))
    if 6 in which:
        ts_int = pt.ts_bell()
        ts_est.append(('Bell Method',ts_int[-1]))
    # generates modevectors to the given TS-estimates
    for name, est in ts_est:
         energy, coords, s0, s1,s_ts,  l, r = est
         cs.set_internals(coords)
         modes =  pt.modeandcurvature(s_ts, l, r, cs)
         addforces = neighborforces(pt, l, r)
         ts_sum.append((name, est, modes, addforces))

    estfrompath(pt, ts_sum, cs, addtoname, which)



def estfrompath(pt2, ts_sum, cs, addtoname, which ):
    """
    Calculates the TS-estimates and their modevectors
    which are choosen and put them back together
    """
    ts_est = []
    #cs_c = cs.copy()
    if 2 in which:
        ts_est.append(('Spline only', pt2.ts_spl()[-1]))
    if 3 in which:
        ts_est.append(('Spline and average', pt2.ts_splavg()[-1]))
    if 4 in which:
        ts_est.append(('Spling and cubic', pt2.ts_splcub()[-1]))

    # generates modevectors to the given TS-estimates
    for name, est in ts_est:
         energy, coords, s0, s1,s_ts,  l, r = est
         cs.set_internals(coords)
         modes =  pt2.modeandcurvature(s_ts, l, r, cs)
         addforces = neighborforces(pt2, l, r)
         ts_sum.append((name + addtoname , est, modes, addforces))

def comparepathes(oldpath, path, thirdpath, gradients, num, file):

    print "Data of new path"
    xs, project = projpath(path, gradients)

    print "Data of old path"
    xso, projecto = projpath(oldpath, gradients)

    #print "Data of third path"
    #xst, projectt = projpath(thirdpath, gradients)

    if not file==None:
        print
        print "==========================================================="
        print "      Comparision of the different path approximations:"
        print "==========================================================="
        if num < 0 :
            num = 0
        rightnum = False
        for line in file:
             if line.startswith('Chain of States Summary'):
                 fields = line.split()
                 if str(num) in fields:
                     rightnum = True
                 else:
                     rightnum = False
             if rightnum:
                 if line.startswith('Para Forces'):
                     fields = line.split()
                     dataline = []
                     datapoints = (len(fields) - 2) / 2
                     for i in range(datapoints):
                         dataline.append(float(fields[3 +2 * i]))
                     print "Difference in the projection of the force on the string:"
                     diff1 = [project[i] - dataline[i] for i in range(len(dataline))]
                     diff2 = [projecto[i] - dataline[i] for i in range(len(dataline))]
                     #diff4 = [projectt[i] - dataline[i] for i in range(len(dataline))]
                     diff3 = [projecto[i] - project[i] for i in range(len(dataline))]
                     #diff5 = [projectt[i] - project[i] for i in range(len(dataline))]
                     #diff6 = [projectt[i] - projecto[i] for i in range(len(dataline))]
                     print "the values from the path approximations (old/new)  and the values stored in the logfile"
                     #print "the values from the path approximations (old/new/third)  and the values stored in the logfile"
                     for i in range(len(dataline)):
                          print "%-d    %-12.7f  %-12.7f | %-12.7f" %  (i, projecto[i], project[i], dataline[i])
                          #print "%-d    %-12.7f  %-12.7f  %-12.7f  | %-12.7f" %  (i, projecto[i], project[i], projectt[i], dataline[i])
                     print "The differences are (logfile to old/new):"
                     #print "The differences are (logfile to old/new/third):"
                     for i in range(len(dataline)):
                          print "%-d    %-12.7f  %-12.7f " % (i, diff2[i], diff1[i])
                          #print "%-d    %-12.7f  %-12.7f  %-12.7f" % (i, diff2[i], diff1[i], diff4[i])
                     print "The differences between the projections:"
                     print "old - new"
                     differ = '   '.join(['%12.7f' % i for i in diff3])
                     print differ
                     #print  "third - new"
                     #differ = '   '.join(['%12.7f' % i for i in diff5])
                     #print differ
                     #print "third - old"
                     #differ = '   '.join(['%12.7f' % i for i in diff6])
                     #print differ
                 if line.startswith('Bead Path Position'):
                     fields = line.split()
                     dataline = []
                     datapoints = (len(fields) - 3) / 2
                     for i in range(datapoints):
                         dataline.append(float(fields[4 +2 * i]))
                     print "Difference of the stored bead positions (to old/new):"
                     #print "Difference of the stored bead positions (to old/new/third):"
                     diff1 = [xso[i]/xso[-1] - dataline[i] for i in range(len(dataline))]
                     diff2 = [xs[i] - dataline[i] for i in range(len(dataline))]
                     #diff3 = [xst[i]/xst[-1] - dataline[i] for i in range(len(dataline))]
                     differ1 = '   '.join(['%12.7f' % i for i in diff1])
                     differ2 = '   '.join(['%12.7f' % i for i in diff2])
                     #differ3 = '   '.join(['%12.7f' % i for i in diff3])
                     print differ1
                     print differ2
                     #print differ3


def projpath(path, gradients):
    print path.get_nodes()
    xs = path.xs
    print xs
    print "Forces on String"
    project = []
    for i, x in enumerate(xs):
         grad = gradients[i].flatten()
         mode = -path.tangent(x).flatten()
         proj = np.dot(mode, grad)
         print proj
         project.append(proj)
    return xs, project


def neighborforces(pt, il, ir):
    paral, perpl = oneneighb(pt, il)
    parar, perpr = oneneighb(pt, ir)
    return (paral, perpl, parar, perpr)

def oneneighb(pt, i):
    mode = -pt.xs.tangent(pt.xs.xs[i]).flatten()
    fr = pt.gradients[i].flatten()
    para, perp = para_perp_forces(mode, fr)
    perprms = np.sqrt(np.dot(perp.flatten(), perp.flatten()))
    return para, perprms

def getforces(ts_sum, cs, file, reloadfile, file2, dump):
    """
    Calculates the energy and the forces of the ts_approximates
    and makes the dot products with all the modevectors for the forces
    """

    # there are different possibilities whereto the output should go
    # default is standart output for all
    # The forces and Energies calculated may be stroed seperately
    if file == "-":
        write = sys.stdout.write
    else:
        write = open(file,"w").write
        write("Forces calculated for the different transition state estimates\n")
        write("Units are eV/Angstrom\n")

    if file2 == None:
        write2 = sys.stdout.write
    else:
        write2 = open(file2,"w").write

    if dump:
        write2(" E in eV: calc, appr., diff; f_max; para forces in eV/A: bead_l, at approx, bead_r;  perp forces in eV/A: bead_l, at approx, bead_r\n")

    # For all the geometries we have do:
    for i, (name, est, modes, addforces) in enumerate(ts_sum):
        # get geometries, string informations and approximated energy
        # form the estimated storage
        energy, coords, s0, s1,s_ts,  l, r = est
        # put the geomtry in the working "faked" atoms object
        cs.set_internals(coords)

        for1, for2, for3, for4 = addforces
        atnames = cs.get_chemical_symbols()
        trueE = None
        cartforces = None

        # Maybe the forces and energies have been stored before
        #Then they only have to be reread
        # Note that this function looks if the wanted approximation is in
        # the file, it decides for each of them seperatly
        # But it does not check if the file is realy for the current molecule
        if not reloadfile == None:
            trueE, cartforces = reloadfande(reloadfile, name, len(atnames))
            # we also want the forces in internal coordinates (especially as there
            # the constraints are used)
            forces = transformforces(cartforces, cs)

        # if the energies and forces have not been stored, they have to be calculated
        # here, use for it the atoms object
        if trueE == None:
            wopl = getcwd()
            wx = "mode%i" % i
            if not path.exists(wx):
               mkdir(wx)
            chdir(wx)
            trueE = cs.get_potential_energy()
            cartforces = cs.get_cartforces()
            forces = cs.get_forces()
            chdir(wopl)

        cartforces = np.asarray(cartforces)

        if dump:
            projection = []
            for  namemd, modevec in modes:
                 projection.append(np.dot(np.asarray(modevec).flatten(), np.asarray(cartforces).flatten()))
            para, perp = para_perp_forces( np.asarray(modevec).flatten(), np.asarray(cartforces).flatten() )
            force2rms = np.sqrt(np.dot(perp.flatten(), perp.flatten()))
            write2("  %16.9e  %16.9e  %16.9e   %16.9e   %16.9e   %16.9e  %16.9e  %16.9e   %16.9e   %16.9e\n" % ( trueE, energy,  (energy - trueE), abs(forces).max(),for1, projection[2], for3 , for2, force2rms, for4))
            continue


        # output for each of the approximations
        write("Looking at the approximation %s\n" % name)
        write("The energy is: %16.9e\n" % trueE)
        for num, force_n in enumerate(cartforces):
            write(_tostr_forces(atnames[num], force_n))

        write2("-------------------------------------------------------------------\n" )
        write2("The observations of energy and forces for the case %s are:\n" % name)
        write2("The energies are approximated %16.9e and true was %19.6e\n" % (energy, trueE))
        write2("The difference in Energy (approx - true) is: %16.9e\n" % (energy - trueE))
        write2("The   maximum  internal force component  is: %16.9e\n" % abs(forces).max() )
        write2("The   maximum Cartesian force component  is: %16.9e\n" % abs(cartforces.flatten()).max() )

        write2("\nThe force component projected on the modevectors\n")
        write2("            modevector     |      value\n")

        para = None
        perp = None
        for namemd, modevec in modes:
        #     write2("       for the modevector %s\n" % namemd)
             projection = np.dot(np.asarray(modevec).flatten(), np.asarray(cartforces).flatten())
        #     write2("       has the value:     %16.9e\n" % projection)
             write2("  %24s | %16.9e\n" % (namemd, projection))
             if namemd == "frompath":
                para, perp = para_perp_forces( np.asarray(modevec).flatten(), np.asarray(cartforces).flatten() )
                force2rms = np.sqrt(np.dot(perp.flatten(), perp.flatten()))

        if not para == None:
            write2("\n    The para/perp forces are:\n")
            write2("              at approximation,                      bead before,                          bead after\n")
            write2("  %16.9e / %16.9e   %16.9e / %16.9e   %16.9e / %16.9e\n" % (para,force2rms , for1, for2, for3, for4 ))


def para_perp_forces( m, f):
    if not ( abs( np.dot(m, m) - 1) < 1e-10):
        m /= np.sqrt(np.dot(m,m))
    para = np.dot(m, f)
    perp = f - para * m
    return para, perp

def _tostr_forces(nam, force):
    force = tuple( map(float, force) )
    fields = (nam,) + force

    return ( "%s      %16.9e %16.9e %16.9e\n" % fields )

def _parse_force(lines, max):
     for i, line in enumerate(lines):
         if i >= max:
             return
         grads = _parse_f1(line)
         yield grads

def _parse_f1(line):
     fields = line.split()
     grad = map(float, fields[1:4])
     return grad

def transformforces(c_forces, cs):
     forces_flat = np.asarray(c_forces)
     forces_flat = forces_flat.flatten()
     transform_matrix, errors = cs.get_transform_matrix(cs._mask(cs._coords))
     forces_coord_sys = np.dot(transform_matrix, forces_flat)

     forces_coord_sys = cs.apply_constraints(forces_coord_sys)
     make_like_atoms(forces_coord_sys)
     return forces_coord_sys

def reloadfande(file, name, num):
     lines = open(file)
     found = False
     energy = None
     grads = None
     for  line in lines:
         if name in line:
              found = True
              break
     if found:
         fields = lines.next().split()
         energy = float(fields[3])
         grads = list(_parse_force(lines, num))
     return energy, grads


def print_estimates(ts_sum, cs, withmodes = False ):
     """
     Prints the transition state estimates with their geometry
     in xyz-style, and their mode vectors if wanted
     """
     print "==================================================="
     print "printing all available transition state estimates"
     print "---------------------------------------------------"
     for name, est, modes, addforces in ts_sum:
          print "TRANSITION STATE ESTIMATE:", name
          energy, coords, s0, s1,s_ts,  l, r = est
          print "Energy was approximated as:", energy
          print "This gives the positition:"
          cs.set_internals(coords)
          print cs.xyz_str()
          print
          if withmodes:
              print "The possible modes are:"
              for namemd, modevec in modes:
                   print "Approximation of mode in way ", namemd
                   for line in modevec:
                       print "   %12.8f  %12.8f  %12.8f" % (line[0], line[1], line[2])

              print

def print_estimatesdump(ts_sum, cs ):
     """
     Prints all the geometries as a (jmol) xyz file
     """
     print
     for name, est, modes, addforces in ts_sum:
          numats = cs.atoms_count
          print numats
          energy, coords, s0, s1,s_ts,  l, r = est
          print "Energy was approximated as:", energy
          cs.set_internals(coords)
          print cs.xyz_str()



if __name__ == "__main__":
    main()


