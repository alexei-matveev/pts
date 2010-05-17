#!/usr/bin/python
import numpy as np
import math
import sys
from sys import stdout

SMALL = 0.00000000001

def readxyz(file, n ):
    '''reads in the positions from a xyz file *file*
    *n* tells when to stop, so the file need not be
    finished
    '''
    for i in range(int(n)):
        line = file.readline()
        fields = line.split()
        #print fields
        positions = [float(fields[1]), float(fields[2]), float(fields[3])]
        #print positions
        yield positions
    return

def getdistancesandanglesfromxyz():
   '''performs all the calculations
   '''
   # these variable may be used but need not
   # they need be there for check anyway
   filein = []
   deg = False
   expand = False
   allval = []
   # read in parameters needed ones are a xyz file
   # and a list what is wanted to be calculated
   if len(sys.argv) < 2:
       print "ERROR: at least one argument (inputfile) is required\n"
       helpfun()
   else:
     second = 0
     for num in range(1, len(sys.argv)):
         if second ==0 :
           # the other arguments
           if sys.argv[num].startswith('--'):
             option = sys.argv[num][2:]
             if option == 'help':
                 helpfun()
             # switch output of angles to degree
             elif option == 'degree':
                 deg = True
             # you can also get values with atoms from different
             # cells, more input is needed (second 3 and 4) and should
             # follow imedately
             elif option == 'expand':
                 expand = True
                 second = 3
             # gives the input for the wanted values
             elif option.startswith('i'):
                 second = 1
             else :
                 print "ERROR: I don't know this option\n"
                 helpfun()
           else :
               # everything else is a xyz file
               filein.append(sys.argv[num])
         elif second ==1:
             # the input of the wanted values can be given directly or by file
             opt2 = sys.argv[num-1][3:]
             allval = []
             # by file
             if opt2 == 'i':
                second = 0
                fileval = open(sys.argv[num],"r" )
                for num, line  in enumerate(fileval):
                    #print line, line[0]
                    fields = line.split()
                    allval.append(interestingvalue(fields[0]))
                    allval[num].partners = []
                    for xval in fields[1:]:
                        allval[num].partners.append(int(xval))
                    #print allval[num].partners
            # directly
             elif opt2 == 'd':
                kval = int(sys.argv[num])
                # print kval
                second = 2
                kiter = 1
                kfinish = True
             else :
                print "ERROR: I don't know this option\n"
                helpfun()
         # still belongs to the direct input values, there are kval of them
         # order see helpfun()
         elif second == 2:
             #print sys.argv[num]
             if kfinish:
                 allval.append(interestingvalue(sys.argv[num]))
                 allval[kiter -1].partners = []
                 kfinish = False
             else :
                 allval[kiter-1].partners.append(int( sys.argv[num]))
                 if (len(allval[kiter-1].partners) == int(allval[kiter-1].lengthneeded())):
                     #print "now to the next one"
                     kiter += 1
                     kfinish = True
             if kiter == kval  + 1:
                  second = 0
             #print second, kfinish, kiter, kval
         # for expansion, first read in cell
         elif second == 3:
             filecell = open(sys.argv[num],"r" )
             cell = np.zeros((3,3))
             for  num, line  in enumerate(filecell):
                    fields = line.split()
                    cell[num,0] = float(fields[0])
                    cell[num,1] = float(fields[1])
                    cell[num,2] = float(fields[2])
             second = 4
         # expansion, get atoms which should be shifted and where they should go
         elif second == 4:
             filemove = open(sys.argv[num],"r" )
             tomove = []
             howmove = []
             for  num, line  in enumerate(filemove):
                    fields = line.split()
                    tomove.append( int(fields[0]))
                    howmove.append([float(fields[1]), float(fields[2]), float(fields[3])])
             second = 0
         else:
             print "ERROR: This should not happen\n"
             helpfun()

     # write to standard output
     write = stdout.write
     if filein == None:
     # there should be at least one xyz input file
         write("ERROR: No input file given\n")
         helpfun()
    # loop over all inputfiles
     for filename in filein:
        file = open(filename, "r")
        loop = 1
        write("#chart of internal coordinates in the run \n")
        write("#observed in file: %s  " % (filename) )
        if deg : write("#the following values were calculated, distances are in Angstroms; angles in degrees\n")
        else : write("#the following values were calculated, distances are in Angstroms; angles in radiens\n")
        # xyz file starts with number of atoms
        nnull = file.readline()
        n = nnull
        if allval == []:
        # by now there should be a list of what values are wanted to be calculated
            print "ERROR: program needs more input"
            print "Give the values to collect"
            helpfun()
        write("#loop; ")
        for k in range(len(allval)):
        # tell what values will be calulated
            write("%s with atoms :" % (allval[k].whatsort()))
            for number in allval[k].partners:
                write("%i " % number )
            write(";")
        write("\n")
        while nnull == n :
           # goes through the whole xyz file, which may contain several geometries
           #( like in a jmol movie)
           # the next line is of no interest
           random = file.readline()
           # read in the positions, give them back as a list
           positionslist = list(readxyz(file, n))
           # expand number of atoms if some of different cells are wanted
           if (expand): expandlist(positionslist, cell, tomove, howmove)
           # make an array of the positions
           positions = np.array(positionslist)
           #print positions
           # the actual writing of the programm
           results = returnall(allval, positions, deg, loop)
           writeall(write, results, loop)
           n = file.readline()
           loop += 1

def writeall(write, results, loop):
      write("%4i " % (loop))
      for res in results[1:]:
          write("%22.12f " % res)
      write("\n")

def returnall(allval, positions, deg, loop):
      results = [loop]
      for value in allval:
           # loop over all wanted values, order of them says how to calculate
           if value.order == 2:
               results.append( radii(positions, value.partners))
           elif value.order == 3 or value.order == 4 :
               results.append( angle(positions, value.partners, deg))
           elif value.order == 5 :
               results.append( dihedral(positions, value.partners, deg))
           elif value.order == 6 :
               results.append( distancetoplane(positions, value.partners))
      return results


def expandlist(positionslist, cell, tomove, howmove):
      # expand list of atoms with atom[num] from cell howmove with given cell is 0
      for num, new in enumerate(tomove):
           expandfor =  np.dot(howmove[num], cell)
           #print cell, howmove[num], expandfor
           pos = positionslist[new-1] + expandfor
           #print positionslist[new-1] , pos
           positionslist.append(pos)


def radii (positions, iconns):
    # distance between two atoms
    a = iconns[0] - 1
    b = iconns[1] - 1
    diff = positions[a]-positions[b]
    rad = math.sqrt(np.vdot(diff,diff))
    return rad

def angle (positions, iconns, deg):
    # angle between two (difference) vectors
    # if the middle atom is the same, case else
    # takes care of it
    if len(iconns) == 4:
         a = iconns[0] - 1
         b = iconns[1] - 1
         c = iconns[2] - 1
         f = iconns[3] - 1
    else :
         a = iconns[0] - 1
         b = iconns[1] - 1
         c = iconns[1] - 1
         f = iconns[2] - 1
    d1 = positions[a] - positions[b]
    d2 = positions[f] - positions[c]
    db1  = math.sqrt( np.vdot(d1, d1) )
    db2  = math.sqrt( np.vdot(d2, d2) )
    x =  np.vdot(d1, d2) / (db1 * db2)
    if  x > 1.0 and x < 1.0 + SMALL : x = 1.0
    if  x < -1.0 and x > -1.0 - SMALL :  x = -1.0
    alpha = np.arccos(x)
    if deg: alpha *= 180 / math.pi
    return alpha

def dihedral (positions, iconns, deg ):
    # dihedral angle
    a = iconns[0] - 1
    b = iconns[1] - 1
    c = iconns[2] - 1
    f = iconns[3] - 1
    d1 = positions[b] - positions[a]
    d2 = positions[c] - positions[b]
    d3 = positions[f] - positions[c]
    e1 = np.cross(d1, d2)
    e2 = np.cross(d2, d3)
    eb1 = math.sqrt(np.vdot(e1, e1))
    eb2 = math.sqrt(np.vdot(e2, e2))
    e1 /= eb1
    e2 /= eb2
    x = np.vdot (e1, e2)
    if  x > 1.0 and x < 1.0 + SMALL : x = 1.0
    if  x < -1.0 and x > -1.0 - SMALL :  x = -1.0
    vor = 1.0
    if ( np.vdot( np.cross(d1, d3), d2) < 0): vor = -1.0
    # cos(gamma) = ((b-a) x (c-b)) * ((c-b) x (d-c)) in unit vectors
    gamma = vor * np.arccos(x)
    if deg : gamma *= 180 / math.pi
    return gamma

def distancetoplane(positions, iconns):
    # f is single point, a, b,c defining plane
    # they should not be in a line
    a = iconns[1] - 1
    b = iconns[2] - 1
    c = iconns[3] - 1
    f = iconns[0] - 1
    d1 = positions[b] - positions[a]
    d2 = positions[c] - positions[b]
    n = np.cross(d1,d2)
    n /= np.sqrt(np.vdot(n,n))
    dis = np.dot(n, positions[f] - positions[a])
    return dis

def helpfun():
    print "This program reads xyz files and gives as output the value of specific coordinates of interest"
    print "A list of all the coordinates of interest have to be provided"
    print "This list can be given as a file via:"
    print "               --ii filename"
    print " or directly via (here num gives the number of coordinates of interest):"
    print "               --id num list"
    print "the list should look contain first a number saying what kind of coordinate is looked at"
    print "then should follow the numbers of the atoms (in the order of the xyz file) involved"
    print "if written in a file, every number of interest should be in its own line"
    print ""
    print "The kind of coordinates can be choosen two different way: by a number, related to the"
    print "choosen kind or with an abrevation. This values can be optained from the following table:"
    print "'kind of coordinate': number; abrevation:'number of atoms needed to calculate'('speciality')"
    print "   distance: 2; dis : 2"
    print "   angle:    3; ang : 3  (angle a-b-c)"
    print "             4; ang4: 4  (if the angle between the two vectors a-b and c-d is wanted)"
    print "   dihedral: 5; dih : 4  (dihedral for the atoms a -b -c-d )"
    print "  dis to pl: 6; dp  : 4  (distance between a and plane defined by b,c and d)"
    print ""
    print "The xyz file name has to be given as an argument, there may be several of them"
    print "or one very long containing several geometries"
    print "additional options are:"
    print "         --help   : shows this help text"
    print "         --degree : angles are given in degree, else in radiens"
    print "         --expand cellfile expandlistfile : if another atom of a neigboring cell is"
    print "                                            wanted, this option helps"
    print "                  cellfile should contain the unit cell in the following format:"
    print "                                            v11 v21 v31                         "
    print "                                            v12 v22 v32                         "
    print "                                            v13 v23 v33                         "
    print "                  expandlist should give the atom number in the original cell   "
    print "                  and three numbers for the shifting in which cell direction    "
    print "                         n   a1  a2 a3 "
    print "                  so for example: "
    print "                  cellfile contains:"
    print "                         10  0  0 "
    print "                          0 10  0 "
    print "                          0  0 10 "
    print "                  expandlist contains:"
    print "                         1  2 0 0"
    print "               then values calcualted with the atomnumber n + 1 (n number of atoms in"
    print "                                          original cell) will be calculated with atom 1"
    print "                                          but as if this would be 20 units in the x direction"
    print "                                          meaning beeing two cells in x direction"
    sys.exit()




class interestingvalue:
    def __init__(self, input = None, partners = []):
        # the order gives the kind of values, function
        # whatsort shows the code
        # the atomnumber needed to know where to calculate are
        # in partners
        self.names = [None, None, "dis", "ang", "ang4", "dih", "dp"]
        self.orders = {}
        for z, name in enumerate(self.names):
            self.orders[name] = z

        try:
            input = int(input)
            self.order = input
            self.name = self.names[input]
        except ValueError:
            self.name = input
            self.order = self.orders[input]

        self.partners = partners

    def lengthneeded(self):
    # shows how many partners wanted to calulate the value
        if self.order == 5 or self.order == 6:
            return 4
        else:
            return self.order


    def whatsort(self):
       if self.order == 5:
            return "dihedral angle"
       elif self.order == 4 or self.order ==3 :
            return "angle"
       elif self.order == 2:
            return "distance"
       elif self.order == 6:
            return "distance to plane"
       else:
            print "This sort of coordinate does not exist"
            sys.exit()


 # main:
def main():
    getdistancesandanglesfromxyz()

if __name__ == "__main__":
    import doctest
    doctest.testmod()
    main()

