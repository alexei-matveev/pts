#!/usr/bin/env python

"""
Use this tool to find out which symmetry (of C1, C2, CS, C2V, C4V, Td)
geometries of an Ar4 Cluster have. This tool really only works on Clusters of the size 4.
The geometries can be given as a path.pickle from a pathsearcher run (string, neb, growingstring, ...)
or as an file containing xyz-geometries. In the later case one has to set the variable --ase in order
to make it easier for the tool to identify the case.

The tool divides the six distances between the four atoms in diagonals and edges.
As a default the diagonals are [1,2] and [3,4]. One may set others by giving
 --diag n1 n2
 Here n1 and n2 are the numbers for ONE diagonal (the other is then clear by default)

 There are other possible options to set:
    --tol n : choose to which accuracy the symmetry is wanted n from {0, 1, 2}, default is 0 tol(0) < tol(1) < tol(2)
    --all   : gives all three tolerances
    --print : instead of plot, print the results
    --onlybeads : if a path.pickle file is used this will make the programm only care about the beads
"""
from numpy import finfo, asarray
from copy import copy, deepcopy
from sys import argv, exit

from ase.io import read

from pts.tools.xyz2tabint import radii
from pts.tools.path2xyz import read_in_path, path_geos
from pts.tools.tab2plot import plot_tabs

tol_absolut = float(finfo(float).eps)
tol_exact = 1.0e-10
tol_rough = 1.0e-5
ar4_diag = ([1,2],[3,4])
ar4_edges = {
 12 : True,
 13 : True,
 14 : True,
 23 : True,
 24 : True,
 34 : True
 }
ar4_sides = [[1,2], [1,3], [1,4], [2,3], [2,4], [3,4]]

def set_sides(diags):
   """
   Sets in global ar4_edges True for all sides wich are edges
   and false for the two diagonals (as seen in the middle bead)
   the number for the diagonals have to be provided
   """
   global ar4_edges
   for ed in ar4_edges.keys():
       ar4_edges[ed] = True
   diag1, diag2 = diags
   d1 = diag1[0]*10 + diag1[1]
   d2 = diag2[0]*10 + diag2[1]
   ar4_edges[d1] = False
   ar4_edges[d2] = False

def equal_sides( geom):
   """
   Finds out which sides have same length (by a tolerance factor)
   Each side will be only given once (for each tolerance level) at least
   They are given back in esi, where all elements are the collection of all
   length (defined by the two atoms) which are the same
   """
   global ar4_sides
   # the esi store the sides
   esa = []
   ese = []
   esr = []
   # to ensure that no side is used several times
   dsa = []
   dse = []
   dsr = []
   for i, side in enumerate(ar4_sides):
         if not side in dsa: # tolerance = floating point exactness
             esa, dsa = find_sides(side, esa, dsa, geom, i, tol_absolut)
         if not side in dse: # low tolerance
             ese, dse = find_sides(side, ese, dse, geom, i, tol_exact)
         if not side in dsr: # hight tolerance
             esr, dsr = find_sides(side, esr, dsr, geom, i, tol_rough)

   return esa, ese, esr

def find_sides(side,es, ds, geom, i, tol):
    """
    Finds all the following sides in ar4_sides
    which have the same length (tolerance) than the
    one of given side, updates also es and ds
    """
    global ar4_sides
    len_s = radii(geom, side)
    ds.append(side)
    line = [side]
    for j in range(i+1, len(ar4_sides)): # only the ones not yet checked
        len_c = radii(geom, ar4_sides[j])
        if abs(len_s - len_c) < tol:
           line.append(ar4_sides[j])
           ds.append(ar4_sides[j])
    if len(line) > 1:
        es.append(line)

    return es, ds

def test_td(es):
    """
    Test if it is symmetry group Td
    All six length should be the same
    """
    if len(es[0]) == 6:
       return True
    else:
       return False

def test_c3(es):
    """
    Test if it is symmetry group C3
    """
    is_c3 = False
    one = 1
    two = 2
    if len(es) == 2: # we need two lenght here
       # one for the bottom triangle and one for
       # the equal tent
       es1 = es[0]
       es2 = es[1]
       if len(es1) == 3 and len(es2) == 3: # both need to have
       # three members
           cn1 = distr(es1) # distribution of atoms numbers
           cn2 = distr(es2)
           # we need the distributions
           # (3, 1, 1, 1) and (2, 2, 2, 0) or any permutation
           # from top to the bottom and the cycle at the bottom
           # no atom can have a distribution of more than three
           if two in cn1 and not one in cn1:
               is_c3 = True
           elif two in cn2 and not one in cn2:
               is_c3 = True

    return is_c3

def distr(lists):
     """
     Counts how often atom numbers are mentioned
     in lists
     """
     count = [0,0,0,0]
     listf = deepcopy(lists)
     listf = asarray(listf).flatten()
     for num in listf:
          count[num -1] += 1
     return count


def test_c2v(es, edg):
    """
    Tests for c2v geometry
    It is assumed that Td geometry has already been checked
    and it is only wanted c2v with no Td, therefore
    there should not be six sides of the same length
    """
    is_c2v = False
    right_axis = False

    large_es = [] # There need to be at least four of the same length
    # in order to have c2v geometry, as there are only six sides at all
    # this one would be the larges of the available equal-sides
    for ei in es:
        if len(ei) > len(large_es):
             large_es = ei

    if len(large_es) == 4: # normal case
         diags, ed_nums =  diag_edg(large_es, edg)
         if  len(diags) == 0:
             is_c2v = True
             right_axis = True
         elif len(diags) == 2:
             zero = 0
             cn = distr(ed_nums)
             if not zero in cn:
                 is_c2v = True
    elif len(large_es) == 5: # one of the other sides is by chance
        # exactly as long as the other four, this way one has in any
        # case c2v geometry
        is_c2v = True
        # one has only to find out if one needs the diagonals or has
        # the geometry in the wanted direction
        diags, ed_nums =  diag_edg(large_es, edg)
        if len(diags) == 1: # one means one can omit the diagonal, two
             # means one has to take them for the four equal sizes
             right_axis = True

    return is_c2v, right_axis

def test_c4v(es):
     """
     Tests if a given c2v geometry is also c4v
     In this case the two diagonals have also to be the
     same, thus the six length have to be divided in
     4 of the same kind and 2 of the same kind, we have
     already verfied that there are 4 of the same kind,
     now the other two may be the same or not
     """
     if len(es) == 2:
        return True
     else:
        return False

def test_cs_and_c2(es, edg):
    """
    Tests if the given geometry is in cs or c2 symmetry
    Test also if it is with the four edges or with the two
    diagonals
    """
    is_cs = False
    is_c2 = False
    zero = 0
    two = 2
    right_axis = False

    # possible equal sizes distributions have either two or three
    # groups
    if len(es) == 2:
       # in principle we need only two equal size groups, but there may be
       # by chance any of the other two length equal to one or two of our
       # groups, first ensure that e1 is larger or equal e2
       if len(es[0]) < len(es[1]):
           e1 = es[1]
           e2 = es[0]
       else:
           e1 = es[0]
           e2 = es[1]

       # groups may have size: (4, 2), (3, 3) , (3, 2) , (2, 2)
       if len(e1) == 4 : # (4, 2)

           # we have in any case cs or c2 geometry, but first find out which:
           # check where the diagonals are, as all six sizes are used
           # we need only to check for one
           d1, r1 = diag_edg(e1, edg)
           if len(d1) == 2: # diags (2, 0)
              right_axis = True # we can use only the non diagonals
              is_cs, is_c2 = helpfun(r1, e2) # decides how the edges are distributed
           if len(d1) == 0: # diags (0, 2)
              # the smaller interval has them opposite to each other, thus
              # c2 geometry, also uses the diagonals
              is_c2 = True
           else: # diags (1, 1)
              # the smaller interval has his sides next to each other
              # this cs, also uses diagonals
              is_cs = True

       elif len(e1) == 3: # (3, 3) or (3, 2)
           # needs to know where the diagonals are
           d1, r1 = diag_edg(e1, edg)
           d2, r2 = diag_edg(e2, edg)

           if len(e2) == 3: # group size (3,3), have to get rid
              # of one length in each group
              # there are the following possibilities for the distribution
              # of the diagonals: (2, 0), (1, 1) and its permutation
              # note that case c3 geometry has been already dealed with
              # therefore in case (1, 1) we have c2 geometry
              is_c2 = True
              if len(d1) == len(d2): # diags (1, 1)
                  right_axis = True

           else: # group size (3, 2)
              # there are following possibilities:
              # (2, 0), (1, 1), (1, 0) and its permutations
              if len(d2) == 2: # diags (0, 2)
                 # two of e1's sides have to be opposite each other
                 is_c2 = True
              elif len(d1) == 2: # diags (2, 0)
                 # e2's two sides can be next to each other (distr (2, 1, 1, 0))
                 # or opposite to each other (distr (1, 1, 1, 1))
                 # no other numbers possible as they are for sure on the four edges
                 cn = distr(e2)
                 if not two in cn: # distr (1, 1, 1,1) = c2
                     is_c2 = True
              elif len(d1) == 0 and len(d2) == 1:
                  # diags (0, 1), none of the checked sym's possible
                  # have it here for completeness
                  pass
              elif len(d1) == 1 and len(d2) == 0: # diags (1, 0)
                  # omit the one diag of e1 and check how the edges are
                  # distributed
                  right_axis = True
                  is_cs, is_c2 = helpfun(r1, e2)
              else: # diags (1, 1)
                  # check if the two edges in e1 are opposite to each other
                  # else we have one which is opposite to the edge in
                  # e2 and thus c2 symmetry (but with use of diagonals)
                  cn = distr(r1)
                  if two in cn:
                      is_c2 = True
       else: # else group sizes (2, 2)
           # first find out diagonal distribution
           d1, r1 = diag_edg(e1, edg)
           d2, r2 = diag_edg(e2, edg)
           if len(d1) == 0 and len(d2) == 0:
               # if there are no diagonals in e1, and e2
               # the symmetry is placed in the right direction
               right_axis = True
           # now let the helfun do the rest:
           is_cs, is_c2 = helpfun(e1, e2)


    elif len(es) == 3:
        # in this case we have to have three groups of two length (there
        # are six altogehter)

        # first identify which one have the diagonals
        diags = [[], [],[]]
        for i, e_s in enumerate(es):
           d2, r2 = diag_edg(e_s, edg)
           diags[len(d2)].append(i) # store which number has 0, 1 or two diagonals

        # there are two possible choices for the diagonal distribution:
        # (2, 0, 0) or (1, 1, 0) and its permutations
        if len(diags[0]) > 1: # (2, 0, 0)
            # the two with 0 gives us c2 or c2 geometry in wanted diag direction
            right_axis = True
            e1 = ei[diags[0,0]]
            e2 = ei[diags[0,1]]
        else: # (1, 1, 0)
            # there might be c2 geometry for the two with 1
            # in any case it is the wrong diag direction
            e1 = es[diags[1,0]]
            e2 = es[diags[1,1]]

        is_cs, is_c2 = helpfun(e1, e2) # check if it is cs or c2

    return is_cs, is_c2, right_axis

def helpfun(e1, e2):
    """
    Given are two 2-equal side functions
    check if they are in cs or c2 geometry if in any
    of them
    """
    zero = 0
    is_c2 = False
    is_cs = False
    all_e = [e1, e2]

    cn = distr(all_e) # needs (2, 2, 2,2) for both cs and c2
    if asarray([cn_i == 2 for cn_i in cn]).all():
       c1 = distr(e1) # decides if cs or c2
       # is a subset of cn
       if zero in c1: # (2, 1, 1, 0) = cs
           is_cs = True
       else: # (1, 1, 1, 1) = c2
           is_c2 = True
    return is_cs, is_c2

def diag_edg(e_s, edg):
     """
     Given a set e_s and a dictionary telling which sides
     are edges and which diagonals, it divides e_s in
     diags and edges (e_s is supposed to be a subset of the
     complete set of sides)
     """
     diag = []
     e_ed = []
     for ei in e_s:
         if not edg[ei[0] * 10 + ei[1]]:
            diag.append(ei)
         else:
             e_ed.append(ei)
     assert len(e_ed) + len(diag) == len(e_s)
     return diag, e_ed

def interprete_sym(es, edges):
     """
     Given which sides are equal, and which are
     supposed to be the preferred diagonals, find out
     which symmetry class we have
     """
     sym = 0 # C1 symmetry
     # sym = 0 will stay if there is no symmetry group found
     if es == []: # if there are no equal sides, this is useless
        return sym
     if test_td(es): # test for Td
         sym = 6 # Td
     elif test_c3(es): # test also for C3
         sym = 5 # C3

     has, sign = test_c2v(es, edges) # test for c2v (and C4v which is a subclass)
     # sign = False means that the mirror axis is not along a diagonal
     if has:
         if test_c4v(es): # test if also c4v
             sym = 4 # C4v
         else:
             sym = 3 # C2v
     else:
          # if there is no c2v geometry, it must not change the sign
          sign = True

     if sym == 0:
         # Test at the same time for cs and c2 geometry
         # Test only if there has not any other geometry found before
         # as c2v would be cs and c2 also, for example
         hascs, hasc2, sign = test_cs_and_c2(es, edges)
         if hascs:
              sym = 2 # Cs
         if hasc2:
              sym = 1 # C2

     if not sign: # symmetry does not be ordered around diagonals as expected
          sym *= -1

     return sym

def symmetry_of_picture(geom):
     """
     Given a geometry of an Ar4 Cluster (or similar) this routine
     searches for a given geometry to which symmetry group it belongs
     it will only check for: Td, C3, C4v, C2v, Cs, C2 and C1
     each symmetry is asociated with a number, two diagonals are stored
     beforehand, they will help to identify the symmeetry group and alow also
     to find out, if a symmetry group is orderd according to the wanted direction
     """
     global ar4_edges
     # equal_sides finds out which sides have the same length
     # the tolerances are given by global variables ta < te < tr
     esa, ese, esr = equal_sides(geom)
     syma = interprete_sym(esa, ar4_edges)
     syme = interprete_sym(ese, ar4_edges)
     symr = interprete_sym(esr, ar4_edges)
     return [syma, syme, symr]

def how_exact_picture_c2v(geom, diag):
    global ar4_sides

    exactness = []
    for j,i in enumerate(diag):
        if j == 0:
           ni = diag[1]
        else:
           ni = diag[0]
        e_sides = []
        for side in ar4_sides:
           if i in side and not ni in side:
              e_sides.append(side)
        assert len(e_sides) == 2
        len_1 = radii(geom, e_sides[0])
        len_2 = radii(geom, e_sides[1])
        exactness.append(abs(len_1 - len_2))

    max_exact = exactness[0]
    par1 = copy(diag)
    par1.sort()
    par2 = []
    for i in range(4):
        if not i+1 in diag:
           par2.append(i+1)
    pars = [par1, par2]
    for side in ar4_sides:
        if not side in pars:
           for side2 in ar4_sides:
               if not side2 in pars:
                   len_1 = radii(geom, side)
                   len_2 = radii(geom, side2)
                   if abs(len_1 - len_2) > max_exact:
                       max_exact = abs(len_1 - len_2)

    exactness.append(max_exact)
    return exactness

def how_exact_picture_td(geom):
    global ar4_sides

    max_exact = 0.0
    for side in ar4_sides:
           for side2 in ar4_sides:
                len_1 = radii(geom, side)
                len_2 = radii(geom, side2)
                if abs(len_1 - len_2) > max_exact:
                    max_exact = abs(len_1 - len_2)

    return max_exact


def main():
     global ar4_diag
     set_sides(ar4_diag)
     num = 25

     onlybeads = False
     geomsbeads = []
     geomspath = []
     bead_s = []
     path_s = []
     tol = 0
     pr = False
     pl_all = False
     pl_exact = False
     with_ase = False

     args = argv[1:]
     while args[0].startswith("--"):
        if args[0] == "--tol":
           tol = int(args[1])
           args = args[2:]
        elif args[0] == "--print":
           args = args[1:]
           pr = True
        elif args[0] == "--all":
           args = args[1:]
           pl_all = True
        elif args[0] == "--exact":
           args = args[1:]
           pl_exact = True
        elif args[0] == "--diag":
           par1 = [int(args[1]), int(args[2])]
           par2 = []
           for i in range(4):
               if not i+1 in par1:
                  par2.append(i+1)
           ar4_diag_new = (par1, par2)
           ar4_diag = ar4_diag_new
           set_sides(ar4_diag_new)
           args = args[3:]
        elif args[0]== "--onlybeads":
            args = args[1:]
            onlybeads = True
        elif args[0] in ["--ase", "--direct"]:
           args = args[1:]
           with_ase = True
        else:
           print "ERROR: unknown option:", args[0], "not known"
           exit()

     if with_ase:
         gb = [read(file) for file in args[:]]
         geomsbeads = [[g.get_positions() for g in gb]]
         bead_s = [[i for i in range(len(geomsbeads[0]))]]
         onlybeads = True
     else:
         for file in args[:]:
             xs, ys, obj = read_in_path(file)
             b1 = [obj.int2cart(y) for y in ys]
             geomsbeads.append(b1)
             bead_s.append(xs)
             p1, p_s = path_geos(xs, ys, obj, num)
             geomspath.append(p1)
             path_s.append(p_s)

     bead_syms = []
     if pl_exact:
         for bgeo in geomsbeads:
             sym1 = [how_exact_picture_c2v(geo1,ar4_diag[0]) for geo1 in bgeo]
             bead_syms.append(sym1)
     else:
         for bgeo in geomsbeads:
             sym1 = [symmetry_of_picture(geo1) for geo1 in bgeo]
             bead_syms.append(sym1)

     if not onlybeads:
         path_syms = []
         if pl_exact:
             for pgeo in geomspath:
                  sym1 = [how_exact_picture_c2v(geo1,ar4_diag[0]) for geo1 in pgeo]
                  path_syms.append(sym1)
         else:
             for pgeo in geomspath:
                  sym1 = [symmetry_of_picture(geo1) for geo1 in pgeo]
                  path_syms.append(sym1)

     if pr:
        print bead_s
        print bead_syms
        if not onlybeads:
            print path_s
            print path_syms
     else:

         pl = plot_tabs()
         for i in range(len(bead_s)):
             if pl_all:
                 for t in range(3):
                      plot_tol(pl, i, bead_s, bead_syms, path_s, path_syms, onlybeads, t)
             else:
                 plot_tol(pl, i, bead_s, bead_syms, path_s, path_syms, onlybeads, tol)

         pl.plot_data()

def plot_tol(pl, i, bead_s ,bead_syms, path_s, path_syms, onlybeads, tol):
    sb = asarray(bead_syms[i])
    beads = [bead_s[i], sb[:,tol]]
    beads = asarray(beads)
    path = None
    if not onlybeads:
        ps = asarray(path_syms[i])
        path = [path_s[i], ps[:,tol]]
        path = asarray(path)
    name_p = str(i+1) + " with tol " + str(tol) + " "
    pl.prepare_plot( path, name_p, beads, "_nolegend_", None, None, None)


if __name__ == "__main__":
     if argv[1] == "--help":
         print __doc__
         exit()
     main()
