#!/usr/bin/env python
"""
Run

    $ paratools transform-zmatrix <format> <file1> ..

Where the recognized original <format> of the zmatrix in the file can be any of the following:
  gx   : gxfile format of ParaGauss (recoginizes also if variables appear more than once)
  gauss: gaussian zmatrix format

"""
import sys
from pts.io.read_COS import read_zmt_from_gauss, read_zmt_from_gx

def main(args):
   if "--help" in args:
      print __doc__
      sys.exit()

   # User has to provide the format of his original zmatrix
   origin = args[0].lower()
   if origin in ["gx", "gxfile", "paragauss"]:
      which_origin = "gx"
   elif origin in ["gauss", "gaussian"]:
      which_origin = "gauss"
   else:
      print >> sys.stderr, "ERROR: unrecognized zmatrix format", origin
      print >> sys.stderr, "ERROR: aborting"
      sys.exit()

   for file in args[1:]:
      # Interprete zmatrix
      if which_origin == "gx":
         symbs, conns, var_names, mult, __, __, __ = read_zmt_from_gx(file)
      else:
         symbs, conns, var_names, mult, __, __, __ = read_zmt_from_gauss(file)

      # build the string of the ParaTools Zmatrix
      string = ""
      j = 0
      for s, con in zip(symbs, conns):
         string = string + "%-3s" % (s)
         for c in con:
             string = string + " %4i  %6s" % (c, var_names[j])
             j = j + 1
         string = string + "\n"

      print string

if __name__ == "__main__":
    main(sys.argv[1:])
