#!/usr/bin/python

import math    as M # works only with floats
import Numeric as O # Old predecessor of NumPy, deprecated, but recommended by Scientific.*
import numpy   as N # (mostly?) works in combination with Scientific.*

from Scientific.Functions.Derivatives import * # DerivVar constructor
from Scientific.Geometry import * # 3D Vector with overloaded arithmetics

a = 2
print "sin(2) = ",M.sin(a)
print "cos(2) = ",M.cos(a)
a = DerivVar(2)
print "Taylor of sin(2) = ",O.sin(a),"(by Numeric with DerivVar)"
print "Taylor of sin(2) = ",N.sin(a),"(by NumPy with DerivVar)"

# more convenient names:
angle  = Vector.angle
length = Vector.length

def distance(a,b):
  return length(a-b)

def rcorr(a,b,c):
  """Fake reaction coordinate, zero for equilateral triangle"""
  return distance(a,c) + distance(b,c) - 2 * distance(a,b)

def test(a,b,c):
  print "a=",a
  print "c=",c
  print "b=",b

  print "distance(a-b)=",distance(a,b)
  print "angle(a-c-b)=",angle(a-c,b-c)
  print "rcorr(a,b,c)=",rcorr(a,b,c)

a = Vector(0,0,1)
c = Vector(0,0,0)
b = Vector(1,0,0)

test(a,b,c)

# a modified copy eclipsing the function from Scientific.Functions.Derivative,
# ( there is a problem with import Vector in Ubuntu 8.10 version )
def DerivVector(x, y, z, index=0, order = 1):

    """Returns a vector whose components are DerivVar objects.

    Arguments:

    |x|, |y|, |z| -- vector components (numbers)

    |index| -- the DerivVar index for the x component. The y and z
               components receive consecutive indices.

    |order| -- the derivative order
    """

    if isDerivVar(x) and isDerivVar(y) and isDerivVar(z):
        return Vector(x, y, z)
    else:
        return Vector(DerivVar(x, index, order),
                      DerivVar(y, index+1, order),
                      DerivVar(z, index+2, order))

a = DerivVector(0,0,1, index=0) # independent vars 0,1,2
c = DerivVector(0,0,0, index=3) # independent vars 3,4,5
b = DerivVector(1,0,0, index=6) # independent vars 6,7,8

test(a,b,c)
