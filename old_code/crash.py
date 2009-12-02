import openbabel
import numpy

from scipy.optimize.lbfgsb import fmin_l_bfgs_b

def f(x):
    return x[0]**2 + x[1]**2

def g(x):
    return numpy.array([2*x[0], 2*x[1]])

x0 = numpy.array([3,1])

opt, energy, dict = fmin_l_bfgs_b(f, x0, fprime=g)

print opt
