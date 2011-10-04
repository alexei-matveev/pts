from numpy import array, sqrt, dot, eye
from pts.metric import Default, Metric, Metric_reduced
from pts.dimer_rotate import rotate_dimer, rotate_dimer_mem
from numpy import zeros, dot
from scipy.linalg import eigh, eig
from copy import deepcopy
from sys import exit
from random import seed, random, shuffle
from pts.func import Func
"""
For modify slighlty the test, please uncommend another lam (=lambda) Series of
eigenvalues below or write your own set
"""

met = Default(None)

class pes_simple(Func):
    """
    Ideal PES for eigenvalue search of second derivative matrix H:
    Here first H is created, fprime = H * (x-x0)
    and f = (x-x0)^T * H * (x - x0)
    """
    def __init__(self, lambdas, start):
       """
       The eigenvalues are given as lambdas
       Then a pseudo-random (note the fixed seed for
       reproduction) Matrix V is generated, which is
       (enforced as symmetric matrix) used to generate
       eigenvectors. New H is these eigenvectors with the
       given lambdas as replacement for the eigenvalues
       """
       self.lambdas = lambdas
       self.H = eye(len(lambdas))
       for i, lam in enumerate(self.lambdas):
            self.H[i,i] = lam

       # keep it reroducable (at the moment)
       seed(11.2)

       # Random generating Matrix
       self.V = zeros(self.H.shape)
       for i in range(len(self.V)):
           for j in range(len(self.V[0])):
               self.V[i,j] = random()

       self.V = self.V / sqrt(dot(self.V, self.V))

       # We want orthogonalized Matrix, like EV
       # of a orthogonal matrix (this is none but
       # thread it as if was, thus having no complex
       # eigenvectors)
       __, self.V = eigh(self.V)

       self.H = dot(self.V.T, dot(self.H, self.V))

       self.start = start

    def eigs(self):
	# The correct eigenvales, eigenvectors without rounding
	# errors
        return self.lambdas, self.V

    def f(self, x1):
        x = x1 - self.start
        return dot(x.T, dot(self.H, x))

    def fprime(self, x1):
        x = x1 - self.start
        return dot(self.H, x)

# 1.: this is an easy example for the rotating methods as lambda_min == max|lambda|
#lam = [-339.6, -12.8, -4.3, 1.3, 1.4,1.5, 1.9, 2.98, 4.555, 10.,10., 13., 13.4, 25.89, 60.1, 70.2]
# 2.: Much harder: there are several eigenvalues of the same order of magnitute
#lam = [-33.6, -12.8, -4.3, 1.3, 1.4,1.5, 1.9, 2.98, 4.555, 10.,10., 13., 13.4, 25.89, 60.1, 70.2]
# 3.: Here max|lambda| > |lambda_min| : should make it harder for krylov methods
#lam = [-33.6, -1.8, -4.3, 1.3, 1.4,1.5, 1.9, 2.98, 4.555, 10.,10., 13., 13.4, 25.89, 600.1, 700.2]
# 4.: This one would fail with power method, we use power methods basis
#lam = [-70.2, -12.8, -4.3, 1.3, 1.4,1.5, 1.9, 2.98, 4.555, 10.,10., 13., 13.4, 25.89, 60.1, 70.2]
# 5.: Very hard: max|lambda| >> |lambda_min|
#lam = [-3.6, -1.8, -4.3, 1.3, 1.4,1.5, 1.9, 2.98, 4.555, 10.,10., 13., 13.4, 25.89, 600.1, 7000.2]
# 6.: Taking a much larger one:
#lam = [-3.6, -1.8, -4.3, 1.3, 1.4,1.5, 1.9, 2.98, 4.555, 10.,10., 13., 13.4, 25.89, 600.1, 7000.2, \
#       -300.7, 8.6, 999.9, 8.7, 8.6, 5.0003, - 0.03, -0.008, 0.05, 0.03, 985., 53., 33., 12., 888., \
#       12., 776., 345., 9., 6., 4., 4., -0.887, 643., -99.7, 0.34, 126., 60., 65., 43., 22., 111.]
# 7.: They around zero should be the hardest
lam = [3.6, 1.8, 4.3, 1.3, 1.4,1.5, 1.9, 2.98, 4.555, 10.,10., 13., 13.4, 25.89, 600.1, 7000.2, \
       300.7, 8.6, 999.9, 8.7, 8.6, 5.0003, 0.03, 0.008, 0.05, 0.03, 985., 53., 33., 12., 888., \
       12., 776., 345., 9., 6., 4., 4., 0.887, 643., 99.7, 0.34, 126., 60., 65., 43., 22., 111.]

print "using", len(lam), "degrees of freedom"

start = zeros(len(lam))
shuffle(lam)
pes = pes_simple(array(lam), start)

mode = zeros(len(start))
seed(34.23)
for i in range(len(start)):
    mode[i] = random()

mode = mode / sqrt(dot(mode, mode))
print "Cartesian Coordinates"
# rotate the dimer at position start, search for lowest mode
# First conjugate gradient reference implementation
curv1, n_mode1, dict = rotate_dimer(pes, start, pes.fprime(start), mode, met,
                           dimer_distance = 0.000001,
                           phi_tol = 1e-12, max_rotations = len(start) * 10)
print "Converged", dict["rot_convergence"]
print "Results:", dict

# New one with memorizing the results
curv2, n_mode2, dict2 = rotate_dimer_mem(pes, start, pes.fprime(start), mode, met,
                           dimer_distance = 0.000001,
                           phi_tol = 1e-12, max_rotations = len(start)-1 )
print "Converged", dict2["rot_convergence"]
print "Results:", dict2

#Real results from our "functions" storage
a1, V1 = pes.eigs()
m_s = a1.argmin()
a_min = a1[m_s]
V_min = V1[m_s]

# the second derivative at finish
d_v1 = min(dot(V_min - n_mode1, V_min - n_mode1), dot(V_min + n_mode1, V_min + n_mode1))
d_v2 = min(dot(V_min - n_mode2, V_min - n_mode2), dot(V_min + n_mode2, V_min + n_mode2))
print "Differences lowest EV-mode", d_v1, d_v2
print "Differences EV", a_min - curv1, a_min - curv2
a1 = list(a1)
a1.sort()
a1 = array(a1)
print "EIGENVALUES", a1
print "EV dimer", dict2["all_curvs"]
#diff = (abs(array(a1) - array(dict2["all_curvs"])))
#print "Differences", diff
print "EIGENMODE (real) 0", V_min
print "EIGENMODE (conj_grad) 0", n_mode1
print "EIGENMODE (lanczos) 0", n_mode2
