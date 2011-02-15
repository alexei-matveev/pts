from numpy import asarray, sqrt
from pts.func import Func
from math import cos, sin

class diagsandhight(Func):
     """
     Function diagsandhight: generates Cartesian coordinates (and the derivatives)
     from given values for the two diagonals and the high between them.
     The First two atoms will be situated on the X-Axis, equal far away from O.
     The other two atoms will have the same on the Y-Axis, but they are shifted in z-direction
     about hight.

     >>> fun = diagsandhight()
     >>> from pts.func import NumDiff
     >>> from numpy import max, abs

     Verify the correctness of the analytical derivative:
     >>> fun2 = NumDiff(diagsandhight())
     >>> t = asarray([1., 1., 0.7])
     >>> max(abs(fun.fprime(t) - fun2.fprime(t))) < 1e-12
     True
     """
     def __init__(self):
         """
         The derivatives are always the same, thus store them
         once.
         """
         self.tmat = asarray([[[ 0.5,   0., 0.],
                               [  0.,   0., 0.],
                               [  0.,   0., 0.]],
                              [[-0.5,   0., 0.],
                               [  0.,   0., 0.],
                               [  0.,   0., 0.]],
                              [[  0.,   0., 0.],
                               [  0.,  0.5, 0.],
                               [  0.,   0., 1.]],
                              [[  0.,   0., 0.],
                               [  0., -0.5, 0.],
                               [  0.,   0., 1.]]])


     def f(self, vec):
         """
         Function gets d1, d2 and h and calculates Cartesian coordinates
         """
         d1 = float(vec[0]) # diagonal with small changes
         d2 = float(vec[1]) # diagonal with large changes
         h = float(vec[2])  # hight of two last atoms in z-direction

         return asarray([[d1 / 2., 0., 0.], [-d1 / 2., 0., 0.], [0., d2 / 2., h], [0., -d2 / 2., h]])

     def fprime(self, vec):
         """
         The derivatives are constant
         """
         return self.tmat


# python func.py [-v]:
if __name__ == "__main__":
    import doctest
    doctest.testmod()

