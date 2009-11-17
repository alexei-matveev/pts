from numpy import cos, sin, array, hstack, dot
from numpy.linalg import norm
import scipy.optimize as opt
import numpy as np

import ase

tmp = "2 0 0 1 1 2 8 6 7 3 5 4 6 5 3 8"
tmp = "2 0 0 1 7 3 8 6 4 8 5 4 6 5 3 7"
tmp = "2 0 0 1 1 2 7 3 8 6 4 8 5 4 6 5 3 7"


def my_main(order_str, fn1, fn2):

    # get re-ordering list
    order = order_str.split()
    order = array([int(o) for o in order])
    tot = len(order)
    assert tot % 2 == 0
    order = order.reshape(-1,2).tolist()

    a1 = ase.read(fn1)
    a2 = ase.read(fn2)
    assert len(a1) == len(a2)
    n = len(a1)

    # get coords
    geom1 = a1.get_positions()
    geom2 = a2.get_positions()


    g1 = []
    g2 = []
    chem_symbols = []
    for o in order:
        i,j = o
        g1.append(geom1[i])
        g2.append(geom2[j])
        print "ij", i, j
        assert a1.get_chemical_symbols()[i] == a2.get_chemical_symbols()[j]
        chem_symbols.append(a2.get_chemical_symbols()[j])

    g1 = array(g1)
    g2 = array(g2)

    a1.set_positions(g1)

    print g1
    print g2

    r = Rotate(g1.copy(), g2.copy(), [0,1,2])

    x0 = array([0.,0,0,0,0,.001])
    x = opt.fmin_bfgs(r.diff, x0)
    print x


    g2_new = r.trans(g2, x)
    print g2
    print g2_new
    a1.set_chemical_symbols(chem_symbols)
    a2.set_chemical_symbols(chem_symbols)

    a1.set_positions(g1)
    a2.set_positions(g2_new)

    #ase.view([a1,a2])
    ase.write(fn1 + '.t', a1, format='xyz')
    ase.write(fn2 + '.t', a2, format='xyz')




            


class Rotate:
    def __init__(self, geom1, geom2, ixs=None):

        if ixs == None:
            self.g1 = geom1
            self.g2 = geom2
        else:
            self.g1 = []
            self.g2 = []
            for i in ixs:
                self.g1.append(geom1[i])
                self.g2.append(geom2[i])


        
    def mat(self, v):
        assert len(v) == 3
        phi = norm(v)
        a = cos(phi/2)
        q2 = sin(phi/2) * v / phi

        b,c,d = q2

        m = np.array([[ a*a + b*b - c*c - d*d , 2*b*c + 2*a*d,         2*b*d - 2*a*c  ],
                   [ 2*b*c - 2*a*d         , a*a - b*b + c*c - d*d, 2*c*d + 2*a*b  ],
                   [ 2*b*d + 2*a*c         , 2*c*d - 2*a*b        , a*a - b*b - c*c + d*d  ]])

        return m

    def trans(self, geom, v):
        shift = v[:3] # displacement
        imq   = v[3:] # imag_quaternion

        mat = self.mat(imq)

        g_moved = []
        for g in geom:
            g_ = dot(mat, g)
            g_ += shift
            g_moved.append(g_)

        g_moved = array(g_moved)

        return g_moved

    def diff(self, x):
        g2_rot = self.trans(self.g2, x)
        diff = (self.g1 - g2_rot)**2
        s = sum(diff.flatten())
        print s
        return s

if __name__ == "__main__":
    my_main(tmp, "../inputs/cyclopropane.xyz", "../inputs/propylene.xyz")

