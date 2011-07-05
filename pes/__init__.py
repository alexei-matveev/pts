import time
import random
import numpy as np
from mueller_brown import energy as mbenergy
from mueller_brown import gradient as mbgradient
from pts.func import Func

class MuellerBrown(Func):
    """
    >>> mb = MuellerBrown()

    >>> mb.fprime([1.,2.])
    array([ 5608.10419613,  4287.36943935])

    >>> mb.f([1.,2.])
    1649.1505581288852
    """
    def f(self, v):
        return mbenergy(v) * 0.001

    def fprime(self, v):
        return mb.gradient(v) * 0.001

class QuarticPES(Func):
    def __init__(self):
        pass

    def fprime(self, a):

        if len(a) != 2:
            raise Exception("Wrong dimension")

        x = a[0]
        y = a[1]
        dzdx = 4*x**3 - 3*80*x**2 + 2*1616*x + 2*2*x*y**2 - 2*8*y*x - 80*y**2 
        dzdy = 2*2*x**2*y - 8*x**2 - 2*80*x*y + 2*1616*y + 4*y**3 - 3*8*y**2
        return np.array([dzdy, dzdx])

    def f(self, a):

        if len(a) != 2:
            raise Exception("Wrong dimension")

        x = a[0]
        y = a[1]
        z = (x**2 + y**2) * ((x - 40)**2 + (y - 4) ** 2)
        return (z)

class GaussianPES(Func):
    def __init__(self, fake_delay=None):
        """
        fake_delay:
            float, average number of seconds to wait after running a job, just 
            used in testing to simulate more cpu intensive jobs

        >>> x = GaussianPES()
        >>> x.energy([0., 1.0])
        """
        self.fake_delay = fake_delay

    def __str__(self):
        return "GaussianPES"

    def f(self, v):

        x = v[0]
        y = v[1]

        if self.fake_delay:
            time.sleep(2*self.fake_delay*random.random())

        return (-np.exp(-(x**2 + y**2)) - np.exp(-((x-3)**2 + (y-3)**2)) + 0.01*(x**2+y**2) - 0.5*np.exp(-((1.5*x-1)**2 + (y-2)**2)))

    def fprime(self, v):

        x = v[0]
        y = v[1]
        dfdx = 2*x*np.exp(-(x**2 + y**2)) + (2*x - 6)*np.exp(-((x-3)**2 + (y-3)**2)) + 0.02*x + 0.5*(4.5*x-3)*np.exp(-((1.5*x-1)**2 + (y-2)**2))
        dfdy = 2*y*np.exp(-(x**2 + y**2)) + (2*y - 6)*np.exp(-((x-3)**2 + (y-3)**2)) + 0.02*y + 0.5*(2*y-4)*np.exp(-((1.5*x-1)**2 + (y-2)**2))
        if self.fake_delay:
            time.sleep(2*self.fake_delay*random.random())

        g = np.array((dfdx,dfdy))
        return g

class PlanePES(Func):
    def f(self, v):
        x = v[0]
        y = v[1]
        return x - y

    def fprime(self, v):
        x = v[0]
        y = v[1]
        dfdx = 1
        dfdy = -1
        g = np.array((dfdx,dfdy))
        return g

class PlanePES2(Func):
    def f(self, v):
        x = v[0]
        y = v[1]
        return x + y

    def fprime(self, v):
        x = v[0]
        y = v[1]
        dfdx = 1
        dfdy = 1
        g = np.array((dfdx,dfdy))
        return g


class GaussianPES2(Func):
    def __init__(self):
        pass

    def f(self, v):

        x = v[0]
        y = v[1]
        return (-exp(-(x**2 + 0.2*y**2)) - exp(-((x-3)**2 + (y-3)**2)) + 0.01*(x**2+y**2) - 0.5*exp(-((x-1.5)**2 + (y-2.5)**2)))

    def fprime(self, v):

        x = v[0]
        y = v[1]
        dfdx = 2*x*exp(-(x**2 + 0.2*y**2)) + (2*x - 6)*exp(-((x-3)**2 + (y-3)**2)) + 0.02*x + 0.5*(2*x-3)*exp(-((x-1.5)**2 + (y-2.5)**2))
        dfdy = 2*y*exp(-(x**2 + 0.2*y**2)) + (2*y - 6)*exp(-((x-3)**2 + (y-3)**2)) + 0.02*y + 0.3*(2*y-5)*exp(-((x-1.5)**2 + (y-2.5)**2))

        return np.array((dfdx,dfdy))

class FourWellPot(Func):
    """From "Dey, Janicki, and Ayers, J. Chem. Phys., Vol. 121, No. 14, 8 October 2004" """
    def __init__(self):

        self.v0 = 4.0
        self.a0 = 0.6
        self.b1 = 0.1
        self.b2 = 0.1
        ais = 2.0 * ones(4)
        sxs = [0.3, 1.0, 0.4, 1.0]
        sys = [0.4, 1.0, 1.0, 0.1]
        alphas = [1.3, -1.5, 1.4, -1.3]
        betas = [-1.6, -1.7, 1.8, 1.23]

        self.params_list = zip(ais, sxs, sys, alphas, betas)


    def f(self, v):

        x, y = v[0], v[1]

        def f_well(args):
            a, sx, sy, alpha, beta = args
            return a * exp(-sx * (x-alpha)**2 -sy * (y-beta)**2)
        
        e = self.v0 + self.a0*exp(-(x-self.b1)**2 -(y-self.b2)**2) \
            - sum (map (f_well, self.params_list))

        return e
        
    def fprime(self, v):

        x, y = v[0], v[1]

        def df_welldx(args):
            a, sx, sy, alpha, beta = args
            return a * (-sx * (2*x-2*alpha)) * exp(-sx * (x-alpha)**2 -sy * (y-beta)**2)

        def df_welldy(args):
            a, sx, sy, alpha, beta = args
            return a * (-sy * (2*y-2*beta)) * exp(-sx * (x-alpha)**2 -sy * (y-beta)**2)
       
        dedx = -(2*x - 2*self.b1)*self.a0*exp(-(x-self.b1)**2 -(y-self.b2)**2) \
            - sum (map (f_welldx, self.params_list))

        dedy = -(2*y - 2*self.b2)*self.a0*exp(-(x-self.b1)**2 -(y-self.b2)**2) \
            - sum (map (f_welldy, self.params_list))

        return (dedx, dedy)

if __name__ == "__main__":
    import doctest
    doctest.testmod()

