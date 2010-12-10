import time
import random

import numpy as np

import mueller_brown as mb
from plot import Plot2D
import pts
from copy import deepcopy
import pts.metric as mt
from pts.metric import setup_metric

class QCDriver:
    def __init__(self, dimension):
        self.dimension = dimension
        self.__g_calls = 0
        self.__e_calls = 0

    def get_calls(self):
        return (self.__e_calls, self.__g_calls)

    def gradient(self):
        self.__g_calls += 1
        pass

    def energy(self):
        self.__e_calls += 1
        pass

class MuellerBrown():
    """
    >>> mb = MuellerBrown()

    >>> mb.gradient([1.,2.])
    array([ 5608.10419613,  4287.36943935])

    >>> mb.energy([1.,2.])
    1649.1505581288852
    """
    def energy(self, v):
        return mb.energy(v) * 0.001

    def gradient(self, v):
        return mb.gradient(v) * 0.001

class QuarticPES(QCDriver):
    def __init__(self):
        QCDriver.__init__(self,2)

    def gradient(self, a):
        QCDriver.gradient(self)

        if len(a) != self.dimension:
            raise QCDriverException("Wrong dimension")

        x = a[0]
        y = a[1]
        dzdx = 4*x**3 - 3*80*x**2 + 2*1616*x + 2*2*x*y**2 - 2*8*y*x - 80*y**2 
        dzdy = 2*2*x**2*y - 8*x**2 - 2*80*x*y + 2*1616*y + 4*y**3 - 3*8*y**2
        return np.array([dzdy, dzdx])

    def energy(self, a):
        QCDriver.energy(self)

        if len(a) != self.dimension:
            raise Exception("Wrong dimension")

        x = a[0]
        y = a[1]
        z = (x**2 + y**2) * ((x - 40)**2 + (y - 4) ** 2)
        return (z)

class GaussianPES():
    def __init__(self, fake_delay=None):
        """
        fake_delay:
            float, average number of seconds to wait after running a job, just 
            used in testing to simulate more cpu intensive jobs

        >>> x = GaussianPES()
        >>> x.energy([0., 1.0])
        """
        self.fake_delay = fake_delay
        # we need an metric object, this one should
        # lower and raise indices without changing anything
        def identity(x):
            return deepcopy(x)
        setup_metric(identity)
        mt.metric.version()

    def __str__(self):
        return "GaussianPES"

    def energy(self, v):

        x = v[0]
        y = v[1]

        if self.fake_delay:
            time.sleep(2*self.fake_delay*random.random())

        return (-np.exp(-(x**2 + y**2)) - np.exp(-((x-3)**2 + (y-3)**2)) + 0.01*(x**2+y**2) - 0.5*np.exp(-((1.5*x-1)**2 + (y-2)**2)))

    def gradient(self, v):

        x = v[0]
        y = v[1]
        dfdx = 2*x*np.exp(-(x**2 + y**2)) + (2*x - 6)*np.exp(-((x-3)**2 + (y-3)**2)) + 0.02*x + 0.5*(4.5*x-3)*np.exp(-((1.5*x-1)**2 + (y-2)**2))
        dfdy = 2*y*np.exp(-(x**2 + y**2)) + (2*y - 6)*np.exp(-((x-3)**2 + (y-3)**2)) + 0.02*y + 0.5*(2*y-4)*np.exp(-((1.5*x-1)**2 + (y-2)**2))
        if self.fake_delay:
            time.sleep(2*self.fake_delay*random.random())

        g = np.array((dfdx,dfdy))
        return g

    def run(self, i):
        j = i.job
        e = self.energy(j.v)
        g = self.gradient(j.v)

        return pts.common.Result(j.v, e, g)

class PlanePES():
    def energy(self, v):
        x = v[0]
        y = v[1]
        return x - y

    def gradient(self, v):
        x = v[0]
        y = v[1]
        dfdx = 1
        dfdy = -1
        g = np.array((dfdx,dfdy))
        return g

class PlanePES2():
    def energy(self, v):
        x = v[0]
        y = v[1]
        return x + y

    def gradient(self, v):
        x = v[0]
        y = v[1]
        dfdx = 1
        dfdy = 1
        g = np.array((dfdx,dfdy))
        return g


class GaussianPES2(QCDriver):
    def __init__(self):
        QCDriver.__init__(self,2)

    def energy(self, v):
        QCDriver.energy(self)

        x = v[0]
        y = v[1]
        return (-exp(-(x**2 + 0.2*y**2)) - exp(-((x-3)**2 + (y-3)**2)) + 0.01*(x**2+y**2) - 0.5*exp(-((x-1.5)**2 + (y-2.5)**2)))

    def gradient(self, v):
        QCDriver.gradient(self)

        x = v[0]
        y = v[1]
        dfdx = 2*x*exp(-(x**2 + 0.2*y**2)) + (2*x - 6)*exp(-((x-3)**2 + (y-3)**2)) + 0.02*x + 0.5*(2*x-3)*exp(-((x-1.5)**2 + (y-2.5)**2))
        dfdy = 2*y*exp(-(x**2 + 0.2*y**2)) + (2*y - 6)*exp(-((x-3)**2 + (y-3)**2)) + 0.02*y + 0.3*(2*y-5)*exp(-((x-1.5)**2 + (y-2.5)**2))

        return np.array((dfdx,dfdy))

class FourWellPot(QCDriver):
    """From "Dey, Janicki, and Ayers, J. Chem. Phys., Vol. 121, No. 14, 8 October 2004" """
    def __init__(self):
        QCDriver.__init__(self,2)

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


    def energy(self, v):
        QCDriver.energy(self)

        x, y = v[0], v[1]

        def f_well(args):
            a, sx, sy, alpha, beta = args
            return a * exp(-sx * (x-alpha)**2 -sy * (y-beta)**2)
        
        e = self.v0 + self.a0*exp(-(x-self.b1)**2 -(y-self.b2)**2) \
            - sum (map (f_well, self.params_list))

        return e
        
    def gradient(self, v):
        QCDriver.gradient(self)

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

