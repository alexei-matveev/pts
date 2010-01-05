#
# Library of optimisation tools for the transition state searching methods.
# Some borrowed from scipy and modified.
#

from lbfgsb import fmin_l_bfgs_b

from ase.optimize.sciopt import Converged, SciPyOptimizer
from ase.optimize import Optimizer

__all__ = filter(lambda s:not s.startswith('_'),dir())

class SciPyFminLBFGSB(SciPyOptimizer):

    """Quasi-Newton method (Limited Memory Broydon-Fletcher-Goldfarb-Shanno [B])"""
    def call_fmin(self, fmax, steps):
        output = fmin_l_bfgs_b(self.f,
                                self.x0(),
                                fprime=self.fprime,
                                #args=(), 
                                pgtol=fmax * 0.1, #Should never be reached
                                #norm=np.inf,
                                #epsilon=1.4901161193847656e-08, 
                                maxfun=steps,
                                #full_output=1, 
                                #disp=0,
                                #retall=0, 
                                callback=self.callback,
                                factr=10
                              )   


