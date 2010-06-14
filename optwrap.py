"""Provides a uniform interface to a variety of optimisers."""

from aof.cosopt.lbfgsb import fmin_l_bfgs_b
from scipy.optimize import fmin_cg

import ase
import aof
from aof import MustRegenerate, MaxIterations, Converged
from aof.common import important

__all__ = ["opt"]

names = ['scipy_lbfgsb', 'ase_lbfgs', 'ase_fire', 'quadratic_string', 'ase_scipy_cg', 'ase_scipy_lbfgsb', 'ase_lbfgs_line', 'multiopt', 'ase_bfgs']

def runopt(name, CoS, ftol, xtol, etol, maxit, callback, maxstep=0.2, extra=dict()):
    assert name in names

    CoS.maxit = maxit
    def cb(x):
        y = callback(x)
        CoS.test_convergence(etol, ftol, xtol)
        return y

    while True:
        try:
            # tol and maxit are scaled so that they are never reached.
            # Convergence is tested via the callback function.
            # Exceeding maxit is tested during an energy/gradient call.
            runopt_inner(name, CoS, ftol*0.01, maxit*100, cb, extra, maxstep=maxstep)
        except MustRegenerate:
            CoS.update_path()
            print important("Optimisation RESTARTED (respaced)")
            continue

        except MaxIterations:
            print important("Optimisation STOPPED (maximum iterations)")
            break

        except Converged:
            print important("Optimisation Converged")

        if CoS.grow_string():
            print important("Optimisation RESTARTED (string grown)")
            continue
        else:
            break

def runopt_inner(name, CoS, ftol, maxit, callback, extra, maxstep=0.2):

    if name == 'scipy_lbfgsb':
        opt, energy, dict = fmin_l_bfgs_b(CoS.obj_func,
                                  CoS.get_state_as_array(),
                                  fprime=CoS.obj_func_grad,
                                  callback=callback,
                                  pgtol=ftol,
                                  factr=10, # stops when step is < factr*machine_precision
                                  maxstep=maxstep)
        return dict

    elif name == 'multiopt':
        opt = aof.cosopt.MultiOpt(CoS, maxstep=maxstep, **extra)
        opt.string = CoS.string
        opt.attach(lambda: callback(None), interval=1)
        opt.run(fmax=ftol)
        x_opt = CoS.state_vec
        return None      
    elif name[0:4] == 'ase_':

        if name == 'ase_lbfgs':
            opt = ase.LBFGS(CoS, maxstep=maxstep, **extra)
        elif name == 'ase_bfgs':
            opt = ase.BFGS(CoS, maxstep=maxstep, **extra)

        elif name == 'ase_lbfgs_line':
            opt = ase.LineSearchLBFGS(CoS, maxstep=maxstep)
        elif name == 'ase_fire':
            opt = ase.FIRE(CoS)
        elif name == 'ase_scipy_cg':
            opt = ase.SciPyFminCG(CoS)
        elif name == 'ase_scipy_lbfgsb':
            opt = aof.cosopt.SciPyFminLBFGSB(CoS, alpha=400)
        else:
            assert False, ' '.join(["Unrecognised algorithm", name, "not in"] + names)

        opt.string = CoS.string

        # attach optimiser to print out each step in
        opt.attach(lambda: callback(None), interval=1)
        opt.run(fmax=ftol)
        x_opt = CoS.state_vec
        return None

    elif name == 'quadratic_string':
        gqs = aof.searcher.QuadraticStringMethod(CoS, callback=callback, update_trust_rads = True)
        opt = gqs.opt()
    else:
        assert False, ' '.join(["Unrecognised algorithm", name, "not in"] + names)

