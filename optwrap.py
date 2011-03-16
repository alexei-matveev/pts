"""Provides a uniform interface to a variety of optimisers."""

import os
import pickle

from pts.cosopt.lbfgsb import fmin_l_bfgs_b
from scipy.optimize import fmin_cg

import ase
import pts
from pts import MustRegenerate, MaxIterations, Converged
from pts.common import important

__all__ = ["opt"]

names = ['scipy_lbfgsb', 'ase_lbfgs', 'ase_fire', 'quadratic_string', 'ase_scipy_cg', 'ase_scipy_lbfgsb', 'ase_lbfgs_line', 'multiopt', 'ase_bfgs']


def record_event(cos, s):
    print important(s)
    if cos.arc_record:
        pickle.dump('Event: ' + s, cos.arc_record, protocol=2)

def runopt(name, CoS, ftol=0.1, xtol=0.03, etol=0.03, maxit=35, maxstep=0.2
                            , callback=None
                            , clean_after_grow=False
                            , **kwargs):
    assert name in names

    CoS.maxit = maxit

    # FIXME: we need an interface design for callbacks:
    def cb(x):
        if callback is not None:
            y = callback(x)
        else:
            y = None
        CoS.test_convergence(etol, ftol, xtol)
        return y

    while True:
        is_converged = False
        try:
            # tol and maxit are scaled so that they are never reached.
            # Convergence is tested via the callback function.
            # Exceeding maxit is tested during an energy/gradient call.
            runopt_inner(name, CoS, ftol*0.01, maxit*100, cb, maxstep=maxstep, **kwargs)
        except MustRegenerate:
            CoS.respace()
            record_event(CoS, "Optimisation RESTARTED (respaced)")
            continue

        except MaxIterations:
            record_event(CoS, "Optimisation STOPPED (maximum iterations)")
            break

        except Converged:
            s = "Optimisation Converged"
            # if only sustring is converged this variable is reset after enlarged string
            # is restarted
            is_converged = True
            record_event(CoS, s)

        if CoS.grow_string():
            if clean_after_grow:
                os.system('rm -r beadjob??') # FIXME: ugly hack

            s = "Optimisation RESTARTED (string grown)"
            record_event(CoS, s)
            continue
        else:
            break

    return is_converged

def runopt_inner(name, CoS, ftol, maxit, callback, maxstep=0.2, **kwargs):

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
        opt = pts.cosopt.MultiOpt(CoS, maxstep=maxstep, **kwargs)
        opt.string = CoS.string
        opt.attach(lambda: callback(None), interval=1)
        opt.run()
        x_opt = CoS.state_vec
        return None      
    elif name[0:4] == 'ase_':

        if name == 'ase_lbfgs':
            opt = ase.LBFGS(CoS, maxstep=maxstep, **kwargs)
        elif name == 'ase_bfgs':
            opt = ase.BFGS(CoS, maxstep=maxstep, **kwargs)

        elif name == 'ase_lbfgs_line':
            opt = ase.LineSearchLBFGS(CoS, maxstep=maxstep)
        elif name == 'ase_fire':
            opt = ase.FIRE(CoS)
        elif name == 'ase_scipy_cg':
            opt = ase.SciPyFminCG(CoS)
        elif name == 'ase_scipy_lbfgsb':
            opt = pts.cosopt.SciPyFminLBFGSB(CoS, alpha=400)
        else:
            assert False, ' '.join(["Unrecognised algorithm", name, "not in"] + names)

        opt.string = CoS.string

        # attach optimiser to print out each step in
        opt.attach(lambda: callback(None), interval=1)
        opt.run(fmax=ftol)
        x_opt = CoS.state_vec
        return None

    elif name == 'quadratic_string':
        gqs = pts.searcher.QuadraticStringMethod(CoS, callback=callback, update_trust_rads = True)
        opt = gqs.opt()
    else:
        assert False, ' '.join(["Unrecognised algorithm", name, "not in"] + names)

