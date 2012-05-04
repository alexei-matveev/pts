"""Provides a uniform interface to a variety of optimisers."""

import os
import pickle

from copy import copy
import pts.cosopt.lbfgsb as lbfgs

import ase
import pts
from pts import MustRegenerate, Converged
from pts.common import important

names = ['scipy_lbfgsb', 'ase_lbfgs', 'ase_fire', 'ase_scipy_cg', 'ase_scipy_lbfgsb', 'ase_lbfgs_line', 'multiopt', 'ase_bfgs', 'conj_grad', 'steep_des', 'fire']


def record_event(cos, s):
    print important(s)
    if cos.arc_record:
        pickle.dump('Event: ' + s, cos.arc_record, protocol=2)


def runopt(name, CoS, ftol=0.1, xtol=0.03, etol=0.03, maxit=35, maxstep=0.2
                            , callback=None
                            , clean_after_grow=False
                            , **kwargs):
    assert name in names, names

    global opt
    opt = None
    CoS.maxit = maxit
    max_it = copy(maxit)

    # FIXME: we need an interface design for callbacks:
    def cb(x):
        if callback is not None:
            y = callback(x)
        else:
            y = None
        CoS.test_convergence(etol, ftol, xtol)
        return y

    def runopt_inner(name, CoS, ftol, maxit, callback, maxstep=0.2, **kwargs):

        global opt

        if name == 'scipy_lbfgsb':
            def fun(x):
               CoS.state_vec = x
               return CoS.obj_func()

            def fprime(x):
               CoS.state_vec = x
               # Attention: here it is expected that only one
               # gradient call per iteration step is done
               return CoS.obj_func_grad()

            class nums:
                def __init__(self):
                    pass

                def get_number_of_steps(self):
                    return lbfgs.n_function_evals

            opt = nums()

            opt2, energy, dict = lbfgs.fmin_l_bfgs_b(fun,
                                      CoS.get_state_as_array(),
                                      fprime=fprime,
                                      callback=callback,
                                      maxfun = maxit,
                                      pgtol=ftol,
                                      factr=10, # stops when step is < factr*machine_precision
                                      maxstep=maxstep)
            return dict

        elif name == 'multiopt':
            opt = pts.cosopt.MultiOpt(CoS, maxstep=maxstep, **kwargs)
            opt.string = CoS.string
            opt.attach(lambda: callback(None), interval=1)
            opt.run(steps = max_it) # convergence handled by callback
            return None
        elif name == 'conj_grad':
            from pts.cosopt.conj_grad import conj_grad_opt
            opt = conj_grad_opt(CoS, maxstep=maxstep, **kwargs)
            opt.attach(lambda: callback(None), interval=1)
            opt.run(steps = max_it) # convergence handled by callback
            return None
        elif name == 'steep_des':
            from pts.cosopt.conj_grad import conj_grad_opt
            opt = conj_grad_opt(CoS, maxstep=maxstep, reduce_to_steepest_decent = True, **kwargs)
            opt.attach(lambda: callback(None), interval=1)
            opt.run() # convergence handled by callback
            return None
        elif name == 'fire':
            from pts.cosopt.fire import fire_opt
            opt = fire_opt(CoS, maxstep=maxstep, **kwargs)
            opt.attach(lambda: callback(None), interval=1)
            opt.run(steps = maxit) # convergence handled by callback
            return None
        elif name[0:4] == 'ase_':

            if name == 'ase_lbfgs':
                opt = ase.LBFGS(CoS, maxstep=maxstep, **kwargs)
            elif name == 'ase_bfgs':
                opt = ase.BFGS(CoS, maxstep=maxstep, **kwargs)

            elif name == 'ase_lbfgs_line':
                opt = ase.LineSearchLBFGS(CoS, maxstep=maxstep)
            elif name == 'ase_fire':
                opt = ase.FIRE(CoS, maxmove=maxstep)
            elif name == 'ase_scipy_cg':
                opt = ase.SciPyFminCG(CoS)
            elif name == 'ase_scipy_lbfgsb':
                opt = pts.cosopt.SciPyFminLBFGSB(CoS, alpha=400)
            else:
                assert False, ' '.join(["Unrecognised algorithm", name, "not in"] + names)

            opt.string = CoS.string

            # attach optimiser to print out each step in
            opt.attach(lambda: callback(None), interval=1)
            opt.run(fmax=ftol, steps = maxit)
            return None

        else:
            assert False, ' '.join(["Unrecognised algorithm", name, "not in"] + names)

        #End of runopt_inner


    while True:
        is_converged = False
        try:
            # tol and maxit are scaled so that they are never reached.
            # Convergence is tested via the callback function.
            # Exceeding maxit is tested during an energy/gradient call.
            runopt_inner(name, CoS, ftol*0.01, max_it, cb, maxstep=maxstep, **kwargs)
            record_event(CoS, "Optimisation STOPPED (optimizer reached maximum iterations)")
            it = opt.get_number_of_steps()
            max_it = max_it - it - 1
            break
        except MustRegenerate:
            CoS.respace()
            record_event(CoS, "Optimisation RESTARTED (respaced)")
            it = opt.get_number_of_steps()
            max_it = max_it - it - 1
            continue

        except Converged:
            s = "Optimisation Converged"
            # if only sustring is converged this variable is reset after enlarged string
            # is restarted
            is_converged = True
            record_event(CoS, s)
            it = opt.get_number_of_steps()
            max_it = max_it - it - 1

        if CoS.grow_string():
            if clean_after_grow:
                os.system('rm -r beadjob??') # FIXME: ugly hack

            s = "Optimisation RESTARTED (string grown)"
            record_event(CoS, s)
            continue
        else:
            break

    return is_converged


