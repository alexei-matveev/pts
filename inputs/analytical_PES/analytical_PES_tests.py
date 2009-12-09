"""Module to run tests on analytical potentials."""

import aof
from scipy.optimize.lbfgsb import fmin_l_bfgs_b
from scipy.optimize import fmin_bfgs
import ase
from numpy import array

def test_StaticModel(model, qc, reagents, N=8, k=None, alg='scipy_lbfgsb', tol=0.1, maxit=20):
    """Tests non-growing String and NEB on an analytical potential using a 
    particular minimiser.
    

    """

    growing = False
    if model == 'neb':
        CoS = aof.searcher.NEB(reagents, qc, k, beads_count=N)
    elif model == 'string':
        CoS = aof.searcher.GrowingString(reagents, qc, beads_count=N, growing=False)
    elif model == 'growingstring':
        CoS = aof.searcher.GrowingString(reagents, qc, beads_count=N, growing=True)
        growing = True
    else:
        print "Unrecognised model", model

    init_state = CoS.get_state_as_array()

    # Wrapper callback function
    def callback(x):
        print aof.common.line()
        print CoS
        print aof.common.line()
        return x

    def run_opt():
        if alg == 'scipy_lbfgsb':
            opt, energy, dict = fmin_l_bfgs_b(CoS.obj_func,
                                          CoS.get_state_as_array(),
                                          fprime=CoS.obj_func_grad,
                                          callback=callback,
                                          pgtol=tol,
                                          maxfun=maxit)
            print opt
            print dict

        elif alg == 'scipy_bfgs':
            opt = fmin_bfgs(CoS.obj_func,
                                          CoS.get_state_as_array(),
                                          fprime=CoS.obj_func_grad,
                                          callback=callback,
                                          gtol=tol,
                                          maxiter=maxit,
                                          disp=bool)
            print opt

        elif alg == 'ase_lbfgs':
            opt = ase.LBFGS(CoS)
    #        opt.attach(something)
            opt.run(fmax=tol)
            x_opt = CoS.state_vec
            print x_opt

        else:
            assert False, "Unrecognised algorithm " + alg

    run_opt()
    while growing and CoS.grow_string():
        run_opt()

    print CoS.ts_estims()


# python path_representation.py [-v]:
if __name__ == "__main__":
    reagents_MB = [array([ 0.62349942,  0.02803776]), array([-0.558, 1.442])]
#    test_StaticModel('string', aof.pes.MuellerBrown(), reagents_MB, 12, 1., 'ase_lbfgs', tol=0.001)
    test_StaticModel('growingstring', aof.pes.MuellerBrown(), reagents_MB, 8, 1., 'scipy_lbfgsb', tol=0.01)

    reagents = [array([0.,0.]), array([3.,3.])]
    test_StaticModel('neb', aof.pes.GaussianPES(), reagents, 8, 1., 'scipy_lbfgsb')
    test_StaticModel('string', aof.pes.GaussianPES(), reagents, 8, 1., 'scipy_lbfgsb')


# Default options for vim:sw=4:expandtab:smarttab:autoindent:syntax


