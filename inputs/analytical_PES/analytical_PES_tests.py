"""Module to run tests on analytical potentials."""

import aof
import ase
from numpy import array

def test_StaticModel(model, qc, reagents, N=8, k=None, alg='scipy_lbfgsb', tol=0.1, maxit=50):
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
        p = aof.pes.Plot2D()
        p.plot(qc, CoS)
        print aof.common.line()
        aof.common.str2file(CoS.state_vec, "test.txt")
        return x

    run_opt = lambda: aof.runopt(alg, CoS, tol, maxit, callback, maxstep=0.1)
    print run_opt()
    while CoS.must_regenerate or growing and CoS.grow_string():
        CoS.update_path()
        print "Optimisation RESTARTED"
        print run_opt()

    print CoS.ts_estims()

# python path_representation.py [-v]:
if __name__ == "__main__":
    reagents_MB = [array([ 0.62349942,  0.02803776]), array([-0.558, 1.442])]
#    reagents_MB = eval(aof.common.file2str("test.txt"))
#    test_StaticModel('string', aof.pes.MuellerBrown(), reagents_MB, 12, 1., 'ase_lbfgs', tol=0.001)
    test_StaticModel('string', aof.pes.MuellerBrown(), reagents_MB, 11, 2., 'scipy_lbfgsb', tol=0.01, maxit=50)

    exit()
    reagents = [array([0.,0.]), array([3.,3.])]
    test_StaticModel('neb', aof.pes.GaussianPES(), reagents, 8, 1., 'scipy_lbfgsb')
    test_StaticModel('string', aof.pes.GaussianPES(), reagents, 8, 1., 'scipy_lbfgsb')


# Default options for vim:sw=4:expandtab:smarttab:autoindent:syntax


