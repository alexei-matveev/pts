"""Module to run tests on analytical potentials."""

import pts
import ase
from numpy import array
from numpy.linalg import norm
import pts.metric as mt
from pts.cfunc import pass_through

def test_StaticModel(model, qc, reagents, N=8, k=None, alg='scipy_lbfgsb', tol=0.1, maxit=50, real_ts=None, plot='every'):
    """Tests non-growing String and NEB on an analytical potential using a 
    particular minimiser.

    model:
        string, 'neb', 'string' or 'growingstring'
    qc:
        PES object, must have gradient() and energy()
    reagents:
        list of reactant, optional guessed transition states, product
    N:
        number of beads
    k:
        spring constant
    alg:
        optimisation algorithm
    tol:
        tolerance on gradients
    maxit:
        maximum iterations in each sub-optimisation
    real_ts:
        known transition state, to calculate the different between the current
        best estimate and the real one.

    """

    growing = False
    if model == 'neb':
        CoS = pts.searcher.NEB(reagents, qc, k, beads_count=N)
    elif model == 'string':
        CoS = pts.searcher.GrowingString(reagents, qc, beads_count=N, growing=False)
    elif model == 'growingstring':
        CoS = pts.searcher.GrowingString(reagents, qc, beads_count=N, growing=True, max_sep_ratio=0.1, growth_mode='normal')
        growing = True
    elif model == 'searchingstring':
        CoS = pts.searcher.GrowingString(reagents, qc, beads_count=N, growing=True, max_sep_ratio=0.1, growth_mode='search')
        growing = True

    else:
        print "Unrecognised model", model

    mt.setup_metric(pass_through)
    init_state = CoS.get_state_as_array()

    # Wrapper callback function
    def callback(x):
        print pts.common.line()
        print CoS

        # Display the string/band
        if plot == 'every':
            p = pts.pes.Plot2D()
            p.plot(qc, CoS)
        print pts.common.line()
        pts.common.str2file(CoS.state_vec, "test.txt")

        # If we know the true transition state, compare it with the current 
        # best estimate after each iteration.
        if real_ts != None:
            tss = CoS.ts_estims()
            tss.sort()
            ts = tss[-1][-1]
            print "TS Error", norm(ts - real_ts)
        CoS.signal_callback()
        return x

    xtol = 0.0
    etol = 0.000001
    run_opt = lambda: pts.runopt(alg, CoS, tol, xtol, etol, maxit, maxstep=0.05, callback=callback, alpha=0.5)
    print run_opt()
    """while CoS.must_regenerate or growing and CoS.grow_string():
        CoS.update_path()
        print "Optimisation RESTARTED"
        print run_opt()"""

    print CoS.ts_estims()

    p = pts.pes.Plot2D()
    p.plot(qc, CoS)

# python path_representation.py [-v]:
if __name__ == "__main__":
    reagents_MB = [array([ 0.62349942,  0.02803776]), array([-0.558, 1.442])]
#    reagents_MB = eval(pts.common.file2str("initial_pathway.txt"))
#    reagents_MB = array([[ 0.62349942,  0.02803776], [-0.82200156,  0.62431281], [-0.558     ,  1.442     ]])
#    test_StaticModel('neb', pts.pes.MuellerBrown(), reagents_MB, 8, 3., 'ase_lbfgs', tol=0.001, plot='every')
#    exit()

    MB_saddle1 = array([ 0.21248659,  0.29298832]) # energy = -0.072248940112325243
    MB_saddle2 = array([-0.82200156,  0.62431281]) # energy = -0.040664843508657414

    test_StaticModel('searchingstring', pts.pes.MuellerBrown(), reagents_MB, 6, 2., 'multiopt', tol=0.005, maxit=200, real_ts=MB_saddle2, plot='every')

    exit()
    reagents = [array([0.,0.]), array([3.,3.])]
    test_StaticModel('neb', pts.pes.GaussianPES(), reagents, 8, 1., 'scipy_lbfgsb')
    test_StaticModel('string', pts.pes.GaussianPES(), reagents, 8, 1., 'scipy_lbfgsb')


# Default options for vim:sw=4:expandtab:smarttab:autoindent:syntax


