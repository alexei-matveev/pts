#!/usr/bin/python

#from scipy import *
import sys
import inspect

import scipy.integrate
from scipy.optimize import fminbound

from scipy import interpolate
import tempfile, os
import logging
from copy import deepcopy
import pickle

from numpy import linalg, floor, zeros, array, ones, arange, arccos, hstack, ceil, abs, ndarray, sqrt, column_stack, dot, eye, outer, inf, isnan, isfinite, size, vstack, atleast_1d

from path import Path
from func import CubicFunc

from common import * # TODO: must unify
import aof.common as common
import aof


lg = logging.getLogger("aof.searcher")
lg.setLevel(logging.INFO)

if not globals().has_key("lg"):
    sh = logging.StreamHandler()
    sh.setLevel(logging.DEBUG)
    lg.addHandler(sh)

def _functionId(nFramesUp):
    """ Create a string naming the function n frames up on the stack.
    """
    import sys
    co = sys._getframe(nFramesUp+1).f_code
    return "%s (%s @ %d)" % (co.co_name, co.co_filename, co.co_firstlineno)


# Function labeller
"""def flab(*args):
    import sys
    args = [str(a) for a in args]
    lg.info("**** " + sys._getframe(1).f_code.co_name + ' '.join(args))"""

class ReactionPathway:
    """Abstract object for chain-of-state reaction pathway."""
    dimension = -1

    eg_calls = 0
    bead_eg_calls = 0

    # Set to true by GrowingString sub-class to indicate that bead spacing 
    # threshold has been exceeded. TODO: can I get rid of this now?
    must_regenerate = False

    def __init__(self, reagents, beads_count, qc_driver, parallel, reporting=None):

        self.parallel = parallel
        self.qc_driver = qc_driver
        self.beads_count = beads_count

        self.__dimension = len(reagents[0])
        #TODO: check that all reagents are same length

        # forces perpendicular to pathway
        self.perp_bead_forces = zeros(beads_count * self.dimension).reshape(beads_count,-1)
        self.para_bead_forces = []#zeros(beads_count * self.dimension).reshape(beads_count,-1)

#        self.bead_pes_energies = deepcopy(self.default_initial_bead_pes_energies)

        self.tangents = zeros(beads_count * self.dimension)
        self.tangents.shape = (beads_count, self.dimension)

        # energies / gradients of beads, excluding any spring forces / projections
        self.bead_pes_energies = zeros(beads_count)
        self.bead_pes_gradients = zeros(beads_count * self.dimension)
        self.bead_pes_gradients.shape = (beads_count, self.dimension)

        self.prev_state = None
        self.prev_perp_forces = None
        self.prev_para_forces = None
        self.prev_energies = None

        # TODO: can the following be removed?
        self.history = []
        self.energy_history = []
        self.ts_history = []

        # mask of gradients to update at each position
        self.grad_update_mask = [False] + [True for i in range(beads_count-2)] + [False]

        self._maxit = sys.maxint
        if reporting:
            assert type(reporting) == file
        self.reporting = reporting

    def test_convergence(self, tol):
        """
        Raises Converged if converged, applying weaker convergence 
        criterion if growing string is not fully grown.

        TODO: this should probably be called via the optimiser at some point.
        """

        if self.eg_calls == 0:
            return
        elif self.growing and not self.grown() and self.rmsf < tol*10:
            raise aof.Converged
        elif self.rmsf_perp < tol:
            raise aof.Converged

    @property
    def rmsf_perp(self):
        """RMS forces, not including those of end beads."""
        return common.rms(self.perp_bead_forces[1:-1]), [common.rms(f) for f in self.perp_bead_forces]

    @property
    def rmsf_para(self):
        """RMS forces, not including those of end beads."""
        return common.rms(self.para_bead_forces), [common.rms(f) for f in self.para_bead_forces]

    @property
    def step(self):
        """RMS forces, not including those of end beads."""
        if self.prev_state == None:
            return 0., [0. for i in self.state_vec]
        step = self.state_vec - self.prev_state
        return common.rms(step), [common.rms(s) for s in step]

    @property
    def energies(self):
        """RMS forces, not including those of end beads."""
        return common.rms(self.bead_pes_energies), [common.rms(s) for s in self.bead_pes_energies]


    @property
    def state_summary(self):
        s = common.vec_summarise(self.state_vec)
        s_beads = [common.vec_summarise(b) for b in self.state_vec]
        return s, s_beads

    def __str__(self):
        e_total, e_beads = self.energies
        rmsf_perp_total, rmsf_perp_beads = self.rmsf_perp
        rmsf_para_total, rmsf_para_beads = self.rmsf_para

        eg_calls = self.eg_calls

        step_total, step_beads = self.step

        angles = self.angles
        seps = self.update_bead_separations()

        state_sum, beads_sum = self.state_summary

        e_reactant, e_product = e_beads[0], e_beads[-1]
        e_max = max(e_beads)
        barrier_fwd = e_max - e_reactant
        barrier_rev = e_max - e_product

        tab = lambda l: '\t'.join([str(i) for i in l])
        format = lambda f, l: '\t'.join([f % i for i in l])

        f = '%.3e'
        s = ["Chain of States Summary",
             "Grad/Energy calls\t%d" % eg_calls,
             "Beads Count\t %d" % self.beads_count,
             "Total Energy\t%f" % e_total,
             "Bead Energies\t%s" % tab(e_beads),
             "Perp Forces (RMS total)\t%f" % rmsf_perp_total,
             "Perp Forces (RMS bead)\t%s" % format(f, rmsf_perp_beads),
             "Para Forces (RMS total)\t%f" % rmsf_para_total,
             "Para Forces (RMS bead)\t%s" % format(f, rmsf_para_beads),
             "Step Size (RMS total)\t%f" % step_total,
             "Step Size (RMS bead)\t%s" % format(f, step_beads),
             "Bead Angles\t%s" % format('%.0f', angles),
             "Bead Separations (Pythagorean)\t%s" % format(f, seps),
             "State Summary (total)\t%s" % state_sum,
             "State Summary (beads)\t%s" % format('%s', beads_sum),
             "Barriers (Fwd, Rev)\t%f\t%f" % (barrier_fwd, barrier_rev),
             "Archive (bc, N, rmsf, e, maxe, s):\t%d\t%d\t%f\t%f\t%f\t%f" % \
                (self.beads_count,
                 eg_calls,
                 rmsf_perp_total,
                 e_total,
                 e_max,
                 step_total)
             ]

        return '\n'.join(s)

    def record(self):
        list = [(self.prev_state, self.state_vec),
                (self.prev_perp_forces, self.perp_bead_forces),
                (self.prev_para_forces, self.para_bead_forces),
                (self.prev_energies, self.bead_pes_energies)]
        
        for prev, curr in list:
            prev = array(curr)

    def post_obj_func(self, grad):
        if self.reporting:
            if grad:
                self.reporting.write("***Gradient call***\n")
            else:
                self.reporting.write("***Energy call***\n")
        if self.prev_state == None:
            self.prev_state = self.state_vec
        elif (self.state_vec == self.prev_state).all():
            return

        self.eg_calls += 1

        if self.reporting:
            s = [str(self),
                 common.line()]
            s = '\n'.join(s) + '\n'
            self.reporting.write(s)
            self.reporting.flush()
        else:
            assert self.reporting == None

        self.record()

    def grow_string(self):
        return False

    @property
    def angles(self):
        """Returns an array of angles between beed groups of 3 beads."""

        angles = []
        for i in range(len(self.state_vec))[2:]:
            t0 = self.state_vec[i-1] - self.state_vec[i-2]
            t1 = self.state_vec[i] - self.state_vec[i-1]
            angles.append(vector_angle(t1, t0))
        return array(angles)

    def update_bead_separations(self):
        """Updates internal vector of distances between beads."""

        v = self.state_vec.copy()
        seps = []
        for i in range(1,len(v)):
            dv = v[i] - v[i-1]
            seps.append(dot(dv,dv))

        self.bead_separations = array(seps)**0.5

        return self.bead_separations


    def pathfs(self):
        return None

    @property
    def dimension(self):
        return self.__dimension

    def get_state_as_array(self):
        """Returns copy of state as flattened array."""
        
        #Do I really need this as well as the one below?
        return deepcopy(self.state_vec).flatten()

    def get_bead_coords(self):
        """Return copy of state_vec as 2D array."""
        tmp = deepcopy(self.state_vec)
        print self.beads_count, self.dimension
        tmp.shape = (self.beads_count, self.dimension)
        return tmp

    def get_maxit(self):
        return self._maxit
    def set_maxit(self, new):
        assert new > 0
        self._maxit = new
    maxit = property(get_maxit, set_maxit)

    def obj_func(self, x):
        if self.eg_calls >= self.maxit:
            raise aof.MaxIterations

        lg.info("Chain of States Objective Function call.")
#        self.eg_calls += 1
        if self.bead_eg_calls == 0:
            self.bead_eg_calls += self.beads_count
        else:
            self.bead_eg_calls += self.beads_count - 2

        self.history.append(deepcopy(x))

    def obj_func_grad_old(self, x):
        if self.e_calls >= self.maxit or self.g_calls >= self.maxit:
            raise aof.MaxIterations

        lg.info("Chain of States Objective Function *Gradient* call.")

        self.g_calls += 1
        if self.bead_g_calls == 0:
            self.bead_g_calls += self.beads_count
        else:
            self.bead_g_calls += self.beads_count - 2

        self.history.append(deepcopy(x))

    def record_energy(self):
        self.energy_history.append(sum(self.bead_pes_energies))

    def update_bead_pes_energies(self):
        """
        Updates internal vector with the energy of each bead (based on energy 
        calculations which must already have been scheduled and run, if in 
        parallel mode).
        """
        bead_pes_energies = []
        for bead_vec in self.state_vec:
            e = self.qc_driver.energy(bead_vec)
            bead_pes_energies.append(e)

        self.bead_pes_energies = array(bead_pes_energies)

    def record_ts_estim(self, mode='splines_and_cubic'):
        """Records the energy transition state estimate that can be formed 
        from the current pathway.
        """
        estims = self.ts_estims(mode=mode)
        if len(estims) < 1:
            lg.warn("No transition state found.")
        estims.sort()
        self.ts_history.append(estims[-1])
        
    def ts_estims(self, tol=1e-10, mode='splines_and_cubic'):
        """Returns list of all transition state(s) that appear to exist along
        the reaction pathway."""

        lg.info("Estimating the transition states along the pathway, mode = %s" % mode)
        n = self.beads_count
        Es = self.bead_pes_energies.reshape(n)
        dofs = self.state_vec.reshape(n,-1)
        assert len(dofs) == len(Es)

        if mode == 'splines':
            """Uses a spline representation of the energy/coordinates of the entire path."""
            Es.shape = (n,-1)
            ys = hstack([dofs, Es])

            step = 1. / n
            xs = arange(0., 1., step)

            # TODO: using Alexei's isolated Path object, eventually must make the
            # GrowingString object use it as well.
            p = Path(ys, xs)

            E_estim_neg = lambda s: -p(s)[-1]
            E_prime_estim = lambda s: p.fprime(s)[-1]

            ts_list = []
            for x in xs[2:]:#-1]:
                # For each pair of points along the path, find the minimum
                # energy and check that the gradient is also zero.
                E_0 = -E_estim_neg(x - step)
                E_1 = -E_estim_neg(x)
                x_min = fminbound(E_estim_neg, x - step, x, xtol=tol)
                E_x = -E_estim_neg(x_min)
    #            print x_min, abs(E_prime_estim(x_min)), E_0, E_x, E_1

                # Use a looser tollerance when minimising the gradient than for 
                # the energy function. FIXME: can this be done better?
                E_prime_tol = tol * 1E4
                if abs(E_prime_estim(x_min)) < E_prime_tol and (E_0 <= E_x >= E_1):
                    p_ts = p(x_min)
                    ts_list.append((p_ts[-1], p_ts[:-1]))

        elif mode == 'splines_and_cubic':
            """Uses a spline representation of the molecular coordinates and 
            a cubic polynomial defined from the slope / value of the energy 
            for pairs of points along the path.
            """

            ys = dofs.copy()

            step = 1. / n
            ss = arange(0., 1., step)

            # build fresh functional representation of optimisation 
            # coordinates as a function of a path parameter s
            xs = Path(ys, ss)
            print "ss",ss

            ts_list = []

            from numpy.linalg import norm
            print "ys", norm(ys[2]-ys[1]), norm(ys[1]-ys[0])
            for i in range(n)[1:]:#-1]:
                # For each pair of points along the path, find the minimum
                # energy and check that the gradient is also zero.
                E_0 = Es[i-1]
                E_1 = Es[i]
                dEdx_0 = self.bead_pes_gradients[i-1]
                dEdx_1 = self.bead_pes_gradients[i]
                dxds_0 = xs.fprime(ss[i-1])
                dxds_1 = xs.fprime(ss[i])
                print "ang", common.vector_angle(ys[2]-ys[0], dxds_1)

                #energy gradient at "left/right" bead along path
                #print "dEdx_0, dxds_0", dEdx_0, dxds_0
                #print "dEdx_1, dxds_1", dEdx_1, dxds_1
                dEds_0 = dot(dEdx_0, dxds_0)
                dEds_1 = dot(dEdx_1, dxds_1)

                dEdss = array([dEds_0, dEds_1])
                #print "i",i
                print "dEdss", dEdss
                #print "ss[i-1:i+1], Es[i-1:i+1]", ss[i-1:i+1], Es[i-1:i+1]

                if dEds_0 >= 0 and dEds_1 < 0:
                    cub = CubicFunc(ss[i-1:i+1], Es[i-1:i+1], dEdss)

                    # Some versions of SciPy seem to give the function called 
                    # by fminbound a 1d vector, and others give it a scalar.
                    # The workaround here is to use the atleast_1d function.
                    E_estim_neg = lambda s: -cub(atleast_1d(s)[0])
                    E_prime_estim = lambda s: cub.fprime(atleast_1d(s)[0])

                    s_min = fminbound(E_estim_neg, ss[i-1], ss[i], xtol=tol)
                    print "s_min",s_min
                    E_s = -E_estim_neg(s_min)

                    print "E_s", E_s

                    # Use a looser tollerance on the gradient than on minimisation of 
                    # the energy function. FIXME: can this be done better?
                    E_prime_tol = tol * 1E4
                    if abs(E_prime_estim(s_min)) < E_prime_tol and (E_0 <= E_s >= E_1):
                        #print "Found", i
                        xs_ts = xs(s_min)
                        ts_list.append((E_s, xs_ts))
                #print "-------"

        elif mode == 'highest':
            """Just picks the highest energy from along the path.
            """
            i = Es.argmax()
            ts_list = [(Es[i], dofs[i])]
        else:
            raise Exception("Unrecognised TS estimation mode " + mode)

        if len(ts_list) < 1:
            ix = self.beads_count/2
            lg.warn("No transition state found, taking bead with index beads_count/2.")
            ts_list = [(Es[ix], dofs[ix])]

        return ts_list

 
def specialReduceXX(list, ks = [], f1 = lambda a,b: a-b, f2 = lambda a: a**2):
    """For a list of x_0, x_1, ... , x_(N-1)) and a list of scalars k_0, k_1, ..., 
    returns a list of length N-1 where each element of the output array is 
    f2(f1(k_i * x_i, k_i+1 * x_i+1)) ."""

    assert type(list) == ndarray
    assert len(list) >= 2
    assert len(ks) == 0 or len(ks) == len(list)
    
    # Fill with trivial value that won't change the result of computations
    if len(ks) == 0:
        ks = array(ones(len(list)))

    def specialReduce_(head, head1, tail, f1, f2, k, k1, ktail):
        reduction = f2 (f1 (k*head, k1*head1))
        if len(tail) == 0:
            return [reduction]
        else:
            return [reduction] + specialReduce_(head1, tail[0], tail[1:], f1, f2, k1, ktail[0], ktail[1:])

    return array(specialReduce_(list[0], list[1], list[2:], f1, f2, ks[0], ks[1], ks[2:]))

class NEB(ReactionPathway):
    """Implements a Nudged Elastic Band (NEB) transition state searcher.
    
    >>> path = [[0,0],[0.2,0.2],[0.7,0.7],[1,1]]
    >>> qc = aof.pes.GaussianPES()
    >>> neb = NEB(path, qc, 1.0, beads_count = 4)
    >>> neb.state_vec
    array([[ 0. ,  0. ],
           [ 0.2,  0.2],
           [ 0.7,  0.7],
           [ 1. ,  1. ]])

    >>> neb.obj_func()
    -2.6541709711655024

    >>> neb.obj_func_grad().round(3)
    array([-0.   , -0.   , -0.291, -0.309,  0.327,  0.073, -0.   , -0.   ])

    >>> print str(neb)[:len('Chain of States Summary')]
    Chain of States Summary

    >>> neb.step
    (0.0, [0.0, 0.0, 0.0, 0.0])

    >>> neb.obj_func_grad([[0,0],[0.3,0.3],[0.9,0.9],[1,1]]).round(3)
    array([-0.   , -0.   , -0.282, -0.318,  0.714,  0.286, -0.   , -0.   ])

    >>> array(neb.step[1]).round(1)
    array([ 0. ,  0.1,  0.2,  0. ])

    >>> neb.eg_calls
    2
    """

    growing = False
    def __init__(self, reagents, qc_driver, base_spr_const, beads_count=10, 
        parallel=False, reporting=None):

        ReactionPathway.__init__(self, reagents, beads_count, qc_driver, 
            parallel=parallel, reporting=reporting)

        self.base_spr_const = base_spr_const

        # Make list of spring constants for every inter-bead separation
        # For the time being, these are uniform
        self.spr_const_vec = array([self.base_spr_const for x in range(beads_count - 1)])

        self.use_upwinding_tangent = True

        # Generate or copy initial path
        if len(reagents) == beads_count:
            self.state_vec = array(reagents)
        else:
            pr = PathRepresentation(reagents, beads_count, lambda x: 1)
            pr.regen_path_func()
            pr.generate_beads(update = True)
            self.state_vec = pr.state_vec.copy()

    def update_tangents(self):
        # terminal beads
        self.tangents[0]  = self.state_vec[1] - self.state_vec[0]#zeros(self.dimension)
        self.tangents[-1] = self.state_vec[-1] - self.state_vec[-2]#zeros(self.dimension)

        for i in range(self.beads_count)[1:-1]:
            if self.use_upwinding_tangent:
                tang_plus = self.state_vec[i+1] - self.state_vec[i]
                tang_minus = self.state_vec[i] - self.state_vec[i-1]

                Vi = self.bead_pes_energies[i]
                Vi_minus_1 = self.bead_pes_energies[i-1]
                Vi_plus_1 = self.bead_pes_energies[i+1]
                
                delta_V_plus = abs(Vi_plus_1 - Vi)
                delta_V_minus = abs(Vi - Vi_minus_1)

                delta_V_max = max(delta_V_plus, delta_V_minus)
                delta_V_min = min(delta_V_plus, delta_V_minus)

                if Vi_plus_1 > Vi > Vi_minus_1:
                    self.tangents[i] = tang_plus

                elif Vi_plus_1 < Vi < Vi_minus_1:
                    self.tangents[i] = tang_minus

                elif Vi_plus_1 > Vi_minus_1:
                    self.tangents[i] = tang_plus * delta_V_max + tang_minus * delta_V_min

                elif Vi_plus_1 <= Vi_minus_1:
                    self.tangents[i] = tang_plus * delta_V_min + tang_minus * delta_V_max
                else:
                    raise Exception("Should never happen")
            else:
                self.tangents[i] = ( (self.state_vec[i] - self.state_vec[i-1]) + (self.state_vec[i+1] - self.state_vec[i]) ) / 2

            self.tangents[i] /= linalg.norm(self.tangents[i], 2)


    def __len__(self):
        """For compatibility with ASE, pretends that there are atoms with cartesian coordinates."""
        return int(ceil((self.beads_count * self.dimension / 3.)))

    def set_positions(self, x):
        """For compatibility with ASE, pretends that there are atoms with cartesian coordinates."""
        self.state_vec = x.flatten()[0:self.beads_count * self.dimension]
        self.state_vec.shape = (self.beads_count, -1)

    def get_positions(self):
        """For compatibility with ASE, pretends that there are atoms with cartesian coordinates.""" 
        return common.make_like_atoms(self.state_vec.copy())

    def get_forces(self):
        """For compatibility with ASE, pretends that there are atoms with cartesian coordinates."""
        return -common.make_like_atoms(self.obj_func_grad())

    def get_potential_energy(self):
        """For compatibility with ASE, pretends that there are atoms with cartesian coordinates."""
        return self.obj_func()

    def obj_func_old(self, new_state_vec = None):
        assert False, "Deprecated. Now use special case of obj_func_grad."

        assert size(self.state_vec) == self.beads_count * self.dimension

        ReactionPathway.obj_func(self, new_state_vec)

        if new_state_vec != None:
            self.state_vec = array(new_state_vec)
            self.state_vec.shape = (self.beads_count, self.dimension)

        # request and process parallel QC jobs
        if self.parallel:

            for i in range(self.beads_count): # [1:-1]:

                bead_vec = self.state_vec[i]
                self.qc_driver.request_gradient(bead_vec)

            self.qc_driver.proc_requests()
        
        # update vector of energies of individual beads`
        # this is required for the tangent calculation
        self.update_bead_pes_energies()

        self.update_tangents()
        self.update_bead_separations()
        
        spring_energies = self.base_spr_const * self.bead_separations**2
        spring_energies = 0.5 * numpy.sum (spring_energies)
        lg.info("Spring energies %s" % spring_energies)

#        pes_energies = sum(self.bead_pes_energies[1:-1])

        # should spring_enrgies be included here?
        return self.bead_pes_energies.sum()# + spring_energies

    def obj_func_grad(self, new_state_vec = None):
        return self.obj_func(new_state_vec, grad=True)

    def obj_func(self, new_state_vec = None, grad=False):

        ReactionPathway.obj_func(self, new_state_vec)

        # If a state vector has been specified, return the value of the 
        # objective function for this new state and set the state of self
        # to the new state.
        if new_state_vec != None:
            self.state_vec = array(new_state_vec).reshape(self.beads_count, -1)

         # request and process parallel QC jobs
        if self.parallel:

            for i in range(self.beads_count): #[1:-1]:
                self.qc_driver.request_gradient(self.state_vec[i])

            self.qc_driver.proc_requests()

        self.update_bead_pes_energies()
        self.update_bead_separations()
        self.update_tangents()

        result_bead_forces = []
        perp_bead_forces  = []
        para_bead_forces  = []

        # get PES forces / project out stuff
        for i in range(self.beads_count):#[1:-1]: # don't include end beads, leave their gradients as zero
            g = self.qc_driver.gradient(self.state_vec[i])
            self.bead_pes_gradients[i] = g.copy()

            perp_force, para_force = project_out(self.tangents[i], -g)
            perp_bead_forces.append(perp_force)
            para_bead_forces.append(para_force)

            spring_force_mag = 0
            # no spring force for end beads
            if i > 0 and i < self.beads_count - 1:
                dx1 = self.bead_separations[i]
                dx0 = self.bead_separations[i-1]
                spring_force_mag = self.base_spr_const * (dx1 - dx0)

            spring_force = spring_force_mag * self.tangents[i]

            total = perp_force + spring_force

            result_bead_forces.append(total)

        
        self.perp_bead_forces = perp_bead_forces
        self.para_bead_forces = para_bead_forces

        # at this point, parallel component has been projected out
        result_bead_forces = array(result_bead_forces)

        # set force of end beads to zero
        z = zeros(self.dimension)
        result_bead_forces[0] = z
        result_bead_forces[-1] = z

        self.post_obj_func(grad)

        if grad:
            g = -result_bead_forces.flatten()
            return g
        else:
            spring_energies = self.base_spr_const * self.bead_separations**2
            spring_energies = 0.5 * numpy.sum (spring_energies)
            lg.info("Spring energies %s" % spring_energies)
            return self.bead_pes_energies.sum()# + spring_energies

class Func():
    def f():
        pass
    def fprime():
        pass

class LinFunc():
    def __init__(self, xs, ys):
        self.fs = scipy.interpolate.interp1d(xs, ys)
        self.grad = (ys[1] - ys[0]) / (xs[1] - xs[0])

    def f(self, x):
        return self.fs(x) #[0]

    def fprime(self, x):
        return self.grad

class QuadFunc(Func):
    def __init__(self, coefficients):
        self.coefficients = coefficients

    def f(self, x):
        return dot(array((x**2, x, 1)), self.coefficients)

    def fprime(self, x):
        return 2 * self.coefficients[0] * x + self.coefficients[1]

class SplineFunc(Func):
    def __init__(self, xs, ys):
        self.spline_data = interpolate.splrep(xs, ys, s=0, k=3)

    def f(self, x):
        return interpolate.splev(x, self.spline_data, der=0)

    def fprime(self, x):
        return interpolate.splev(x, self.spline_data, der=1)

def dup2val(dup):
    (x,y) = dup
    return x

class PathRepresentation(object):
    """Supports operations on a path represented by a line, parabola, or a 
    spline, depending on whether it has 2, 3 or > 3 points."""

    def __init__(self, state_vec, beads_count, rho = lambda x: 1, str_resolution = 10000):

        # vector of vectors defining the path
        self.__state_vec = array(state_vec)

        # number of vectors defining the path
        self.beads_count = beads_count
        self.__dimension = len(state_vec[0])

        self.__str_resolution = str_resolution
        self.__step = 1.0 / self.__str_resolution

        self.__fs = []
        self.__path_tangents = []

        self.__unit_interval = array((0.0,1.0))

        # generate initial paramaterisation density
        # TODO: Linear at present, perhaps change eventually
        points_cnt = len(self.__state_vec)
        self.__normalised_positions = arange(0.0, 1.0 + 1.0 / (points_cnt - 1), 1.0 / (points_cnt - 1))
        self.__normalised_positions = self.__normalised_positions[0:points_cnt]

        self.__max_integral_error = 1e-4

        self.__rho = self.set_rho(rho)

        # TODO check all beads have same dimensionality

    """def get_fs(self):
        return self.__fs"""

    @property
    def fs(self):
        return self.__fs
    @property
    def path_tangents(self):
        return self.__path_tangents

    def recalc_path_tangents(self):
        """Returns the unit tangents to the path at the current set of 
        normalised positions."""

        tangents = []
        for str_pos in self.__normalised_positions:
            tangents.append(self.__get_tangent(str_pos))

        tangents = array(tangents)
        return tangents

    def get_state_vec(self):
        return self.__state_vec

    def set_state_vec(self, new_state_vec):
        self.__state_vec = array(new_state_vec).flatten()
        self.__state_vec.shape = (self.beads_count, -1)

    state_vec = property(get_state_vec, set_state_vec)

    @property
    def dimension(self):
        return self.__dimension

    def regen_path_func(self):
        """Rebuild a new path function and the derivative of the path based on 
        the contents of state_vec."""

        assert len(self.__state_vec) > 1

        self.__fs = []

        for i in range(self.__dimension):

            ys = self.__state_vec[:,i]

            # linear path
            if len(self.__state_vec) == 2:
                self.__fs.append(LinFunc(self.__unit_interval, ys))

            # parabolic path
            elif len(self.__state_vec) == 3:

                # FIXME: at present, transition state assumed to be half way ebtween reacts and prods
                ps = array((0.0, 0.5, 1.0))
                ps_x_pow_2 = ps**2
                ps_x_pow_1 = ps
                ps_x_pow_0 = ones(len(ps_x_pow_1))

                A = column_stack((ps_x_pow_2, ps_x_pow_1, ps_x_pow_0))

                quadratic_coeffs = linalg.solve(A,ys)

                self.__fs.append(QuadFunc(quadratic_coeffs))

            else:
                # spline path
                xs = self.__normalised_positions
                assert len(self.__normalised_positions) == len(self.__state_vec)
                self.__fs.append(SplineFunc(xs,ys))


    def __arc_dist_func(self, x):
        output = 0
        for a in self.__fs:
            output += a.fprime(x)**2
        return sqrt(output)

    def __get_total_str_len_exact(self):

        (integral, error) = scipy.integrate.quad(self.__arc_dist_func, 0.0, 1.0)
        return integral

    def get_bead_separations(self):
        """Returns the arc length between beads according to the current 
        parameterisation.
        """

        a = self.__normalised_positions
        N = len(a)
        seps = []
        for i in range(N)[1:]:
            l = scipy.integrate.quad(self.__arc_dist_func, a[i-1], a[i])
            seps.append(l[0])
        return array(seps)

    def __get_total_str_len(self):
        """Returns the a duple of the total length of the string and a list of 
        pairs (x,y), where x a distance along the normalised path (i.e. on 
        [0,1]) and y is the corresponding distance along the string (i.e. on
        [0,string_len])."""
        

        # number of points to chop the string into
        param_steps = arange(0, 1, self.__step)

        list = []
        cummulative = 0

        (str_len_precise, error) = scipy.integrate.quad(self.__arc_dist_func, 0, 1, limit=100)
        lg.debug("String length integration error = " + str(error))
        assert error < self.__max_integral_error

        for i in range(self.__str_resolution):
            pos = (i + 0.5) * self.__step
            sub_integral = self.__step * self.__arc_dist_func(pos)
            cummulative += sub_integral
            list.append(cummulative)

        #lg.debug('int_approx = {0}, int_accurate = {1}'.format(cummulative, str_len_precise))

        return (list[-1], zip(param_steps, list))

    def sub_str_lengths(self, normd_poses):
        """Finds the lengths of the pieces of the string specified in 
        normd_poses in terms of normalised coordinate."""

        my_normd_poses = array(normd_poses).flatten()

        from scipy.integrate import quad
        x0 = 0.0
        lengths = []
        for pos in my_normd_poses:
            (len,err) = quad(self.__arc_dist_func, x0, pos)
            lengths.append(len)
            x0 = pos

        lengths = array(lengths).flatten()

    def generate_beads_exact(self, update = False):
        """Returns an array of the self.__beads_count vectors of the coordinates 
        of beads along a reaction path, according to the established path 
        (line, parabola or spline) and the parameterisation density."""


        assert len(self.__fs) > 1

        # Find total string length and incremental distances x along the string 
        # in terms of the normalised coodinate y, as a list of (x,y).
        total_str_len = self.__get_total_str_len_exact()

        # For the desired distances along the string, find the values of the
        # normalised coordinate that achive those distances.
        normd_positions = self.__generate_normd_positions_exact(total_str_len)

        self.sub_str_lengths(normd_positions)

        bead_vectors = []
        bead_tangents = []
        for str_pos in normd_positions:
            bead_vectors.append(self.__get_bead_coords(str_pos))
            bead_tangents.append(self.__get_tangent(str_pos))


        (reactants, products) = (self.__state_vec[0], self.__state_vec[-1])
        bead_vectors = array([reactants] + bead_vectors + [products])
        bead_tangents = array([self.__get_tangent(0)] + bead_tangents + [self.__get_tangent(1)])

        if update:
            self.state_vec = bead_vectors

            self.__path_tangents = bead_tangents

        return bead_vectors

    def generate_beads(self, update = False):
        """Returns an array of the self.__beads_count vectors of the coordinates 
        of beads along a reaction path, according to the established path 
        (line, parabola or spline) and the parameterisation density."""

        assert len(self.__fs) > 1

        # Find total string length and incremental distances x along the string 
        # in terms of the normalised coodinate y, as a list of (x,y).
        (total_str_len, incremental_positions) = self.__get_total_str_len()

        # For the desired distances along the string, find the values of the
        # normalised coordinate that achive those distances.
        normd_positions = self.__generate_normd_positions(total_str_len, incremental_positions)

        bead_vectors = []
        bead_tangents = []
        for str_pos in normd_positions:
            bead_vectors.append(self.__get_bead_coords(str_pos))
            bead_tangents.append(self.__get_tangent(str_pos))

        reactants = self.__state_vec[0]
        products = self.__state_vec[-1]
        bead_vectors = array([reactants] + bead_vectors + [products])
        bead_tangents = array([self.__get_tangent(0)] + bead_tangents + [self.__get_tangent(1)])

        if update:
            self.state_vec = bead_vectors
            lg.debug("New beads generated: " + str(self.__state_vec))

            self.__path_tangents = bead_tangents
            lg.debug("Tangents updated:" + str(self.__path_tangents))

            self.__normalised_positions = array([0.0] + normd_positions + [1.0])
            lg.debug("Normalised positions updated:" + str(self.__normalised_positions))

        return bead_vectors

    def update_tangents(self):
        points_cnt = len(self.__state_vec)
        even_spacing = arange(0.0, 1.0 + 1.0 / (points_cnt - 1), 1.0 / (points_cnt - 1))
        even_spacing = even_spacing[0:points_cnt]

        ts = []
        for i in self.__normalised_positions:
             ts.append(self.__get_tangent(i))
 
        self.__path_tangents = array(ts)
        
    def __get_str_positions_exact(self):
        from scipy.integrate import quad
        from scipy.optimize import fmin

        integrated_density_inc = 1.0 / (self.beads_count - 1.0)
        requirement_for_prev_bead = 0.0
        requirement_for_next_bead = integrated_density_inc

        x_min = 0.0
        x_max = -1

        str_poses = []

        def f_opt(x):
            (i,e) = quad(self.__rho, x_min, x)
            tmp = (i - integrated_density_inc)**2
            return tmp

        for i in range(self.beads_count - 2):

            x_max = fmin(f_opt, requirement_for_next_bead, disp=False)
            str_poses.append(x_max[0])
            x_min = x_max[0]
            requirement_for_prev_bead = requirement_for_next_bead
            requirement_for_next_bead += integrated_density_inc

        return str_poses

    def dump_rho(self):
        res = 0.02
        print "rho: ",
        for x in arange(0.0, 1.0 + res, res):
            if x < 1.0:
                print self.__rho(x),
        print
        raw_input("that was rho...")


    def __get_str_positions(self):
        """Based on the provided density function self.__rho(x) and 
        self.beads_count, generates the fractional positions along the string 
        at which beads should occur."""

        param_steps = arange(0, 1 - self.__step, self.__step)
        integrated_density_inc = 1.0 / (self.beads_count - 1.0)
        requirement_for_next_bead = integrated_density_inc

        integral = 0
        str_positions = []
        prev_s = 0
        for s in param_steps:
            integral += 0.5 * (self.__rho(s) + self.__rho(prev_s)) * self.__step
            if integral > requirement_for_next_bead:
                str_positions.append(s)
                requirement_for_next_bead += integrated_density_inc
            prev_s = s
        
#        dump_diffs("spd", str_positions)

        # Return first self.beads_count-2 points. Reason: sometimes, due to
        # inaccuracies in the integration of the density function above, too
        # many points are generated in str_positions.
        return str_positions[0:self.beads_count-2]

    def __get_bead_coords(self, x):
        """Returns the coordinates of the bead at point x <- [0,1]."""
        bead_coords = []
        for f in self.__fs:
            bead_coords.append(f.f(x))

        return (array(bead_coords).flatten())

    def __get_tangent(self, x):
        """Returns the tangent to the path at point x <- [0,1]."""

        path_tangent = []
        for f in self.__fs:
            path_tangent.append(f.fprime(x))

        t = array(path_tangent).flatten()
        t = t / linalg.norm(t)
        return t

    def set_rho(self, new_rho, normalise=True):
        """Set new bead density function, ensuring that it is normalised."""
        if normalise:
            (integral, err) = scipy.integrate.quad(new_rho, 0.0, 1.0)
        else:
            int = 1.0
        self.__rho = lambda x: new_rho(x) / integral
        return self.__rho

    def __generate_normd_positions_exact(self, total_str_len):

        #  desired fractional positions along the string
        fractional_positions = self.__get_str_positions_exact()

        normd_positions = []

        prev_norm_coord = 0
        prev_len_wanted = 0

        from scipy.integrate import quad
        from scipy.optimize import fmin
        for frac_pos in fractional_positions:
            next_norm_coord = frac_pos
            next_len_wanted = total_str_len * frac_pos
            length_err = lambda x: \
                (dup2val(quad(self.__arc_dist_func, prev_norm_coord, x)) - (next_len_wanted - prev_len_wanted))**2
            next_norm_coord = fmin(length_err, next_norm_coord, disp=False)

            normd_positions.append(next_norm_coord)

            prev_norm_coord = next_norm_coord
            prev_len_wanted = next_len_wanted

        return normd_positions
            
    def __generate_normd_positions(self, total_str_len, incremental_positions):
        """Returns a list of distances along the string in terms of the normalised 
        coordinate, based on desired fractional distances along string."""

        # Get fractional positions along string, based on bead density function
        # and the desired total number of beads
        fractional_positions = self.__get_str_positions()

        normd_positions = []

        lg.debug("fractional_positions: %s" % fractional_positions)
        for frac_pos in fractional_positions:
            for (norm, str) in incremental_positions:

                if str >= frac_pos * total_str_len:
                    normd_positions.append(norm)
                    break

        return normd_positions

class PiecewiseRho:
    """Supports the creation of piecewise functions as used by the GrowingString
    class as the bead density function."""
    def __init__(self, a1, a2, rho, max_beads):
        self.a1, self.a2, self.rho = a1, a2, rho
        self.max_beads = max_beads

    def f(self, x):
        if 0 <= x <= self.a1:
            return self.rho(x)
        elif self.a1 < x <= self.a2:
            return 0.0
        elif self.a2 < x <= 1.0:
            return self.rho(x)
        else:
            lg.error("Value of (%f) not on [0,1], should never happen" % x)
            lg.error("a1 = %f, a2 = %f" % (self.a1, self.a2))

class GrowingString(ReactionPathway):
    """Implements growing and non-growing strings.

    >>> path = [[0,0],[0.2,0.2],[0.7,0.7],[1,1]]
    >>> qc = aof.pes.GaussianPES()
    >>> s = GrowingString(path, qc, beads_count=4, growing=False)
    >>> s.state_vec.round(1)
    array([[ 0. ,  0. ],
           [ 0.3,  0.3],
           [ 0.7,  0.7],
           [ 1. ,  1. ]])

    >>> new = s.state_vec.round(2).copy()
    >>> s.obj_func()
    -2.5884273157684441

    >>> s.obj_func_grad().round(3)
    array([ 0.   ,  0.   ,  0.021, -0.021,  0.11 , -0.11 ,  0.   ,  0.   ])

    >>> print str(s)[:len('Chain of States Summary')]
    Chain of States Summary

    >>> s.step
    (0.0, [0.0, 0.0, 0.0, 0.0])

    >>> s.obj_func_grad(new)
    array([ 0.        ,  0.        ,  0.02041863, -0.02041863,  0.10998242, -0.10998242,  0.        ,  0.        ])
    >>> array(s.step[1])
    array([ 0.        ,  0.00149034,  0.000736  ,  0.        ])

    >>> s.obj_func_grad([[0,0],[0.3,0.3],[0.9,0.9],[1,1]]).round(3)
    Traceback (most recent call last):
        ...
    MustRegenerate

    >>> s.eg_calls
    3

    """
    def __init__(self, reagents, qc_driver, beads_count = 10, rho = lambda x: 1, growing=True, parallel=False, head_size=None, max_sep_ratio = 0.1):

        self.__qc_driver = qc_driver

        self.__final_beads_count = beads_count

        self.growing = growing
        if growing:
            initial_beads_count = 4
        else:
            initial_beads_count = self.__final_beads_count

        # TODO: this was moved from the first line recently, check
        ReactionPathway.__init__(self, reagents, initial_beads_count, qc_driver, parallel)

        # create PathRepresentation object
        self.__path_rep = PathRepresentation(reagents, initial_beads_count, rho)

        # final bead spacing density function for grown string
        # make sure it is normalised
        (int, err) = scipy.integrate.quad(rho, 0.0, 1.0)
        self.__final_rho = lambda x: rho(x) / int

        # current bead spacing density function for incompletely grown string
        self.update_rho()

        # Build path function based on reagents
        self.__path_rep.regen_path_func()

        # Space beads along the path
        self.__path_rep.generate_beads(update = True)

        # dummy energy of a reactant/product
        # TODO: remove?
        self.__reagent_energy = 0

        self.parallel = parallel

        # Number of beads in growing heads of growing string to calculate 
        # gradients for. All others are left as zero.
        assert head_size == None, "Not yet properly implemented, problem when beads are respaced."
        assert head_size == None or (growing and beads_count / 2 -1 >= head_size > 0)
        self.head_size = head_size

        # maximum allowed ratio between (max bead sep - min bead sep) and (average bead sep)
        self.__max_sep_ratio = max_sep_ratio

        # TODO: can I get rid of this?
        self.must_regenerate  = False

    def __len__(self):
        """For compatibility with ASE, pretends that there are atoms with cartesian coordinates."""

        return len(common.make_like_atoms(self.state_vec))

    def pathfs(self):
        return self.__path_rep.fs

    def set_positions(self, x):
        """For compatibility with ASE, pretends that there are atoms with cartesian coordinates."""
        new_state_vec = x.flatten()[0:self.beads_count * self.dimension]
        self.update_path(new_state_vec, respace = False)

    def get_positions(self):
        """For compatibility with ASE, pretends that there are atoms with cartesian coordinates.""" 
        return common.make_like_atoms(self.state_vec)

    def get_forces(self):
        """For compatibility with ASE, pretends that there are atoms with cartesian coordinates."""
        return -common.make_like_atoms(self.obj_func_grad())

    def get_potential_energy(self):
        """For compatibility with ASE, pretends that there are atoms with cartesian coordinates."""
        return self.obj_func()

    def get_state_vec(self):
        return deepcopy(self.__path_rep.state_vec)

    state_vec = property(get_state_vec)

    def get_beads_count(self):
        return self.__path_rep.beads_count

    def set_beads_count(self, new):
        self.__path_rep.beads_count = new

    beads_count = property(get_beads_count, set_beads_count)

    @property
    def tangents(self):
        return self.__path_rep.path_tangents

    def expand(self, old, new_size):

        assert len(old) % 2 == 0

        new = zeros(old.shape[1] * new_size).reshape(new_size, -1)
        for i in range(len(old) / 2):
            new[i] = old[i]
            new[-i] = old[-i]

        return new

    def grown(self):
        return self.beads_count == self.__final_beads_count

    def grow_string(self):
        """
        Adds 2, 1 or 0 beads to string (such that the total number of 
        beads is less than or equal to self.__final_beads_count).
        """

        assert self.beads_count <= self.__final_beads_count

        if self.grown():
            return False
        elif self.__final_beads_count - self.beads_count == 1:
            self.beads_count += 1
        else:
            self.beads_count += 2

        self.bead_pes_gradients = self.expand(self.bead_pes_gradients, self.beads_count)

        self.__path_rep.beads_count = self.beads_count

        # build new bead density function based on updated number of beads
        self.update_rho()
        self.__path_rep.generate_beads(update = True)

        # Build mask of gradients to calculate, effectively freezing some if 
        # head_size, i.e. the size of the growing heads, is specified.
        if self.head_size:
            e = self.beads_count / 2 - self.head_size + self.beads_count % 2
        else:
            e = 1
        m = self.beads_count - 2 * e
        self.grad_update_mask = [False for i in range(e)] + [True for i in range(m)] + [False for i in range(e)]
        lg.debug("Bead Freezing MASK: " + str(self.grad_update_mask))

        lg.info("******** String Grown to %d beads ********", self.beads_count)

        return True

    def update_rho(self):
        """Update the density function so that it will deliver beads spaced
        as for a growing (or grown) string."""

        assert self.beads_count <= self.__final_beads_count

        from scipy.optimize import fmin
        from scipy.integrate import quad

        if self.beads_count == self.__final_beads_count:
            self.__path_rep.set_rho(self.__final_rho)

            return self.__final_rho

        # Value that integral must be equal to to give desired number of beads
        # at end of the string.
        end_int = (self.beads_count / 2.0) \
            / self.__final_beads_count

        f_a1 = lambda x: (quad(self.__final_rho, 0.0, x)[0] - end_int)**2
        f_a2 = lambda x: (quad(self.__final_rho, x, 1.0)[0] - end_int)**2

        a1 = fmin(f_a1, end_int)[0]
        a2 = fmin(f_a2, end_int)[0]

        assert a2 > a1

        pwr = PiecewiseRho(a1, a2, self.__final_rho, self.__final_beads_count)
        self.__path_rep.set_rho(pwr.f)

    def obj_func_old(self, new_state_vec = None, grad):
        if self.must_regenerate:
            return self.bead_pes_energies.sum()

        ReactionPathway.obj_func(self, new_state_vec)

        self.update_path(new_state_vec, respace = False)

        es = [] #[self.__reagent_energy]

        # request and process parallel QC jobs
        if self.parallel:
            for bead_vec in self.__path_rep.state_vec: #[1:-1]:
                self.qc_driver.request_gradient(bead_vec)
            self.qc_driver.proc_requests()

        self.update_bead_pes_energies()

        es = self.bead_pes_energies

        # assume that energies of end beads are not updated every time
        self.bead_eg_calls += len(es) - 2

        # If bead spacings become too uneven, set a flag so that the string 
        # is not modified until it is updated.
        if self.lengths_disparate():
            raise aof.MustRegenerate
            #self.must_regenerate = True

        return es.sum() #es.max() #total_energies
       
    def lengths_disparate(self):
        """Returns true if the ratio between the (difference of longest and 
        shortest segments) to the average segment length is above a certain 
        value (self.__max_sep_ratio).
        """
        seps = self.__path_rep.get_bead_separations()

        # If string is incompletely grown, remove inter-bead distance, 
        # corresponding to the ungrown portion of the string.
        if self.beads_count < self.__final_beads_count:
            l = seps.tolist()
            i = seps.argmax()
            l.pop(i)
            seps = array(l)

        max_sep = seps.max()
        min_sep = seps.min()
        mean_sep = seps.mean()

        r = (max_sep - min_sep) / mean_sep
        lg.info("Bead spacing ratio is %f, max is %f" % (r, self.__max_sep_ratio))
        return r > self.__max_sep_ratio

    def obj_func(self, new_state_vec = None, grad=False):

        # TODO: do I still need this?
        if self.must_regenerate:
            return array(self.__path_rep.state_vec * 0.).flatten()

        #ReactionPathway.obj_func_grad(self, new_state_vec)

        self.update_path(new_state_vec, respace = False)

        gradients = []

        ts = self.__path_rep.path_tangents

        # request and process parallel QC jobs
        if self.parallel:
            for i, bead_vec in enumerate(self.__path_rep.state_vec): #[1:-1]:
                if self.grad_update_mask[i] or self.e_calls == 0 or self.g_calls == 0:
                    self.qc_driver.request_gradient(bead_vec)
            self.qc_driver.proc_requests()

        self.update_bead_pes_energies()

        # get gradients / perform projections
        for i in range(self.beads_count):
            if self.grad_update_mask[i]:
                g = self.__qc_driver.gradient(self.__path_rep.state_vec[i])
                self.bead_pes_gradients[i] = g.copy()
                t = ts[i]
                g, para = project_out(t, g)
            else:
                g = zeros(self.__path_rep.dimension)
            gradients.append(g)

        gradients = array(gradients)
        self.perp_bead_forces = gradients.copy()

        if new_state_vec != None:
            self.record_energy()

        self.post_obj_func(True)

        # If bead spacings become too uneven, set a flag so that the string 
        # is not modified until it is updated.
        if self.lengths_disparate():
            raise aof.MustRegenerate
            #self.must_regenerate = True

        return gradients.copy().flatten()

    def update_path(self, state_vec = None, respace = True):
        """After each iteration of the optimiser this function must be called.
        It rebuilds a new (spline) representation of the path and then 
        redestributes the beads according to the density function."""

        if state_vec != None:
            self.__path_rep.state_vec = state_vec

        # rebuild line, parabola or spline representation of path
        self.__path_rep.regen_path_func()

        # respace the beads along the path
        if respace:
            self.__path_rep.generate_beads(update = True)
            self.must_regenerate = False
        else:
            self.__path_rep.update_tangents()

    def plot(self):
#        self.__path_rep.generate_beads_exact()
        plot2D(self.__path_rep)

def project_out(component_to_remove, vector):
    """Projects the component of 'vector' that list along 'component_to_remove'
    out of 'vector' and returns it."""
    projection = dot(component_to_remove, vector)
    removed = projection * component_to_remove
    output = vector - removed
    return output, removed



# Testing the examples in __doc__strings, execute
# "python gxmatrix.py", eventualy with "-v" option appended:
if __name__ == "__main__":
    import doctest
    doctest.testmod()

# You need to add "set modeline" and eventually "set modelines=5"
# to your ~/.vimrc for this to take effect.
# Dont (accidentally) delete these lines! Unless you do it intentionally ...
# Default options for vim:sw=4:expandtab:smarttab:autoindent:syntax


