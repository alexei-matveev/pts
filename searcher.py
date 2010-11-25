#!/usr/bin/python

import sys
import inspect

import scipy.integrate
from scipy.optimize import fminbound

from scipy import interpolate
import tempfile, os
import logging
from copy import deepcopy
import pickle

# TODO: change to import numpy as np???
from numpy import linalg, floor, zeros, array, ones, arange, arccos, hstack, ceil, abs, ndarray, sqrt, column_stack, dot, eye, outer, inf, isnan, isfinite, size, vstack, atleast_1d

from path import Path
from func import CubicFunc, QuadFunc, RhoInterval

from common import * # TODO: must unify
import pts.common as common
import pts

from history import History

previous = []
def test_previous(state_vec):
    """
    Mainly for testing. Keeps an additional record of calcs that have 
    previously been performed.
    """
    prev = []
    for v in state_vec:
        matches = sum([(v == p).all() for p in previous])
        if matches == 0:
            prev.append(False)
            previous.append(v)
        elif matches == 1:
            prev.append(True)
        else:
            assert False
            
    return prev

lg = logging.getLogger("pts.searcher")
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

def masked_assign(mask, dst, src):
    """Assigns y to x if mask allows it.

    >>> m = [True, True]
    >>> orig = arange(4).reshape(2,-1)
    >>> x = orig.copy()
    >>> y = array([5,4,3,2]).reshape(2,-1)
    >>> x = masked_assign(m, x, y)
    [[0 1]
     [2 3]]
    [[5 4]
     [3 2]]
    [True, True]
    >>> (x == y).all()
    True
    >>> m = [False, False]
    >>> x = orig.copy()
    >>> (x != masked_assign(m, x, y)).all()
    True
    """

    if len(dst) != len(src):
        return src.copy()

    dstc = dst.copy()
    print dstc
    print src
    print mask
    assert len(dstc) == len(src), "%d != %d" % (len(dstc), len(src))
    assert mask is None or len(mask) == len(dstc)


    for i, src_ in enumerate(src):
        if mask is None or mask[i]:
            dstc[i] = src_

    return dstc


def minimal_update(new, old, new_ixs):
    """
    Replaces vectors in |new| with corresponding ones in |old|, where 
    they do not correspond to the indices in new_ixs.

    >>> new = array([1,1,1,1,1])
    >>> old = array([0,0,0])
    >>> new_ixs = (2,3)

    >>> minimal_update(new, old, new_ixs)
    >>> new
    array([0, 0, 1, 1, 0])

    """
    assert len(new) == len(old) + len(new_ixs)

    new_ixs = list(new_ixs)
    new_ixs.sort()

    for i in range(len(new)):
        if i < new_ixs[0]:
#            print "yes",i
            new[i] = old[i]
        elif i > new_ixs[-1]:
#            print "yes",i
            new[i] = old[i-len(new_ixs)]
        else:
#            print "no",i
            pass

def freeze_ends(bc):
    return [False] + [True for i in range(bc-2)] + [False]

class ReactionPathway(object):
    """Abstract object for chain-of-state reaction pathway."""
    dimension = -1

    eg_calls = 0
    bead_eg_calls = 0

    # incremented by callback function
    callbacks = 0

    # no times path has been regenerated
    respaces = 0

    string = False

    def __init__(self, 
            reagents, 
            beads_count, 
            qc_driver, 
            parallel, 
            reporting=None, 
            convergence_beads=3, 
            steps_cumm=3, 
            freeze_beads=False, 
            output_level = 3,
            conv_mode='gradstep'):
        """
        convergence_beads:
            number of highest beads to consider when testing convergence

        steps_cumm:
            number of previous steps to consider when testing convergence

        freeze_beads:
            freeze some beads if they are not in the highest 3 or subject to low forces.

        """

        self.parallel = parallel
        self.qc_driver = qc_driver
        self.beads_count = beads_count
        self.prev_beads_count = beads_count

        self.convergence_beads = convergence_beads
        self.steps_cumm = steps_cumm
        self.freeze_beads = freeze_beads

        self.__dimension = len(reagents[0])
        #TODO: check that all reagents are same length

        self.initialise()
        
        self.history = History()

        # mask of gradients to update at each position
        self.bead_update_mask = freeze_ends(self.beads_count)

        self._maxit = sys.maxint
        if reporting:
            assert type(reporting) == file
        self.reporting = reporting

        legal_conv_modes = ('gradstep', 'energy', 'gradstep_new')
        assert conv_mode in legal_conv_modes
        self.conv_mode = conv_mode

        # This can be set externally to a file object to allow recording of picled archive data
        self.arc_record = None
        self.output_level = output_level

    def initialise(self):
        beads_count = self.beads_count

        shape = (beads_count, self.dimension)

        # forces perpendicular to pathway
        self.perp_bead_forces = zeros(shape)
        self.para_bead_forces = zeros(beads_count)

        self.tangents = zeros(shape)

        # energies / gradients of beads, excluding any spring forces / projections
        self.bead_pes_energies = zeros(beads_count)
        self.bead_pes_gradients = zeros(shape)

        self.prev_state = None
        self.prev_perp_forces = None
        self.prev_para_forces = None
        self.prev_energies = None
        self._step = zeros(shape)

    def update_mask(self, perp_max=0.3):
        assert False, "not using this"
        #self.bead_update_mask = [False, False, False, True, True, False, False, False]
        #return
        top3 = self.bead_pes_energies.argsort()[-3:]
        max_f = abs(self.perp_bead_forces[i]).max()
        self.bead_update_mask[i] = max_f > perp_max or i in top3
        lg.info("Update mask set to %s" % str(self.bead_update_mask))

    def lengths_disparate(self):
        return False

    def signal_callback(self):
        self.callbacks += 1

        # This functionality below was used with generic 3rd party optimisers,
        # i.e. it would force the optimiser to exit, so that a respace could 
        # be done.
        if self.lengths_disparate():
            raise pts.MustRegenerate

    def test_convergence(self, etol, ftol, xtol):
        
        if self.conv_mode == 'gradstep':
            return self.test_convergence_GS(ftol, xtol)

        if self.conv_mode == 'gradstep_new':
            return self.test_convergence_GS_new(ftol, xtol)

        if self.conv_mode == 'energy':
            return self.test_convergence_E(etol)

        assert False, "Should have been prevented by test in constructor."
            
    def test_convergence_E(self, etol):
        if self.eg_calls < 2:
            # because forces are always zero at zeroth iteration
            return

        bc_history = self.history.bead_count(2)
        if bc_history[1] != bc_history[0]:
            lg.info("Testing Convergence: string just grown, so skipping.")
            return


        es = array(self.history.e(2)) / (self.beads_count - 2)
        # TODO: should remove abs(), only converged if energy went down
        # Or maybe should look at difference between lowest energies so far.
        diff = abs(es[0] - es[1]) 

        lg.info("Testing Convergence of %f to %f eV / moving bead / step." % (diff, etol))
        if diff < etol:
            raise pts.Converged
        
    def test_convergence_GS(self, f_tol, x_tol, always_tight=False):
        """
        Raises Converged if converged, applying weaker convergence 
        criteria if growing string is not fully grown.

        During growth: rmsf < 10 * f_tol
        When grown:    rmsf < f_tol OR
                       max_step < x_tol AND rmsf < 5*f_tol

                       where max_step is max step in optimisation coordinates 
                           taken by H highest beads and C cummulative beads,

                       where H is self.convergence_beads
                       and C is self.steps_cumm

                       both set by __init__

                       At time of writing (19/02/2010): convergence is tested 
                       in callback function which is called after every 
                       iteration, not including backtracks. The state is 
                       recorded at every 

        """

        rmsf = self.rmsf_perp[0]
        if self.eg_calls == 0:
            # because forces are always zero at zeroth iteration
            return
        elif not always_tight and self.growing and not self.grown():
            lg.info("Testing During-Growth Convergence to %f: %f" % (f_tol*5, rmsf))
            if rmsf < f_tol*5:
                raise pts.Converged
        else:
            max_step = self.history.step(self.convergence_beads, self.steps_cumm)
            max_step = abs(max_step).max()
            lg.info("Testing Non-Growing Convergence to f: %f / %f, x: %f / %f" % (rmsf, f_tol, max_step, x_tol))
            if rmsf < f_tol or (self.eg_calls > self.steps_cumm and max_step < x_tol and rmsf < 5*f_tol):
                raise pts.Converged

    def test_convergence_GS_new(self, f_tol, x_tol):
        """
        Raises Converged if converged, applying weaker convergence 
        criteria if growing string is not fully grown.

        During growth: rmsf < 5 * f_tol
        When grown:    rmsf < f_tol OR
                       max_step < x_tol AND rmsf < 5*f_tol

                       where max_step is max step in optimisation coordinates 
                           taken by H highest beads and C cummulative beads,

                       where H is self.convergence_beads
                       and C is self.steps_cumm

                       both set by __init__

                       At time of writing (19/02/2010): convergence is tested 
                       in callback function which is called after every 
                       iteration, not including backtracks. The state is 
                       recorded at every 

        """

        f = self.maxf_perp

        max_step = self.history.step(self.convergence_beads, self.steps_cumm)
        max_step = abs(max_step).max()

        if self.eg_calls == 0:
            # because forces are always zero at zeroth iteration
            return
        elif self.growing and not self.grown():
            lg.info("Testing During-Growth Convergence to f: %f / %f (max), x: %f / %f" % (f, 5*f_tol, max_step, 2*x_tol))
            if f < f_tol*5 or (self.eg_calls > self.steps_cumm and max_step < 2*x_tol):
                raise pts.Converged
        else:
            lg.info("Testing Non-Growing Convergence to f: %f / %f (max), x: %f / %f" % (f, f_tol, max_step, x_tol))
            if f < f_tol or (self.eg_calls > self.steps_cumm and max_step < x_tol):
                raise pts.Converged

    @property
    def rmsf_perp(self):
        """RMS forces, not including those of end beads."""
        return common.rms(self.perp_bead_forces[1:-1]), [common.rms(f) for f in self.perp_bead_forces]

    def pathpos(self):
         return None, None

    @property
    def maxf_perp(self):
        """RMS forces, not including those of end beads."""
        return abs(self.perp_bead_forces[1:-1]).max()

    @property
    def rmsf_para(self):
        """RMS forces, not including those of end beads."""
        return common.rms(self.para_bead_forces), [f for f in self.para_bead_forces]

    @property
    def step(self):
        """RMS forces, not including those of end beads."""
        return common.rms(self._step[1:-1]), array([common.rms(s) for s in self._step]), abs(self._step)

    @property
    def energies(self):
        """RMS forces, not including those of end beads."""
        return self.bead_pes_energies.sum(), self.bead_pes_energies

    def set_positions(self, x):
        """For compatibility with ASE, pretends that there are atoms with cartesian coordinates."""

        tmp = x.flatten()[0:self.beads_count * self.dimension]
        self.state_vec = x.flatten()[0:self.beads_count * self.dimension]

    def get_positions(self):
        """For compatibility with ASE, pretends that there are atoms with cartesian coordinates.""" 

        return common.make_like_atoms(self.state_vec.copy())

    positions = property(get_positions, set_positions)

    @property
    def state_summary(self):
        s = common.vec_summarise(self.state_vec)
        s_beads = [common.vec_summarise(b) for b in self.state_vec]
        return s, s_beads

    def path_tuple(self):
        state, energies, gradients, (pathps, pathpsold)  = \
            self.state_vec.reshape(self.beads_count,-1), \
            self.bead_pes_energies.reshape(-1), \
            self.bead_pes_gradients.reshape(self.beads_count,-1), \
            self.pathpos()
        return state, energies, gradients, pathps, pathpsold
        
    def __str__(self):
        e_total, e_beads = self.energies
        rmsf_perp_total, rmsf_perp_beads = self.rmsf_perp
        rmsf_para_total, rmsf_para_beads = self.rmsf_para
        maxf_beads = [abs(f).max() for f in self.bead_pes_gradients]
        path_pos, path_pos_old  = self.pathpos()

        if path_pos == None:
             # set up dummy spline abscissa for non-spline methods
             path_pos = [0.0 for i in range(len(self.bead_pes_gradients))]

        eg_calls = self.eg_calls

        step_total, step_beads, step_raw = self.step

        angles = self.angles
        seps_pythag = self.update_bead_separations()

        total_len_pythag = seps_pythag.sum()
        total_len_spline = 0
        if self.string:
            total_len_spline = self._path_rep.path_len
            seps_spline = self._path_rep.get_bead_separations()
            diff = seps_spline.sum() - total_len_spline
            diff1, _ = self._path_rep._get_total_str_len()
            diff1 = abs(diff1 - total_len_spline)
            assert diff < 1e-6, "%e %e" % (diff, diff1)

            # differences between spline and pythag distances
            pyth_spl_diff = (seps_pythag - seps_spline) / seps_spline.max() * 100
            print 'Spl / Pythag differences (%)',  ' '.join(['%.0f' % i for i in pyth_spl_diff])
            


        state_sum, beads_sum = self.state_summary

        e_reactant, e_product = e_beads[0], e_beads[-1]
        e_max = max(e_beads)
        barrier_fwd = e_max - e_reactant
        barrier_rev = e_max - e_product

        tab = lambda l: '\t'.join([str(i) for i in l])
        format = lambda f, l: ' | '.join([f % i for i in l])

        all_coordinates = ("%-24s : %s\n" % ("    Coordinate %3d " % 1 , format('%10.4f',(self.state_vec[:,0]))))
        (coord_dim1, coord_dim2) = self.state_vec.shape
        for i in range(1,coord_dim2 ):
            all_coordinates += ("%-24s : %s\n" % ("    Coordinate %3d " % (i+1) , format('%10.4f',self.state_vec[:,i])))

        ts_ix = 0
        # max cummulative step over steps_cumm iterations
        step_max_bead_cumm = self.history.step(self.convergence_beads, self.steps_cumm).max()
        step_ts_estim_cumm = self.history.step(ts_ix, self.steps_cumm).max()

        if self.output_level > 2:
            arc = {'bc': self.beads_count,
                   'N': eg_calls,
                   'resp': self.respaces,
                   'cb': self.callbacks, 
                   'rmsf': rmsf_perp_total, 
                   'maxf': self.maxf_perp,
                   'e': e_total,
                   'maxe': e_max,
                   's': abs(step_raw).max(),
                   's_ts_cumm': step_ts_estim_cumm,
                   's_max_cumm': step_max_bead_cumm,
                   'ixhigh': self.bead_pes_energies.argmax()}# ,
                   #'bead_positions': self.bead_positions}

            if hasattr(self.qc_driver, 'eg_counts'):
                bead_es, bead_gs = self.qc_driver.eg_counts()
            else:
                bead_es = bead_gs = (self.beads_count - 2) * eg_calls + 2
            arc['bead_es'] = bead_es
            arc['bead_gs'] = bead_gs

        # Write out the cartesian coordinates of two transition state
        # estimates: highest bead and from spline estimation.
        if hasattr(self, 'bead2carts'):

            # History.func(n) returns list of last n values of func
            # as a tuple. For ts_estim values are a tuple: (energy, vector)
            n = 1
            which = 0
            ts_estim_energy, estim, _, _, _, _, _ = self.history.ts_estim(n)[which]
            ts_estim = self.bead2carts(estim)
            bead_carts = [self.bead2carts(v) for v in self.state_vec]
            bead_carts = zip(e_beads, bead_carts)

            delnl = lambda a: ''.join(repr(a).split('\n'))
            a2s = lambda a: '[' + ', '.join(['%.3f'%f for f in a]) + ']'
            if self.output_level > 2:
                arc["ts_estim_carts"] = ts_estim_energy, delnl(ts_estim)
                arc["bead_carts"] = delnl(bead_carts)


        f = '%10.3e'
        s = [ "\n----------------------------------------------------------",
             "Chain of States Summary for %d gradient/energy calculations" % eg_calls,
             "VALUES FOR WHOLE STRING",
             "%-24s : %10.4f" %  ("Total Energy"  , e_total) ,
             "%-24s : %10.4f | %10.4f" % ("RMS Forces (perp|para)", rmsf_perp_total, rmsf_para_total),
             "%-24s : %10.4f | %10.4f" %  ("Step Size (RMS|MAX)", step_total, step_raw.max()),
             "%-24s : %10.4f" % ("Cumulative steps (max bead)", step_max_bead_cumm),
             "%-24s : %10.4f" % ("Cumulative steps (ts estim)", step_ts_estim_cumm),
             "%-24s : %8s %10.4f" % ("Bead Sep ratio (Pythagorean)", "|", seps_pythag.max() / seps_pythag.min()),

             "VALUES FOR SINGLE BEADS",
             "%-24s : %s" % ("Bead Energies",format('%10.4f', e_beads)) ,
             "%-24s : %s" % ("RMS Perp Forces", format(f, rmsf_perp_beads)),
             "%-24s : %s" % ("Para Forces", format(f, rmsf_para_beads)),
             "%-24s : %s" % ("MAX Forces", format(f, maxf_beads)),
             "%-24s : %s" % ("RMS Step Size", format(f, step_beads)),
             "%-24s : %12s %s |" % ("Bead Angles","|" , format('%10.0f', angles)),
             "%-24s : %6s %s" % ("Bead Separations (Pythagorean)", "|", format(f, seps_pythag)),
             "%-24s : %s" % ("Bead Path Position", format(f, path_pos)),
             "%-24s :" % ("Raw State Vector"),
             all_coordinates,
             "GENERAL STATS",
             "%-24s : %10d" % ("Callbacks", self.callbacks),
             "%-24s : %10d" % ("Beads Count", self.beads_count),
             "%-24s : %.4f %.4f" % ("Total Length (Pythag|Spline)", total_len_pythag, total_len_spline),
             "%-24s : %10s" % ("State Summary (total)", state_sum),
             "%-24s : %s" % ("State Summary (beads)", format('%10s', beads_sum)),
             "%-24s : %10.4f | %10.4f " % ("Barriers (Fwd|Rev)", barrier_fwd, barrier_rev)]

        if self.output_level > 2:
            s += ["Archive %s" % arc]

        if self.arc_record and self.output_level > 2:
            assert type(self.arc_record) == file, type(self.arc_record)
            sv = self.state_vec.reshape(self.beads_count,-1)
            arc['state_vec'] = sv
            arc['energies']  = self.bead_pes_energies.reshape(-1)
            arc['gradients'] = self.bead_pes_gradients.reshape(self.beads_count, -1)
            arc['pathps'] = self.pathpos()[0]
            arc['pathpsold'] = self.pathpos()[1]

            pickle.dump(arc, self.arc_record, protocol=2)
            self.arc_record.flush()

        return '\n'.join(s)

    def post_obj_func(self, grad):
        if self.reporting:
            if grad:
                self.reporting.write("***Gradient call (E was %f)***\n" % self.bead_pes_energies.sum())
            else:
                self.reporting.write("***Energy call   (E was %f)***\n" % self.bead_pes_energies.sum())

        if self.prev_state == None:
            # must be first iteration
            assert self.eg_calls == 0
            self.prev_state = self.state_vec.copy()

        # skip reporting if the state hasn't changed
        elif (self.state_vec == self.prev_state).all() and self.beads_count == self.prev_beads_count:
            return

        self.eg_calls += 1
        self._step = self.state_vec - self.prev_state
        self.prev_state = self.state_vec.copy()

        self.record()

        if self.reporting:
            s = [str(self),
                 common.line()]


            s = '\n'.join(s) + '\n'

            self.reporting.write(s)
            self.reporting.flush()
        else:
            assert self.reporting == None

        self.prev_beads_count = self.beads_count


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
        seps = common.pythag_seps(v)
        self.bead_separations = seps
        return seps

    # commented out 06/05/2010 -> not needed? YES NEEDED
    def pathfs(self):
        """Used for plotting Analytical PES's"""
        return None
        return self.bead_separations

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
        tmp.shape = (self.beads_count, self.dimension)
        return tmp

    def get_maxit(self):
        return self._maxit
    def set_maxit(self, new):
        assert new >= 0
        self._maxit = new
    maxit = property(get_maxit, set_maxit)

    def get_state_vec(self):
        return self._state_vec.copy()
    def set_state_vec(self, x):
        if x != None:
            tmp = array(x).reshape(self.beads_count, -1)

            for i in range(self.beads_count):
                if self.bead_update_mask[i]:
                    self._state_vec[i] = tmp[i]

    state_vec = property(get_state_vec, set_state_vec)

    def obj_func(self, new_state_vec=None, grad=False):
        # diffinbeads is only needed for growing string, for everything else it
        # can be set to zero

        # update mask of beads to freeze i.e. not move or recalc energy/grad for
        #if self.freeze_beads:
        #    self.update_mask()

        # NOTE: this automatically skips if new_state_vec == None
        self.state_vec = new_state_vec

#        self.reporting.write("self.prev_state updated")
#        self.prev_state = self.state_vec.copy()

        if self.eg_calls >= self.maxit:
            raise pts.MaxIterations

        print "Objective function call: gradient = %s" % grad
        # tests whether this has already been requested
        print "Objective function call: Previously made calcs:", test_previous(self.state_vec)
        print "Objective function call: Bead update mask:     ", self.bead_update_mask
        print self.state_vec

        if self.bead_eg_calls == 0:
            self.bead_eg_calls += self.beads_count
        else:
            self.bead_eg_calls += self.beads_count - 2

         # request and process parallel QC jobs
        if self.parallel:

            for i in range(self.beads_count):
                # if request of gradients are given, give also the number of the bead
                self.qc_driver.request_gradient(self.state_vec[i], self.get_final_bead_ix(i))

            self.qc_driver.proc_requests()

        self.update_bead_pes_energies()
        self.update_bead_separations()
        self.update_tangents()

        # get PES forces / project out stuff
        for i in range(self.beads_count):

            g = self.qc_driver.gradient(self.state_vec[i])
            self.bead_pes_gradients[i] = g.copy()

            t = self.tangents[i]
            perp_force, para_force = project_out(t, -g)

            self.perp_bead_forces[i] = perp_force
            self.para_bead_forces[i] = para_force

        self.post_obj_func(grad)

    def exp_project(self, vector):
        """Exports projection functionality from CoS object."""
        N = self.beads_count
        assert vector.size >= self.tangents.size
        v = vector.reshape(-1).copy()
        v.resize(self.tangents.size)
        v.shape = (N,-1)

        for i in range(N):
            t = self.tangents[i]
            v[i], _ = project_out(t, v[i])

        return v
       
    def get_final_bead_ix(self, i):
        """
        Based on bead index |i|, returns final index once string is fully 
        grown.
        """
        return i

    def record(self):
        """Records snap-shot of chain."""
        es = self.bead_pes_energies.copy()

        state = self.state_vec.copy()
        perp_forces = self.perp_bead_forces.copy()
        para_forces = self.para_bead_forces.copy()
        ts_estim = self.ts_estims()[-1]

        a = [es, state, perp_forces, para_forces, ts_estim]

        self.history.rec(a)
        
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

    def ts_estims(self, alsomodes = False, converter = None):
        """TODO: Maybe this whole function should be made external."""

        self.pt = pts.tools.PathTools(self.state_vec, self.bead_pes_energies, self.bead_pes_gradients)

        estims = self.pt.ts_splcub()
        f = open("tsplot.dat", "w")
        f.write(self.pt.plot_str)
        f.close()

        tsisspl = True
        if len(estims) < 1:
            lg.warn("No transition state found, using highest bead.")
            estims = self.pt.ts_highest()
            tsisspl = False
        estims.sort()
        if alsomodes:
            cx = converter
            bestmode = []
            for est in estims:
                energies, coords, s0, s1, s_ts,  l, r = est
                cx.set_internals(coords)
                modes =  self.pt.modeandcurvature(s_ts, l, r, cx)
                for mo in modes:
                    mo_name, value = mo
                    if tsisspl and mo_name=="frompath":
                        bestmode.append(value)
                    if not tsisspl and mo_name=="directinternal":
                        bestmode.append(value)
            return estims, bestmode
        return estims
       
        
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
    >>> qc = pts.pes.GaussianPES()
    >>> neb = NEB(path, qc, 1.0, beads_count = 4)
    >>> neb.state_vec
    array([[ 0. ,  0. ],
           [ 0.2,  0.2],
           [ 0.7,  0.7],
           [ 1. ,  1. ]])

    >>> neb.obj_func()
    Objective function call: gradient = False
    Objective function call: Previously made calcs: [True, False, False, True]
    Objective function call: Bead update mask:      [False, True, True, False]
    [[ 0.   0. ]
     [ 0.2  0.2]
     [ 0.7  0.7]
     [ 1.   1. ]]
    -2.6541709711655024

    >>> neb.obj_func_grad().round(3)
    Objective function call: gradient = True
    Objective function call: Previously made calcs: [True, True, True, True]
    Objective function call: Bead update mask:      [False, True, True, False]
    [[ 0.   0. ]
     [ 0.2  0.2]
     [ 0.7  0.7]
     [ 1.   1. ]]
    array([-0.   , -0.   , -0.291, -0.309,  0.327,  0.073, -0.   , -0.   ])

    >>> neb.step
    (0.0, array([ 0.,  0.,  0.,  0.]), array([[ 0.,  0.],
           [ 0.,  0.],
           [ 0.,  0.],
           [ 0.,  0.]]))

    >>> neb.obj_func_grad([[0,0],[0.3,0.3],[0.9,0.9],[1,1]]).round(3)
    Objective function call: gradient = True
    Objective function call: Previously made calcs: [True, True, True, True]
    Objective function call: Bead update mask:      [False, True, True, False]
    [[ 0.   0. ]
     [ 0.3  0.3]
     [ 0.9  0.9]
     [ 1.   1. ]]
    array([-0.   , -0.   , -0.282, -0.318,  0.714,  0.286, -0.   , -0.   ])

    >>> neb.step[1].round(1)
    array([ 0. ,  0.1,  0.2,  0. ])

    >>> neb.eg_calls
    2

    >>> neb = NEB([[0,0],[3,3]], pts.pes.GaussianPES(), 1., beads_count = 10)
    Starting slow function: _get_total_str_len()
    Finishing slow function: _get_total_str_len()
    >>> neb.angles
    array([ 180.,  180.,  180.,  180.,  180.,  180.,  180.,  180.])
    >>> neb.obj_func()
    Objective function call: gradient = False
    Objective function call: Previously made calcs: [True, False, False, False, False, False, False, False, False, False]
    Objective function call: Bead update mask:      [False, True, True, True, True, True, True, True, True, False]
    [[ 0.      0.    ]
     [ 0.3333  0.3333]
     [ 0.6666  0.6666]
     [ 0.9999  0.9999]
     [ 1.3329  1.3329]
     [ 1.6662  1.6662]
     [ 1.9995  1.9995]
     [ 2.3328  2.3328]
     [ 2.6661  2.6661]
     [ 3.      3.    ]]
    -4.5561921505021239
    >>> neb.tangents
    array([[ 0.70710678,  0.70710678],
           [ 0.70710678,  0.70710678],
           [ 0.70710678,  0.70710678],
           [ 0.70710678,  0.70710678],
           [ 0.70710678,  0.70710678],
           [ 0.70710678,  0.70710678],
           [ 0.70710678,  0.70710678],
           [ 0.70710678,  0.70710678],
           [ 0.70710678,  0.70710678],
           [ 0.70710678,  0.70710678]])

    >>> neb.state_vec += 0.1
    >>> abs(neb.step[1] - ones(neb.beads_count) * 0.1).round()
    array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.])

    >>> neb = NEB([[0,0],[1,1]], pts.pes.GaussianPES(), 1., beads_count = 3)
    Starting slow function: _get_total_str_len()
    Finishing slow function: _get_total_str_len()
    >>> neb.angles
    array([ 180.])
    >>> neb.obj_func([[0,0],[0,1],[1,1]])
    Objective function call: gradient = False
    Objective function call: Previously made calcs: [True, False, True]
    Objective function call: Bead update mask:      [False, True, False]
    [[ 0.  0.]
     [ 0.  1.]
     [ 1.  1.]]
    -1.6878414761432885
    >>> neb.tangents
    array([[ 0.,  1.],
           [ 1.,  0.],
           [ 1.,  0.]])
    >>> neb.bead_pes_energies = array([0,1,0])

    >>> neb.update_tangents()
    >>> neb.tangents
    array([[ 0.        ,  1.        ],
           [ 0.70710678,  0.70710678],
           [ 1.        ,  0.        ]])
    >>> neb.bead_pes_energies = array([1,0,-1])
    >>> neb.update_tangents()
    >>> neb.tangents
    array([[ 0.,  1.],
           [ 0.,  1.],
           [ 1.,  0.]])

    """

    growing = False
    def __init__(self, reagents, qc_driver, base_spr_const, beads_count=10, 
        parallel=False, reporting=None, output_level = 3):

        ReactionPathway.__init__(self, reagents, beads_count, qc_driver, 
            parallel=parallel, reporting=reporting, output_level = output_level)

        self.base_spr_const = base_spr_const

        # Make list of spring constants for every inter-bead separation
        # For the time being, these are uniform
        self.spr_const_vec = array([self.base_spr_const for x in range(beads_count - 1)])

        self.use_upwinding_tangent = True

        # Generate or copy initial path
        if len(reagents) == beads_count:
            self._state_vec = array(reagents)
        else:
            pr = PathRepresentation(reagents, beads_count, lambda x: 1)
            pr.regen_path_func()
            pr.generate_beads(update = True)
            self._state_vec = pr.state_vec.copy()

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

        for i in range(self.beads_count):
            self.tangents[i] /= linalg.norm(self.tangents[i], 2)

#        print "self.tangents", self.tangents

    def __len__(self):
        """For compatibility with ASE, pretends that there are atoms with cartesian coordinates."""
        return int(ceil((self.beads_count * self.dimension / 3.)))

    def pathpos(self):
        return (None, None)

    def get_forces(self):
        """For compatibility with ASE, pretends that there are atoms with cartesian coordinates."""
        return -common.make_like_atoms(self.obj_func_grad())

    def get_potential_energy(self):
        """For compatibility with ASE, pretends that there are atoms with cartesian coordinates."""
        return self.obj_func()


    def obj_func_grad(self, new_state_vec = None):
        ReactionPathway.obj_func(self, new_state_vec, grad=True)

        result_bead_forces = zeros((self.beads_count, self.dimension))
#        print "pbf", self.perp_bead_forces.reshape((-1,2))
        for i in range(self.beads_count):
            if not self.bead_update_mask[i]:
                continue

            spring_force_mag = 0
            # no spring force for end beads
            if i > 0 and i < self.beads_count - 1:
                dx1 = self.bead_separations[i]
                dx0 = self.bead_separations[i-1]
                spring_force_mag = self.base_spr_const * (dx1 - dx0)

            spring_force = spring_force_mag * self.tangents[i]

            total = self.perp_bead_forces[i] + spring_force
#            print "spring", spring_force

            result_bead_forces[i] = total

        g = -result_bead_forces.flatten()
        return g

    def obj_func(self, new_state_vec = None, grad=False):

        ReactionPathway.obj_func(self, new_state_vec)

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

"""
class QuadFunc(Func):
    def __init__(self, coefficients):
        self.coefficients = coefficients

    def f(self, x):
        return dot(array((x**2, x, 1)), self.coefficients)

    def fprime(self, x):
        return 2 * self.coefficients[0] * x + self.coefficients[1]
"""

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
        self.__old_normalised_positions = self.__normalised_positions

        self.__max_integral_error = 1e-4

        self.__rho = self.set_rho(rho)

        self._funcs_stale = True
        self._integrals_stale = True

        # TODO check all beads have same dimensionality

    """def get_fs(self):
        return self.__fs"""

    @property
    def fs(self):
        return self.__fs
    @property
    def path_tangents(self):
        return self.__path_tangents

    def positions_on_string(self):
        return self.__normalised_positions

    #def recalc_path_tangents(self):
        """Returns the unit tangents to the path at the current set of 
        normalised positions."""

    """        tangents = []
        for str_pos in self.__normalised_positions:
            tangents.append(self.__get_tangent(str_pos))

        tangents = array(tangents)
        return tangents"""

    def get_state_vec(self):
        return self.__state_vec

    def set_state_vec(self, new_state_vec):
        self.__state_vec = array(new_state_vec).reshape(self.beads_count, -1)
        self._funcs_stale = True
        self._integrals_stale = True

    state_vec = property(get_state_vec, set_state_vec)

    __path_len = -1
    @property
    def path_len(self):
        return self.__path_len

    @property
    def dimension(self):
        return self.__dimension

    def regen_path_func(self, normalised_positions=None):
        """Rebuild a new path function and the derivative of the path based on 
        the contents of state_vec."""

        assert self._funcs_stale or (not normalised_positions is None), self._funcs_stale

        if not normalised_positions is None:
            assert len(normalised_positions) == self.beads_count
            self.__normalised_positions = normalised_positions

        assert len(self.__state_vec) > 1

        self.__fs = []

#        assert self.__state_vec.shape == (self.beads_count, self.__dimension), "%s" % str(self.__state_vec.shape)
        for i in range(self.__dimension):

            ys = self.__state_vec[:,i]

            # linear path
            if len(self.__state_vec) == 2:
                self.__fs.append(LinFunc(self.__unit_interval, ys))

            # parabolic path
            elif len(self.__state_vec) == 3:

                # FIXME: at present, transition state assumed to be half way ebtween reacts and prods
                ps = array((0.0, 0.5, 1.0))

                self.__fs.append(QuadFunc(ps, ys))

            else:
                # spline path
                xs = self.__normalised_positions
                assert len(self.__normalised_positions) == len(self.__state_vec)
                self.__fs.append(SplineFunc(xs,ys))

        (str_len_precise, error) = scipy.integrate.quad(self.__arc_dist_func, 0, 1, limit=100)
        lg.debug("String length integration error = " + str(error))
        assert error < self.__max_integral_error, "error = %f" % error

        self.__path_len = str_len_precise

        self._funcs_stale = False


    def __arc_dist_func(self, x):
        output = 0
        for a in self.__fs:
            output += a.fprime(x)**2
        return sqrt(output)

    def get_bead_separations(self):
        """Returns the arc length between beads according to the current 
        parameterisation.
        """

        assert not self._funcs_stale
        if self._integrals_stale:
            a = self.__normalised_positions
            N = len(a)
            seps = []
            for i in range(N)[1:]:
                l, _ = scipy.integrate.quad(self.__arc_dist_func, a[i-1], a[i])
                seps.append(l)

            self.seps = array(seps)

            self._integrals_stale = False

        return self.seps

    def _get_total_str_len(self):
        """Returns a duple of the total length of the string and a list of 
        pairs (x,y), where x a distance along the normalised path (i.e. on 
        [0,1]) and y is the corresponding distance along the string (i.e. on
        [0,string_len])."""
        
        print "Starting slow function: _get_total_str_len()"
        # number of points to chop the string into
        param_steps = arange(0, 1, self.__step)

        list = []
        cummulative = 0

        for i in range(self.__str_resolution):
            pos = (i + 0.5) * self.__step
            sub_integral = self.__step * self.__arc_dist_func(pos)
            cummulative += sub_integral
            list.append(cummulative)

        #lg.debug('int_approx = {0}, int_accurate = {1}'.format(cummulative, str_len_precise))
        err = abs(list[-1] - self.__path_len)
        assert err < 1e-4, "%e" % err

        print "Finishing slow function: _get_total_str_len()"

        return (list[-1], zip(param_steps, list))

    def sub_str_lengths(self, normd_poses):
        # delete this function? 07/05/10
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

    def generate_beads(self, update = False, update_mask=None):
        """Returns an array of the self.__beads_count vectors of the coordinates 
        of beads along a reaction path, according to the established path 
        (line, parabola or spline) and the parameterisation density."""

        assert len(self.__fs) > 1

        assert not self._funcs_stale

        # Find total string length and incremental distances x along the string 
        # in terms of the normalised coodinate y, as a list of (x,y).
        (total_str_len, incremental_positions) = self._get_total_str_len()

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
            # OLD: self.state_vec = bead_vectors
            self.state_vec = masked_assign(update_mask, self.state_vec, bead_vectors)

            # OLD: self.__path_tangents = bead_tangents
            self.__path_tangents = masked_assign(update_mask, self.__path_tangents, bead_tangents)

            self.__old_normalised_positions = self.__normalised_positions
            self.__normalised_positions = array([0.0] + normd_positions + [1.0])

        return bead_vectors

    def update_tangents(self):
        assert not self._funcs_stale
        ts = []
        for i in self.__normalised_positions:
             ts.append(self.__get_tangent(i))
 
        self.__path_tangents = array(ts)
        
    def dump_rho(self):
        res = 0.02
        print "rho: ",
        for x in arange(0.0, 1.0 + res, res):
            if x < 1.0:
                print self.__rho(x),
        print
        raw_input("that was rho...")

    def pathpos(self):
        return (self.__normalised_positions, self.__old_normalised_positions)

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
            integral = 1.0
        self.__rho = lambda x: new_rho(x) / integral
        return self.__rho

          
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


def get_bead_positions(E, ps):
    points = arange(ps[0], ps[-1], (ps[-1] - ps[0]) / 20.0)

    Es = array([E(p) for p in points])
    p_max = points[Es.argmax()]

    new_p = -1
    for i in range(len(ps))[1:]:
        if ps[i] > p_max:
            new_p = ps[i-1] + (ps[i] - ps[i-1]) / 2.0

            new_i = i

            break

    assert new_p > 0, new_p
    
    psa = ps.tolist()
    new_ps = psa[:new_i] + [new_p] + psa[new_i:]
    assert (array(new_ps).argsort() == arange(len(new_ps))).all()

    return array(new_ps), new_i
            
def get_bead_positions_grad(Es, gradients, tangents, ps):

    # perform safety check
    bad = 0
    for i, p in list(enumerate(ps))[1:-1]:
        dxds = tangents[i]
        dEdx = gradients[i]
        dEds = numpy.dot(dEdx, dxds)

        if Es[i-1] < Es[i] < Es[i+1] and dEds < 0:
            bad += 1
        elif Es[i-1] > Es[i] > Es[i+1] and dEds > 0:
            bad += 1

    if bad:
        print "WARNING: %d errors in calculated gradients along path, get_bead_positions_grad() might give unreliable results" % bad
#        assert False # for the time being...

    i_max = Es.argmax()
    p_max = ps[i_max]
    dEdx = gradients[i_max]

    dxdp = tangents[i_max]

    dEdp = numpy.dot(dEdx, dxdp)

    if dEdp < 0:
        new_p = (ps[i_max-1] + ps[i_max]) / 2.0
        new_i = i_max
    else:
        new_p = (ps[i_max] + ps[i_max+1]) / 2.0
        new_i = i_max+1

    psa = ps.tolist()
    new_ps = psa[:new_i] + [new_p] + psa[new_i:]
    assert (array(new_ps).argsort() == arange(len(new_ps))).all()

    return array(new_ps), new_i


class GrowingString(ReactionPathway):
    """Implements growing and non-growing strings.

    >>> path = [[0,0],[0.2,0.2],[0.7,0.7],[1,1]]
    >>> qc = pts.pes.GaussianPES()
    >>> s = GrowingString(path, qc, beads_count=4, growing=False)
    Starting slow function: _get_total_str_len()
    Finishing slow function: _get_total_str_len()
    [[ 0.   0. ]
     [ 0.2  0.2]
     [ 0.7  0.7]
     [ 1.   1. ]]
    [[ 0.          0.        ]
     [ 0.32149034  0.32149034]
     [ 0.660736    0.660736  ]
     [ 1.          1.        ]]
    None
    >>> s.state_vec.round(1)
    array([[ 0. ,  0. ],
           [ 0.3,  0.3],
           [ 0.7,  0.7],
           [ 1. ,  1. ]])

    >>> new = s.state_vec.round(2).copy()
    >>> s.obj_func()
    Objective function call: gradient = False
    Objective function call: Previously made calcs: [False, False, False, False]
    Objective function call: Bead update mask:      [False, True, True, False]
    [[ 0.          0.        ]
     [ 0.32149034  0.32149034]
     [ 0.660736    0.660736  ]
     [ 1.          1.        ]]
    -2.5884273157684441

    >>> s.obj_func_grad().round(3)
    Objective function call: gradient = True
    Objective function call: Previously made calcs: [True, True, True, True]
    Objective function call: Bead update mask:      [False, True, True, False]
    [[ 0.          0.        ]
     [ 0.32149034  0.32149034]
     [ 0.660736    0.660736  ]
     [ 1.          1.        ]]
    result_bead_forces [[ 0.          0.        ]
     [ 0.02061132 -0.02061132]
     [ 0.11027833 -0.11027833]
     [ 0.          0.        ]]
    array([ 0.   ,  0.   ,  0.021, -0.021,  0.11 , -0.11 ,  0.   ,  0.   ])

    >>> s.step
    (0.0, array([ 0.,  0.,  0.,  0.]), array([[ 0.,  0.],
           [ 0.,  0.],
           [ 0.,  0.],
           [ 0.,  0.]]))

    >>> s.obj_func_grad(new)
    Objective function call: gradient = True
    Objective function call: Previously made calcs: [True, False, False, True]
    Objective function call: Bead update mask:      [False, True, True, False]
    [[ 0.    0.  ]
     [ 0.32  0.32]
     [ 0.66  0.66]
     [ 1.    1.  ]]
    result_bead_forces [[ 0.          0.        ]
     [ 0.02041863 -0.02041863]
     [ 0.10998242 -0.10998242]
     [ 0.          0.        ]]
    array([ 0.        ,  0.        ,  0.02041863, -0.02041863,  0.10998242, -0.10998242,  0.        ,  0.        ])
    >>> array(s.step[1])
    array([ 0.        ,  0.00149034,  0.000736  ,  0.        ])

    >>> s.obj_func_grad([[0,0],[0.3,0.3],[0.9,0.9],[1,1]]).round(3)
    Objective function call: gradient = True
    Objective function call: Previously made calcs: [True, False, False, True]
    Objective function call: Bead update mask:      [False, True, True, False]
    [[ 0.   0. ]
     [ 0.3  0.3]
     [ 0.9  0.9]
     [ 1.   1. ]]
    result_bead_forces [[ 0.          0.        ]
     [ 0.01796772 -0.01796772]
     [ 0.21435123 -0.21435123]
     [ 0.          0.        ]]
    array([ 0.   ,  0.   ,  0.018, -0.018,  0.214, -0.214,  0.   ,  0.   ])
    >>> s.lengths_disparate()
    True

    >>> s.eg_calls
    3

    """

    string = True

    def __init__(self, reagents, qc_driver, beads_count = 10, 
        rho = lambda x: 1, growing=True, parallel=False, head_size=None, output_level = 3,
        max_sep_ratio = 0.1, reporting=None, growth_mode='normal', freeze_beads=False):

        self.__qc_driver = qc_driver

        self.__final_beads_count = beads_count

        self.growing = growing
        if growing:
            initial_beads_count = 4
        else:
            initial_beads_count = self.__final_beads_count

        # create PathRepresentation object
        self._path_rep = PathRepresentation(reagents, initial_beads_count, rho)
        ReactionPathway.__init__(self, reagents, initial_beads_count, qc_driver, parallel, reporting=reporting, output_level = output_level)

        # final bead spacing density function for grown string
        # make sure it is normalised
        (int, err) = scipy.integrate.quad(rho, 0.0, 1.0)
        self.__final_rho = lambda x: rho(x) / int

        # setup growth method
        self.growth_funcs = {
            'normal': (self.grow_string_normal, self.update_rho),
            'search': (self.grow_string_search, self.search_string_init)
            }
        if not growth_mode in self.growth_funcs:
            t = growth_mode, str(self.growth_funcs.keys())
            msg = 'Unrecognised growth mode: %s, possible values are %s' % t
            raise Exception(msg)

        # TODO: this is a bit of a hack, can it be done better?
        if growth_mode == 'normal':
            self.get_final_bead_ix = self.get_final_bead_ix_classic_grow

        (self.grow_string, init) = self.growth_funcs[growth_mode]
        init()

        # Build path function based on reagents
        self._path_rep.regen_path_func()

        # Space beads along the path
        self._path_rep.generate_beads(update = True)
        self._path_rep.regen_path_func()

        self.parallel = parallel

        # Number of beads in growing heads of growing string to calculate 
        # gradients for. All others become frozen.
        #assert head_size == None, "Not yet properly implemented, problem when beads are respaced."
        assert head_size == None or (growing and beads_count / 2 - 1 >= head_size > 0 and freeze_beads)
        self.head_size = head_size

        # maximum allowed ratio between (max bead sep - min bead sep) and (average bead sep)
        self.__max_sep_ratio = max_sep_ratio

        self.freeze_beads = freeze_beads


    def search_string_init(self):
        self.bead_positions = arange(self.beads_count) * 1.0 / (self.beads_count - 1.0)

    def __len__(self):
        """For compatibility with ASE, pretends that there are atoms with cartesian coordinates."""

        return len(common.make_like_atoms(self.state_vec))

    def pathfs(self):
        """Used for plotting Analytical PES's."""
        return self._path_rep.fs

    def pathpos(self):
        return self._path_rep.pathpos()

    def path_tuple(self):
        state, energies, gradients, (pathps, pathpsold) = \
            (self.state_vec.reshape(self.beads_count,-1), \
            self.bead_pes_energies.reshape(-1), \
            self.bead_pes_gradients.reshape(self.beads_count,-1), \
            self.pathpos())

        return state, energies, gradients, pathps, pathpsold

    def get_state_vec(self):
        assert not '_state_vec' in self.__dict__
        return self._path_rep.state_vec.copy()

    def set_state_vec(self, x):
        assert not '_state_vec' in self.__dict__

        if x != None:

            self.update_path(x, respace = False)

#            self._path_rep.state_vec = array(x).reshape(self.beads_count, -1)

    state_vec = property(get_state_vec, set_state_vec)

    def update_tangents(self):
        self.tangents = self._path_rep.path_tangents

    def get_forces(self):
        """For compatibility with ASE, pretends that there are atoms with cartesian coordinates."""
        return -common.make_like_atoms(self.obj_func_grad())

    def get_potential_energy(self):
        """For compatibility with ASE, pretends that there are atoms with cartesian coordinates."""
        return self.obj_func()

    def get_beads_count(self):
        return self._path_rep.beads_count

    def set_beads_count(self, new):
        self._path_rep.beads_count = new

    beads_count = property(get_beads_count, set_beads_count)

    def expand_internal_arrays(self, size):
        # arrays which must be expanded if string is grown

        # Grr, no pointers in python, but can this be done better?
        self.bead_pes_gradients = self.expand(self.bead_pes_gradients, size)
        self.perp_bead_forces = self.expand(self.perp_bead_forces, size)
        self.para_bead_forces = self.expand(self.para_bead_forces, size)
        self.bead_pes_energies = self.expand(self.bead_pes_energies, size)

    def expand(self, old, new_size):

        assert len(old) % 2 == 0

        old.shape = (len(old), -1)
        new = zeros(old.shape[1] * new_size).reshape(new_size, -1)
        for i in range(len(old) / 2):
            new[i] = old[i]
            new[-i] = old[-i]

        if new.shape[1] == 1:
            new.shape = -1
        return new

    def get_final_bead_ix_classic_grow(self, i):
        """
        Based on bead index |i|, returns final index once string is fully 
        grown. This is only valid for 'classic' ends-inwards growth.
        """
        if self.growing and not self.grown():
            assert self.beads_count % 2 == 0
            end = self.beads_count / 2
            if i >= end:
                gap = self.__final_beads_count - self.beads_count
                return i + gap

        return i

    def grown(self, max_beads_equiv=13):
        """Returns true if the number of beads has reached the max allowed 
        number, or if the interbead spacing has become smaller than would be
        experienced if there were max_beads_equiv beads.
        """

        ps = self.bead_positions
        diffs = [ps[i+1] - ps[i] for i in range(self.beads_count-1)]
        print "grown(): Difference in bead spacings:", diffs
        min_diff = min(diffs)
        too_closely_spaced = min_diff < 1.0 / (max_beads_equiv - 1.0)

        return self.beads_count == self.__final_beads_count or too_closely_spaced

    def grow_string_search(self, energy_only=False):
        assert self.beads_count <= self.__final_beads_count

        if self.grown():
            return False
        else:
            self.beads_count += 1

        
        # Build piecewise bead density function
        _, es = self.energies
        path = Path(es, self.bead_positions)
        if energy_only:
            self.bead_positions, new_i = get_bead_positions(path, self.bead_positions)
        else:
            self.bead_positions, new_i = get_bead_positions_grad(self.bead_pes_energies, self.bead_pes_gradients, self.tangents, self.bead_positions)
        new_ixs = (new_i,)

        if new_i == self.beads_count - 2:
            moving_beads = [new_i-2, new_i-1, new_i]
        elif new_i == 0:
            moving_beads = [new_i+1, new_i+2, new_i+3]
        else:
            moving_beads = [new_i-1, new_i, new_i+1]
        

        rho = RhoInterval(self.bead_positions).f
        self._path_rep.set_rho(rho)

        # The following block of code ensures that all beads other than the
        # newly added one stay in exactly the same position. Otherwise, 
        # numerical inaccuraties cause them to move and additional beads
        # to have their energies calculated.
        old = self.state_vec.copy()
        new = self._path_rep.generate_beads(update = True)
        minimal_update(new, old, new_ixs) # only updates new_ixs
        self.bead_update_mask = freeze_ends(self.beads_count)
        self.state_vec = new
        print "old", old
        print "new", new
        print "state_vec", self.state_vec

        # Mask of beads to freeze, includes only end beads at present
        if self.freeze_beads:
            self.bead_update_mask = [i in moving_beads for i in range(self.beads_count)]
        else:
            e = 1
            m = self.beads_count - 2 * e
            self.bead_update_mask = [False for i in range(e)] + [True for i in range(m)] + [False for i in range(e)]

        # HCM 26/05/10: following line now done by self.state_vec assignment
        #self._path_rep.regen_path_func() #TODO: for grow class as well


        # HCM 06/05/10: the following line does nothing, see definition of setter for self.beads_count
        # self._path_rep.beads_count = self.beads_count

        self.initialise()

        lg.info("******** String Grown to %d beads ********", self.beads_count)

        self.prev_state = self.state_vec.copy()

        return True

    def grow_string_normal(self):
        """
        Adds 2, 1 or 0 beads to string (such that the total number of 
        beads is less than or equal to self.__final_beads_count).
        """

        assert self.beads_count <= self.__final_beads_count

        if self.grown():
            return False
        elif self.__final_beads_count - self.beads_count == 1:
            new_ixs = (self.beads_count / 2,)
            self.beads_count += 1
        else:
            i = self.beads_count / 2
            new_ixs = (i, i+1)
            self.beads_count += 2


        #self.expand_internal_arrays(self.beads_count)

        # Does nothing? self.beads_count references self._path_rep.beads_count anyway. - HCM 26/05/10
        self._path_rep.beads_count = self.beads_count

        # build new bead density function based on updated number of beads
        self.update_rho()
        old = self.state_vec.copy()
        new = self._path_rep.generate_beads(update = True)
        minimal_update(new, old, new_ixs)

        # create a new bead_update_mask that permits us to set the state to what we want
        self.bead_update_mask = freeze_ends(self.beads_count)
        self.state_vec = new
        print "old", old
        print "new", new
        print "state_vec", self.state_vec

        self.initialise()

        self.prev_state = self.state_vec.copy()

        # Build mask of beads to calculate, effectively freezing some if 
        # head_size, i.e. the size of the growing heads, is specified.
        if self.freeze_beads:
            e = self.beads_count / 2 - self.head_size + self.beads_count % 2
        else:
            e = 1
        m = self.beads_count - 2 * e
        self.bead_update_mask = [False for i in range(e)] + [True for i in range(m)] + [False for i in range(e)]
        lg.info("Bead Freezing MASK: " + str(self.bead_update_mask))

        print "******** String Grown to %d beads ********" % self.beads_count
        return True

    def update_rho(self):
        """Update the density function so that it will deliver beads spaced
        as for a growing (or grown) string."""

        assert self.beads_count <= self.__final_beads_count

        from scipy.optimize import fmin
        from scipy.integrate import quad

        # Set up vector of fractional positions along the string.
        # TODO: this function should be unified with the grow_string_search()
        # at some point.
        fbc = self.__final_beads_count
        all_bead_ps = arange(fbc) * 1.0 / (fbc - 1)
        end = self.beads_count / 2.0
        self.bead_positions = array([all_bead_ps[i] for i in range(len(all_bead_ps)) \
            if i < end or i >= (fbc - end)])

        if self.beads_count == self.__final_beads_count:
            self._path_rep.set_rho(self.__final_rho)

            return self.__final_rho

        assert self.beads_count % 2 == 0

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
        self._path_rep.set_rho(pwr.f)


    def lengths_disparate(self):
        """Returns true if the ratio between the (difference of longest and 
        shortest segments) to the average segment length is above a certain 
        value (self.__max_sep_ratio).

        FIXME: Is this description out of date?
        """

        seps = self._path_rep.get_bead_separations()
        assert len(seps) == self.beads_count - 1

        seps_ = zeros(seps.shape)
        seps_[0] = seps[0]
        for i in range(len(seps))[1:]:
            seps_[i] = seps[i] + seps_[i-1]

        assert len(seps_) == len(self.bead_positions) - 1, "%s\n%s" % (seps_, self.bead_positions)
        diffs = (self.bead_positions[1:] - seps_/seps.sum())

        return diffs.max() > self.__max_sep_ratio


    def obj_func(self, new_state_vec=None, individual=False):
        # growing string object needs to know how many beads should be added
        # for knowing the correct bead number _____AN
#        diffbeads = self.__final_beads_count - self.beads_count
        ReactionPathway.obj_func(self, new_state_vec)

        if individual:
            return self.bead_pes_energies
        else:
            return self.bead_pes_energies.sum()

    def obj_func_grad(self, new_state_vec = None, raw=False):
        # growing string object needs to know how many beads should be added
        # for knowing the correct bead number _____AN
        #diffbeads = self.__final_beads_count - self.beads_count
        ReactionPathway.obj_func(self, new_state_vec, grad=True)

        result_bead_forces = zeros((self.beads_count, self.dimension))
        if raw:
            from_array = self.bead_pes_gradients
        else:
            from_array = -self.perp_bead_forces

        for i in range(self.beads_count):
            if self.bead_update_mask[i]:
                result_bead_forces[i] = from_array[i]

        print "result_bead_forces", result_bead_forces
        g = result_bead_forces.flatten()
        return g

    def respace(self):
        self.update_path(respace=True)

    def update_path(self, state_vec=None, respace=True, smart_abscissa=True):
        """After each iteration of the optimiser this function must be called.
        It rebuilds a new (spline) representation of the path and then 
        redestributes the beads according to the density function."""

        if state_vec != None:
            new = array(state_vec).reshape(self.beads_count, -1)
            assert new.size == self.state_vec.size

            tmp = self._path_rep.state_vec.copy()

            for i in range(self.beads_count):
                if self.bead_update_mask[i]:
                    tmp[i] = new[i]
            self._path_rep.state_vec = tmp # was '= new' until 07/05/10

            # rebuild line, parabola or spline representation of path
            self._path_rep.regen_path_func()

        # respace the beads along the path
        if respace:
            if smart_abscissa:
                pythag_seps = common.pythag_seps(self.state_vec)

                if len(self.state_vec) > 4:
                    # This is a kind of hack to prevent the bits of
                    # spline between the end beads from becoming too curved.
                    pythag_seps[0] *= 0.7
                    pythag_seps[-1] *= 0.7
                new_abscissa = cumm_sum(pythag_seps, start_at_zero=True)
                new_abscissa /= new_abscissa[-1]
                self._path_rep.regen_path_func(normalised_positions=new_abscissa)

            self._path_rep.generate_beads(update=True, update_mask=self.bead_update_mask)

            # The following line ensure that the path used for the next 
            # step is the same as the one that is generated, at a later date,
            # based on an outputted path pickle.
            self._path_rep.regen_path_func()

            self.respaces += 1

        # TODO: this must eventually be done somewhere behind the scenes.
        # I.e. Why would one ever want to update the path but not the tangents?
        self._path_rep.update_tangents()


    # never used? commented out 06/05/2010
    #def plot(self):
#        self._path_rep.generate_beads_exact()
    #    plot2D(self._path_rep)

def project_out(component_to_remove, vector):
    """Projects the component of 'vector' that lies along 'component_to_remove'
    out of 'vector' and returns it."""
    projection = dot(component_to_remove, vector)
    removed = projection * component_to_remove
    output = vector - removed
    return output, projection



# Testing the examples in __doc__strings, execute
# "python gxmatrix.py", eventualy with "-v" option appended:
if __name__ == "__main__":
    import doctest
    doctest.testmod()

# You need to add "set modeline" and eventually "set modelines=5"
# to your ~/.vimrc for this to take effect.
# Dont (accidentally) delete these lines! Unless you do it intentionally ...
# Default options for vim:sw=4:expandtab:smarttab:autoindent:syntax


