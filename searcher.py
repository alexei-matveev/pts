#!/usr/bin/env python

import sys
import inspect

import scipy.integrate

from scipy import interpolate
import tempfile, os
import logging
from copy import deepcopy
import pickle
from os import path, mkdir, chdir, getcwd

# TODO: change to import numpy as np???
from numpy import linalg, array, asarray, ceil, abs, sqrt, dot, size
from numpy import empty, zeros, ones, linspace, arange

from path import Path, Arc, scatter, scatter1
from func import RhoInterval
from pts.memoize import Elemental_memoize

from common import * # TODO: must unify
import pts.common as common
import pts
import pts.metric as mt

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

def masked_assign(mask, dst, src):
    """Assigns y to x if mask allows it.

    >>> m = [True, True]
    >>> orig = arange(4).reshape(2,-1)
    >>> x = orig.copy()
    >>> y = array([5,4,3,2]).reshape(2,-1)
    >>> x = masked_assign(m, x, y)
    >>> (x == y).all()
    True
    >>> m = [False, False]
    >>> x = orig.copy()
    >>> (x != masked_assign(m, x, y)).all()
    False
    """

    if len(dst) != len(src):
        return src.copy()

    dstc = dst.copy()
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
            pes,
            parallel, 
            result_storage,
            reporting=None, 
            convergence_beads=3, 
            steps_cumm=3, 
            pmap=map,
            workhere = False,
            freeze_beads=False, 
            output_level = 3,
            output_path = ".",
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
        self.pes = pes
        self.beads_count = beads_count
        self.prev_beads_count = beads_count
        self.output_path = output_path

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

        self.allvals = Elemental_memoize(self.pes, pmap=pmap, cache = result_storage, workhere = workhere, format = "bead%02d")

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


    @property
    def state_summary(self):
        s = common.vec_summarise(self.state_vec)
        s_beads = [common.vec_summarise(b) for b in self.state_vec]
        return s, s_beads

    def path_tuple(self):
        state, energies, gradients, (pathps, pathpsold) = \
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

            bead_es = bead_gs = (self.beads_count - 2) * eg_calls + 2
            arc['bead_es'] = bead_es
            arc['bead_gs'] = bead_gs

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

    def obj_func(self, grad=False):
       ## NOTE: this automatically skips if new_state_vec == None
       #self.state_vec = new_state_vec

#        self.reporting.write("self.prev_state updated")
#        self.prev_state = self.state_vec.copy()

        if self.eg_calls >= self.maxit:
            raise pts.MaxIterations

#       print "Objective function call: gradient = %s" % grad
#       # tests whether this has already been requested
#       print "Objective function call: Previously made calcs:", test_previous(self.state_vec)
#       print "Objective function call: Bead update mask:     ", self.bead_update_mask
#       print self.state_vec

        if self.bead_eg_calls == 0:
            self.bead_eg_calls += self.beads_count
        else:
            self.bead_eg_calls += self.beads_count - 2

        # Note how these arrays are indexed below:
        assert len(self.bead_pes_energies) == self.beads_count
        assert len(self.bead_pes_gradients) == self.beads_count

        # calculation output should go to another place, thus change directory
        wopl = getcwd()
        if not path.exists(self.output_path):
            mkdir(self.output_path)
        chdir(self.output_path)


        # get PES energy/gradients
        es, gs = self.allvals.taylor( self.state_vec)

        # return to former directory
        chdir(wopl)

        # FIXME: does it need to be a a destructive update?
        self.bead_pes_energies[:] = es
        self.bead_pes_gradients[:] = gs

        #
        # NOTE: update_tangents() is not implemented by this class!
        #       Consult particular subclass.
        #
        #       Also note that at least in NEB the definition
        #       of the tangent may depend on the relative bead energies.
        #       Therefore update tangents only after self.bead_pes_energies
        #       was updated first. Not sure about bead separations though.
        #
        self.update_bead_separations()
        self.update_tangents()

        # project gradients in para/perp components:
        for i in range(self.beads_count):

            # previously computed gradient:
            g = self.bead_pes_gradients[i]

            # contravariant tangent coordiantes:
            T = self.tangents[i]

            # covariant tangent coordiantes:
            t = mt.metric.lower(T, self.state_vec[i])

            # components of force parallel and orthogonal to the tangent:
            para_force = - dot(T, g) / dot(T, t)
            perp_force = - g - para_force * t

            self.para_bead_forces[i] = para_force
            self.perp_bead_forces[i] = perp_force

        self.post_obj_func(grad)

    def set_positions(self, x):
        """For compatibility with ASE, pretends that there are atoms with cartesian coordinates."""

        tmp = x.flatten()[0:self.beads_count * self.dimension]
        self.state_vec = x.flatten()[0:self.beads_count * self.dimension]

    def get_positions(self):
        """For compatibility with ASE, pretends that there are atoms with cartesian coordinates.""" 

        return common.make_like_atoms(self.state_vec.copy())

    positions = property(get_positions, set_positions)

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

        # FIXME: wrong place? 
        ts_estim = ts_estims(self.state_vec, self.bead_pes_energies, self.bead_pes_gradients)[-1]

        a = [es, state, perp_forces, para_forces, ts_estim]

        self.history.rec(a)
        
def ts_estims(beads, energies, gradients, alsomodes = False, converter = None):
    """TODO: Maybe this whole function should be made external."""

    pt = pts.tools.PathTools(beads, energies, gradients)

    estims = pt.ts_splcub()
    f = open("tsplot.dat", "w")
    f.write(pt.plot_str)
    f.close()

    tsisspl = True
    if len(estims) < 1:
        lg.warn("No transition state found, using highest bead.")
        estims = pt.ts_highest()
        tsisspl = False
    estims.sort()
    if alsomodes:
        cx = converter
        bestmode = []
        for est in estims:
            energies, coords, s0, s1, s_ts,  l, r = est
            cx.set_internals(coords)
            modes =  pt.modeandcurvature(s_ts, l, r, cx)
            for mo in modes:
                mo_name, value = mo
                if tsisspl and mo_name=="frompath":
                    bestmode.append(value)
                if not tsisspl and mo_name=="directinternal":
                    bestmode.append(value)
        return estims, bestmode
    return estims

class NEB(ReactionPathway):
    """Implements a Nudged Elastic Band (NEB) transition state searcher.
    
    >>> path = [[0,0],[0.2,0.2],[0.7,0.7],[1,1]]
    >>> qc = pts.pes.GaussianPES()
    >>> neb = NEB(path, qc, 1.0, None, beads_count = 4)
    >>> neb.state_vec
    array([[ 0. ,  0. ],
           [ 0.2,  0.2],
           [ 0.7,  0.7],
           [ 1. ,  1. ]])

    >>> neb.obj_func()
    -2.6541709711655024

    Was changed because beads are put differently, was before:

    >>> neb.obj_func_grad().round(3)
    array([-0.   , -0.   , -0.291, -0.309,  0.327,  0.073, -0.   , -0.   ])

    >>> neb.step
    (0.0, array([ 0.,  0.,  0.,  0.]), array([[ 0.,  0.],
           [ 0.,  0.],
           [ 0.,  0.],
           [ 0.,  0.]]))

    >>> neb.state_vec = [[0,0],[0.3,0.3],[0.9,0.9],[1,1]]
    >>> neb.obj_func_grad().round(3)
    array([-0.   , -0.   , -0.282, -0.318,  0.714,  0.286, -0.   , -0.   ])

    >>> neb.step[1].round(1)
    array([ 0. ,  0.1,  0.2,  0. ])

    >>> neb.eg_calls
    2

    >>> neb = NEB([[0,0],[3,3]], pts.pes.GaussianPES(), 1., None, beads_count = 10)
    >>> neb.angles
    array([ 180.,  180.,  180.,  180.,  180.,  180.,  180.,  180.])
    >>> neb.obj_func()
    -4.5571068838569122

    Was changed because of different spacing of beads, was before
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

    >>> neb = NEB([[0,0],[1,1]], pts.pes.GaussianPES(), 1., None, beads_count = 3)
    >>> neb.angles
    array([ 180.])
    >>> neb.state_vec = [[0,0],[0,1],[1,1]]
    >>> neb.obj_func()
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
    def __init__(self, reagents, pes, base_spr_const, result_storage, beads_count=10, pmap = map,
        parallel=False, workhere = False, reporting=None, output_level = 3, output_path = "."):

        ReactionPathway.__init__(self, reagents, beads_count, pes, parallel, result_storage, pmap = pmap,
            reporting=reporting, output_level = output_level, output_path = output_path, workhere = workhere)

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
            pr.generate_beads(mt.metric)
            self._state_vec = pr.state_vec.copy()

    def update_tangents(self):
        """
        WARNING: uses self.bead_pes_energies to determine tangents
        """
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
            #self.tangents[i] /= linalg.norm(self.tangents[i], 2)
            self.tangents[i] /= mt.metric.norm_up(self.tangents[i], self.state_vec[i])

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


    def obj_func_grad(self):
        ReactionPathway.obj_func(self, grad=True)

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

    def obj_func(self, grad=False):

        ReactionPathway.obj_func(self)

        spring_energies = self.base_spr_const * self.bead_separations**2
        spring_energies = 0.5 * numpy.sum (spring_energies)
        lg.info("Spring energies %s" % spring_energies)
        return self.bead_pes_energies.sum()# + spring_energies

class PathRepresentation(Path):
    """Supports operations on a path represented by a line, parabola, or a 
    spline, depending on whether it has 2, 3 or > 3 points."""

    def __init__(self, state_vec, beads_count, rho = lambda s: 1.0):

        # vector of vectors defining the path
        self.__state_vec = array(state_vec)

        # number of vectors defining the path
        self.beads_count = beads_count

        # NOTE: here we assume that state_vec[i] is a 1D array:
        self.__dimension = len(state_vec[0])

        self.__path_tangents = []

        # generate initial paramaterisation density
        # TODO: Linear at present, perhaps change eventually
        self.__normalised_positions = linspace(0.0, 1.0, len(self.__state_vec))
        self.__old_normalised_positions = self.__normalised_positions.copy()

        self.set_rho(rho)

        self._funcs_stale = True
        self._integrals_stale = True

        # FIXME: please provide matric as an argument, explicit is better than
        # implicit:
        self.__metric = mt.metric

        # TODO check all beads have same dimensionality

        # use Path functionality:
        Path.__init__(self, self.__state_vec, self.__normalised_positions)


    @property
    def path_tangents(self):
        return self.__path_tangents

    def positions_on_string(self):
        return self.__normalised_positions

    def get_state_vec(self):
        return self.__state_vec

    def set_state_vec(self, new_state_vec):
        self.__state_vec = array(new_state_vec).reshape(self.beads_count, -1)
        self._funcs_stale = True
        self._integrals_stale = True

    state_vec = property(get_state_vec, set_state_vec)

    @property
    def dimension(self):
        return self.__dimension

    def regen_path_func(self, normalised_positions=None):
        """Rebuild a new path function and the derivative of the path based on 
        the contents of state_vec."""

        assert self._funcs_stale or (normalised_positions is not None), self._funcs_stale

        if not normalised_positions is None:
            assert len(normalised_positions) == self.beads_count
            self.__normalised_positions = normalised_positions

        assert len(self.__state_vec) > 1

        # use Path functionality, on setting nodes a new parametrizaiton is generated:
        self.nodes = self.__normalised_positions, self.__state_vec

        self._funcs_stale = False

    def __arc_dist_func(self, x, metric):

        value, tangent = self.taylor(x)

        return metric.norm_up(tangent, value)

    def get_bead_separations(self):
        """Returns the arc length between beads according to the current 
        parameterisation.
        """

        assert not self._funcs_stale
        if self._integrals_stale:
            a = self.__normalised_positions
            N = len(a)
            seps = []
            def arc_fun(x):
                # arc_dist_func needs also knowledge of some kind of metric
                return self.__arc_dist_func(x, self.__metric)

            for i in range(N)[1:]:
                l, _ = scipy.integrate.quad(arc_fun, a[i-1], a[i])
                seps.append(l)

            self.seps = array(seps)

            self._integrals_stale = False

        return self.seps

    def generate_beads(self, metric, update_mask=None):
        """Returns an array of the self.__beads_count vectors of the coordinates 
        of beads along a reaction path, according to the established path 
        (line, parabola or spline) and the parameterisation density."""

        assert not self._funcs_stale

        # For the desired distances along the string, find the values of the
        # normalised coordinate that achive those distances.
        moving = self.__generate_normd_positions(metric)

        # FIXME: for historical reasons self.__generate_normd_positions does
        # not return abscissas for terminal beads:
        self.__old_normalised_positions = self.__normalised_positions
        self.__normalised_positions = empty(len(moving) + 2)
        self.__normalised_positions[0] = 0.0
        self.__normalised_positions[1:-1] = moving
        self.__normalised_positions[-1] = 1.0

        # NOTE: Dont use list concatenation as in [reactant] + beads +
        # [product] this is going to break with numpy arrays.

        # use Path functionality:
        pairs = map(self.taylor, self.__normalised_positions)

        bead_vectors, bead_tangents = zip(*pairs)

        # convert to arrays:
        bead_vectors = asarray(bead_vectors)
        bead_tangents = asarray(bead_tangents)

        # OLD: self.state_vec = bead_vectors
        self.state_vec = masked_assign(update_mask, self.state_vec, bead_vectors)

        # FIXME: this appears suspicious, changing a single node, or a single
        # abscissa invalidates all tangents as one has to assume a different
        # path parametrization. A "masked update" of tangents may seem
        # unjustified! Is it done on purpose?
        # OLD: self.__path_tangents = bead_tangents
        self.__path_tangents = masked_assign(update_mask, self.__path_tangents, bead_tangents)

        return bead_vectors

    def update_tangents(self):
        assert not self._funcs_stale
        ts = []
        for i in self.__normalised_positions:
             ts.append(self.fprime(i))
 
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

    def set_rho(self, new_rho):
        """Set new bead density function, ensuring that it is normalised."""

        integral, err = scipy.integrate.quad(new_rho, 0.0, 1.0)

        self.__rho = lambda x: new_rho(x) / integral

    def __generate_normd_positions(self, metric):
        """Returns a list of distances along the string in terms of the normalised 
        coordinate, based on desired fractional distances along string."""

        # Get fractional positions along string, based on bead density function
        # and the desired total number of beads
        weights = linspace(0.0, 1.0, self.beads_count)

        # FIXME: I guess this method is not supposed to return terminal beads?
        # This need to work also for piecewise rho:
        arcs = scatter(self.__rho, weights[1:-1])

        # this is a Func(s(t), ds/dt) that computes the path length, here we
        # abuse it to provide the length of the tangent, ds/dt:
        arc = Arc(self, norm=metric.norm_up)

        # Other scatter variants are available, use the one with integrating
        # over the tangent lenght, see also scatter(), scatter2():
        args = scatter1(arc.fprime, arcs)

        return args

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

    if i_max == 0:
        print "WARNING: bead with highest energy is first bead"
        new_p = (ps[i_max] + ps[i_max+1]) / 2.0
        new_i = i_max+1

    elif i_max == len(Es) -1:
        print "WARNING: bead with highest energy is last bead"
        new_p = (ps[i_max] + ps[i_max-1]) / 2.0
        new_i = i_max

    else:
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
    >>> s = GrowingString(path, qc,None, beads_count=4, growing=False)
    >>> s.state_vec.round(1)
    array([[-0. , -0. ],
           [ 0.3,  0.3],
           [ 0.7,  0.7],
           [ 1. ,  1. ]])

    >>> new = s.state_vec.round(2).copy()
    >>> s.obj_func()
    -2.5882373808383816

    Was changed because of bead scattering changed, was before
    -2.5884273157684441

    >>> s.obj_func_grad().round(3)
    array([ 0.   ,  0.   ,  0.021, -0.021,  0.11 , -0.11 ,  0.   ,  0.   ])

    >>> s.step
    (0.0, array([ 0.,  0.,  0.,  0.]), array([[ 0.,  0.],
           [ 0.,  0.],
           [ 0.,  0.],
           [ 0.,  0.]]))

    #Because of line breaking do max(abs(s.obj_func_grad - true_result)) nearly 0
    # instead of looking at s.obj_func_grad directly
    >>> s.state_vec = new
    >>> a1 =  s.obj_func_grad()
    >>> ac = array([ 0.        ,  0.        ,  0.02041863, -0.02041863,  0.10998242, -0.10998242,  0.        ,  0.        ])
    >>> max(abs(a1 - ac)) < 1e-7
    True
    >>> array(s.step[1])
    array([ 0.        ,  0.00162017,  0.00081008,  0.        ])

    Changed code of respacing, old results were:
    array([ 0.        ,  0.00158625,  0.00074224,  0.        ])
    array([ 0.        ,  0.00149034,  0.000736  ,  0.        ])

    >>> s.state_vec = [[0,0],[0.3,0.3],[0.9,0.9],[1,1]]
    >>> s.obj_func_grad().round(3)
    array([ 0.   ,  0.   ,  0.018, -0.018,  0.214, -0.214,  0.   ,  0.   ])
    >>> s.lengths_disparate()
    False

    >>> s.eg_calls
    3

    """

    string = True

    def __init__(self, reagents, pes, result_storage, beads_count = 10, pmap = map,
        rho = lambda x: 1.0, growing=True, parallel=False, head_size=None, output_level = 3,
        max_sep_ratio = 0.1, reporting=None, growth_mode='normal', freeze_beads=False,
        output_path = ".", workhere = False):

        self.__final_beads_count = beads_count

        self.growing = growing
        if growing:
            initial_beads_count = 4
        else:
            initial_beads_count = self.__final_beads_count

        # create PathRepresentation object
        self._path_rep = PathRepresentation(reagents, initial_beads_count, rho)
        ReactionPathway.__init__(self, reagents, initial_beads_count, pes, parallel, result_storage,
                 reporting=reporting, output_level = output_level,
                 pmap = pmap, output_path = output_path, workhere = workhere)

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
        self._path_rep.generate_beads(mt.metric)
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

    def pathpos(self):
        return self._path_rep.pathpos()

    def get_state_vec(self):
        assert not '_state_vec' in self.__dict__
        return self._path_rep.state_vec.copy()

    def set_state_vec(self, x):
        assert not '_state_vec' in self.__dict__

        if x != None:

            self.update_path(x)

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
        new = self._path_rep.generate_beads(mt.metric)
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
        new = self._path_rep.generate_beads(mt.metric)
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

            # see below and how it is used, update_rho() does not return anything:
            return None # self.__final_rho

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


    def obj_func(self, individual=False):
        ReactionPathway.obj_func(self)

        if individual:
            return self.bead_pes_energies
        else:
            return self.bead_pes_energies.sum()

    def obj_func_grad(self,  raw=False):
        ReactionPathway.obj_func(self,  grad=True)

        result_bead_forces = zeros((self.beads_count, self.dimension))
        if raw:
            from_array = self.bead_pes_gradients
        else:
            from_array = -self.perp_bead_forces

        for i in range(self.beads_count):
            if self.bead_update_mask[i]:
                result_bead_forces[i] = from_array[i]

#       print "result_bead_forces", result_bead_forces
        g = result_bead_forces.flatten()
        return g

    def respace(self, metric, smart_abscissa=True):
        # respace the beads along the path
        if smart_abscissa:
            pythag_seps = common.pythag_seps(self.state_vec, metric)

            if len(self.state_vec) > 4:
                # This is a kind of hack to prevent the bits of
                # spline between the end beads from becoming too curved.
                pythag_seps[0] *= 0.7
                pythag_seps[-1] *= 0.7
            new_abscissa = cumm_sum(pythag_seps, start_at_zero=True)
            new_abscissa /= new_abscissa[-1]
            self._path_rep.regen_path_func(normalised_positions=new_abscissa)

        self._path_rep.generate_beads(metric, update_mask=self.bead_update_mask)

        # The following line ensure that the path used for the next
        # step is the same as the one that is generated, at a later date,
        # based on an outputted path pickle.
        self._path_rep.regen_path_func()

        self.respaces += 1

        # TODO: this must eventually be done somewhere behind the scenes.
        # I.e. Why would one ever want to update the path but not the tangents?
        self._path_rep.update_tangents()

    def update_path(self, state_vec):
        """
        After each iteration of the optimiser this function must be called.
        It rebuilds a new (spline) representation of the path
        """

        new = array(state_vec).reshape(self.beads_count, -1)
        assert new.size == self.state_vec.size

        tmp = self._path_rep.state_vec.copy()

        for i in range(self.beads_count):
            if self.bead_update_mask[i]:
                tmp[i] = new[i]
        self._path_rep.state_vec = tmp # was '= new' until 07/05/10

        # rebuild line, parabola or spline representation of path
        self._path_rep.regen_path_func()

        # TODO: this must eventually be done somewhere behind the scenes.
        # I.e. Why would one ever want to update the path but not the tangents?
        self._path_rep.update_tangents()


# Testing the examples in __doc__strings, execute
# "python gxmatrix.py", eventualy with "-v" option appended:
if __name__ == "__main__":
    import doctest
    doctest.testmod()

# You need to add "set modeline" and eventually "set modelines=5"
# to your ~/.vimrc for this to take effect.
# Dont (accidentally) delete these lines! Unless you do it intentionally ...
# Default options for vim:sw=4:expandtab:smarttab:autoindent:syntax


