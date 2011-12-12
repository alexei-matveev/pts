#!/usr/bin/env python

from sys import stderr, maxint, exit
import inspect # unused?

import logging
from copy import deepcopy
import pickle
from os import path, mkdir, chdir, getcwd

from numpy import array, asarray, ceil, abs, sqrt, dot
from numpy import empty, zeros, linspace, arange
from numpy import argmax

from path import Path, Arc, scatter1
from pts.memoize import Elemental_memoize

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
       There are three different cases for each element possible:
       m = 0: keep the old value of dst
       m = 1: use new value of src
       m = 2: only represented in src (not in dst)
              thus of course not keep it and
              make sure it is not counted in dst

    >>> m = [1, 1]
    >>> orig = arange(4).reshape(2,-1)
    >>> x = orig.copy()
    >>> y = array([5,4,3,2]).reshape(2,-1)
    >>> x = masked_assign(m, x, y)
    >>> (x == y).all()
    True
    >>> m = [0, 0]
    >>> x = orig.copy()
    >>> (x != masked_assign(m, x, y)).all()
    False
    """
    dstc = src.copy()
    assert len(mask) == len(dstc)

    j = 0
    for i, m in enumerate(mask):
        if mask[i] == 1:
            j = j + 1
        elif mask[i] == 0:
            dstc[i] = dst[j]
            j = j + 1

    return dstc

def freeze_ends(bc):
    """
    Generates a mask for masked_assign function which fixes
    the two termination beads (with 0 ) and updates the rest
    of them (1)
    """
    return [0] + [1 for i in range(bc-2)] + [0]

def new_bead_positions(new_abscissa, unchanged_abscissa, ci_num):
    """
    gives a new abcissa, calculated from the original and the new values as followes:
    the abscissa from ci_num (climbing image) will be taken. The other
    abscissa are thus distributed on the remaining space, that their relation to
    the climbing image is kept

    >>> xo = [0., 0.2, 0.25, 0.4, 0.5, 0.75, 1.]
    >>> xn = [0., 0.3, 0.25, 0.45, 0.4, 0.12, 1.]

    >>> out = new_bead_positions(xn, xo, 3)
    >>> print " %4.3f"* 7 %( tuple(out))
     0.000 0.225 0.281 0.450 0.542 0.771 1.000

    The number at 1 should be still have as big as 3:
    >>> 0.5 * out[3] - out[1] < 1e-7
    True

    >>> out = new_bead_positions(xn, xo, 4)
    >>> print " %4.3f"* 7 %( tuple(out))
     0.000 0.160 0.200 0.320 0.400 0.700 1.000

    Image 2 should be halve as big:
    >>> 0.5 * out[4] - out[2] < 1e-7
    True

    Image 5 should be exactly in the middle between image 4 and
    1.
    >>> 1. - out[5] - (out[5] - out[4]) < 1e-7
    True

    Squeeze everything into small area:
    >>> out = new_bead_positions(xn, xo, 5)
    >>> print " %4.3f"* 7 %( tuple(out))
     0.000 0.032 0.040 0.064 0.080 0.120 1.000

    If the postion of it is unchanged, so are all:
    >>> out = new_bead_positions(xn, xo, 2)
    >>> print " %4.3f"* 7 %( tuple(out))
     0.000 0.200 0.250 0.400 0.500 0.750 1.000
    """
    assert new_abscissa[-1] == unchanged_abscissa[-1]
    assert new_abscissa[0] == unchanged_abscissa[0]

    abscissa = deepcopy(new_abscissa)
    ab_ci_n = abscissa[ci_num]
    ab_ci_ref = unchanged_abscissa[ci_num]
    end = abscissa[-1]

    def new_absc(old):
        if old < ab_ci_ref:
            # Scale from the left
            return old * ab_ci_n / ab_ci_ref
        else:
            #Start from the other end:
            return end - (end - old) * (end - ab_ci_n) / (end - ab_ci_ref)

    for i in range(len(abscissa)):
        if i == ci_num:
            # This is the climbing image, it will stay fixed
            continue

        abscissa[i] = new_absc(unchanged_abscissa[i])

    return abscissa

def new_abscissa(state_vec, metric):
    """
    Generates   a  new   (normalized)  abscissa,   using   the  metric
    "metric". The abscissa will be according to the sum of pythagorean
    distances in metric for all the beads before, normed to 1.
    """

    separations = common.pythag_seps(state_vec, metric)

    new_abscissa = common.cumm_sum(separations)
    new_abscissa /= new_abscissa[-1]
    return new_abscissa

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
            workhere = 1,
            freeze_beads=False, 
            output_level = 3,
            output_path = ".",
            climb_image = False,
            start_climb = 5,
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
        self.opt_iter = 0

        self.__dimension = len(reagents[0])
        #TODO: check that all reagents are same length

        self.initialise()
        
        self.history = History()

        # mask of gradients to update at each position
        self.bead_update_mask = freeze_ends(self.beads_count)

        self._maxit = maxint
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

        self.climb_image = climb_image
        self.start_climb = start_climb
        self.ci_num = None

    def initialise(self):
        beads_count = self.beads_count

        shape = (beads_count, self.dimension)

        # forces perpendicular to pathway
        self.perp_bead_forces = zeros(shape)
        self.para_bead_forces = zeros(beads_count)

        # energies / gradients of beads, excluding any spring forces / projections
        self.bead_pes_energies = zeros(beads_count)
        self.bead_pes_gradients = zeros(shape)

        self.prev_state = None
        self.prev_perp_forces = None
        self.prev_para_forces = None
        self.prev_energies = None
        self._step = zeros(shape)

    def lengths_disparate(self, metric):
        return False

    def signal_callback(self):
        self.callbacks += 1

        # This functionality below was used with generic 3rd party optimisers,
        # i.e. it would force the optimiser to exit, so that a respace could 
        # be done.
        if self.lengths_disparate(mt.metric):
            raise pts.MustRegenerate

    def test_convergence(self, etol, ftol, xtol):
        
        self.opt_iter = self.opt_iter + 1

        t_clim = self.climb_image
        if self.growing and self.climb_image:
             # If used together with string let it work only on complete strings
             t_clim = self.grown()

        if t_clim:
            # If climbing image is used on NEB or a completely grown string
            # try to find the image, which should actually climb
            if self.opt_iter >= self.start_climb and self.ci_num == None:
                clim = self.try_set_ci_num()
                if clim:
                    print "Turned image", self.ci_num, " into climbing image"

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

    def try_set_ci_num(self):
        """
        The bead with the maximal energy will become the climbing image
        But if it would be one of the termination beads this does not make
        much sense, then tell wrong and maybe retry another time
        """
        assert self.climb_image
        i = argmax(self.bead_pes_energies)
        if i > 0 and i < len(self.bead_pes_energies) - 1:
            self.ci_num = i
            return True

        return False

    @property
    def rmsf_perp(self):
        """RMS forces, not including those of end beads."""
        return common.rms(self.perp_bead_forces[1:-1]), [common.rms(f) for f in self.perp_bead_forces]

    def pathpos(self):
        return None

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

    def __str__(self):
        e_total, e_beads = self.energies
        rmsf_perp_total, rmsf_perp_beads = self.rmsf_perp
        rmsf_para_total, rmsf_para_beads = self.rmsf_para
        maxf_beads = [abs(f).max() for f in self.bead_pes_gradients]
        path_pos  = self.pathpos()

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
             "%-24s : %10d" % ("Number of iteration", self.opt_iter),
             "%-24s : %10d" % ("Callbacks", self.callbacks),
             "%-24s : %10d" % ("Number of respaces", self.respaces),
             "%-24s : %10d" % ("Beads Count", self.beads_count)]

        if self.climb_image:
             if self.ci_num == None:
                 s += ["%-24s : %10s" % ("Climbing image", "None")]
             else:
                 s += ["%-24s : %10d" % ("Climbing image", self.ci_num)]

        s += ["%-24s : %.4f %.4f" % ("Total Length (Pythag|Spline)", total_len_pythag, total_len_spline),
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
            arc['pathps'] = self.pathpos()

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
            #FIXME: this is needed for the growing methods. Here the first iteration after growing
            # will set prev_state = state_vec
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
            angles.append(common.vector_angle(t1, t0))
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
            # Update mask: 1 update, 0 stay fixed, 2 new bead
                if self.bead_update_mask[i] == 1:
                    self._state_vec[i] = tmp[i]
        else:
           print >> stderr, "ERROR: setting state vector to NONE, aborting"
           exit()

    state_vec = property(get_state_vec, set_state_vec)

    def taylor(self, state):
       ## NOTE: this automatically skips if new_state_vec == None
       #self.state_vec = new_state_vec

#        self.reporting.write("self.prev_state updated")
#        self.prev_state = self.state_vec.copy()

        # This is now done by the optimizers themselves, as there were
        # problems with not knowing which obj_func calls to count and
        # which not
#       if self.eg_calls >= self.maxit:
#           raise pts.MaxIterations

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
        es, gs = self.allvals.taylor( state)

        # return to former directory
        chdir(wopl)

        # FIXME: does it need to be a a destructive update?
        self.bead_pes_energies[:] = es
        self.bead_pes_gradients[:] = gs
        return array(es), array(gs)

    def obj_func(self):
        es, __ = self.taylor(self.state_vec)
        self.post_obj_func(False)
        return es

    def obj_func_grad(self):
        __, g_all = self.taylor(self.state_vec)
        tangents = self.update_tangents()
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

        # project gradients in para/perp components:
        for i in range(self.beads_count):

            # previously computed gradient:
            g = g_all[i]

            # contravariant tangent coordiantes:
            T = tangents[i]

            # covariant tangent coordiantes:
            t = mt.metric.lower(T, self.state_vec[i])

            # components of force parallel and orthogonal to the tangent:
            para_force = - dot(T, g) / dot(T, t)
            perp_force = - g - para_force * t

            self.para_bead_forces[i] = para_force
            self.perp_bead_forces[i] = perp_force

        self.post_obj_func(True)
        return self.para_bead_forces, self.perp_bead_forces


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

        # FIXME: wrong place? Yes, do now only after last iteration
        #ts_estim = ts_estims(self.state_vec, self.bead_pes_energies, self.bead_pes_gradients)[-1]

        a = [es, state, perp_forces, para_forces]

        self.history.rec(a)
        
def ts_estims(beads, energies, gradients, alsomodes = False, converter = None):
    """TODO: Maybe this whole function should be made external."""

    pt = pts.tools.PathTools(beads, energies, gradients)

    estims = pt.ts_splcub()
   #f = open("tsplot.dat", "w")
   #f.write(pt.plot_str)
   #f.close()

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
    
    >>> from numpy import ones

    >>> path = [[0,0],[0.2,0.2],[0.7,0.7],[1,1]]
    >>> qc = pts.pes.GaussianPES()
    >>> neb = NEB(path, qc, 1.0, None, beads_count = 4, workhere= 0)
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

    >>> neb = NEB([[0,0],[3,3]], pts.pes.GaussianPES(), 1., None, beads_count = 10,
    ...             workhere= 0)
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

    >>> neb = NEB([[0,0],[1,1]], pts.pes.GaussianPES(), 1., None, beads_count = 3,
    ...             workhere= 0)
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
        parallel=False, workhere = 1, reporting=None, output_level = 3, output_path = ".",
        climb_image = False, start_climb = 5
        ):

        ReactionPathway.__init__(self, reagents, beads_count, pes, parallel, result_storage, pmap = pmap,
            reporting=reporting, output_level = output_level, output_path = output_path, workhere = workhere,
            climb_image = climb_image, start_climb = start_climb)

        self.base_spr_const = base_spr_const


        # Make list of spring constants for every inter-bead separation
        # For the time being, these are uniform
        self.spr_const_vec = array([self.base_spr_const for x in range(beads_count - 1)])

        self.use_upwinding_tangent = True

        # Generate or copy initial path
        if len(reagents) == beads_count:
            self._state_vec = array(reagents)
        else:
            weights = linspace(0.0, 1.0, beads_count)
            dist =  new_abscissa(reagents, mt.metric)
            pr = PathRepresentation(reagents, dist)
            #Space beads along the path, as it is for start set all of them anew
            pos = generate_normd_positions(pr, weights, mt.metric)
            self._state_vec = pr.generate_beads( pos)

    def update_tangents(self):
        """
        WARNING: uses self.bead_pes_energies to determine tangents
        """
        # terminal beads
        tangents = zeros((self.beads_count,len(self._state_vec[0])))
        tangents[0]  = self.state_vec[1] - self.state_vec[0]#zeros(self.dimension)
        tangents[-1] = self.state_vec[-1] - self.state_vec[-2]#zeros(self.dimension)

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
                    tangents[i] = tang_plus

                elif Vi_plus_1 < Vi < Vi_minus_1:
                    tangents[i] = tang_minus

                elif Vi_plus_1 > Vi_minus_1:
                    tangents[i] = tang_plus * delta_V_max + tang_minus * delta_V_min

                elif Vi_plus_1 <= Vi_minus_1:
                    tangents[i] = tang_plus * delta_V_min + tang_minus * delta_V_max
                else:
                    raise Exception("Should never happen")
            else:
                tangents[i] = ( (self.state_vec[i] - self.state_vec[i-1]) + (self.state_vec[i+1] - self.state_vec[i]) ) / 2

        for i in range(self.beads_count):
            tangents[i] /= mt.metric.norm_up(tangents[i], self.state_vec[i])

        return tangents

    def __len__(self):
        """For compatibility with ASE, pretends that there are atoms with cartesian coordinates."""
        return int(ceil((self.beads_count * self.dimension / 3.)))

    def pathpos(self):
        return None

    def get_forces(self):
        """For compatibility with ASE, pretends that there are atoms with cartesian coordinates."""
        return -common.make_like_atoms(self.obj_func_grad())

    def get_potential_energy(self):
        """For compatibility with ASE, pretends that there are atoms with cartesian coordinates."""
        return self.obj_func()

    def respace(self, metric):
        # For NEB: nothing to be done
        pass


    def obj_func_grad(self):
        __, f_perp = ReactionPathway.obj_func_grad(self)

        result_bead_forces = zeros((self.beads_count, self.dimension))
#        print "pbf", self.perp_bead_forces.reshape((-1,2))
        tangents = self.update_tangents()
        for i in range(self.beads_count):
            # Update mask: 1 update, 0 stay fixed, 2 new bead
            if not self.bead_update_mask[i] == 1:
                continue

            total = f_perp[i]

            # Climbing image is special case
            if self.climb_image and i == self.ci_num:
                t = mt.metric.lower(tangents[i], self.state_vec[i])
                total = total - self.para_bead_forces[i] * t
            # no spring force for end beads
            elif i > 0 and i < self.beads_count - 1:
                dx1 = self.bead_separations[i]
                dx0 = self.bead_separations[i-1]
                spring_force_mag = self.base_spr_const * (dx1 - dx0)

                spring_force = spring_force_mag * tangents[i]
                total = total + spring_force

#            print "spring", spring_force

            result_bead_forces[i] = total

        g = -result_bead_forces.flatten()
        return g

    def obj_func(self):

        es = ReactionPathway.obj_func(self)

        return es.sum()

class PathRepresentation(Path):
    """Supports operations on a path represented by a line, parabola, or a 
    spline, depending on whether it has 2, 3 or > 3 points."""

    def __init__(self, state_vec, positions):

        # vector of vectors defining the path
        self.__state_vec = array(state_vec)

        # generate initial paramaterisation density
        self.__normalised_positions = array(positions)

        # TODO check all beads have same dimensionality

        # use Path functionality:
        # creates first path
        Path.__init__(self, self.__state_vec, self.__normalised_positions)


    @property
    def path_tangents(self):
        ss, xs = self.nodes
        return array(map(self.fprime, ss))

    def regen_path_func(self, normalised_positions, state_vec):
        """Rebuild a new path function and the derivative of the path based on 
        the contents of state_vec."""

        assert len(normalised_positions) == len(state_vec), "%i != %i" % \
              (len(normalised_positions), state_vec)
        self.__normalised_positions = normalised_positions
        self.__state_vec = state_vec

        assert len(self.__state_vec) > 1

        # use Path functionality, on setting nodes a new parametrizaiton is generated:
        self.nodes = self.__normalised_positions, self.__state_vec

    def get_bead_separations(self, metric):
        """Returns the arc length between beads according to the current 
        parameterisation.
        """

        #
        # This is a Func(s(t), ds/dt) that computes the path length,
        # here we use it to compute arc lengths between points on the
        # path. Here "self" iherits a Path interface:
        #
        arc = Arc(self, norm=metric.norm_up)

        arcs = array(map(arc, self.__normalised_positions))

        #
        # This function is supposed to return pairwise distances, not
        # comulative arc lengths:
        #
        return arcs[1:] - arcs[:-1]

    def generate_beads(self, positions):
        """
        Updated the bead informations of the path by setting
        self.__normalised_positions and self.state_vec
        self.state_vec will contain all the coordinates
        of beads along a reaction path, according to the established path
        (line, parabola or spline) and the parameterisation density in
        self.__normalised_positions.

        Not all beads will be updated, only those which have a value > 0
        in update_mask, thus allowing for example the termination beads
        to stay exactly the same during the complete run.

        __beads_count might be different to the length of the self.state_vec
        before applying generate_beads. It is assumed that the new beads are
        indices as 2 in the update_mask.
        """

        # use Path functionality:
        pairs = map(self.taylor, positions)

        bead_vectors, bead_tangents = zip(*pairs)

        # convert to arrays:
        bead_vectors = asarray(bead_vectors)
        bead_tangents = asarray(bead_tangents)

        return bead_vectors

def generate_normd_positions(path, weights, metric):
    """Returns a list of distances along the string in terms of the normalised
    coordinate, based on desired fractional distances along string."""

    #This function is only valid for normed pathlenghts where the weights
    # of the terminal beads are 0 and 1
    assert weights[0] == 0.
    assert weights[-1] == 1.

    #
    # This is a Func(s(t), ds/dt) that computes the path length, here
    # we abuse it to provide the length of the tangent, ds/dt:
    #
    arc = Arc(path, norm=metric.norm_up)

    # Other scatter variants are available, use the one with integrating
    # over the tangent lenght, see also scatter(), scatter2():
    normalised_positions = empty(len(weights))
    normalised_positions[0] = 0.0
    normalised_positions[1:-1] = scatter1(arc.fprime, weights[1:-1])
    normalised_positions[-1] = 1.0

    return normalised_positions

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
            new_p = (ps[i-1] + ps[i] ) / 2.0

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
        dEds = dot(dEdx, dxds)

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
        new_i = 1

    elif i_max == len(Es) -1:
        print "WARNING: bead with highest energy is last bead"
        new_p = (ps[i_max] + ps[i_max-1]) / 2.0
        new_i = i_max

    else:
        p_max = ps[i_max]
        dEdx = gradients[i_max]

        dxdp = tangents[i_max]

        dEdp = dot(dEdx, dxdp)

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
    >>> s = GrowingString(path, qc,None, beads_count=4, growing=False, workhere=0)
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
    >>> s.lengths_disparate(mt.metric)
    False

    >>> s.eg_calls
    3

    """

    string = True

    def __init__(self, reagents, pes, result_storage, beads_count = 10, pmap = map,
        weights = None, growing=True, parallel=False, head_size=None, output_level = 3,
        max_sep_ratio = 0.1, reporting=None, growth_mode='normal', freeze_beads=False,
        output_path = ".", workhere = 1, climb_image = False, start_climb = 5
        ):

        self.__final_beads_count = beads_count

        self.growing = growing
        if growing:
            initial_beads_count = 4
        else:
            initial_beads_count = self.__final_beads_count

        if growing:
            if growth_mode == 'normal':
               self.weights = zeros(4)
               h_weights = linspace(0.0, 1.0, beads_count)
               self.weights[:2] = h_weights[:2]
               self.weights[-2:] = h_weights[-2:]
            elif growth_mode == 'search':
                self.weights = linspace(0.0, 1.0, 4)
            else:
                print >> stderr, "For this growth_mode ", growth_mode, "there is no way specified for getting inital weights distribution."
                print >> stderr, "Plase check if it really exists"
                exit()
        else:
            if weights == None:
                self.weights = linspace(0.0, 1.0, beads_count)
            else:
                self.weights = weights
                assert len(self.weights) == beads_count

        # Build path function based on reagents
        dist =  new_abscissa(reagents, mt.metric)
        # create PathRepresentation object
        path_rep = PathRepresentation(reagents, dist)
        self.beads_count = initial_beads_count

        ReactionPathway.__init__(self, reagents, initial_beads_count, pes, parallel, result_storage,
                 reporting=reporting, output_level = output_level, climb_image = climb_image, start_climb = 5,
                 pmap = pmap, output_path = output_path, workhere = workhere)

        # setup growth method
        self.growth_funcs = {
            'normal': (self.grow_string_normal, self.growing_string_init),
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

        self.growth_mode = growth_mode

        # Space beads along the path, as it is for start set all of them anew
        pos = generate_normd_positions(path_rep, self.weights, mt.metric)
        self._state_vec = path_rep.generate_beads( pos)

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
        """
        ATTENTION: keep this consistent with abscissas generated for tangents and respace
          Maybe let it die soon.
        """
        return new_abscissa(self.state_vec, mt.metric)

    def update_tangents(self):
        dist =  new_abscissa(self.state_vec, mt.metric)
        path_rep = PathRepresentation(self.state_vec, dist)
        return path_rep.path_tangents

    def get_forces(self):
        """For compatibility with ASE, pretends that there are atoms with cartesian coordinates."""
        return -common.make_like_atoms(self.obj_func_grad())

    def get_potential_energy(self):
        """For compatibility with ASE, pretends that there are atoms with cartesian coordinates."""
        return self.obj_func()

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

        # This additional "finish growing" test makes only sense for
        # Searching string, where the beads are set to refine the string
        # Do not use it for growing string
        if self.growth_mode == "search":
            diffs = [ps[i+1] - ps[i] for i in range(self.beads_count-1)]
            print "grown(): Difference in bead spacings:", diffs
            min_diff = min(diffs)
            too_closely_spaced = min_diff < 1.0 / (max_beads_equiv - 1.0)
        else:
            too_closely_spaced = False

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
            self.bead_positions, new_i = get_bead_positions_grad(self.bead_pes_energies, self.bead_pes_gradients, self.update_tangents(), self.bead_positions)

        if new_i == self.beads_count - 2:
            moving_beads = [new_i-2, new_i-1, new_i]
        elif new_i == 1:
            moving_beads = [new_i, new_i+1, new_i+2]
        else:
            moving_beads = [new_i-1, new_i, new_i+1]
        
        new_weight = (self.weights[new_i] + self.weights[new_i -1])/2.
        old_weights = deepcopy(self.weights)
        assert len(old_weights) == self.beads_count - 1
        self.weights = zeros(self.beads_count)
        self.weights[:new_i] = old_weights[:new_i]
        self.weights[new_i] = new_weight
        self.weights[new_i+1:] = old_weights[new_i:]

        # The following block of code ensures that all beads other than the
        # newly added one stay in exactly the same position. Otherwise, 
        # numerical inaccuraties cause them to move and additional beads
        # to have their energies calculated.

        mask = [0 for i in range(self.beads_count)]
        mask[new_i] = 2

        dist =  new_abscissa(self.state_vec, mt.metric)
        path_rep = PathRepresentation(self.state_vec, dist)
        pos = generate_normd_positions(path_rep, self.weights, mt.metric)
        # Mask tells which beads a new (2), stay fixed (0) or should be updated(1)
        places = path_rep.generate_beads( pos)
        self._state_vec = masked_assign(mask, self.state_vec, places)

        self.initialise()

        # ATTENTION: No, we do not want the state_vec from the last iteration here but
        # rather reinitalize it so that the next convergence test will be skipped
        # Therefore we want self.prev_state == self.state_vec
        self.prev_state = self.state_vec.copy()

        self.bead_update_mask = freeze_ends(self.beads_count)

        # Mask of beads to freeze, includes only end beads at present
        # 0 for fix, 1 for updated (as number of beads now fixed, no 2 any more needed)
        self.bead_update_mask = [0 for i in range(self.beads_count)]
        if self.freeze_beads:
            for i in moving_beads:
                self.bead_update_mask[i] = 1
        else:
            e = 1
            m = self.beads_count - 2 * e
            for i in range(m):
                self.bead_update_mask[e+i] = 1

        lg.info("******** String Grown to %d beads ********", self.beads_count)


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


        self.weights = zeros(self.beads_count)
        h_weights = linspace(0.0, 1.0, self.__final_beads_count)
        self.weights[:self.beads_count/2] = h_weights[:self.beads_count/2]
        self.weights[-self.beads_count/2:] = h_weights[-self.beads_count/2:]
        #self.expand_internal_arrays(self.beads_count)

        # build new bead density function based on updated number of beads
        self.growing_string_init()

        # All beads are fixed, the newly ones need their data anyhow from the new set of beads
        # As the path is build on them the other should anyhow only get rounding errors here
        mask = [0 for i in range(self.beads_count)]
        for i in new_ixs:
            mask[i] = 2

        dist =  new_abscissa(self.state_vec, mt.metric)
        path_rep = PathRepresentation(self.state_vec, dist)
        pos = generate_normd_positions(path_rep, self.weights, mt.metric)
        places = path_rep.generate_beads( pos)
        self._state_vec = masked_assign(mask, self.state_vec, places)

        # create a new bead_update_mask that permits us to set the state to what we want
        self.bead_update_mask = freeze_ends(self.beads_count)

        self.initialise()

        # ATTENTION: No, we do not want the state_vec from the last iteration here but
        # rather reinitalize it so that the next convergence test will be skipped
        # Therefore we want self.prev_state == self.state_vec
        self.prev_state = self.state_vec.copy()

        # Build mask of beads to calculate, effectively freezing some if 
        # head_size, i.e. the size of the growing heads, is specified.
        # 0 means fix, 1 means update, (bead number fix: no 2 needed)
        if self.freeze_beads:
            e = self.beads_count / 2 - self.head_size + self.beads_count % 2
        else:
            e = 1
        m = self.beads_count - 2 * e
        self.bead_update_mask = [0 for i in range(self.beads_count)]
        for i in range(m):
            self.bead_update_mask[e+i] = 1

        lg.info("Bead Freezing MASK: " + str(self.bead_update_mask))

        print "******** String Grown to %d beads ********" % self.beads_count
        return True

    def growing_string_init(self):
        """
        Sets bead_positions for searching string
        FIXME: Why not using a general setting of them?
        """

        assert self.beads_count <= self.__final_beads_count

        # Set up vector of fractional positions along the string.
        # TODO: this function should be unified with the grow_string_search()
        # at some point.
        fbc = self.__final_beads_count
        all_bead_ps = arange(fbc) * 1.0 / (fbc - 1)
        end = self.beads_count / 2.0
        self.bead_positions = array([all_bead_ps[i] for i in range(len(all_bead_ps)) \
            if i < end or i >= (fbc - end)])


    def lengths_disparate(self, metric):
        """Returns true if the ratio between the (difference of longest and 
        shortest segments) to the average segment length is above a certain 
        value (self.__max_sep_ratio).

        FIXME: Is this description out of date?
        """

        dist =  new_abscissa(self.state_vec, mt.metric)
        path_rep = PathRepresentation(self.state_vec, dist)
        seps = path_rep.get_bead_separations(metric)
        assert len(seps) == self.beads_count - 1

        seps_ = zeros(seps.shape)
        seps_[0] = seps[0]
        for i in range(len(seps))[1:]:
            seps_[i] = seps[i] + seps_[i-1]

        assert len(seps_) == len(self.weights) - 1, "%s\n%s" % (seps_, self.bead_positions)
        diffs = abs(self.weights[1:] - seps_/seps.sum())

        return diffs.max() > self.__max_sep_ratio


    def obj_func(self, individual=False):
        es = ReactionPathway.obj_func(self)

        if individual:
            return es
        else:
            return es.sum()

    def obj_func_grad(self, raw=False):
        # Perpendicular compontent should be vanish
        # difference for climbing image case
        g_para, g_minimize = ReactionPathway.obj_func_grad(self)

        result_bead_forces = zeros((self.beads_count, self.dimension))
        if raw:
            from_array = self.bead_pes_gradients
        else:
            tangents = self.update_tangents()
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

            from_array = -self.perp_bead_forces

            if self.climb_image and not self.ci_num == None:
                assert self.bead_update_mask[self.ci_num] > 0
                t = mt.metric.lower(tangents[self.ci_num], self.state_vec[self.ci_num])
                g_minimize[self.ci_num] = g_minimize[self.ci_num] + g_para[self.ci_num] * t \
                  / sqrt(dot(t, tangents[self.ci_num]))

        for i in range(self.beads_count):
            if self.bead_update_mask[i]  > 0:
                result_bead_forces[i] = g_minimize[i]

#       print "result_bead_forces", result_bead_forces
        g = result_bead_forces.flatten()
        return g

    def respace(self, metric, smart_abscissa=True):
        if not self.lengths_disparate(mt.metric):
            # Only do respace if it is necessary
            # This test seems to be done separately for most optimizer but not at all
            # for multiopt, this way it should work for all
            return

        #print "Respacing beads"
        # respace the beads along the path
        dist =  new_abscissa(self.state_vec, mt.metric)
        path_rep = PathRepresentation(self.state_vec, dist)
        # Reuse old abscissa, if CI does not tell others

        if self.climb_image and not self.ci_num == None:
            #Scale the bead positions, strings have been fully grown
            # Thus no fear to interact with their changing rhos
            bd_pos = new_bead_positions(new_abscissa, self.bead_positions, self.ci_num)
        else:
             bd_pos = generate_normd_positions(path_rep, self.weights, mt.metric)

        mask = deepcopy(self.bead_update_mask)
        if self.climb_image and not self.ci_num == None:
            mask[self.ci_num] = 0
        # Mask tells which beads a new (2), stay fixed (0) or should be updated(1)
        places = path_rep.generate_beads( bd_pos)
        self._state_vec = masked_assign(mask, self.state_vec, places)

        self.respaces += 1


    def ci_abscissa(self, x, metric):
        """
        Find a new abscissa for the climbing image, by thinking that its
        step consists of a component perpendicular to the path and one
        parallel to the path (here shortly in direction of parallel
        tangent)
        Find the component for the position of the old path, where this
        parallel step is nearest to.

        The paths defined in the PathRepresentation objects are not
        metric invariant as they do not consider our metric. Thus 
        to find out approximated path length it does not make
        sense to have it considered here
        """

        assert self.climb_image and not self.ci_num == None

        x0 = x[self.ci_num]

        s_old = self.pathpos()[self.ci_num]
        x_old = self.get_state_vec()[self.ci_num]

        s_l = self.pathpos()[self.ci_num - 1]
        s_r = self.pathpos()[self.ci_num + 1]

        d_s_all = s_r - s_l
        dx_l = x0 -  self.get_state_vec()[self.ci_num - 1]
        dx_r = -x0 +  self.get_state_vec()[self.ci_num + 1]
        fact = metric.norm_up(dx_l, x0) / (metric.norm_up(dx_l, x0) + metric.norm_up(dx_r, x0))

        s_new = s_l + d_s_all * fact

        if s_new >= 1. or s_new <= 0. \
           or s_new <= s_l or s_new >= s_r:
           print >> stderr, "WARNING: Got suspitious result for new climbing image abcissa"
           print >> stderr, "         Reusing result from last iteration", s_old, "instead of ", s_new
           s_new = s_old

        return s_new

# Testing the examples in __doc__strings, execute
# "python gxmatrix.py", eventualy with "-v" option appended:
if __name__ == "__main__":
    import doctest
    doctest.testmod()

# You need to add "set modeline" and eventually "set modelines=5"
# to your ~/.vimrc for this to take effect.
# Dont (accidentally) delete these lines! Unless you do it intentionally ...
# Default options for vim:sw=4:expandtab:smarttab:autoindent:syntax


