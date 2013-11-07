#!/usr/bin/env python
from numpy import zeros, dot, sqrt
import pts.metric as mt
from copy import deepcopy
from sys import stderr

VERBOSE = 0

class conj_grad_opt():
    """
    Optimizer for the string/neb methods of path_searcher
    It is specialized for these methods, especially it uses
    massively their interface

    Does a conjugate gradient step
    It is supposed that a outer function will break the run method,
    therefore it is endless,
    the convergence should be detected by one of the observers

    Conjugate gradient knows that it is working on a string as it needs
     to respace from time to time
     to process one image after the other, if metric is applied
     has to deal with some beads having old results with 0 norm

    But in general all the beads will be treated as one long vector

    The direction for the step can be reduced to the steepest decent case
    The length of the step is determined with a quadratic interpolation with one
    trial step in the direction
    """
    def __init__(self, reaction_pathway, maxstep = 0.1, respace = True,\
      trial_step = 0.01, backtrack_border = 0.9, dummy_backtracking = False,\
      reduce_to_steepest_descent = False, **kwargs):
        # Reaction_Pathway is an object that gives forces/tangents for
        # all beads:
        self.atoms = reaction_pathway

        # flags on how conjugate gradient behaves
        self.respace = respace
        self.reduce = reduce_to_steepest_descent

        # metric cannot handle more than one bead (in general) at the same time
        self.size = reaction_pathway.beads_count

        # Maximal and start step size
        self.ms = maxstep * self.size
        self.trial_step = trial_step #* self.size

        # Need to be there when used the first time
        self.nsteps = 0
        self.observers = []
        self.old_g = None

        self.old_dir = None
        # For backtracking
        self.dummy_backtracking = dummy_backtracking
        self.bt_border = backtrack_border
        self.old_length = 0.0
        self.second = False

    def step(self, g):
        """
        The actual step
        """
        # Shape because the metric can handle only one bead at a time
        g.shape = (self.size, -1)
        r = self.atoms.state_vec
        r.shape = (self.size, -1)
        dir = zeros(g.shape)


        # FIRST part: define conjugate gradient direction

        # We want process all beads together but metric can only
        # handle one bead at a time (if not Default metric)
        for i, gi in enumerate(g):
           dir[i] = -mt.metric.raises(gi, r[i])

        # From the second step on, we have conj grad
        if not self.old_g == None and not self.reduce:
           old_norm = 0.0
           for i in range(self.size):
               old_norm = old_norm + dot(self.old_g[i], mt.metric.raises(self.old_g[i], self.old_r[i]))

           if old_norm != 0.0:
               gamma = max((dot(g.flatten() - self.old_g.flatten(), -dir.flatten()) / old_norm), 0.0)
               if VERBOSE > 0:
                   if gamma == 0.0: print "CG: RESETED conjugate gradient"
           else:
               gamma = 0.0
               if VERBOSE > 0:
                   print "CG: Old Norm is zero"

           if VERBOSE > 0:
               print "CG: gamma = ", gamma

           for i in range(self.size):
               dir[i] = dir[i] + gamma * self.old_dir[i]

        need_backtracking = test_for_backtrack( g.flatten(), self.old_dir, self.old_g, self.bt_border, self.second)
        if VERBOSE and not self.old_dir == None:
            print "CG B: projections: (old/new)", dot(self.old_g.flatten(), self.old_dir.flatten())\
              / sqrt(dot(self.old_dir.flatten(), self.old_dir.flatten())), \
              dot(g.flatten(), self.old_dir.flatten()) / sqrt(dot(self.old_dir.flatten(), self.old_dir.flatten()))
        self.second = False

        if need_backtracking:
            length_bt = backtrack( g.flatten(), self.old_dir, self.old_g, self.old_length)
            self.second = True
            if VERBOSE > 0:
                print "BACKTRACKING: in iteration", self.nsteps + 1
                print "BACKTRACKING: projections: (old/new)", dot(self.old_g.flatten(), self.old_dir.flatten())\
                  / sqrt(dot(self.old_dir.flatten(), self.old_dir.flatten())), \
                  dot(g.flatten(), self.old_dir.flatten()) / sqrt(dot(self.old_dir.flatten(), self.old_dir.flatten()))
                print "BACKTRACKING: reduced length / old length", length_bt, self.old_length

            if not self.dummy_backtracking:
                #Return to previous iteration, which is still stored in the old results.
                r = self.old_r
                dir = self.old_dir
                g = self.old_g


        self.old_g = deepcopy(g)
        self.old_dir = deepcopy(dir)
        self.old_r = deepcopy(r)

        length = 0
        for di, ri in zip(dir, r):
            length = length + mt.metric.norm_up(di, ri)**2
        length = sqrt(length)

        # From now on there is no need for metric considerations
        # Therefore reshape the vectors
        dir.shape = (-1)
        r.shape = (-1)
        g.shape = (-1)

        dir = dir / length

        # SECOND PART: find step length

        if self.dummy_backtracking or not need_backtracking:
            # line search like algorithm
            length = line_search(r, dir, g, self.atoms, self.trial_step, self.ms)

            # but do not forget step length restriction
            if length > self.ms:
                length = self.ms
        else:
            # This is a backtracking step. The step length has already been
            # calculated.
            length = length_bt

        # Remember the length in case the next step is a backtracking step.
        self.old_length = length

        self.atoms.state_vec = (r + dir * length)

        if self.respace:
            # This is needed by the string method
            self.atoms.respace(mt.metric )

        self.nsteps = self.nsteps + 1

    def run(self, steps = 10000000):
        #FIXME: do we really need this function here? convergence and
        #        maximal steps are done with the call observers
        #        only the step is really from here
        while self.nsteps < steps: # convergence will be checked by call_observers
                   # Test here only if maximum allowed steps are exceeded
            f = self.atoms.obj_func_grad() # Like the gradients, but more specialized
            # string methods will give only the perpendicular direction of all beads
            # neb will already have added the spring forces

            self.call_observers()
            # Check for convergence and such things

            self.step(f)
            # Make the actual conjugate gradient step, be aware that the included line search
            # needs also to use obj_func_grad

    def attach(self, function, interval=1, *args, **kwargs):
        """Attach callback function.

        At every *interval* steps, call *function* with arguments
        *args* and keyword arguments *kwargs*."""

        self.observers.append((function, interval, args, kwargs))

    def call_observers(self):
        for function, interval, args, kwargs in self.observers:
            if self.nsteps % interval == 0:
                function(*args, **kwargs)

    def get_number_of_steps(self):
        return self.nsteps

def test_for_backtrack( g, old_dir, old_g, required, second):
    """
    Tests if a backtracking might be desiable.
    Backtrack if the projection changed its sign and
    did not loose more than 1 - required of its
    size.
    """
    if old_dir == None:
        # First iteration, nothing to backtrack to.
        return False
    else:
        new_proj = dot(g.flatten(), old_dir.flatten())
        old_proj = dot(old_g.flatten(), old_dir.flatten())
        change_sign = new_proj < 0 <  old_proj or old_proj < 0 < new_proj

        want_backtrack = abs(new_proj) > abs(old_proj) * required and change_sign
        if VERBOSE > 0:
            if second and want_backtrack:
               print "CG WARNING: backtrack rejected: Last step was already a backtrack!"

        return want_backtrack and not second

def backtrack( g, old_dir, old_g, length_old):
    """
    Find direction and step length for backtracking.
    """
    old_proj = dot(old_g.flatten(), old_dir.flatten())
    curv = dot(g, old_dir.flatten()) - old_proj
    length = - length_old * old_proj / curv
    if VERBOSE > 0:
        print "Old projection, curvature, old length"
        print old_proj, curv, length_old

    assert abs(length) < abs(length_old)

    return length

def line_search(r, dir, g, atoms, trial_step, default_step):
    """
    Maybe better called interpolation step
    Do a test step in the direction dir and find out by
    a curvature approximation in this direction how long
    the step might go with it.

    If the curvature is negative, we could in principle go
    for ever in the minimization direction. Then do not
    go to the maximum but rather use the default step, provided
    by the calling program
    """
    atoms.state_vec = r + dir * trial_step
    g2 = atoms.obj_func_grad()
    c = dot(g2 - g, dir) / trial_step
    g3 = dot(g2 + g, dir) / 2.

   #atoms.state_vec = r + dir * trial_step / 2.
   #g4 = atoms.obj_func_grad()
   #c2 = dot(g4 -g, dir)/trial_step * 2.
   #if abs(c2 - c)/c > 1e-4:
   #    print "Different curvature", c, c2
   #atoms.state_vec = r + dir * trial_step * 2.
   #g4 = atoms.obj_func_grad()
   #c2 = dot(g4 -g, dir)/trial_step / 2.
   #if abs(c2 - c)/c > 1e-4:
   #    print "Different curvature 2", c, c2
    if dot(g, dir) > 0. :
        print >> stderr, "WARNING: positive gradient projection", dot(g, dir)
        if VERBOSE > 0:
            print "WARNING: positive gradient projection", dot(g, dir)

    if VERBOSE > 0:
        print "CG: Curvature, interpolated force, force projections"
        print c, g3, dot(g, dir), dot(g2, dir)

    if c > 0:
        step_len = trial_step / 2. - g3 / c
        if VERBOSE > 0:
            print "CG: Step_length", step_len, trial_step
    else:
        if VERBOSE > 0:
            print "CG WARNING: Use DEFAULT step", default_step

        if dot(g, dir) > 0:
            print >> stderr, "WARNING: negative curvature and positive gradient projection", c, dot(g, dir)
            print >> stderr, "WARNING: This should happen seldom and is not very well explored. Take care!"
            if VERBOSE > 0:
                print "WARNING: positive curvature and gradeint projection",c , dot(g, dir)
            step_len = -default_step
        else:
            step_len = default_step

    return step_len

# python conj_grad.py [-v]:
if __name__ == "__main__":
     import doctest
     doctest.testmod()

# Default options for vim:sw=4:expandtab:smarttab:autoindent:syntax
