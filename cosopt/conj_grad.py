#!/usr/bin/env python
from numpy import zeros, dot, sqrt
import pts.metric as mt
from copy import deepcopy

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
      trial_step = 0.01, \
      reduce_to_steepest_decent = False, **kwargs):
        # Reaction_Pathway is an object that gives forces/tangents for
        # all beads:
        self.atoms = reaction_pathway

        # flags on how conjugate gradient behaves
        self.respace = respace
        self.reduce = reduce_to_steepest_decent

        # metric cannot handle more than one bead (in general) at the same time
        self.size = reaction_pathway.beads_count

        # Maximal and start step size
        self.ms = maxstep * self.size
        self.trial_step = trial_step #* self.size

        # Need to be there when used the first time
        self.nsteps = 0
        self.observers = []
        self.old_g = None

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
               #if gamma == 0.0: print "Reseted conjugate gradient"
           else:
               gamma = 0.0
               #print "Old Norm is zero"

           for i in range(self.size):
               dir[i] = dir[i] + gamma * self.old_dir[i]

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

        # line search like algorithm
        length = line_search(r, dir, g, self.atoms, self.trial_step, self.ms)

        # but do not forget step length restriction
        if length > self.ms:
            length = self.ms

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

    if c > 0:
        step_len = trial_step / 2. - g3 / c
    else:
        step_len = default_step

    return step_len

# python conj_grad.py [-v]:
if __name__ == "__main__":
     import doctest
     doctest.testmod()

# Default options for vim:sw=4:expandtab:smarttab:autoindent:syntax
