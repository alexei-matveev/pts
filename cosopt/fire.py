#!/usr/bin/env python
from numpy import zeros, dot, sqrt
import pts.metric as mt
from copy import deepcopy

class fire_opt():
    """
    Optimizer for the string/neb methods of path_searcher
    It is specialized for these methods, especially it uses
    massively their interface

    This one is about the FIRE (fast inertial relaxation engine)
     (Phys. Rev. Let. 97, 170201 (2006).)
     It treads the system with a molecular dynamic approach, changing
     the positions with speed v and time t rather than direct steps, driving
     the system to a minimum (here of all images merged to a discrete path)

     The beads are threaded as a long system, only where norm is applied they
     have to dealed with separately

     Another difference is the respacing for the beads at the end (if
     found necessary by the searcher routine)
    """
    def __init__(self, atoms, maxstep = 0.1, respace = True,\
      n_min = 5, f_inc = 1.1, f_dec = 0.5, alpha = 0.1, f_alpha = 0.99,\
      dt_max = 1.0, dt = 0.1, \
      **kwargs):
        """
        Parameters as used by the FIRE algorithm. n_min, f_inc, f_dec, alpha,
        f_alpha are all taken from the paper, where they where reocommended for
        general use. The only parameter they proposed for adapting was the maximal
        time step (dt_max). Start time step (dt) is also easily adaptable)

        FIXME: do we really need aditional a maximal step?
        """
        # Atoms to give forces/tangents
        self.atoms = atoms

        # flags on how conjugate gradient behaves
        self.respace = respace

        # metric cannot handle more than one bead (in general) at the same time
        self.size = atoms.beads_count

        # Maximal and start step size
        self.ms = maxstep * self.size

        # Need to be there when used the first time
        self.nsteps = 0
        self.observers = []

        # Parameter
        self.n_min = n_min
        self.f_inc = f_inc
        self.f_dec = f_dec
        self.alpha = alpha
        self.alpha_start = alpha
        self.f_alpha = f_alpha
        self.dt_max = dt_max

        self.v = None
        self.n_pp = 0
        self.dt = dt

    def step(self, g):
        """
        The actual step
        """

        if self.v == None:
            self.v = zeros(g.shape)

        f = - deepcopy(g)
        P = dot(f, self.v)

        # Shape because the metric can handle only one bead at a time
        f.shape = (self.size, -1)
        x = self.atoms.state_vec
        x.shape = (self.size, -1)

        self.v.shape = f.shape


        # Norm of both f and v
        norm_f = 0
        norm_v = 0
        for  vi, fi, xi in zip(self.v, f, x):
             norm_f = norm_f + mt.metric.norm_down(fi, xi)**2
             norm_v = norm_v + mt.metric.norm_up(vi, xi)**2
        norm_f = sqrt(norm_f)
        norm_v = sqrt(norm_v)

        # choose a direction steeper than the current one: (if not reset
        # by stop criteria later)
        for i, xi, fi in zip(range(self.size), x, f):
            self.v[i] = (1. - self.alpha) * self.v[i] + \
                     self.alpha * mt.metric.raises(fi, xi) * norm_v / norm_f

        f = f.flatten()
        self.v = self.v.flatten()

        if P > 0:
             # Go further in the same direction, if it was stable long enough
             # increase the step size
             self.n_pp = self.n_pp + 1
             if self.n_pp > self.n_min:
                 self.dt = min(self.dt * self.f_inc, self.dt_max)
                 self.alpha = self.alpha * self.f_alpha

        else:
            # We run into an energy minimum (in this direction)
            # Restart for a new one
            self.n_pp = 0
            self.dt = self.dt * self.f_dec
            self.alpha = self.alpha_start
            self.v = zeros(f.shape)

        # delta v = - gradient(E) * delta t
        self.v = self.v + f * self.dt
        # delta x = v * delta t
        dx = self.v * self.dt

        dx.shape = (self.size, -1)

        norm_dx = 0
        for dxi, xi in zip(dx, x):
            norm_dx = norm_dx + mt.metric.norm_up(dxi, xi)**2
        norm_dx = sqrt(norm_dx)

        if norm_dx > self.ms:
           dx = dx / norm_dx * self.ms

        # For putting them back
        x = x.flatten()
        dx = dx.flatten()

        self.atoms.state_vec = x + dx

        if self.respace:
            # This is needed by the string method
            self.atoms.respace(mt.metric )

        self.nsteps = self.nsteps + 1

    def run(self):
        #FIXME: do we really need this function here? convergence and
        #        maximal steps are done with the call observers
        #        only the step is really from here
        while True:
            # it is okay to have a endless loop here, call_observers
            # will terminate it at some time
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

# python fire.py [-v]:
if __name__ == "__main__":
     import doctest
     doctest.testmod()

# Default options for vim:sw=4:expandtab:smarttab:autoindent:syntax
