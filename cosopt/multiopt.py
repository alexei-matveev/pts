# -*- coding: utf-8 -*-
import sys
import numpy as np
from scipy.optimize.optimize import fminbound

from pts.func import Func
from pts.common import ObjLog
import pts.metric as mt
from numpy import sqrt
from time import localtime
from pts.bfgs import Hughs_Hessian

def disp_step(dr, f):
    dr_f = np.dot(dr.flatten(), f.flatten())
    dr_norm = (dr**2).sum()**0.5
    dr_max = np.abs(dr).max()

    return 'x.f = %.4e, norm(x) = %.4e, max(x) = %.4e' % (dr_f, dr_norm, dr_max)



class MiniBFGS(ObjLog):
    """
    """

    def __init__(self, dims, B0=None, init_step_scale=0.5, max_step_scale=0.5, max_H_resets=1e10, id=-1):
        """
        max_scale:
            Maximum scale factor to be applied to the Quasi-Newton step. This
            is quite conservative because, even if the Hessian is very
            accurate in one aprticular direction, in might be inaccurate in
            other directions and so a very accurately predicted energy step
            in one direction should not be taken as an indication that large
            steps are now possible.
        """

        ObjLog.__init__(self, 'BeadBFGS')
        self._dims = dims
        self._its = 0
        self._grad0 = None
        self._pos0 = None
        self._init_step_scale = init_step_scale
        self._max_step_scale = max_step_scale
        self._step_scale = init_step_scale

        self.hessian_params = {}
        if B0 is None:
           self.hessian_params["B0"] = 1.
        else:
           self.hessian_params["B0"] = B0
        self.hessian_params["update"] = 'SR1'
        self.hessian_params["id"] = id

        self._max_H_resets = max_H_resets
        self._H_resets = 0
        self.H_method = Hughs_Hessian
        self.id = id
        self.H = self.H_method( **self.hessian_params)

    def predictE(self, pos):
        dx = pos - self._pos0
        E = self._E0 + np.dot(dx, self._grad0) + 0.5 * np.dot(dx, self.H.app(dx))
        return E

    def calc_step_scale(self, energy, pos, err_per_step=0.1):
        """Calculates a factor by which to scale the step size.

        err_per_step:
            Allowed error per step (sort of).

        """

        E_predicted = self.predictE(pos)
        self._rho = (energy - self._E0) / (E_predicted - self._E0 + 1e-8) # guard against divide by zero
        if np.isnan(self._rho):
            print "pos", pos
            print "E_predicted", E_predicted
            print "self._E0", self._E0
            print "energy", energy
            print "self.H", self.H
            print "self._grad0", self._grad0


        # Assumes that the error in the 2nd order Taylor series energy varies
        # linearly with the step length and adjusts the step length to achieve
        # the desired fractional error |err_per_step|. FIXME: bad?
        if self._rho < 0:
            # curvature is wrong, choose small factor to scale Quasi-Newton step by
            scale = 0.1
        else:
            scale = err_per_step / (np.abs(1 - self._rho) + 0.001)
            scale = min(scale, self._max_step_scale)

        self.slog("Bead %d: Energy change: Actual / Predicted %f" % (self.id, self._rho), when='later')
        self.slog("Bead %d: Step scale:                       %f" % (self.id, scale), when='later')

        return scale

    def _update(self, energy, grad, pos):

        if self._its > 0:
            # Calculate trust radius based on approx energy using existing
            # hessian estimate.
            self._step_scale = self.calc_step_scale(energy, pos)

            dE = energy - self._E0
            self.slog("Bead %d: dE = %f" % (self.id, dE))
            dr = pos - self._pos0
            df = grad - self._grad0

            # Update Hessian
            energy_went_up      = dE > 0.
            more_resets_allowed = self._H_resets < self._max_H_resets
            hessian_inaccurate  = np.abs(self._rho - 1) > 0.1
            if energy_went_up and more_resets_allowed and hessian_inaccurate:
                self.H = self.H_method( **self.hessian_params)
                self.slog("Bead %d: Energy went up, Hessian reset to" % self.id, when='always')
                self._H_resets += 1
                # Reset also Scaling factor
                self._step_scale = self._init_step_scale
            else:
                self.H.update(dr, df)

        self._its += 1

    def step(self, energy, grad, pos, t, remove_neg_modes=True):
        """Returns a step direction by updating the Hessian (BFGS) calculating a Quas-Newton step."""

        self._update(energy, grad, pos)


        # If tangent is available, minimise energy by stepping only along
        # the force. I.e. this is kind of a line search on a quadratic
        # model of the surface.
        grad_co = mt.metric.raises(grad, pos)
        t_co = mt.metric.lower(t, pos)
        dir = -(grad_co - np.dot(grad, t)*t /np.dot(t_co, t))
        dir = np.asarray(dir)
        # NNNN: This norm only scales the vector, it will be scaled later, so
        # it is only needed to change it here if the scaling here has some importance
        norm = np.linalg.norm(dir)

        # guards against divide by zero
        if norm < 1e-8:
            step = np.zeros(self._dims)
        else:
            dir = dir / norm
            #FIXME: magical 2: why another border of minimization here?
            step_len = calc_step(dir, self.H, grad, [0.,2.])
            if step_len == 0.:
                # would mean either converged (should be detected before)
                # or wrong curvature, with 0. being the lowest endpoint. Anyhow
                # in this case the calculation is in a completely wrong area,
                # Thus make a maxstep instead
                step_len = 2.

            step = dir * step_len
            self.slog("Recomended non-scaled step dist:", step, when='always')

        self._pos0 = pos
        self._grad0 = grad
        self._E0 = energy
        return step

def calc_step(dir, H, grad, interval):
    """
    >>> print np.round(100*calc_step(np.array([-1.]), Hughs_Hessian(B0 = 2.), np.array([2.]), [0.,2.]))
    100.0

    >>> print np.round(100*calc_step((-1,1), Hughs_Hessian(B0 = 2.), (2.,2), [0.,2.]))
    -0.0

    >>> print np.round(100*calc_step((-1,-1), Hughs_Hessian(B0 = 2.), (2.,2), [0.,2.]))
    100.0


    Finds minimum of lambda *dir*H*dir*lambda + lambda*grad*dir
    dir is a direction, H the second derivative matrix and grad the first derivatives

    The interval, in which the minimum is searcher, is given relative to vector dir, thus
    it is searched in between interval[0] * dir and interval[1] * dir
    """
    dir = np.asarray(dir)
    grad = np.asarray(grad)

    # Find extremun via first derivative
    g = np.dot(grad, dir)
    b = np.dot(dir, H.app(dir))
    s_min = - g / b

    # This would be a maximum
    if b < 0.:
        #Take the border with smaller energy (the one farer away from the maximum)
        if abs(interval[0] - s_min) > abs(interval[1] - s_min):
            s_min = interval[0]
        else:
            s_min = interval[1]

    # Now restrict search on interval
    if s_min < interval[0]:
        s_min = interval[0]
    if s_min > interval[1]:
        s_min = interval[1]

    return s_min

class MultiOpt(ObjLog):
    """ Description

    """
    string = False
    def __init__(self, atoms, maxstep=0.05, alpha = 70., respace=True, **kwargs): # alpha was 70, memory was 100
        """
        THIS DESCRIPTION IS A BIT OUT OF DATE.

        Parameters:

        maxstep: float
            How far is a single atom allowed to move. This is useful for DFT
            calculations where wavefunctions can be reused if steps are small.
            Default is 0.04 Angstrom.

        alpha: float
            Initial guess for the Hessian (curvature of energy surface). A
            conservative value of 70.0 is the default, but number of needed
            steps to converge might be less if a lower value is used. However,
            a lower value also means risk of instability.
        logfile: file object or str
            If *logfile* is a string, a file with that name will be opened.
            Use '-' for stdout.
        [[atoms: Atoms object
            The Atoms object to relax.]]
        """

        ### Opt Code
        self.atoms = atoms
        self.observers = []
        self.nsteps = 0
        ###

        # this will also interprete kwargs['logfile'], if defined:
        ObjLog.__init__(self, 'MultiOpt', **kwargs)

        self.slog("Optimiser (MultiOpt): parameters: alpha =", alpha, when='always')

        self.bs = atoms.beads_count
        d = atoms.dimension
        self.respace = respace
        self.maxstep = maxstep

        # list of per-bead optimisers
        self.bead_opts = [MiniBFGS(d, B0=alpha, id=i) for i in range(self.bs)]
        self.slog("Optimiser (MultiOpt): initial step scale factors", [m._step_scale for m in self.bead_opts], when='always')


    def step(self, g):
        """Take a single step

        Use the given forces, update the history and calculate the next step --
        then take it"""

        step_str = ""
        bs = self.bs

        r = self.atoms.state_vec.reshape(bs, -1)
        e = self.atoms.obj_func(individual=True)
        ts = self.atoms.tangents.copy()
        g.shape = (bs, -1)

        # get initial direction from per-bead optimisers
        dr = np.array([self.bead_opts[i].step(e[i], g[i], r[i], t=ts[i]) for i in range(bs)])


        self.slog("DR", dr.reshape((bs,-1)))
        self.slog("G", g.reshape((bs,-1)))

        step_scales = np.array([self.bead_opts[i]._step_scale for i in range(bs)])

        dr = self.scale_step(dr, step_scales)

       #NNNN: change norm description in output?
        self.slog("DB: Lengths of steps of each bead:", ['%.5f' % np.linalg.norm(dr_bead) for dr_bead in dr], when='always')

        self.atoms.state_vec = (r + dr)

        if self.respace:
            self.slog("Respacing Respacing Respacing Respacing Respacing ")
            self.atoms.respace(mt.metric )

            dr_respace = self.atoms.state_vec.reshape(bs,-1) - (r + dr)

        dr_total = self.atoms.state_vec - r

        #dr_total.shape = (bs, -1)

    def scale_step(self, dr, step_scales):
        """Determine step to take according to the given trust radius

        Normalize all steps as the largest step. This way
        we still move along the eigendirection.
        """

        for i in range(self.atoms.beads_count):
            dr[i] *= step_scales[i]
            longest = np.abs(dr[i]).max()
            if longest > self.maxstep:
                dr[i] *= (self.maxstep / longest)

        return dr

# ASE optimizer function, inherited before
    def attach(self, function, interval=1, *args, **kwargs):
        """Attach callback function.

        At every *interval* steps, call *function* with arguments
        *args* and keyword arguments *kwargs*."""

        self.observers.append((function, interval, args, kwargs))

    def run(self, steps=100000000):
        """Run structure optimization algorithm.

        This method will return  when the number of steps exceeds
        *steps*."""

        step = 0
        while step < steps:
            f = self.atoms.obj_func_grad(raw=True)
            self.slog(f) # ObjLog method
            self.call_observers()
            self.step(f)
            self.nsteps += 1
            step += 1

    def call_observers(self):
        for function, interval, args, kwargs in self.observers:
            if self.nsteps % interval == 0:
                function(*args, **kwargs)

# python path_representation.py [-v]:
if __name__ == "__main__":
    import doctest
    doctest.testmod()

# Default options for vim:sw=4:expandtab:smarttab:autoindent:syntax

