# -*- coding: utf-8 -*-
import sys
import numpy as np
from scipy.optimize.optimize import fminbound
from ase.optimize.optimize import Optimizer

from pts.func import Func
from pts.common import ObjLog

def disp_step(dr, f):
    dr_f = np.dot(dr.flatten(), f.flatten())
    dr_norm = (dr**2).sum()**0.5
    dr_max = np.abs(dr).max()

    return 'x.f = %.4e, norm(x) = %.4e, max(x) = %.4e' % (dr_f, dr_norm, dr_max)

class MiniBFGS(Func, ObjLog):
    """
    1-D Parabola, initial hessian perfect
        >>> bfgs = MiniBFGS(1, np.eye(1) * 2)
        >>> e, g = lambda x: x**2, lambda x: 2*x
        >>> x0 = np.array([1.0])
        >>> x0 + bfgs(e(x0), g(x0), x0)
        array([ 0.])

    2-D parabola, initial hessian perfect
        >>> bfgs = MiniBFGS(2, np.eye(2) * 2)
        >>> g = lambda x: np.array((2*x[0], 2*x[1]))
        >>> e = lambda x: (x**2).sum()
        >>> x0 = np.array([1.0, -1.0])
        >>> x0 + bfgs(e(x0), g(x0), x0)
        array([ 0.,  0.])

    1-D parabola, initial hessian half what ti should be
        >>> bfgs = MiniBFGS(1, np.eye(1))
        >>> f = lambda x: 2*x
        >>> e = lambda x: x**2
        >>> x = np.array([1.0])
        >>> x = x + bfgs(e(x), f(x), x)
        >>> x = x + bfgs(e(x), f(x), x)
        >>> bfgs.H
        array([[ 2.]])
        >>> x
        array([ 0.])

    2-D parabola, initial hessian half what it should be
        >>> bfgs = MiniBFGS(2, np.eye(2))
        >>> f = lambda x: np.array((2*x[0], 2*x[1]))
        >>> e = lambda x: (x**2).sum()
        >>> x = np.array([1.0, -1.0])
        >>> x = x + bfgs(e(x), f(x), x)
        >>> x = x + bfgs(e(x), f(x), x)
        >>> bfgs.H
        array([[ 1.5, -0.5],
               [-0.5,  1.5]])

   After two steps reaches minimum.
        >>> x
        np.array([ 0.,  0.])
        >>> x + bfgs(e(x), f(x), x)
        np.array([ 0.,  0.])
        >>> bfgs.H
        array([[ 1.5, -0.5],
               [-0.5,  1.5]])

    Dummy trust radius at present:
        >>> round(10* bfgs.trust_rad)
        1.0


    """

    def __init__(self, dims, H0=None, init_step_scale=0.5, max_step_scale=0.5, max_H_resets=1e10, id=-1):
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

        if H0 is None:
            self.H = np.eye(dims)
        else:
            self.H = H0.copy()

        self.initH = self.H.copy()
        self._max_H_resets = max_H_resets
        self._H_resets = 0
        self.id = id
        self.method = 'SR1'

    def predictE(self, pos):
        dx = pos - self._pos0
        E = self._E0 + np.dot(dx, self._grad0) + 0.5 * np.dot(dx, np.dot(self.H, dx))
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
        step_len = np.linalg.norm(pos - self._pos0)
        if self._rho < 0:
            # curvature is wrong, choose small factor to scale Quasi-Newton step by
            scale = 0.1
        else:
            scale = err_per_step / (np.abs(1 - self._rho) + 0.001)
            scale = min(scale, self._max_step_scale)
        
        self.slog("Bead %d: Energy change: Actual / Predicted %f" % (self.id, self._rho), when='always')
        self.slog("Bead %d: Step scale:                       %f" % (self.id, scale), when='always')

        return scale

    def _update(self, energy, grad, pos):

        if self._its > 0:
            # Calculate trust radius based on approx energy using existing
            # hessian estimate.
            self._step_scale = self.calc_step_scale(energy, pos)

            self.slog("Bead %d: dE = %f" % (self.id, energy - self._E0))

            # Update Hessian
            energy_went_up      = energy > self._E0
            more_resets_allowed = self._H_resets < self._max_H_resets
            hessian_inaccurate  = np.abs(self._rho - 1) > 0.1
            if energy_went_up and more_resets_allowed and hessian_inaccurate:# or (self.id==6 and self._its=10):
                # reset Hessian to conservative value if energy goes up
#                self.H *= np.eye(self._dims)
                self.H = self.initH.copy()#np.min(self.initH[0,0], np.abs(self.H[0,0]).sum() * 2) * np.eye(self._dims)
                self.slog("Bead %d: Energy went up, Hessian reset to" % self.id, when='always')
                self.slog(self.H)
                self._H_resets += 1
                self._step_scale = self._init_step_scale
            else:
                dr = pos - self._pos0
                # do nothing if the step is tiny (and probably hasn't changed at all)
                if np.abs(dr).max() >= 1e-7:
                    # BFGS update
                    df = grad - self._grad0
                    dg = np.dot(self.H, dr)

                    print "dr", dr
                    if self.method == 'SR1':
                        c = df - np.dot(self.H, dr)
                        print "c",c

                        # guard against division by very small denominator
                        if np.linalg.norm(c) * np.linalg.norm(c) > 1e-8:
                            self.H += np.outer(c, c) / np.dot(c, dr)
                        else:
                            self.slog("Bead %d: skipping SR1 update, denominator too small" % self.id, when='always')
                    elif self.method == 'BFGS':
                        a = np.dot(dr, df)
                        b = np.dot(dr, dg)
                        print "a", a
                        print "b", b
                        self.H += np.outer(df, df) / a - np.outer(dg, dg) / b
                    else:
                        assert False, 'Should never happen'
                    self.slog("Bead %d: Hessian (BFGS) updated to" % self.id)
                    self.slog(self.H)

            self.slog('DB: Hessian min/max abs vals: %.2f %.2f' % (abs(self.H).min(), abs(self.H).max()), when='always')

        self._its += 1
       
    def f(self, energy, grad, pos, t=None, remove_neg_modes=True):
        """Returns a step direction by updating the Hessian (BFGS) calculating a Quas-Newton step."""

        self._update(energy, grad, pos)


        if t is None:
            if remove_neg_modes:
                evals, evecs = np.linalg.eigh(self.H)
                step = - np.dot(evecs, np.dot(grad, evecs) / np.fabs(evals))
            else:
                Hinv = np.linalg.inv(self.H)
                step = -np.dot(Hinv, grad)
        else:
            # If tangent is available, minimise energy by stepping only along
            # the force. I.e. this is kind of a line search on a quadratic 
            # model of the surface.
            dir = -(grad - np.dot(grad, t)*t)
            norm = np.linalg.norm(dir)

            # guards against divide by zero
            if norm < 1e-8:
                step = np.zeros(self._dims)
            else:
                dir = dir / norm
                step = dir * calc_step(dir, self.H, grad, energy)
                self.slog("Recomended non-scaled step dist:", step, when='always')

        self._pos0 = pos
        self._grad0 = grad
        self._E0 = energy
        return step

def calc_step(dir, H, grad, energy):
    """
    >>> np.round(100*calc_step((-1.,), (2.,), (2.,), 1.))
    100.0

    >>> np.round(100*calc_step((-1,1), ((2,0),(0,2)), (2.,2), 1.))
    0.0

    >>> np.round(100*calc_step((-1,-1), ((2,0),(0,2)), (2.,2), 1.))
    141.0



    """
    dir = np.asarray(dir)
    H = np.asarray(H)
    grad = np.asarray(grad)
    dir = dir / np.linalg.norm(dir)
    def quadradic_energy(s):
        return s*np.dot(grad, dir) + 0.5*s*s*np.dot(dir, np.dot(H, dir))

    # This is silly: I need to change it to solving teh quadratic
    dist = np.atleast_1d(fminbound(quadradic_energy, 0., 2.))[0]
    assert dist > 0.

    return dist

class MultiOpt(Optimizer, ObjLog):
    """ Description

    """
    string = False
    def __init__(self, atoms, restart=None, logfile='-', trajectory=None,
                 maxstep=0.05, alpha = 70., respace=True, backtracking=None): # alpha was 70, memory was 100
        """
        THIS DESCRIPTION IS A BIT OUT OF DATE.

        Parameters:

        restart: string
            Pickle file used to store vectors for updating the inverse of Hessian
            matrix. If set, file with such a name will be searched and information
            stored will be used, if the file exists.

        maxstep: float
            How far is a single atom allowed to move. This is useful for DFT
            calculations where wavefunctions can be reused if steps are small.
            Default is 0.04 Angstrom.

        alpha: float
            Initial guess for the Hessian (curvature of energy surface). A
            conservative value of 70.0 is the default, but number of needed
            steps to converge might be less if a lower value is used. However,
            a lower value also means risk of instability.
            
        """
        Optimizer.__init__(self, atoms, restart, logfile, trajectory)
        ObjLog.__init__(self, 'MultiOpt')

        self.slog("Optimiser (MultiOpt): parameters: alpha =", alpha, when='always')

        class Dummy():
            e = 1e100 # internet wasn't working, couldn't get to docs
        self.best = Dummy()

        self.bs = atoms.beads_count
        d = atoms.dimension
        self.respace = respace
        self.its = 0
        self.maxstep = maxstep

        # list of per-bead optimisers
        self.bead_opts = [MiniBFGS(d, H0=np.eye(d)*alpha, id=i) for i in range(self.bs)]
        self.slog("Optimiser (MultiOpt): initial step scale factors", [m._step_scale for m in self.bead_opts], when='always')

        if not backtracking is None:
            print "WARNING: backtracking does not exist for multiopt optimiser. Keyword ignored."


    def step(self, dummy):
        """Take a single step
        
        Use the given forces, update the history and calculate the next step --
        then take it"""
        
        step_str = ""
        bs = self.bs

        r = self.atoms.state_vec.reshape(bs, -1)
        g = self.atoms.obj_func_grad(raw=True)
        e = self.atoms.obj_func(individual=True)
        ts = self.atoms.tangents.copy()
        g.shape = (bs, -1)

        # get initial direction from per-bead optimisers
        dr_raw = np.array([self.bead_opts[i](e[i], g[i], r[i], t=ts[i]) for i in range(bs)])
        self.slog("DR_raw", dr_raw.reshape((bs,-1)))


        # project out parallel part of step
        dr = self.atoms.exp_project(dr_raw) # - np.dot(dr, t) * t
        self.slog("DR", dr.reshape((bs,-1)))
        self.slog("G", g.reshape((bs,-1)))
        self.slog("G_proj", self.atoms.exp_project(g))

        step_scales = np.array([self.bead_opts[i]._step_scale for i in range(bs)])

        dr = self.scale_step(dr, step_scales)

        self.slog("DB: Lengths of steps of each bead:", ['%.5f' % np.linalg.norm(dr_bead) for dr_bead in dr], when='always')
 
        self.atoms.state_vec = (r + dr) 

        if self.respace:
            self.slog("Respacing Respacing Respacing Respacing Respacing ")
            self.atoms.respace()

            dr_respace = self.atoms.state_vec.reshape(bs,-1) - (r + dr)

        dr_total = self.atoms.state_vec - r

        #dr_total.shape = (bs, -1)

        self.its += 1

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

# python path_representation.py [-v]:                              
if __name__ == "__main__":                                         
    import doctest                                                 
    doctest.testmod()                                              
                                                                   
# Default options for vim:sw=4:expandtab:smarttab:autoindent:syntax

