#!/usr/bin/python
from numpy import dot, sqrt, pi
from copy import deepcopy
from pts.bfgs import LBFGS, BFGS, SR1
from pts.metric import Default
from pts.dimer_rotate import rotate_dimer, rotate_dimer_mem
from numpy import arccos
from sys import stdout
from pts.trajectories import empty_traj
"""
dimer method:

References:
J. K\{a"}stner, P. Sherwood; J. Chem. Phys. 128 (2008), 014106
A. Heyden, A. T. Bell, F. J. Keil; J. Chem. Phys. 123 (2005), 224101
G. Henkelman, H. J\{o'}nsson; J. Chem. Phys. 111 (1999), 7010
"""

class translate_cg():
    def __init__(self, metric, trial_step):
        """
        use conjugate gradient to determine in which direction to do the next step
        conjugate gradient update as Polak Ribiere with reset for negative values

        >>> from numpy import array
        >>> from pts.pes.mueller_brown import MB
        >>> from pts.metric import Default

        >>> met = Default(None)
        >>> trans = translate_cg(met, 0.5)

        >>> start = array([-0.5, 0.5])
        >>> mode = array([-1., 0.])

        >>> step, info = trans(MB, start, MB.fprime(start), mode, -1., {})
        >>> print step
        [-0.78443852 -0.07155202]

        >>> step, info = trans(MB, start + step, MB.fprime(start + step), mode, -1., {})
        >>> print step
        [ 2.67119938 -1.01683994]

        >>> trans = translate_cg(met, 0.5)
        >>> start = array([-0.25, 0.75])
        >>> step, info = trans(MB, start, MB.fprime(start), mode, -1., {})
        >>> print step
        [-0.00316205 -0.31033489]
        """
        self.metric = metric
        self.old_force = None
        self.old_step = None
        self.trial_step = trial_step
        self.old_geo = None

    def __call__(self, pes, start_geo, geo_grad, mode_vector, curv, unused):
        """
        the actual step
        """
        # translation "force"
        shape = geo_grad.shape
        mode_vec_down = self.metric.lower(mode_vector, start_geo).flatten()
        #print "Forces raw PTS", dot(geo_grad.flatten(), geo_grad.flatten())

        if curv > 0.0:
            # No negative curvature found, go in direction of lowest mode then
            force = negpara_force( -geo_grad.flatten(), mode_vector.flatten(), mode_vec_down)
        else:
            force = trans_force( -geo_grad.flatten(), mode_vector.flatten(), mode_vec_down)

        # find direction
        step = self.metric.raises(force, start_geo)

        if not self.old_force == None: # if not first iteration
           # attention: here step is simply force with upper indice
           old_norm = dot(self.old_force, self.metric.raises(self.old_force, self.old_geo))
           if old_norm != 0.0:
               gamma = max((dot(force - self.old_force, step) / old_norm), 0.0)
               #if gamma == 0.0: print "Reseted conjugate gradient"
           else:
               gamma = 0.0
               #print "Old Norm is zero"
           step = step + gamma * self.old_step
           #self.trial_step = self.trial_step * old_norm /  dot(force, step)
        #print "Direction", step

        # store for the next iteration:
        # FIXME: make copy else the next iteration will have for at least one of them wrong
        # value and convert this to a steepest decent calculation
        self.old_force = deepcopy(force)
        self.old_step = deepcopy(step)
        self.old_geo = deepcopy(start_geo)

        step /= self.metric.norm_up(step, start_geo)

        if curv > 0.0:
            # We are (supposely) in complete wrong region
            # Take maximal step out of here
            trial_step = self.trial_step
            grad_calc = 0
        else:
            # find how far to go
            # line search, first trial
            trial_step, grad_calc = line_search(start_geo, step, self.trial_step, pes, self.metric, mode_vector, force)

        step = trial_step * step


        info = {"trans_abs_force" : self.metric.norm_down(force, start_geo),
                "trans_gradient_calculations": grad_calc}

        step.shape = shape

        return step, info

def line_search( start_geo, direction, trial_step, pes, metric, mode_vector, force ):
        """
        Find the minimum in direction from strat_geo on, uses second point
        makes quadratic approximation with the "forces" of these two points
        """
        assert ( abs(metric.norm_up(direction, start_geo) -1.) < 1e-7)

        grad_calc = 0
        force_l = deepcopy(force)
        i = 0
        t_s = deepcopy(trial_step)

        geo = start_geo + direction * t_s
        mode_vec_down = metric.lower(mode_vector, start_geo).flatten()
        grad_calc += 1
        force_r = trans_force( -pes.fprime(geo), mode_vector, mode_vec_down)
        # interpolate force in middle between two steps
        f_mid = dot(force_r + force_l, direction) /2.
        # estimate curvature in middle between two steps
        cr = dot(force_r - force_l, direction) / t_s
        # search 0 = f_mid + t_s1 * cr (t_s1 starting from middle between two points)
        t_s = (- f_mid / cr + t_s/ 2.0)

        return t_s, grad_calc

def trans_force(force_raw_trial, mode_vector, mode_vector_down):
    """
    Force used for translation
    F = F_perp - F_para
    F_perp = F_old - F_para
    F_para = (F_old, mode) * mode
    """
    return force_raw_trial - 2. * dot(force_raw_trial, mode_vector) * mode_vector_down

def negpara_force(force_raw_trial, mode_vector, mode_vector_down):
    """
    Force used for translation
    F = F_perp - F_para
    F_perp = F_old - F_para
    F_para = (F_old, mode) * mode
    """
    return  - dot(force_raw_trial, mode_vector) * mode_vector_down

class translate_lbfgs():
    def __init__(self, metric, unused):
        """
        The parameter unused is there for consistence, as the other
        translation steps needs a parameter more

        Calculates a translation step of the dimer (not scaled)
        Uses an approximated hessian hess

        Combination of a Dimer\Lanczos and a L-BFGS quasi-Newton algorithm:
           * A (positive definite) L-BFGS update maintans an approximate
           hessian. A quasi-Newton step with this is done the usual way.
           * Additionally the mode and curvature of the dimer is used to
           generate an Hessian update of SR1 kind (should keep negative
           curvature). A step modification using this "Matrix" is added
           to the normal step.

        If the Dimer\Lanczos curvature is positive instead the usual
        step for the Dimer/Lanczos method is done to get as fast away
        from the minima as possible (maximal step along negative of
        Dimer/Lanczos mode vector).

        Consider metric

        >>> from numpy import array
        >>> from pts.pes.mueller_brown import MB
        >>> from pts.metric import Default

        >>> met = Default(None)
        >>> trans = translate_lbfgs(met, None)

        >>> start = array([-0.5, 0.5])
        >>> mode = array([-1., 0.])

        >>> step, info = trans(MB, start, MB.fprime(start), mode, -1000., {})
        >>> print step
        [-0.07293125 -0.0950339 ]
        >>> step, info = trans(MB, start + step, MB.fprime(start + step), mode, -1000., {})
        >>> print step
        [-0.07935383  0.0896875 ]

        >>> trans = translate_lbfgs(met, None)
        >>> start = array([-0.25, 0.75])
        >>> step, info = trans(MB, start, MB.fprime(start), mode, -100., {})
        >>> print step
        [-0.02558608 -3.58730495]
        """
        self.hess = BFGS()
        self.old_grad = None
        self.old_geo = None
        self.metric = metric

    def __call__(self, pes, start_geo, geo_grad, mode_vector, curv, info):
        """
        the actual step
        """
        force_raw = - geo_grad
        shape = force_raw.shape
        force_raw = force_raw.flatten()
        mode_vec_down = self.metric.lower(mode_vector, start_geo).flatten()

        # These are mainly there for output and printing, force_para is
        # also needed if the curvature is positive.
        force_para = dot(force_raw, mode_vector.flatten()) * mode_vec_down
        force_perp = force_raw - force_para

        if "rot_updates" in info:
            # Lanczos method keeps some informations for our update.
            # If it created the mode, we can now use them for the
            # step hessian.
            for dr, dg in info["rot_updates"]:
                # Be aware that the direction goes from middle point
                # to the dimer end point, while the forces are from
                # the dimer end point  minus the middle point.
                self.hess.update(dr, dg)

        if not self.old_grad == None:
            # This is the update of the Hessian with the gradient
            # change from the last step. It is always the last
            # update as it is the most relevant one.
            dr = start_geo - self.old_geo
            dg = geo_grad - self.old_grad
            self.hess.update(dr, dg)

        # BFGS hessian is positive definite. Thus it has the wrong
        # eigenvalue in direction of the mode (when it has as desired
        # a negative direction).
        step_hess = self.hess.inv(force_raw)

        # Like SR1 update on the hessian with the mode vector:
        # result is (H* - H) * tau.
        def H_times_tau(tau):
             y_k = curv * mode_vector
             # y_k = curv * mode_vector is a gradient.
             u_k = mode_vector.flatten() - self.hess.inv(y_k)
             return u_k / (dot(u_k, y_k)) * dot(u_k, tau)

        # Like a SR1 update step, using mode and curvature of
        # dimer/lanczos method.
        step_add = H_times_tau(force_raw)

        # step = H_BFGS * f + update(SR1) * f
        step = step_hess + step_add

        if curv > 0:
            # Undesired Hessian, there is no negative eigenmode.
            # Therefore do a maximal step away. Relax as long as relaxation
            # might be going, additionally do rest of the step along mode
            # (should become negative eigenvalue direction). Go there
            # opposite to the forces (climb)
            step_relax = step - dot( step, mode_vec_down) * mode_vector
            length_relax = self.metric.norm_up(step_relax, start_geo)
            if length_relax > info["max_step"]:
                # will be scaled down later
                step = step_relax
            else:
                # The step climbing along mode_vector should have the opposite
                # direction than the force.
                sign = 1.
                if dot(mode_vector, force_raw) > 0:
                    sign = -1.

                step = step_relax + sign * mode_vector * sqrt(info["max_step"]**2 - length_relax**2)

        # Test if the perpendicular part of the force is perpendicular to the mode
        # (Angle == pi/2)
        #test_step = H_times_tau(force_raw - dot(force_raw, mode_vector) * mode_vec_down )
        #print "Angle", dot(test_step, mode_vector)

        self.old_grad = geo_grad
        self.old_geo = start_geo

        #print "Forces mod BFGS",self.metric.norm_down(force_raw, start_geo), self.metric.norm_down(force_para, start_geo), self.metric.norm_down(force_perp, start_geo)
        #print "Steps mod BFGS", self.metric.norm_up(step, start_geo) ,self.metric.norm_up(step_hess, start_geo), self.metric.norm_up(step_add, start_geo)

        info_out = {"trans_perp_force" : self.metric.norm_down(force_perp, start_geo),
                "trans_para_force": self.metric.norm_down(force_para, start_geo),
                "trans_step_hess" : self.metric.norm_up(step_hess, start_geo),
                "trans_step_mod" : self.metric.norm_up(step_add, start_geo),
                "trans_gradient_calculations": 0}

        step.shape = shape

        return step, info_out

class translate_sd():
    def __init__(self, metric, trial_step):
        """
        use steepest decent to determine in which direction to do the next step

        >>> from numpy import array
        >>> from pts.pes.mueller_brown import MB

        >>> tr_step = 0.3
        >>> met = Default(None)
        >>> trans = translate_sd(met, tr_step)

        >>> start = array([-0.5, 0.5])
        >>> mode = array([-1., 0.])

        >>> step, info = trans(MB, start, MB.fprime(start), mode, -1., {})
        >>> print step
        [-1.64564686 -0.15010654]

        >>> trans = translate_sd(met, tr_step)
        >>> start = array([-0.25, 0.75])
        >>> step, info = trans(MB, start, MB.fprime(start), mode, -1., {})
        >>> print step
        [-0.00256136 -0.25138143]
        """
        self.metric = metric
        self.trial_step = trial_step

    def __call__(self, pes, start_geo, geo_grad, mode_vector, curv, unused):
        """
        the actual step
        """
        force_raw = - geo_grad
        shape = force_raw.shape
        force_raw = force_raw.flatten()
        mode_vec_down = self.metric.lower(mode_vector, start_geo).flatten()
        if curv > 0.0:
            # No negative curvature found, go in direction of lowest mode then
            force = negpara_force( force_raw, mode_vector.flatten(), mode_vec_down)
        else:
            force = trans_force( force_raw, mode_vector.flatten(), mode_vec_down)

        # actual steepest decent step
        step = self.metric.raises(force, start_geo)
        step /= sqrt(dot(step, force))

        if curv > 0.0:
            # We are (supposely) in complete wrong region
            # Take maximal step out of here
            trial_step = self.trial_step
            grad_calc = 0
        else:
            # find how far to go
            # line search, first trial
            trial_step, grad_calc = line_search(start_geo, step, self.trial_step, pes, self.metric, mode_vector, force)
        step = trial_step * step

        step.shape = shape

        info = {"trans_abs_force" : self.metric.norm_down(force, start_geo),
                "trans_gradient_calculations": grad_calc}
        return step, info


trans_dict = {
               "conj_grad" : translate_cg,
               "lbfgs"     : translate_lbfgs,
               "steep_dec" : translate_sd
             }

rot_dict = {
           "dimer" : rotate_dimer,
           "lanczos" : rotate_dimer_mem
           }


def dimer(pes, start_geo, start_mode, metric, max_translation = 100000000, max_gradients = None, \
       trans_converged = 0.00016, trans_method = "conj_grad", start_step_length = 0.001, \
       rot_method = "dimer", trajectory = empty_traj, logfile = None, **params):
    """ The complete dimer algorithm
    Parameters for rotation and translation are handed over together. Each of the two
    grabs what it needs.
    Required input is:
    pes :  potential surface to calculate on, needs f and fprime function
    start_geo : one geometry on pes, as start for search
    start_mode : first mdoe_vector suggestion
    metric     : might affect everything, as defines distances and angles, from pts.metric module
    """
    # for translation

    trans = trans_dict[trans_method](metric, start_step_length)
    rot = rot_dict[rot_method]

    # do not change them
    geo = deepcopy(start_geo) # of dimer middle point
    mode = deepcopy(start_mode) # direction of dimer
    old_force = None
    step_old = 0
    conv = False
    grad_calc = 0
    error_old = 0
    mode_old = deepcopy(mode)

    if logfile == None or logfile == "-":
        selflogfile = stdout
    else:
        selflogfile = open(logfile, "a")

    selflogfile.write("Intermediate steps for translation and rotation informations\n")
    selflogfile.write("Values are in eV, Angstrom, degrees or combinations of them\n")
    selflogfile.write("Trans. Infos:       Energy            ABS. Force          Max. Force        Step:")
    selflogfile.write("          Perp\Para           Angle to mode\n")
    selflogfile.write("Rot. Infos:  Conv.   Steps.       curvature      rot. force        Angle to last\n")
    selflogfile.flush()

    i = 0
    # main loop:
    while i < max_translation:
         #ATTENTION: 2 break points, first for convergence, second at end
         # two ways of maximum iteration: as max_translation or
         #        as count of gradient calls, second case needs
         #        additional break point
         #        it is only checked for only maximum number of gradient calls
         #        if max_translation > maximum number of gradient calls
         energy, grad = pes.taylor(geo)
         grad_calc += 1

         # Test for convergence, converged if saddle point is reached
         abs_force = metric.norm_down(grad, geo)
         error = max(abs(grad.flatten()))
         if error < trans_converged:
              # Breakpoint 1: calculation has converged
              conv = True
              selflogfile.write("Calculation is converged with max(abs(force)) %8.5f < %8.5f \n" % \
               (error, trans_converged))
              break

         # calculate one step of the dimer, also update dimer direction
         # res is dictionary with additional results
         step, mode, res = _dimer_step(pes, geo, grad, mode, trans, rot, metric, **params)
         grad_calc += res["rot_gradient_calculations"] + res["trans_gradient_calculations"]
         #print "iteration", i, error, metric.norm_down(step, geo)

         #collect things for output
         step_len = metric.norm_up(step, geo)
         step_para = dot(step, metric.lower(mode, geo))
         step_perp = metric.norm_up(step - step_para * mode, geo)
         angle = abs(dot(step,mode) / step_len)
         if angle > 1.0: #will be a rounding error
            angle = 1.0
         angle2 = abs(dot(mode, mode_old))
         if angle2 > 1.0: #will be a rounding error
            angle2 = 1.0

         # Give some report during dimer optimization
         selflogfile.write("Step %5i with sum of Grad. calcs. %7i\n" % (i, grad_calc))
         selflogfile.write("Trans. Infos:   %12.5f       %12.5f       %12.5f       %9.5f    %9.5f \%9.5f      %9.5f\n" % \
               (energy, abs_force, error, step_len,step_perp, step_para, arccos(angle) * 180 /pi))

         if res["rot_convergence"]:
             selflogfile.write("Rot. Infos: True  %5i     %12.5f       %12.5f       %12.5f\n" % \
               ( res["rot_iteration"] , res["curvature"], res["rot_abs_forces"], arccos(angle2) * 180 /pi))
         else:
             selflogfile.write("Rot. Infos: False %5i     %12.5f       %12.5f       %12.5f\n" % \
               ( res["rot_iteration"] , res["curvature"], res["rot_abs_forces"], arccos(angle2) * 180 /pi))
         selflogfile.flush()
        #if i > 0 and error_old - error < 0:
        #    print "Error is growing"
        #if i > 0:
        #    if dot(step, metric.lower(step_old, geo)) < 0:
        #       print "Step changed direction"
         step_old = step
         mode_old = mode
         geo = geo + step
         #print "Step",i , pes(geo-step), abs_force, max(grad), res["trans_last_step_length"], res["curvature"], res["rot_gradient_calculations"]
         i += 1
         error_old = error
         traj_content = [( grad, "grads", "Gradients"),(mode, "modes", "Mode")]
         trajectory(geo, i, traj_content )

         if not max_gradients == None and max_gradients <= grad_calc:
              # Breakpoint 2: for counting gradient calculations instead of translation steps
              break

    if conv:
        print "Calculation is converged"
    else:
        selflogfile.write("Calculation is not converged\n")
        print "No convergence reached"

    # all gradient calculations makes sense
    # gradient calculations only of last rotation not so much
    try:
        del res["rot_gradient_calculations"]
        del res["trans_gradient_calculations"]
    except UnboundLocalError:
        # Convergence criteria was fulfilled from the start.
        res = {}

    res["gradient_calculations"] = grad_calc

    # add some more results to give back
    res["trans_convergence"] = conv
    res["abs_force"] = abs_force
    res["trans_steps"] = i
    res["mode"] = mode
    res["conv_criteria"] = error

    return geo, res

def _dimer_step(pes, start_geo, geo_grad, start_mode, trans, rot, metric, max_step = 0.1, scale_step = 1.0, **params):
    """
    Calculates the step the dimer should take
    First improves the mode start_mode to mode_vec to identify the dimer direction
    Than calculates the step for the modified force
    Scales the step (if required)
    """
    curv, mode_vec, info = rot(pes, start_geo, geo_grad, start_mode, metric, **params)

    info["max_step"] = max_step
    step_raw, info_t = trans(pes, start_geo, geo_grad, mode_vec, curv, info)

    info.update(info_t)

    st_sq = metric.norm_up(step_raw, start_geo)
    if st_sq > max_step or curv > 0:
        assert st_sq > 0
        step_raw *= max_step / st_sq
        st_sq = max_step

    info["trans_last_step_length"] = st_sq

    return step_raw * scale_step, mode_vec, info


def main(args):
    from pts.io.read_inp_dimer import read_dimer_input
    from pts.trajectories import empty_traj, traj_every, traj_long, traj_last
    from ase.io import write
    pes, start_geo, start_mode, params, atoms, funcart = read_dimer_input(args[1:], args[0] )
    metric = Default()
    if "trajectory" in params:
        if params["trajectory"] in ["empty", "None", "False"]:
            params["trajectory"] = empty_traj
        elif params["trajectory"] == "every":
            params["trajectory"] = traj_every(atoms, funcart)
        elif params["trajectory"] == "one_file":
            params["trajectory"] = traj_long(atoms, funcart, ["modes", "grads"])
        else:
            params["trajectory"] = traj_last(atoms, funcart)
    else:
            params["trajectory"] = traj_last(atoms, funcart)

    start_mode = start_mode / metric.norm_up(start_mode, start_geo)
    geo, res = dimer(pes, start_geo, start_mode, metric, **params)

    print ""
    print ""
    print "Results as given back from dimer routine"
    print geo
    print res
    print ""
    print "Results DIMER calculation:"
    print "Dimer converged: ", res["trans_convergence"]
    print "Forces left (RMS):", res["abs_force"]
    print "Number of gradient calculations:", res["gradient_calculations"]
    print "Mode at last iteration:"
    print res["mode"]
    print "Final geometry:"
    atoms.set_positions(funcart(geo))
    write("-", atoms)

# python dimer.py [-v]:
if __name__ == "__main__":
    import doctest
    doctest.testmod()
