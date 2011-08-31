from numpy import dot, array, sqrt, arctan, sin, cos, pi, zeros
from copy import deepcopy
from pts.bfgs import LBFGS, BFGS, SR1
from scipy.linalg import eigh
from pts.func import NumDiff
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

        >>> from pts.pes.mueller_brown import MB
        >>> from pts.metric import Default

        >>> met = Default(None)
        >>> trans = translate_cg(met, 0.5)

        >>> start = array([-0.5, 0.5])
        >>> mode = array([-1., 0.])

        >>> step, dict = trans(MB, start, MB.fprime(start), mode)
        >>> print step
        [-0.78443852 -0.07155202]

        >>> step, dict = trans(MB, start + step, MB.fprime(start + step), mode)
        >>> print step
        [ 0.64788906 -0.24663059]

        >>> trans = translate_cg(met, 0.5)
        >>> start = array([-0.25, 0.75])
        >>> step, dict = trans(MB, start, MB.fprime(start), mode)
        >>> print step
        [-0.00316205 -0.31033489]
        """
        self.metric = metric
        self.old_force = None
        self.old_step = None
        self.trial_step = trial_step
        self.old_geo = None

    def __call__(self, pes, start_geo, geo_grad, mode_vector):
        """
        the actual step
        """
        # translation "force"
        shape = geo_grad.shape
        mode_vec_down = self.metric.lower(mode_vector, start_geo).flatten()
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
        self.old_force = force
        self.old_step = step
        self.old_geo = start_geo

        step /= self.metric.norm_up(step, start_geo)

        # find how far to go
        # line search, first trial
        trial_step, grad_calc = line_search(start_geo, step, self.trial_step, pes, self.metric, mode_vector, force)
        step = trial_step * step


        dict = {"trans_abs_force" : self.metric.norm_down(force, start_geo),
                "trans_gradient_calculations": grad_calc}

        step.shape = shape

        return step, dict

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


class translate_lbfgs():
    def __init__(self, metric):
        """
        Calculates a translation step of the dimer (not scaled)
        Uses an approximated hessian hess

        step is Newton (Hessian) step for grads:

        g = g_0 - 2 * (g_0, m) * m

        Consider metric

        >>> from pts.pes.mueller_brown import MB
        >>> from pts.metric import Default

        >>> met = Default(None)
        >>> trans = translate_lbfgs(met)

        >>> start = array([-0.5, 0.5])
        >>> mode = array([-1., 0.])

        >>> step, dict = trans(MB, start, MB.fprime(start), mode)
        >>> print step
        [-1.04187497 -0.0950339 ]
        >>> step, dict = trans(MB, start + step, MB.fprime(start + step), mode)
        >>> print step
        [-4157.56285284  -445.26019983]

        >>> trans = translate_lbfgs(met)
        >>> start = array([-0.25, 0.75])
        >>> step, dict = trans(MB, start, MB.fprime(start), mode)
        >>> print step
        [-0.03655155 -3.58730495]
        """
        self.hess = SR1()
        self.old_force = None
        self.old_geo = None
        self.metric = metric

    def __call__(self, pes, start_geo, geo_grad, mode_vector):
        """
        the actual step
        """
        force_raw = - geo_grad
        shape = force_raw.shape
        force_raw = force_raw.flatten()
        mode_vec_down = self.metric.lower(mode_vector, start_geo).flatten()
        force_para = dot(force_raw, mode_vector.flatten()) * mode_vec_down
        force_perp = force_raw - force_para
        force = force_perp - force_para
        if not self.old_force == None:
            self.hess.update(start_geo - self.old_geo, force - self.old_force)

        step = self.hess.inv(force)
        mat = self.hess.H
        a, V = eigh(mat)

        self.old_force = force
        self.old_geo = start_geo

        dict = {"trans_abs_force" : self.metric.norm_down(force, start_geo),
                "trans_gradient_calculations": 0}

        step.shape = shape

        return step, dict

class translate_sd():
    def __init__(self, metric, trial_step):
        """
        use steepest decent to determine in which direction to do the next step

        >>> from pts.pes.mueller_brown import MB
        >>> from pts.metric import Default

        >>> tr_step = 0.3
        >>> met = Default(None)
        >>> trans = translate_sd(met, tr_step)

        >>> start = array([-0.5, 0.5])
        >>> mode = array([-1., 0.])

        >>> step, dict = trans(MB, start, MB.fprime(start), mode)
        >>> print step
        [-72.93124759  -6.65237324]

        >>> trans = translate_sd(met, tr_step)
        >>> start = array([-0.25, 0.75])
        >>> step, dict = trans(MB, start, MB.fprime(start), mode)
        >>> print step
        [  -2.55860839 -251.11134643]
        """
        self.metric = metric
        self.trial_step = trial_step

    def __call__(self, pes, start_geo, geo_grad, mode_vector):
        """
        the actual step
        """
        force_raw = - geo_grad
        shape = force_raw.shape
        force_raw = force_raw.flatten()
        mode_vec_down = self.metric.lower(mode_vector, start_geo).flatten()
        force = force_raw -2. * dot(force_raw, mode_vector.flatten()) * mode_vec_down

        # actual steepest decent step
        step = self.metric.raises(force, start_geo)
        step /= sqrt(dot(step, force))

        trial_step, grad_calc = line_search(start_geo, step, self.trial_step, pes, self.metric, mode_vector, force)
        step = trial_step * step

        step.shape = shape

        dict = {"trans_abs_force" : self.metric.norm_down(force, start_geo),
                "trans_gradient_calculations": grad_calc}
        return step, dict


trans_dict = {
               "conj_grad" : translate_cg,
               "lbfgs"     : translate_lbfgs,
               "steep_dec" : translate_sd
             }

def dimer(pes, start_geo, start_mode, metric, max_translation = 100000000, max_gradients = None, \
       trans_converged = 0.00016, trans_method = "conj_grad", start_step_length = 0.7,   **params):
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

    # do not change them
    geo = deepcopy(start_geo) # of dimer middle point
    mode = deepcopy(start_mode) # direction of dimer
    old_force = None
    step_old = 0
    conv = False
    grad_calc = 0
    error_old = 0

    i = 0
    # main loop:
    while i < max_translation:
         #ATTENTION: 2 break points, first for convergence, second at end
         # two ways of maximum iteration: as max_translation or
         #        as count of gradient calls, second case needs
         #        additional break point
         #        it is only checked for only maximum number of gradient calls
         #        if max_translation > maximum number of gradient calls
         grad = pes.fprime(geo)
         grad_calc += 1

         # Test for convergence, converged if saddle point is reached
         abs_force = metric.norm_down(grad, geo)
         error = max(abs(grad.flatten()))
         if error < trans_converged:
              # Breakpoint 1: calculation has converged
              conv = True
              break

         # calculate one step of the dimer, also update dimer direction
         # res is dictionary with additional results
         step, mode, res = _dimer_step(pes, geo, grad, mode, trans, metric, **params)
         grad_calc += res["rot_gradient_calculations"] + res["trans_gradient_calculations"]
         #print "iteration", i, error, metric.norm_down(step, geo)
         if i > 0 and error_old - error < 0:
             print "Error is growing"
         if i > 0:
             if dot(step, metric.lower(step_old, geo)) < 0:
                print "Step changed direction"
         step_old = step
         geo = geo + step
         #print "Step",i , pes(geo-step), abs_force, max(grad), res["trans_last_step_length"], res["curvature"], res["rot_gradient_calculations"]
         i += 1
         error_old = error

         if not max_gradients == None and max_gradients <= grad_calc:
              # Breakpoint 2: for counting gradient calculations instead of translation steps
              break

    # all gradient calculations makes sense
    # gradient calculations only of last rotation not so much
    del res["rot_gradient_calculations"]
    del res["trans_gradient_calculations"]
    res["gradient_calculations"] = grad_calc

    # add some more results to give back
    res["trans_convergence"] = conv
    res["abs_force"] = abs_force
    res["trans_steps"] = i
    res["mode"] = mode

    return geo, res

def _dimer_step(pes, start_geo, geo_grad, start_mode, trans, metric, max_step = 0.265, scale_step = 1.0, **params):
    """
    Calculates the step the dimer should take
    First improves the mode start_mode to mode_vec to identify the dimer direction
    Than calculates the step for the modified force
    Scales the step (if required)
    """
    mode_vec, dict = _rotate_dimer(pes, start_geo, geo_grad, start_mode, metric, **params)

    step_raw, dict_t = trans(pes, start_geo, geo_grad, mode_vec )

    dict.update(dict_t)

    st_sq = metric.norm_down(step_raw, start_geo)
    if st_sq > max_step:
        step_raw *= max_step / st_sq
        st_sq = max_step

    dict["trans_last_step_length"] = st_sq

    return step_raw * scale_step, mode_vec, dict


def _rotate_dimer(pes, mid_point, grad_mp, start_mode_vec, metric, dimer_distance = 0.01, \
    max_rotations = 10, phi_tol = 0.001, rot_conj_gradient = True, **params):
    """
    Rotate the dimer to the mode of lowest curvature

    Rotate after the method of J. K\"{a}stner and Paul Sherwood, J. Chem. Phys. 128 (2008),
                                   014106

    >>> from pts.pes.mueller_brown import MB
    >>> from pts.metric import Default, Metric
    >>> met = Default(None)

    Try at a point:
    >>> start = array([-0.5, 0.5])

    This is far of:
    >>> mode = array([ 0., 1.])
    >>> mode = mode / met.norm_up(mode, start)
    >>> d = 0.0001

    >>> n_mode, dict = _rotate_dimer(MB, start, MB.fprime(start), mode, met, dimer_distance = d,
    ...                             phi_tol = 1e-7, max_rotations = 100 )
    >>> dict["rot_convergence"]
    True

    >>> from pts.func import NumDiff
    >>> from scipy.linalg import eigh
    >>> grad = NumDiff(MB.fprime, h = d)
    >>> h = grad.fprime(start)
    >>> a, V = eigh(h)
    >>> min(a) - dict["curvature"] < 0.1
    True

    Here the minimal value of a is the first
    >>> dot(V[0] - n_mode, V[0] - n_mode) < 1e-7
    True

    Thus only rough estimate for the curvature but the direction has
    is quite near

    Try another point:
    >>> start = array([-1.0, 1.0])

    This is far of:
    >>> mode = array([-1., 0.])
    >>> mode = mode / met.norm_up(mode, start)
    >>> d = 0.0001

    >>> n_mode1, dict = _rotate_dimer(MB, start, MB.fprime(start), mode, met, dimer_distance = d)
    >>> dict["rot_convergence"]
    True

    >>> h = grad.fprime(start)
    >>> a, V = eigh(h)
    >>> min(a) - dict["curvature"] < 0.1
    True

    Here the minimal value of a is the first
    (and the direction of the mode vector is reversed)
    >>> (dot(V[0] - n_mode1, V[0] - n_mode1) < 1e-7)
    ...  or (dot(V[0] + n_mode1, V[0] + n_mode1) < 1e-7)
    True

    # test Metric and different funcs
    >>> from pts.test.testfuns import mb1, mb2
    >>> from pts.func import compose

    For mb1:
    Use the same start point as before
    >>> start_c = array([-0.5, 0.5])
    >>> fun = mb2()
    >>> start = fun.pinv(start_c)
    >>> p1 = compose(MB, fun)

    Prepare Metric functions
    >>> met1 = Metric(fun)

    This is far of:
    >>> mode = array([-1., 5.])

    >>> n_mode2, dict = _rotate_dimer(p1, start, p1.fprime(start), mode, met1, dimer_distance = d*0.01,
    ...                             phi_tol = 1e-7, max_rotations = 100 )
    >>> dict["rot_convergence"]
    True

    Result should be the same as before:
    >>> n_c = (fun(start + d * n_mode2) - fun(start))
    >>> n_c = n_c / met.norm_up(n_c, start)
    >>> dot(n_c + n_mode, n_c + n_mode) < 1e-3 or dot(n_c - n_mode, n_c - n_mode) < 1e-3
    True

    A bigger example using Ar4
    >>> from ase import Atoms
    >>> ar4 = Atoms("Ar4")
    >>> from pts.qfunc import QFunc
    >>> from pts.cfunc import Justcarts
    >>> pes = compose(QFunc(ar4), Justcarts())

    >>> w=0.39685026
    >>> C = array([[-w,  w,  w],
    ...            [ w, -w,  w],
    ...            [ w, -w, -w],
    ...            [-w,  w, -w]])
    >>> start = C.flatten()
    >>> from numpy import zeros
    >>> mode = zeros(12)
    >>> mode[1] = 1
    >>> n_mode, dict = _rotate_dimer(pes, start, pes.fprime(start), mode, met,
    ...                             dimer_distance = d,
    ...                             phi_tol = 1e-7, max_rotations = 100 )
    >>> dict["rot_convergence"]
    True
    """
    shape = start_mode_vec.shape
    # don't change start values, but use flat modes for easier handling
    mode = deepcopy(start_mode_vec)
    mode = mode.flatten()
    mode = mode / metric.norm_up(mode, mid_point)

    g0 = deepcopy(grad_mp)

    md2 = deepcopy(mode)
    md2.shape = mid_point.shape
    # first dimer, only look at one of the images
    x = mid_point + md2 * dimer_distance
    fr_old = None
    dir_old = None

    # variables needed for keeping track of calculation
    conv = False
    l_curv = None
    l_ang = None
    grad_calc = 0

    i = 0
    while i < max_rotations: # ATTENTION: two break points
        g1 = pes.fprime(x)
        grad_calc += 1

        #"rotation force"
        fr = rot_force(g0, g1, mode, metric, mid_point)


        # minimization will take place in plane spanned by mode and dir
        dir = metric.raises(fr, mid_point)

        # modified conjugate gradient
        if rot_conj_gradient and not fr_old == None:
            # attention: here dir is just fr with upper indices
            gamma = dot((fr - fr_old), dir) / dot(fr, dir)
            # direction of old part:
            # dir_s should be orthogonal to mode but lying on old plane
            dir_old_down = metric.lower(dir_old, x)
            dir_s =  dir_old - ( dot(dir_old_down, mode) * mode\
                                / metric.norm_up(mode, mid_point))
            dir_s /= metric.norm_up(dir_s, x)
            # modifated conjugate gradient
            dir = dir + gamma * metric.norm_up(dir_old, mid_point) * dir_s

        fr_old = fr
        # dir_old with length
        dir_old = dir
        dir /= metric.norm_up(dir, mid_point)

        # first angle approximation, (we need two picture for minimization)
        phi1 = phi_start(g0, g1, dir, mode, metric, mid_point)

        l_ang = phi1
        if abs(phi1) < phi_tol:
            # FIRST BREAK POINT: first approximation did nearly not move dimer
            conv = True
            l_curv = curv(g0, g1, mode, dimer_distance, metric, mid_point)
            break

        # do not rotate for a too small value, else the differences will be useless
        # better interpolate
        if phi1 < 0:
            phi1 = -pi/4.
        else:
            phi1 = pi/4.

        # calculate values for dimer rotated for phi1
        x2, m2  = rotate_phi(mid_point, mode, dir, phi1, dimer_distance, metric)
        g2 = pes.fprime(x2)
        grad_calc += 1

        # curvature approximations
        c1 = curv(g0, g1, mode, dimer_distance, metric, mid_point)
        c2 = curv(g0, g2, m2, dimer_distance, metric, mid_point)

        # approximate rotated curvature with:
        # C(phi) = a0/2 + a1 cos(2 phi) + b1 sin(2 phi)
        b1 =  dot((g1 - g0), dir) / dimer_distance
        a1 = ( c1 - c2 + b1 * sin(2. * phi1)) / (1. - cos(2. * phi1))
        a0 = 2. * (c1 - a1)

        # then minimize
        phi_m = 0.5 * arctan(b1/a1)
        cm = a0/2. + a1 * cos(2. * phi_m) + b1 * sin(2. * phi_m)

        # was search for extremum, could as well be maximum
        if cm > min(c1, c2):
            # in this case minimum is perpendicular to it and also in plane:
            phi_m = phi_m + pi / 2.
            cm = a0/2. + a1 * cos(2. * phi_m) + b1 * sin(2. * phi_m)

        l_ang = phi_m
        if abs(phi_m) < phi_tol:
            # SECOND BREAK POINT: minimum in plane does hardly move
            conv = True
            l_curv = c1
            break

        # prepare for next rotation step: start with minimum of last iteration
        xm, mm  = rotate_phi(mid_point, mode, dir, phi_m, dimer_distance,  metric)
        x_old = x
        x = xm
        mode = mm
        l_curv = cm
        #print i,  metric.norm_down(fr, mid_point), l_curv, metric.norm_down(x-x_old,mid_point)
        i += 1

    # this was the shape of the starting mode vector
    mode.shape = shape

   #grad = NumDiff(pes.fprime)
   #h = grad.fprime(mid_point)
   #a, V = eigh(h)
   #print "EIGENMODES", a
   #print l_curv
   #print "Difference to lowest mode", dot(V[0] - mode, V[0] - mode), a[0] - l_curv

    res = { "rot_convergence" : conv, "rot_iteration" : i + 1,
            "curvature" : l_curv, "rot_abs_forces" : metric.norm_down(fr,mid_point),
            "rot_last_angle": l_ang, "rot_gradient_calculations": grad_calc}

    return mode, res

def rot_force(g0, g1, m, metric, mid):
    """
    Rotation force: factor 2 because dimer normally at x0+/-d
    Use force perpendicular to the mode direction m:
    g_pr = g - (g,m) m

    consider metric
    """
    assert abs(metric.norm_up(m, mid) - 1.) < 1e-7
    f = - 2. * ((g1 - g0) - dot((g1 - g0), m) * metric.lower(m, mid))
    return f

def phi_start(g0, g1, d, m, met, mid):
    """
    phi1 = - 0.5 arctan (dC/dphi) / (2 |C|)

    dC/dphi = 2 delta(g) * d / Delta

    C = delta(g) * m / Delta
    """
    assert abs(met.norm_up(d, mid) - 1.) < 1e-7
    assert abs(met.norm_up(m, mid) - 1.) < 1e-7
    var = dot( (g1 - g0), d) / abs(dot(g1-g0, m))
    return - 0.5 * arctan(var)

def rotate_phi(mid, m, d, phi, l, met):
    """
    Rotate the dimer from mid +(/-) m * l to mid + d * l
    """
    assert abs(met.norm_up(m, mid) - 1.) < 1e-7
    assert abs(met.norm_up(d, mid) - 1.) < 1e-7
    vec = m * cos(phi) + d * sin(phi)
    vec = vec / met.norm_up(vec, mid)
    vec.shape = mid.shape
    return mid + vec * l, vec.flatten()

def curv(g0, g1, m, d, met, mid):
   """
   Curvature in direction m
   The dimer has force g1 (and g2 not shown), dimer midpoint has
   g0 and x1 and x0 have distance d
   """
   assert abs(met.norm_up(m, mid) - 1.) < 1e-7
   return dot((g1 - g0), m) / d

# python dimer.py [-v]:
if __name__ == "__main__":
    import doctest
    doctest.testmod()
