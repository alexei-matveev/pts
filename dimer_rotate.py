#!/usr/bin/python
from numpy import dot, array, sqrt, arctan, sin, cos, pi, zeros
from copy import deepcopy
from pts.bfgs import LBFGS, BFGS, SR1
from scipy.linalg import eig, eigh
from pts.func import NumDiff
from pts.metric import Default

def rotate_dimer(pes, mid_point, grad_mp, start_mode_vec, metric, dimer_distance = 0.0001, \
    max_rotations = 10, phi_tol = 0.1, rot_conj_gradient = True, **params):
    """
    Rotate the dimer to the mode of lowest curvature

    Rotate after the method of J. K\"{a}stner and Paul Sherwood, J. Chem. Phys. 128 (2008),
                                   014106

    >>> from pts.pes.mueller_brown import MB
    >>> from pts.metric import Metric
    >>> met = Default(None)

    Try at a point:
    >>> start = array([-0.5, 0.5])

    This is far of:
    >>> mode = array([ 0., 1.])
    >>> mode = mode / met.norm_up(mode, start)
    >>> d = 0.0001

    >>> curv, n_mode, dict = rotate_dimer(MB, start, MB.fprime(start), mode, met, dimer_distance = d,
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

    >>> curv, n_mode1, dict = rotate_dimer(MB, start, MB.fprime(start), mode, met, dimer_distance = d)
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

    >>> curv, n_mode2, dict = rotate_dimer(p1, start, p1.fprime(start), mode, met1, dimer_distance = d*0.01,
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
    >>> curv, n_mode, dict = rotate_dimer(pes, start, pes.fprime(start), mode, met,
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

    return l_curv, mode, res

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

# python dimer_rotate.py [-v]:
if __name__ == "__main__":
    import doctest
    doctest.testmod()
