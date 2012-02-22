#!/usr/bin/python
from numpy import dot, array, arctan, sin, cos, pi, zeros
from numpy import sqrt, arccos
from copy import deepcopy
from scipy.linalg import eigh
from pts.func import NumDiff
from pts.metric import Default

VERBOSE = 0

def rotate_dimer_mem(pes, mid_point, grad_mp, start_mode_vec, met, pickle_log, dimer_distance = 0.01, \
    max_rotations = 10, phi_tol = 0.1, interpolate_grad = True, restart = None, **params):
    """
    Rotates  the  dimer  while  keeping  its old  results  in  memory,
    therefore building  slowly a picture of how  the second derivative
    matrix  of the  potential energy  surface at  the  point mid_point
    looks  like.  The  code  is  inspired by  the  Lanczos method  for
    finding  eigenvalues  (and   eigenvectors)  but  as  the  smallest
    eigenvalue  rather than  the one  with largest  absolute  value is
    searched for there could not be used MINRES directly.

    >>> from pts.pes.mueller_brown import MB
    >>> from pts.metric import Metric
    >>> met = Default(None)

    Try at a point:

    >>> start = array([-0.5, 0.5])

    This is far of:

    >>> mode = array([ 0., 1.])
    >>> mode = mode / met.norm_up(mode, start)
    >>> d = 0.000001

    >>> curv, n_mode, info = rotate_dimer_mem(MB, start, MB.fprime(start), mode, met, dimer_distance = d,
    ...                       restart = 1, phi_tol = 1e-7, max_rotations = 10 )
    >>> info["rot_convergence"]
    True

    >>> from pts.func import NumDiff
    >>> from scipy.linalg import eigh
    >>> grad = NumDiff(MB.fprime, h = d)
    >>> h = grad.fprime(start)
    >>> a, V = eigh(h)
    >>> min(a) - info["curvature"] < 0.1
    True

    Here the minimal value of a is the first
    >>> dot(V[0] + n_mode, V[0] + n_mode) < 1e-7
    True

    Thus only rough  estimate for the curvature but  the direction has
    is quite near

    Try another point:
    >>> start = array([-1.0, 1.0])

    This is far of:
    >>> mode = array([-1., 0.])
    >>> mode = mode / met.norm_up(mode, start)
    >>> d = 0.0000001

    >>> curv, n_mode1, info = rotate_dimer_mem(MB, start, MB.fprime(start), mode, met, dimer_distance = d,
    ...                       restart = 1, phi_tol = 1e-7, max_rotations = 10 )
    >>> info["rot_convergence"]
    True

    >>> h = grad.fprime(start)
    >>> a, V = eigh(h)
    >>> min(a) - info["curvature"] < 0.1
    True

    Here the minimal value of a is the first (and the direction of the
    mode vector is reversed)

    >>> ((dot(V[0] - n_mode1, V[0] - n_mode1) < 1e-7) or
    ...   (dot(V[0] + n_mode1, V[0] + n_mode1) < 1e-7))
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

    >>> curv, n_mode2, info = rotate_dimer_mem(p1, start, p1.fprime(start), mode, met1, dimer_distance = d*0.01,
    ...                       restart = 1, phi_tol = 1e-7, max_rotations = 10 )
    >>> info["rot_convergence"]
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
    >>> curv, n_mode, info = rotate_dimer_mem(pes, start, pes.fprime(start), mode, met,
    ...                             dimer_distance = d,
    ...                       restart = 11, phi_tol = 1e-7, max_rotations = 100 )
    >>> info["rot_convergence"]
    True
    """
    shape = start_mode_vec.shape
    # don't change start values, but use flat modes for easier handling
    mode = deepcopy(start_mode_vec)
    mode = mode.flatten()
    mode = mode / met.norm_up(mode, mid_point)

    g0 = deepcopy(grad_mp)

    global grad_calc
    grad_calc = 0
    def grad(vm):
        global grad_calc
        grad_calc = grad_calc + 1
        x = mid_point + dimer_distance *vm
        pickle_log("Lanczos", x)
        return pes.fprime(x) - g0

    # keep all the basis vectors and their forces
    m_basis = [mode]
    g_for_mb = [grad(mode)]

    # Build up matrix for eigenvalues
    H = array([[dot(m_basis[0], g_for_mb[0])]])
    # No possibility of choosing a better start for the first iteration
    new_mode = mode
    new_g    = g_for_mb[0]

    # If we are already converged
    a = array([H[0,0]])
    min_curv = a[0] / dimer_distance

    # ensure that the start value will not pass the test
    old_mode = zeros(new_mode.shape)

    conv = test_lanczos_convergence(new_mode, new_g, old_mode, phi_tol, met, mid_point)

    i = 1
    while i < max_rotations and not conv:

       i = i + 1
       # New basis vector from the interpolation of the last iteration:
       #n_bas = orthogonalize(met.raises(new_g, mid_point), m_basis, met, mid_point)
       n_bas = orthogonalize(met.raises(g_for_mb[-1], mid_point), m_basis, met, mid_point)
       n_bas = n_bas / met.norm_up(n_bas, mid_point)

       m_basis.append(n_bas)
       g_for_mb.append(grad(n_bas))

       # Build up new (larger) matrix
       H_old = H
       hl = len(H_old) + 1
       H = zeros((hl, hl))
       H[:hl-1,:hl-1] = H_old
       H_old = None

       for j, m, g in zip(range(hl), m_basis, g_for_mb):
          # Hessian is symmetric, or should be, enforce it here:
          if j > hl - 3:
              H[j, -1] = (dot(m, g_for_mb[-1]) + dot(m_basis[-1], g)) /2.
              H[-1, j] = H[j, -1]

       #FIXME: maybe we want to use a more specialized algorithm here?
       a, V = eigh(H)

       # We want them in the other order
       V = V.T
       # The vector of the minimal eigenvalue should be dimer direction
       # Here vectors are in m_basis
       min_j = a.argmin()
       min_curv = a[min_j] / dimer_distance
       v_min = V[min_j]
       #print "Iteration", i, a / dimer_distance
       old_mode = new_mode
       new_mode = zeros(mode.shape)
       new_gi   = zeros(new_g.shape)

       # Vector in internal coordinate basis
       for gamma, mb, gm in zip(v_min, m_basis, g_for_mb):
           new_mode = new_mode + gamma * mb
           new_gi = new_gi + gamma * gm

       mode_len = met.norm_up(new_mode, mid_point)
       new_mode = new_mode / mode_len
       new_gi = new_gi / mode_len

       if VERBOSE > 0:
           print ""
           print "For Lanczos iteration", i
           print "New Force: norm, projection "
           print met.norm_down(new_g, mid_point), dot(m_basis[-1], g_for_mb[-1]), dot(m_basis[-2], g_for_mb[-1])
           print "Force difference between g and last g:", met.norm_down(g_for_mb[-1] - g_for_mb[-2], mid_point)
           print "Distances: dimer_distance, between last and current position"
           print dimer_distance, met.norm_up(m_basis[-1] - m_basis[-2], mid_point) * dimer_distance
           print "Eigenvalues:"
           print a / dimer_distance

       if interpolate_grad:
          # need restarts means: we have only one vector in
          # g_for_mb, no use to recalculate it again
          new_g = new_gi
       else:
          new_g = grad(new_mode)
          if VERBOSE > 0:
              print "Differences in grads", sqrt(dot(new_g - new_gi, new_g - new_gi))
              print "Difference in curvature approximations", min_curv - (dot(new_g, new_mode) / dimer_distance)

          # We know an exacter approximation for the curvature:
          min_curv = dot(new_g, new_mode) / dimer_distance

       # Check for convergence
       conv = test_lanczos_convergence(new_mode, new_g, old_mode, phi_tol, met, mid_point)
       if conv:
           break

       if need_restart(restart, i):
           m_basis, g_for_mb, H = start_setting(new_mode, grad)

    mode = new_mode
    pickle_log("Lowest_Mode", mode)
    # this was the shape of the starting mode vector
    mode.shape = shape

    if VERBOSE > 1:
        grad2 = NumDiff(pes.fprime)
        h = grad2.fprime(mid_point)
        a2, V2 = eigh(h)
        print "EIGENMODES", a2
        print "LC", a / dimer_distance
        print "own", a / dimer_distance
        print "Difference to lowest mode", dot(V2[:,0] - mode, V2[:,0] - mode), a2.min() - min_curv
        print V2[:,0]
        print mode
        mode = V2[:,0]
        min_curv = a2[0]
        print mode
        print min_curv

    # some more statistics (to compare to other version)
    fr = rot_force(g0, pes.fprime(mid_point + dimer_distance * mode), mode, met, mid_point)

    # Have them in the right norm
    m_basis = [ m_bas * dimer_distance for m_bas in m_basis]

    res = { "rot_convergence" : conv, "rot_iteration" : i,
            "curvature" : min_curv,"rot_abs_forces" : met.norm_down(fr,mid_point),
            "all_curvs" : a / dimer_distance,
            "rot_updates" : zip(m_basis, g_for_mb),
            "rot_gradient_calculations": grad_calc}

    return min_curv, mode, res

def test_lanczos_convergence(mode, grad, old_mode, phi_tol, met, mid_point):
     """
     If the new gradient is parallel to its mode (= eigenmode)
     or if the new mode did not differ much from the one from the previous
     calculation convergence has been reached.
     """
     fr = rot_force(zeros(len(grad)), grad, mode, met, mid_point)
     fr = fr /met.norm_down(fr, mid_point)
     phi1 = phi_start(zeros(len(grad)), grad, fr, mode, met, mid_point)

     f_phi = dot(mode, grad) / met.norm_down(grad, mid_point)
     mode_down = met.lower(mode, mid_point)
     conv1 = abs(f_phi)
     c_phi1 = dot(mode_down, old_mode)
     conv2 = abs(c_phi1)

     if VERBOSE > 0:
         print "Angle expectation", phi1
         print "Change in modes"
         print c_phi1
         print "Convergence criterias", conv1, conv2, cos(phi_tol)

     if (conv1 > cos(phi_tol)) or \
        (conv2 > cos(phi_tol)) :
         # not arccos(conv2) < phi_tol to allow rounding errors like
         # conv2 = 1.0000000000001
         if VERBOSE > 0:
             print "Rotation convergence criteria 1 (forces parallel to mode):", 1.0 - conv1, arccos(conv1)
             print "Rotation convergence criteria 2 (change in mode approximations):", 1.0 - conv2, arccos(conv2)
         return True
     else:
         return False

def start_setting(mode, grad):
    m_basis = [mode]
    g_for_mb = [grad(mode)]

    # Build up matrix for eigenvalues
    H = array([[dot(m_basis[0], g_for_mb[0])]])

    return m_basis, g_for_mb, H

def need_restart(restart, i):
    """
    Test if the restart option has been set and if yes
    if in the current iteration a restart is wanted
    """
    res = not (restart == None)
    if res:
       res = (i % restart == 0)
    return res

def rotate_dimer(pes, mid_point, grad_mp, start_mode_vec, metric, pickle_log, \
    dimer_distance = 0.0001, max_rotations = 10, phi_tol = 0.1, rot_conj_gradient = True, **params):
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

    >>> curv, n_mode, info = rotate_dimer(MB, start, MB.fprime(start), mode, met, dimer_distance = d,
    ...                             phi_tol = 1e-7, max_rotations = 100 )
    >>> info["rot_convergence"]
    True

    >>> from pts.func import NumDiff
    >>> from scipy.linalg import eigh
    >>> grad = NumDiff(MB.fprime, h = d)
    >>> h = grad.fprime(start)
    >>> a, V = eigh(h)
    >>> min(a) - info["curvature"] < 0.1
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

    >>> curv, n_mode1, info = rotate_dimer(MB, start, MB.fprime(start), mode, met, dimer_distance = d)
    >>> info["rot_convergence"]
    True

    >>> h = grad.fprime(start)
    >>> a, V = eigh(h)
    >>> min(a) - info["curvature"] < 0.1
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

    >>> curv, n_mode2, info = rotate_dimer(p1, start, p1.fprime(start), mode, met1, dimer_distance = d*0.01,
    ...                             phi_tol = 1e-7, max_rotations = 100 )
    >>> info["rot_convergence"]
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
    >>> curv, n_mode, info = rotate_dimer(pes, start, pes.fprime(start), mode, met,
    ...                             dimer_distance = d,
    ...                             phi_tol = 1e-7, max_rotations = 100 )
    >>> info["rot_convergence"]
    True
    """
    shape = start_mode_vec.shape
    # don't change start values, but use flat modes for easier handling
    mode = deepcopy(start_mode_vec)
    mode = mode.flatten()
    mode = mode / metric.norm_up(mode, mid_point)

    old_mode = deepcopy(mode)

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
    m_basis = []
    g_for_mb = []

    i = 1
    while i < max_rotations: # ATTENTION: two break points
        pickle_log("Dimer", x)
        g1 = pes.fprime(x)
        grad_calc += 1
        m_basis.append(x - mid_point)
        g_for_mb.append(g1 -g0)

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

        phi1_1 = phi1
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
        pickle_log("Rot_Trial", x2)
        g2 = pes.fprime(x2)
        grad_calc += 1
        m_basis.append(x2 - mid_point )
        g_for_mb.append(g2 -g0)

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

        if VERBOSE > 0:
            print "Left Informations for Rotation iteration", i -1
            print "Left curvature", curv(g0, g1, old_mode, dimer_distance, metric, mid_point)
            print "For Rotation iteration", i
            print "Norm of rotation force:",  metric.norm_down(fr,mid_point)
            print "Force projection: g1-gm, g2-gm both on dir and mode "
            print dot((g1 - g0), dir), dot((g2 - g0), dir), dot((g1 - g0), mode), dot((g2 - g0), mode)
            print "Force projection in curvature direction:"
            print dot((g1 - g0), mode), dot((g2 - g0), m2)
            print "Force difference between g1 and g2:", metric.norm_down(g1 -g2, mid_point)
            print "Force sizes (m, 1, 2):"
            print metric.norm_down(g0, mid_point), metric.norm_down(g1, mid_point), metric.norm_down(g2, mid_point)
            print "Distances: dimer_distance, between 1 and 2, x1 and new value"
            print dimer_distance, metric.norm_up(x2 -x, mid_point), metric.norm_up(xm -x, mid_point)
            print "Variables for approximation"
            print a0, a1, b1
            print "Rotation Angle: real (first approximation)", phi_m, "(", phi1_1, ")"
            print "curvatures (1,2):", c1, c2
            print "curvature approximation for new point:", cm

        x_old = x
        x = xm
        old_mode = mode
        mode = mm
        l_curv = cm
        #print i,  metric.norm_down(fr, mid_point), l_curv, metric.norm_down(x-x_old,mid_point)
        i += 1

    # this was the shape of the starting mode vector
    mode.shape = shape
    pickle_log("Lowest_Mode", mode)

    if VERBOSE > 1:
        grad = NumDiff(pes.fprime)
        h = grad.fprime(mid_point)
        a, V = eigh(h)
        print "EIGENMODES", a
        print l_curv
        print "Difference to lowest mode", dot(V[:,0] - mode, V[:,0] - mode), a[0] - l_curv

    res = { "rot_convergence" : conv, "rot_iteration" : i,
            "curvature" : l_curv, "rot_abs_forces" : metric.norm_down(fr, mid_point),
            "rot_updates" : zip(m_basis, g_for_mb),
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

def orthogonalize(v_new, vs, met, geo):
    """
    Given a vector (with upper indice) v_new
    and a list of vectors vs (also upper indice)
    returns a vector of v_new where the parallel
    parts to the other vectors are removed
    """
    s = deepcopy(v_new)
    s_down = met.lower(s, geo)
    sum = zeros(s.shape)
    for v in vs:
        sum = sum + dot(v,s_down) * v
    s = s - sum
    s = s / met.norm_up(s, geo)
    return s

def main(args):
    from pts.io.read_inp_dimer import read_dimer_input
    pes, start_geo, start_mode, params, atoms, funcart = read_dimer_input(args[1:], args[0] )
    metric = Default()

    start_mode = start_mode / metric.norm_up(start_mode, start_geo)
    if params["rot_method"] == "lanczos":
        min_curv, mode, res = rotate_dimer_mem(pes, start_geo, pes.fprime(start_geo), start_mode, metric, **params) 
    else:
        min_curv, mode, res = rotate_dimer(pes, start_geo, pes.fprime(start_geo), start_mode, metric, **params) 

    print "Results finding minimal curvature mode:"
    print "Is converged: ", res["rot_convergence"]
    print "Number of gradient calculations:", res["rot_gradient_calculations"]
    print "Curvature, which is supposed to be the lowest: ", min_curv
    print "Mode at last iteration:"
    for md in mode:
        print "  %15.11f " % (md)

# python dimer_rotate.py [-v]:
if __name__ == "__main__":
    import doctest
    doctest.testmod()
