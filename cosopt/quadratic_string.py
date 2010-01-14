def test_QSM():
    from scipy.optimize import fmin_bfgs
    f_test = lambda x: True
    rho_quartic = lambda x: (x*(x-1))**2
    rho_flat = lambda x: 1
    surf_plot = SurfPlot(GaussianPES())
    qc_driver = GaussianPES()

    reagents = [reactants, products]
    gs = GrowingString(reagents, qc_driver, 
        beads_count=8, rho=rho_flat, growing=False)

    # Wrapper callback function
    def mycb(x):
        gs.update_path(x, respace = True)
        gs.plot()
        return gs.get_state_vec()

    from scipy.optimize.lbfgsb import fmin_l_bfgs_b
    from scipy.optimize import fmin_cg

    qs = QuadraticStringMethod(gs, callback = mycb, update_trust_rads = True)
    
    opt = qs.opt_global_local_wrap()

    gs.plot()
    surf_plot.plot(opt)

def test_GQSM():
    """Test the GROWING Quadratic String Method"""
    from scipy.optimize import fmin_bfgs
    f_test = lambda x: True
    rho_quartic = lambda x: (x*(x-1))**2
    rho_flat = lambda x: 1
    surf_plot = SurfPlot(GaussianPES())
    qc_driver = GaussianPES()

    reagents = [reactants, products]
    gs = GrowingString(reagents, qc_driver, 
        beads_count=8, rho=rho_flat, growing=True)

    # Wrapper callback function
    def mycb(x):
        gs.update_path(x, respace = True)
        gs.plot()
        return gs.get_state_vec()

    from scipy.optimize.lbfgsb import fmin_l_bfgs_b
    from scipy.optimize import fmin_cg

    qs = QuadraticStringMethod(gs, callback = mycb)
    
    while True:
        opt = qs.opt_global_local_wrap()

        # grow the string, but break if not possible
        if not gs.grow_string():
            break

    gs.plot()
    surf_plot.plot(opt)

def vecnorm(x, ord=2):
    import numpy
    if ord == Inf:
        return numpy.amax(abs(x))
    elif ord == -Inf:
        return numpy.amin(abs(x))
    else:
        return numpy.sum(abs(x)**ord,axis=0)**(1.0/ord)


def wrap_function(function, args):
    ncalls = [0]
    def function_wrapper(x):
        ncalls[0] += 1
        return function(x, *args)
    return ncalls, function_wrapper

#_epsilon = sqrt(finfo(float).eps)

def dump_mat(mat):
    for row in mat:
        for col in row:
            if col > 0:
                print "+",
            elif col < 0:
                print "-",
            else:
                print "0",
        print

class QuadraticStringMethod():
    """Quadratic String Method Functions
       ---------------------------------

    The functions in this class are described in the reference:

    [QSM] Burger and Yang, J Chem Phys 2006 vol 124 054109."""

    def __init__(self, string = None, callback = None, gtol = 0.05, update_trust_rads = False):
        self.__string = string
        self.__callback = callback
        
        self.__init_trust_rad = 0.05
        self.__h0 = 0.05
        self.__max_step_err = 0.01
        self.__dims = self.__string.dimension

        self.__TRUST_EXCEEDED = 1
        self.__DIRECTION_CHANGE = 2
        self.__MAX_QUAD_ITERATIONS = 100
        self.__gtol = gtol

        self.__update_trust_rads = update_trust_rads

    def mytest(self):
        dims = 3

        # test update_H()
        delta = arange(dims)
        gamma = ones(dims)
        H = arange(dims*dims)
        H.shape = (-1, dims)

        newH = self.update_H(delta, gamma, H)
        print "newH =", newH

        # test update_trust_rads()
        e = array((10,10))
        prev_e = array((12,12))
        m = array((10.2, 10.3))
        prev_m = array((12.5, 12.8))
        dx = array((0.5, 0.5, 0.6, 0.7))
        dims = 2
        prev_trust_rads = array((0.25, 0.25))
        new_trust_rads = self.update_trust_rads(e, prev_e, m, prev_m, dx, dims, prev_trust_rads)
        print new_trust_rads

    def mytest_rk45(self):
        trust_rad = 1000.
        """x0 = 1.0
        dx_on_dt = lambda xp: -0.1*xp
        x, deltaX = self.int_rk45(x0, dx_on_dt, trust_rad, verbose=True)
        print "Answers: x =", x, "deltaX =", deltaX"""

        # projectile motion
        vx0 = 100.0
        vy0 = 1000.0
        dx_on_dt = lambda x: array([vx0, vy0 - 9.8*x[0]/1.0])
        x0 = array((3.,4.))
        x, deltaX, path = self.int_rk45(x0, dx_on_dt, trust_rad, verbose=True)
        print "Answers: x =", x, "deltaX =", deltaX
        for p in path:
            print "%d\t%d" % (p[0], p[1])

    def opt_global_local_wrap(self):
        """Optimises a string by optimising its control points separately."""

        x0 = self.__string.get_state_vec()

        assert len(x0[0]) == self.__dims

        x = deepcopy(x0)
        N = self.__string.beads_count

        def update_eg(my_x):
            """Returns energy and gradient for state vector my_x."""
            e = self.__string.obj_func(my_x, individual_energies = True)
            g = self.__string.obj_func_grad(my_x).flatten()
#            g.shape = (-1, self.__dims)

            return e, g

        # initial parameters for optimisation
        e, g = update_eg(x)
        x = self.__callback(x)

        trust_rad = ones(N) * self.__init_trust_rad # ???
        H = []
        Hi = linalg.norm(g) * eye(self.__dims)
        for i in range(N):
            H.append(deepcopy(Hi))
        H = array(H)

#        print "type(x)",type(x)
#        raw_input()
        # optimisation of whole string to semi-global min
        k = 0
        m = e # quadratically estimated energy
        while True:
            prev_x = deepcopy(x)

            # optimisation of whole string on local quadratic surface
            x = self.quadratic_opt(x, g, H)

            # respace, recalculate splines
            self.__string.update_path(x, respace=True)

#            print "x",x
#            print "prev_x",prev_x
            delta = x - prev_x

            # update quadratic estimate of new energy
            prev_m = m
            m = e + self.mydot(delta, g) + 0.5 * self.mydot(delta, self.mydot(H, delta))

            # update real energy
            prev_g = g
            prev_e = e
            e, g = update_eg(x) # TODO: Question: will the state of the string be updated(through respacing)?

            # callback. Note, in previous versions, self.update_path was called by callback function
            print common.line()
            x = self.__callback(x)

#            print "linalg.norm(g)", linalg.norm(g)
            if common.rms(g) < self.__gtol:
                print "Sub cycle finished", common.rms(g), "<", self.__gtol
                break

            if self.__update_trust_rads:
                prev_trust_rad = trust_rad
                trust_rad = self.update_trust_rads(e, prev_e, m, prev_m, delta, prev_trust_rad)

            gamma = g - prev_g
            prev_H = H
            H = self.update_H(delta, gamma, prev_H)

            k += 1

        return x
            
    def opt(self):
        """Convenience wrapper for main optimisation function."""
        return self.opt_global_local_wrap()

    def update_H(self, deltas, gammas, Hs, use_tricky_update = True):
        """Damped BFGS Hessian Update Scheme as described in Ref. [QSM]., equations 14, 16, 18, 19."""

        deltas.shape = (-1, self.__dims)
        gammas.shape = (-1, self.__dims)

        Hs_new = []
        for i in range(self.__string.beads_count):
            H = Hs[i]
            delta = deltas[i]
            gamma = gammas[i]

            if use_tricky_update:
                if dot(delta, gamma) <= 0:
                    H_new = H
                    Hs_new.append(H_new)
                    continue

                if dot(delta, gamma) > 0.2 * dot(delta, dot(H, delta)):
                            theta = 1
                else:
                    theta = 0.8 * dot(delta, dot(H, delta)) / (dot(delta, dot(H, delta)) - dot(delta, gamma))
                
                gamma = theta * gamma + (1 - theta) * dot(H, delta)

            tmp1 = dot(delta, H)
            numerator1 = dot(H, outer(delta, tmp1))
            denominator1 = dot(delta, dot(H, delta))

            numerator2 = outer(gamma, gamma)
            denominator2 = dot(gamma, delta)

            H_new = H - numerator1 / denominator1 + numerator2 / denominator2

            # Guard against when gradient doesn't change for a particular point.
            # This typically happens for the reactant/product points which are
            # already at a minimum.
            if isfinite(H_new).flatten().tolist().count(False) > 0: # is there a more elegant expression?
                H_new = H * 0

            """    if linalg.norm(dot(H_new, delta)) > 0.2:
                    H_new = eye(self.__dims) * 0.01""" # what was this for?

            Hs_new.append(H_new)

        return array(Hs_new)

    def mydot(self, super_vec1, super_vec2):
        """Performs element wise dot multiplication of vectors of 
        vectors/matrices (super vectors/matrices) with each other."""
        N = self.__string.beads_count
        d = self.__dims

        def set_shape(v):
            if v.size == N * d: # vector of vectors
                v.shape = (N, d)
            elif v.size % (N * d) == 0: # vector of matrices
                v.shape = (N, d, -1)
            else:
                raise Exception("vector %s inappropriate size for resizing with %d and %d" % (v, N, d))

        super_vec1 = deepcopy(super_vec1)
        set_shape(super_vec1)

        super_vec2 = deepcopy(super_vec2)
        set_shape(super_vec2)

        list = []
        for i in range(N):
            v1 = super_vec1[i]
            v2 = super_vec2[i]

            list.append(dot(v1, v2))
        a = array(list).flatten()

        return a
            
    def update_trust_rads(self, e, prev_e, m, prev_m, dx, prev_trust_rads):
        """Equations 21a and 21b from [QSM]."""
        
        s = (-1, self.__dims)
        dx.shape = s
        N = self.__string.beads_count

        new_trust_rads = []

        assert len(prev_trust_rads) == N

#        lg.debug("e = {0}".format(e))
#        lg.debug("prev_e = {0}".format(prev_e))
#        lg.debug("m = {0}".format(m))
#        lg.debug("prev_m = {0}".format(prev_n))
        for i in range(N):
            rho = (e[i] - prev_e[i]) / (m[i] - prev_m[i])

            # guards against case e.g. when end points are not being moved and
            # hence the denominator is zero
            if isnan(rho):
                rho = 1
            lg.info("rho = " + str(rho))

            rad = prev_trust_rads[i]
            if rho > 0.75 and 1.25 * linalg.norm(dx[i]) > rad:
                rad = 2 * rad # was 2*rho

            elif rho < 0.25:
                rad = 0.25 * linalg.norm(dx[i])

            new_trust_rads.append(rad)

        lg.info("new trust radii = " + str(new_trust_rads))
        #wt()
        return new_trust_rads

    def calc_tangents(self, state_vec):
        """Based on a path represented by state_vec, returns its unit tangents."""

        path_rep = PathRepresentation(state_vec, self.__string.beads_count)
        path_rep.regen_path_func()
        tangents = path_rep.recalc_path_tangents()
        return tangents

    def quadratic_opt(self, x0, g0, H):
        """Optimiser used to optimise on quadratic surface."""

        from numpy.linalg import norm

        dims = self.__dims
        x = deepcopy(x0)
        x0.shape = (-1, dims)
        x.shape = (-1, dims)
        prev_x = deepcopy(x)
        N = self.__string.beads_count
        g0 = deepcopy(g0)
        g0.shape = (-1, dims)     # initial gradient of quadratic surface
        assert(len(H[0])) == dims # hessian of quadratic surface

        # temporary
        trust_rad = ones(N) * self.__init_trust_rad

        h = ones(N) * self.__h0 # step size
        k = flag = 0
        while True:
            
            tangents = self.calc_tangents(x)

            # optimize each bead in the string
            prev_g = deepcopy(g0)
            for i in range(N):

#                print "i =", i
#                print "H =", H[i]
#                print "x =", x[i]
                dx_on_dt = lambda myx: self.dx_on_dt_general(x0[i], myx, g0[i], H[i], tangents[i])
                
                step4, step5 = self.rk45_step(x[i], dx_on_dt, h[i])

                if linalg.norm(step4, ord=inf) == 0.0:
                    continue

                prev_x[i] = x[i]
        
                # guard against case when even initial step goes over trust radius
                if linalg.norm(step4) > trust_rad[i]:
                    step4 = step4 / linalg.norm(step4) * trust_rad[i]

                x[i] += step4

                g = -dx_on_dt(x[i])

                err = norm(step5 - step4)

                if norm(x[i] - x0[i]) > trust_rad[i]:
                    #lg.debug("Trust radius exceeded for point {0}".format(i))
                    flag = self.__TRUST_EXCEEDED

                elif dot(g, prev_g[i]) < 0: # angle_change >= pi / 2: # is this correct?
                    #lg.debug("Direction change for point {0}".format(i))
                    print "g = ", g, "g_prev =", prev_g[i]
                    flag = self.__DIRECTION_CHANGE

                if True:
                    # adaptive step size
                    #print "Step size for point", i, "scaled from", h[i],
                    h[i] = h[i] * abs(self.__max_step_err / err)**(1./5.)
                    #print "to", h[i], ". Error =", err


                prev_g[i] = g

            k += 1

            if k > self.__MAX_QUAD_ITERATIONS or flag != 0:
                #print "flag = ",flag, "k =", k
                #raw_input("Wait...\n")
                break

        #x = x.flatten()
        return x 

    def dx_on_dt_general(self, x0, x, g0, H, tangent):
#        print g0
#        print H
#        print x
#        print x0
        approx_grad = g0 + dot(H, x - x0)
        perp_component = eye(self.__dims) - outer(tangent, tangent)
        dx_on_dt_tmp = dot(approx_grad, perp_component)
        return -dx_on_dt_tmp

    def int_rk45(self, x0, dx_on_dt, trust_rad, verbose = False):
        """Performs adaptive step size Runge-Kutta integration. Based on:
            http://www.ecs.fullerton.edu/~mathews/n2003/RungeKuttaFehlbergMod.html
        and
            http://en.wikipedia.org/wiki/Runge-Kutta"""
        
        # TODO: dummy constant value, eventually must be generated 
        # in accordance with reference [QSM].
        eps0 = 0.1

        x = array([x0]).flatten()
        h = self.__h0

        srch_dir = dx_on_dt(x)
        k = 0
        path = []
        while True:
            # two integration steps for Runge Kutta 4 and 5
            step4, step5 = self.rk45_step(x, dx_on_dt, h)

            err = linalg.norm(step5 - step4)

            if verbose:
                print "step4 =", step4, "step5 =", step5
                print "x =", x, "srch_dir =", srch_dir
                print "h =", h
                print "err =", err

            path.append(deepcopy(x))
            x_prev = x
            x += step4
            prev_srch_dir = srch_dir
            srch_dir = dx_on_dt(x)

            if linalg.norm(x - x0) > trust_rad:
                print "trust rad"
                break
            #print "dot(srch_dir, prev_srch_dir) =", dot(srch_dir, prev_srch_dir)
            if dot(srch_dir, prev_srch_dir) < 0:
                print "direc change: sd =", srch_dir, "prev_sd =", prev_srch_dir
                break
            if k > 500:
                print "max iters"
                break

            s = (eps0 * h / 2. / err)**0.25
            h = min(s*h, 0.2)
            k += 1
            print

        return x, (x-x0), path


    def rk45_step(self, x, f, h):
        
        k1 = h * f(x)
        x2 = x + 1.0/4.0 * k1
        k2 = h * f(x2)
        x3 = x + 3.0/32.0 * k1 + 9.0/32.0 * k2
        k3 = h * f(x3)
        x4 = x + 1932./2197. * k1 - 7200./2197. * k2 + 7296./2197. * k3
        k4 = h * f(x4)
        x5 = x + 439./216. * k1 - 8. * k2 + 3680./513. * k3 - 845./4104. * k4
        k5 = h * f(x5)
        x6 = x - 8./27.*k1 + 2.*k2 - 3544./2565. * k3 + 1859./4104.*k4 - 11./40. * k5
        k6 = h * f(x6)

        xs = array((x,x2,x3,x4,x5,x6))
        ks = array((k1, k2, k3, k4, k5, k6))

        step4 = 25./216.*k1 + 1408./2565.*k3 + 2197./4104.*k4 - 1./5.*k5
        step5 = 16./135.*k1 + 6656./12825.*k3 + 28561./56430.*k4 - 9./50.*k5 + 2./55.*k6

        #print "*******STEP", x, h, step4
        return step4, step5


