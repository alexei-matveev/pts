def my_fmin_bfgs(f, x0, fprime=None, args=(), gtol=1e-5, norm=Inf,
              epsilon=_epsilon, maxiter=None, full_output=0, disp=1,
              retall=0, callback=None):
    """Minimize a function using the BFGS algorithm.
    
    :Parameters:

      f : the Python function or method to be minimized.
      x0 : ndarray
        the initial guess for the minimizer.

      fprime : a function to compute the gradient of f.
      args : extra arguments to f and fprime.
      gtol : number
        gradient norm must be less than gtol before succesful termination
      norm : number
        order of norm (Inf is max, -Inf is min)
      epsilon : number
        if fprime is approximated use this value for
                 the step size (can be scalar or vector)
      callback : an optional user-supplied function to call after each
                  iteration.  It is called as callback(xk), where xk is the
                  current parameter vector.

    :Returns: (xopt, {fopt, gopt, Hopt, func_calls, grad_calls, warnflag}, <allvecs>)

      xopt : ndarray
        the minimizer of f.

      fopt : number
        the value of f(xopt).
      gopt : ndarray
        the value of f'(xopt).  (Should be near 0)
      Bopt : ndarray
        the value of 1/f''(xopt).  (inverse hessian matrix)
      func_calls : number
        the number of function_calls.
      grad_calls : number
        the number of gradient calls.
      warnflag : integer
                  1 : 'Maximum number of iterations exceeded.'
                  2 : 'Gradient and/or function calls not changing'
      allvecs  :  a list of all iterates  (only returned if retall==1)

    :OtherParameters:

      maxiter : number
        the maximum number of iterations.
      full_output : number
        if non-zero then return fopt, func_calls, grad_calls,
                     and warnflag in addition to xopt.
      disp : number
        print convergence message if non-zero.
      retall : number
        return a list of results at each iteration if non-zero

    :SeeAlso:

      fmin, fmin_powell, fmin_cg,
             fmin_bfgs, fmin_ncg -- multivariate local optimizers
      leastsq -- nonlinear least squares minimizer

      fmin_l_bfgs_b, fmin_tnc,
             fmin_cobyla -- constrained multivariate optimizers

      anneal, brute -- global optimizers

      fminbound, brent, golden, bracket -- local scalar minimizers

      fsolve -- n-dimenstional root-finding

      brentq, brenth, ridder, bisect, newton -- one-dimensional root-finding

      fixed_point -- scalar fixed-point finder
      
    Notes
    
    ----------------------------------

      Optimize the function, f, whose gradient is given by fprime using the
      quasi-Newton method of Broyden, Fletcher, Goldfarb, and Shanno (BFGS)
      See Wright, and Nocedal 'Numerical Optimization', 1999, pg. 198.
      """
    import numpy
    import scipy.optimize.linesearch as linesearch

    x0 = asarray(x0).squeeze()
    if x0.ndim == 0:
        x0.shape = (1,)
    if maxiter is None:
        maxiter = len(x0)*200
    func_calls, f = wrap_function(f, args)
    if fprime is None:
        grad_calls, myfprime = wrap_function(approx_fprime, (f, epsilon))
    else:
        grad_calls, myfprime = wrap_function(fprime, args)
    gfk = myfprime(x0)
    k = 0
    N = len(x0)
    I = numpy.eye(N,dtype=int)
    Hk = I
    old_fval = f(x0)
    old_old_fval = old_fval + 5000
    xk = x0
    if retall:
        allvecs = [x0]
    sk = [2*gtol]
    warnflag = 0
    gnorm = vecnorm(gfk,ord=norm)
    while (gnorm > gtol) and (k < maxiter):
        pk = -numpy.dot(Hk,gfk)
        if False:
            alpha_k, fc, gc, old_fval, old_old_fval, gfkp1 = \
               linesearch.line_search(f,myfprime,xk,pk,gfk,
                                      old_fval,old_old_fval)
            if alpha_k is None:  # line search failed try different one.
                alpha_k, fc, gc, old_fval, old_old_fval, gfkp1 = \
                         line_search(f,myfprime,xk,pk,gfk,
                                     old_fval,old_old_fval)
                if alpha_k is None:

                    # This line search also failed to find a better solution.
                    warnflag = 2
                    break
        else:
            alpha_k = 0.1
        lg.debug("alpha = {0}".format(alpha_k))
        xkp1 = xk + alpha_k * pk #0.3 added by hcm
        print "--------------------------------\npk =", pk
        dump_mat(Hk)
        if retall:
            allvecs.append(xkp1)
        sk = xkp1 - xk
        xk = xkp1
        #if gfkp1 is None:
        gfkp1 = myfprime(xkp1)

        yk = gfkp1 - gfk
        gfk = gfkp1
        if callback is not None:
            #callback(xk)
            xk = callback(xk) # changed to the following line by hcm
        k += 1
        gnorm = vecnorm(gfk,ord=norm)
        if (gnorm <= gtol):
            break

        try: # this was handled in numeric, let it remaines for more safety
            rhok = 1.0 / (numpy.dot(yk,sk))
        except ZeroDivisionError: 
            rhok = 1000.0
            lg.debug("Divide-by-zero encountered: rhok assumed large")
        if isinf(rhok): # this is patch for numpy
            rhok = 1000.0
            lg.debug("Divide-by-zero encountered: rhok assumed large")
        A1 = I - sk[:,numpy.newaxis] * yk[numpy.newaxis,:] * rhok
        A2 = I - yk[:,numpy.newaxis] * sk[numpy.newaxis,:] * rhok
        Hk = numpy.dot(A1,numpy.dot(Hk,A2)) + rhok * sk[:,numpy.newaxis] \
                 * sk[numpy.newaxis,:]

    if disp or full_output:
        fval = old_fval
    if warnflag == 2:
        if disp:
            print "Warning: Desired error not necessarily achieved due to precision loss"
            print "         Current function value: %f" % fval
            print "         Iterations: %d" % k
            print "         Function evaluations: %d" % func_calls[0]
            print "         Gradient evaluations: %d" % grad_calls[0]

    elif k >= maxiter:
        warnflag = 1
        if disp:
            print "Warning: Maximum number of iterations has been exceeded"
            print "         Current function value: %f" % fval
            print "         Iterations: %d" % k
            print "         Function evaluations: %d" % func_calls[0]
            print "         Gradient evaluations: %d" % grad_calls[0]
    else:
        if disp:
            print "Optimization terminated successfully."
            print "         Current function value: %f" % fval
            print "         Iterations: %d" % k
            print "         Function evaluations: %d" % func_calls[0]
            print "         Gradient evaluations: %d" % grad_calls[0]

    if full_output:
        retlist = xk, fval, gfk, Hk, func_calls[0], grad_calls[0], warnflag
        if retall:
            retlist += (allvecs,)
    else:
        retlist = xk
        if retall:
            retlist = (xk, allvecs)

    return retlist

def test_my_bfgs():
    f = lambda v: v[0]**2*(v[0]-3) + v[1]**2
    x0 = array([100,1])
    fprime = lambda v: array([3*v[0]**2-6*v[0], 2*v[1]])

    from scipy.optimize import fmin_bfgs
    x = my_bfgs(f,x0, fprime)
    print x


def vector_interpolate(start, end, beads_count):
    """start: start vector
    end: end vector
    points: TOTAL number of points in path, INCLUDING start and final point"""

    assert len(start) == len(end)
    assert type(end) == ndarray
    assert type(start) == ndarray
    assert beads_count > 2

    # do I still need these lines?
    start = array(start)
    end = array(end)

    inc = (end - start) / (beads_count - 1)
    output = [ start + x * inc for x in range(beads_count) ]

    return array(output)


# delete after setup of auto tests
reactants = array([0,0])
products = array([3,3])

def test_path_rep():
    ts = array((2.5, 1.9))
    ts2 = array((1.9, 2.5))
    r = array((reactants, ts, ts2, products))
    my_rho = lambda x: 30*(x*x*(x-1)*(x-1))
    def my_rho1(x):
        if x < 0.5:
            return 4*x
        else:
            return -4*x + 4

    print "r",r
    x = PathRepresentation(r, 5)

    # Build linear, quadratic or spline representation of the path,
    # depending on the number of points.
    x.regen_path_func()
    x.beads_count = 20
    x.generate_beads(update=True)
    print "tangents =", x.path_tangents

    plot2D(x)

def test_GrowingString():
    from scipy.optimize import fmin_bfgs
    f_test = lambda x: True
    rho_quartic = lambda x: (x*(x-1))**2
    rho_flat = lambda x: 1
    surf_plot = SurfPlot(GaussianPES())
    qc_driver = GaussianPES()

    gs = GrowingString([reactants, products], f_test, qc_driver, 
        beads_count=15, rho=rho_flat)

    # Wrapper callback function
    def mycb(x):
        gs.update_path(x)
        gs.respace()
#        surf_plot.plot(x)
        gs.plot()
        return gs.get_state_vec()

    from scipy.optimize.lbfgsb import fmin_l_bfgs_b
    from scipy.optimize import fmin_cg

    while True:
        # (opt, a, b) = fmin_l_bfgs_b(gs.obj_func, gs.get_state_vec(), fprime = gs.obj_func_grad) 
        #opt = fmin_bfgs(gs.obj_func, gs.get_state_vec(), fprime = gs.obj_func_grad, callback=mycb, gtol=0.05, norm=Inf) 
        # opt = my_fmin_bfgs(gs.obj_func, gs.get_state_vec(), fprime = gs.obj_func_grad, callback=mycb, gtol=0.05, norm=Inf) 

        raw_input("test...\n")
        # opt = gd(gs.obj_func, gs.get_state_vec(), fprime = gs.obj_func_grad, callback = mycb) 
        #opt = my_bfgs(gs.obj_func, gs.get_state_vec(), fprime = gs.obj_func_grad, callback = mycb) 
        opt = my_runge_kutta(gs.obj_func, gs.get_state_vec(), fprime = gs.obj_func_grad, callback = mycb) 
        if not gs.grow_string():
            break

    gs.plot()
    surf_plot.plot(opt)


def my_runge_kutta(f, x0, fprime, callback, gtol=0.05):
    max_step = 0.3

    x = x0
    while True:
        g = fprime(x)
        if linalg.norm(g) < gtol:
            return x

        dt = 0.2
        ki1 = dt * g
        ki2 = dt * fprime(x + 0.5 * ki1)
        ki3 = dt * fprime(x + 0.5 * ki2)
        ki4 = dt * fprime(x + 0.5 * ki3)

        step =  -(1./6.) * ki1 - (1./3.) * ki2 - (1./3.) * ki3 - (1./6.) * ki4

        if linalg.norm(step, ord=inf) > max_step:
            step = max_step * step / linalg.norm(step, ord=inf)

        x = x + step

        if callback != None:
            x = callback(x)

def dump_diffs(pref, list):
    prev = 0
    for p in list:
        print "%s = %f" % (pref, (p - prev))
        prev = p
    print

def test_NEB():
    from scipy.optimize import fmin_bfgs

    default_spr_const = 1.
    neb = NEB([reactants, products], lambda x: True, GaussianPES(), default_spr_const, beads_count = 12)
    init_state = neb.get_state_as_array()

    surf_plot = SurfPlot(GaussianPES())

    # Wrapper callback function
    def mycb(x):
        #surf_plot.plot(path = x)
        print neb
        return x

    from scipy.optimize.lbfgsb import fmin_l_bfgs_b

#    opt = fmin_bfgs(neb.obj_func, init_state, fprime=neb.obj_func_grad, callback=mycb, gtol=0.05)
#    opt, energy, dict = fmin_l_bfgs_b(neb.obj_func, init_state, fprime=neb.obj_func_grad, callback=mycb, pgtol=0.05)
#    opt = opt_gd(neb.obj_func, init_state, neb.obj_func_grad, callback=mycb)

    import ase
    optimizer = ase.LBFGS(neb)
    optimizer.run(fmax=0.04)
    opt = neb.state_vec

    print "opt =", opt
    print dict
    #wt()

    gr = neb.obj_func_grad(opt)
    n = linalg.norm(gr)
    i = 0
    """while n > 0.001 and i < 4:
        print "n =",n
        opt = fmin_bfgs(neb.obj_func, opt, fprime=neb.obj_func_grad)
        gr = neb.obj_func_grad(opt)
        n = linalg.norm(gr)
        i += 1"""

    # Points on grid to draw PES
    ps = 20.0
    xrange = arange(ps)*(5.0/ps) - 1
    yrange = arange(ps)*(5.0/ps) - 1

    # Make a 2-d array containing a function of x and y.  First create
    # xm and ym which contain the x and y values in a matrix form that
    # can be `broadcast' into a matrix of the appropriate shape:
    gpes = GaussianPES()
    g = Gnuplot.Gnuplot(debug=1)
    g('set data style lines')
    g('set hidden')
    g.xlabel('Molecular Coordinate A')
    g.ylabel('Molecular Coordinate B')
    g.zlabel('Energy')

    g('set linestyle lw 5')
    # Get some tmp filenames
    (fd, tmpPESDataFile,) = tempfile.mkstemp(text=1)
    (fd, tmpPathDataFile,) = tempfile.mkstemp(text=1)
    Gnuplot.funcutils.compute_GridData(xrange, yrange, 
        lambda x,y: gpes.energy([x,y]), filename=tmpPESDataFile, binary=0)
    opt.shape = (-1,2)
    print "opt = ", opt
    pathEnergies = array (map (gpes.energy, opt.tolist()))
    print "pathEnergies = ", pathEnergies
    pathEnergies += 0.05
    xs = array(opt[:,0])
    ys = array(opt[:,1])
    print "xs =",xs, "ys =",ys
    data = transpose((xs, ys, pathEnergies))
    Gnuplot.Data(data, filename=tmpPathDataFile, inline=0, binary=0)

    # PLOT SURFACE AND PATH
    g('set xrange [-0.2:3.2]')
    g('set yrange [-0.2:3.2]')
    g('set zrange [-1:2]')
    print tmpPESDataFile,tmpPathDataFile
    g.splot(Gnuplot.File(tmpPESDataFile, binary=0), 
        Gnuplot.File(tmpPathDataFile, binary=0, with_="lines"))
    raw_input('Press to continue...\n')

    os.unlink(tmpPESDataFile)
    os.unlink(tmpPathDataFile)

    return opt


