"""
Hacked version of SciPy's interface to the Fortran L-BFGS-B optimiser.

Modifications include:
    1. addition of a callback function, called as
       x = callback(x) where x is the current optimisation vector
    2. application of a maximum step size

"""

## Automatically adapted for scipy Oct 07, 2005 by convertcode.py


## License for the Python wrapper
## ==============================

## Copyright (c) 2004 David M. Cooke <cookedm@physics.mcmaster.ca>

## Permission is hereby granted, free of charge, to any person obtaining a copy of
## this software and associated documentation files (the "Software"), to deal in
## the Software without restriction, including without limitation the rights to
## use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
## of the Software, and to permit persons to whom the Software is furnished to do
## so, subject to the following conditions:

## The above copyright notice and this permission notice shall be included in all
## copies or substantial portions of the Software.

## THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
## IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
## FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
## AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
## LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
## OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
## SOFTWARE.

## Modifications by Travis Oliphant and Enthought, Inc.  for inclusion in SciPy

from numpy import zeros, float64, array, int32, dot, sqrt

import scipy.optimize._lbfgsb as _lbfgsb
import scipy.optimize as optimize

import scipy.linalg

approx_fprime = optimize.approx_fprime

def fmin_l_bfgs_b(func, x0, fprime=None, args=(),
                  approx_grad=0,
                  bounds=None, m=10, factr=1e7, pgtol=1e-5,
                  epsilon=1e-8,
                  iprint=-1, maxfun=15000,
                  callback = None, maxstep=0.2, scale_steps=True): # added for AOF
    """
    Minimize a function func using the L-BFGS-B algorithm.

    Arguments:

    func    -- function to minimize. Called as func(x, *args)

    x0      -- initial guess to minimum

    fprime  -- gradient of func. If None, then func returns the function
               value and the gradient ( f, g = func(x, *args) ), unless
               approx_grad is True then func returns only f.
               Called as fprime(x, *args)

    args    -- arguments to pass to function

    approx_grad -- if true, approximate the gradient numerically and func returns
                   only function value.

    bounds  -- a list of (min, max) pairs for each element in x, defining
               the bounds on that parameter. Use None for one of min or max
               when there is no bound in that direction

    m       -- the maximum number of variable metric corrections
               used to define the limited memory matrix. (the limited memory BFGS
               method does not store the full hessian but uses this many terms in an
               approximation to it).

    factr   -- The iteration stops when
               (f^k - f^{k+1})/max{|f^k|,|f^{k+1}|,1} <= factr*epsmch

               where epsmch is the machine precision, which is automatically
               generated by the code. Typical values for factr: 1e12 for
               low accuracy; 1e7 for moderate accuracy; 10.0 for extremely
               high accuracy.

    pgtol   -- The iteration will stop when
                  max{|proj g_i | i = 1, ..., n} <= pgtol
               where pg_i is the ith component of the projected gradient.

    epsilon -- step size used when approx_grad is true, for numerically
               calculating the gradient

    iprint  -- controls the frequency of output. <0 means no output.

    maxfun  -- maximum number of function evaluations.


    Returns:
    x, f, d = fmin_lbfgs_b(func, x0, ...)

    x -- position of the minimum
    f -- value of func at the minimum
    d -- dictionary of information from routine
        d['warnflag'] is
            0 if converged,
            1 if too many function evaluations,
            2 if stopped for another reason, given in d['task']
        d['grad'] is the gradient at the minimum (should be 0 ish)
        d['funcalls'] is the number of function calls made.


   License of L-BFGS-B (Fortran code)
   ==================================

   The version included here (in fortran code) is 2.1 (released in 1997). It was
   written by Ciyou Zhu, Richard Byrd, and Jorge Nocedal <nocedal@ece.nwu.edu>. It
   carries the following condition for use:

   This software is freely available, but we expect that all publications
   describing  work using this software , or all commercial products using it,
   quote at least one of the references given below.

   References
     * R. H. Byrd, P. Lu and J. Nocedal. A Limited Memory Algorithm for Bound
       Constrained Optimization, (1995), SIAM Journal on Scientific and
       Statistical Computing , 16, 5, pp. 1190-1208.
     * C. Zhu, R. H. Byrd and J. Nocedal. L-BFGS-B: Algorithm 778: L-BFGS-B,
       FORTRAN routines for large scale bound constrained optimization (1997),
       ACM Transactions on Mathematical Software, Vol 23, Num. 4, pp. 550 - 560.

    See also:
        scikits.openopt, which offers a unified syntax to call this and other solvers

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

    """
    n = len(x0)

    if bounds is None:
        bounds = [(None,None)] * n
    if len(bounds) != n:
        raise ValueError('length of x0 != length of bounds')

    if approx_grad:
        def func_and_grad(x):
            f = func(x, *args)
            g = approx_fprime(x, func, epsilon, *args)
            return f, g
    elif fprime is None:
        def func_and_grad(x):
            f, g = func(x, *args)
            return f, g
    else:
        def func_and_grad(x):
            f = func(x, *args)
            g = fprime(x, *args)
            return f, g

    nbd = zeros((n,), int32)
    low_bnd = zeros((n,), float64)
    upper_bnd = zeros((n,), float64)
    bounds_map = {(None, None): 0,
              (1, None) : 1,
              (1, 1) : 2,
              (None, 1) : 3}
    for i in range(0, n):
        l,u = bounds[i]
        if l is not None:
            low_bnd[i] = l
            l = 1
        if u is not None:
            upper_bnd[i] = u
            u = 1
        nbd[i] = bounds_map[l, u]

    x = array(x0, float64)
    f = array(0.0, float64)
    g = zeros((n,), float64)
    wa = zeros((2*m*n+4*n + 12*m**2 + 12*m,), float64)
    iwa = zeros((3*n,), int32)
    task = zeros(1, 'S60')
    csave = zeros(1,'S60')
    lsave = zeros((4,), int32)
    isave = zeros((44,), int32)
    dsave = zeros((29,), float64)

    task[:] = 'START'

    n_function_evals = 0
    while 1:
        prevx = x.copy()
        # see http://www.math.unm.edu/~vageli/courses/Ma579/lbfgsb.f
        _lbfgsb.setulb(m, x, low_bnd, upper_bnd, nbd, f, g, factr,
                       pgtol, wa, iwa, task, iprint, csave, lsave,
                       isave, dsave)

        # scale step size, added for AOF
        # A question that one might ask is: if the steps are always scaled, 
        # even during the line search, wont this cause problems if points on 
        # the line search are co-linear. The answer is: that subsequent steps 
        # in the line search, as generated by setulb(), never seem to be 
        # co-linear.
        if scale_steps:
            step = x - prevx
            size = sqrt(dot(step, step))
            #print "step size",size
            if size > maxstep:
                #print "*********** scaling step size"
                step = step / size * maxstep
                x = prevx + step

        task_str = task.tostring()
        #print "task_str",task_str
        if task_str.startswith('FG'):
            # minimization routine wants f and g at the current x
            n_function_evals += 1
            # Overwrite f and g:

            #print "step x", str(x)
            f, g = func_and_grad(x)
#            print "Line searching, current f =", f

        elif task_str.startswith('NEW_X'):

            # added for aOF
            print "g_max", g.max(), "pgtol", pgtol
            #print "n_function_evals",n_function_evals
            if callable(callback):
                callback(x)

            # new iteration
            if n_function_evals > maxfun:
                task[:] = 'STOP: TOTAL NO. of f AND g EVALUATIONS EXCEEDS LIMIT'
        else:
            break

    task_str = task.tostring().strip('\x00').strip()
    if task_str.startswith('CONV'):
        warnflag = 0
    elif n_function_evals > maxfun:
        warnflag = 1
    else:
        warnflag = 2


    d = {'grad' : g,
         'task' : task_str,
         'funcalls' : n_function_evals,
         'warnflag' : warnflag
        }
    return x, f, d

if __name__ == '__main__':
    def func(x):
        f = 0.25*(x[0]-1)**2
        for i in range(1, x.shape[0]):
            f += (x[i] - x[i-1]**2)**2
        f *= 4
        return f
    def grad(x):
        g = zeros(x.shape, float64)
        t1 = x[1] - x[0]**2
        g[0] = 2*(x[0]-1) - 16*x[0]*t1
        for i in range(1, g.shape[0]-1):
            t2 = t1
            t1 = x[i+1] - x[i]**2
            g[i] = 8*t2 - 16*x[i]*t1
        g[-1] = 8*t1
        return g

    factr = 1e7
    pgtol = 1e-5

    n=25
    m=10

    bounds = [(None,None)] * n
    for i in range(0, n, 2):
        bounds[i] = (1.0, 100)
    for i in range(1, n, 2):
        bounds[i] = (-100, 100)

    x0 = zeros((n,), float64)
    x0[:] = 3

    x, f, d = fmin_l_bfgs_b(func, x0, fprime=grad, m=m,
                            factr=factr, pgtol=pgtol)
    print x
    print f
    print d
    x, f, d = fmin_l_bfgs_b(func, x0, approx_grad=1,
                            m=m, factr=factr, pgtol=pgtol)
    print x
    print f
    print d


