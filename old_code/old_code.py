#!/usr/bin/python

    def simple_int(self, x0, dx_on_dt, trust_rad, verbose=False):
        
        x = array([x0]).flatten()
        d = zeros(len(x))
        k=0
        while True:
            k += 1
            prev_d = d
            d = dx_on_dt(x)
            x += 0.3 * 1.0/linalg.norm(d) * d
            print "x =", x
            if linalg.norm(x - x0) > trust_rad:
                print "trust rad"
                break
            print "d,dprev=", d, prev_d
            if dot(d, prev_d) < 0:
                print "direc change"
                break
            if k > 10:
                print "max iters"
                break
        return x, (x-x0)


def my_bfgs(f, x0, fprime, callback = None):
    from scipy.optimize import line_search
    max_step_size = 0.3
    i = 0
    dims = len(x0)
    I = eye(dims)
    x = x0

    def get_new_H_inv(sk, yk, H_inv_k):
        p = 1 / dot(yk, sk)
        tmp = yk*sk
        I_minus_psy = eye(dims) - p * outer(sk, yk)
        I_minus_pys = eye(dims) - p * outer(yk, sk)
        pss = p * outer(s, s)

        H_inv_new = dot (I_minus_psy, dot(H_inv_k, I_minus_pys)) + pss
        return H_inv_new

    def get_new_H_inv2(sk, yk, Hk):
        rhok = 1.0 / (dot(yk,sk))
        A1 = I - sk[:,newaxis] * yk[newaxis,:] * rhok
        A2 = I - yk[:,newaxis] * sk[newaxis,:] * rhok
        Hk = dot(A1,dot(Hk,A2)) + rhok * sk[:,newaxis] \
                 * sk[newaxis,:]
        return Hk

    H_inv = eye(dims) * 5
    H_inv2 = eye(dims) * 0.1
    g = fprime(x)
    energy = f(x)
    k=0
    while True:
        p = -dot(H_inv, g)
        alpha = 1 # from line search eventually
        step = alpha * p

        step_size = linalg.norm(step, ord=Inf)
        if step_size > max_step_size:
            print "scaling step"
            step = step * max_step_size / step_size
        
        print "step =", step
        print "p =", p
        print "x =", x
        print "g =", g

        x_old = x
        x = x + step
        if callback != None:
            print "x_before =", x
            x = callback(x)
            print "x_after =", x
        g_old = g
        g = fprime(x)

        energy = f(x)

        if linalg.norm(g, ord=2) < 0.01:
            print k, " iterations"
            return x
            break
        s = x - x_old
        y = g - g_old
        H_inv = get_new_H_inv(s, y, copy.deepcopy(H_inv))
        H_inv2 = get_new_H_inv2(s, y, copy.deepcopy(H_inv))
        print "H_inv", H_inv
#        print "H_inv2", H_inv2
        k += 1
#        print "H_inv =", H_inv.flatten(), "s =", s, "y =", y


def my_bfgs_bad(f, x0, fprime, callback = lambda x: Nothing):
    from scipy.optimize import line_search
    i = 0
    dims = len(x0)
    x = x0

    def get_new_H_inv(sk, yk, H_inv_k):
        A = outer(sk,sk) * (dot(sk,yk) + dot(yk, dot(H_inv_k, yk)))
        B = dot(H_inv_k, outer(yk, sk)) + dot(outer(sk, yk), H_inv_k)
        C = dot(sk, yk)

        H_inv_new = H_inv_k + A/C/C - B/C
        return H_inv_new
    def get_new_H_inv_old(sk, yk, H_inv_k):
        p = 1 / dot(yk, sk)
        I_minus_psy = eye(dim) - p * outer(sk, yk)
        I_minus_pys = eye(dim) - p * outer(yk, sk)
        pss = p * outer(s, s)

        H_inv_new = dot (I_minus_psy, dot(H_inv_k, I_minus_pys)) + pss
        return H_inv_new

    H_inv = eye(dims)
    g = fprime(x)
    energy = f(x)
    while True:
        s = dot(H_inv, -g)
        #res = line_search(f, fprime, x, -1*g, g, energy, copy.deepcopy(energy))
#        print "res =", res
#        print "x =", x,
#        print "g =", g
#        alpha, d1, d2, d3, d4 = res
        alpha = 1
        x = x + alpha * s
        x = callback(x)
        g_old = g
        g = fprime(x)

        energy = f(x)

        if linalg.norm(g, ord=inf) < 0.001:
            break
        y = g - g_old
        H_inv = get_new_H_inv(s, y, H_inv)
        print "H_inv =", H_inv.shape, "s =", s, "y =", y




# parabolas
# (x**2 + y**2)*((x-40)**2 + (y-4)**2)

# gaussians
# f(x,y) = -exp(-(x**2 + y**2)) - exp(-((x-3)**2 + (y-3)**2))
# df/dx = 2*x*exp(-(x**2 + y**2)) + (2*x - 6)*exp(-((x-3)**2 + (y-3)**2))
# df/dy = 2*y*exp(-(x**2 + y**2)) + (2*y - 6)*exp(-((x-3)**2 + (y-3)**2))

def e_test(v):
    x = v[0]
    y = v[1]
    return (-exp(-(x**2 + y**2)) - exp(-((x-3)**2 + (y-3)**2)) + 0.01*(x**2+y**2))

def g_test(v):
    x = v[0]
    y = v[1]
    dfdx = 2*x*exp(-(x**2 + y**2)) + (2*x - 6)*exp(-((x-3)**2 + (y-3)**2)) + 0.02*x
    dfdy = 2*y*exp(-(x**2 + y**2)) + (2*y - 6)*exp(-((x-3)**2 + (y-3)**2)) + 0.02*y
    return array((dfdx,dfdy))

def rosen_der(x):
    xm = x[1:-1]
    xm_m1 = x[:-2]
    xm_p1 = x[2:]
    der = zeros_like(x)
    der[1:-1] = 200*(xm-xm_m1**2) - 400*(xm_p1 - xm**2)*xm - 2*(1-xm)
    der[0] = -400*x[0]*(x[1]-x[0]**2) - 2*(1-x[0])
    der[-1] = 200*(x[-1]-x[-2]**2)
    return der

def rosen(x):
    """The Rosenbrock function"""
    return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)

