import numpy

class NumDiff():
    """Some functions to numerically differential vector functions."""
    def __init__(self, simpleerr=1e-5, init_h=0.1, method="numrec"):
        """
        method is either
            numrec: modified method from Numerical Recipes
            simple: method without strict error controls or reporting

        simpleerr is maximum error to tolerate when using simple method
        init_h is the initial h (i.e. finite difference) to use when using the
            numrec method"""
        self.max_err = simpleerr
        assert init_h > 0
        self.init_h = init_h
        if method == "numrec":
            self.numdiff = self.numdiff_numrec
        elif method == "simple":
            self.numdiff = self.numdiff_simple
        else:
            raise Exception("Unsupported differentiation method " + str(method))

    def vsin(self, x):
        return numpy.array([numpy.sin(x[0])])
    def vsincos(self, x):
        return numpy.array([numpy.sin(x[0]), numpy.cos(x[1])])

    def numdiff_numrec(self, f, x):
        """Vector version of numerical differentiation code given in Numerical 
        Recipes, 3rd Ed, Press, Teukolsky, Vetterling, Flannery, Cambridge 
        University Press.
        
        f is function to differentiate
        x is point about which to differentiate"""

        from numpy import zeros, float64, finfo, array
        from numpy.linalg import norm
        from copy import copy, deepcopy

        output_dims = len(f(x))
        input_dims = len(x)
        h = self.init_h
        ntab = 10
        con = 1.4
        con2 = con * con

        big = finfo(float64).max
        safe = 2.0

        a = zeros(ntab * ntab * output_dims)
        a.shape = (ntab, ntab, output_dims)

        hh = h

        jacobian = []
        errors = []
        for d in range(input_dims):
            x1 = deepcopy(x)
            x2 = deepcopy(x)
            x1[d] += hh
            x2[d] -= hh
            f1 = f(x1)
            f2 = f(x2)

            a[0][0] = (f1 - f2) / (2.0 * hh)
            err=big
            for i in range(1, ntab):
                hh /= con

                x1 = deepcopy(x)
                x2 = deepcopy(x)
                x1[d] += hh
                x2[d] -= hh
                f1 = f(x1)
                f2 = f(x2)

                a[0][i] = (f1 - f2) / (2.0 * hh)
                fac=con2
                for j in range(1,i+1):
                    a[j][i] = (a[j-1][i]*fac - a[j-1][i-1])/(fac-1.0)
                    fac = con2 * fac
                    errt = max(norm(a[j][i] - a[j-1][i]), norm(a[j][i] - a[j-1][i-1]))
    
                    if (errt <= err):
                        err = errt
                        ans = a[j][i]
                if (norm(a[i][i] - a[i-1][i-1]) >= safe * err):
                    break

            jacobian.append(deepcopy(ans))
            errors.append(err)


        return array(jacobian), array(errors)




    def numdiff_simple(self, f, X):
        """For function f, computes f'(X) numerically based on a finite difference approach."""

        # make sure we don't have ints as these will stop things from working
        X = numpy.array(X)

        N = len(X)
        df_on_dX = []

        def update_estim(dx, ix):
            from copy import deepcopy
            X1 = deepcopy(X)
            X2 = deepcopy(X)
            X1[ix] += dx
            X2[ix] -= dx
            f1, f2 = f(X1), f(X2)
            estim = (f1 - f2) / (2*dx)
            return estim

        it = 0
        for i in range(N):
            dx = self.init_h
            df_on_dx = update_estim(dx, i)
            while True:
                prev_df_on_dx = df_on_dx

                dx /= 2.0
                df_on_dx = update_estim(dx, i)
                #norm = numpy.linalg.norm(prev_df_on_dx)

                err = self.calc_err(df_on_dx, prev_df_on_dx)
                it += 1
                if err/dx < self.max_err:
                    break

            df_on_dX.append(df_on_dx)

        print "Average iterations per variable", it*1.0/N
        return numpy.array(df_on_dX), numpy.array([err])

    def calc_err(self, estim1, estim2):
        #max_err = numpy.finfo(numpy.float64).max
        diff = estim1 - estim2
        return numpy.linalg.norm(diff, ord=numpy.inf)


