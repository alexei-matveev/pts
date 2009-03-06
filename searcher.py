#!/usr/bin/python

from numpy import *
from scipy import *
import Gnuplot, Gnuplot.PlotItems, Gnuplot.funcutils
import tempfile, os

print "\n\nBegin Program..." 


class QCDriver:
    def __init__(self, dimension):
        self.dimension = dimension

    def gradient(self, a):
        return (-1)

    def energy(self, a):
        return (-1)

"""def g(a):
    x = a[0]
    y = a[1]
    dzdx = 4*x**3 - 3*80*x**2 + 2*1616*x + 2*2*x*y**2 - 2*8*y*x - 80*y**2 
    dzdy = 2*2*x**2*y - 8*x**2 - 2*80*x*y + 2*1616*y + 4*y**3 - 3*8*y**2
    return array([dzdy, dzdx])

def e(a):
    x = a[0]
    y = a[1]
    z = (x**2 + y**2) * ((x - 40)**2 + (y - 4) ** 2)
    return (z)
"""

class GaussianPES(QCDriver):
    def __init__(self):
        QCDriver.__init__(self,2)

    def energy(self, v):
        x = v[0]
        y = v[1]
        return (-exp(-(x**2 + y**2)) - exp(-((x-3)**2 + (y-3)**2)) + 0.01*(x**2+y**2) - 0.3*exp(-((x-1)**2 + (y-2)**2)))

    def gradient(self, v):
        x = v[0]
        y = v[1]
        dfdx = 2*x*exp(-(x**2 + y**2)) + (2*x - 6)*exp(-((x-3)**2 + (y-3)**2)) + 0.02*x + 0.3*(2*x-2)*exp(-((x-1)**2 + (y-2)**2))
        dfdy = 2*y*exp(-(x**2 + y**2)) + (2*y - 6)*exp(-((x-3)**2 + (y-3)**2)) + 0.02*y + 0.3*(2*y-4)*exp(-((x-1)**2 + (y-2)**2))

        return array((dfdx,dfdy))

class GaussianPES2(QCDriver):
    def __init__(self):
        QCDriver.__init__(self,2)

    def energy(self, v):
        x = v[0]
        y = v[1]
        return (-exp(-(x**2 + 0.2*y**2)) - exp(-((x-3)**2 + (y-3)**2)) + 0.01*(x**2+y**2) - 0.5*exp(-((x-1.5)**2 + (y-2.5)**2)))

    def gradient(self, v):
        x = v[0]
        y = v[1]
        dfdx = 2*x*exp(-(x**2 + 0.2*y**2)) + (2*x - 6)*exp(-((x-3)**2 + (y-3)**2)) + 0.02*x + 0.5*(2*x-3)*exp(-((x-1.5)**2 + (y-2.5)**2))
        dfdy = 2*y*exp(-(x**2 + 0.2*y**2)) + (2*y - 6)*exp(-((x-3)**2 + (y-3)**2)) + 0.02*y + 0.3*(2*y-5)*exp(-((x-1.5)**2 + (y-2.5)**2))

        return array((dfdx,dfdy))

class QuarticPES(QCDriver):
    def __init__(self):
        QCDriver.__init__(self,2)

    def gradient(self, a):
        if len(a) != self.dimension:
            raise Exception("Wrong dimension")

        x = a[0]
        y = a[1]
        dzdx = 4*x**3 - 3*80*x**2 + 2*1616*x + 2*2*x*y**2 - 2*8*y*x - 80*y**2 
        dzdy = 2*2*x**2*y - 8*x**2 - 2*80*x*y + 2*1616*y + 4*y**3 - 3*8*y**2
        return array([dzdy, dzdx])

    def energy(self, a):
        if len(a) != self.dimension:
            raise Exception("Wrong dimension")

        x = a[0]
        y = a[1]
        z = (x**2 + y**2) * ((x - 40)**2 + (y - 4) ** 2)
        return (z)

class ReactionPathway:
    dimension = -1
    def __init__(self, reactants, products, f_test = lambda x: True, beadsCount = 10):
        assert type(reactants) == type(products) == ndarray

        if beadsCount <= 2:
            raise Exception("Must have beadsCount > 2 to form a meaningful path")

        self.reactants  = reactants
        self.products   = products
        self.beadsCount = beadsCount
        self.stateVec   = vectorInterpolate(reactants, products, beadsCount)
        
        assert len(reactants) == len(products)
        self.dimension = len(reactants) # dimension of PES
        
        # test to see if molecular geometry is bad or not
        pointsGood = map (f_test, self.stateVec.tolist())
        if not reduce(lambda a,b: a and b, pointsGood):
            raise Exception("Unhandled, some points were bad")

    def objFunc():
        pass

    def objFuncGrad():
        pass

    def dump():
        print "pathdata", pathdata


def specialReduceXX(list, ks = [], f1 = lambda a,b: a-b, f2 = lambda a: a**2):
    """For a list of x_0, x_1, ... , x_(N-1)) and a list of scalars k_0, k_1, ..., 
    returns a list of length N-1 where each element of the output array is 
    f2(f1(k_i * x_i, k_i+1 * x_i+1)) ."""

    assert type(list) == ndarray
    assert len(list) >= 2
    assert len(ks) == 0 or len(ks) == len(list)
    
    # Fill with trivial value that won't change the result of computations
    if len(ks) == 0:
        ks = array(ones(len(list)))

    def specialReduce_(head, head1, tail, f1, f2, k, k1, ktail):
        reduction = f2 (f1 (k*head, k1*head1))
        if len(tail) == 0:
            return [reduction]
        else:
            return [reduction] + specialReduce_(head1, tail[0], tail[1:], f1, f2, k1, ktail[0], ktail[1:])

    return array(specialReduce_(list[0], list[1], list[2:], f1, f2, ks[0], ks[1], ks[2:]))

class NEB_l(ReactionPathway):
    def __init__(self, reactants, products, f_test, baseSprConst, qcDriver, beadsCount = 10):
        ReactionPathway.__init__(self, reactants, products, f_test, beadsCount)
        self.baseSprConst = baseSprConst
        self.qcDriver = qcDriver
        self.tangents = zeros(beadsCount * self.dimension)
        self.tangents.shape = (beadsCount, self.dimension)

        # Make list of spring constants for every inter-bead separation
        # For the time being, these are uniform
        self.sprConstVec = array([self.baseSprConst for x in range(beadsCount - 1)])

class PathRepresentation():
    def __init__(self, state_vec, beads_count, f_spacing_density = lambda x: 1):
        self.state_vec = state_vec
        self.control_pts_cnt = len(state_vec)
        self.dimensions = len(state_vec[0])

        # TODO check all beads have same dimensionality

        self.f_spacing_density = f_spacing_density
        int = integrate_f_spacing_density(f_spacing_density)
        if int != 1
            raise Exception("bad spacing function")


    def regen_path_func(self):
        """Rebuild a new path function and the derivative of the path based on the contents of state_vec."""
        assert len(state_vec) > 1

        for i in range(self.dimensions):

            ys = state_vec[:,i]
            if len(state_vec) == 2:
                # linear path
                self.fs[i] = interpolate.interp1d(unit_interval, ys)
                self.fprimes[i] = ys[1] - ys[0]

            elif len(state_vec) == 3:
                # parabolic path

                ps = array((0.0, 0.5, 1.0))
                for i in range(self.dimensions)
                    ps_x_pow_2 = ps**2
                    ps_x_pow_1 = ps
                    ps_x_pow_0 = 1

                    A = column_stack((ps_x_pow_2, ps_x_pow_1, ps_x_pow_0))

                    quadratic_coeffs = linalg.solve(A,y)

                    self.fs[i] = lambda x: dot(array((x**2, x, 1)), quadratic_coeffs)
                    self.fprimes[i] = lambda x: 2 * quadratic_coeffs[0] * x + quadratic_coeffs[1]

            else
                # spline path
                # TODO rewrite for non-unity density function?
                ps = arange(0.0, 1.0 + 1.0 / len(state_vec), 1.0 / len(state_vec))
                spline_data = interpolate.splrep(ps, y, s=0)

                self.fs[i] = lambda x: interpolate.splev(x, spline_data, der=0)
                self.fprimes[i] = lambda x: interpolate.splev(x, spline_data, der=1)


    def generate_beads():
        """Returns an array of the vectors of the coordinates of beads along a reaction path,
        according to the established path (line, parabola or spline) and the parameterisation
        density"""

        (total_str_len, inremental_positions) = get_total_str_len()

        normd_positions = generate_normd_positions(total_str_len, incremental_positions)

        bead_vectors = []
        for str_pos in str_positions:
            bead_vectors.append(get_bead_coords(str_pos))

        return bead_vectors
        
    def get_str_positions(self):
        param_steps = arange(0, 1, step)
        integrated_density_inc = 1.0 / (beds_count + 1.0)
        integral = 0
        requirement_for_next_bead = integrated_density_inc

        for s in param_steps:
            integral += step * rho(s)
            if integral > requirement_for_next_bead:
                str_positions.append(s)
                requirement_for_next_bead += integrated_density_inc
        
        return str_positions



    def generate_normd_positions(self, total_str_len, incremental_positions):
        "Returns a list of normalised distances based on desired distances along string"""

        str_positions = get_str_positions()

        normd_positions = []

        for desired_str_pos in str_positions:
            for (norm, str) in incremental_positions:
                if str >= desired_str_pos:
                    normd_positions.append(norm)
                    break

        return normd_positions

    def get_total_str_len(self):
        """Returns the a duple of the total length of the string and a list of 
        pairs (x,y), where x a distance along the normalised path (i.e. on 
        [0,1]) and y is the corresponding distance along the string."""
        
        def arc_dist_func(self, x):
            output = 0
            for a in self.fprimes:
                output += a(x)**2
            return sqrt(output)

        # number of points to chop the string into
        str_resolution = 200
        step = 1.0 / str_resolution
        param_steps = arange(0, 1, step)

        list = []
        for i in range(str_resolution):
            lower, upper = i * step, (i + 1) * step
            integral = integrate.quad(arc_dist_func, lower, upper)
            list.append(integral)

        return (sum(list), zip(param_steps, list))

    def tangents():
        pass

class GrowingString(ReactionPathway):
    def __init__(self, reactants, products, f_test, f_density, qcDriver, beadsCount = 10):
        ReactionPathway.__init__(self, reactants, products, f_test, beadsCount)
        self.baseSprConst = baseSprConst
        self.qcDriver = qcDriver

        self.path_rep = PathRepresentation([reactants, products])

    def step_opt():
        


class NEB(ReactionPathway):
    def __init__(self, reactants, products, f_test, baseSprConst, qcDriver, beadsCount = 10):
        ReactionPathway.__init__(self, reactants, products, f_test, beadsCount)
        self.baseSprConst = baseSprConst
        self.qcDriver = qcDriver
        self.tangents = zeros(beadsCount * self.dimension)
        self.tangents.shape = (beadsCount, self.dimension)

        # Make list of spring constants for every inter-bead separation
        # For the time being, these are uniform
        self.sprConstVec = array([self.baseSprConst for x in range(beadsCount - 1)])

    def specialReduce(self, list, ks = [], f1 = lambda a,b: a-b, f2 = lambda a: a**2):
        """For a list of x_0, x_1, ... , x_(N-1)) and a list of scalars k_0, k_1, ..., 
        returns a list of length N-1 where each element of the output array is 
        f2(f1(k_i * x_i, k_i+1 * x_i+1)) ."""

#       print type(self.stateVec)
#       print type(list)
        assert type(list) == ndarray
        assert len(list) >= 2
        assert len(ks) == 0 or len(ks) == len(list)
        
        # Fill with trivial value that won't change the result of computations
        if len(ks) == 0:
            ks = array(ones(len(list)))

        assert type(ks) == ndarray
        for a in range(len(ks)):
            list[a] = list[a] * ks[a]

        print "list =",list
        currDim = list.shape[1]  # generate zero vector of the same dimension of the list of input dimensions
        print "cd = ", currDim
        z = array(zeros(currDim))
        listPos = vstack((list, z))
        listNeg = vstack((z, list))
#        print "listPos =",listPos
#        print "listNeg =",listNeg

        list = f1 (listPos, listNeg)
#        print "list2 =",list

        list = f2 (list[1:-1])
#        print "list =",list

        return list

    def updateTangents(self):
        # terminal beads have no tangent
        self.tangents[0]  = zeros(self.dimension)
        self.tangents[-1] = zeros(self.dimension)
        for i in range(self.beadsCount)[1:-1]:
            self.tangents[i] = ( (self.stateVec[i] - self.stateVec[i-1]) + (self.stateVec[i+1] - self.stateVec[i]) ) / 2
            self.tangents[i] /= linalg.norm(self.tangents[i], 2)

#        print "tangents =", self.tangents

    def updateBeadSeparations(self):
        self.beadSeparationSqrsSums = array( map (sum, self.specialReduce(self.stateVec).tolist()) )
        self.beadSeparationSqrsSums.shape = (self.beadsCount - 1, 1)
#        print "beadSeparations =", self.beadSeparationSqrsSums


    def getStateAsArray(self):
        return self.stateVec.flatten()

    def objFunc(self, newStateVec = []):
        assert size(self.stateVec) == self.beadsCount * self.dimension

        if newStateVec != []:
            self.stateVec = array(newStateVec)
            self.stateVec.shape = (self.beadsCount, self.dimension)

        self.updateTangents()
        self.updateBeadSeparations()
        
#        print "self.beadSeparationSqrsSums =", self.beadSeparationSqrsSums
        forceConstsBySeparationsSquared = multiply(self.sprConstVec, self.beadSeparationSqrsSums.flatten()).transpose()
#        print "forceConstsBySeparationsSquared =", forceConstsBySeparationsSquared
        springEnergies = 0.5 * ndarray.sum (forceConstsBySeparationsSquared)

        # The following code block will need to be replaced for parallel operation
        pesEnergies = 0
        for beadVec in self.stateVec[1:-1]:
            pesEnergies += self.qcDriver.energy(beadVec)

#        print "pesEnergies =", pesEnergies, "springEnergies =", springEnergies
        return (pesEnergies + springEnergies)

    def objFuncGrad(self, newStateVec = []):

        # If a state vector has been specified, return the value of the 
        # objective function for this new state and set the state of self
        # to the new state.
        if newStateVec != []:
            self.stateVec = array(newStateVec)
            self.stateVec.shape = (self.beadsCount, self.dimension)

        self.updateBeadSeparations()
        self.updateTangents()

        separationsVec = self.beadSeparationSqrsSums ** 0.5
#        print "sv =", separationsVec
        separationsDiffs = self.specialReduce(separationsVec, self.sprConstVec, f2 = lambda x: x)
        assert len(separationsDiffs) == self.beadsCount - 2

#        print "sd =", separationsDiffs.flatten(), "t =", self.tangents[1:-1]
        springForces = multiply(separationsDiffs.flatten(), self.tangents[1:-1].transpose()).transpose()
        springForces = vstack((zeros(self.dimension), springForces, zeros(self.dimension)))
        print "sf =", springForces

        pesForces = array(zeros(self.beadsCount * self.dimension))
        pesForces.shape = (self.beadsCount, self.dimension)
#        print "pesf =", pesForces

        for i in range(self.beadsCount)[1:-1]:
            pesForces[i] = -self.qcDriver.gradient(self.stateVec[i])
#            print "pesbefore =", pesForces[i]
            pesForces[i] = pesForces[i] - dot(pesForces[i], self.tangents[i]) * self.tangents[i]
#            print "pesafter =", pesForces[i], "t =", self.tangents[i]

        print "pesf =", pesForces
        gradientsVec = -1 * (pesForces + springForces)
        print "gradients =", gradientsVec

        return gradientsVec.flatten()


def vectorInterpolate(start, end, beadsCount):
    """start: start vector
    end: end vector
    points: TOTAL number of points in path, INCLUDING start and final point"""

    assert len(start) == len(end)
    assert type(end) == ndarray
    assert type(start) == ndarray
    assert beadsCount > 2

    start = array(start, dtype=float64)
    end = array(end, dtype=float64)

    inc = (end - start) / (beadsCount - 1)
    output = [ start + x * inc for x in range(beadsCount) ]

    return array(output)



reactants = array([0,0])
products = array([3,3])
if len(reactants) != len(products):
    print "Reactants/Products must be the same size"

print "Reactants vector size =", len(reactants), "Products vector size =", len(products)

defaultSprConst = 0.01
neb = NEB(reactants, products, lambda x: True, defaultSprConst, GaussianPES(), beadsCount = 15)

def main():
    from scipy.optimize import fmin_bfgs

    initState = neb.getStateAsArray()
    opt = fmin_bfgs(neb.objFunc, initState, fprime=neb.objFuncGrad)
    gr = neb.objFuncGrad(opt)
    n = linalg.norm(gr)
    i = 0
    while n > 0.001 and i < 4:
        print "n =",n
        opt = fmin_bfgs(neb.objFunc, opt, fprime=neb.objFuncGrad)
        gr = neb.objFuncGrad(opt)
        n = linalg.norm(gr)
        i += 1


    # Points on grid to draw PES
    ps = 20.0
    xrange = arange(ps)*(5.0/ps) - 1
    yrange = arange(ps)*(5.0/ps) - 1

    # Make a 2-d array containing a function of x and y.  First create
    # xm and ym which contain the x and y values in a matrix form that
    # can be `broadcast' into a matrix of the appropriate shape:
    gpes = GaussianPES2()
    g = Gnuplot.Gnuplot(debug=1)
    g('set data style lines')
    g('set hidden')
    g.xlabel('x')
    g.ylabel('y')

    # Get some tmp filenames
    (fd, tmpPESDataFile,) = tempfile.mkstemp(text=1)
    (fd, tmpPathDataFile,) = tempfile.mkstemp(text=1)
    Gnuplot.funcutils.compute_GridData(xrange, yrange, lambda x,y: gpes.energy([x,y]),filename=tmpPESDataFile, binary=0)
    opt.shape = (-1,2)
    print "opt = ", opt
    pathEnergies = array (map (gpes.energy, opt.tolist()))
    print "pathEnergies = ", pathEnergies
    pathEnergies += 0.02
    xs = array(opt[:,0])
    ys = array(opt[:,1])
    print "xs =",xs, "ys =",ys
    data = transpose((xs, ys, pathEnergies))
    Gnuplot.Data(data, filename=tmpPathDataFile, inline=0, binary=0)

    # PLOT SURFACE AND PATH
    g.splot(Gnuplot.File(tmpPESDataFile, binary=0), Gnuplot.File(tmpPathDataFile, binary=0, with_="linespoints"))
    raw_input('Press to continue...\n')

    os.unlink(tmpPESDataFile)
    os.unlink(tmpPathDataFile)

    return opt

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


