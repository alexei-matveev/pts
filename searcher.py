#!/usr/bin/python

from numpy import *
from scipy import *
from scipy import interpolate
import Gnuplot, Gnuplot.PlotItems, Gnuplot.funcutils
import tempfile, os
import logging
import copy
import pickle

logger = logging.getLogger("searcher.py")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

import scipy.integrate

print "\n\nBegin Program..." 

# Function labeller
def flab(tag = "", tag1 = ""):
    import sys
#    import inspect
#    print inspect.stack(sys._getframe())
    print "**** ", sys._getframe(1).f_code.co_name, tag, tag1

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
        return (-exp(-(x**2 + y**2)) - exp(-((x-3)**2 + (y-3)**2)) + 0.01*(x**2+y**2) - 0.5*exp(-((1.5*x-1)**2 + (y-2)**2)))

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
    def __init__(self, reactants, products, f_test = lambda x: True):
        assert type(reactants) == type(products) == ndarray

        self.reactants  = reactants
        self.products   = products
        
        assert len(reactants) == len(products)
        self.dimension = len(reactants) # dimension of PES
        
    def obj_func():
        pass

    def obj_func_grad():
        pass

    def dump(self):
        pass


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
    def __init__(self, reactants, products, f_test, baseSprConst, qcDriver, beadsCount = 10, str_resolution = 100):
        ReactionPathway.__init__(self, reactants, products, f_test, beadsCount)
        self.baseSprConst = baseSprConst
        self.qcDriver = qcDriver
        self.tangents = zeros(beadsCount * self.dimension)
        self.tangents.shape = (beadsCount, self.dimension)

        # Make list of spring constants for every inter-bead separation
        # For the time being, these are uniform
        self.sprConstVec = array([self.baseSprConst for x in range(beadsCount - 1)])

class Func():
    def f():
        pass
    def fprime():
        pass

class LinFunc():
    def __init__(self, xs, ys):
        self.fs = scipy.interpolate.interp1d(xs, ys)
        self.grad = (ys[1] - ys[0]) / (xs[1] - xs[0])

    def f(self, x):
        return self.fs(x)[0]

    def fprime(self, x):
        return self.grad


class QuadFunc(Func):
    def __init__(self, coefficients):
        self.coefficients = coefficients

    def f(self, x):
        return dot(array((x**2, x, 1)), self.coefficients)

    def fprime(self, x):
        return 2 * self.coefficients[0] * x + self.coefficients[1]

class SplineFunc(Func):
    def __init__(self, xs, ys):
        self.spline_data = interpolate.splrep(xs, ys, s=0)
        
    def f(self, x):
        return interpolate.splev(x, self.spline_data, der=0)

    def fprime(self, x):
        return interpolate.splev(x, self.spline_data, der=1)

def dup2val(dup):
    (x,y) = dup
    return x

class PathRepresentation():
    """Supports operations on a path represented by a line, parabola, or a 
    spline, depending on whether it has 2, 3 or > 3 points."""

    def __init__(self, state_vec, beads_count, rho = lambda x: 1, str_resolution = 100):

        # vector of vectors defining the path
        if (isinstance(state_vec, ndarray)):
            self.__state_vec = state_vec
        else:
            self.__state_vec = array(state_vec)

        # number of vectors defining the path
        self.beads_count = beads_count
        self.__dimensions = len(state_vec[0])

        self.__str_resolution = str_resolution
        self.__step = 1.0 / self.__str_resolution

        self.__fs = []
        self.__path_tangents = []

        self.__unit_interval = array((0.0,1.0))

        # TODO check all beads have same dimensionality

        self.__max_integral_error = 1e-5

        self.__rho = self.set_rho(rho)

        msg = "beads_count = %d\nstr_resolution = %d" % (beads_count, str_resolution)
        print msg

    def get_fs(self):
        return self.__fs
    def get_path_tangents(self):
        return self.__path_tangents
    def set_state_vec(self, new_state_vec):
        self.__state_vec = array(new_state_vec).flatten()
        self.__state_vec.shape = (self.beads_count, -1)
    def get_state_vec(self):
        return self.__state_vec
    def get_dimensions(self):
        return self.__dimensions

    def regen_path_func(self):
        """Rebuild a new path function and the derivative of the path based on 
        the contents of state_vec."""
        assert len(self.__state_vec) > 1

        self.__fs = []

        #print "state_vec2 =", self.__state_vec
        #print "self.__dimensions =", self.__dimensions
        for i in range(self.__dimensions):

            ys = self.__state_vec[:,i]

            # linear path
            if len(self.__state_vec) == 2:
                self.__fs.append(LinFunc(self.__unit_interval, ys))

            # parabolic path
            elif len(self.__state_vec) == 3:

                # TODO: at present, transition state assumed to be half way ebtween reacts and prods
                ps = array((0.0, 0.5, 1.0))
                ps_x_pow_2 = ps**2
                ps_x_pow_1 = ps
                ps_x_pow_0 = ones(len(ps_x_pow_1))

                A = column_stack((ps_x_pow_2, ps_x_pow_1, ps_x_pow_0))

                quadratic_coeffs = linalg.solve(A,ys)

                self.__fs.append(QuadFunc(quadratic_coeffs))

            else:
                # spline path
                points_cnt = len(self.__state_vec)
                xs = arange(0.0, 1.0 + 1.0 / (points_cnt - 1), 1.0 / (points_cnt - 1))
#                print "points_cnt =", points_cnt
#                print "xs =", xs
                self.__fs.append(SplineFunc(xs,ys))

    def __arc_dist_func(self, x):
        output = 0
        for a in self.__fs:
            output += a.fprime(x)**2
        return sqrt(output)

    def __get_total_str_len_exact(self):

        (integral, error) = scipy.integrate.quad(self.__arc_dist_func, 0.0, 1.0)
        return integral


    def __get_total_str_len(self):
        """Returns the a duple of the total length of the string and a list of 
        pairs (x,y), where x a distance along the normalised path (i.e. on 
        [0,1]) and y is the corresponding distance along the string (i.e. on
        [0,string_len])."""
        
        # function, integral of which gives total path length
        def arc_dist_func(x):
            output = 0
            for a in self.__fs:
                output += a.fprime(x)**2
            return sqrt(output)

        # number of points to chop the string into
        param_steps = arange(0, 1, self.__step)

        list = []
        cumm_dist = 0
        for i in range(self.__str_resolution):
            lower, upper = i * self.__step, (i + 1) * self.__step
            (integral, error) = scipy.integrate.quad(arc_dist_func, lower, upper)
#            print "int = %lf, err = %lf, lower = %lf, upper = %lf" % (integral, 
#                error, lower, upper)
            cumm_dist += integral

            assert error < self.__max_integral_error

            list.append(cumm_dist)

        return (list[-1], zip(param_steps, list))

    def sub_str_lengths(self, normd_poses):
        """Finds the lengths of the pieces of the string specified in 
        normd_poses in terms of normalised coordinate."""

        my_normd_poses = array(normd_poses).flatten()

        from scipy.integrate import quad
        x0 = 0.0
        lengths = []
        for pos in my_normd_poses:
            (len,err) = quad(self.__arc_dist_func, x0, pos)
            lengths.append(len)
            x0 = pos

        lengths = array(lengths).flatten()
        print "sub_str_lengths: normd_poses:", normd_poses
        print "sub_str_lengths: sub lengths of string", lengths

    def generate_beads_exact(self, update = False):
        """Returns an array of the self.__beads_count vectors of the coordinates 
        of beads along a reaction path, according to the established path 
        (line, parabola or spline) and the parameterisation density."""

        flab("called", update)

        assert len(self.__fs) > 1

        # Find total string length and incremental distances x along the string 
        # in terms of the normalised coodinate y, as a list of (x,y).
        total_str_len = self.__get_total_str_len_exact()

        # For the desired distances along the string, find the values of the
        # normalised coordinate that achive those distances.
        normd_positions = self.__generate_normd_positions_exact(total_str_len)

        self.sub_str_lengths(normd_positions)

        bead_vectors = []
        bead_tangents = []
#        print "normd_positions =", normd_positions
        for str_pos in normd_positions:
            bead_vectors.append(self.__get_bead_coords(str_pos))
            bead_tangents.append(self.__get_tangent(str_pos))


        (reactants, products) = (self.__state_vec[0], self.__state_vec[-1])
        bead_vectors = [reactants] + bead_vectors + [products]
        bead_tangents = [self.__get_tangent(0)] + bead_tangents + [self.__get_tangent(1)]
        print "bead_vectors =", bead_vectors

        if update:
            self.__state_vec = bead_vectors
            print "New beads generated:", self.__state_vec

            self.__path_tangents = bead_tangents
            print "Tangents updated:", self.__path_tangents

        return bead_vectors

    def generate_beads(self, update = False):
        """Returns an array of the self.__beads_count vectors of the coordinates 
        of beads along a reaction path, according to the established path 
        (line, parabola or spline) and the parameterisation density."""

        assert len(self.__fs) > 1

        # Find total string length and incremental distances x along the string 
        # in terms of the normalised coodinate y, as a list of (x,y).
        (total_str_len, incremental_positions) = self.__get_total_str_len()

        # For the desired distances along the string, find the values of the
        # normalised coordinate that achive those distances.
        normd_positions = self.__generate_normd_positions(total_str_len, incremental_positions)

        bead_vectors = []
        bead_tangents = []
#        print "normd_positions =", normd_positions
        for str_pos in normd_positions:
            bead_vectors.append(self.__get_bead_coords(str_pos))
            bead_tangents.append(self.__get_tangent(str_pos))


        reactants = self.__state_vec[0]
        products = self.__state_vec[-1]
        bead_vectors = [reactants] + bead_vectors + [products]
        bead_tangents = [self.__get_tangent(0)] + bead_tangents + [self.__get_tangent(1)]
        print "bead_vectors =", bead_vectors

        if update:
            self.__state_vec = bead_vectors
            print "New beads generated:", self.__state_vec

            self.__path_tangents = bead_tangents
            print "Tangents updated:", self.__path_tangents

        return bead_vectors
        
    def __get_str_positions_exact(self):
        from scipy.integrate import quad
        from scipy.optimize import fmin

        integrated_density_inc = 1.0 / (self.beads_count - 1.0)
        requirement_for_prev_bead = 0.0
        requirement_for_next_bead = integrated_density_inc
        print "idi =", integrated_density_inc

        x_min = 0.0
        x_max = -1

        str_poses = []

        def f_opt(x):
            (i,e) = quad(self.__rho, x_min, x)
            tmp = (i - integrated_density_inc)**2
            return tmp

        for i in range(self.beads_count - 2):

            x_max = fmin(f_opt, requirement_for_next_bead, disp=False)
            str_poses.append(x_max[0])
            x_min = x_max[0]
            requirement_for_prev_bead = requirement_for_next_bead
            requirement_for_next_bead += integrated_density_inc

        """print "fractions along string:", str_poses
        dump_diffs("spd_exact", str_poses)"""
        return str_poses

    def __get_str_positions(self):
        """Based on the provided density function self.__rho(x) and 
        self.bead_count, generates the fractional positions along the string 
        at which beads should occur."""

        param_steps = arange(0, 1 - self.__step, self.__step)
        integrated_density_inc = 1.0 / (self.beads_count - 1.0)
        requirement_for_next_bead = integrated_density_inc

#        print "param_steps =", param_steps

        integral = 0
        str_positions = []
        for s in param_steps:
#            (i, err) = scipy.integrate.quad(self.__rho, s, s + self.step)
            integral += self.__rho(s) * self.__step
            if integral > requirement_for_next_bead:
#                msg = "rfnb = %f integral =  %f" % (requirement_for_next_bead, integral)
#                print msg
                str_positions.append(s)
#                print "req = ", requirement_for_next_bead
                requirement_for_next_bead += integrated_density_inc
        
#        print "str_positions =", str_positions
        dump_diffs("spd", str_positions)
        return str_positions

    def __get_bead_coords(self, x):
        """Returns the coordinates of the bead at point x <- [0,1]."""
        bead_coords = []
#        print "len(self.__fs) =", len(self.__fs)
        for f in self.__fs:
            bead_coords.append(f.f(x))

        return (array(bead_coords).flatten())

    def __get_tangent(self, x):
        """Returns the tangent to the path at point x <- [0,1]."""

        path_tangent = []
        for f in self.__fs:
            path_tangent.append(f.fprime(x))

        t = array(path_tangent).flatten()
        t = t / linalg.norm(t)
        return t

    def set_rho(self, new_rho):
        """Set new bead density function, ensuring that it is normalised."""
        (int, err) = scipy.integrate.quad(new_rho, 0.0, 1.0)
        self.__rho = lambda x: new_rho(x) / int
        return self.__rho

    def __generate_normd_positions_exact(self, total_str_len):

        #  desired fractional positions along the string
        fractional_positions = self.__get_str_positions_exact()

        normd_positions = []

        prev_norm_coord = 0
        prev_len_wanted = 0

        from scipy.integrate import quad
        from scipy.optimize import fmin
        for frac_pos in fractional_positions:
            next_norm_coord = frac_pos
            next_len_wanted = total_str_len * frac_pos
            length_err = lambda x: \
                (dup2val(quad(self.__arc_dist_func, prev_norm_coord, x)) - (next_len_wanted - prev_len_wanted))**2
            next_norm_coord = fmin(length_err, next_norm_coord, disp=False)

            normd_positions.append(next_norm_coord)

            prev_norm_coord = next_norm_coord
            prev_len_wanted = next_len_wanted

        return normd_positions
            
    def __generate_normd_positions(self, total_str_len, incremental_positions):
        """Returns a list of distances along the string in terms of the normalised 
        coordinate, based on desired fractional distances along string."""

        fractional_positions = self.__get_str_positions()

        normd_positions = []

        print "fractional_positions: ", fractional_positions, "\n"
        for frac_pos in fractional_positions:
#            print "frac_pos = ", frac_pos, "total_str_len = ", total_str_len
            for (norm, str) in incremental_positions:

                if str >= frac_pos * total_str_len:
#                    print "norm = ", norm
                    normd_positions.append(norm)
                    break

        print "normd_positions =", normd_positions
        dump_diffs("npd", normd_positions)
        return normd_positions


class GrowingString(ReactionPathway):
    def __init__(self, reactants, products, f_test, f_density, qc_driver, 
        beads_count = 10):

        ReactionPathway.__init__(self, reactants, products, f_test)
        self.__qc_driver = qc_driver

        self.__path_rep = PathRepresentation([reactants, products], beads_count)
        self.__path_rep.regen_path_func()
        self.__path_rep.generate_beads(update = True)

    def obj_func(self, new_state_vec = []):
        flab("called")
        self.update_path(new_state_vec, respace = True)

        # The following code block will need to be replaced for parallel operation
        pes_energies = 0
        for bead_vec in self.__path_rep.get_state_vec()[1:-1]:
            pes_energies += self.__qc_driver.energy(bead_vec)

        print "ENERGY =", pes_energies
        return pes_energies
       
    def obj_func_grad(self, new_state_vec = []):
        flab("called")
        self.update_path(new_state_vec, respace = True)

        gradients = []

        ts = self.__path_rep.get_path_tangents()
        for i in range(self.__path_rep.beads_count)[1:-1]:
            g = self.__qc_driver.gradient(self.__path_rep.get_state_vec()[i])
            t = ts[i]
#            print "g_before = ", g,
#            print "t =", t
            g = project_out(t, g)
#            print "g_after =", g
            gradients.append(g)

        react_gradients = prod_gradients = zeros(self.__path_rep.get_dimensions())
        gradients = [react_gradients] + gradients + [prod_gradients]
        
        gradients = array(gradients).flatten()
        print "gradients =", gradients
        return (array(gradients).flatten())

    def update_path(self, state_vec = [], respace = True):
        """After each iteration of the optimiser this function must be called.
        It rebuilds a new (spline) representation of the path and then 
        redestributes the beads according to the density function."""

        flab("called", respace)

        if len(state_vec) > 2:
            self.__path_rep.set_state_vec(state_vec)

#        print "update_path: self.__path_rep.get_state_vec() =", self.__path_rep.get_state_vec()
        # rebuild line, parabola or spline representation of path
        self.__path_rep.regen_path_func()

        # respace the beads along the path
        if respace:
            self.__path_rep.generate_beads_exact(update = True)

    def plot(self):
        flab("(GS) called")
#        self.__path_rep.generate_beads_exact()
        plot2D(self.__path_rep)

    def get_state_vec(self):
        return array(self.__path_rep.get_state_vec()).flatten()

#    def get_dimensions(self):
#        return self.__dimensions


def project_out(component_to_remove, vector):
    """Projects the component of 'vector' that list along 'component_to_remove'
    out of 'vector' and returns it."""
    projection = dot(component_to_remove, vector)
    output = vector - projection * component_to_remove
    return output

class NEB(ReactionPathway):
    """Implements a Nudged Elastic Band (NEB) transition state searcher."""

    def __init__(self, reactants, products, f_test, baseSprConst, qcDriver, beadsCount = 10):
        ReactionPathway.__init__(self, reactants, products, f_test, beadsCount)
        self.baseSprConst = baseSprConst
        self.qcDriver = qcDriver
        self.tangents = zeros(beadsCount * self.dimension)
        self.tangents.shape = (beadsCount, self.dimension)

        # Make list of spring constants for every inter-bead separation
        # For the time being, these are uniform
        self.sprConstVec = array([self.baseSprConst for x in range(beadsCount - 1)])

    def special_reduce(self, list, ks = [], f1 = lambda a,b: a-b, f2 = lambda a: a**2):
        """For a list of x_0, x_1, ... , x_(N-1)) and a list of scalars k_0, k_1, ..., 
        returns a list of length N-1 where each element of the output array is 
        f2(f1(k_i * x_i, k_i+1 * x_i+1)) ."""

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

        list = f1 (listPos, listNeg)
        list = f2 (list[1:-1])

        return list

    def update_tangents(self):
        # terminal beads have no tangent
        self.tangents[0]  = zeros(self.dimension)
        self.tangents[-1] = zeros(self.dimension)
        for i in range(self.beadsCount)[1:-1]:
            self.tangents[i] = ( (self.stateVec[i] - self.stateVec[i-1]) + (self.stateVec[i+1] - self.stateVec[i]) ) / 2
            self.tangents[i] /= linalg.norm(self.tangents[i], 2)

    def update_bead_separations(self):
        self.beadSeparationSqrsSums = array( map (sum, self.specialReduce(self.stateVec).tolist()) )
        self.beadSeparationSqrsSums.shape = (self.beadsCount - 1, 1)

    def get_state_as_array(self):
        return self.stateVec.flatten()

    def obj_func(self, newStateVec = []):
        assert size(self.stateVec) == self.beadsCount * self.dimension

        if newStateVec != []:
            self.stateVec = array(newStateVec)
            self.stateVec.shape = (self.beadsCount, self.dimension)

        self.updateTangents()
        self.updateBeadSeparations()
        
        forceConstsBySeparationsSquared = multiply(self.sprConstVec, self.beadSeparationSqrsSums.flatten()).transpose()
        springEnergies = 0.5 * ndarray.sum (forceConstsBySeparationsSquared)

        # The following code block will need to be replaced for parallel operation
        pesEnergies = 0
        for beadVec in self.stateVec[1:-1]:
            pesEnergies += self.qcDriver.energy(beadVec)

        return (pesEnergies + springEnergies)

    def obj_func_grad(self, newStateVec = []):

        # If a state vector has been specified, return the value of the 
        # objective function for this new state and set the state of self
        # to the new state.
        if newStateVec != []:
            self.stateVec = array(newStateVec)
            self.stateVec.shape = (self.beadsCount, self.dimension)

        self.updateBeadSeparations()
        self.updateTangents()

        separationsVec = self.beadSeparationSqrsSums ** 0.5
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
            # OLD LINE:
#            pesForces[i] = pesForces[i] - dot(pesForces[i], self.tangents[i]) * self.tangents[i]

            # NEW LINE:
            pesForces[i] = project_out(self.tangents[i], pesForces[i])

#            print "pesafter =", pesForces[i], "t =", self.tangents[i]

        gradientsVec = -1 * (pesForces + springForces)

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

    x = PathRepresentation(r, 5)

    # Build linear, quadratic or spline representation of the path,
    # depending on the number of points.
    x.regen_path_func()
    x.beads_count = 20
    x.generate_beads(update=True)
    print "tangents =", x.get_path_tangents()

    plot2D(x)


def plot2D(react_path, path_res = 0.01):
    """Given a path object react_path, displays the a 2D depiction of it's 
    first two dimensions as a graph."""
    g = Gnuplot.Gnuplot(debug=1)

    g.xlabel('x')
    g.ylabel('y')
    g('set xrange [0:3]')
    g('set yrange [0:3]')

    # Get some tmp filenames
    (fd, tmp_file1,) = tempfile.mkstemp(text=1)
    (fd, tmp_file2,) = tempfile.mkstemp(text=1)
    (fd, tmp_file3,) = tempfile.mkstemp(text=1)

    params = arange(0, 1 + path_res, path_res)
    f_x = react_path.get_fs()[0].f
    f_y = react_path.get_fs()[1].f
    xs = array ([f_x(p) for p in params])
    ys = array ([f_y(p) for p in params])
#    print "params: ", params
#    print "xs: ", xs
#    print "ys: ", ys

    # smooth path
    smooth_path = vstack((xs,ys)).transpose()
#    print "smooth_path =", smooth_path
    Gnuplot.Data(smooth_path, filename=tmp_file1, inline=0, binary=0)
    
    # state vector
    data2 = react_path.get_state_vec()
    print "plot2D: react_path.get_state_vec() =", data2
    Gnuplot.Data(data2, filename=tmp_file2, inline=0, binary=0)

    # points along path
    beads = react_path.generate_beads_exact()
    Gnuplot.Data(beads, filename=tmp_file3, inline=0, binary=0)

    # draw tangent to the path
    pt_ix = 4
    t0_grad = react_path.get_path_tangents()[pt_ix][1] / react_path.get_path_tangents()[pt_ix][0]
    t0_str = "%f * (x - %f) + %f" % (t0_grad, react_path.get_state_vec()[pt_ix][0], react_path.get_state_vec()[pt_ix][1])
    t0_func = Gnuplot.Func(t0_str)

    # PLOT THE VARIOUS PATHS
    g.plot(t0_func, Gnuplot.File(tmp_file1, binary=0, title="Smooth", 
        with_ = "lines"), Gnuplot.File(tmp_file2, binary=0, 
        with_ = "linespoints", title = "get_state_vec()"), Gnuplot.File(tmp_file3, binary=0, 
        title="points on string from optimisation", with_ = "points"))
    raw_input('Press to continue...\n')

    os.unlink(tmp_file1)
    os.unlink(tmp_file2)
    os.unlink(tmp_file3)

def test_GrowingString():
    from scipy.optimize import fmin_bfgs
    f_test = lambda x: True
    rho = lambda x: 1
    surf_plot = SurfPlot(GaussianPES())

    gs = GrowingString(reactants, products, f_test, rho, qc_driver = GaussianPES(), 
        beads_count = 16)

    # Wrapper callback function
    def mycb(x):
        flab("called")
#       gs.update_path(x, respace = True)
#        surf_plot.plot(x)
        return gs.get_state_vec()

    from scipy.optimize.lbfgsb import fmin_l_bfgs_b
    from scipy.optimize import fmin_cg

    print "gsv =", gs.get_state_vec()
#    (opt, a, b) = fmin_l_bfgs_b(gs.obj_func, gs.get_state_vec(), fprime = gs.obj_func_grad) 
#    opt = fmin_bfgs(gs.obj_func, gs.get_state_vec(), fprime = gs.obj_func_grad, callback=surf_plot.plot) 
    opt = gd(gs.obj_func, gs.get_state_vec(), fprime = gs.obj_func_grad, callback = mycb) 

    gs.plot()
    print "path to plot (for surface) =", opt
    surf_plot.plot(opt)

def dump_diffs(pref, list):
    prev = 0
    for p in list:
        print "%s = %f" % (pref, (p - prev))
        prev = p
    print

def gd(f, x0, fprime, callback = lambda x: Nothing):
    i = 0
    x = copy.deepcopy(x0)
    while 1:
        g = fprime(x)
        print "it(x) =", x
        if linalg.norm(g, ord=inf) < 0.0005:
            print "%d iterations" % i
            break

        i += 1
        x = callback(x)
        x -= g * 0.5

    x = callback(x)
    return x

class SurfPlot():
    def __init__(self, pes):
        self.__pes = pes

    def plot(self, path):
        flab("called")
        opt = copy.deepcopy(path)

        # Points on grid to draw PES
        ps = 20.0
        xrange = arange(ps)*(3.0/ps)
        yrange = arange(ps)*(3.0/ps)

        # Make a 2-d array containing a function of x and y.  First create
        # xm and ym which contain the x and y values in a matrix form that
        # can be `broadcast' into a matrix of the appropriate shape:
        g = Gnuplot.Gnuplot(debug=1)
        g('set data style lines')
#        g('set hidden')
        g('set contour')
        g('set cntrparam levels 100')
        g.xlabel('x')
        g.ylabel('y')

        # Get some tmp filenames
        (fd, tmpPESDataFile,) = tempfile.mkstemp(text=1)
        (fd, tmpPathDataFile,) = tempfile.mkstemp(text=1)
        Gnuplot.funcutils.compute_GridData(xrange, yrange, 
            lambda x,y: self.__pes.energy([x,y]), filename=tmpPESDataFile, binary=0)
        opt.shape = (-1,2)
        pathEnergies = array (map (self.__pes.energy, opt.tolist()))
        pathEnergies += 0.05
        xs = array(opt[:,0])
        ys = array(opt[:,1])
        print "xs =",xs, "ys =",ys
        data = transpose((xs, ys, pathEnergies))
        Gnuplot.Data(data, filename=tmpPathDataFile, inline=0, binary=0)

        # PLOT SURFACE AND PATH
        g.splot(Gnuplot.File(tmpPESDataFile, binary=0), 
            Gnuplot.File(tmpPathDataFile, binary=0, with_="linespoints"))
        print "Path to plot (SurfPlot) =", path
        raw_input('Press to continue...\n')

        os.unlink(tmpPESDataFile)
        os.unlink(tmpPathDataFile)


def mytest_NEB():
    from scipy.optimize import fmin_bfgs

    defaultSprConst = 0.01
    neb = NEB(reactants, products, lambda x: True, defaultSprConst,
        GaussianPES(), beadsCount = 15)
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
    Gnuplot.funcutils.compute_GridData(xrange, yrange, 
        lambda x,y: gpes.energy([x,y]), filename=tmpPESDataFile, binary=0)
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
    g.splot(Gnuplot.File(tmpPESDataFile, binary=0), 
        Gnuplot.File(tmpPathDataFile, binary=0, with_="linespoints"))
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


