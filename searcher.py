#!/usr/bin/python

#from scipy import *
import sys
import inspect

import scipy.integrate
from scipy.optimize import fminbound

from scipy import interpolate
import tempfile, os
import logging
from copy import deepcopy
import pickle

from numpy import linalg, floor, zeros, array, ones, arange, arccos, hstack, ceil, abs, ndarray, sqrt, column_stack, dot

from path import Path

from common import * # TODO: must unify
import aof.common as common


# TODO: remove global
plot_count = 0

lg = logging.getLogger("aof.searcher")
lg.setLevel(logging.INFO)

if not globals().has_key("lg"):
    sh = logging.StreamHandler()
    sh.setLevel(logging.DEBUG)
    lg.addHandler(sh)


def _functionId(nFramesUp):
    """ Create a string naming the function n frames up on the stack.
    """
    import sys
    co = sys._getframe(nFramesUp+1).f_code
    return "%s (%s @ %d)" % (co.co_name, co.co_filename, co.co_firstlineno)


# Function labeller
def flab(*args):
    import sys
    args = [str(a) for a in args]
    lg.info("**** " + sys._getframe(1).f_code.co_name + ' '.join(args))

class ReactionPathway:
    """Abstract object for chain-of-state reaction pathway."""
    dimension = -1

    e_calls = 0
    g_calls = 0
    bead_g_calls = 0
    bead_e_calls = 0

    def __init__(self, reagents, beads_count, qc_driver, parallel):

        self.parallel = parallel
        self.qc_driver = qc_driver
        self.beads_count = beads_count

        self.__dimension = len(reagents[0])
        #TODO: check that all reagents are same length

        # forces perpendicular to pathway
        self.bead_forces = zeros(beads_count * self.dimension)

#        self.bead_pes_energies = deepcopy(self.default_initial_bead_pes_energies)

        self.tangents = zeros(beads_count * self.dimension)
        self.tangents.shape = (beads_count, self.dimension)

        self.bead_pes_energies = zeros(beads_count)
        self.history = []
        self.energy_history = []

        self.grad_update_mask = [False] + [True for i in range(beads_count-2)] + [False]

    def __str__(self):
        strrep = "Bead Energies: " + str(self.bead_pes_energies) + "\n"
        total_energy = 0
        for i in range(len(self.bead_pes_energies)): #[1:-1]:
            total_energy += self.bead_pes_energies[i]
        max_energy = self.bead_pes_energies.max()
        strrep += "Total Band Energy: " + str(total_energy)
        strrep += "\nRMS Perpendicular bead forces: " + str(common.rms(self.bead_forces))
        strrep += "\nFunction calls: " + str(self.e_calls)
        strrep += "\nGradient calls: " + str(self.g_calls)
        strrep += "\nArchive (bc, bgc, gc, ec, rmsf, e, maxe): %d\t%d\t%d\t%d\t%f\t%f\t%f" % \
            (self.beads_count,
                self.bead_g_calls, 
                self.g_calls, 
                self.e_calls, 
                common.rms(self.bead_forces), 
                total_energy, 
                max_energy)

        strrep += "\nAngles: %s" % str(self.get_angles())

        return strrep

    def get_angles(self):
        """Returns an array of angles between beed groups of 3 beads."""

        angles = []
        for i in range(len(self.state_vec))[2:]:
            t0 = self.state_vec[i-1] - self.state_vec[i-2]
            t1 = self.state_vec[i] - self.state_vec[i-1]
            angles.append(vector_angle(t1, t0))
        return array(angles)

    def pathfs(self):
        return None

    @property
    def dimension(self):
        return self.__dimension

    def get_state_as_array(self):
        """Returns copy of state as flattened array."""
        
        #Do I really need this as well as the one below?
        return deepcopy(self.state_vec).flatten()

    def get_bead_coords(self):
        """Return copy of state_vec as 2D array."""
        tmp = deepcopy(self.state_vec)
        print self.beads_count, self.dimension
        tmp.shape = (self.beads_count, self.dimension)
        return tmp

    def obj_func(self, x):
        lg.info("Chain of States Objective Function call.")
        self.e_calls += 1
        if self.bead_g_calls == 0:
            self.bead_g_calls += self.beads_count
        else:
            self.bead_g_calls += self.beads_count - 2

        self.history.append(deepcopy(x))

    def obj_func_grad(self, x):
        lg.info("Chain of States Objective Function *Gradient* call.")

        self.g_calls += 1
        if self.bead_g_calls == 0:
            self.bead_g_calls += self.beads_count
        else:
            self.bead_g_calls += self.beads_count - 2

        self.history.append(deepcopy(x))

    def record_energy(self):
        self.energy_history.append(sum(self.bead_pes_energies))

    def update_bead_pes_energies(self):
        """
        Updates internal vector with the energy of each bead (based on energy 
        calculations which must already have been scheduled and run, if in 
        parallel mode).
        """
        bead_pes_energies = []
        for bead_vec in self.state_vec:
            e = self.qc_driver.energy(bead_vec)
            bead_pes_energies.append(e)

        self.bead_pes_energies = array(bead_pes_energies)

    def ts_estims(self, tol=1e-10):
        """Returns list of all transition state(s) that appear to exist along
        the reaction pathway."""

        n = self.beads_count
        Es = self.bead_pes_energies.reshape(n,-1)
        dofs = self.state_vec.reshape(n,-1)
        assert len(dofs) == len(Es)
        ys = hstack([dofs, Es])

        step = 1. / n
        xs = arange(0., 1., step)

        # TODO: using Alexei's isolated Path object, eventually must make the
        # GrowingString object use it as well.
        p = Path(ys, xs)

        E_estim_neg = lambda s: -p(s)[-1]
        E_prime_estim = lambda s: p.fprime(s)[-1]

        ts_list = []
        for x in xs[2:]:#-1]:
            E_0 = -E_estim_neg(x - step)
            E_1 = -E_estim_neg(x)
            x_min = fminbound(E_estim_neg, x - step, x, xtol=tol)
            E_x = -E_estim_neg(x_min)
#            print x_min, abs(E_prime_estim(x_min)), E_0, E_x, E_1

            # Use a looser tollerane on the gradient than on minimisation of 
            # the energy function. FIXME: can this be done better?
            E_prime_tol = tol * 1E4
            if abs(E_prime_estim(x_min)) < E_prime_tol and (E_0 <= E_x >= E_1):
                p_ts = p(x_min)
                ts_list.append((p_ts[-1], p_ts[:-1]))

        return ts_list

 
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

class NEB(ReactionPathway):
    """Implements a Nudged Elastic Band (NEB) transition state searcher."""

    def __init__(self, reagents, qc_driver, base_spr_const, beads_count = 10, parallel = False):

        ReactionPathway.__init__(self, reagents, beads_count, qc_driver, parallel)

        self.base_spr_const = base_spr_const

        # Make list of spring constants for every inter-bead separation
        # For the time being, these are uniform
        self.spr_const_vec = array([self.base_spr_const for x in range(beads_count - 1)])

        self.use_upwinding_tangent = True

        #  TODO: generate initial path using all geoms in reagents. Must use
        # path representation object.
        reactants, products = reagents[0], reagents[-1]
        self.state_vec = vector_interpolate(reactants, products, beads_count)

    def special_reduce_old(self, list, ks = [], f1 = lambda a,b: a-b, f2 = lambda a: a**2):
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

        curr_dim = list.shape[1]  # generate zero vector of the same dimension of the list of input dimensions
        z = array(zeros(curr_dim))
        list_pos = vstack((list, z))
        list_neg = vstack((z, list))

        list = f1 (list_pos, list_neg)
        list = f2 (list[1:-1])

        return list

    def update_tangents(self):
        # terminal beads have no tangent
        self.tangents[0]  = zeros(self.dimension)
        self.tangents[-1] = zeros(self.dimension)

        for i in range(self.beads_count)[1:-1]:
            if self.use_upwinding_tangent:
                tang_plus = self.state_vec[i+1] - self.state_vec[i]
                tang_minus = self.state_vec[i] - self.state_vec[i-1]

                Vi = self.bead_pes_energies[i]
                Vi_minus_1 = self.bead_pes_energies[i-1]
                Vi_plus_1 = self.bead_pes_energies[i+1]
                
                delta_V_plus = abs(Vi_plus_1 - Vi)
                delta_V_minus = abs(Vi - Vi_minus_1)

                delta_V_max = max(delta_V_plus, delta_V_minus)
                delta_V_min = min(delta_V_plus, delta_V_minus)

                if Vi_plus_1 > Vi > Vi_minus_1:
                    self.tangents[i] = tang_plus

                elif Vi_plus_1 < Vi < Vi_minus_1:
                    self.tangents[i] = tang_minus

                elif Vi_plus_1 > Vi_minus_1:
                    self.tangents[i] = tang_plus * delta_V_max + tang_minus * delta_V_min

                elif Vi_plus_1 <= Vi_minus_1:
                    self.tangents[i] = tang_plus * delta_V_min + tang_minus * delta_V_max
                else:
                    raise Exception("Should never happen")
            else:
                self.tangents[i] = ( (self.state_vec[i] - self.state_vec[i-1]) + (self.state_vec[i+1] - self.state_vec[i]) ) / 2

            self.tangents[i] /= linalg.norm(self.tangents[i], 2)


    def __len__(self):
        """For compatibility with ASE, pretends that there are atoms with cartesian coordinates."""
        return int(ceil((self.beads_count * self.dimension / 3.)))

    def update_bead_separations(self):
        """Updates internal vector of distances between beads."""
#        self.bead_separation_sqrs_sums = array( map (sum, self.special_reduce(self.state_vec).tolist()) )

        v = self.state_vec.copy()
        seps = []
        for i in range(1,len(v)):
            dv = v[i] - v[i-1]
            seps.append(dot(dv,dv))

        self.bead_separations = array(seps)**0.5

        print self.bead_separations
#        self.bead_separation_sqrs_sums.shape = (self.beads_count - 1, 1)

    def set_positions(self, x):
        """For compatibility with ASE, pretends that there are atoms with cartesian coordinates."""
        self.state_vec = x.flatten()[0:self.beads_count * self.dimension]
        self.state_vec.shape = (self.beads_count, -1)

    def get_positions(self):
        """For compatibility with ASE, pretends that there are atoms with cartesian coordinates.""" 
        return common.make_like_atoms(self.state_vec.copy())

    def get_forces(self):
        """For compatibility with ASE, pretends that there are atoms with cartesian coordinates."""
        return -common.make_like_atoms(self.obj_func_grad())

    def get_potential_energy(self):
        """For compatibility with ASE, pretends that there are atoms with cartesian coordinates."""
        return self.obj_func()

    def obj_func(self, new_state_vec = None):
        assert size(self.state_vec) == self.beads_count * self.dimension

        ReactionPathway.obj_func(self, new_state_vec)

        if new_state_vec != None:
            self.state_vec = array(new_state_vec)
            self.state_vec.shape = (self.beads_count, self.dimension)

        # request and process parallel QC jobs
        if self.parallel:

            for i in range(self.beads_count): # [1:-1]:

                bead_vec = self.state_vec[i]
                self.qc_driver.request_gradient(bead_vec)

            self.qc_driver.proc_requests()
        
        # update vector of energies of individual beads`
        # this is required for the tangent calculation
        self.update_bead_pes_energies()

        self.update_tangents()
        self.update_bead_separations()
        
        spring_energies = self.base_spr_const * self.bead_separations**2
        spring_energies = 0.5 * numpy.sum (spring_energies)
#        print "spring_energies", spring_energies

        # Hmm, why do I do this? WRONG???
        #self.bead_pes_energies = self.default_initial_bead_pes_energies

        pes_energies = sum(self.bead_pes_energies[1:-1])

        print "self.bead_pes_energies", self.bead_pes_energies
        return (pes_energies + spring_energies) #self.bead_pes_energies.max() #(pes_energies + spring_energies)

      
    def obj_func_grad(self, new_state_vec = None):

        ReactionPathway.obj_func_grad(self, new_state_vec)

        # If a state vector has been specified, return the value of the 
        # objective function for this new state and set the state of self
        # to the new state.
        if new_state_vec != None:
            self.state_vec = array(new_state_vec)
            self.state_vec.shape = (self.beads_count, self.dimension)

         # request and process parallel QC jobs
        if self.parallel:

            for i in range(self.beads_count): #[1:-1]:
                self.qc_driver.request_gradient(self.state_vec[i])

            self.qc_driver.proc_requests()

        self.update_bead_pes_energies()
        self.update_bead_separations()
        self.update_tangents()

        total_bead_forces = []
        perp_bead_forces = []

        # get PES forces / project out stuff
        for i in range(self.beads_count)[1:-1]: # don't include end beads, leave their gradients as zero
            pes_force = -self.qc_driver.gradient(self.state_vec[i])
            pes_force = project_out(self.tangents[i], pes_force)
            perp_bead_forces.append(pes_force)

            spring_force = (self.base_spr_const * 
                            (self.bead_separations[i] - self.bead_separations[i-1]) * 
                            self.tangents[i])

            total = pes_force + spring_force

            total_bead_forces.append(total)

        # at this point, parallel component has been projected out
        total_bead_forces = array(total_bead_forces)
        z = zeros(self.dimension)
        total_bead_forces = vstack([z, total_bead_forces, z])

        self.bead_forces = array(perp_bead_forces).flatten()

        if new_state_vec != None:
            self.record_energy()

        return -total_bead_forces.flatten()

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
        return self.fs(x) #[0]

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

class PathRepresentation(object):
    """Supports operations on a path represented by a line, parabola, or a 
    spline, depending on whether it has 2, 3 or > 3 points."""

    def __init__(self, state_vec, beads_count, rho = lambda x: 1, str_resolution = 500):

        # vector of vectors defining the path
        if (isinstance(state_vec, ndarray)):
            self.__state_vec = state_vec
        else:
            self.__state_vec = array(state_vec)

        # number of vectors defining the path
        self.beads_count = beads_count
        self.__dimension = len(state_vec[0])

        self.__str_resolution = str_resolution
        self.__step = 1.0 / self.__str_resolution

        self.__fs = []
        self.__path_tangents = []

        self.__unit_interval = array((0.0,1.0))

        # generate initial paramaterisation density
        # TODO: Linear at present, perhaps change eventually
        points_cnt = len(self.__state_vec)
        self.__normalised_positions = arange(0.0, 1.0 + 1.0 / (points_cnt - 1), 1.0 / (points_cnt - 1))
        self.__normalised_positions = self.__normalised_positions[0:points_cnt]

        self.__max_integral_error = 1e-4

        self.__rho = self.set_rho(rho)

        # TODO check all beads have same dimensionality

    """def get_fs(self):
        return self.__fs"""

    @property
    def fs(self):
        return self.__fs
    @property
    def path_tangents(self):
        return self.__path_tangents

    def recalc_path_tangents(self):
        """Returns the unit tangents to the path at the current set of 
        normalised positions."""

        tangents = []
        for str_pos in self.__normalised_positions:
            tangents.append(self.__get_tangent(str_pos))

        tangents = array(tangents)
        return tangents

    def get_state_vec(self):
        return self.__state_vec

    def set_state_vec(self, new_state_vec):
        self.__state_vec = array(new_state_vec).flatten()
        self.__state_vec.shape = (self.beads_count, -1)

    state_vec = property(get_state_vec, set_state_vec)

    @property
    def dimension(self):
        return self.__dimension

    def regen_path_func(self):
        """Rebuild a new path function and the derivative of the path based on 
        the contents of state_vec."""

        assert len(self.__state_vec) > 1

        self.__fs = []

        for i in range(self.__dimension):

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
                xs = self.__normalised_positions
                assert len(self.__normalised_positions) == len(self.__state_vec)
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
        

        # number of points to chop the string into
        param_steps = arange(0, 1, self.__step)

        list = []
        cummulative = 0

        (str_len_precise, error) = scipy.integrate.quad(self.__arc_dist_func, 0, 1, limit=100)
        lg.debug("String length integration error = " + str(error))
        assert error < self.__max_integral_error

        for i in range(self.__str_resolution):
            pos = (i + 0.5) * self.__step
            sub_integral = self.__step * self.__arc_dist_func(pos)
            cummulative += sub_integral
            list.append(cummulative)

        #lg.debug('int_approx = {0}, int_accurate = {1}'.format(cummulative, str_len_precise))

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

    def generate_beads_exact(self, update = False):
        """Returns an array of the self.__beads_count vectors of the coordinates 
        of beads along a reaction path, according to the established path 
        (line, parabola or spline) and the parameterisation density."""


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
        for str_pos in normd_positions:
            bead_vectors.append(self.__get_bead_coords(str_pos))
            bead_tangents.append(self.__get_tangent(str_pos))


        (reactants, products) = (self.__state_vec[0], self.__state_vec[-1])
        bead_vectors = array([reactants] + bead_vectors + [products])
        bead_tangents = array([self.__get_tangent(0)] + bead_tangents + [self.__get_tangent(1)])

        if update:
            self.state_vec = bead_vectors

            self.__path_tangents = bead_tangents

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
        for str_pos in normd_positions:
            bead_vectors.append(self.__get_bead_coords(str_pos))
            bead_tangents.append(self.__get_tangent(str_pos))

        reactants = self.__state_vec[0]
        products = self.__state_vec[-1]
        bead_vectors = array([reactants] + bead_vectors + [products])
        bead_tangents = array([self.__get_tangent(0)] + bead_tangents + [self.__get_tangent(1)])

        if update:
            self.state_vec = bead_vectors
            lg.debug("New beads generated: " + str(self.__state_vec))

            self.__path_tangents = bead_tangents
            lg.debug("Tangents updated:" + str(self.__path_tangents))

            self.__normalised_positions = array([0.0] + normd_positions + [1.0])
            lg.debug("Normalised positions updated:" + str(self.__normalised_positions))

        return bead_vectors
        
    def __get_str_positions_exact(self):
        from scipy.integrate import quad
        from scipy.optimize import fmin

        integrated_density_inc = 1.0 / (self.beads_count - 1.0)
        requirement_for_prev_bead = 0.0
        requirement_for_next_bead = integrated_density_inc

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

        return str_poses

    def dump_rho(self):
        res = 0.02
        print "rho: ",
        for x in arange(0.0, 1.0 + res, res):
            if x < 1.0:
                print self.__rho(x),
        print
        raw_input("that was rho...")


    def __get_str_positions(self):
        """Based on the provided density function self.__rho(x) and 
        self.beads_count, generates the fractional positions along the string 
        at which beads should occur."""

        param_steps = arange(0, 1 - self.__step, self.__step)
        integrated_density_inc = 1.0 / (self.beads_count - 1.0)
        requirement_for_next_bead = integrated_density_inc

        integral = 0
        str_positions = []
        prev_s = 0
        for s in param_steps:
            integral += 0.5 * (self.__rho(s) + self.__rho(prev_s)) * self.__step
            if integral > requirement_for_next_bead:
                str_positions.append(s)
                requirement_for_next_bead += integrated_density_inc
            prev_s = s
        
#        dump_diffs("spd", str_positions)

        # Return first self.beads_count-2 points. Reason: sometimes, due to
        # inaccuracies in the integration of the density function above, too
        # many points are generated in str_positions.
        return str_positions[0:self.beads_count-2]

    def __get_bead_coords(self, x):
        """Returns the coordinates of the bead at point x <- [0,1]."""
        bead_coords = []
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

    def set_rho(self, new_rho, normalise=True):
        """Set new bead density function, ensuring that it is normalised."""
        if normalise:
            (integral, err) = scipy.integrate.quad(new_rho, 0.0, 1.0)
        else:
            int = 1.0
        self.__rho = lambda x: new_rho(x) / integral
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

        # Get fractional positions along string, based on bead density function
        # and the desired total number of beads
        fractional_positions = self.__get_str_positions()

        normd_positions = []

        lg.debug("fractional_positions: %s" % fractional_positions)
        for frac_pos in fractional_positions:
            for (norm, str) in incremental_positions:

                if str >= frac_pos * total_str_len:
                    normd_positions.append(norm)
                    break

        return normd_positions

class PiecewiseRho:
    """Supports the creation of piecewise functions as used by the GrowingString
    class as the bead density function."""
    def __init__(self, a1, a2, rho, max_beads):
        self.a1, self.a2, self.rho = a1, a2, rho
        self.max_beads = max_beads

    def f(self, x):
        if 0 <= x <= self.a1:
            return self.rho(x)
        elif self.a1 < x <= self.a2:
            return 0.0
        elif self.a2 < x <= 1.0:
            return self.rho(x)
        else:
            lg.error("Value of (%f) not on [0,1], should never happen" % x)
            lg.error("a1 = %f, a2 = %f" % (self.a1, self.a2))

class GrowingString(ReactionPathway):
    def __init__(self, reagents, qc_driver, beads_count = 10, rho = lambda x: 1, growing=True, parallel=False, head_size=None):

        self.__qc_driver = qc_driver

        self.__final_beads_count = beads_count
        if growing:
            initial_beads_count = 4
        else:
            initial_beads_count = self.__final_beads_count

        # TODO: this was moved from the first line recently, check
        ReactionPathway.__init__(self, reagents, initial_beads_count, qc_driver, parallel)

        # create PathRepresentation object
        print "reagents", reagents
        self.__path_rep = PathRepresentation(reagents, initial_beads_count, rho)

        # final bead spacing density function for grown string
        # make sure it is normalised
        (int, err) = scipy.integrate.quad(rho, 0.0, 1.0)
        self.__final_rho = lambda x: rho(x) / int

        # current bead spacing density function for incompletely grown string
        self.update_rho()

        # Build path function based on reagents and possibly transitionstate
        # then place beads along that path
        self.__path_rep.regen_path_func()
        self.__path_rep.generate_beads(update = True)

        # dummy energy of a reactant/product
        # TODO: remove?
        self.__reagent_energy = 0

        self.parallel = parallel

        # Number of beads in growing heads of growing string to calculate 
        # gradients for. All others are left as zero.
        assert head_size == None, "Not yet properly implemented, problem when beads are respaced."
        assert head_size == None or (growing and beads_count / 2 -1 >= head_size > 0)
        self.head_size = head_size

    def pathfs(self):
        return self.__path_rep.fs

    def set_positions(self, x):
        """For compatibility with ASE, pretends that there are atoms with cartesian coordinates."""
        new_state_vec = x.flatten()[0:self.beads_count * self.dimension]
        self.update_path(new_state_vec, respace = True)

    def get_positions(self):
        """For compatibility with ASE, pretends that there are atoms with cartesian coordinates.""" 
        return common.make_like_atoms(self.state_vec)

    def get_forces(self):
        """For compatibility with ASE, pretends that there are atoms with cartesian coordinates."""
        return -common.make_like_atoms(self.obj_func_grad())

    def get_potential_energy(self):
        """For compatibility with ASE, pretends that there are atoms with cartesian coordinates."""
        return self.obj_func()

    def get_state_vec(self):
        return deepcopy(self.__path_rep.state_vec)

    state_vec = property(get_state_vec)

    def get_beads_count(self):
        return self.__path_rep.beads_count

    def set_beads_count(self, new):
        self.__path_rep.beads_count = new

    beads_count = property(get_beads_count, set_beads_count)

    @property
    def tangents(self):
        return self.__path_rep.path_tangents

    def grow_string(self):
        """
        Adds 2, 1 or 0 beads to string (such that the total number of 
        beads is less than or equal to self.__final_beads_count).
        """

        assert self.beads_count <= self.__final_beads_count

        if self.beads_count == self.__final_beads_count:
            return False
        elif self.__final_beads_count - self.beads_count == 1:
            self.beads_count += 1
        else:
            self.beads_count += 2

        self.__path_rep.beads_count = self.beads_count

        # build new bead density function based on updated number of beads
        self.update_rho()
        self.__path_rep.generate_beads(update = True)

        # Build mask of gradients to calculate, effectively freezing some if 
        # head_size, i.e. the size of the growing heads, is specified.
        if self.head_size:
            e = self.beads_count / 2 - self.head_size + self.beads_count % 2
        else:
            e = 1
        m = self.beads_count - 2 * e
        self.grad_update_mask = [False for i in range(e)] + [True for i in range(m)] + [False for i in range(e)]
        lg.debug("Bead Freezing MASK: " + str(self.grad_update_mask))

        lg.info("******** String Grown to %d beads ********", self.beads_count)

        return True

    def update_rho(self):
        """Update the density function so that it will deliver beads spaced
        as for a growing (or grown) string."""

        assert self.beads_count <= self.__final_beads_count

        from scipy.optimize import fmin
        from scipy.integrate import quad

        if self.beads_count == self.__final_beads_count:
            self.__path_rep.set_rho(self.__final_rho)

            return self.__final_rho

        # Value that integral must be equal to to give desired number of beads
        # at end of the string.
        end_int = (self.beads_count / 2.0) \
            / self.__final_beads_count

        f_a1 = lambda x: (quad(self.__final_rho, 0.0, x)[0] - end_int)**2
        f_a2 = lambda x: (quad(self.__final_rho, x, 1.0)[0] - end_int)**2

        a1 = fmin(f_a1, end_int)[0]
        a2 = fmin(f_a2, end_int)[0]

        assert a2 > a1

        pwr = PiecewiseRho(a1, a2, self.__final_rho, self.__final_beads_count)
        self.__path_rep.set_rho(pwr.f)

    def obj_func(self, new_state_vec = None, individual_energies = False):
        ReactionPathway.obj_func(self, new_state_vec)

        self.update_path(new_state_vec, respace = True)

        es = [] #[self.__reagent_energy]

        # request and process parallel QC jobs
        if self.parallel:
            for bead_vec in self.__path_rep.state_vec: #[1:-1]:
                self.qc_driver.request_gradient(bead_vec)
            self.qc_driver.proc_requests()

        self.update_bead_pes_energies()

        es = self.bead_pes_energies

        # assume that energies of end beads are not updated every time
        self.bead_e_calls += len(es) - 2

        total_energies = sum(es)
        if individual_energies:
            return deepcopy(es)
        else:
            return total_energies #es.max() #total_energies
       
    def obj_func_grad(self, new_state_vec = None):
        ReactionPathway.obj_func_grad(self, new_state_vec)

        self.update_path(new_state_vec, respace = True)

        gradients = []

        ts = self.__path_rep.path_tangents

        # request and process parallel QC jobs
        if self.parallel:
            for i, bead_vec in enumerate(self.__path_rep.state_vec): #[1:-1]:
                if self.grad_update_mask[i]:
                    self.qc_driver.request_gradient(bead_vec)
            self.qc_driver.proc_requests()

        self.update_bead_pes_energies()

        # get gradients / perform projections
        for i in range(self.beads_count):
            if self.grad_update_mask[i]:
                g = self.__qc_driver.gradient(self.__path_rep.state_vec[i])
                t = ts[i]
                g = project_out(t, g)
            else:
                g = zeros(self.__path_rep.dimension)
            gradients.append(g)

        gradients = array(gradients).flatten()
        self.bead_forces = deepcopy(gradients)

        if new_state_vec != None:
            self.record_energy()

        return gradients

    def update_path(self, state_vec = None, respace = True):
        """After each iteration of the optimiser this function must be called.
        It rebuilds a new (spline) representation of the path and then 
        redestributes the beads according to the density function."""

        flab("update_pathcalled", respace)

        if state_vec != None:
            self.__path_rep.state_vec = state_vec

        # rebuild line, parabola or spline representation of path
        self.__path_rep.regen_path_func()

        # respace the beads along the path
        if respace:
            self.__path_rep.generate_beads(update = True)

    def plot(self):
#        self.__path_rep.generate_beads_exact()
        plot2D(self.__path_rep)



def project_out(component_to_remove, vector):
    """Projects the component of 'vector' that list along 'component_to_remove'
    out of 'vector' and returns it."""
#    print len(component_to_remove)
#    print len(vector)
    projection = dot(component_to_remove, vector)
    output = vector - projection * component_to_remove
    return output


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
        gs.update_path(x, respace = True)
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


if __name__ == '__main__':
    # FIXME: change to doctest
    pass

