#!/usr/bin/env python

import ConfigParser
import sys
import getopt
import re
import logging

import molinterface
import sched
import os

from common import *

import searcher

file_dump_count = 0

# setup logging
import logging
print "Defining logger"
lg = logging.getLogger(__name__)
lg.setLevel(logging.INFO)

if not globals().has_key("ch"):
    print "Defining stream handler"
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    lg.addHandler(ch)
formatter = logging.Formatter("%(name)s (%(levelname)s): %(message)s")
ch.setFormatter(formatter)



class Usage(Exception):
    def __init__(self, msg):
        self.msg = msg
    def __str__(self, msg):
        return self.msg

class ParseError(Exception):
    def __init__(self, msg):
        self.msg = "Parse Error: " + msg
    def __str__(self):
        return self.msg


def main(argv=None):
    """Not Yet Implemented
        1. read input file containing
           a. method to use, number of processors, etc.
           b. reactant, transition state(s), product, in that order
           c. ...
        2. initialise searcher, molinterface, drivers, etc.
        3. start searcher
    """

    if argv is None:
        argv = sys.argv[1:]
    try:
        try:
            opts, args = getopt.getopt(argv[1:], "h", ["help"])
        except getopt.error, msg:
             raise Usage(msg)

        print "argv =", argv
        if len(argv) != 1:
            raise Usage("Exactly 1 input file must be specified.")
        inputfile = argv[0]

    except Usage, err:
        print >>sys.stderr, err.msg
        print >>sys.stderr, "for help use --help"
        return 2

    # try to parse config file
    mol_files = []
    params = []
    try:
        config = ConfigParser.RawConfigParser()
        config.read(inputfile)

        if config.has_section('parameters'):
            params = dict(config.items('parameters'))
            params["inputfile"] = inputfile
        else:
            raise ParseError("Could not find section 'parameters'")

        print "parameters: ", params

        if not "processors" in params:
            raise ParseError("Processor configuration not specified")
        
        proc_spec_str = re.findall(r"(\d+)\s*,(\d+)\s*,(\d+)\s*", params["processors"])
        if proc_spec_str != []:
            total, max, norm = proc_spec_str[0]
            proc_spec = (int(total), int(max), int(norm))
            params["processors"] = proc_spec
        else:
            raise ParseError("Couldn't parse processor configuration")
        

        for geom_ix in range(MAX_GEOMS):
            section_name = "geom" + str(geom_ix)
            if config.has_section(section_name):
                if config.has_option(section_name, 'file'):
                    mol_files.append(config.get(section_name, 'file'))
                else:
                    raise ParseError("No filename given for " + section_name)
            else:
                if geom_ix < 1:
                    raise ParseError("There must be at least two geometries given (i.e. a reactant and a product)")
                else:
                    break

    except ParseError, err:
        print err.msg
        return 1

    logging.info("Molecules list: " + str(mol_files))
    logging.info("Parameters list: " + str(params))

    mol_strings = []
    try:
        for file in mol_files:
            f = open(file, "r")
            mystr = f.read()
            f.close()
            mol_strings.append(mystr)

    except Exception, e:
        print "Exception:", str(e)
        return 1

    setup_and_run(mol_strings, params)

def setup_and_run(mol_strings, params):
    """1. Setup all objects based on supplied parameters and molecular 
    geometries. 2. Run searcher. 3. Print Summary."""

    # setup MoIinterface
    #mol_interface_params = setup_params(params)
    mol_interface = molinterface.MolInterface(mol_strings, params)

    calc_man = sched.CalcManager(mol_interface, 
                           params["processors"]) # TODO: check (earlier) that this param is a tuple / correct format

    # SETUP / RUN SEARCHER
    reagent_coords = mol_interface.reagent_coords
    print ("Initial state of path: %s" % reagent_coords)

    max_iterations = DEFAULT_MAX_ITERATIONS
    if "max_iterations" in params:
        max_iterations = int(params["max_iterations"])

    if params["method"] == "neb":
        spr_const = float(params["spr_const"])
        beads_count = int(params["beads_count"])
        neb = searcher.NEB(reagent_coords, 
                  mol_interface.geom_checker, 
                  spr_const, 
                  calc_man, 
                  beads_count,
                  parallel=True)
        # initial path
        dump_beads(mol_interface, neb, params)
        dump_steps(neb)


        # callback function
        def mycb(x):
            print line()
#                logging.info("Current path: %s" % str(x))
            print neb
#            print neb.gradients_vec
#            print numpy.linalg.norm(neb.gradients_vec)
            dump_beads(mol_interface, neb, params)
            dump_steps(neb)
            print line()
            return x


        print "Launching optimiser..."
        if params["optimizer"] == "l_bfgs_b":
            """default_spr_const = 1.
            reactants = array([0,0])
            products = array([3,3])

            from searcher import NEB
            neb = NEB([reactants, products], lambda x: True, default_spr_const,
                GaussianPES(), beads_count = 10)
            init_state = neb.get_state_as_array()


            # Wrapper callback function
            def mycb(x):
                print "x:",x
                return x

            from scipy.optimize.lbfgsb import fmin_l_bfgs_b
            opt, energy, dict = fmin_l_bfgs_b(neb.obj_func, init_state, fprime=neb.obj_func_grad, callback=mycb, pgtol=0.05)"""

            import scipy.optimize.lbfgsb as so

            opt, energy, dict = so.fmin_l_bfgs_b(neb.obj_func,
                                              neb.get_state_as_array(),
                                              fprime=neb.obj_func_grad,
                                              callback=mycb,
                                              pgtol=0.005,
                                              maxfun=max_iterations)
            print opt
            print energy
            print dict
        elif params["optimizer"] == "bfgs":
            from scipy.optimize import fmin_bfgs
            opt = fmin_bfgs(neb.obj_func, 
                neb.get_state_as_array(), 
                fprime=neb.obj_func_grad, 
                callback=mycb,
                maxiter=max_iterations)

        elif params["optimizer"] == "grad_descent":
            opt = opt_gd(neb.obj_func, 
                neb.get_state_as_array(), 
                fprime=neb.obj_func_grad, 
                callback=mycb)

        else:
            raise ParseError("Unknown optimizer: " + params["optimizer"])

    else:
        raise ParseError("Unknown method: " + params["method"])

    # PRODUCE OUTPUT
    print "Finished"
#    print opt
#    print energy

    dump_beads(mol_interface, neb, params)
    print "steps"
    dump_steps(neb)


def dump_beads(mol_interface, path_rep, params):
    """Writes the states along the reaction path to a file in a form that can
    be read by a molecule viewing program."""

    from copy import deepcopy

    global file_dump_count
    file_dump_count += 1

    local_bead_forces = deepcopy(path_rep.bead_forces)
    local_bead_forces.shape = (path_rep.beads_count, -1)

    mystr = ""
    print path_rep.bead_pes_energies
    for bead, i in zip(path_rep.get_bead_coords(), range(path_rep.beads_count)):
        mystr += str(mol_interface.natoms)
        mystr += "\nBead " + str(i) + ": Energy = " + str(path_rep.bead_pes_energies[i]) + "\n"
#        mystr += "Gradients = " + str(local_bead_forces[i])
        molstr, coords = mol_interface.coords2xyz(bead)
        mystr += molstr
#        mystr += "\n\n"

    path = os.path.splitext(params["inputfile"])[0] + str(file_dump_count) + LOGFILE_EXT

    f = open(path, "w")
    f.write(mystr)
    f.close()

def dump_steps(path_rep):
    """Prints the steps taken during the optimisation."""

    print "Steps taken during the optimisation"

    for i in range(len(path_rep.history))[1:]:
        prev = path_rep.history[i-1]
        curr = path_rep.history[i]

        diff = curr - prev

        diff.shape = (path_rep.beads_count, -1)
        diff = [numpy.linalg.norm(i) for i in diff]
        print diff

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt, e:
        print e
    import threading
    print "active threads were..."
    print threading.enumerate()
    sys.exit()

