from __future__ import with_statement

# FIXME: this module is imported even if its functionality, and that of ASE, is
# not used. E.g. in doctests of pes/mueller_brown.py.
try:
    from ase.io import write
except:
    def write(path, atoms, format = "?"):
        pass

from numpy import savetxt
from pickle import dump

def empty_traj(geo, iter, adds, adds2):
    """
    Do nothing
    """
    pass

class traj_every:
    """
    Writes to every iteration iter a geometry file geo<iter>
    and a mode vector file (in internal coordinates) mode<iter>
    """
    def __init__(self, atoms, funcart):
        self.atoms = atoms
        self.fun = funcart
        self.logger = dimer_log(self.atoms.get_chemical_symbols(), funcart)

    def __call__(self, geo, iter, adds_files, adds_only_pickle):
        self.atoms.set_positions(self.fun(geo))
        write(("geo" + str(iter)), self.atoms, format = "xyz")

        for item in adds_files:
            val, name, text = item
            savetxt( name + str(iter), val)

        self.logger([(geo, None, "Center")] + adds_files + adds_only_pickle)

class traj_last:
    """
    After each iteration it updates geometry file actual_geo
    with the geometry of the system and actual_mode with
    the current mode vector
    """
    def __init__(self, atoms, funcart):
        self.atoms = atoms
        self.fun = funcart
        self.logger = dimer_log(self.atoms.get_chemical_symbols(), funcart)

    def __call__(self, geo, iter, adds_files, adds_only_pickle):
        self.atoms.set_positions(self.fun(geo))
        write("actual_geo", self.atoms, format = "xyz")

        for item in adds_files:
            val, name, text = item
            savetxt("actual_" + name, val)

        self.logger([(geo, None, "Center")] + adds_files + adds_only_pickle)

class traj_long:
    """
    After each iteration it updates geometry file actual_geo
    with the geometry of the system and actual_mode with
    the current mode vector
    """
    def __init__(self, atoms, funcart, names):
        from os import remove
        self.atoms = atoms
        self.fun = funcart
        self.logger = dimer_log(self.atoms.get_chemical_symbols(), funcart)
        try:
            remove("all_geos")
        except OSError:
            pass

        for name in names:
            try:
                remove("all" + name)
            except OSError:
                pass

    def __call__(self, geo, iter, adds_files, adds_only_pickle):
        self.atoms.set_positions(self.fun(geo))
        write("actual_geo", self.atoms, format = "xyz")

        with open("actual_geo", "r") as f_in:
            gs = f_in.read()

        with open("all_geos", "a") as f_out:
            f_out.write(gs)

        for item in adds_files:
            val, name, text = item
            savetxt("actual_" + name, val)

            with open("actual_" + name, "r") as f_in:
                gs = f_in.read()

            with open("all_" + name, "a") as f_out:
                line = text + " of iteration " + str(iter) + "\n"
                f_out.write(line)
                f_out.write(gs)

        self.logger([(geo, None, "Center")] + adds_files + adds_only_pickle)

def dimer_log(symbols, funcart, filename = "progress.pickle"):
    """
    Returns a  callback funciton  that implements a  general interface
    that appends dimer  state to a file. Overwrites  the file, if that
    exists.
    """
    #
    # A valid logfile  will contain pickled Atoms object  as the first
    # record:
    #
    obj = symbols, funcart
    with open(filename, "w") as logfile:
        dump(obj, logfile)

    def callback(content):
        """
        Append geometry to  a file, prepending the key.  Has to adhere
        to the same interface as the callback prepared by empty_log()
        """

        with open(filename, "a") as logfile:
            dump(content, logfile)

    return callback
