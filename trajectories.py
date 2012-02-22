from ase.io import write
from numpy import savetxt
from pickle import dump

def empty_traj(geo, iter, adds):
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

    def __call__(self, geo, iter, adds):
        self.atoms.set_positions(self.fun(geo))
        write(("geo" + str(iter)), self.atoms, format = "xyz")

        for item in adds:
            val, name, text = item
            savetxt( name + str(iter), val)

class traj_last:
    """
    After each iteration it updates geometry file actual_geo
    with the geometry of the system and actual_mode with
    the current mode vector
    """
    def __init__(self, atoms, funcart):
        self.atoms = atoms
        self.fun = funcart

    def __call__(self, geo, iter, adds):
        self.atoms.set_positions(self.fun(geo))
        write("actual_geo", self.atoms, format = "xyz")

        for item in adds:
            val, name, text = item
            savetxt("actual_" + name, val)

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
        try:
            remove("all_geos")
        except OSError:
            pass
       
        for name in names:
            try:
                remove("all" + name)
            except OSError:
                pass

    def __call__(self, geo, iter, adds):
        self.atoms.set_positions(self.fun(geo))
        write("actual_geo", self.atoms, format = "xyz")
        f_in = open("actual_geo", "r")
        gs = f_in.read()
        f_in.close()
        f_out = open("all_geos", "a")
        f_out.write(gs)
        f_out.close()

        for item in adds:
            val, name, text = item
            savetxt("actual_" + name, val)


            f_in = open("actual_" + name, "r")
            gs = f_in.read()
            f_in.close()
            f_out = open("all_" + name, "a")
            line = text + " of iteration " + str(iter) + "\n"
            f_out.write(line)
            f_out.write(gs)
            f_out.close()

class empty_log:
    def __init__(self):
        pass

    def __call__(self, key, geo, mode = None):
        pass

    def record(self):
        pass

class dimer_log:
    def __init__(self, atoms, filename = "dimer.log.pickle"):
        from os import remove
        self.atoms = atoms
        self.filename = filename
        self.dict = {}

        try:
            remove(filename)
        except OSError:
            pass

        logfile = open(filename, "w")
        dump(self.atoms, logfile)
        logfile.close()

    def __call__(self, key, geo):
        if key in self.dict:
            self.dict[key].append(geo)
        else:
            self.dict[key] = [geo]

    def record(self):
        logfile = open(self.filename, "a")
        dump(self.dict, logfile)
        logfile.close()
        self.dict.clear()
