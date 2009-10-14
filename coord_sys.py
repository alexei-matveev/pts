class Atoms():
    pass

class CoordSys(Atoms):
    def __init__(self, cartesian_positions=None):
        pass

    def get_positions(self):
        assert False, "Abstract function"

    def set_positions(self, x):
        assert False, "Abstract function"

    def get_forces(self):
        assert False, "Abstract function"

    def set_internals(self, x):
        assert False, "Abstract function"

    def get_internals(self):
        assert False, "Abstract function"


class XYZ(CoordSys):
    def get_forces(self):
        forces_cartesian = Atoms.get_forces()
        transform_matrix = self.get_transform_matrix(self.internals)
        forces_opt = numpy.dot(transform_matrix, forces_cartesian)
        return forces_opt

    def get_positions(self):
        return make_like_atoms(self.__coords)

    def set_positions(self, x):
        self.__coords = x.flatten()

    def set_internals(self, x):
        return set_positions(x)

    def get_internals(self):
        return self.__coords.copy()


# OO bhaviour testing
class A():
    x = 3

class B(A):
    pass

class C(B):
    def z(self):
        A.x = 1
    def y(self):
        return A.x
