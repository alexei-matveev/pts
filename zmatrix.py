class ZMatrix:
    def __init__(self, string=None,data_struct=None):
        pass

    def to_internal(self):
        # vector3 n,nn,v1,v2,v3,avec,bvec,cvec;
        
        r = numpy.float64(0)
        sum = numpy.float64(0)

        for atom in atoms:
            if atom.a == None:
                atom.vector = numpy.zeros(3)
            else:
                avec = atom.a.vector
                dst = atom.dst
