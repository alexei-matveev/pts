import numpy as np

"""Class for recording the hsitory of a Chain of State optimisation."""

class Record():
    """A single record in the history of a Chain of State.
    
    >>> r = Record([1,2], [[1,2],[2,3]], [[0,0],[0.01,0]], [[1,2],[3,4]])

    >>> r.maxix
    1

    >>> r.max
    array([2, 3])

    >>> r.maxixs
    array([0, 1])

    """
    def __init__(self, es, state, perp_forces, para_forces, ts_estim):

        assert len(es) == len(state) == len(perp_forces) == len(para_forces)

        self.es = np.array(es)
        self.state = np.array(state)
        self.perp_forces = np.array(perp_forces)
        self.para_forces = np.array(para_forces)
        self.bead_count = len(self.es)

        self.maxix = self.es.argmax()
        self.e = self.es.sum()
        self.max = self.state[self.maxix]

        self.maxixs = self.es.argsort()

        self.ts_estim = ts_estim

    def highest(self, n):
        return self.state.take(self.maxixs[:-n])

    def perp_forces_highest(self, n):
        return self.highest(n).max()
    
    def __str__(self):
        return str(self.state)
        

class History():
    """The history of a Chain of State.
    
    >>> r1 = Record([1,4], [[0,1,2],[0,2,3]], [[0,0,0.1],[0.01,0,0]], [[1,2,1],[3,4,1]])
    >>> r2 = Record([2,4], [[1,2,2],[0,2,3]], [[0,0,0],[0.0,0,0]], [[1,2,1],[3,4,1]])

    >>> h = History()
    >>> h.rec(r1)
    >>> h.rec(r2)

    >>> h.step(2,1)
    array([[0, 0, 0],
           [0, 0, 0]])

    >>> h.step(2,2)
    array([[1, 1, 0],
           [0, 0, 0]])

    >>> h.bead_count(2)
    [2, 2]

    """
    def __init__(self):
        self.list = []

    def rec(self, r):
        """Records a snapshot."""
        if r.__class__.__name__ != 'Record':
            r = Record(*r)
        assert r.__class__.__name__ == 'Record'

        self.list.append(r)

    def __len__(self):
        return len(self.list)

    def __getattr__(self, name):
        """
        Brings through fields of Records and allows them to be accessed as,
        e.g. history.bead_count(n) where |n| specifies to return the last n
        values in the history.
        """
        if len(self.list) == 0:
            return lambda n: []

        if name in self.list[0].__dict__:
            return lambda n: [getattr(r, name) for r in self.list[-n:]]

    def step(self, n, recs):
        """Returns the array of total step sizes of the |n| highest energy 
        beads over the last |recs| records. If n == 0, then return step sizes 
        of the transition state estim.
        """
        assert n >= 0

        if self.list == []:
            return np.array([0])

        tmp = []
        if n == 0:
            for r in self.list[-recs:]:
                tmp.append(r.ts_estim[1])
        else:
            # use max bead ix's from most recent record
            ixs = self.list[-1].maxixs[-n:]

            for r in self.list[-recs:]:
                tmp.append(r.state.take(ixs, axis=0))

        tmp = np.array(tmp)

        mins = tmp.min(axis=0)#.flatten()
        maxs = tmp.max(axis=0)#.flatten()

        return np.abs(maxs-mins)
            

# Testing the examples in __doc__strings, execute
# "python gxmatrix.py", eventualy with "-v" option appended:
if __name__ == "__main__":
    import doctest
    doctest.testmod()

# You need to add "set modeline" and eventually "set modelines=5"
# to your ~/.vimrc for this to take effect.
# Dont (accidentally) delete these lines! Unless you do it intentionally ...
# Default options for vim:sw=4:expandtab:smarttab:autoindent:syntax


