import Queue
import aof.common
import numpy as np

class Strategy:
    def generate(self, procs, topology, job_costs=None):
        """Generate a scheduling strategy
        procs: procs tuple (total, max, min)
        """

        raise False, "Abstract Function"

class Strategy_HCM_Simple(Strategy):
    def generate(self, procs, topology, jobs_count, job_costs=None):
        """Generate a scheduling strategy
        procs: procs tuple (total, max, min)
        topology: model of system cpu sets
        """

        total, max, min = procs
        cpu_ranges = []

        simtop = topology.copy()

        remaining_jobs = jobs_count
        while remaining_jobs * min > simtop.total():

            # keep filling up machine with jobs using minimum processor count
            while simtop.largest() > min:
                range = simtop.get_range(min)
                cpu_ranges.append(range)
                remaining_jobs -= 1

            simtop.reset()

        while remaining_jobs > 0:
            combinations = self.gen_all(min, max, remaining_jobs)
            assert len(combinations) == remaining_jobs

            combinations = filter(simtop.used, combinations)

            max_ix = free.argmax()

            combination = combinations[max_ix]

            for p in combination:
                range = simtop.get_range(p)
                cpu_ranges.append(range)
                remaining_jobs -= 1

            assert remaining_jobs == 0

        assert len(cpu_ranges) == jobs_count

        return cpu_ranges

class Topology(object):
    """A model of a multiprocessor system.
    
    Used to plan a scheduling sequence.
    Used to keep track of free/occupied cpus.
    
    Contents e.g.:
        [n]         Single system image, n processors.
        [a,b,c]     Multiple system images, each with a,b,c processors.


    >>> t = Topology([4])
    >>> t.get_range(2)
    (array([0, 1]), 0, array([0, 1]), 1)

    >>> print t
    [2] / [4] / [[False, False, True, True]]

    >>> t.get_range(2)
    (array([2, 3]), 0, array([2, 3]), 2)

    >>> print t
    [0] / [4] / [[False, False, False, False]]

    >>> t.put_range(2)
    >>> print t
    [2] / [4] / [[False, False, True, True]]



    >>> t = Topology([1,1])
    >>> t.get_range(2)
    >>> t.get_range(1)
    (array([0]), 0, array([0]), 1)
    >>> t.get_range(1)
    (array([1]), 1, array([0]), 2)
    >>> t.get_range(1)
    >>> print t
    [0, 0] / [1, 1] / [[False], [False]]

    >>> t = Topology(range(1,5))
    >>> t.get_range(2)
    (array([1, 2]), 1, array([0, 1]), 1)
    >>> print t
    [1, 0, 3, 4] / [1, 2, 3, 4] / [[True], [False, False], [True, True, True], [True, True, True, True]]

    >>> t.get_range(1)
    (array([0]), 0, array([0]), 2)

    >>> t.get_range(1)
    (array([6]), 3, array([0]), 3)

    >>> t.get_range(1)
    (array([3]), 2, array([0]), 4)

    >>> t.get_range(2)
    (array([4, 5]), 2, array([1, 2]), 5)


    >>> t2 = t.copy()

    >>> for i in [1, 2, 3, 4, 5]:
    ...     t2.put_range(i)
    >>> print t2
    [1, 2, 3, 4] / [1, 2, 3, 4] / [[True], [True, True], [True, True, True], [True, True, True, True]]




    """
    def __init__(self, shape):
        self.state = []
        for i in shape:
            assert i > 0
            self.state.append([True for j in range(i)])

        self.id = 0
        self._alloc = dict()

        #TODO: create lock


    def __str__(self):
        msg = "%s / %s / %s" % (self.available, self.all, self.state)
        return msg

    @property
    def available(self):
        return [sum(i) for i in self.state]

    @property
    def all(self):
        return [len(i) for i in self.state]

    
    def put_range(self, id):
        """Relinquish ownership of cpus allocated with id."""

        ix_part, ixs_local = self._alloc.pop(id)
        part = self.state[ix_part]
        for i in ixs_local:
            assert not part[i]
            part[i] = True
        
    def get_range(self, n):
        """Try to find a range of n cpus in the system.
        
        If possible, (A) returns cpus in a partition of exactly n cpus, otherwise
        (B) returns cpus in a partition with exactly n remaining cpus, otherwise
        (C) returns the range that maximises the number of leftover cpus in the 
        partition.
        """

        assert n > 0

        if n in self.available:
            if n in self.all and self.all.index(n) == self.available.index(n):
                ix_part = self.all.index(n)
            
            else:
                ix_part = self.available.index(n)

        else:
            diffs = np.array([N - n for N in self.available])

            if diffs.max() < 0:
                return None

            ix_part = diffs.argmax()


        mask = self.state[ix_part]
        assert sum(mask) >= n

        ixs_local = np.array(self.frees(mask)[:n])

        ix_global_start = sum(self.all[:ix_part])

        ixs_global = ixs_local + ix_global_start

        for i in ixs_local:
            self.state[ix_part][i] = False


        # Keep a record of allocation so that cpus can be relinquished just
        # by quoting the id.
        self.id += 1
        self._alloc[self.id] = (ix_part, ixs_local)
        return (ixs_global, ix_part, ixs_local, self.id)

    def frees(self, mask):
        """Returns indices of all free cpus in a mask."""

        return [i for i in range(len(mask)) if mask[i]]

    def calc_used(self, task_cpus):
        """Finds the number of used processors when running jobs 
        with the specified CPUs."""

        tmp = self.copy()
        tmp.reset()

        used = 0
        for tc in task_cpus:
            r = tmp.get_range(tc)

            # If ever unable to fit a job, it means the combination of cpus
            # cannot yield a valid occupation of the system, and so the total
            # used processors is zero.
            if r == []:
                return 0
            used += sum(r)

        assert used <= sum(self.all)
        return used
    
    def copy(self):
        from copy import deepcopy
        return deepcopy(self)

class SchedQueue(Queue.Queue):
    """A thread-safe queue object that comes annotates it's contents with scheduling information."""

    _sleeptime = 0.3

    def __init__(self, procs, topology=None):
        self.procs = procs
        self._topology = topology

        Queue.Queue.__init__(self)

    def put(self, item):
        
        #TODO: in here, alter schedule info

        return Queue.Queue.put(self, item)

    def get(self):
        while not self._topology.fits(item):
            time.sleep(self._sleeptime)

        return Queue.Queue.get(self, item)

if __name__ == "__main__":
    import doctest
    doctest.testmod()

