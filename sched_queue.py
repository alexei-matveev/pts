from __future__ import with_statement
import Queue
import numpy as np
import threading

import aof.common

class SchedStrategy:
    def __init__(self, procs):
        """
        procs: procs tuple (total, max, min)
        """
        self.procs = procs
        ptotal, pmax, pmin = self.procs
        assert ptotal >= pmax >= pmin

    def generate(self, topology, job_costs=None):
        """Generate a scheduling strategy

        topology:   model of system cpu sets
        job_count:  total number of jobs to schedule
        job_costs:  relative cost of each job (optional)
        """

        raise False, "Abstract Function"

class SchedStrategy_HCM_Simple(SchedStrategy):
    def generate(self, topology, job_count, job_costs=None):
        """Generate a scheduling strategy

        >>> procs = (4,2,1)
        >>> s = SchedStrategy_HCM_Simple(procs)
        >>> sched = s.generate(Topology([4]), 8)
        >>> [r[0] for r in sched]
        [[0], [1], [2], [3], [0], [1], [2], [3]]

        >>> sched = s.generate(Topology([4]), 6)
        >>> [r[0] for r in sched]
        [[0], [1], [2], [3], [0, 1], [2, 3]]

        >>> sched = s.generate(Topology([4]), 7)
        >>> [r[0] for r in sched]
        [[0], [1], [2], [3], [0, 1], [2], [3]]

        >>> s = SchedStrategy_HCM_Simple((4,4,1))
        >>> sched = s.generate(Topology([4]), 7)
        >>> [r[0] for r in sched]
        [[0], [1], [2], [3], [0, 1], [2], [3]]

        >>> sched = s.generate(Topology([4]), 5)
        >>> [r[0] for r in sched]
        [[0], [1], [2], [3], [0, 1, 2, 3]]

        >>> s = SchedStrategy_HCM_Simple((8,4,1))
        >>> sched = s.generate(Topology([4,4]), 8)
        >>> [r[0] for r in sched]
        [[0], [4], [1], [5], [2], [3], [6], [7]]

        >>> s = SchedStrategy_HCM_Simple((8,3,2))
        >>> sched = s.generate(Topology([4,4]), 5)
        >>> [r[0] for r in sched]
        [[0, 1], [2, 3], [4, 5], [6, 7], [0, 1, 2]]

        >>> s = SchedStrategy_HCM_Simple((8,4,2))
        >>> sched = s.generate(Topology([4,4]), 3)
        >>> [r[0] for r in sched]
        [[0, 1, 2], [4, 5], [6, 7]]

        >>> s = SchedStrategy_HCM_Simple((16,4,2))
        >>> sched = s.generate(Topology([4,4,4,4]), 30)
        >>> [r[0] for r in sched]
        [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15], [0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15], [0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15], [0, 1, 2], [4, 5, 6], [8, 9], [10, 11], [12, 13], [14, 15]]

     """

        if job_costs != None:
            assert len(job_costs) == job_count

        ptotal, pmax, pmin = self.procs
        assert ptotal == sum(topology.all)
        cpu_ranges = []

        # create a copy of the system topology for planning purposes
        simtop = topology.copy()

        remaining_jobs = job_count

        # Keep filling up using minimum number of cpus per job (pmin), until 
        # there are not enough jobs to fill the system when taking pmin cpus
        # per job.
        while remaining_jobs * pmin >= sum(simtop.all):

            # keep filling up machine with jobs using minimum processor count
            while max(simtop.available) >= pmin:
                range = simtop.get_range(pmin)
                cpu_ranges.append(range)
                remaining_jobs -= 1

            simtop.reset()

        # Once we are unable to completely fill the system using the minimum
        # CPU count for each job, switch to a different strategy to try to
        # maximise system usage: attempt to fill the available CPUs by 
        # increasing the number of CPUs that each job is run on, as evenly
        # as possible accross all jobs.
        while remaining_jobs > 0:
            combinations = self._gen_combs(pmin, pmax, remaining_jobs)
            assert combinations.shape[1] == remaining_jobs

            # number of cpus left over for every combination of cpus
            leftovers = np.array([simtop.leftover(c) for c in combinations])

            # get combination which fits into available cpus the best
            min_ix = np.argmin(leftovers)
            combination = combinations[min_ix]

            for p in combination:
                range = simtop.get_range(p)
                cpu_ranges.append(range)
                remaining_jobs -= 1

            assert remaining_jobs == 0

        assert len(cpu_ranges) == job_count

        return cpu_ranges

    def _gen_combs(self, pmin, pmax, n):
        """Generates all possible distributions of n jobs on between min and 
        max cpus. Order does't matter.
        
        >>> s = SchedStrategy_HCM_Simple((1,1,1))
        >>> s._gen_combs(2,2,2)
        array([[2, 2]])

        >>> s._gen_combs(1,3,4)
        array([[1, 1, 1, 1],
               [2, 1, 1, 1],
               [2, 2, 1, 1],
               [2, 2, 2, 1],
               [2, 2, 2, 2],
               [3, 2, 2, 2],
               [3, 3, 2, 2],
               [3, 3, 3, 2],
               [3, 3, 3, 3]])

        >>> s._gen_combs(2,3,2)
        array([[2, 2],
               [3, 2],
               [3, 3]])
        
        >>> s._gen_combs(3,2,0)
        Traceback (most recent call last):
            ...
        AssertionError

        """
        assert pmax >= pmin
        assert n > 0
        tmp = np.array([pmin for i in range(n)])

        list = [tmp]
        for i in range(pmin, pmax):
            for j in range(n):
                tmp = tmp.copy()
                tmp[j] += 1
                list.append(tmp)
        return np.array(list)
                

class Topology(object):
    """A model of a multiprocessor system.
    
    Used to plan a scheduling sequence.
    Used to keep track of free/occupied cpus.

    *Functions return lists, but NumPy arrays used internally.
    
    Contents e.g.:
        [n]         Single system image, n CPUs; e.g. shared memory SMP
        [a,b,c,...] Multiple system images, with a,b,c,... CPUs, e.g. a cluster


    >>> t = Topology([4])
    >>> t.get_range(2)
    ([0, 1], 0, [0, 1], 1)

    >>> print t
    [2] / [4] / [[False, False, True, True]]

    >>> t.get_range(2)
    ([2, 3], 0, [2, 3], 2)

    >>> print t
    [0] / [4] / [[False, False, False, False]]

    >>> t.put_range(2)
    >>> print t
    [2] / [4] / [[False, False, True, True]]



    >>> t = Topology([1,1])
    >>> t.get_range(2)
    >>> t.get_range(1)
    ([0], 0, [0], 1)
    >>> t.get_range(1)
    ([1], 1, [0], 2)
    >>> t.get_range(1)
    >>> print t
    [0, 0] / [1, 1] / [[False], [False]]

    >>> t = Topology(range(1,5))
    >>> t.get_range(2)
    ([1, 2], 1, [0, 1], 1)
    >>> print t
    [1, 0, 3, 4] / [1, 2, 3, 4] / [[True], [False, False], [True, True, True], [True, True, True, True]]

    >>> t.get_range(1)
    ([0], 0, [0], 2)

    >>> t.get_range(1)
    ([6], 3, [0], 3)

    >>> t.get_range(1)
    ([3], 2, [0], 4)

    >>> t.get_range(2)
    ([4, 5], 2, [1, 2], 5)


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

        self.lock = threading.RLock()

    def __str__(self):
        msg = "%s / %s / %s" % (self.available, self.all, self.state)
        return msg

    @property
    def available(self):
        return [sum(i) for i in self.state]

    @property
    def all(self):
        return [len(i) for i in self.state]

    def leftover(self, task_cpus):
        """Calculates the number of leftover cpus when tasks with the numbers 
        of cpus in task_cpus are all placed in the system, biggest first. Returns 
        the total number of cpus if the requested job sizes don't fit.

        >>> t = Topology([1])
        >>> t.leftover([1,2,3])
        1
        >>> t = Topology([1,1,1])
        >>> t.leftover([1,2,1])
        3
        >>> t.leftover([1,1,1])
        0
        >>> t = Topology([4,4])
        >>> t.leftover([1,1,2])
        4

        >>> t.leftover([1,1,5])
        8

        >>> t.leftover([1,1,4])
        2

        >>> t.leftover([1 for i in range(9)])
        8

        """
        
        decreasing = list(task_cpus)
        decreasing.sort(reverse=True)

        tmp = self.copy()
        for n in decreasing:
            assert n > 0
            if tmp.get_range(n) == None:
                return sum(tmp.all)

        return sum(tmp.available)

    def put_range(self, id):
        """Relinquish ownership of cpus allocated with id."""

        with self.lock:
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

        Returns a 4-tuple (ixs_global, ix_part, ixs_local, id)
            ixs_global: list of global cpu indices in system
            ix_part:    index of the system partition/node node
            ixs_local:  list of cpu indices within a partition/node
            id:         unique number for every allocation made, intended to
                        facilitate simple relinquishment of cpu ranges.
        """

        assert n > 0

        with self.lock:

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

            return (ixs_global.tolist(), ix_part, ixs_local.tolist(), self.id)

    @staticmethod
    def frees(mask):
        """Returns indices of all free cpus in mask. True implies free.
        
        >>> Topology.frees([False, True, True, False])
        [1, 2]

        >>> Topology.frees([False, False])
        []

        """

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
        with self.lock:
            # Cannot do a deepcopy of the whole object because locks cannot 
            # be copied. (Is there a way around this?)
            new = Topology(self.all)
            new.state = deepcopy(self.state)
            new.id = self.id
            new._alloc = self._alloc.copy()
        return new

    def reset(self):
        """Resets Topology object's internal state to totally unoccupied. 
        Allocation dictionary remains unchanged.
        
        >>> t = Topology([3,2,1])
        >>> r0 = t.get_range(2)
        >>> t.reset()
        >>> r1 = t.get_range(2)
        >>> r1[0] == r0[0]
        True
        """

        with self.lock:
            shape = self.all
            self.state = []
            for i in shape:
                self.state.append([True for j in range(i)])
       

class SchedQueue(Queue.Queue):
    """A thread-safe queue object that annotates it's contents with scheduling information."""

    _sleeptime = 0.3

    def __init__(self, procs, topology):
        self._procs = procs
        self._topology = Topology(topology)

        Queue.Queue.__init__(self) # do I want this?

        self.lock = threading.RLock()

    def put(self, items):
        if type(items) == np.ndarray:
            items = items.tolist()
        elif type(items) != list:
            items = [items]
        
        #TODO: in here, alter schedule info

        return Queue.Queue.put(self, item)

    def get(self):
        while not self._topology.fits(item):
            time.sleep(self._sleeptime)

        return Queue.Queue.get(self, item)

if __name__ == "__main__":
    import doctest
    doctest.testmod()

