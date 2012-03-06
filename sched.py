from __future__ import with_statement
import time

import threading
import logging
import numpy as np
from config import DEFAULT_TOPOLOGY, DEFAULT_PMIN, DEFAULT_PMAX

# setup logging
lg = logging.getLogger("pts.sched")
lg.setLevel(logging.INFO)

class Item():
    def __init__(self, job, tag):
        self.job          = job
        self.tag          = tag
        self.id           = tag[3]
        self.range_global = tag[0]
        self.part_ix      = tag[1]
        self.range_local  = tag[2]
    def __str__(self):
        return str((self.job, self.tag))

class Strategy:
    """
    Wrapper around the SchedStrategy class,
    Make scheduling strategy easily available for other functions
    This wrapper gives only the Strategy and has no other functions
    which interact with the rest of the scheduling algorithm

    >>> strat = Strategy([1],1,1)
    >>> strat(4)
    [([0], 0, [0]), ([0], 0, [0]), ([0], 0, [0]), ([0], 0, [0])]

    >>> strat = Strategy([4,4],2,4)
    >>> strat(5)
    [([0, 1], 0, [0, 1]), ([2, 3], 0, [2, 3]), ([4, 5], 1, [0, 1]), ([6, 7], 1, [2, 3]), ([0, 1, 2, 3], 0, [0, 1, 2, 3])]
    >>> strat(4)
    [([0, 1], 0, [0, 1]), ([2, 3], 0, [2, 3]), ([4, 5], 1, [0, 1]), ([6, 7], 1, [2, 3])]


    FIXME: Strategy class seems a bit too complicated for what it is (now) supposed to
           do, maybe simplify or replace it

    Needed Interface: input: get topology (how many processes per node for all nodes
                                  in a list
                             pmin: minimal number of processors a single point calculation
                                   might use
                             pmax: maximal number of processors a single point calculation
                                   might use

                       Output: A list for all the n (given as input to call) tasks and how and
                               where to run them. Should be an optimized task distribution, where
                               no task has less than pmin and no more than pmax tasks, no task is distributed
                               over several nodes and there is as less as possible overhead due to empty processors

                               For every task there should be the following output:
                              distribution: a list of processor numbers, on which the task should run
                                             (global numbers)
                               node: the number of the node to run on
                               cpus: a list of cpus on the node to run on (local numbers)

                       for example: strategy for 4 tasks on two four-core machines:
                           [([0, 1], 0, [0, 1]), ([2, 3], 0, [2, 3]), ([4, 5], 1, [0, 1]), ([6, 7], 1, [2, 3])]
    """
    def __init__(self, topology = None, pmin = None, pmax = None):

        if topology == None:
           self.topstring = DEFAULT_TOPOLOGY
        else:
           self.topstring = topology

        if pmin == None:
           self.pmin = DEFAULT_PMIN
        else:
           self.pmin = pmin

        if pmax == None:
           self.pmax = DEFAULT_PMAX
        else:
           self.pmax = pmax

        self.s = SchedStrategy_HCM_Simple((self.pmax,self.pmin))
        self.top = Topology(self.topstring)

    def __call__(self, n):
        sched = self.s.generate(self.top, n)
        scheds = [r[:3] for r in sched]
        return scheds

class SchedStrategy:
    """Abstract object representing a method of placing jobs on some parallel computing infrastructure."""
    def __init__(self, procs):
        """
        procs:
            2-tuple (max, min): allowable range of processors on which to 
            schedule jobs. Intended to provide a simple mechanism to deal with 
            decreasing efficiency as the degree of parallelism increases.
        """
        self.procs = procs
        pmax, pmin = self.procs
        assert pmax >= pmin

    def generate(self, topology, job_count, job_costs=None, params=None):
        """Generate a scheduling strategy

        topology:   model of system cpu sets
        job_count:  total number of jobs to schedule
        job_costs:  relative cost of each job (optional)
        params:     any other params, instructions

        returns:
            A list of tuples of length |job_count| where each tuple describes 
            the scheduling info for each job. The tuples have the following form:

                (global cpu indices, partition index, local cpu indices, id)

                where:
                    global cpu indices:
                        list of cpu indices 
                    partition index:
                        index of partition in which job runs
                    local cpu indices:
                        list of cpu indices *within* the partition
                    id:
                        unique number per allocation to facilitate the 
                        relinquishing of groups of cpus, just by quoting the 
                        number

                    NOTE:
                        global cpu index = partition index + local cpu index
        """

        assert False, "Abstract Function"

class SchedStrategy_HCM_Simple(SchedStrategy):
    def __init__(self, procs, favour_less_parallel=True):
        SchedStrategy.__init__(self, procs)
        self.favour_less_parallel = favour_less_parallel

    def generate2(self, topology, job_count):
        """Generate a scheduling strategy"""

        """
        >>> s = SchedStrategy_HCM_Simple((4,1))
        >>> sched = s.generate2(Topology([4,4]), 8+7)
        >>> [r[0] for r in sched]

        >>> s = SchedStrategy_HCM_Simple((4,2))
        >>> sched = s.generate2(Topology([4,4]), 8+7)
        >>> [r[0] for r in sched]


        >>> s = SchedStrategy_HCM_Simple((4,2))
        >>> sched = s.generate2(Topology([4,4]), 8+6)
        >>> [r[0] for r in sched]

        >>> s = SchedStrategy_HCM_Simple((4,1))
        >>> sched = s.generate2(Topology([4,4]), 6)
        >>> [r[0] for r in sched]

        >>> s = SchedStrategy_HCM_Simple((4,1))
        >>> sched = s.generate2(Topology([4,4]), 7)
        >>> [r[0] for r in sched]

        >>> s = SchedStrategy_HCM_Simple((4,1))
        >>> sched = s.generate2(Topology([4,4]), 8)
        >>> [r[0] for r in sched]


        """


        pmax, pmin = self.procs
        cpu_ranges = []

        # create a copy of the system topology for planning purposes
        simtop = topology.copy()

        remaining_jobs = job_count

        # Keep filling up using minimum number of cpus per job (pmin), until 
        # there are not enough jobs to fill the system when taking pmin cpus
        # per job.
        while remaining_jobs > 0:


            if pmin < pmax and remaining_jobs * pmin < sum(simtop.all):
                pmin += 1
#               print "pmin",pmin
#               print "remaining_jobs", remaining_jobs

            # keep filling up machine with jobs using minimum processor count
            jobs_in_round = 0
            while max(simtop.available) >= pmin and remaining_jobs > 0:
                range = simtop.get_range(pmin)
                assert range != None
                remaining_jobs -= 1
                jobs_in_round += 1

            simtop.reset()

            best = self.find_best_fit(pmin, pmax, jobs_in_round, topology.copy())
            for p in best:
                range = simtop.get_range(p)
                assert range != None, "%s %s" % (simtop, p)
                cpu_ranges.append(range)

            simtop.reset()




        return cpu_ranges
        # Once we are unable to completely fill the system using the minimum
        # CPU count for each job, switch to a different strategy to try to
        # maximise system usage: attempt to fill the available CPUs by 
        # increasing the number of CPUs that each job is run on, as evenly
        # as possible accross all jobs.
        while remaining_jobs > 0:
            combinations = self._gen_combs(pmin, pmax, remaining_jobs)

#            lg.error("cs %s"% combinations)
            assert combinations.shape[1] == remaining_jobs

            # number of cpus left over for every combination of cpus
            leftovers = np.array([simtop.leftover(c) for c in combinations])

            # get combination which fits into available cpus the best
            min_ix = np.argmin(leftovers)
#            lg.error("ix %d"%min_ix)

            combination = combinations[min_ix]

            for p in combination:
                range = simtop.get_range(p)
#                lg.error("%s"%p)
                assert range != None, "%s %s" % (simtop, p)
                cpu_ranges.append(range)
                remaining_jobs -= 1

            assert remaining_jobs == 0

        assert len(cpu_ranges) == job_count

        return cpu_ranges

    def find_best_fit(self, pmin, pmax, jobs, simtop):

            combinations = self._gen_combs(pmin, pmax, jobs)

#            lg.error("cs %s"% combinations)
            assert combinations.shape[1] == jobs

            # number of cpus left over for every combination of cpus
            leftovers = np.array([simtop.leftover(c) for c in combinations])

            # get combination which fits into available cpus the best
            min_ix = np.argmin(leftovers)
#            lg.error("ix %d"%min_ix)

            combination = combinations[min_ix]

            return list(combination)

    def generate(self, topology, job_count):
        """Generate a scheduling strategy


        >>> procs = (2,1)
        >>> s = SchedStrategy_HCM_Simple(procs)
        >>> sched = s.generate(Topology([4]), 8)
        >>> [r[0] for r in sched]
        [[0], [1], [2], [3], [0], [1], [2], [3]]

        >>> sched = s.generate(Topology([4]), 6)
        >>> [r[0] for r in sched]
        [[0], [1], [2], [3], [0, 1], [2, 3]]

        >>> sched = s.generate(Topology([4]), 7)
        >>> [r[0] for r in sched]
        [[0], [1], [2], [3], [0], [1], [2, 3]]

        >>> s = SchedStrategy_HCM_Simple((4,1))
        >>> sched = s.generate(Topology([4]), 7)
        >>> [r[0] for r in sched]
        [[0], [1], [2], [3], [0], [1], [2, 3]]

        >>> sched = s.generate(Topology([4]), 5)
        >>> [r[0] for r in sched]
        [[0], [1], [2], [3], [0, 1, 2, 3]]

        >>> s = SchedStrategy_HCM_Simple((4,1))
        >>> sched = s.generate(Topology([4,4]), 8)
        >>> [r[0] for r in sched]
        [[0], [4], [1], [5], [2], [3], [6], [7]]

        >>> s = SchedStrategy_HCM_Simple((3,2))
        >>> sched = s.generate(Topology([4,4]), 5)
        >>> [r[0] for r in sched]
        [[0, 1], [2, 3], [4, 5], [6, 7], [0, 1, 2]]

        >>> s = SchedStrategy_HCM_Simple((4,2))
        >>> sched = s.generate(Topology([4,4]), 3)
        >>> [r[0] for r in sched]
        [[0, 1], [2, 3], [4, 5, 6]]

        >>> s = SchedStrategy_HCM_Simple((4,2))
        >>> sched = s.generate(Topology([4,4,4,4]), 30)
        >>> [r[0] for r in sched]
        [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15], [0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15], [0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15], [0, 1], [2, 3], [4, 5], [6, 7], [8, 9, 10], [12, 13, 14]]

        """

        pmax, pmin = self.procs
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
                assert range != None
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

#            lg.error("cs %s"% combinations)
            assert combinations.shape[1] == remaining_jobs

            # number of cpus left over for every combination of cpus
            leftovers = np.array([simtop.leftover(c) for c in combinations])

            # get combination which fits into available cpus the best
            min_ix = np.argmin(leftovers)
#            lg.error("ix %d"%min_ix)

            combination = combinations[min_ix]

            for p in combination:
                range = simtop.get_range(p)
#                lg.error("%s"%p)
                assert range != None, "%s %s" % (simtop, p)
                cpu_ranges.append(range)
                remaining_jobs -= 1

            assert remaining_jobs == 0

        assert len(cpu_ranges) == job_count

        return cpu_ranges

    def _gen_combs(self, pmin, pmax, n):
        """Generates all possible distributions of n jobs on between min and 
        max cpus. Order does't matter.
        
        >>> s = SchedStrategy_HCM_Simple((1,1),favour_less_parallel=False)
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

        >>> s._gen_combs(2,4,4)
        array([[2, 2, 2, 2],
               [3, 2, 2, 2],
               [3, 3, 2, 2],
               [3, 3, 3, 2],
               [3, 3, 3, 3],
               [4, 3, 3, 3],
               [4, 4, 3, 3],
               [4, 4, 4, 3],
               [4, 4, 4, 4]])

        """
        assert pmax >= pmin
        assert n > 0
        tmp = np.array([pmin for i in range(n)])

        list = [tmp]
        for i in range(pmin, pmax):
            for j in range(n):
                tmp = tmp.copy()
                tmp[j] += 1

                # Sort jobs in order of increasing number of CPUs.
                # In this way, longer running jobs i.e. those allocated to fewer 
                # CPUs (under the assumption that all jobs will consumer the same 
                # amount of total CPU time), will run first, and the more parallel
                # jobs will be run later.
                if self.favour_less_parallel:
                    list.append(tmp[::-1])
                else:
                    list.append(tmp)
        return np.array(list)
                

class Topology(object):
    """A model of a multiprocessor system.
    
    Used to plan a scheduling sequence.
    Used to keep track of free/occupied cpus.
    Used to record statistics on system usage, etc. (eventually)

    *Functions return lists, but NumPy arrays used internally.
    
    Contents e.g.:
        [n]         Single system image, n CPUs; e.g. shared memory SMP
        [a,b,c,...] Multiple system images, with a,b,c,... CPUs, e.g. a cluster


    TODO: 
     - add test that upon get_range failure, state doesn't change
     - split functionality => higher no processors first/last
       - adjust test cases accordingly

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

    # Test writing of stuff to file
    >>> import tempfile
    >>> f = tempfile.TemporaryFile()
    >>> t = Topology([1], f_timing=f)
    >>> r = t.get_range(1)
    >>> time.sleep(0.1)
    >>> t.put_range(r[3])
    >>> time.sleep(0.05)
    >>> r = t.get_range(1)
    >>> f.seek(0)
    >>> s = f.read()
    >>> sp = s.split("\\n")
    >>> len(sp)
    4


    """
    def __init__(self, shape, f_timing=None):
        self.state = []
        for i in shape:
            assert i > 0
            self.state.append([True for j in range(i)])

        self.id = 0
        self._alloc = dict()

        self._lock = threading.RLock()

        # recording stats?
        self.f_timing = f_timing
        if type(f_timing) == file:
            self.start_time = time.time()

    def __str__(self):
        msg = "%s / %s / %s" % (self.available, self.all, self.state)
        return msg

    @property
    def available(self):
        """List of numbers of available CPUs in each partition."""
        return [sum(i) for i in self.state]

    @property
    def all(self):
        """List of total numbers of CPUs in each partition."""
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
        8

        >>> t.leftover([1 for i in range(9)])
        8

        >>> t1 = Topology([4,3,2])
        >>> t2 = Topology([2,3,4])
        >>> list = [[1,2,3],[3,2,1],[3,1,1],[1,3,3],[2,2,2],[1,1,1],[2,3,3],[3,2,4]]
        >>> l1 = [t1.leftover(c) for c in list]
        >>> l2 = [t2.leftover(c) for c in list]
        >>> l1 == l2
        True

        """
        
        task_cpus = list(task_cpus)
#        if reverse:
#            decreasing.sort(reverse=True)

        tmp = self.copy()
        for n in task_cpus:
            assert n > 0
            if tmp.get_range(n) == None:
                return sum(tmp.all)

        return sum(tmp.available)

    def put_range(self, id):
        """Relinquish ownership of cpus allocated with id."""

        with self._lock:
            if self.f_timing:
                self._record()

            ix_part, ixs_local = self._alloc.pop(id)
            part = self.state[ix_part]
            for i in ixs_local:
                assert not part[i]
                part[i] = True

    def _record(self):
        assert self.f_timing
        t = time.time() - self.start_time
        s = "%.3f\t%d\t%s\n" % (t, sum(self.available), self)
        self.f_timing.write(s)
        self.f_timing.flush()
 
    def get_range(self, n):
        """Try to find a range of n cpus in the system.
        
        If possible, (A) returns cpus in a partition of exactly n cpus, otherwise
        (B) returns cpus in a partition with exactly n remaining cpus, otherwise
        (C) returns the range that maximises the number of leftover cpus in the 
        partition, otherwise (D) returns None, indicating failure.

        Returns a 4-tuple (ixs_global, ix_part, ixs_local, id)
            ixs_global: list of global cpu indices in system
            ix_part:    index of the system partition/node node
            ixs_local:  list of cpu indices within a partition/node
            id:         unique number for every allocation made, intended to
                        facilitate simple relinquishment of cpu ranges.
        """

        assert n > 0

        with self._lock:
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

#            lg.error("l n=%d %s %s" % (n,ix_part, ixs_local.tolist()))
            if self.f_timing:
                self._record()

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

        assert False, "I don't use this function I think...?"
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
        with self._lock:
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

        with self._lock:
            shape = self.all
            self.state = []
            for i in shape:
                self.state.append([True for j in range(i)])


if __name__ == "__main__":
    import doctest
    doctest.testmod()


