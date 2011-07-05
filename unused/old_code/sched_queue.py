from __future__ import with_statement
import time
import Queue
import numpy as np
import threading
from collections import deque

assert "False", "Deprecated (02/12/2009): this code has been moved to sched.py"

class SchedStrategy:
    def __init__(self, procs):
        """
        procs: procs 2-tuple (max, min)
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
        """

        raise False, "Abstract Function"

class SchedStrategy_HCM_Simple(SchedStrategy):
    def generate(self, topology, job_count, favour_less_parallel=True):
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

            # Sort jobs in order of increasing number of CPUs.
            # In this way, longer running jobs i.e. those allocated to fewer 
            # CPUs (under the assumption that all jobs will consumer the same 
            # amount of total CPU time), will run first, and the more parallel
            # jobs will be run later.
            if favour_less_parallel:
                combination.sort()

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
        
        >>> s = SchedStrategy_HCM_Simple((1,1))
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

    """
    def __init__(self, shape):
        self.state = []
        for i in shape:
            assert i > 0
            self.state.append([True for j in range(i)])

        self.id = 0
        self._alloc = dict()

        self._lock = threading.RLock()

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

        with self._lock:
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

class SchedQueueEmpty(Exception):
    "SchedQueue was empty."
    pass

class SchedQueue():
    """A thread-safe queue object that 
        - annotates it's contents with scheduling information
        - keeps track of which processors in the system are free via an 
          internal model
        - re-generate scheduling info when new jobs are added to the queue

    Calls to get() will block until the internal model of the cluster 
    indicates that there are sufficient CPUs free to run the net job.
    
    >>> sq = SchedQueue((2,1), [4])
    >>> sq.empty()
    True

    >>> sq.put(1)
    >>> sq.put(2)
    >>> sq.put([3,4,5,6,7])

    >>> l = []
    >>> l += [sq.get(), sq.get()]
    >>> l += [sq.get(), sq.get()]
    >>> for id in [li.id for li in l]:
    ...     sq.task_done(id)
    >>> l = []
    >>> l += [sq.get(), sq.get(), sq.get()]
    >>> print sq
    [0] / [4] / [[False, False, False, False]]
    >>> sq.task_done(100)
    Traceback (most recent call last):
        ...
    KeyError: 100

    """

    class Item():
        def __init__(self, job, tag):
            self.job = job
            self.tag = tag
            self.id = tag[3]
            self.range_global = tag[0]
            self.part_ix      = tag[1]
            self.range_local  = tag[2]
        def __str__(self):
            return str((self.job, self.tag))

    _sleeptime = 0.01

    def __init__(self, procs, topology, sched_strategy=None):
        self._topology = Topology(topology)
        if sched_strategy == None:
            self._sched_strategy = SchedStrategy_HCM_Simple(procs)
        else:
            self._sched_strategy = sched_strategy

        self._deque = deque()
        self._sched_info = deque()
        self._lock = threading.RLock()

    def __str__(self):
        with self._lock:
            s = str(self._topology)
            for i, j in zip(self._deque, self._sched_info):
                s += '\n' + str(i) + ': ' + str(j)
            return s

    def put(self, items):
        """Places an item, or list of items, into the queue, and update the 
        scheduling information based on (a) the total number of items now in 
        the queue and (b) the currently in-use cpu set."""

        if type(items) == np.ndarray:
            items = items.tolist()
        elif type(items) != list:
            items = [items]
        
        with self._lock:
            # add
            for i in items:
                self._deque.append(i)

            # generate scheduling info
            jobs = len(self._deque)
            cpu_ranges = self._sched_strategy.generate(self._topology, jobs)

            # decorate tasks with their allocated range of CPUs
            self._sched_info = deque(cpu_ranges)

            assert len(self._deque) == len(self._sched_info)


    def task_done(self, id):
        assert type(id) == int
        self._topology.put_range(id)

    def empty(self):
        """Return True if and only the queue is empty."""
        with self._lock:
            return len(self._deque) == 0

    def get(self):
        """Gets an item from the queue"""
        with self._lock:
            if len(self._deque) == 0:
                raise SchedQueueEmpty()

            job = self._deque.popleft()
            range = self._sched_info.popleft()
            cpus = len(range[0])

            # A noteworthy design decision has been made here...
            # Instead of respecting the precise system CPU indices that were
            # originally generated for this job, only the *total number* is
            # used, in case we can find a better slot using the current 
            # occupation of the system.

            range = self._topology.get_range(cpus)
            while not range:
                time.sleep(self._sleeptime)
                range = self._topology.get_range(cpus)

            assert len(self._deque) == len(self._sched_info)

            return self.Item(job, range)

def test_SchedQueue(cpus, topology, thread_count, items):
    """Launches a whole lot of threads to test the functionality of SchedQueue.

    cpus:         max, min numbers of cpus per job
    topology:     list of processors in each system image
    thread_count: total threads to launch
    items:        items to process, can be any list, basically ignored
    
    >>> test_SchedQueue((2,1), [4], 8, range(100))
    >>> test_SchedQueue((2,1), [4,4,2,1], 100, range(1000))
    >>> test_SchedQueue((5,1), [4,4,2,1], 100, range(1000))
    >>> test_SchedQueue((3,2), [1,1,1,2,3,4,5,6,7], 200, range(1000))
    >>> test_SchedQueue((4,2), [4,4,4,4,4], 100, range(1000))

    """

    import random

    sq = SchedQueue(cpus, topology)
    finished = Queue.Queue()

    def worker(inq, outq):
        while not inq.empty():
            try:
                j = inq.get()
            except SchedQueueEmpty, sqe:
                return
            time.sleep(0.01*random.random())
            inq.task_done(j.id)
            outq.put(j.job)

    sq.put(items)

    for i in range(thread_count):
        t = threading.Thread(target=worker, args=(sq,finished))
        t.daemon = True
        t.start()

    while threading.activeCount() > 1: # this is a bit crude, can it be done better?
        time.sleep(0.1)

    input = list(items)
    input.sort()
    output = list(finished.queue.__iter__())
    output.sort()

#    print input
#    print output
    assert input == output
    assert sq._topology.all == sq._topology.available
#    print sq



if __name__ == "__main__":
    import doctest
    doctest.testmod()

