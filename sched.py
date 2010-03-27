from __future__ import with_statement
import time
from collections import deque

import random
import threading
import thread
from Queue import Queue
import time
import logging
import numpy as np

from aof.common import Job, QCDriverException, Result, is_same_e, is_same_v, ERROR_STR
import aof

# setup logging
lg = logging.getLogger("aof.sched")
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

        assert False, "Abstract Function"

class SchedStrategy_HCM_Simple(SchedStrategy):
    def __init__(self, procs, favour_less_parallel=True):
        SchedStrategy.__init__(self, procs)
        self.favour_less_parallel = favour_less_parallel

    def generate2(self, topology, job_count):
        """Generate a scheduling strategy


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
                print "pmin",pmin
                print "remaining_jobs", remaining_jobs

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
    """
    """
    
    >>> sq = SchedQueue(([4],2,1))
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

    _sleeptime = 0.01

    def __init__(self, processors, sched_strategy=None):
#        lg.info(self.__class__.__name__ + ": Topology: %s CPUs: %s" % (topology, procs))
        topology, max_CPUs, min_CPUs = processors
        procs = max_CPUs, min_CPUs

        f = open("cpu_occupation_timing.txt", "a")
        self._topology = Topology(topology, f_timing=f)
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
    def __len__(self):
        with self._lock:
            return len(self._deque)


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

            return Item(job, range)

def test_SchedQueue(cpus, thread_count, items):
    """Launches a whole lot of threads to test the functionality of SchedQueue.
    """
    """

    cpus:         topology, max, min numbers of cpus per job
    thread_count: total threads to launch
    items:        items to process, can be any list, basically ignored
    
    >>> test_SchedQueue(([4],2,1), 8, range(100))
    >>> test_SchedQueue(([4,4,2,1], 2,1), 100, range(1000))
    >>> test_SchedQueue(([4,4,2,1], 5,1), 100, range(1000))
    >>> test_SchedQueue(([1,1,1,2,3,4,5,6,7], 3,2), 200, range(1000))
    >>> test_SchedQueue(([4,4,4,4,4], 4,2), 100, range(1000))

    """

    import random

    sq = SchedQueue(cpus)
    finished = Queue()

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

    assert input == output
    assert sq._topology.all == sq._topology.available

class SchedException(Exception):
    def __init__(self, msg):
        self.msg = msg 
    def __str__(self, msg):
        return self.msg

class ParaJobLauncher():
    def run(self, j, ix):
        params = dict()
        params["placement_command"] = mol_interface.placement_progam + \
                                      " " + \
                                      mol_interface.placement_arg
        mol_interface.run_job(j.v, params)

def hugh_pmap(f, l, processors):
    """Parallel map using the scheduling infrastructure in this module."""
    assert False, "work in progress"

    ps = ParaSched(f, procesors)
    ps.batch_run(l)
    return ps.get_results()
    

class ParaSched(object):
    """Manages the threads responsible for running programs that perform 
    electronic structure calculations. Builds queues and generates sceduling
    info.
    """
    """

    >>> ps = ParaSched(aof.pes.GaussianPES(fake_delay=0.2), ([8], 3, 1))
    >>> vecs = np.arange(12).reshape(-1,2)
    >>> jobs = [Job(i,[]) for i in vecs]
    >>> ps.batch_run(jobs)
    >>> res = np.array([r.g for r in ps.get_results()]).flatten()
    >>> correct = np.array([aof.pes.GaussianPES().gradient(v) for v in vecs]).flatten()
    >>> correct.sort()
    >>> res.sort()
    >>> (res == correct).all()
    True

    """
    def __init__(self, qc_driver, processors):
        """Constructor.
        
        qc_driver:
            object with a run() function
        processors:
            3-tuple: ([N1,N2,...], max_CPUs, min_CPUs)
            i.e. shape of system, max CPUs per job, min CPUs per job
        """
        
        # pending queue contains self-generates scheduling info
        self._pending = SchedQueue(processors)
        self._finished = Queue()
        self._qc_driver = qc_driver

        sys_shape, _, min_cpus = processors

        # no of workers to start
        self._workers_count = sum(sys_shape) / min_cpus

    def __worker(self, pending, finished, ix):
        """Runs as a Python thread. "pending" and "finished" are both thread
        safe queues of jobs to be consumed / added to by each worker. ix
        is the index of each worker, used in node placement."""

        my_id = thread.get_ident()

        lg.info("Worker starting, id = %s ix = %s" % (my_id, ix))

        while not pending.empty():

            try:
                item = pending.get()
            except SchedQueueEmpty, sqe:
#                lg.error(msg)
                break
            """except Exception, e:
                msg = ' '.join(["Worker", str(my_id), "threw", str(e)])
                lg.error(msg)
                return"""

            # just for testing what happens when a worker thread experiences an exception
            # if ix % 2 == 0:
            #   raise Exception("Dummy")

            # setup parameter dictionary
            params = dict()

            # call quantum chem driver
            try:
                res = self._qc_driver.run(item)
            except QCDriverException, inst:
                # TODO: perhaps this needs to be done differently, when a worker 
                # encounters an exception it should empty the queue rethrow, 
                # and maybe kill all running QC jobs.
                l = ["Worker", str(my_id), ": Exception thrown when", 
                     "calling self.__qc_driver.run(item):", str(type(inst)),
                     ":", str(inst.args)]
                msg = ' '.join(l)
                
                res = Result(item.v, 0.0, flags = dict(ERROR_STR = msg))

            finished.put(res)
            lg.info("Worker %s finished a job." % my_id)
            pending.task_done(item.id)
            lg.debug("thread " + str(my_id) + ": item " + str(item) + " complete: " + str(res))

        lg.info("Queue empty, worker %s exiting." % my_id)

    def batch_run(self, jobs_list):
        """Start threads to process jobs in queue."""

        # place jobs in a queue
        self._pending.put(jobs_list)

#        lg.info("%d jobs in pending queue" % len(self._pending))
        # start workers
        lg.info("%s spawning %d worker threads" % (self.__class__.__name__, self._workers_count))
        for i in range(self._workers_count):
            t = threading.Thread(target=self.__worker, args=(self._pending, self._finished, i))
            t.daemon = True
            t.start()

        # The normal method would be to join() the thread, but if one of the 
        # threads dies then we get deadlock.
        # this is a bit crude, can it be done better?
        while threading.activeCount() > 1:
            time.sleep(0.3333)

        lg.debug("All worker threads exited")

        if not self._pending.empty():
            raise SchedException("Pending queue not empty but all threads exited.")

    def get_results(self):
        results = []
        while not self._finished.empty():
            results.append(self._finished.get())
            self._finished.task_done()

        return results

if __name__ == "__main__":
    import doctest
    doctest.testmod()


