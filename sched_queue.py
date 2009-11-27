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
    """
    def __init__(self, shape):
        self.state = []
        for i in shape:
            self.state.append([False for j in range(i)])

    @property
    def filled(self):
        return [sum(i) for i in self.state]

    @property
    def all(self):
        return [len(i) for i in self.state]

    
    def get_range(self, n):
        """Try to find a range of n cpus in the system.
        
        Returns the range that maximises the number of leftover cpus in the 
        partition.
        """

        diffs = np.array([N - n for N in self.filled])
        partition_ix = diffs.argmax()

        mask = self.state[partition_ix]
        assert sum(mask) >= n

        local_range = np.array(self.get_frees(mask)[:n])

        global_range = local_range + sum(self.all[:partition_ix+1])

        return global_range

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

    def get_free_cpus(self):
        """Return a list of the total number of free slots in each system partition."""

        return self.all - self.filled

    #TODO: define getters and setters?

class SchedQueue(Queue.Queue):

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
