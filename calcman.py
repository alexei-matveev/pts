import logging

from pts.sched import ParaSched
from pts.common import Job, QCDriverException, Result, is_same_e, is_same_v, ERROR_STR, vec_summarise
import pts
import numpy as np
import pickle

# setup logging
lg = logging.getLogger("pts.calcman")
lg.setLevel(logging.INFO)

class CalcManagerException(Exception):
    pass


class CalcManager():
    """
    Memorizes the results of previous calculations and, depending on the 
    closeness of requested calculations to previous ones either (a) runs a 
    whole new calculation, (b) interpolates existing data or (c) returns 
    existing data unmodified.

    Only (a) and (c) implemented at present.


    >>> cm = CalcManager(pts.pes.GaussianPES())
    """

    def __init__(self, qc_driver, processors=None, to_cache=None, from_cache=None):

        self.qc_driver = qc_driver
        self.__para_sched = None

        if processors:
            shape, max_cpus, min_cpus = processors
            cpus = [max_cpus, min_cpus]
            if max(shape) < max(cpus):
                lg.warn("Max number of CPUs larger than max CPUs per node: %d > %d" % (max(cpus), max(shape)))
            lg.info("CPUs: %s QC Driver: %s", processors, qc_driver)
            self.__para_sched = ParaSched(qc_driver, processors)

        self.__pending_jobs = []

        try:
            if from_cache != None:
                self.__result_dict = pickle.load(open(from_cache))
                lg.info("Loading previously ResultDict from " + from_cache)
            else:
                self.__result_dict = ResultDict()
            self.__to_cache = to_cache

            # dump empty ResultDict to make sure the file is writable
            if to_cache != None:
                pickle.dump(self.__result_dict, open(to_cache, 'w'), protocol=2)
                lg.info("Storing ResultDict in " + to_cache)
        except IOError, msg:
            raise CalcManagerException(msg)

    def __str__(self):
        s = self.__class__.__name__ + ":"
        for j in self.__pending_jobs:
            s += j.__str__() + "\n"
        return s

    def request_energy(self, v, bead_ix):
        self.request_job(v, 'E', bead_ix)

    def request_gradient(self, v, bead_ix):
        self.request_job(v, 'G', bead_ix)

    def request_job(self, v, type, bead_ix):
        """Place into queue a request for calculations of type |type|, 
        corresponding to bead index |bead_ix|.
        """

        result = self.__result_dict.get(v)

        # calculation not already performed
        if result == None: 
            # check jobs already in current pending list
            for i in range(len(self.__pending_jobs)):
                j = self.__pending_jobs[i]
                if j.geom_is(v):
                    j.add_calc(type)
                    return

        # calculation has aleady been performed, will use cached version
        elif result.has_field(type):
            lg.info("Already have result for %s, bead %d using cached version." % (vec_summarise(v), bead_ix))
            return

        # find dir containing previous calc to use as guess for wavefunction
#        closest = self.__result_dict.get_closest(v)
        dir = None
#        if closest != None:
#           dir = closest.dir

        # calc is not already in list so must add
        # gives also num (number of bead, take care for growingstring)
        # as a hand over to the job ______AN
        j = Job(v, type, bead_ix=bead_ix, prev_calc_dir=dir)
        self.__pending_jobs.append(j)
        lg.debug("Requesting job " + str(j))

    def proc_requests(self):
        """Process all jobs in queue."""

        s_jobs = ' '.join([str(j) for j in self.__pending_jobs])
        lg.info(self.__class__.__name__ + "Queued Jobs (%d) %s" % (len(self.__pending_jobs), s_jobs))

        if self.__para_sched != None:
            self.__para_sched.batch_run(self.__pending_jobs)
            self.__pending_jobs = []

            for r in self.__para_sched.get_results():
                if ERROR_STR in r.flags:
                    raise CalcManagerException("Error encountered in computation, result was: " + r)
                self.__result_dict.add(r.v, r)
#            lg.info("%d results in result dictionary" % len(self.__result_dict))

        # running serially
        else:
            lg.info(self.__class__.__name__ + " operating serially")

            for j in self.__pending_jobs:
                res = self.qc_driver.run(j)
                self.__result_dict.add(j.v, res)
                self.__pending_jobs = [] # added 03/12/2009

        if self.__to_cache != None:
            pickle.dump(self.__result_dict, open(self.__to_cache, 'w'), protocol=2)

    def eg_counts(self):
        return self.__result_dict.eg_counts()

    def energy(self, v):
        """Returns the already computed energy of vector v."""
        res = self.__result_dict.get(v)
        if res == None:
            raise CalcManagerException("No result found for vector %s (energy requested)." %v)

        return res.e

    def gradient(self, v):
        """Returns the already computed gradient of vector v (gradient requested)."""
        res = self.__result_dict.get(v)
        if res == None:
            raise CalcManagerException("No result found for vector %s." %v)
        elif res.g == None:
            raise CalcManagerException("No gradient found for vector %s, only energy." %v)

        return res.g      
        
class ResultDict():
    """Maintains a dictionary of results i.e. energy / gradient calculations."""
    def __init__(self):
        self.list = []
#        self.parent = parent

    def __len__(self):
        return len(self.list)

    def get_closest(self, v):
        """Returns the result with the vector closest to |v|."""
        if len(self.list) == 0:
            return None

        dists = [np.linalg.norm(v - x.v) for x in self.list]
        dmin = np.inf
        for i, d in enumerate(dists):
            if d < dmin:
                dmin = d
                imin = i

        return self.list[imin]

    def add(self, v, res):
        """Add result res for vector |v| to the dictionary."""
        f = lambda x: is_same_v(v, x.v)
        matches_list = filter(f, self.list)

        if len(matches_list) > 1:
            lg.debug("More than 1 result for vector %s already in dictionary (add)." % v)

        if len(matches_list) >= 1:
            match_ix = self.list.index(v) #TODO: check whether I can use instance from matches_list
            self.list[match_ix].merge(res)

        else:
            self.list.append(res)

    def get(self, v):
        """Get previously calculated results for vector v from the dictionary."""
        f = lambda x: is_same_v(v, x.v)
        matches_list = filter (f, self.list)

        if len(matches_list) > 1:
            lg.debug("More than 1 result for vector %s already in dictionary (get)." % v)

        if len(matches_list) >= 1:
            return matches_list[0]
        else:
            return None

    def eg_counts(self):
        """Here it is assumed that every time we ask for a gradient we are 
        also asking for an energy.
        """
        l = ''.join([r.type() for r in self.list])
        e,g = l.count('E'), l.count('G')
        assert e >= g
        return e,g

def test_CalcManager(qc_driver, inputs, procs, to_cache=None, from_cache=None):
    """Perform more comprehensive tests of CalcManager / ResultDict infrastructure.

    >>> from numpy import *
    >>> from random import random
    >>> tmp = array((0.5, 0.7))
    >>> input1 = [tmp * x for x in range(10)]
    >>> input2 = [[random(),random()] for x in range(1000)]
    >>> procs = ([4,3,2,1], 6,1)
    >>> test_CalcManager(pts.pes.GaussianPES(fake_delay=0.3), input1, procs)
    >>> test_CalcManager(pts.pes.GaussianPES(fake_delay=0.3), input1, procs, to_cache="")
    Traceback (most recent call last):
        ...
    CalcManagerException: [Errno 2] No such file or directory: ''
    >>> test_CalcManager(pts.pes.GaussianPES(fake_delay=0.3), input1, procs, to_cache='test_CalcManager.tmp')
    >>> test_CalcManager(pts.pes.GaussianPES(fake_delay=0.3), input1, procs, from_cache='test_CalcManager.tmp')
    """

    cm = CalcManager(qc_driver, procs, from_cache=from_cache, to_cache=to_cache)

    # request gradients, energies and both
    dummy_bead_ix = 0
    N = len(inputs)
    for i in range(N):
        if i > N * 6 / 10:
            cm.request_energy(inputs[i], dummy_bead_ix)
        if i > N * 3 / 10:
            cm.request_gradient(inputs[i], dummy_bead_ix)

    cm.proc_requests()

    # Get results back, CalcManager, only allowing exceptions for calculations
    # that were not asked for.
    for j, i in enumerate(inputs):
        try:
            e = cm.energy(i)
        except CalcManagerException, cme:
            if j > N * 3 / 10:
                print cme, j
                assert False, "This should not happen."
            


if __name__ == "__main__":
    import doctest
    doctest.testmod()


