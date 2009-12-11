import logging

from aof.sched import ParaSched
from aof.common import Job, QCDriverException, Result, is_same_e, is_same_v, ERROR_STR, vec_summarise
import aof
import numpy as np

# setup logging
lg = logging.getLogger("aof.calcman")
lg.setLevel(logging.INFO)

class CalcManagerException(Exception):
    def __init__(self, msg):
        self.msg = msg 
    def __str__(self):
        return self.msg


class CalcManager():
    """
    Memorizes the results of previous calculations and, depending on the 
    closeness of requested calculations to previous ones either (a) runs a 
    whole new calculation, (b) interpolates existing data or (c) returns 
    existing data unmodified.

    Only (a) and (c) implemented at present.


    >>> cm = CalcManager(aof.common.GaussianPES())
    """

    def __init__(self, qc_driver, processors=None):

        self.qc_driver = qc_driver
        self.__para_sched = None

        if processors:
            shape, max_cpus, min_cpus = processors
            if max(shape) < max([max_cpus, min_cpus]):
                lg.warn("Max number of CPUs larger than max CPUs per node: %d > %d" % (max(cpus), max(shape)))
            lg.info("CPUs: %s QC Driver: %s", processors, qc_driver)
            self.__para_sched = ParaSched(qc_driver, processors)

        self.__pending_jobs = []
        self.__result_dict = ResultDict(self)

    def __str__(self):
        s = self.__class__.__name__ + ":"
        for j in self.__pending_jobs:
            s += j.__str__() + "\n"
        return s

    def request_energy(self, v):
        self.request_job(v, Job.E())

    def request_gradient(self, v):
        self.request_job(v, Job.G())

    def request_job(self, v, type):
        """Place into queue a request for calculations of type 'type'."""

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
            lg.info("Already have result for %s, using cached version." % vec_summarise(v))
            return

        # find dir containing previous calc to use as guess for wavefunction
        guess_dir = self.__result_dict.get_closest(v)

        # calc is not already in list so must add
        self.__pending_jobs.append(Job(v, type)) #PLAN: add guess_dir

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
    def __init__(self, parent):
        self.list = []
        self.parent = parent

    def __len__(self):
        return len(self.list)

    def get_closest(self, v):
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
        """Add result res for vector v to the dictionary."""
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
#        print "GET: %s in %s" % (v, self.list)
        f = lambda x: is_same_v(v, x.v)
        matches_list = filter (f, self.list)
#        lg.info("Matches list %s" % matches_list)

        if len(matches_list) > 1:
            lg.debug("More than 1 result for vector %s already in dictionary (get)." % v)

        if len(matches_list) >= 1:
            return matches_list[0]
        else:
            return None

def test_CalcManager(qc_driver, inputs, procs):
    """Perform more comprehensive tests of CalcManager.
    

    >>> from numpy import *
    >>> from random import random
    >>> tmp = array((0.5, 0.7))
    >>> input1 = [tmp * x for x in range(10)]
    >>> input2 = [[random(),random()] for x in range(1000)]
    >>> procs = ([4,3,2,1], (6,1))
    >>> test_CalcManager(aof.common.GaussianPES(fake_delay=0.3), input1, procs)

    """

    cm = CalcManager(qc_driver, procs)

    # request gradients, energies and both
    N = len(inputs)
    for i in range(N):
        if i > N * 6 / 10:
            cm.request_energy(inputs[i])
        if i > N * 3 / 10:
            cm.request_gradient(inputs[i])

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


