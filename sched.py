import threading
import thread
from Queue import Queue
import time
import logging
from numpy import floor, zeros, array, ones, arange

from aof.common import Job, QCDriverException, Result, is_same_e, is_same_v, ERROR_STR

# setup logging
lg = logging.getLogger("aof.sched")
lg.setLevel(logging.INFO)

class CalcManagerException(Exception):
    def __init__(self, msg):
        self.msg = msg 
    def __str__(self, msg):
        return self.msg

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

class G03Launcher():
    """Just for testing."""
    def run(self, j, ix):
        import subprocess
        import os
        import re

        n = j.v[0]
        inputfn = "job" + str(n) + ".com"
        outputfn = "job" + str(n) + ".log"
        p = subprocess.Popen("g03 " + inputfn, shell=True)
        sts = os.waitpid(p.pid, 0)
        f = open(outputfn, 'r')
        results = re.findall(r"SCF Done.+?=.+?\d+\.\d+", f.read(), re.S)
        f.close()
        return Result(j.v, results[-1])

def test_parallel():
    cm = CalcManager(G03Launcher(), (8,2,1))

    inputs = range(1,11)
    lg.info("inputs are %s", inputs)

    for i in inputs:
        cm.request_energy(array((i,i)))

    cm.proc_requests()

    for i in inputs:
        e = cm.energy(i)
        print e


class MiniQC():
    """Mini qc driver. Just for testing"""
    def run(self, j, ix):
        x = j.v[0]
        y = j.v[1]

        e = x + y

        if j.is_gradient():
            g = array((x + 1, y + 1))
            res = Result(j.v, e, g)
        else:
            res = Result(j.v, e)

        return res


def test_CalcManager():
    cm = CalcManager(MiniQC(), (4,1,2))

    tmp = array((0.5, 0.7))
    inputs = [tmp * x for x in range(10)]
    lg.info("inputs are %s", inputs)

    for i in range(len(inputs)):
        cm.request_energy(inputs[i])
        if i < 4:
            cm.request_energy(inputs[i])
        if i > 3:
            cm.request_gradient(inputs[i])

    cm.proc_requests()

    for i in inputs:
        lg.info("Vector %s" % i)
        try:
            e = cm.energy(i)
            lg.info("e = %s" % e)
        except Exception, err:
            lg.error(err)

        try:
            g = cm.gradient(i)
            lg.info("g = %s" % g)
        except Exception, err:
           lg.error(err)

    # tests caching of previously calculated results
    lg.info("Starting tests of result caching")
    inputs = [array((20.,30.))] + inputs
    print "inputs =", inputs
    for i in range(len(inputs))[:-3]:
        cm.request_energy(inputs[i])
        if i < 7:
            cm.request_energy(inputs[i])
        if i > 8:
            cm.request_gradient(inputs[i])

    cm.proc_requests()

    for i in inputs:
        lg.info("Vector %s" % i)
        try:
            e = cm.energy(i)
            lg.info("e = %s" % e)
        except Exception, err:
            lg.error(err)

        try:
            g = cm.gradient(i)
            lg.info("g = %s" % g)
        except Exception, err:
           lg.error(err)

class CalcManager():
    """
    Memorizes the results of previous calculations and, depending on the 
    closeness of requested calculations to previous ones either (a) runs a 
    whole new calculation, (b) interpolates existing data or (c) returns 
    existing data unmodified.

    Only (a) and (c) implemented at present.
    """

    def __init__(self, qc_driver, params):

        self.qc_driver = qc_driver
        self.__para_sched = None

        if 'processors' in params:
            total, max, norm = params['processors']

            lg.info("%d %d %d %s", total, norm, max, qc_driver)
            self.__para_sched = ParaSched(qc_driver, total, norm, max)

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
            lg.debug("Already have result for %s, will used cached value" % v)
            return

        # calc is not already in list so must add
        self.__pending_jobs.append(Job(v, type))

    def proc_requests(self):
        """Process all jobs in queue."""
        lg.info(self.__class__.__name__ + " %s jobs in queue" % len(self.__pending_jobs))

        if self.__para_sched != None:
            lg.info(self.__class__.__name__ + " parallel on up to " + str(self.__para_sched.total_procs) + " processors.")
            self.__para_sched.run_all(self.__pending_jobs)

            for r in self.__para_sched.get_results():
                if ERROR_STR in r.flags:
                    raise CalcManagerException("Error encountered in computation, result was: " + r)
                self.__result_dict.add(r.v, r)

        # running serially
        else:
            lg.info(self.__class__.__name__ + " operating serially")

            for j in self.__pending_jobs:
                res = self.qc_driver.run(j)
                self.__result_dict.add(j.v, res)


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

    def add(self, v, res):
        """Add result res for vector v to the dictionary."""
        f = lambda x: is_same_v(v, x.v)
        matches_list = filter(f, self.list)

        """if len(matches_list) > 1:
            for i in matches_list:
                print i
            raise Exception("More than 1 result for vector %s already in dictionary (add)." % v)"""

        if len(matches_list) >= 1:
            match_ix = self.list.index(v) #TODO: check whether I can use instance from matches_list
            self.list[match_ix].merge(res)

        else:
            self.list.append(res)

    def get(self, v):
        """Get previously calculated results for vector v from the dictionary."""
        f = lambda x: is_same_v(v, x.v)
        matches_list = filter (f, self.list)

        """if len(matches_list) > 1:
            raise Exception("More than 1 result for vector %s already in dictionary (get)." % v)"""

        if len(matches_list) >= 1:
            return matches_list[0]

        else:
            return None

class ParaSched(object):
    def __init__(self, qc_driver, total_procs = 4, normal_procs = 2, max_procs = 4):
        
        self.__pending = Queue()
        self.__finished = Queue()
        self.__qc_driver = qc_driver

        # no of workers to start
        if total_procs % normal_procs != 0:
            raise SchedException("total_procs must be a whole multiple of min_procs")

        self.__total_procs = total_procs
        self.__workers_count = int (floor (total_procs / normal_procs))
        self.__normal_procs = normal_procs

    @property
    def total_procs(self):
        return self.__total_procs

    def __worker(self, pending, finished, ix):
        """Runs as a Python thread. "pending" and "finished" are both thread
        safe queues of jobs to be consumed / added to by each worker. ix
        is the index of each worker, used in node placement."""

        my_id = thread.get_ident()

        lg.debug("Worker starting, id = %s ix = %s" % (my_id, ix))

        while not pending.empty():

            try:
                item = self.__pending.get()

            except Queue.Empty, ex:
                lg.error("Thrown by worker: " + str(my_id) + " " + str(ex))
                return

            # just for testing what happens when a worker thread experiences an exception
#            if ix % 2 == 0:
#                raise Exception("Dummy")

            # setup parameter dictionary
            params = dict()

            # TODO: eventually processor ranges must become dynamic to allow 
            # jobs to be run on variable numbers of processors. I think I will 
            # need to implement a super-class of Queue to generate processor
            # ranges dynamically.
            item.processor_ix_start = self.__normal_procs * ix
            item.processor_ix_end = self.__normal_procs * ix + self.__normal_procs - 1

            # call quantum chem driver
            try:
                res = self.__qc_driver.run(item)
            except common.QCDriverException, inst:
                # TODO: this needs to be done differently, when a worker encounters 
                # an exception it should empty the queue and then rethrow, otherwise
                # the other jobs will continue to run
                l = ["Worker", str(my_id), ": Exception thrown when", 
                     "calling self.__qc_driver.run(item):", str(type(inst)),
                     ":", str(inst.args)]
                msg = ' '.join(l)
                
                res = Result(item.v, 0.0, flags = dict(ERROR_STR = msg))


            finished.put(res)
            lg.info("Worker %s finished a job." % my_id)
            self.__pending.task_done()
            lg.debug("thread " + str(my_id) + ": item " + str(item) + " complete: " + str(res))

        lg.info("Queue empty, worker %s exiting." % my_id)

    def run_all(self, jobs_list):
        """Start threads to process jobs in queue."""

        # place jobs in a queue
        while len(jobs_list) > 0:
            self.__pending.put(jobs_list.pop())

        # start workers
        lg.info("%s spawning %d worker threads" % (self.__class__.__name__, self.__workers_count))
        for i in range(self.__workers_count):
            t = threading.Thread(target=self.__worker, args=(self.__pending, self.__finished, i))
            t.daemon = True
            t.start()

        # The normal method would be to join() the thread, but if one of the 
        # threads dies then we get deadlock.
        while threading.activeCount() > 1: # this is a bit crude, can it be done better?
            time.sleep(0.3333)

        lg.debug("All worker threads exited")

        if not self.__pending.empty():
            lg.error("Pending queue not empty but threads exited")

    def get_results(self):
        results = []
        while not self.__finished.empty():
            results.append(self.__finished.get())
            self.__finished.task_done()

        return results

def test_threads():
    def worker(q):
        logging.info("worker started")
        while not q.empty():
            import subprocess
            import os

            item = q.get()
            p = subprocess.Popen("g03 " + item, shell=True)
            sts = os.waitpid(p.pid, 0)

            my_id = get_ident()
###            print "thread " + str(my_id) + ": item " + item + " complete"
            q.task_done()

    queue = Queue()
    
    items = ["j1.com", "j2.com", "j3.com", "j4.com", "j5.com", "j6.com", "j7.com"]
    for item in items:
#        print "Adding ", item
        queue.put(item)

    no_of_workers = 3
    for i in range(no_of_workers):
         t = Thread(target=worker, args=(queue,))
         t.start()

    if queue.empty():
        print "Problem: queue was empty"
    else:
        queue.join()


if __name__ == "__main__":
    test_parallel()

