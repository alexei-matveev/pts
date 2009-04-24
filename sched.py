from threading import *
from thread import *
from Queue import *
from numpy import *


SAMENESS_THRESH = 1e-6
def is_same_v(v1, v2):
    return linalg.norm(v1 - v2) < SAMENESS_THRESH

def test_threads():
    def worker(q):
        print "worker started"
        while not q.empty():
            import subprocess
            import os

            item = q.get()
            p = subprocess.Popen("g03 " + item, shell=True)
            sts = os.waitpid(p.pid, 0)

            my_id = get_ident()
            print "thread " + str(my_id) + ": item " + item + " complete"
            q.task_done()

    queue = Queue()
    
    items = ["j1.com", "j2.com", "j3.com", "j4.com", "j5.com", "j6.com", "j7.com"]
    for item in items:
        print "Adding ", item
        queue.put(item)

    no_of_workers = 3
    for i in range(no_of_workers):
         t = Thread(target=worker, args=(queue,))
         t.start()

    if queue.empty():
        print "Problem: queue was empty"
    else:
        queue.join()

class MiniQC():
    """Mini qc driver. Just or testing"""
    def run(self, j):
        x = j.v[0]
        y = j.v[1]

        e = x + y

        if j.is_grad():
            g = array((x + 1, y + 1))
            res = Result(j.v, e, g)
        else:
            res = Result(j.v, e)

        return res

def test_calcManager():
    cm = CalcManager(MiniQC(), (4,1,2))

    tmp = array((0.5, 0.7))
    inputs = [tmp * x for x in range(10)]

    for i in range(len(inputs)):
        cm.request_energy(inputs[i])
        if i < 4:
            cm.request_energy(inputs[i])
        if i > 3:
            cm.request_grad(inputs[i])

    print cm
    cm.proc_requests()

    for i in inputs:
        print "Vector ", i,
        try:
            e = cm.energy(i)
            print "e =", e,
        except Exception, err:
            print err

        try:
            g = cm.grad(i)
            print "g =", g
        except Exception, err:
           print err

class CalcManager():
    """
    Memorizes the results of previous calculations and, depending on the 
    closeness of requested calculations to previous ones either (a) runs a 
    whole new calculation, (b) interpolates existing data or (c) returns 
    existing data unmodified.

    Only (a) and (c) implemented at present.
    """

    def __init__(self, qc_driver, sched_args = None):

        self.qc_driver = qc_driver
        self.__para_sched = None

        if sched_args != None:
            total, max, norm = sched_args

            print total, norm, max, qc_driver
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

    def request_grad(self, v):
        self.request_job(v, Job.G())

    def request_job(self, v, type):
        """Place into queue, request for calculations of type 'type'."""

        for i in range(len(self.__pending_jobs)):
#            print i, len(self.__pending_jobs)
            j = self.__pending_jobs[i]
            if j.geom_is(v):
                j.add_calc(type)
                return

        self.__pending_jobs.append(Job(v, type))
        print self.__pending_jobs[-1]

    def proc_requests(self):
        """Process all jobs in queue."""
        if self.__para_sched != None:
            print self.__class__.__name__ + " operating in parallel"
            self.__para_sched.run_all(self.__pending_jobs)

            for r in self.__para_sched.get_results():
                self.__result_dict.add(r.v, r)

        # running serially
        else:
            print self.__class__.__name__ + " operating serially"

            for j in self.__pending_jobs:
                res = self.qc_driver.run(j)
                self.__result_dict.add(j.v, res)


    def energy(self, v):
        """Returns the already computed energy of vector v."""
        res = self.__result_dict.get(v)
        if res == None:
            raise Exception("No result found for vector %s." %v)

        return res.e

    def grad(self, v):
        """Returns the already computed gradient of vector v."""
        res = self.__result_dict.get(v)
        if res == None:
            raise Exception("No result found for vector %s." %v)
        elif res.g == None:
            raise Exception("No gradient found for vector %s." %v)

        return res.g      
        
class Job():
    """Specifies calculations to perform on a particular geometry v."""
    def __init__(self, v, l):
        self.v = v
        if not isinstance(l, list):
            l = [l]
        self.calc_list = l
    
    def __str__(self):
        s = ""
        for j in self.calc_list:
            s += j.__str__()
        return self.__class__.__name__ + " %s: %s" % (self.v, s)

    def geom_is(self, v_):
        assert len(v_) == len(self.v)
        return is_same_v(v_, self.v)

    def add_calc(self, calc):
        if self.calc_list.count(calc) == 0:
            self.calc_list.append(calc)

    def is_energy(self):
        return self.calc_list.count(self.E()) > 0
    def is_grad(self):
        return self.calc_list.count(self.G()) > 0

    class E():
        def __eq__(self, x):
            return isinstance(x, self.__class__) and self.__dict__ == x.__dict__
        def __str__(self):  return "E"

    class G():
        def __eq__(self, x):
            return isinstance(x, self.__class__) and self.__dict__ == x.__dict__
        def __str__(self):  return "G"

class ResultDict():
    """Maintains a dictionary of results i.e. energy / gradient calculations."""
    def __init__(self, parent):
        self.list = []
        self.parent = parent
        self.__name__ = "ResultDict"

    def add(self, v, res):
        """Add result res for vector v to the dictionary."""
        f = lambda x: is_same_v(v, x.v)
        matches_list = filter(f, self.list)

        if len(matches_list) > 1:
            raise Exception("More than 1 result for vector %s already in dictionary." % v)

        elif len(matches_list) == 1:
            match_ix = self.list.index(v)
            self.list[match_ix].merge(res)

        else:
            self.list.append(res)

    def get(self, v):
        """Get already calculated results for vector v from the dictionary."""
        f = lambda x: x == v
        matches_list = filter (f, self.list)

        if len(matches_list) > 1:
            raise Exception("More than 1 result for vector %s already in dictionary" % v)

        elif len(matches_list) == 1:
            match_ix = self.list.index(v)
            return self.list[match_ix]

        else:
            return None
       

class Result():
    def __init__(self, v, energy, gradient = None):
        self.v = v
        self.e = energy
        self.g = gradient

    def __eq__(self, r):
        return (isinstance(r, self.__class__) and is_same_v(r.v, self.v)) or (r != None and is_same_v(r, self.v))

    def __str__(self):
        s = self.__class__.__name__ + ": " + str(self.v) + " E = " + str(self.e) + " G = " + str(self.g)
        return s

    def merge(self, res):
        assert self.is_same_v(self.v, res.v)
        assert self.is_same_e(self.e, res.e)

        if self.g == None:
            self.g = res.g
        else:
            raise Exception("Trying to add a gradient result when one already exists")
        

class ParaSched:
    def __init__(self, qc_driver, total_procs = 4, min_job_procs = 1, max_job_procs = 2):
        
        self.__pending = Queue()
        self.__finished = Queue()
        self.__qc_driver = qc_driver

        # no of workers to start
        self.__workers_count = int (floor (total_procs / min_job_procs))

    def __worker(self, pending, finished):

        my_id = get_ident()
        print "worker started, id =", my_id
        while not pending.empty():

            try:
                item = self.__pending.get()
                res = self.__qc_driver.run(item)
                finished.put(res)

                print "thread " + str(my_id) + ": item " + str(item) + " complete"

            except Exception, ex:
                print "Thrown by worker: " + str(my_id) + " " + str(ex)

            finally:
                self.__pending.task_done()

        print "worker finished, id =", my_id

    def run_all(self, jobs_list):
        """Start threads to process jobs in queue."""

        # place jobs in a queue
        for j in jobs_list:
            self.__pending.put(j)

        # start workers
        print "%s spawning %d worker threads" % (self.__class__.__name__, self.__workers_count)
        for i in range(self.__workers_count):
            t = Thread(target=self.__worker, args=(self.__pending, self.__finished))
            t.start()

        """if self.__pending.empty():
            print "Problem: queue was empty"
        else:"""
        self.__pending.join()

    def get_results(self):
        results = []
        while not self.__finished.empty():
            results.append(self.__finished.get())
            self.__finished.task_done()

        return results

