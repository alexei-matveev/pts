from threading import *
from thread import *
from Queue import *

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
        
class CalcManager():
    """
    Memorizes the results of previous calculations and, depending on the 
    closeness of requested calculations to previous ones either (a) runs a 
    whole new calculation, (b) interpolates existing data or (c) returns 
    existing data unmodified.

    Only (a) and (c) implemented at present.
    """

    def __init__(self, qc_driver, f_closeness = None, sched_args = None):
        self.__para_sched = None
        if sched_args != None:
            self.__para_sched = ParaSched(qc_driver, *sched_args)

        self.__pending_jobs = []
        self.__result_dict = ResultDict(self)

    def add_energy_request(self, v):
        self.add_job_request(v, self.Job.E())

    def add_grad_request(self, v):
        self.add_job_request(v, self.Job.G())

    def add_job_request(self, v, type):
        for i in range(len(self.__pending_jobs)):
            j = self.__pending_jobs[i]
            if j.geom_is(v):
                self.__pending_jobs[i] = j.add_calcs(type)
                return

        self.__pending_jobs.append(Job(v, type))
                

    def proc_requests(self):
        """Process all jobs in queue."""
        if self.__para_sched != None:
            self.__para_sched.run_all()

        # running serially
        else
            for j in self.__pending_jobs:
                if j == Job.E()
                    res = self.qc_driver.energy(j.v)
                    result_dict.add(v, res)
                elif j == Job.G()
                    res = self.qc_driver.energy(j.v)
                    self.__result_dict.add(v, res)

    def energy(self, v):
        res = self.__result_dict.get(v)
        if res == None:
            raise Exception("No result found for vector %s." %v)

        return res.e

    def grad(self, v):
        res = self.__result_dict.get(v)
        if res == None:
            raise Exception("No result found for vector %s." %v)
        elif res.g == None:
            raise Exception("No gradient found for vector %s." %v)

        return res.g      
        
    class Job():
        """Specifies calculations to perform on a particular geometry v."""
        def __init__(self, v, list):
            self.v = v
            self.calc_list = list
            self.sameness_thresh = 1e6
        
        def geom_is(self, v_):
            assert len(v_) = self.v
            d = linalg.norm(self.v - v_)
            return d < self.sameness_thresh

        def add_calcs(self, list):
            for j in list:
                if self.calc_list.count(j) == 0:
                    self.calc_list.append(j)

        class E():
            def __eq__(self, x):
                return isinstance(x, self.__class__) and self.__dict__ == x.__dict__
        class G():
            def __eq__(self, x):
                return isinstance(x, self.__class__) and self.__dict__ == x.__dict__

    class ResultDict():
        """Maintains a dictionary of results i.e. energy / gradient calculations."""
        def __init__(self, parent):
            self.list = []
            self.parent = parent

        def add(self, v, res):
            """Add result res for vector v to the dictionary."""
            f = lambda x: is_same_v(v, x)
            matches_list = filter(f, self.list)

            if len(matches_list) > 1:
                raise Exception("Results for vector %s already in dictionary." % v)

            elif len(matches_list) == 1:
                match_ix = self.list.index(v)
                self.list[match_ix].merge(res)

            else
                self.list.append(res)

        def get(self, v):
            """Get already calculated results for vector v from the dictionary."""
            f = lambda x: is_same_v(v, x)
            matches_list = filter (f, self.list)

            if len(matches_list) > 1:
                raise Exception("More than 1 result for vector %s already in dictionary" % v)

            elif len(matches_list) == 1:
                match_ix = self.list.index(v)
                self.list[match_ix].merge(res)

            else
                return None
           

    class Result():
        def __init__(self, v, energy, gradient = None):
            self.v = v
            self.e = energy
            self.g = gradient

        def __eq__(self, r):
            return isinstance(r, self.__class__) and is_same_v(r.v, self.v) or is_same_v(r, self.v)

        def merge(self, res):
            assert self.is_same_v(self.v, res.v)
            assert self.is_same_e(self.e, res.e)

            if self.g == None:
                self.g = res.g
            else
                raise Exception("Trying to add a gradient result when one already exists")
            

class ParaSched:
    def __init__(qc_driver, total_procs = 4, min_job_procs = 1, max_job_procs = 2)
        
        self.__queue = Queue()

        # no of workers to start
        self.__workers = floor (total_procs / min_job_procs)


    def add_job(self, j, i = None):
        self.__queue.put((v,i))

    def run_all(self):
        for i in range(self, ):
            t = Thread(target=worker, args=(queue,))
            t.start()

        if queue.empty():
            print "Problem: queue was empty"
        else:
            queue.join()

    def get_result(v, i = None):
        pass

