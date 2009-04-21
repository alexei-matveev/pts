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
    """Memorizes the results of previous calculations and, depending on the closeness
    of newly requested calculations, either (a) runs a whole new calculation, 
    (b) interpolates existing data or (c) returns existing data unmodified."""

    def __init__(self, qc_driver, f_closeness = None, sched_args = None):
        self.__para_sched = None
        if sched_args != None:
            self.__para_sched = ParaSched(qc_driver, *sched_args)

    def add_energy_request(self, v):
        pass

    def add_grad_request(self, v):
        pass

    def proc_requests(self):
        if self.__para_sched != None:
            self.__para_sched.run_all()

    def energy(self, v):
        pass

    def grad(self, v):
        pass
        


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

