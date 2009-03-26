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
        


"""class ParaSched(threading.Thread):
    def __init__(self, total_procs, norm_procs, max_procs, input_list, f):
        self.__total_procs = total_procs
        self.__norm_procs = norm_procs
        self.__max_procs = norm_procs

        # setup queue
        self.__queue = Queue()
        for x in input_list:
            self.__queue.put(x)

        self.__results = []

        # start workers
"""
