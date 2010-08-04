#!/usr/bin/python
"""
  Module for calculating several choice of variables for a function at once.
  There are several choices for the way of calculating, with Threads
  or with processes.

  Test functions:

       >>> def g1(x):
       ...     return [ 2 * x[0] * x[1] * x[2], x[1]**2, x[2] ]

       >>> from time import sleep
       >>> def g2(x):
       ...     print "g2: entered"
       ...     sleep(1)
       ...     print "g2: exit"
       ...     return g1(x)

       >>> from os import getenv, system
       >>> def g3(x):
       ...  #  system("echo $AOF_SCHED_JOB_HOST")
       ...  #  system("echo $AOF_SCHED_JOB_NPROCS")
       ...  #  system("echo $AOF_SCHED_JOB_CPUS")
       ...     return g1(x)

       >>> class g4(object):
       ...     def __init__(self):
       ...        pass
       ...     def perform(self, x):
       ...         system("echo $AOF_SCHED_JOB_HOST")
       ...         system("echo $AOF_SCHED_JOB_NPROCS")
       ...         system("echo $AOF_SCHED_JOB_CPUS")
       ...         return g1(x)

   Arguments for the test functions g1, g2:

       >>> x1 = [[1, 4, 5], [2, 2, 2], [2, 5, 7], [1, 0, 0]]
       >>> x2 = [[1, 4, 5], [2, 2, 2], [2, 5, 7], [1, 0, 0], [9, 1, 0]]

  Now calculates some things in parallel for one of the own functions,
  taking lists and giving some back two.

  But first serial variant, just for comparing:

       >>> map(g1, x1)
       [[40, 16, 5], [16, 4, 2], [140, 25, 7], [0, 0, 0]]

  Parallel variants:

       >>> pmap(g1, x1)
       [[40, 16, 5], [16, 4, 2], [140, 25, 7], [0, 0, 0]]

       >>> pmap2(g1, x1)
       [[40, 16, 5], [16, 4, 2], [140, 25, 7], [0, 0, 0]]

  To see the time it takes: same function as before but with
  sleep(1) between entry and exit:

       >>> map(g2, x1)
       g2: entered
       g2: exit
       g2: entered
       g2: exit
       g2: entered
       g2: exit
       g2: entered
       g2: exit
       [[40, 16, 5], [16, 4, 2], [140, 25, 7], [0, 0, 0]]

  For some reason the stdout output from process-based pmap is not
  visible here, use threaded version for testing:

       >>> tmap(g2, x1)
       g2: entered
       g2: entered
       g2: entered
       g2: entered
       g2: exit
       g2: exit
       g2: exit
       g2: exit
       [[40, 16, 5], [16, 4, 2], [140, 25, 7], [0, 0, 0]]

       >>> pmap2(g3, x1)
       [[40, 16, 5], [16, 4, 2], [140, 25, 7], [0, 0, 0]]
       >>> g5 = g4()
       >>> g6 = g5.perform
       >>> pmap2(g6, x1)
       [[40, 16, 5], [16, 4, 2], [140, 25, 7], [0, 0, 0]]

       >>> sched = Strategy(topology = [4], pmin = 2, pmax = 3)
       >>> pmap_2 = PMap2(strat = sched)
       >>> pmap_2(g3, x1)
       [[40, 16, 5], [16, 4, 2], [140, 25, 7], [0, 0, 0]]
       >>> pmap_2(g3, x2)
       [[40, 16, 5], [16, 4, 2], [140, 25, 7], [0, 0, 0], [0, 1, 0]]

  Here testing togehter with the derivatef function from the vib module,
  which is the first application for it, here it is testted for severak
  of the functions given above.

       >>> from vib import derivatef

       >>> hessian = derivatef(g1, [1.0, 2.0, 1.0], pmap = pmap)
       >>> print hessian
       [[ 4.  0.  0.]
        [ 2.  4.  0.]
        [ 4.  0.  1.]]

 test that the results only comes if all finished and that they start
 not one after another (for the parallel versions)
       >>> hessian = derivatef(g2, [1.0, 2.0, 1.0], pmap = tmap)
       g2: entered
       g2: entered
       g2: entered
       g2: entered
       g2: entered
       g2: entered
       g2: exit
       g2: exit
       g2: exit
       g2: exit
       g2: exit
       g2: exit
       >>> print hessian
       [[ 4.  0.  0.]
        [ 2.  4.  0.]
        [ 4.  0.  1.]]
"""

__all__ = ["pmap"]

from threading import Thread, activeCount
from Queue import Queue as TQueue
from os import environ
from sched import Strategy, SchedQueue, SchedQueueEmpty
from time import sleep
try:
    from multiprocessing import Process
    from multiprocessing import Queue as PQueue
    from multiprocessing import Pool, Manager, Event
except:
    from processing import Process
    from processing import Queue as PQueue
    from processing import Pool, Manager, Event

def job(jid, queue, func, args=(), kwds={}):
    """Each process should do its job and
    store the result in the queue.
    """
    queue.put((jid, func(*args, **kwds)))

class PMap(object):
    """A "classy" implementation of parallel map.
    """

    def __init__(self, Worker=Process, Queue=PQueue):
        self.__Worker = Worker
        self.__Queue = Queue

    def __call__(self, f, xs, processes = None):
        # processes just given for consistency (not needed anywhere
        # in this algorithm, compare pool_map)

        # aliases:
        Worker = self.__Worker
        Queue = self.__Queue

        # force evaluation of arguments, some callers may pass
        # enumerate() or generator objects:
        xs = [x for x in xs]

        # prepare placeholder for the return values
        fxs = [None for x in xs]

        # here I can put the results and get them back
        queue = Queue()

        # Initialize the porcesses that should be used
        workers = [ Worker(target=job, args=(jid, queue, f, (x,))) for jid, x in enumerate(xs) ]

        # start all processes
        for w in workers:
            w.start()

        for w in workers:
            # put the results in the return value
            # so far I'm not sure if they would be in the correct oder
            # by themselves
            w.join()
            jid, fx = queue.get()
            fxs[jid] = fx

        return fxs

pmap = PMap(Process, PQueue)
tmap = PMap(Thread, TQueue)

MAXPROCS = 12

def pool_map(f, xs, processes=MAXPROCS):
    """
    Variant of map using the map function of the pool module
    Here a pool with MAXPROCS or user defined number of processes.

    The pool object (of our python version at least) has some
    problems with interactive use, therefore there are no tests
    for it in the doctests of this module.
    """

    # Initializing the Pool() with processes=None will
    # start as many workers as there are CPUs on the workstation.

    # initializes the pool object
    pool = Pool(processes=processes)

    res = pool.map(f, xs)
    pool.close()

    return res

class f_schedwr():
     """
     Wrapper around the function to
     take first care of the environment variable

     Described as seperate class to please pool_map
     """
     def __init__(self, f):
          self.f = f

     def __call__(self, xplus, num = None):
          x, env = xplus
          node, processes = env
          np = len(processes)
          value = ""
          for proc in processes[:-1]:
               value += "%s," % (proc)
          value += "%s" % (processes[-1])
          # the environment variables should contain something
          # like: node number_of_procs number of the procs (on the node)
          environ['AOF_SCHED_JOB_HOST'] = "%s" % (node)
          environ['AOF_SCHED_JOB_NPROCS'] = "%s" % (np)
          environ['AOF_SCHED_JOB_CPUS'] = value
          if num == None:
              return self.f(x)
          else:
              return self.f(x, num)

class PMap3(object):
    """A implemention of parallel map, which uses the scheduler of sched.py
       for generating a fix strategy (via the strategy wrapper).
       The wrapper f_schedwr makes the strategy available at the call of the
       function by setting some environment variables (for the specific process).
    """

    def __init__(self, Worker=Process, Queue=PQueue, strat = Strategy(), wait = 0.333):
        self.__Worker = Worker
        self.__Queue = Queue
        self.strat = strat
        # FIXME: is this waiting time reasonable? Or is it too large?
        self.wait = wait

    def __call__(self, f, xs, processes = None):
        # processes just given for consistency (not needed anywhere
        # in this algorithm, compare pool_map)

        # aliases:
        Worker = self.__Worker
        Queue = self.__Queue
        manager = Manager()

        # here I can put the results and get them back
        queue = Queue()

        # wrapper around the function, to generate environment variables
        ffun = f_schedwr(f)

        # store for all processes accessible which CPUs are occupied
        # manager ensures that it is update for all processes if one changes it
        occupied = manager.dict()

        # event changing the occupied CPUs, if there would be only
        # one, this should lock the other processes by its own, but as
        # there may be several for the given job, this event ensures that
        # the other may not interfere during setting
        event = Event()
        event.set()

        def worker(inq, outq):
                # worker for a single job
                #print occupied.keys()
                # gets as input all that is needed
                jid, fun, x, sched_job = inq
                # distr for occupied dictionary (easier too compare)
                # node and cpus for handling over to the function call
                distr, node, cpus = sched_job

                still_occupied = True
                while still_occupied:
                     # as long as one or more of the processes for the job
                     # are occupied, test from time to time if they are released

                     # only one should work on the occupied dict at the same time
                     # (as there are several things to do the lock of the manager
                     # should not be enough)
                     event.wait()

                     # lock dict for the other processes (they will wait at event.wait)
                     event.clear()
                     #print occupied.keys()
                     still_occupied = False
                     for cpu in distr:
                         if cpu in occupied:
                              # if any of the cpus is still occupied, the job
                              # mustn't start
                              still_occupied = True
                     if not still_occupied:
                         # all cpus are free, so leave loop and start job
                         #print "Job %s started %s" % (str(sched_job), str(jid))
                         # release occupied only after the choosen cpus are in it
                         break
                     else:
                         # release the occupied dict for the other processes
                         # (maybe they will find what they need)
                         event.set()
                         # wait some time before trying again
                         sleep(self.wait)

                # starting job
                for cpu in distr:
                    # the cpus in distr should be used for the job
                    # thus now they are "occupied"
                    occupied[cpu] = True
                    # now the other proccesses can try occupied again
                    event.set()

                # the function needs also the information on the node and cpus
                xplus = (x, (node, cpus))
                # the calculation
                result = (jid, fun(xplus, jid))
                # this queue should hold the results
                outq.put(result)
                # release the cpus
                # this needn't be made locked, as only one proccess
                # wants to release this special cpu at this time
                for cpu in distr:
                    del occupied[cpu]


        # force evaluation of arguments, some callers may pass
        # enumerate() or generator objects:
        xs = [x for x in xs]

        # calculate the scheduling strategy
        # the extended gives also the distribution in numbers of the cpus
        # this is just easier to handle
        sched = self.strat.call_extended(len(xs))

        # this is the input per job, stored in input for all of them
        input = [(jid, ffun, x, sched[jid]) for jid, x in enumerate(xs)]

        # prepare placeholder for the return values
        fxs = [None for x in xs]

        # define a worker for each job
        workers = [ Worker(target=worker, args=(inp, queue)) for inp in input]

        # start all processes, so all jobs start at the same time
        # but some will wait, before the actual QM-calculation starts,
        # as for each cpu ony one job should run at once
        for w in workers:
            w.daemon = True
            w.start()

        #print "All started"
        for w in workers:
            # put the results in the return value
            # so far I'm not sure if they would be in the correct oder
            # by themselves
            w.join()
            jid, fx = queue.get()
            fxs[jid] = fx
        #print "All ended"

        return fxs

class PMap2():
    """
    The same as Pmap but uses aditionally the
    Strategy class to generate a environment variables
    called "AOF_SCHED_JOB_*" which stores informations
    to the scheduling strategy regarding the special job
    AOF_SCHED_JOB_HOST gives number of host to calculate this
           special job on, starts with zero
    AOF_SCHED_JOB_NPROCS gives number of CPUs this job should
           run in parallel on
    AOF_SCHED_JOB_CPUS gives the number of the CPUs (starting with 0)
           on the host, which should be used
    """
    def __init__ (self, strat = Strategy(), p_map = pmap):
        self.strat = strat
        self.p_map = p_map

    def __call__ (self, f, xs):
         n = len(xs)
         sched = self.strat(n)

         ffun = f_schedwr(f)

         # Find out how many (count) jobs should run in parallel (at beginning)
         # only helpful if scheduling strategy does not change this to often
         first = sched[0]
         host1, cpus1 = first
         count = 1
         finished = False
         for next in sched[1:]:
               hostn, cpusn = next
               if hostn == host1:
                   for cp in cpusn:
                       if cp in cpus1:
                           finished = True
                           break
               if finished: break
               count += 1
         #print "Number of jobs running in parallel", count

         fxs = self.p_map(ffun, zip(xs, sched), count)
         return fxs

pmap2 = PMap2()
pmap3 = PMap3()

from os import system, getenv
def test(x, num = None):
  # system("echo $AOF_SCHED_JOB_HOST")
  # system("echo $AOF_SCHED_JOB_NPROCS")
  # system("echo $AOF_SCHED_JOB_CPUS")
    host = getenv("AOF_SCHED_JOB_HOST")
    nprocs = getenv("AOF_SCHED_JOB_NPROCS")
    number = getenv("AOF_SCHED_JOB_CPUS")
    print "%s waits" % (num)
    sleep(0.5)
    print "%s Scheduling informatiom: %s %s %s" % (num, host, nprocs, number)

    return x


if __name__ == "__main__":
    x1 = [[1, 4, 5], [2, 2, 2], [2, 5, 7], [1, 0, 0], [2, 3, 4], [1, 1, 1]]
    sched = Strategy(topology = [4], pmin = 1, pmax = 2)
    pmap4 = PMap2(strat = sched, p_map = pool_map)
    pmap5 = PMap3(strat = sched)
    print pmap4(test, x1)
    print pmap5(test, x1)
    print pmap3(test, x1)
    import doctest
    doctest.testmod()

# Default options for vim:sw=4:expandtab:smarttab:autoindent:syntax
