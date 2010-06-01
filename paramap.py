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

   Arguments for the test functions g1, g2:

       >>> x1 = [[1, 4, 5], [2, 2, 2], [2, 5, 7], [1, 0, 0]]

  Now calculates some things in parallel for one of the own functions,
  taking lists and giving some back two.

  But first serial variant, just for comparing:

       >>> map(g1, x1)
       [[40, 16, 5], [16, 4, 2], [140, 25, 7], [0, 0, 0]]

  Parallel variants:

       >>> pmap(g1, x1)
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

from threading import Thread
from Queue import Queue as TQueue
try:
    from multiprocessing import Process
    from multiprocessing import Queue as PQueue
    from multiprocessing import Pool
except:
    from processing import Process
    from processing import Queue as PQueue
    from processing import Pool

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

    def __call__(self, f, xs):
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


if __name__ == "__main__":
    import doctest
    doctest.testmod()

# Default options for vim:sw=4:expandtab:smarttab:autoindent:syntax
