#!/usr/bin/python
"""
  Module for calculating several choice of variables for a function at once.
  There are several choices for the way of calculating, with Threads
  or with processes.

  Test functions:

       >>> def g1(x):
       ...     return [ 2 * x[0] * x[1] * x[2]  , x[1]**2 , x[2] ]

       >>> from time import sleep
       >>> def g2(x):
       ...     print "I take a nap"
       ...     sleep(1)
       ...     print "I'm back"
       ...     return g1(x)

   Arguments for respective functions g1-g4:

       >>> x1 = [[1, 4, 5], [2, 2, 2], [2, 5, 7], [1, 0, 0]]

  Now calculates some things in parallel for one of the own functions,
  taking lists and giving some back two.

  But first serial variant (from python) , just for comparing:

       >>> map(g1, x1)
       [[40, 16, 5], [16, 4, 2], [140, 25, 7], [0, 0, 0]]

  Parallel variants:

       >>> pmap(g1, x1)
       [[40, 16, 5], [16, 4, 2], [140, 25, 7], [0, 0, 0]]

  To see the time it takes: Same function as before but with
  sleep(1) between I take a nap and I'm back

       >>> map(g2, x1)
       I take a nap
       I'm back
       I take a nap
       I'm back
       I take a nap
       I'm back
       I take a nap
       I'm back
       [[40, 16, 5], [16, 4, 2], [140, 25, 7], [0, 0, 0]]

  For some reason the stdout output from process-based pmap is not
  visible here, use threaded version for testing:

       >>> tmap(g2, x1)
       I take a nap
       I take a nap
       I take a nap
       I take a nap
       I'm back
       I'm back
       I'm back
       I'm back
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
       I take a nap
       I take a nap
       I take a nap
       I take a nap
       I take a nap
       I take a nap
       I'm back
       I'm back
       I'm back
       I'm back
       I'm back
       I'm back
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

def _pmap(f, xs, Worker=Process, Queue=PQueue):
    """
    Variant of map which works both with Threads or Processes (of
    the processing/multiprocessing module)
    The selection from those two is done by choosing Worker (and Queue as
    the Queue.Queue does not work together with the Process)
    The result is put in a Queue, the input is given directly
    It also gives the number where it calculates, if there would be
    a mixup in the queue
    """
    # prepare placeholder for the return values
    outcome = [None] * len(xs)

    # each process should do its job and
    # store the result in the queue "q":
    def workshare(i, x, q):
        q.put((i, f(x)))

    # here I can put the results and get them back
    q = Queue()

    # Initialize the porcesses that should be used
    workers = [ Worker(target=workshare, args=(i, x, q)) for i, x in enumerate(xs) ]

    # start all processes
    for w in workers:
        w.start()

    for w in workers:
        # put the results in the return value
        # so far I'm not sure if they would be in the correct oder
        # by themselves
        w.join()
        i, res = q.get()
        outcome[i] = res

    return outcome

def pmap(f, xs):
    return _pmap(f, xs, Process, PQueue)

def tmap(f, xs):
    return _pmap(f, xs, Thread, TQueue)

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

