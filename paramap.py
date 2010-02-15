#!/usr/bin/python
"""
  Module for calculating several choice of variables for a function at once.
  There are several choices for the way of calculating, with Threads
  as in td_map or pmap , with processes as in ps_map and with either of them
  in pa_map (look here were Worker and Queue is gotten from)

  Test of the Module:

  Testfunctions:

       >>> def g(x):
       ...     return [ 2 * x[0] * x[1] * x[2]  , x[1]**2 , x[2] ]

       >>> def g2(x):
       ...     return [4 * x[0] ]

       >>> def g3(x):
       ...     return [ 2 * x[0] * x[1] ]

       >>> from time import sleep
       >>> def g4(x):
       ...     print "I take a nap"
       ...     sleep(1)
       ...     print "I'm back"
       ...     return g(x)

   The places for testing purposes

       >>> x1 = [[1, 4, 5], [2, 2, 2], [2, 5, 7], [1, 0, 0]]

       >>> x2 = [[1],[ 2], [3], [4], [5]]
       >>> x5 = [1, 2, 3, 4, 5]

       >>> x3 = [[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4],
       ...       [5, 5, 5], [6, 6, 6], [7, 7, 7], [8, 8, 8]]

  Now calculates some things in parallel for one of the own functions,
  taking lists and giving some back two.

  But first serial variant (from python) , just for comparing:

       >>> map(g, x1)
       [[40, 16, 5], [16, 4, 2], [140, 25, 7], [0, 0, 0]]

   New variants:

       >>> td_map(g, x1)
       [[40, 16, 5], [16, 4, 2], [140, 25, 7], [0, 0, 0]]
       >>> ps_map(g, x1)
       [[40, 16, 5], [16, 4, 2], [140, 25, 7], [0, 0, 0]]
       >>> pa_map(g, x1)
       [[40, 16, 5], [16, 4, 2], [140, 25, 7], [0, 0, 0]]
       >>> pmap(g, x1)
       [[40, 16, 5], [16, 4, 2], [140, 25, 7], [0, 0, 0]]

   To see the time it takes: Same function as before but with
   sleep(1) between I take a nap and I'm back

       >>> map(g4, x1)
       I take a nap
       I'm back
       I take a nap
       I'm back
       I take a nap
       I'm back
       I take a nap
       I'm back
       [[40, 16, 5], [16, 4, 2], [140, 25, 7], [0, 0, 0]]
       >>> td_map(g4, x1)
       I take a nap
       I take a nap
       I take a nap
       I take a nap
       I'm back
       I'm back
       I'm back
       I'm back
       [[40, 16, 5], [16, 4, 2], [140, 25, 7], [0, 0, 0]]
       >>> ps_map(g4, x1)
       [[40, 16, 5], [16, 4, 2], [140, 25, 7], [0, 0, 0]]


       >>> pa_map(g, x3)
       [[2, 1, 1], [16, 4, 2], [54, 9, 3], [128, 16, 4], [250, 25, 5], [432, 36, 6], [686, 49, 7], [1024, 64, 8]]
       >>> pa_map(g2, x2)
       [[4], [8], [12], [16], [20]]

  Here testing togehter with the derivatef function from the vib module,
  which is the first application for it, here it is testted for severak
  of the functions given above.

       >>> from aof.vib import derivatef

       >>> hessian = derivatef(g, [1.0, 2.0, 1.0] )
       >>> print hessian
       [[ 4.  0.  0.]
        [ 2.  4.  0.]
        [ 4.  0.  1.]]

  Here is a test for a single argument to single argument test
       >>> derivatef(g2, 2.4)
       array([[ 4.]])

  test also 2 (3) -> 1
       >>> derivatef(g3, [1.0, 2.0, 1.0] )
       array([[ 4.],
              [ 2.],
              [ 0.]])

 test that the results only comes if all finished and that they start
 not one after another (for the parallel versions)
       >>> hessian = derivatef(g4, [1.0, 2.0, 1.0], p_map = td_map )
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

  Testing the processes variant:
       >>> derivatef(g, [1.0, 2.0, 1.0], p_map = ps_map )
       array([[ 4.,  0.,  0.],
              [ 2.,  4.,  0.],
              [ 4.,  0.,  1.]])


  Do they start at once? Slow variant
       >>> hessian = derivatef(g4, [1.0, 2.0, 1.0], p_map = ps_map )
       >>> print hessian
       [[ 4.  0.  0.]
        [ 2.  4.  0.]
        [ 4.  0.  1.]]
"""
from threading import Thread
#from threading import Thread as Worker
#from Queue import Queue
from processing import Process as Worker
from processing import Queue
from processing import Process

# -------- Variante with own thread class-
class Mythread(Thread):
      def __init__(self, func, place):
          Thread.__init__(self)
          self.func = func
          self.place = place
      def run(self):
          self.result = self.func(self.place)

def td_map(g, xs):
    """
    Variant of map using threads to "paralize"
    things, meaning maping each run of the
    function g on its own thread
    This variant uses a Class, called Mythread
    for performing and getting back the result
    it would not work with processes
    """
    rs = []
    outcome = [None for xval in xs]

    # start the actual threads
    for i, xval in enumerate(xs):
        t = Mythread(func = g, place = xval )
        rs.append(t)
        t.start()

    # wait for the threads to finish
    # (several possibilities?)
    #1) while threading.activeCount()>1:
    #1)   sleep(1)
    #2) for t in threading.enumerate():
    #2)    if t is not threading.currentThread():
    #2)        t.join()
    # or 3):
    for t in rs:
        t.join()

    # now gather the results
    for i, t in enumerate(rs):
        outcome[i] = t.result

    return outcome

# ------------ Variant with original thread----
def pmap(f, xs):
    """
    Variant of map using threads to "paralize"
    things, meaning maping each run of the
    function g on its own thread
    This variant uses only the origional Thread
    implemention and stores the results in a
    list, build before the threads seperate
    it would not work with processes
    """
    results = [None] * len(xs)

    def workshare(i):
        results[i] = f(xs[i])

#   # 1) serial version:
#   for i in range(len(xs)):
#       workshare(i)

    # 2) threaded version:
    # create thread objects:
    threads = [ Thread(target=workshare, args=(i,)) for i in range(len(xs)) ]

    # start all threads:
    for t in threads:
        t.start()

    # wait for completion:
    for t in threads:
        t.join()

    # now that "results" must have been set:
    return results

# ----------- Variant with the processes module(working similar to threads)-----

def ps_map(g, xs):
    """
    Variant of map which paralize with the Process class
    found in the processing/multiprocessing module
    It uses a Queue to store the results
    it also stores from which start it came at is it
    not completly clear to me if the order could be changed
    The results are only gatherd after all processes have finished
    """
    # prepare some variable for the return value
    outcome = [None] * len(xs)

    # each process should do a job on a single geometry and
    # store the result somewhere reachable (in the result queue)
    def worksingle(i, result):
        work = g(xs[i])
        result.put([i, work])

    # here I can put the results and get them back
    result = Queue()
    # Initialize the porcesses that should be used
    procs = [ Process(target=worksingle, args=(i,result )) for i in range(len(xs)) ]

    # start all processes
    for ps in procs:
        ps.start()

    # finish all processes
    for ps in procs:
        ps.join()

    # put the results in the return value
    # so far I'm not sure if they would be in the correct oder
    # by themselves
    for i in range(len(procs)):
          j, res = result.get()
          outcome[j] = res

    return outcome

# ------------- Variant working either with Thread or Processes---
#               depending on what is choosen as Worker and Queue
#               in the import part

def pa_map(f, xs):
    """
    Variant of map which works both with Threads or Processes (of
    the processing/multiprocessing module)
    The selection from those two is done by choosing Worker (and Queue as
    the Queue.Queue does not work together with the Process)
    The result is put in a Queue, the input is given directly
    It also gives the number where it calculates, if there would be
    a mixup in the queue
    """
    # prepare some variable for the return value
    outcome = [None] * len(xs)

    # each process should do a job on a single geometry and
    # store the result somewhere reachable (in the queue "q")
    def worksingle(i, x, q):
        q.put((i, f(x)))

    # here I can put the results and get them back
    q = Queue()

    # Initialize the porcesses that should be used
    workers = [ Worker(target=worksingle, args=(i, x, q)) for i, x in enumerate(xs) ]

    # start all processes
    for w in workers:
        w.start()

    # finish all processes
    for w in workers:
        w.join()

        # put the results in the return value
        # so far I'm not sure if they would be in the correct oder
        # by themselves
        i, res = q.get()
        outcome[i] = res

    return outcome

if __name__ == "__main__":
    import doctest
    doctest.testmod()

