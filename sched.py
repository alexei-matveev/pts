class ParaSched(threading.Thread):
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

