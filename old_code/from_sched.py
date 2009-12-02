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

