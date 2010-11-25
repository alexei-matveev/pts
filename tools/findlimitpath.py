#!/usr/bin/python
import sys
import re
from pydoc import help

class storedata:
    """
    Class for storing the data, which will be used some times
    """
    def __init__(self, file, argv):
        # argv = "which variable" limit  specialopt
        if len(argv) < 3:
            print "ERROR: there are 4 values needed\n"
            sys.exit()
        self.file = file
        self.argument = argv[0]
        self.inval = float(argv[1])
        # there are several specialopts possible
        # as default they are all false:
        self.procent = False
        self.difference = False
        self.withvalues = False
        self.maxbead = None
        # the specialopts are decided here:
        specialalrg = argv[2][1:]
        if specialalrg =='mb':
            self.maxbead = []
            specialalrg = 'n'
        elif specialalrg.startswith('mb'):
            self.maxbead = []
            specialalrg = specialalrg[2:]

        if specialalrg == '%':
            self.procent = True
        elif specialalrg == 'd':
            self.difference = True
        elif specialalrg == '%w':
            self.procent = True
            self.withvalues = True
        elif specialalrg == 'dw':
            self.difference = True
            self.withvalues = True
        elif specialalrg == 'w':
            self.withvalues = True
        elif specialalrg == 'n':
            # if one does not want to have to decide which of the
            # arguments is a specialopt or the next value, one needs
            # a specialopt saying nothing special
            pass
        else:
            print "ERROR: This option does not exist!",specialalrg
            sys.exit()
        self.uninteresting = len(self.argument.split())
        # to find out if there is growing string and for
        # filling up the data then
        self.lengthconstant = True
        self.numberofbeads = None
        # here will the data be stored
        self.data = []

    def emaxbead(self):
         if not self.maxbead == None:
             return True
         else:
             return False

    def findmaxbead(self):
        """
        finds the number of the bead for each iteration
        which holds the maximum energy
        """
        filein = open(self.file, "r")
        for line in filein:
             # Data is stored in file as:
             # which variable : num1 | num2 | ...
             # the lengt of "which variable" has to be omitted(self.uninteresting)
             # then every second value is wanted
             if line.startswith('%s' % "Bead Energies"):
                 fields = line.split()
                 datapoints = (len(fields) - 2) / 2
                 max_e = float(fields[3])
                 numin = 0
                 for i in range(datapoints-1):
                    curen = float(fields[2 + 3 + 2 * i])
                    if max_e < curen:
                         max_e = curen
                         numin = i + 1
                 self.maxbead.append(numin)
        #print self.maxbead

    def createlist(self):
        """
        reads in the data from the self.file and filters the lines of
        interest, picks derin the data and makes the difference between
        them if wanted
        stores data in self.data
        """
        filein = open(self.file, "r")
        self.data = []
        self.lengthconstant = True
        storeline = []
        for line in filein:
             # Data is stored in file as:
             # which variable : num1 | num2 | ...
             # the lengt of "which variable" has to be omitted(self.uninteresting)
             # then every second value is wanted
             if line.startswith('%s' % self.argument):
                 fields = line.split()
                 dataline = []
                 datapoints = (len(fields) - self.uninteresting) / 2
                 for i in range(datapoints):
                       dataline.append(float(fields[self.uninteresting + 1 +2 * i]))
                 if not self.numberofbeads == None:
                      if not (self.numberofbeads == datapoints):
                          # Here the growing string method has produced the results
                          # lengthconstant says now there are lines with different lengt
                          # We hold the first line of the bigger length (we assume that
                          # the lengt increases all the time), we will fill up the smaller
                          # lines with the data from this line
                          # (as values should go down all the time tis way the first lines
                          # with the big values should give no hit for the limit
                          #FIXME: can we tread growing string case better?
                          self.lengthconstant = False
                          storeline = dataline
                 self.numberofbeads = datapoints
                 self.data.append(dataline)
        if self.difference:
            # Here not the original data is wanted but the difference between the two
            # succeding lines, starts with a zero line (line1 - line1)
            oldline = self.data[0]
            interstore = []
            for line in self.data:
                if not len(line) == len(oldline):
                    # some workaround for the growing string method, should give only
                    # data where there were values in the line before
                    already = len(oldline)
                    fillin = len(line) - already
                    diff = [abs(line[i] - oldline[i]) for i in range(already/2)]
                    diff += [abs(line[i + fillin] - oldline[i]) for i in range(already/2,already)]
                else:
                    # just the difference to the line before
                    diff = [abs(line[i] - oldline[i]) for i in range(len(line))]
                    interstore.append(diff)
                if not self.numberofbeads == len(oldline):
                    # for consistency with the case if not (self.numberofbeads == datapoints) above
                    self.lengthconstant = False
                    storeline = diff
                oldline = line
            self.data = interstore

        if not self.lengthconstant:
            # Here we fill up the lines with smaller length with the values from the first
            # line where the length has been full FIXME:? See above
            interstore = []
            for line in self.data:
                if len(line) < self.numberofbeads:
                    already = len(line)
                    fillin = self.numberofbeads - already
                    intermediate = [ line[i] for i in range(already/2)]
                    intermediate += [ storeline[already/2 + i] for i in range(fillin)]
                    intermediate += [ line[ already/2 + i] for i in range(already/2)]
                    assert(len(intermediate)==self.numberofbeads)
                    interstore.append(intermediate)
                else:
                    interstore.append(line)
            self.data = interstore

    def set_limit(self):
        '''
        Makes own limit for every bead, if a procentual values is
        searched for, the limits differ, else they are all the same
        '''
        self.limit = [ self.inval for i in range(self.numberofbeads)]
        if self.procent:
            for i in range(self.numberofbeads):
                diff = self.data[-1][i] - self.data[0][i]
                self.limit[i] = self.inval * diff + self.data[0][i]

class holdresult():
    """
    Stores the results, counts in which bead the limit was reached
    and when the value was above for the last time (stores this value
    + 1 as reachedlimitandstay)
    Does this for each bead and for all of them together
    """
    def __init__(self,numberofbeads):
        self.numberofbeads = numberofbeads
        self.reachedlimit = [None for i in range(self.numberofbeads)]
        self.reachedlimitall = None
        self.maxbeadlimit = None
        self.fallenbackall = -1
        self.fallenback = [-1 for i in range(self.numberofbeads)]
        self.fallenbackmb = -1
        self.reachedlimitandstay = [None for i in range(self.numberofbeads)]
        self.reachedlimitandstayall = None
        self.reachedlimitandstaymb = None
        self.allbel = 0

    def count_num(self, i, j, countyes):
        """
        counts for a bead
        """
        if countyes:
            # allbel for count_all
            self.allbel += 1
            if self.reachedlimit[j] == None:
                 self.reachedlimit[j] = i
                 self.reachedlimitandstay[j] = i
                 # sets fallenback to 0 (-1 so far)
                 self.fallenback[j] += 1
        else:
            if self.fallenback[j] > -1:
                 # if we have already once reached the limit, it is interesting
                 # if we are above again, this case we count the number we oversteped it
                 self.fallenback[j] += 1
                 # if we do not reach this point in another iteration, the number of
                 # this iteration + 1 (threaded as in other case) should be the first iteration
                 # after which we do not fall again above the limit
                 self.reachedlimitandstay[j] = i+1

    def count_all(self, i):
        """
        counts for whole string
        """
        # We count like for every bead (see above)
        # but allbel tells us how many of the beads
        # are below there limit
        if self.allbel == self.numberofbeads:
             if self.reachedlimitall == None:
                 self.reachedlimitall = i
                 self.reachedlimitandstayall = i
                 self.fallenbackall += 1
        else:
            if self.fallenbackall > -1:
                self.fallenbackall += 1
                self.reachedlimitandstayall = i+1

    def count_mb(self, i,countyes):
        """
        counts for the bead with maximal energy
        """
        if countyes:
            if self.maxbeadlimit == None:
                 self.maxbeadlimit = i
                 self.reachedlimitandstaymb = i
                 # sets fallenback to 0 (-1 so far)
                 self.fallenbackmb += 1
        else:
            if self.fallenbackmb > -1:
                 # if we have already once reached the limit, it is interesting
                 # if we are above again, this case we count the number we oversteped it
                 self.fallenbackmb += 1
                 # if we do not reach this point in another iteration, the number of
                 # this iteration + 1 (threaded as in other case) should be the first iteration
                 # after which we do not fall again above the limit
                 self.reachedlimitandstaymb = i + 1

class interpretvalues():
    """
    finds out if there is only one variable looked at or several
    handles the interaction between them (if several)
    holds the output formats
    """
    def __init__(self):
        # We need at least 3 arguments
        assert( len(sys.argv) > 3)
        # The first one should be the filename, in which we search
        self.file = sys.argv[1]
        if len(sys.argv[2:]) < 4:
            # in this case there are 3 or 4 arguments all together
            # (one for the file) and 2 to tree for the rest
            # we also have only one variable to look at:
            self.vals = 1
            arg2 = sys.argv[2:]
            if len(arg2) < 3:
                # if there are no special options we can allow orself
                # the luxury in not saying it, but the storedataclass
                # wants it
                arg2.append('-n')
            # create one storedataobject, extract wanted data from file
            # and produce the limits
            self.st1 = storedata(self.file, arg2)
            self.st1.createlist()
            if self.st1.emaxbead():
                self.st1.findmaxbead()
            self.st1.set_limit()
        else:
            # if there are more than one variable, we need to know, how to
            # connect them, there exist the options and or or
            if sys.argv[-1] == '-a' :
                self.merge = 0
            elif sys.argv[-1] == '-o':
                self.merge = 1
            else:
                print "ERROR: You need to specify the interaction between the different"
                print "values to look at"
                sys.exit()
            # the other arguments given are the variables
            # in this case we insist on the special options
            arg = sys.argv[2:-1]
            self.vals = len(arg)/3
            assert(len(arg) == 3 * self.vals)
            self.stn = [None for i in range(self.vals)]
            # create a storedataobject for each of them
            for k in range(self.vals):
                argm = arg[k*3:k*3+3]
                self.stn[k] = storedata(self.file, argm)
                self.stn[k].createlist()
                if self.stn[k].emaxbead():
                    self.stn[k].findmaxbead()
                self.stn[k].set_limit()

    def searchlimit(self, stx):
        """
        The smallest system, for only one variable
        """
        res = holdresult(stx.numberofbeads)
        for i, dataline in enumerate(stx.data[1:]):
             res.allbel = 0
             for j, dats in enumerate(dataline):
                 res.count_num(i+2, j, (dats <= stx.limit[j]))
                 if stx.emaxbead():
                     if j == stx.maxbead[i+1]:
                         res.count_mb(i+2,(dats <= stx.limit[j]))
             res.count_all(i+2)
        return res


    def searchlimitand(self, stx):
        """
        For several variables, connected with and
        (all have to be fullfiled at the same iteration)
        """
        numbbeads = stx[0].numberofbeads
        res = holdresult(numbbeads)
        for i in range(len(stx[0].data[1:])):
             # the number of the current iteration is i + 2
             # starting in the second iteration because for
             # differences (made with special option or are
             # differences in the file itsself) will have zeros
             # in the first iteration (0 < limit)
             res.allbel = 0
             for j in range(numbbeads):
                 onetrue = 0
                 for k in range(self.vals):
                     if (stx[k].data[i+1][j]  <= stx[k].limit[j]):
                         onetrue += 1
                 res.count_num(i+2, j, (onetrue == self.vals ))
                 if stx[0].emaxbead():
                     if j == stx[0].maxbead[i+1]:
                           res.count_mb(i+2,(onetrue == self.vals ))
             res.count_all(i+2)
        return res


    def searchlimitor(self, stx):
        """
        For several variables, connected with or
        (only one has to be fullfiled in the same iteration)
        """
        numbbeads = stx[0].numberofbeads
        res = holdresult(numbbeads)
        for i in range(len(stx[0].data[1:])):
             res.allbel = 0
             for j in range(numbbeads):
                 onetrue = 0
                 for k in range(self.vals):
                     if (stx[k].data[i+1][j]  <= stx[k].limit[j]):
                         onetrue += 1
                 res.count_num(i+2, j, (onetrue > 0))
                 if stx[0].emaxbead():
                     if j == stx[0].maxbead[i+1]:
                         res.count_mb(i+2,(onetrue > 0 ))
             res.count_all(i)
        return res

    def output(self, stx, res):
        print " "
        print "calculation was performed for",res.numberofbeads, "beads for",len(stx.data),"iterations"
        if stx.difference:
            print "The limit", stx.limit[0],"for the difference in the argument", stx.argument, "has been reached"
        else:
            print "The limit", stx.limit[0],"for the argument", stx.argument, "has been reached"
        print "In the beads                     ",res.reachedlimit
        print "how often fallen above afterwards", res.fallenback
        print "stayed below after:              ", res.reachedlimitandstay
        print "for all beads",res.reachedlimitall,"with", res.fallenbackall,"times fallen above again"
        print "but stayed there after", res.reachedlimitandstayall
        if stx.withvalues:
            print "The last iteration gave"
            print stx.data[-1]
        if stx.emaxbead():
            print "For the bead with maximum energy (highest bead) the limits are reached in:"
            print  res.maxbeadlimit, "but fallen above afterwards", res.fallenbackmb ,"times to stay there after", res.reachedlimitandstaymb
            if not res.maxbeadlimit == None:
                print "Here the maximum bead was number (counting starts with 1):",stx.maxbead[res.maxbeadlimit-1]+1
        print " "

    def outputprocent(self, stx, res):
        print " "
        print "calculation was performed for",stx.numberofbeads, "beads for",len(stx.data),"iterations"
        print "The search was for an procentual limit",stx.inval,"for the argument",stx.argument
        if stx.withvalues:
            print "This resembles by a starting geometry of:"
            print stx.data[0]
            if not stx.lengthconstant:
                print "Be careful by the growing string method, not all start geometries lie in the first iteration"
            print "And a endresult of:"
            print stx.data[-1]
        print "This leads to a limit of:"
        print stx.limit
        print "It was reached in the beads      ",res.reachedlimit
        print "how often fallen above afterwards", res.fallenback
        print "stayed below after:              ", res.reachedlimitandstay
        print "for all beads",res.reachedlimitall,"with", res.fallenbackall,"times fallen above again"
        print "but stayed there after", res.reachedlimitandstayall
        if stx.emaxbead():
            print "For the bead with maximum energy (highest bead) the limits are reached in:"
            print  res.maxbeadlimit, "but fallen above afterwards", res.fallenbackmb ,"times to stay there after", res.reachedlimitandstaymb
            if not res.maxbeadlimit == None:
                print "Here the maximum bead was number (counting starts with 1):",stx.maxbead[res.maxbeadlimit-1]+1
        print " "

    def outputall(self, stx, res):
        print " "
        print "calculation was performed for",res.numberofbeads, "beads for",len(stx[0].data),"iterations"
        print "The limits in the single variables were:"
        for sts in stx:
            if sts.difference:
                print "The search was for a limit", sts.limit[0],"in the difference of the argument", sts.argument
                if sts.withvalues:
                    print "The last iteration gave", sts.data[-1]
            elif sts.procent:
                print "The search was for an procentual limit",sts.inval,"for the argument",sts.argument
                if sts.withvalues:
                    print "This resembles by a starting geometry of:"
                    print sts.data[0]
                    if not sts.lengthconstant:
                        print "Be careful by the growing string method, not all start geometries lie in the first iteration"
                    print "And a endresult of:"
                    print sts.data[-1]
                print "This leads to a limit of:"
                print sts.limit
            else:
                print "The search was for a limit", sts.limit[0],"for the argument", sts.argument
                if sts.withvalues:
                    print "The last iteration gave"
                    print sts.data[-1]
        if self.merge == 0:
            print "It was searched for the point where all the limits were reached"
        elif self.merge == 1:
            print "It was searched for the point where at least one limit was reached"
        print "This happens at:"
        print "In the beads                     ",res.reachedlimit
        print "how often fallen above afterwards", res.fallenback
        print "stayed below after:              ", res.reachedlimitandstay
        print "for all beads",res.reachedlimitall,"with", res.fallenbackall,"times fallen above again"
        print "but stayed there after", res.reachedlimitandstayall
        if stx[0].emaxbead():
            print "For the bead with maximum energy (highest bead) the limits are reached in:"
            print  res.maxbeadlimit, "but fallen above afterwards", res.fallenbackmb ,"times to stay there after", res.reachedlimitandstaymb
            if not res.maxbeadlimit == None:
                print "Here the maximum bead was number (counting starts with 1):",stx[0].maxbead[res.maxbeadlimit-1]+1
        print " "

def findlimitpath():
    """
This programme was designed to interpret the output data one gets with the neb, string or growing string method from
the ParaTools framework
It uses the *.log file for it (created by the ParaTools framework), which contains data like:
Bead Energies            :   -61.4932 |   -61.1503 |   -60.2848 |   -59.9914 |   -60.4461 |   -60.9149 |   -61.1336
RMS Perp Forces          :  8.609e-03 |  9.350e-01 |  1.006e+00 |  6.541e-01 |  6.370e-01 |  3.819e-01 |  9.167e-03
The programme takes one of this variables and searches for the iteration in which a given limit is first reached,
it also takes track how often afterwards the variable oversteps this limit and in which iteration it stays below
for ever (if this iteration is given a value higher (+1 normally) than the maximum number of iterations the limit
is never reachd without overstepping.
The programme looks for the reaching of the limit in each bead and for all beads alltogether.

A typical call for it may look like:

  *** Example: ***
findlimitpath.py chch2_to_cch2_with_params.log "RMS Perp Forces" 0.5 -n "Bead Energies" 0.9 % "RMS Step Size" 0.01 -n  -a
  ***----------***

In this case it looks for the variables RMS Perp Forces, Bead Energies and RMS Step Size.

The -a at the and says, it looks for the case where all limits are reached in all variables (the and case);
it can also look for the case, where only one variable has to reach its limit (the or case) anounced by -o.
If there are several variables at once, this flag has to be specified; if there is only one, there is of course no
need and no allowance for it.
The settings for one variable looks something like:
   "Full name of Variable to look at" limit -special_option

available special options are:
  -%  : limit is procentual value of change in variable (of each bead)
  -d  : the limit of the difference of two suceeding iterations is wanted
  -w  : gives also the values of the variable for the last iteration
  -%w : % and w together
  -dw : d and w together
  -mb : the limit is also searched for the maximum bead, if several
        parameters are set, mb has to be set for the first to be used
        mb can be combined with any other special option, but in this
        case mb has to be first, so mbd is maximum bead search and the
        values is taken as differences
  -n  : nothing special
the last option is of course to ensure that the length fo each variable block
is the same, this is only needed if there are several variables for one the
settings
"Full name of Variable to look at" limit n
"Full name of Variable to look at" limit
are the same.

To return to the example above:
findlimitpath.py chch2_to_cch2_with_params.log "RMS Perp Forces" 0.5 -n "Bead Energies" 0.9 % "RMS Step Size" 0.01 -n  -a
This means that in the file chch2_to_cch2_with_params.log there is a search when the variable RMS Perp Forces is below 0.5,
the Bead Energies has fallen 90% of its change and at the same iteration the RMS Step Size has fallen below 0.01

This will give as an output something like:

*** example output ***
calculation was performed for 7 beads for 35 iterations
The limits in the single variables were:
The search was for a limit 0.5 for the argument RMS Perp Forces
The search was for an procentual limit 0.9 for the argument Bead Energies
This leads to a limit of:
[-61.493200000000002, -61.27216, -60.908589999999997, -60.73498, -61.010310000000004, -61.074560000000005, -61.133600000000001]
The search was for a limit 0.01 for the argument RMS Step Size
It was searched for the point where all the limits were reached
This happens at:
In the beads                      [2, 4, 21, 21, 23, 22, 2]
how often fallen above afterwards [0, 12, 0, 2, 2, 0, 0]
stayed below after:               [2, 26, 21, 26, 26, 22, 2]
for all beads 26 with 0 times fallen above again
but stayed there after 26
***---------------***

The first line of it gives the number of beads and iterations
the next lines give the limits which should be reached in the given variables
For procentual limits the limit is given also in concrete
Then it is said how the limits are connected (only if there are several)
The next three lines belong to the limits in each bead. There should be a value for each of them,
in which the first time the limit is reached is anounced. As in differences (if given by d or written
directly in the output file) the first line normally starts with zeros and thus would be below every reasonable
limit and thus corrupt the result, the search only starts in the second row. Therefore in the not moving beads 0
and the last one, there is normally a 2 to expected, 2 means here either the 2 or the 1 iteration or beiing below
the limit.
If the calculation would not find any limit at all, the output would look something like:

*** example output ***
calculation was performed for 7 beads for 35 iterations
The limit 1e-05 for the argument RMS Perp Forces has been reached
In the beads                      [None, None, None, None, None, None, None]
how often fallen above afterwards [-1, -1, -1, -1, -1, -1, -1]
stayed below after:               [None, None, None, None, None, None, None]
for all beads None with -1 times fallen above again
but stayed there after None
***---------------***

Here the None's say that he has not find anything, -1 is the default value for
not fallen above afterwards
    """
    find = interpretvalues()
    if find.vals == 1:
        result = find.searchlimit(find.st1)
        if find.st1.procent:
            find.outputprocent(find.st1, result)
        else:
            find.output(find.st1, result)
    else:
        if find.merge == 0:
            result = find.searchlimitand(find.stn)
        elif find.merge == 1:
            result = find.searchlimitor(find.stn)
        find.outputall(find.stn, result)

if sys.argv[1] == '--help':
    help(findlimitpath)
else:
    findlimitpath()

