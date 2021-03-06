from ase.calculators import emt
from ase import Vasp, LennardJones
import numpy

mycell = numpy.array([[ 5.59180043,  0.        ,  0.        ],
       [ 2.79590021,  4.84264122,  0.        ],
       [ 0.        ,  0.        ,  5.59180043]])

mypbc = numpy.array([ True,  True,  True], dtype=bool)

mycalc = Vasp( ismear = '1'
             , sigma  = '0.15'
             , xc     = 'VWN'
             , isif   = '2'
             , enmax  = '300'
             , idipol = '3'
             , enaug  = '300'
             , ediffg = '-0.02'
             , voskown= '1'
             , istart = '1'
             , icharg = '1'
             , nelmdl = '0'
             , kpts   = [1,1,1]
             )

# enable this for testing
#mycalc = emt.EMT()

