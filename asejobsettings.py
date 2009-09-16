from ase.calculators import emt
from ase import Vasp, LennardJones

mycell = [ ( 5.6362,  0.000,   0.000),
                ( 2.8181,  4.881,   0.000),
                ( 0.0000,  0.000,  10.000) ]
mypbc = (True, True, True)

"""mycalc = Vasp( ismear = '1'
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
             )"""

mycalc = emt.EMT()

