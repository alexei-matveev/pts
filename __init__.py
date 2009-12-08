import searcher
import common
import test
import sched
#import zmatrix
import coord_sys
import qcdrivers
import pes

# locally modified l_bfgs_b optimiser from scipy
import cosopt

# bring through top level functions / classes for users' scripts
from parasearch import neb_calc, string_calc, read_files, generic_callback, dump_steps
from calcman import CalcManager
from asemolinterface import MolInterface

def cleanup(gs):
    """Perform error checking and other stuff on an environment used to run
    a search."""

    # empty for the time being
    pass

