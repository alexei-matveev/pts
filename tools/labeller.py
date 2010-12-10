#!/usr/bin/env python

"""\
Displays a couple of ASE GUIs to facilitate the ordering of atoms in pairs of 
molecules.

Usage:

    $ python labeller.py [-h|--help] mol1.xyz mol2.xyz

After the first window appears, you must click close. This doesn't actually 
close it but allows the second window to open. I know that doesn't make sense,
it must be to do with a bug/quirk in ASE.

When the user clicks on an atom with the MIDDLE button, the program prints to 
stdout the indices of the atoms in the order that they are clicked on, and 
marks the atoms that have already been clicked.

By clicking on atoms in molecule 1, then molecule 2, then molecule 1, etc. the
text which is written to stdout can by used by tools/rotate.py to re-order and
align molecules given in cartesian coordinates.
"""

from ase.gui.images import Images
from ase.gui.gui import GUI

import ase

import sys
import getopt

class Usage(Exception):
    def __init__(self, msg):
        self.msg = msg

def run(args):
    f1 = ase.io.read(args[0])
    f2 = ase.io.read(args[1])
    i1 = Images([f1])
    i2 = Images([f2])

    gui1 = GUI(i1, '', 1, False)
    gui1.run(None)
    gui2 = GUI(i2, '', 1, False)
    gui2.run(None)


def main(argv=None):
    if argv is None:
        argv = sys.argv
    try:
        try:
            opts, args = getopt.getopt(argv[1:], "h", ["help"])
        except getopt.error, msg:
             raise Usage(msg)
        
        if len(args) < 2:
            raise Usage(__doc__)
        run(args)

    except Usage, err:
        print >>sys.stderr, err.msg
        print >>sys.stderr, "for help use --help"
        return 2

if __name__ == "__main__":
    sys.exit(main())



