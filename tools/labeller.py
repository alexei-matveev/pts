#!/usr/bin/env python

"""Displays a couple of ASE GUIs to facilitate the ordering of atoms in pairs of molecules."""

from ase.gui.images import Images
from ase.gui.gui import GUI

import ase

import sys
import getopt

class Usage(Exception):
    def __init__(self, msg):
        self.msg = msg

def run(args):
    f1 = ase.read(args[0])
    f2 = ase.read(args[1])
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
            raise Usage("Must specify two files")
        run(args)

    except Usage, err:
        print >>sys.stderr, err.msg
        print >>sys.stderr, "for help use --help"
        return 2

if __name__ == "__main__":
    sys.exit(main())



