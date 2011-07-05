# Example paragauss input deck
"""
nuclear   x coordinate     y coordinate     z coordinate    count index of the UNIQUE_ATOM
charge    in bohr          in bohr          in bohr         namelist defining that atom
|         |                |                |               |

8.00      0.000000000000   0.000000000000   0.000000000000  1  1    0  0  0    0  0  0
1.00      1.400000000000   0.000000000000   1.150000000000  2  2    0  0  0    0  0  0
1.00     -1.400000000000   0.000000000000   1.150000000000  2  3    0  0  0    0  0  0
-7.0
                                                               |    |  |  |    |  |  |
 |
                                                atom count index    def. of    numbering
 count index of                                                     internal   of internal
 geometry step                                                      coord.     coordinates
"""

class GXFileInterface():
    def __init__(gxfile):
        pass

    def parse_gxfile(gxstr):
        lines = split(gxstr, "\n")
        line_pat = re.compile(r"(?P<charge>\S+)\s+(?P<xcoord>\S+)\s+(?P<ycoord>\S+)\s+(?P<zcoord>\S+)\s+(?P<ucountix>\d+)\s+(?P<countix>\d+)\s+(?P<ic1>\d+)\s+(?P<ic2>\d+)\s+(?P<ic3>\d+)\s+(?P<ic4>\d+)\s+(?P<ic5>\d+)\s+(?P<ic6>\d+)")

        atoms = []
        for line in lines:
            match = line_pat.search(line)
            if match == None:
                raise Exception("Line not matched: " + line)
            
            else:
                """
                1. find out which coordinates are specified
                2. find their value from the cartesians
                3. construct atom
                """
        """
        1. once list of atoms is built, generate z-matrix
        """
            
