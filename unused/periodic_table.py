#!/usr/bin/env python
"""
Contains atomic data that can be inquired by atomic number:

    >>> for z in range(1,11):
    ...   print symbol(z),mass(z),radius(z),row(z)
    ...
    H  1.007825 0.32 1
    He 4.002603 0.93 1
    Li 7.016005 1.23 2
    Be 9.012182 0.9 2
    B  11.009305 0.82 2
    C  12.0 0.77 2
    N  14.003074 0.75 2
    O  15.994915 0.73 2
    F  18.998403 0.72 2
    Ne 19.992439 0.71 2

... or by atomic name:

    >>> for a in [ 'H ', 'He', 'Li', 'Be', 'B ', 'C ', 'N ', 'O ', 'F ', 'Ne']:
    ...   print symbol(a),mass(a),radius(a),row(a)
    ...
    H  1.007825 0.32 1
    He 4.002603 0.93 1
    Li 7.016005 1.23 2
    Be 9.012182 0.9 2
    B  11.009305 0.82 2
    C  12.0 0.77 2
    N  14.003074 0.75 2
    O  15.994915 0.73 2
    F  18.998403 0.72 2
    Ne 19.992439 0.71 2

These subroutines have a fallback for a "dummy" atom
that is associated with the atomic number zero:

    >>> print symbol(0),mass(0),radius(0)
    X 0.0 None

The symolic name for dummy atom is "X"

    >>> print symbol('X'),mass('X'),radius('X')
    X 0.0 None

The mass of a dummy atom is set to zero so that it
does not offset mass center or mass tensor.

So far there is no atomic radius associated with a
dumy atom.

"""

__all__ = ["mass", "symbol", "radius", "number", "row"]

def mass(atom):
    return _mass[number(atom)]

def symbol(atom):
    return _symbol[number(atom)]

def radius(atom):
    return _radius[number(atom)]

def number(atom):
    if type(atom) == type(1):
        return atom
    elif type(atom) == type(1.0):
        return int(atom)
    elif type(atom) == type("H"):
        return _number[atom]
    else:
        raise TypeError("Not an atom " ++ str(atom) )

def row(atom):
    """Purpose: return the row of the periodic table where the
    element with atom is located.
    """

    Z = number(atom)

    if Z == 1 or Z == 2:
       return 1
    elif Z >= 3 and Z <= 10:
       return 2
    elif Z >= 11 and Z <= 18:
       return 3
    elif Z >= 19 and Z <= 36:
       return 4
    elif Z >= 37 and Z <= 54:
       return 5
    elif Z >= 55 and Z <= 86:
       return 6
    elif Z >= 87 and Z <= 103:
       return 7
    else:
       raise ValueError("Atomic number out of range: " ++ str(Z))
#end def

_symbol = [ 'X', # for dummy atoms
       'H',  'He', 'Li', 'Be', 'B ', 'C',  'N',  'O',  'F',  'Ne', 'Na',
       'Mg', 'Al', 'Si', 'P ', 'S ', 'Cl', 'Ar', 'K ', 'Ca', 'Sc', 'Ti', 'V',  'Cr',
       'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
       'Rb', 'Sr', 'Y ', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In',
       'Sn', 'Sb', 'Te', 'I',  'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm',
       'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W',  'Re',
       'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra',
       'Ac', 'Th', 'Pa', 'U',  'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm',
       'Md', 'No', 'Lr' ]
       # FIXME: X at 99 is in fact Es!

_number = dict([ (_symbol[z], z) for z in range(len(_symbol)) ])

_mass = [ 0.0,  # for dummy atoms
       1.007825,  4.002603,  7.016005,  9.012182,
       11.009305, 12.000000, 14.003074, 15.994915,
       18.998403, 19.992439, 22.989770, 23.985045,
       26.981541, 27.976928, 30.973763, 31.973909,
       34.968853, 39.962383, 38.963708, 39.962591,
       44.955914, 47.947947, 50.943963, 51.940510,
       54.938046, 55.934939, 58.933198, 57.935347,
       62.929599, 63.929145, 68.925581, 73.921179,
       74.921596, 79.916521, 78.918336, 83.911506,
       84.911800, 87.956250, 88.905856, 89.904708,
       92.906378, 97.905405, 97.907110, 101.904348,
       102.90550, 106.903480, 106.905095, 113.903361,
       114.90388, 119.902199, 120.903824, 129.906230,
       126.904477, 131.90415, 132.905770, 137.905240,
       138.906360, 139.90544, 140.907660, 141.907730,
       144.912691, 151.91974, 152.921240, 157.924110,
       158.92535, 163.929180, 164.930330, 167.930310,
       168.93423, 173.938870, 174.940790, 177.943710,
       180.94801, 183.950950, 186.955770, 191.961490,
       192.96294, 194.964790, 196.966560, 201.970630,
       204.97441, 207.976640, 208.980390, 208.982420,
       219.01130, 222.017570, 223.019734, 226.025406,
       227.02775, 232.038050, 231.035881, 238.050786,
       237.048169, 244.06420, 243.0614  , 247.0703  ,
       247.0703  , 251.0796 , 252.0829  , 257.0951  ,
       258.0986  , 259.1009 , 262.1100 ]
       # older versions used mass 250 for elements > Np and < 99

_radius = [ None,
       0.3200, 0.9300, 1.2300, 0.9000, 0.8200,
       0.7700, 0.7500, 0.7300, 0.7200, 0.7100,
       1.5400, 1.3600, 1.1800, 1.1100, 1.0600,
       1.0200, 0.9900, 0.9800, 2.0300, 1.7400,
       1.4400, 1.3200, 1.2200, 1.1800, 1.1700,
       1.1700, 1.1600, 1.1500, 1.1700, 1.2500,
       1.2600, 1.2200, 1.2000, 1.1600, 1.1400,
       1.1200, 2.1600, 1.9100, 1.6200, 1.4500,
       1.3400, 1.3000, 1.2700, 1.2500, 1.2500,
       1.2800, 1.3400, 1.4800, 1.4400, 1.4100,
       1.4000, 1.3600, 1.3300, 1.3100, 2.3500,
       1.9800, 1.6900, 1.6500, 1.6500, 1.6400,
       1.6300, 1.6200, 1.8500, 1.6100, 1.5900,
       1.5900, 1.5800, 1.5700, 1.5600, 1.5600,
       1.5600, 1.4400, 1.3400, 1.3000, 1.2800,
       1.2600, 1.2700, 1.3000, 1.3400, 1.4900,
       1.4800, 1.4700, 1.4600, 1.4600, 1.4500,
       1.0000, 1.0000, 1.0000, 1.0000, 1.6500,
       1.0000, 1.4200, 1.0000, 1.0000, 1.0000,
       1.0000, 1.0000, 1.0000, 1.0000, 0.8000,
       1.0000, 1.0000, 1.0000 ]

if __name__ == "__main__":
    import doctest
    doctest.testmod()

# Default options for vim:sw=4:expandtab:smarttab:autoindent:syntax
