from numpy import array, sqrt, dot, cross
from numpy import pi, arccos
"""
Attention: only working with a special set of coordinates with:

Ar
Ar 1 var1
Ar 1 var2 2 var3
Ar 2 var4 3 var5 1 var6
"""
def vn(v):
    return float(sqrt(dot(v,v)))


def t_c2v(x):
    """
    variables as stated above
    constraints 1 and 2: two length should be equal
    constraint 3: adding up of the two angles
    on the sixth coordinate (dihedral angle) there is
    no constraint
    Checking for the constraints in real number and relavive
    """
    assert len(x) > 5
    c1 = abs(x[0] - x[1])
    c2 = abs(x[0] - x[3])
    c3 = abs(x[2] +  2. * x[4] - pi)
    return c1, c2, c3, c1/abs(x[0]), c2/abs(x[0]), abs(c3/x[2])

def t_c2v_prime(x):
    """
    Checking if a change in internal coordinates as above
    would disturb the symmetry. Here x is the change.
    first checking for real number than relative
    """
    assert len(x) > 5
    c1 = abs(x[0] - x[1])
    c2 = abs(x[0] - x[3])
    c3 = abs(x[2] +  2. * x[4])
    lenx = sqrt(dot(x, x))
    return [c1, c2, c3] , [c1/lenx, c2/lenx, c3/lenx]


def t_c2v_cartforces(f2, x):
    """
    Given the four cartesian force vectors (f) and the four
    positions of the atoms, check if the forces are respecting
    C2V symmetry.
    Check if the forces along the diagonal are equal on both sides
    and if the remaining forces are orthogonal to the direction of
    the perpendicular part of the other diagonal.
    """
    # The two diagonals
    v14 = x[3] - x[0]
    v14 /= vn(v14)
    v23 = x[2] - x[1]
    v23 /= vn(v23)


    # remove global translation through forces
    midf = sum(f2)/ 4.
    f = f2 - midf

    # Forces projected on the diagonals going through the atoms place
    f14 = dot(f[0],v14)
    f41 = dot(f[3], v14)
    f23 = dot(f[1], v23)
    f32 = dot(f[2], v23)

    # First two constraints: both atoms at a diagonal have to have
    # forces projected on it with the same length but different
    # direction
    c1 = f14 + f41
    c2 = f23 + f32

    # There should be no forces in direction of the other diagonal
    # if the diagonals are not perpendicular to each other, this has
    # to be considered too
    c3 = dot(f[0], v23) - f14 * dot(v14, v23)
    c4 = dot(f[3], v23) - f41 * dot(v14, v23)
    c5 = dot(f[2], v14) - f32 * dot(v14, v23)
    c6 = dot(f[1], v14) - f23 * dot(v14, v23)

    # Return the constraints as well as the relative ones on them
    return [[abs(c1), abs(c2), abs(c3), abs(c4), abs(c5), abs(c6)] ,
           [ abs(c1 / f14), abs(c2 /f23), abs(c3 / vn(f[0])),
            abs(c4 / vn(f[3])), abs(c5 / vn(f[2])), abs(c6 / vn(f[1]))]]

def t_td(x):
    """
    variables as stated above
    constraints 1 and 2: two length should be equal
    constraints 3 and 4: the angles should be 60
    constraint 5: dihedral angle = arrcos 1/3
    """
    assert len(x) > 5
    c1 = abs(x[0] - x[1])
    c2 = abs(x[0] - x[3])
    c3 = abs(x[2] - pi/3. )
    c4 = abs(x[4] - pi/3. )
    c5 = abs(abs(x[5]) - arccos(1./3.) )
    return [c1, c2, c3, c4, c5], [c1/abs(x[0]), c2/abs(x[0]), c3/abs(x[2]), c4 / abs(x[4]), c5 / abs(x[5])]

def t_td_prime(x):
    """
    variables as stated above
    """
    assert len(x) > 5
    c1 = abs(x[0] - x[1])
    c2 = abs(x[0] - x[3])
    c3 = abs(x[2])
    c4 = abs(x[4])
    c5 = abs(x[5])
    lenx = sqrt(dot(x, x))
    return [c1, c2, c3,c4, c5] ,[c1/lenx, c2/lenx, c3/lenx, c4/lenx, c5 / lenx]

def t_td_cartforces(f2, x):
    """
    Given the four cartesian force vectors (f) and the four
    positions of the atoms, check if the forces are respecting
    C2V symmetry.
    Check if the forces along the diagonal are equal on both sides
    and if the remaining forces are orthogonal to the direction of
    the perpendicular part of the other diagonal.
    """
    # The two diagonals
    v14 = x[3] - x[0]
    v14 /= vn(v14)
    v23 = x[2] - x[1]
    v23 /= vn(v23)


    # remove global translation through forces
    midf = sum(f2)/ 4.
    f = f2 - midf

    # Forces projected on the diagonals going through the atoms place
    f14 = dot(f[0],v14)
    f41 = dot(f[3], v14)
    f23 = dot(f[1], v23)
    f32 = dot(f[2], v23)

    # First two constraints: both atoms at a diagonal have to have
    # forces projected on it with the same length but different
    # direction
    c1 = f14 + f41
    c2 = f23 + f32
    c3 = f14 + f32

    # There should be no forces in direction of the other diagonal
    # if the diagonals are not perpendicular to each other, this has
    # to be considered too
    c4 = dot(f[0], v23) - f14 * dot(v14, v23)
    c5 = dot(f[3], v23) - f41 * dot(v14, v23)
    c6 = dot(f[2], v14) - f32 * dot(v14, v23)
    c7 = dot(f[1], v14) - f23 * dot(v14, v23)

    # Return the constraints as well as the relative ones on them
    return [[abs(c1), abs(c2), abs(c3), abs(c4), abs(c5), abs(c6), abs(c7)] ,
           [ abs(c1 / f14), abs(c2 /f23), abs(c3 / f14), abs(c4 / vn(f[0]))
           , abs(c5 / vn(f[3])), abs(c6 / vn(f[2])), abs(c7 / vn(f[1]))]]
