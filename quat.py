"""
Quaternions and rotation matrices here.

    >>> from numpy import pi, round, max, abs

Three unitary quaternions:

    >>> qz = uquat((0., 0., pi/4.))
    >>> qy = uquat((0., pi/4., 0.))
    >>> qx = uquat((pi/4., 0., 0.))

Product of three quaternions:

    >>> q = Quat(qx) * Quat(qy) * Quat(qz)

Rotation matrix corresponding to quaternion product:

    >>> print round(qrotmat(q), 7)
    [[ 0.5       -0.5        0.7071068]
     [ 0.8535534  0.1464466 -0.5      ]
     [ 0.1464466  0.8535534  0.5      ]]

Three rotation matrices:

    >>> mz = qrotmat(qz)
    >>> my = qrotmat(qy)
    >>> mx = qrotmat(qx)

Product of three matrices:

    >>> m = dot(mx, dot(my, mz))
    >>> print round(m, 7)
    [[ 0.5       -0.5        0.7071068]
     [ 0.8535534  0.1464466 -0.5      ]
     [ 0.1464466  0.8535534  0.5      ]]

    >>> max(abs(m - qrotmat(q))) < 1e-10
    True

The funciton rotmat(v) is a composition of two:

    rotmat(v) = qrotmat(uquat(v))

    >>> random = (0.1, 0.8, 5.)
    >>> (rotmat(random) == qrotmat(uquat(random))).all()
    True

"""
from numpy import asarray, empty, dot, sqrt, sin, cos, abs

def uquat(v):
    """Returns unitary quaternion corresponding to rotation vector "v",
    whose length specifies the rotation angle and whose direction
    specifies an axis about which to rotate.

        >>> from numpy import pi

        >>> uquat((0., 0., pi/2.))
        (0.70710678118654757, 0.0, 0.0, 0.70710678118654746)
    """

    v = asarray(v)

    assert len(v) == 3 # (axial) vector

    phi = sqrt(dot(v, v))

    #   u = v * sin(phi/2) / phi
    #     = (v/2) * sin(phi/2) / (phi/2)
    #     = (v/2) * sinc(phi/2)

    # real component of the quaternion:
    w = cos(phi/2.)

    # imaginary components:
    x, y, z = v/2. * sinc(phi/2.)

    return (w, x, y, z)

def rotmat(v):
    """Generates rotation matrix based on vector v, whose length specifies 
    the rotation angle and whose direction specifies an axis about which to
    rotate.

        >>> from numpy import pi, round

        >>> print round(rotmat((0., 0., pi/2.)), 7)
        [[ 0. -1.  0.]
         [ 1.  0.  0.]
         [ 0.  0.  1.]]

        >>> print round(rotmat((0., pi/2., 0.)), 7)
        [[ 0.  0.  1.]
         [ 0.  1.  0.]
         [-1.  0.  0.]]

        >>> print round(rotmat((pi/2., 0., 0.)), 7)
        [[ 1.  0.  0.]
         [ 0.  0. -1.]
         [ 0.  1.  0.]]

        >>> print round(rotmat((0., 0., pi/4.)), 7)
        [[ 0.7071068 -0.7071068  0.       ]
         [ 0.7071068  0.7071068  0.       ]
         [ 0.         0.         1.       ]]
    """

    return qrotmat(uquat(v))

def qrotmat(q):

    assert len(q) == 4 # quaternion!
    assert abs(dot(q, q) - 1.) < 1e-10 # unitary quaternion!

    a, b, c, d = q

    # transposed:
#   m = [[ a*a + b*b - c*c - d*d, 2*b*c + 2*a*d,         2*b*d - 2*a*c  ],
#        [ 2*b*c - 2*a*d        , a*a - b*b + c*c - d*d, 2*c*d + 2*a*b  ],
#        [ 2*b*d + 2*a*c        , 2*c*d - 2*a*b        , a*a - b*b - c*c + d*d ]]

    # this definition makes quaternion- and matrix multiplicaiton consistent:
    m = [[ a*a + b*b - c*c - d*d, 2*b*c - 2*a*d,         2*b*d + 2*a*c  ],
         [ 2*b*c + 2*a*d        , a*a - b*b + c*c - d*d, 2*c*d - 2*a*b  ],
         [ 2*b*d - 2*a*c        , 2*c*d + 2*a*b        , a*a - b*b - c*c + d*d ]]

    return asarray(m)

def sinc(x):
    """sinc(x) = sin(x)/x

        >>> sinc(0.0)
        1.0

        >>> sinc(0.010001)
        0.99998333008319973

        >>> sinc(0.009999)
        0.99998333674979978
    """

    if abs(x) > 0.01:
        return sin(x) / x
    else:
        #   (%i1) taylor(sin(x)/x, x, 0, 8);
        #   >                           2    4      6       8
        #   >                          x    x      x       x
        #   > (%o1)/T/             1 - -- + --- - ---- + ------ + . . .
        #   >                          6    120   5040   362880
        #
        #   Below x < 0.01, terms greater than x**8 contribute less than 
        #   1e-16 and so are unimportant for double precision arithmetic.

        return 1 - x**2/6. + x**4/120. - x**6/5040. + x**8/362880.

class Quat(object):
    """Minimal quaternions

        >>> e = Quat()
        >>> i = Quat((0., 1., 0., 0.))
        >>> j = Quat((0., 0., 1., 0.))
        >>> k = Quat((0., 0., 0., 1.))
        >>> e * i == i, e * j == j, e * k == k
        (True, True, True)
        >>> i * i
        Quat((-1.0, 0.0, 0.0, 0.0))
        >>> j * j
        Quat((-1.0, 0.0, 0.0, 0.0))
        >>> k * k
        Quat((-1.0, 0.0, 0.0, 0.0))
    """
    def __init__(self, q=(1., 0., 0., 0.)):
        self.__q = asarray(q)

    def __len__(self): return 4

    def __getitem__(self, i): return self.__q[i]

    def __mul__(self, other):
        p = self.__q
        q = other.__q

        r = empty(4)

        r[0] = p[0] * q[0] - p[1] * q[1] - p[2] * q[2] - p[3] * q[3] 

        r[1] = p[0] * q[1] + p[1] * q[0] + p[2] * q[3] - p[3] * q[2] 
        r[2] = p[0] * q[2] + p[2] * q[0] - p[1] * q[3] + p[3] * q[1] 
        r[3] = p[0] * q[3] + p[3] * q[0] + p[1] * q[2] - p[2] * q[1] 
        return Quat(r)

    def __repr__(self):
        return "Quat(%s)" % str(tuple(self.__q))

    __str__ = __repr__

    def __eq__(self, other):
        return (self.__q == other.__q).all()

# "python quat.py", eventualy with "-v" option appended:
if __name__ == "__main__":
    import doctest
    doctest.testmod()

# Default options for vim:sw=4:expandtab:smarttab:autoindent:syntax
