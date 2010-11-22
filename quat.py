"""
Quaternions and rotation matrices here.

    >>> from numpy import pi, round, max, abs, all

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
    >>> all(rotmat(random) == qrotmat(uquat(random)))
    True

Derivatives:
Compare the direct calculated derivatives with the ones from
NumDiff:

    >>> from func import NumDiff

First for the uquat function:

    >>> value1, derivative1 = NumDiff(uquat).taylor(random)

The functions prefixed by underscore give also the
analytical derivative:

    >>> value2, derivative2 = _uquat(random)

    >>> max(abs(derivative2 - derivative1)) < 1e-10
    True

The same for the qrotmat function

    >>> value1, derivative1 = NumDiff(qrotmat).taylor(q)
    >>> value2, derivative2 = _qrotmat(q)
    >>> max(abs(derivative2 - derivative1)) < 1e-10
    True

Rotmat has also an analytical derivative:

    >>> value1, derivative1 = NumDiff(rotmat).taylor(random)
    >>> value2, derivative2 = _rotmat(random)
    >>> max(abs(derivative2 - derivative1)) < 1e-10
    True
"""
from numpy import asarray, empty, dot, sqrt, sin, cos, abs, array, eye, diag, zeros
from numpy import arccos

def uquat(v):
    """Returns unitary quaternion corresponding to rotation vector "v",
    whose length specifies the rotation angle and whose direction
    specifies an axis about which to rotate.

        >>> from numpy import pi

        >>> uquat((0., 0., pi/2.))
        array([ 0.70710678,  0.        ,  0.        ,  0.70710678])

    #  (0.70710678118654757, 0.0, 0.0, 0.70710678118654746)
    """
    uq, __  = _uquat(v)
    return uq


def _uquat(v):
    """
    Exponential of a purely imaginary quaternion:

    q = exp(v) = cos(phi/2) + (v/|v|) * sin(phi/2), phi = |v|

    Note that:

      (v/|v|) * sin(phi/2) = (v/2) * sin(phi/2) / (phi/2)
                           = (v/2) * sinc(phi/2)
    """

    v = asarray(v)

    assert len(v) == 3 # (axial) vector

    phi = sqrt(dot(v, v))

    # real component of the quaternion:
    w = cos(phi/2.)

    # Note: dcos(phi/2)/dv = -sin(phi/2) * 1/2 * (v/|v|)
    dw = - 1/4. * sinc(phi/2.) * v

    # imaginary components:
    x, y, z = v/2. * sinc(phi/2.)

    # dx/v = sinc(phi/2) * 1/2 * dv0/dv + v0/8 * dsinc(phi/2)/dphi/2 * v
    # dy/v and dz/v the same way
    dx = 1/2. * sinc(phi/2.) * array([1, 0, 0]) + v[0]/8. * dsincx(phi/2.) * v
    dy = 1/2. * sinc(phi/2.) * array([0, 1, 0]) + v[1]/8. * dsincx(phi/2.) * v
    dz = 1/2. * sinc(phi/2.) * array([0, 0, 1]) + v[2]/8. * dsincx(phi/2.) * v

    q = asarray([w, x, y, z])

    dq = asarray([dw, dx, dy, dz])

    return q, dq

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

def rotvec(m):
    """
    Given orthogonal rotation matrix |m| compute the corresponding
    rotation vector |v| so that m == rotmat(v). This should always
    hold:

        m == rotmat(rotvec(m))

    But this holds only "modulo 2*pi" for the vector length:

        v == rotvec(rotmat(v))

    Examples:

        >>> from numpy import pi, round

        >>> v = array((0.5, -0.3, 1.0))

        >>> max(abs(rotvec(rotmat(v)) - v)) < 1.0e-10
        True

        >>> max(abs(rotvec(rotmat(-v)) + v)) < 1.0e-10
        True

    Note that the length of the skew-vector is only defined
    modulo 2 pi:

        >>> v = array((0.0, 0.0, 6.0 * pi + 0.1))
        >>> rotvec(rotmat(v))
        array([ 0. ,  0. ,  0.1])
    """

    #
    # cos \phi = ( Tr(m) - 1 ) / 2
    #
    phi = arccos((m[0, 0] + m[1, 1] + m[2, 2] - 1.0) / 2.0)

    #
    # To get axis look at skew-symmetrix matrix (m - m'):
    #
    n = zeros(3)
    if ( phi != 0.0 ):
        n[0] = (m[2, 1] - m[1, 2]) / (2.0 * sin(phi))
        n[1] = (m[0, 2] - m[2, 0]) / (2.0 * sin(phi))
        n[2] = (m[1, 0] - m[0, 1]) / (2.0 * sin(phi))

    return phi * n


def _rotmat(v):
   uq, duq = _uquat(v)
   qrot, dqrot = _qrotmat(uq)

   # rot = qrot(uquat(v))
   # drot/dv = dqrot/du * duquat(u)/dv
   return qrot, dot(dqrot, duq)


def qrotmat(q):
    m, __ = _qrotmat(q)
    return m


def _qrotmat(q):
    assert len(q) == 4 # quaternion!
    #assert abs(dot(q, q) - 1.) < 1e-10 # unitary quaternion!
    # not if as function, only knowm from relation, thus not with NumDiff usable

    a, b, c, d = q

    # transposed:
#   m = [[ a*a + b*b - c*c - d*d, 2*b*c + 2*a*d,         2*b*d - 2*a*c  ],
#        [ 2*b*c - 2*a*d        , a*a - b*b + c*c - d*d, 2*c*d + 2*a*b  ],
#        [ 2*b*d + 2*a*c        , 2*c*d - 2*a*b        , a*a - b*b - c*c + d*d ]]

    # this definition makes quaternion- and matrix multiplicaiton consistent:
    m = empty((3, 3))
    m[...] = [[ a*a + b*b - c*c - d*d, 2*b*c - 2*a*d,         2*b*d + 2*a*c  ],
              [ 2*b*c + 2*a*d        , a*a - b*b + c*c - d*d, 2*c*d - 2*a*b  ],
              [ 2*b*d - 2*a*c        , 2*c*d + 2*a*b        , a*a - b*b - c*c + d*d ]]

    #
    # Derivatives dm  / dq  at dm[i, j, k]
    #               ij    k
    dm = empty((3, 3, 4))

    # factor 2 will be added later:
    dm[..., 0] = [[ a, -d,  c],
                  [ d,  a, -b],
                  [-c,  b,  a]]

    dm[..., 1] = [[ b,  c,  d],
                  [ c, -b, -a],
                  [ d,  a, -b]]

    dm[..., 2] = [[-c,  b,  a],
                  [ b,  c,  d],
                  [-a,  d, -c]]

    dm[..., 3] = [[-d, -a,  b],
                  [ a, -d,  c],
                  [ b,  c,  d]]
    dm *= 2

    return m, dm

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

def dsincx(x):
    """This evaluates in a "numerically stable" fashion

        1    d  sin x
        - * --- -----
        x   dx    x

        >>> dsincx(0.0)
        -0.33333333333333331

        >>> dsincx(0.010001)
        -0.33332999934464169

        >>> dsincx(0.009999)
        -0.3333300006785333
    """
    if abs(x) > 0.01:
        return cos(x) / x**2 - sin(x) / x**3
    else:
        #    (%i2) taylor( cos(x)/x**2 - sin(x)/x**3, x, 0, 9);
        #                              2    4      6        8
        #                         1   x    x      x        x
        #    (%o2)/T/           - - + -- - --- + ----- - ------- + . . .
        #                         3   30   840   45360   3991680

        return - 1/3. + x**2/30. - x**4/840. + x**6/45360. + x**8/3991680.


class Quat(object):
    """Minimal quaternions

        >>> e = Quat()
        >>> i = Quat((0., 1., 0., 0.))
        >>> j = Quat((0., 0., 1., 0.))
        >>> k = Quat((0., 0., 0., 1.))

        >>> i * j == k, j * k == i, k * i == j
        (True, True, True)

        >>> j * i
        Quat((0.0, 0.0, 0.0, -1.0))

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
        "Multiplication of self * other in that order"

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
