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
from numpy import trace, pi, finfo, cross
from numpy import max
from numpy import outer, hstack, vstack
from func import Func

__all__ = ["r3", "reper"]

#FIXME: have a bit more to decide?
machine_precision = finfo(float).eps * 2

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
    # Tr(m) = 1 + 2 cos(phi):
    #
    phi = arccos((trace(m) - 1.0) / 2.0)

    #
    # To get axis look at skew-symmetrix matrix (m - m'):
    #
    n = zeros(3)
    n[0] = m[2, 1] - m[1, 2]
    n[1] = m[0, 2] - m[2, 0]
    n[2] = m[1, 0] - m[0, 1]

    #
    # FIXME: problem with angles close to |pi|:
    #
    return n / ( 2.0 * sinc(phi))


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

def cart2rot(v1, v2):
    """
    v1 and v2 are two three point-objects
    Here a rotation matrix is build, which rotatats
    v1 on v2
    (For the coordinate objects we have:
    C2 = MAT * C1.T)

        >>> from numpy import max, abs

        >>> vec1 = array([[0., 0., 0.], [0., 0., 1.], [0., 1., 0.]])
        >>> vec2 = array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])

    See if identity is reproduced:

        >>> m = cart2rot(vec1, vec1)
        >>> max(abs(m - eye(3))) < 1e-15
        True

    Rotation matrix that brings vec1 to vec2:

        >>> m = cart2rot(vec1, vec2)

    This way rotation matrix is applied to a vector:

        >>> transform = lambda v: dot(m, v)

        >>> max(abs(vec2 - array(map(transform, vec1)))) < 1e-15
        True

    """

    c1 = reper([v1[1] - v1[0], v1[2] - v1[0]])
    c2 = reper([v2[1] - v2[0], v2[2] - v2[0]])

    return dot(c2.T, c1)

def rot2quat(mat):
    """
    transforms a rotation matrix mat in a quaternion
    there should be two different quaternions belonging
    to the same matrix, we take the positive one

    >>> from numpy import max, abs

    >>> mat1 = eye(3)
    >>> max(abs(mat1 - qrotmat(rot2quat(mat1)))) < 1e-12
    True

    >>> psi = pi/2.
    >>> mat2 = array([[ 1., 0., 0.], [0., cos(psi), -sin(psi)], [ 0., sin(psi), cos(psi)]])
    >>> max(abs(mat2 - qrotmat(rot2quat(mat2)))) < 1e-12
    True

    >>> mat3 = array([[ 1., 0., 0.], [0., -1., 0.], [ 0., 0., -1.]])
    >>> max(abs(mat3 - qrotmat(rot2quat(mat3)))) < 1e-12
    True

    >>> psi = pi - 0.3
    >>> mat4 = array([[ 1., 0., 0.], [0., cos(psi), -sin(psi)], [ 0., sin(psi), cos(psi)]])
    >>> max(abs(mat4 - qrotmat(rot2quat(mat4)))) < 1e-12
    True

    >>> psi = pi
    >>> mat5 = array([[ cos(psi), 0., sin(psi)], [0., 1, 0.], [ -sin(psi), 0, cos(psi)]])
    >>> max(abs(mat5 - qrotmat(rot2quat(mat5)))) < 1e-12
    True

    >>> ql = array([0,0,0.,1.])
    >>> mat6 = qrotmat(ql)
    >>> max(abs(ql - rot2quat(mat6))) < 1e-12
    True

    Code from: Ken Shoemake "Animation rotation with quaternion curves.",
    Computer Graphics 19(3):245-254, 1985
    """
    qu = zeros(4)
    s = 0.25 * (1 + trace(mat))
    if s > machine_precision:
        qu[0] = sqrt(s)
        qu[1] =  (mat[2,1] - mat[1,2]) / (4 * qu[0])
        qu[2] =  (mat[0,2] - mat[2,0]) / (4 * qu[0])
        qu[3] =  (mat[1,0] - mat[0,1]) / (4 * qu[0])
    else:
        # (qu[0] = 0)
        s = -0.5 * (mat[1,1] + mat[2,2])
        if s > machine_precision:
            qu[1] = sqrt(s)
            qu[2] = mat[1,0] / (2 * qu[1])
            qu[3] = mat[2,0] / (2 * qu[1])
        else:
            # (qu[1] = 0)
            s = 0.5 * (1 - mat[2,2])
            if s > machine_precision:
                qu[2] = sqrt(s)
                qu[3] = mat[2,1] / ( 2 * qu[2])
            else:
                # qu[2] = 0
                qu[3] = 1

    return qu

def cart2quat(v1, v2):
    """
    Gives back the quaternion, belonging to the rotation from
    v1 onto v2, where v1 and v2 are each three points, defining an
    plane (and the top of the plane)

    >>> vec1 = array([[0.,0,0],[0,0,1],[0,1,0]])
    >>> vec2 = array([[0.,0,0],[1,0,0],[0,1,0]])

    >>> ql = cart2quat(vec1, vec2)
    >>> m1 = qrotmat(ql)
    >>> transform = lambda vec3d: dot(m1, vec3d)
    >>> max(abs(vec2 - array(map(transform, vec1)))) < 1e-15
    True
    """
    return rot2quat(cart2rot(v1, v2))

def quat2vec(qa):
    """
    Gives back a vector, as specified by the quaternion q,
    in the representation of length(vec) = rot_angle,
    V/ |v| is vector to rotate around

    >>> v = [0., 0., pi/2.]
    >>> max(abs(v - quat2vec(uquat(v)))) < 1e-12
    True

    >>> v = [0, 0., 0.]
    >>> max(abs(v - quat2vec(uquat(v)))) < 1e-12
    True

    >>> v = [1., 2, 3]
    >>> max(abs(v - quat2vec(uquat(v)))) < 1e-12
    True

    >>> v = [ 0., 0, pi * 6 + pi/2]
    >>> abs(v - quat2vec(uquat(v)))[2] / pi - 8.0 < 1e-12
    True
    """
    qaa = asarray(qa)[1:]
    ang = arccos(qa[0]) * 2
    if abs(dot(qaa, qaa)) == 0:
        lqaa = 1
    else:
        lqaa = sqrt(dot(qaa, qaa))
    # give back as vector
    vall = array([qai/lqaa * ang for qai in qaa])
    return vall

def cart2vec( vec1, vec2):
    """
    given two three point objects vec1 and vec2
    calculates the vector representing the rotation

    >>> vec1 = array([[0.,0,0],[0,0,1],[0,1,0]])
    >>> vec2 = array([[0.,0,0],[1,0,0],[0,1,0]])

    >>> v = cart2vec(vec1, vec2)
    >>> m1 = rotmat(v)
    >>> transform = lambda vec3d: dot(m1, vec3d)
    >>> max((abs(vec2 - array(map(transform, vec1))))) < 1e-15
    True

    >>> vec3 = array([[0.,0,0],[0,0,1],[0,1,0]])
    >>> v2 = cart2vec(vec1, vec3)
    >>> m2 = rotmat(v2)
    >>> transform = lambda vec3d: dot(m2, vec3d)
    >>> max(abs(vec3 - array(map(transform, vec1)))) < 1e-15
    True
    """
    return quat2vec(cart2quat(vec1, vec2))

def cart2veclin(v1, v2):
    """
    v1 and v2 are two two point-objects
    Here a rotation matrix is build, which rotatats
    v1 on v2
    (For the coordinate objects we have:
    C2 = MAT * C1.T)

    >>> vec1 = array([[0.,0,0],[0,0,1]])
    >>> vec2 = array([[0.,0,0],[1,0,0]])

    >>> v = cart2veclin(vec1, vec2)
    >>> m1 = rotmat(v)
    >>> transform = lambda vec3d: dot(m1, vec3d)
    >>> max(abs(vec2 - array(map(transform, vec1)))) < 1e-15
    True

    >>> vec3 = array([[0.,0,0],[1,0,0]])
    >>> vec4 = array([[0.,0,0],[0,1,0]])
    >>> vec5 = array([[0.,0,0],[0,0,1]])

    >>> v = cart2veclin(vec3, vec4)
    >>> m2 = rotmat(v)
    >>> transform = lambda vec3d: dot(m2, vec3d)
    >>> max(abs(vec4 - array(map(transform, vec3)))) < 1e-15
    True
    >>> vec5 = array([[0.,0,0],[0,0,1]])
    >>> max(abs(vec5 - array(map(transform, vec5)))) < 1e-15
    True

    >>> v = cart2veclin(vec3, vec3)
    WARNING: two objects are alike
    >>> m2 = rotmat(v)
    >>> transform = lambda vec3d: dot(m2, vec3d)
    >>> max(abs(vec3 - array(map(transform, vec3)))) < 1e-15
    True

    >>> v = cart2veclin(vec4, vec4)
    WARNING: two objects are alike
    >>> m2 = rotmat(v)
    >>> transform = lambda vec3d: dot(m2, vec3d)
    >>> max(abs(vec4 - array(map(transform, vec4)))) < 1e-15
    True

    >>> v = cart2veclin(vec5, vec5)
    WARNING: two objects are alike
    >>> m2 = rotmat(v)
    >>> transform = lambda vec3d: dot(m2, vec3d)
    >>> max(abs(vec5 - array(map(transform, vec5)))) < 1e-15
    True
    """
    assert (v1[0] == v2[0]).all()

    vec1 = zeros((3,3))
    vec2 = zeros((3,3))
    vec1[0] = v1[0]
    vec1[1] = v1[1]
    vec2[0] = v2[0]
    vec2[1] = v2[1]

    vec1[2] = v2[1]

    if (abs(v1[1] - v2[1]) < machine_precision).all():
        print "WARNING: two objects are alike"
        # in this case there should be no rotation at all
        # thus give back zero vector
        return array([0., 0,0])
    else:
        n2 = planenormal(vec1)

    vec1[2] = n2 - v1[0]
    vec2[2] = n2 - v2[0]
    return quat2vec(cart2quat(vec1, vec2))


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

def planenormal(threepoints):
    """
    gives back normalised plane as defined by the
    three points stored in numpy array threepoints
    """
    v_1 = asarray(threepoints[1] - threepoints[0])
    v_1 /= sqrt(dot(v_1, v_1))
    v_2 = asarray(threepoints[2] - threepoints[0])
    v_2 /= sqrt(dot(v_2, v_2))
    n = cross(v_1, v_2 - v_1 )
    n_norm = sqrt(dot(n, n))
    if n_norm != 0:
        n /= n_norm
    return n

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

def unit(v):
    "Normalize a vector"
    n = sqrt(dot(v, v))
    # numpy will just return NaNs:
    if n == 0.0: raise Exception("divide by zero")
    return v / n

def M(x):
    "M_ij = delta_ij - x_i * x_j / x**2"
    n = unit(x)
    return eye(3) - outer(n, n)

def E(x):
    """E_ij = epsilon_ijk * x_k (sum over k)

        >>> print E([1, 2, 3])
        [[ 0.  3. -2.]
         [-3.  0.  1.]
         [ 2. -1.  0.]]
    """
    e = zeros((3,3))

    e[0, 1] = x[2]
    e[1, 2] = x[0]
    e[2, 0] = x[1]

    e[1, 0] = - e[0, 1]
    e[2, 1] = - e[1, 2]
    e[0, 2] = - e[2, 0]
    return e

class _Reper(Func):
    """Returns orthogonal basis [i, j, k] where
    "k" is parallel to U
    "i" is in UV plane and
    "j" is orthogonal to that plane

    FIXME: there must be a better way to do this ...

    Example:
        >>> from numpy import max, abs
        >>> from func import NumDiff

        >>> r = _Reper()
        >>> u = array((1., 0., 0.))
        >>> v = array((0., 1., 0.))
        >>> print r([u, v])
        [[-0.  1.  0.]
         [ 0.  0. -1.]
         [ 1.  0.  0.]]

        >>> u = array((1.1, 0.3, 0.7))
        >>> v = array((0.5, 1.9, 0.8))
        >>> uv = [u, v]

        >>> r1 = NumDiff(r)
        >>> max(abs(r.fprime(uv) - r1.fprime(uv))) < 1e-10
        True
    """

    def taylor(self, args):

        u, v = args

        lu = sqrt(dot(u, u))
        lv = sqrt(dot(v, v))

        if lu == 0.0: raise Exception("divide by zero")
        if lv == 0.0: raise Exception("divide by zero")

        #
        # Unit vectors for local coordiante system:
        #

        # unit vector in U-direciton:
        k = u / lu

        # dk/du:
        ku = M(k) / lu

        # dk/dv:
        kv = zeros((3,3))

        # orthogonal to UV plane:
        w = cross(v, u)
        lw = sqrt(dot(w, w))

        j = w / lw

        # dj/dw
        jw = M(j) / lw

        # dj/du:
        jv = dot(jw, E(u))

        # dj/dv:
        ju = dot(jw, E(-v))

        # in UV-plane, orthogonal to U:
        i = cross(k, j) # FIXME: not cross(j, k)!

        # di/du = di/dk * dk/du + di/dj * dj/du:
        iu = dot(E(j), ku) + dot(E(-k), ju)

        # di/du =                 di/dj * dj/du:
        iv =                 dot(E(-k), jv)

        f = array([i, j, k])

        # convention: fprime[i, k] = df_i / dx_k
        iprime = hstack((iu, iv))
        jprime = hstack((ju, jv))
        kprime = hstack((ku, kv))

        fprime = vstack((iprime,\
                         jprime,\
                         kprime))

        return f, fprime.reshape((3, 3, 2, 3))

# one instance of Reper(Func):
reper = _Reper()

class _R3(Func):
    """Spherical to cartesian transformation.

        >>> from func import NumDiff

        >>> r3 = _R3()

        >>> vz = (8., 0., 0.)
        >>> vx = (8., pi/2., 0.)
        >>> vy = (8., pi/2., pi/2.)

        >>> print r3(vz)
        [ 0.  0.  8.]

        >>> from numpy import round, max, abs

        >>> print round(r3(vx), 4)
        [ 8.  0.  0.]

        >>> print round(r3(vy), 4)
        [ 0.  8.  0.]

        >>> r4 = NumDiff(r3)
        >>> max(abs(r3.fprime(vz) - r4.fprime(vz))) < 1e-10
        True
        >>> max(abs(r3.fprime(vx) - r4.fprime(vx))) < 1e-10
        True
        >>> max(abs(r3.fprime(vy) - r4.fprime(vy))) < 1e-10
        True
    """

    def f(self, args):

        r, theta, phi = args

        z = r * cos(theta)
        x = r * sin(theta) * cos(phi)
        y = r * sin(theta) * sin(phi)

        return array([x, y, z])

    def fprime(self, args):

        r, theta, phi = args

        ct, st =  cos(theta), sin(theta)
        cp, sp =  cos(phi),   sin(phi)

        z = ct
        x = st * cp
        y = st * sp

        fr = array([x, y, z])

        z = - st
        x = + ct * cp
        y = + ct * sp

        ft = array([x, y, z]) * r

        z = 0.0
        x = - st * sp
        y = + st * cp

        fp = array([x, y, z]) * r

        # convention: fprime[i, k] = df_i / dx_k
        return array([fr, ft, fp]).transpose()

# one instance of _R3(Func):
r3 = _R3()

def test():
    """
    interactive test for the rotation vector,
    uses the next three arguments to build an initial
    vector, describing a rotation.

    Using a random vector from the standard input, generates
    the rotation belonging to the coresponding quaternion for it
    Then uses a rotated inital vector to generate the quaternion
    in the new way
    prints comparisions
    """
    from sys import argv
    v_random = array([0, 0, 0])
    v_random[0] = float(argv[1])
    v_random[1] = float(argv[2])
    v_random[2] = float(argv[3])
    #v_random = v_random/sqrt(dot(v_random, v_random))

    quat_random = uquat(v_random)
    mat_random = rotmat(v_random)
    v_init = array([[0.,0,0],[0,0,1],[0,1,0]])
    v_init = array([[0.,0,0],[1,0,0],[0,1,0]])
    v_end = array([[0.,0,0],[0,0,0],[0,0,0]])
    for i, v in enumerate(v_init):
        v_end[i] = dot(mat_random, v)
    vec_calc = cart2vec(v_init, v_end)
    quat_calc = uquat(vec_calc)
    print "Quaternions: given, calculated back"
    print quat_random
    print quat_calc
    print "differences are", quat_random - quat_calc
    print "Vectors: start and end"
    print v_random
    print vec_calc
    print "differences are", v_random - vec_calc
    print "Angles"
    print sqrt(dot(v_random, v_random))
    print sqrt(dot(vec_calc, vec_calc))
    print "Does rotation work?"
    m1 = rotmat(vec_calc)
    transform = lambda vec3d: dot(m1, vec3d)
    print (abs(v_end - array(map(transform, v_init))) < 1e-15).all()

# "python quat.py", eventualy with "-v" option appended:
if __name__ == "__main__":
    import doctest
    doctest.testmod()

# Default options for vim:sw=4:expandtab:smarttab:autoindent:syntax
