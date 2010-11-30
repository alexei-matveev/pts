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

def vec_to_coord_mat(threepoints):
    """
    generates a coordinate system out of three points defining
    vector and plane for the first two directions
    returns the result as an matrix

    >>> vec1 = array([[0.,0,0],[0,0,1],[0,1,0]])
    >>> abs(dot(planenormal(vec1), vec1[1] - vec1[0])) < 1e-10
    True
    >>> abs(dot(planenormal(vec1), vec1[2] - vec1[0])) < 1e-10
    True
    >>> abs(dot(planenormal(vec1), planenormal(vec1)) - 1) < 1e-10
    True

    >>> vec1 = array([[0.,0,0],[1,0,1],[2,1,0]])
    >>> abs(dot(planenormal(vec1), vec1[1] - vec1[0])) < 1e-10
    True
    >>> abs(dot(planenormal(vec1), vec1[2] - vec1[0])) < 1e-10
    True
    >>> abs(dot(planenormal(vec1), planenormal(vec1)) - 1) < 1e-10
    True
    """
    coords = zeros((3,3))
    x1, y1, z1 = vec_to_coord(threepoints)
    assert abs(dot(x1, x1) -1) < 1e-12
    assert abs(dot(y1, y1) -1) < 1e-12
    assert abs(dot(z1, z1) -1) < 1e-12

    coords[0,:] = x1
    coords[1,:] = y1
    coords[2,:] = z1
    return coords

def cart2rot(v1, v2):
    """
    v1 and v2 are two three point-objects
    Here a rotation matrix is build, which rotatats
    v1 on v2
    (For the coordinate objects we have:
    C2 = MAT * C1.T)

    >>> from numpy import max, abs

    >>> vec1 = array([[0.,0,0],[0,0,1],[0,1,0]])
    >>> vec2 = array([[0.,0,0],[1,0,0],[0,1,0]])
    >>> m1 = cart2rot(vec1, vec2)
    >>> m2 = cart2rot(vec1, vec1)
    >>> transform = lambda vec3d: dot(m1, vec3d)
    >>> max(abs(vec2 - array(map(transform, vec1)))) < 1e-15
    True
    >>> max(abs(m2 - eye(3))) < 1e-15
    True
    """
    c1 = vec_to_coord_mat(v1)
    c2 = vec_to_coord_mat(v2)
    mat = dot(c2.T, c1)
    return mat

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
    >>> (abs(vec2 - array(map(transform, vec1)))).all() < 1e-15
    True
    """
    return rot2quat(cart2rot(v1, v2))

def quat2vec(qa):
    """
    Gives back a vector, as specified by the quaternion q,
    in the representation of length(vec) = rot_angle,
    V/ |v| is vector to rotate around

    >>> v = [0., 0., pi/2.]
    >>> (abs(v - quat2vec(uquat(v)))).all() < 1e-12
    True

    >>> v = [0, 0., 0.]
    >>> (abs(v - quat2vec(uquat(v)))).all() < 1e-12
    True

    >>> v = [1., 2, 3]
    >>> (abs(v - quat2vec(uquat(v)))).all() < 1e-12
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

def vec_to_coord(threepoints):
    """
    generates a three coordinate system from three points,
    where the first vector is in direction points[1] - points[0]
    the second vector is in the plane spanned by the three points
    while the third one is the normal vector to this plane

    >>> v_init = array([[0.,0,0],[0,0,1],[0,1,0]])
    >>> vec_to_coord(v_init)
    (array([ 0.,  0.,  1.]), array([-0.,  1., -0.]), array([ 1., -0., -0.]))
    >>> v_init = array([[0.,0,0],[1,0,0],[0,1,0]])
    >>> vec_to_coord(v_init)
    (array([ 1.,  0.,  0.]), array([-0.,  1.,  0.]), array([-0.,  0., -1.]))
    >>> v_init = array([[0.,0,0],[1,2,0],[0.5,1.2,0]])
    >>> vec = vec_to_coord(v_init)
    >>> sqrt(dot(vec[0], vec[0])) - 1 < 10e-10
    True
    """
    x1 = asarray(threepoints[1] - threepoints[0])
    x1 /= sqrt(dot(x1, x1))
    z1 = -planenormal(threepoints)
    assert abs(dot(z1, z1) -1) < 1e-12
    #FIXME: other order than in rest of code?
    y1 = -cross(z1, x1)
    y1 /= sqrt(dot(y1, y1))

    return x1, y1, z1



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
