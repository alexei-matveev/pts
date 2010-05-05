#!/usr/bin/python
"""
Motivation: Tensor (aka Kronecker, aka outer) product works like this

    C     =     A     *   B
     ij..pq..    ij..      pq..

e.g. all products of elements of rank-2 tensor  A
                                                 ij

with all elements of rank-3 tensor  B    form a rank-5 tensor  C     .
                                     pqr                        ijpqr

Most notably products with rank-0 tensors (numbers with no indices)
does not increase the tensor rank. In this respect the numpy.outer()
behaves counter intuitive, e.g. numpy.outer(2, 3) == array([[6]]),
is a rank-2 array (albeit with both index ranges limited to one value).

Matrix product can be viewed as a tensor product followed
by contraction over innermost indices, e.g.

    C   =  A   *  B     (sum over repeated k)
     mn     mk     kn

The contraction step can be extended to several indices.

In |matmul| we restrict ourselves to contracting over
several INNERMOST indices. This differs from the conventions
used by dot- and inner product funcitons in NumPy.

See also:

    numpy.outer
    numpy.kron
    numpy.dot
    numpy.inner
    numpy.matrixmultiply
"""

__all__ = ["matmul", "outer", "dots", "sums"]

from numpy import asarray, empty, shape, dot, sum

def prod(ns): # name clash with numpy.prod

    return reduce(lambda x, y: x * y, ns, 1)

def outer(A, B):
    """Outer product, differs from numpy version for special cases
    of scalar arguments:

        >>> outer([2, 3], [10, 100])
        array([[ 20, 200],
               [ 30, 300]])

        >>> outer([2, 3], 10)
        array([20, 30])

        >>> outer(10, [2, 3])
        array([20, 30])
    """
    return matmul(shape(A), shape(B), (), A, B)

def sums(m, n, k, A):
    """S(m, n) = SUM(k) A[m, k, n]

        >>> from numpy import ones, all
        >>> a = ones((2,3,4,5,6))

        >>> c = sums((2,3), (), (4,5,6), a)
        >>> shape(c)
        (2, 3)

        >>> c[0,0]
        120.0

        >>> c = sums((), (5,6), (2,3,4), a)
        >>> shape(c)
        (5, 6)

        >>> c[0,0]
        24.0

        >>> all( a == sums((2,3), (4,5,6), (), a) )
        True
    """

    A = asarray(A)

    assert shape(A) == m + k + n

    M = prod(m)
    N = prod(n)
    K = prod(k)

    A.shape = (M, K, N)

    C = sum(A, axis=1)

    A.shape = m + k + n
    C.shape = m + n

    return C

def dots(m, n, k, A, B):
    """C[m, n] = A[m, k] * B[m, k, n] (sum over repeated k, not over m)

        >>> from numpy import ones
        >>> a = ones((2,3,4,5,6))
        >>> b = ones((2,3,4,5,6,3,2))

        >>> c = dots((2,3,4), (3,2), (5,6), a, b)

        >>> shape(c)
        (2, 3, 4, 3, 2)

        >>> c[0,0,0,0,0]
        30.0

    Can this be done with vectorize() ?
    """

    A = asarray(A)
    B = asarray(B)

    assert shape(A) == m + k
    assert shape(B) == m + k + n

    M = prod(m)
    N = prod(n)
    K = prod(k)

    A.shape = (M, K)
    B.shape = (M, K, N)

    C = empty((M, N))

    # FIXME: how to avoid copying?
    for i in xrange(M):
        C[i, :] = dot(A[i], B[i])

    A.shape = m + k
    B.shape = m + k + n
    C.shape = m + n

    return C

def matmul(m, n, k, A, B, transA=False, transB=False):
    """Specialized dot(), inspired by DGEMM

        >>> from numpy import zeros
        >>> a = zeros((2, 3, 4, 5))
        >>> b = zeros((3, 4, 5, 9))
        >>> c = matmul( (2,), (9,), (3, 4, 5), a, b)
        >>> shape(c)
        (2, 9)

        >>> c = matmul( (), (3, 4, 5, 9), (), 5.0, b)
        >>> shape(c)
        (3, 4, 5, 9)
    """

#   print "m, n, k =", m, n, k

    # sizes:
    M = prod(m)
    K = prod(k)
    N = prod(n)
#   print "M, N, K =", M, N, K

#   print "matmul: A, B =", A, B, type(A), type(B)
    A = asarray(A)
    B = asarray(B)

    assert A.size == M * K
    assert B.size == K * N

    if transA:
        assert A.shape == k + m
        A.shape = (K, M)

        # transposed view of A:
        opA = transpose(A)

    else:
        assert A.shape == m + k
        A.shape = (M, K)

        # alias:
        opA = A

    if transB:
        assert B.shape == n + k
        B.shape = (N, K)

        # transposed view of B:
        opB = transpose(B)

    else:
        assert B.shape == k + n
        B.shape = (K, N)

        # alias:
        opB = B

    C = dot(opA, opB)

    # reshape back:
    A.shape = m + k
    B.shape = k + n
    C.shape = m + n
#   print "matmul: C =", C, type(C)

#   # FIXME: this is ugly, but to make doctests succed
#   # we treat the case of scalar C somewhat special:
#   if C.shape == ():
#       # 1) we got a numpy scalar, tough input A and B were
#       #    probably plain python scalars, return plain python:
#       # 2) dot(vecA, vecB) returns
#       C = C.item()

    # NO, this is indeed ugly and broken!

    return C

# python npx.py [-v]:
if __name__ == "__main__":
    import doctest
    doctest.testmod()

# Default options for vim:sw=4:expandtab:smarttab:autoindent:syntax
