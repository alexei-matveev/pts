#!/usr/bin/python
"""
"""

__all__ = ["matmul"]

from numpy import asarray, shape, dot

def matmul(m, n, k, A, B):
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

    # product() appears to return floats?
    mul = lambda x, y: x * y

    # sizes:
    M = reduce(mul, m, 1)
    K = reduce(mul, k, 1)
    N = reduce(mul, n, 1)
#   print "M, N, K =", M, N, K

#   print "matmul: A, B =", A, B, typeA, typeB
    A = asarray(A)
    B = asarray(B)

    assert A.size == M * K
    assert B.size == K * N

    # somewhat redundant:
    assert A.shape == m + k
    assert B.shape == k + n

    # reshape temporarily (does it have side-effects?):
    A.shape = (M, K)
    B.shape = (K, N)

    C = dot(A, B)
    
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
