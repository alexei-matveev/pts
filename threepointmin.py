#!/usr/bin/env python
"""
Test on Mueller Brown Potential:

    >>> from pts.pes.mueller_brown import MuellerBrown
    >>> mb = MuellerBrown()

    Function which just returns the needed values for the
    Approximation Function:

    >>> def mub(x1, x2, x3):
    ...      return x1, x2, x3, mb.fprime(x1).flatten(), mb.fprime(x2).flatten(), mb.fprime(x3).flatten(), mb.f(x1), mb.f(x2), mb.f(x3)

    >>> a = mub( [-0.55822362,  1.44172583], [-0.82200123,  0.62430438],[-0.05001084,  0.46669421])
    >>> print a
    ([-0.55822362000000003, 1.44172583], [-0.82200123000000003, 0.62430437999999999], [-0.050010840000000001, 0.46669421], array([  5.43831125e-05,  -5.31756199e-05]), array([-0.00523521,  0.00046528]), array([  1.18691394e-05,   1.50976424e-04]), -146.69951720995329, -40.66484351147777, -80.767818129651189)

    >>> ts_3p_gr(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8])
    (array([-0.4388412 ,  1.20162258]), array([ -3.46944695e-18,   3.25260652e-19]), -130.03944969666441, -3.4643888169298034e-18, -7.5592804931911682e-19, True)

    >>> ts_3p_gr(a[0], a[1], a[2], a[3], a[4], a[5])
    (array([-0.4388412 ,  1.20162258]), array([ -3.46944695e-18,   3.25260652e-19]), -3.4643888169298034e-18, -7.5592804931911682e-19, True)

    >>> mb.fprime(np.array([-0.19577454,  0.03811052])), mb.f(np.array([-0.19577454,  0.03811052]))
    (array([-101.73697899,  -93.82146582]), -30.234251920790779)

    >>> from pts.pes.mueller_brown import show_chain, CHAIN_OF_STATES
    >>> from path import Path
    >>> points = np.array(CHAIN_OF_STATES)
    >>> pth = Path(points, np.linspace(-1., 1., len(points)))
    >>> pths = np.array([ pth(t) for t in np.linspace(-1., 1., 10)])
    >>> x1, x2, x3 = pths[1:4]
    >>> p1, p2, p3, g1, g2, g3, e1, e2, e3 = mub(x1, x2, x3)

    >>> xts, gts, ets, gpr1, gpr2, work =  ts_3p_gr(p1, p2, p3, g1, g2, g3, e1, e2, e3)
    >>> print xts, gts, ets, gpr1, gpr2, work
    [-0.76505965  0.59237172] [  6.88338275e-15  -5.32907052e-15] -42.800650422 8.40802710791e-15 -6.57556851718e-15 True
    >>> print xts, mb.fprime(xts), mb.f(xts) 
    [-0.76505965  0.59237172] [-30.86794654  26.75993863] -42.044187973

    >>> x1, x2, x3 = pths[6:9]
    >>> p1, p2, p3, g1, g2, g3, e1, e2, e3 = mub(x1, x2, x3)

    >>> xts2, gts2, ets2, gpr12, gpr22, work2 =  ts_3p_gr(p1, p2, p3, g1, g2, g3, e1, e2, e3)
    >>> print xts2, gts2, ets2, gpr12, gpr22, work2
    [ 0.20243333  0.29680714] [ -8.88178420e-16   1.06581410e-14] -70.6644241613 -8.10519034122e-15 9.53587417705e-15 True
    >>> print xts2, mb.fprime(xts2), mb.f(xts2)
    [ 0.20243333  0.29680714] [ 0.04725785 -6.93897701] -72.2625180165

"""

import numpy as np



def ts_3p_gr(geo1, geo2, geo3, grad1, grad2, grad3, ener1 = None, ener2 = None, ener3 = None):
    """
    This function should calculate an approximated transition state (gradient should vanish)
    with just three points, it will minimize the
          vec(g)*vec(s_{+-})
    with the approximatation:
          g = grad2 + (grad3 - grad2)/|geo3 - geo2|  + (grad1- grad2)/|geo2 - geo1|
    if the three points are in a line, there will be used a linear approximation instead
    the geo (geoemtries) and grad (gradients) should be given as a vector, meaning flattened if
    they have been arrays
    If the energy is also given, the approximation for the energy at the point where the ts may be
    is also calculated
    Output:
    geometry_ts-approx, approximated_gradient_ts-approx, (approximated_energy_ts-approx), approximated_gradient_ts-approx projected
     in direction 2->3, approximated_gradient_ts-approx projected in direction 2->1, if the tool thinks the result reasonable
    """
    geo1 = np.asarray(geo1)
    geo2 = np.asarray(geo2)
    geo3 = np.asarray(geo3)
    grad1 = np.asarray(grad1)
    grad2 = np.asarray(grad2)
    grad3 = np.asarray(grad3)


    # the directions are:
    s23 = geo3 - geo2
    s12 = geo1 - geo2
    ls23 = np.sqrt(np.dot(s23, s23))
    ls12 = np.sqrt(np.dot(s12, s12))
    s23 /= ls23
    s12 /= ls12
    #s23 = np.array([1,0])
    #s12 = np.array([0,1])
    # the cos(angle):
    #cosphi = np.dot(s23, s12)
    #project the gradient on the directions:
    gin23 = np.dot(s23, grad2)
    gin12 = np.dot(s12, grad2)
    # approximated hessian:
    h33 = np.dot((grad3 - grad2), s23) / ls23
    h11 = np.dot((grad1 - grad2), s12) / ls12
    h31 = np.dot((grad1 - grad2), s23) / ls12
    h13 = np.dot((grad3 - grad2), s12) / ls23

    determ = h11 * h33 - h13 * h31

    if  determ * determ < 10**-18  :
        print "Linear approximation: all three points are on a line"
        # solve y23 * h33 - h11 * y12 = -gin23
        y23 = - gin23 / ( h33 + h11)
        y12 = -y23
    else:
        # solve H y = g_{in}
        y23 = - (gin23 * h11 - gin12 * h31) / determ
        y12 = - (gin12 * h33 - gin23 * h13) / determ

    geots = geo2 + y23 * s23 + y12 * s12
    gradts = grad2 + y23 * (grad3 - grad2) / ls23  + y12 * (grad1 - grad2) / ls12
    gradtsinplane23 = np.dot(gradts, s23)
    gradtsinplane12 = np.dot(gradts, s12)
    # if the energies are also there, approximate energy for
    # comparision:
    if not ener2 == None:
         ets = ener2 + (ener3 - ener2) * y23 / ls23 + (ener1 - ener2) * y12 / ls12

    goodrange = True
    if y23**2 > ls23**2  or y12**2 > ls12**2:
         goodrange = False

    if ener2 == None:
        return (geots, gradts, gradtsinplane23, gradtsinplane12, goodrange)
    else:
        return (geots, gradts, ets, gradtsinplane23, gradtsinplane12, goodrange)

if __name__ == "__main__":
    import doctest
    doctest.testmod()

