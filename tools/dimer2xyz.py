#!/usr/bin/env python
def read_from_pickle_log(filename ):
    """
    read the geometry of given points as well as modes, curvaturese
    from log.pickle file.
    always extract center curvature and lowest mode, addtional
    points can be specified
    """
    from pickle import load
    from numpy.linalg import norm

    geos = []
    modes = []
    curvatures = []
    energy = []
    gradients = []

    logfile = open(filename, "r")

    # first item is the the object function,
    # contains symbols and the transformation in Cartesian geometries.
    object = load(logfile)

    # read in data until reach the end of the file
    while True:
        try:
            items = load(logfile)
        except EOFError:
            break

        for item in items:
            value, __, name = item
            if name == "Mode":
                # normalize when read in
                #FIXME: really use norm here? Not better from the
                # metric module
                modes.append(value / norm(value))
            if name == "Curvature":
                curvatures.append(value)
            elif name == "Center":
                geos.append(value)
            elif name == "Energy":
                energy.append(value[0])
            elif name == "Gradients":
                gradients.append(value)

    logfile.close()

    return object, geos, modes, curvatures, energy, gradients

def main(argv):
     """
     Takes the geometries from the pickle file and prints them in
     xyz format to standard output
     """
     from pts.ui.cmdline import get_options_to_xyz
     from pts.ui.write_COS import print_xyz_with_direction
     from sys import stdout
     from numpy import array

     opts, geo_files = get_options_to_xyz("progress", argv, None)

     assert len(geo_files) == 1

     geo_file = geo_files[0]

     assert not opts.beads
     assert opts.num == None

     obj, geos, modes, __, energies, __ = read_from_pickle_log(geo_file)

     if opts.add_modes:
        for geo, mode,  energy in zip(geos, modes,  energies):
            text = "E = %f" % energy
            print_xyz_with_direction(stdout.write, array(geo), obj, text = text, direction = array(mode))
     else:
        for geo, energy in zip(geos, energies):
            text = "E = %f" % energy
            print_xyz_with_direction(stdout.write, array(geo), obj, text = text)

if __name__ == "__main__":
    import doctest
    doctest.testmod()
    main(sargv[1:])
