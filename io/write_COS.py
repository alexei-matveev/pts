#!/usr/bin/env python
def print_xyz_with_direction(write_method, coord, cs, text = "", direction = None):
    """
    Prints the geometry coord (after transformed in Cartesian)
    and the direction into the write_method.
    INPUT:
    write_method: should be a write method, like fopen(filename, "w") or
                  std.write.
    coord:        geometries to store (should be in internal coordinates).
    cs:           like others: Symbols of atoms, trafo from internal to Cartesians
    text:         if some special text is been wanted for the otherwise not used line.
    direction:    internal direction vector (maybe mode, forces)
    """
    from numpy import dot

    symbs, trafo = cs
    numats = len(symbs)
    write_method("%i\n" % (numats))
    write_method("%s\n" % (text))
    carts = trafo(coord)

    if direction == None:
        for s, pos in zip(symbs, carts):
             write_method("%-2s %22.15f %22.15f %22.15f\n" % (s, pos[0], pos[1], pos[2]))

    else:
        transfer = trafo.fprime(coord)
        dirs = dot( transfer, direction)
        for s, pos, dir in zip(symbs, carts, dirs):
             write_method("%-2s %22.15f %22.15f %22.15f    %12.8f %12.8f %12.8f\n" \
              % (s, pos[0], pos[1], pos[2], dir[0], dir[1], dir[2]))


