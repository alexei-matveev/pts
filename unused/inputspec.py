class InputSpec:
    params = "params"
    geom = "geom0, geom1, ..."
    sections = [params, geom]

    class params:
        class processors(str):
             __doc__ = "Tuple (x,y,z) where..."

        class method:
            neb = "neb"
            l_bfgs_b = "l_bfgs_b"
            bfgs = "bfgs"

    class Geom:
        inputfile = "inputfile"
        values = [inputfile]

