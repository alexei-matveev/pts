
def log2grads(filename):
    """
    Extracts data from a table in the log file that looks like this:

     Variable       Old X    -DE/DX   Delta X   Delta X   Delta X     New X
                                     (Linear)    (Quad)   (Total)
        ch        2.05980   0.01260   0.00000   0.01134   0.01134   2.07114
       hch        1.75406   0.09547   0.00000   0.24861   0.24861   2.00267
       hchh       2.09614   0.01261   0.00000   0.16875   0.16875   2.26489
             Item               Value     Threshold  Converged?

    """
    import re

    f = open(filename, "r")
    log_file_as_str = f.read()
    grads_pattern = re.compile(r"""Variable\s+Old X\s+-DE/DX   Delta X   Delta X   Delta X     New X(.+?)\(Total\)(.+?)Item""", re.S)

    match = grads_pattern.search(log_file_as_str)
    f.close()
    if match == None:
        raise Exception("Gradient information not found in " + filename)

    table = match.group(2)

    der_str_list = re.findall(r"\s+\S+\s+\S+\s+(\S+)\s+\S+\s+\S+\s+\S+\s+\S+", table)
    dE_on_dX_vec = [float(s) for s in der_str_list]
    return dE_on_dX_vec





