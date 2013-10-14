#!/usr/bin/env python

def main(args):
    from sys import maxint, stderr, exit
    from pts.tools.path2plot import visualize_path
    from pts.tools.dimer2plot import visualize_dimer
    from pts.ui.cmdline import visualize_input

    args_red = []
    names_fun = []
    break_at = []

    i = 0
    while i < len(args):
       if args[i] in ["--path", "--progress"]:
           # We need to know the next file AND leave it in the arguments.
           # But remove the current argument as optpare in visualize_input
           # cannot handle it.
           names_fun.append(args[i][2:])
           break_at.append(args[i+1])
       else:
           args_red.append(args[i])
       i = i + 1

    # Extract the parameter from the command line.
    files, data_ase, other_input, values, path_look, dimer_special, for_plot = visualize_input("mixed", "plot", args_red, 100)

    num_break = []

    for break_point in break_at:
        # We need the number in the left over arguments of the break points
        # not the name of them.
        num_break.append(files.index(break_point) -1)

    # Do we start with path or progress?
    assert (-1 in num_break), "ERROR: needs to know what format first FILE is!"

    # The last ones belong all to the last function.
    num_break.append(maxint)

    for i, name in enumerate(names_fun):
        # The last should plot, the other not yet.
        if i == len(names_fun) -1:
            hold = False
        else:
            hold = True

        # These are the files belonging to the current call.
        # The tool will get all to handle naming and colors but process only those
        # within that range.
        plot_range = [num_break[i] + 1, num_break[i+1]]

        if name == "progress":
            visualize_dimer(files, values, dimer_special, for_plot, hold, plot_range)
        elif name == "path":
            visualize_path(files, data_ase, other_input, values, path_look, for_plot, hold, plot_range )
        else:
            print >> stderr, "ERROR: Name", name, "is not know. Aborting!"
            exit()

if __name__ == "__main__":
    from sys import argv
    main(argv[1:])
