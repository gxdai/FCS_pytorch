import time
import sys

def progress_bar(total_width):

    # setup toolbar
    sys.stdout.write("[%s]" % (" " * toolbar_width))
    sys.stdout.flush()
    sys.stdout.write("\b" * (toolbar_width+1)) # return to start of line, after '['

    for i in xrange(toolbar_width):
        # update the bar
        sys.stdout.write("-")
        sys.stdout.flush()

    sys.stdout.write("\n")

def get_parameter_group(model):
    """Separate parameters into different groups."""
    all_params = list(model.parameters())
    print("all parameter #: {}".format(len(all_params)))
    fc_params = list(model.fc.parameters())
    print("fc parameter #: {}".format(len(fc_params)))
    base_params = [param for param in all_params if param not in fc_params]
    print("base parameter #: {}".format(len(base_params)))
    return fc_params, base_params
