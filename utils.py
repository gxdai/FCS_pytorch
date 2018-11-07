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
