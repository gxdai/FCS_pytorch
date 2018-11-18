import time
import sys
import torch.optim as optim
import torch
from PIL import Image

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


def configure_optimizer(param_lr_list, optimizer):
    """This is configure different optimizer.
    """
    gpu_number = torch.cuda.device_count()

    if optimizer == 'rmsprop':
        optimizer = optim.RMSprop(param_lr_list, lr=0.001)
    elif optimizer == 'sgd':
        optimizer = optim.SGD(param_lr_list, lr=0.001, momentum=0.9)

    return optimizer

def create_thumbnail_image(infile, outfile, size=(128, 128)):
    """Given an input image, return a thumbnail image"""
    try:
        im = Image.open(infile)
        im.thumbnail(size)
        im.save(outfile, 'png')
    except IOError:
        print("cannot create thumbnail for {}".format(infile))
