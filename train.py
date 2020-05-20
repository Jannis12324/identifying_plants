
# Imports
from argparse import ArgumentParser
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image
from numpy import asarray
import numpy as np
import train_utils

# Create a parser to handle the passed arguments
parser = ArgumentParser(description = "Train and save a neural network")
# add the arguments handled by the parser
parser.add_argument("data_directory", metavar = "", help = "the directory of the data to be trained and tested on")
parser.add_argument("-sd", "--save_dir", metavar = "", help = "the directory where the model is saved")
parser.add_argument("--arch", metavar = "", help = "the pretrained model that should be used for training")
parser.add_argument("-lr", "--learning_rate", type = float, metavar = "", help = "The learning rate as a float")

# function as a new type to handle the boundaries of the hidden units
def hidden_units_type(x):
    '''
    Defines the boundaries of the acceptable hidden units.
    '''
    x = int(x)
    if x < 102 or x > 25088:
        raise argparse.ArgumentTypeError("Value needs to be between 102 and 25088 units")
    return x
parser.add_argument("-hu", "--hidden_units", type = hidden_units_type, metavar = "", help = "The number of nodes in the hidden layer")

parser.add_argument("-e", "--epochs", type = int, metavar = "", help = "The number of epochs used for training (Should be a small number as 1 epoch takes about 5-8 min")
parser.add_argument("-gpu", "--gpu", action = "store_true", help = "Uses the gpu for training if available")
# save the passed arguments in a variable
args = parser.parse_args()


if __name__ == "__main__":
    print("The passed Argumens are '{}'".format(args.data_directory))
