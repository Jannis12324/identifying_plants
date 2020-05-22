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
from train_utils import *

# Create a parser to handle the passed arguments
parser = ArgumentParser(description = "Train and save a neural network")
# add the arguments handled by the parser
parser.add_argument("data_directory", metavar = "", help = "the directory of the data to be trained and tested on")
parser.add_argument("-sd", "--save_dir", nargs = "?", default = "checkpoint.pth", metavar = "", help = "the directory and checkpoint name where the model is saved")
parser.add_argument("--arch", nargs = "?", default = "vgg13",type = str, metavar = "", help = "the pretrained model that should be used for training")
parser.add_argument("-lr", "--learning_rate",nargs = "?", default = 0.003, type = float, metavar = "", help = "The learning rate as a float")

# function as a new type to handle the boundaries of the hidden units
def hidden_units_type(x):
    '''
    Defines the boundaries of the acceptable hidden units.
    '''
    x = int(x)
    if x < 102 or x > 25088:
        raise argparse.ArgumentTypeError("Value needs to be between 102 and 25088 units")
    return x
parser.add_argument("-hu", "--hidden_units", type = hidden_units_type,nargs = "?", default = 512, metavar = "", help = "The number of nodes in the hidden layer")

parser.add_argument("-e", "--epochs", type = int, nargs = "?", default = 1, metavar = "", help = "The number of epochs used for training (Should be a small number as 1 epoch takes about 5-8 min")
parser.add_argument("-gpu", "--gpu", action = "store_true", help = "Uses the gpu for training if available")
parser.add_argument("-v", "--verbose", action = "store_true", help = "Prints status messages")

# save the passed arguments in a variable
args = parser.parse_args()
v = args.verbose
verbose = v
# Load the data
if v:
    print("Loading data")
data_dir = args.data_directory
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

trainloader = train_transforms(train_dir)
validloader = test_transforms(valid_dir)
testloader = test_transforms(test_dir)

# Load mapping dict:
if v:
    print("Loading mapping dict")
cat_to_name = get_cat_to_name()

# load pretrained model
if v:
    print("Loading pretrained model {}".format(str(args.arch)))
model = load_pretrained_nn(args.arch)

# build and attach new classifier
if v:
    print("Building new classifier with {} hidden units".format(str(args.hidden_units)))
model.classifier = build_new_classifier(args.hidden_units)

# train the model
optimizer = optim.Adam(model.classifier.parameters(), float(args.learning_rate))
model, optimizer = train_model(model,optimizer, trainloader, validloader, args.epochs, args.gpu, verbose)

if v:
    print("Saving checkpoint with name {}".format(str(args.save_dir))
# save the model
save_model(args.arch, model, optimizer, cat_to_name, train_data.class_to_idx, args.save_dir)

print("Model saved at {}".format(str(args.save_dir))
