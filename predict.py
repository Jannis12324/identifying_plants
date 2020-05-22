# Imports
from argparse import ArgumentParser
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from PIL import Image
from numpy import asarray
import numpy as np
from predict_utils import *



# Create a parser to handle the passed arguments
parser = ArgumentParser(description = "Return the prediction and probabability of a given image.")
# add the arguments handled by the parser
parser.add_argument("image_path", metavar = "", default = "flowers/test/28/image_05214.jpg", help = "The path to a single image.")
parser.add_argument("checkpoint", metavar = "", default = "checkpoint_vgg13.pth", help = "The checkpoint file which should be used to build the model."
parser.add_argument("-v", "--verbose", action = "store_true", help = "Prints status messages.")
parser.add_argument("-tk", "--top_k",metavar = "", type = int, nargs = 1, default = 3, help = "The number of classes with the highest probability")
parser.add_argument("-gpu", "--gpu", action = "store_true", help = "Uses the GPU for inference if available")
parser.add_argument("-cat", "--category_names", default = 'cat_to_name.json', help = "The json file with the mapping of categories to names")


# save the passed arguments in a variable
args = parser.parse_args()

# Load the model
model, criterion, optimizer = load_checkpoint(args.checkpoint, args.gpu, args.verbose)

# predict
predict(args.image_path, model, args.top_k, args.gpu, args.verbose)
