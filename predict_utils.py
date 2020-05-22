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


# Load the model
def load_checkpoint(filepath, gpu, verbose):
    '''
    input: filepath to a checkpoint file -.pth
    output:
        model - pretrained network with frozen parameters and a trained classifier with loaded state dict
        criterion - the criterion to be used
        optimizer - the optimizer with loaded state dict

    returns a trained model ready for predictions with a criterion and optimizer in case more training is needed
    '''
    checkpoint = torch.load(filepath)

    if verbose:
        print("Loaded checkpoint {}.".format(str(filepath)))
    model_name = checkpoint["arch"]
    model = getattr(models, str(model_name))(pretrained = True)

    if verbose:
        print("Loaded model with {} architecture.".format(checkpoint["arch"]))

    model.to(device)
    for param in model.parameters():
        param.requires_grad = False

    if hasattr(model, 'classifier'):
        model.classifier = checkpoint["classifier"]
        model.classifier.load_state_dict(checkpoint["classifier_state_dict"])
    elif hasattr(model, 'fc'):
        model.fc = checkpoint["classifier"]
        model.fc.load_state_dict(checkpoint["classifier_state_dict"])

    else:
        raise Exception('The classifier of the loaded model could not be identified. .classifier & .fc was tried')

    if verbose:
        print("Loaded pretrained classifier with structure ".)

    optimizer = checkpoint["optimizer"]
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    model.class_to_name = checkpoint["class_to_name"]
    model.class_to_idx = checkpoint["class_to_idx"]
    criterion = nn.NLLLoss()


    return model, criterion, optimizer


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    transform = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])])

    return transform(image)


def predict(image_path, model, topk, gpu, verbose):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''

    # loads and preprocesses the image
    image = Image.open(image_path, "r")
    image = process_image(image)

    # get the input tensor in the correct dimension for the model
    image = image.view(1,3,224,224)

    # don't load the model again if it is already loaded to save time

    if gpu and torch.cuda.is_available():
        if verbose:
                print("GPU used for prediction")
        model.to("gpu")
        model.eval()
        ps = torch.exp(model(image.to("gpu")))

    else:
        if verbose:
                print("GPU not available or not chosen: CPU is used.")
        model.to("cpu")
        model.eval()
        ps = torch.exp(model(image))
            

    #get the top 5 predictions
    top_p, top_class = ps.topk(topk, dim = 1)
    # convert tensor to a list
    top_p = top_p.detach().numpy()
    top_p=list(top_p.flatten())
    classes = list(top_class.detach().numpy().flatten())

    # map the predictions to the classes
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    classes = list(top_class.detach().numpy().flatten())
    class_preds = []
    for item in classes:
        item = idx_to_class[item]
        class_preds.append(model.class_to_name[item])

    if verbose:
        print("The predictions are:")
    for i in range(top_p):
        print("{}. {} with probability {}".format(i+1, class_preds[i], top_p[i]*100))
    return None
