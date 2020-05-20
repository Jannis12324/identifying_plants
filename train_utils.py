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


def train_transforms(train_dir):
    '''
    Takes the directory of the training data as an input and returns a trainloader with images transformed for training.
    '''
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])])

    train_data = datasets.ImageFolder(train_dir, transform = train_transforms)
    trainloader = torch.utils.data.DataLoader(train_data, batch_size = 32, shuffle = True)

    return trainloader

def test_transforms(test_dir):
    '''
    Takes the directory of the test data and returns a dataloader with images transformed for testing.
    '''
    test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])])
    test_data = datasets.ImageFolder(test_dir, transform = test_transforms)
    testloader = torch.utils.data.DataLoader(test_data, batch_size = 32)

    return testloader

def get_cat_to_name():
    '''
    When called returns the dict which translates the category numbers to flower names.
    '''
    import json

    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name

def load_pretrained_nn(model_name):
    '''
    Takes the model architecture as a string input and returns the loaded and pretrained model with all parameters frozen.
    '''
    # get the requested model
    model = getattr(models, str(model_name))(pretrained = True)
    # freeze parameters of pretrained model
    for param in model.parameters():
        param.requires_grad = False
    return model

def build_new_classifier(hidden_units):
    '''
    Builds a new classifier with the requested amount of hidden units.
    '''
    classifier = nn.Sequential(
                           nn.Linear(25088, hidden_units),
                           nn.ReLU(inplace = True),
                           nn.Dropout(p=0.2),
                           nn.Linear(hidden_units, 102),
                           nn.LogSoftmax(dim = 1))
    return classifier

def train_model(model, optimizer, trainloader, validloader, epoch, gpu, verbose):
    '''
    Trains the classifier attached to the pretrained model.
    Inputs:
    model - pretrained neural network
    optimizer - optimizer objekt
    trainloader - object to supply the training images
    validloader - object to supply the validation images
    epoch - int - the number of epochs to be run
    gpu - Bool - If the gpu should be used if possible

    '''
    criterion = nn.NLLLoss()


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if gpu:
        model.to(device)
        if device == "gpu":
            print("GPU training activated")

    model.train()
    epochs = epoch
    steps = 0
    print_every = 5
    running_loss = 0


    for epoch in range(epochs):
        print("Training in epoch {}".format(str(epoch)))
        for images, labels in trainloader:
            steps +=1
            if gpu:
                images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            logps = model(images)
            loss = criterion(logps, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                if verbose:
                    print("Current batch: {}".format(str(steps)))
                valid_loss = 0
                accuracy = 0
                model.eval()

                with torch.no_grad():
                    for images, labels in validloader:
                        if gpu:
                            images, labels = images.to(device), labels.to(device)
                        logps = model(images)

                        valid_loss += criterion(logps, labels).item()

                        #accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim = 1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor))

                    print("Validation loss: {}".format(valid_loss/len(validloader)))
                    print("Validation Accuracy: {}".format(accuracy/len(validloader)))
                    model.train()
            print("Train loss: {}".format(running_loss/len(trainloader)))
    return model, optimizer

def save_model(arch, model, optimizer, cat_to_name, class_to_idx, save_dir):
    checkpoint = {"transfer_model": arch,
             "classifier": model.classifier,
             "classifier_state_dict": model.classifier.state_dict(),
             "optimizer": optimizer,
             "optimizer_state_dict": optimizer.state_dict(),
             "class_to_idx": class_to_idx,
             "class_to_name": model.class_to_name}
    torch.save(checkpoint, save_dir)
