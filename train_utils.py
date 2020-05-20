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
    model = getattr(models, model_name)(pretrained = True)
    for param in model.parameters():
        param.requires_grad = False
    return model

def build_new_classifier(hidden_units):
    classifier = nn.Sequential(
                           nn.Linear(25088, hidden_units),
                           nn.ReLU(inplace = True),
                           nn.Dropout(p=0.2),
                           nn.Linear(hidden_units, 102),
                           nn.LogSoftmax(dim = 1))
    model.classifier = classifier
