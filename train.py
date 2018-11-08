import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, models, transforms
from collections import OrderedDict
import numpy as np
from PIL import Image
import seaborn as sns
import argparse
from torch.utils.data import DataLoader
from utils import validation

# Define command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, help='Path to dataset ')
parser.add_argument('--arch', type=str, default='vgg16',help='Choose architecture between VGG and Alexnet')
parser.add_argument('--gpu', action='store_true', default=True,help='Use GPU if available')
parser.add_argument('--epochs', type=int, default=5,help='Number of epochs')
parser.add_argument('--learning_rate', type=float, default=0.001,help='Learning rate')
parser.add_argument('--hidden_units', type=int, default=[4096,1024],help='Number of hidden units')
parser.add_argument('--checkpoint', type=str, default='',help='Save trained model checkpoint to file')

args = parser.parse_args()

device = 'cuda' if args.gpu == True else 'cpu'

train_dir = args.data_dir + '/train'
valid_dir = args.data_dir + '/valid'
test_dir = args.data_dir + '/test'

# : Define your transforms for the training, validation, and testing sets
data_transforms = {'train' : transforms.Compose([transforms.RandomRotation(30),
                                                 transforms.RandomResizedCrop(224),                                                 
                                                 transforms.RandomHorizontalFlip(),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize([0.485, 0.456, 0.406],
                                                                      [0.229, 0.224, 0.225]),
                                                 ]),
                   
                   'valid' : transforms.Compose([transforms.Resize(256),
                                                 transforms.CenterCrop(224),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize([0.485, 0.456, 0.406],
                                                                      [0.229, 0.224, 0.225])
                                                ]),
                   
                   'test' : transforms.Compose([transforms.Resize(256),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406],
                                                                      [0.229, 0.224, 0.225]),
                                               ])
                  }
                                                
# : Load the datasets with ImageFolder
image_datasets = {'train' : datasets.ImageFolder(train_dir, transform=data_transforms['train']),
                  'valid' : datasets.ImageFolder(valid_dir, transform=data_transforms['valid']),
                  'test' : datasets.ImageFolder(test_dir, transform=data_transforms['test'])
                 }

# : Using the image datasets and the trainforms, define the dataloaders
dataloaders = {'train' : DataLoader(image_datasets['train'], batch_size=64, shuffle=True),
               'valid' : DataLoader(image_datasets['valid'], batch_size=32),
               'test' : DataLoader(image_datasets['test'], batch_size=32)
              }

input_size = 25088 if args.arch == 'vgg16' else 4096
hidden_units = [4096, 1024]
output_size = 102

if args.arch == 'vgg16':
    model = models.vgg16(pretrained=True)
elif args.arch == 'alexnet':
     model = models.alexnet(pretrained=True)
else:
     raise valueError('{0} architecture is not supported '.format(arch))

for param in model.parameters():
        param.requires_grad = False        
        
classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(input_size, hidden_units[0])),
                          ('relu1', nn.ReLU()),
                          ('drop1', nn.Dropout(0.5)),
                          ('fc2',  nn.Linear(hidden_units[0], hidden_units[1])),
                          ('relu2', nn.ReLU()),
                          ('drop2',nn.Dropout(0.5)),
                          ('fc3',  nn.Linear(hidden_units[1], output_size)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))

model.classifier = classifier        

if args.gpu:
    model.cuda()
else:
    model.cpu()

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr = args.learning_rate)

model.train()

print_every = 40
steps = 0

for epoch in range(args.epochs):
    accuracy = 0
    running_loss = 0.0
    for inputs, labels in dataloaders['train']:

        steps += 1
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        output = model(inputs)
        _, preds = torch.max(output.data, 1)
        loss = criterion(output, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
        if steps % print_every == 0:
            test_loss, test_accuracy = validation(model, dataloaders['valid'], criterion, device)
            print("Epoch: {}/{}".format(epoch+1, args.epochs),
                  "Train Loss: {:.4f}".format(running_loss/print_every),
                  "Train Accuracy : {:.4f}".format(accuracy/print_every),
                  "Validation Loss : {:.4f}".format(test_loss),
                  "Validation Accuracy : {:.4f}".format(test_accuracy))
            model.train()
            accuracy = 0
            running_loss = 0

# Do validation on the test set, print results
test_loss, test_accuracy = validation(model, dataloaders['test'], criterion, device)
print("Test Loss : {:.4f}".format(test_loss),
    "Test Accuracy : {:.4f}".format(test_accuracy))

# Save the checkpoint
model.class_to_idx = image_datasets['train'].class_to_idx
checkpoint = {'arch' : args.arch,
              'classifier' : model.classifier,
              'state_dict' : model.state_dict(),
              'optimizer' : optimizer,
              'optimizer_dict' : optimizer.state_dict(),
              'epochs' : args.epochs,
              'class_to_idx' : model.class_to_idx}

torch.save(checkpoint, 'checkpoint.pth')