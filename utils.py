import numpy as np
from torchvision import datasets, models
import torchvision
import torch
from torchvision import transforms
from PIL import Image

def validation(model, loader, criterion,device):
    
    model.to(device)
    
    loss = 0
    accuracy = 0
    
    model.eval()
    
    for images, labels in loader:

        images, labels = images.to(device) , labels.to(device)
        
        output = model.forward(images)
        loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return loss/len(loader), accuracy/len(loader)

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    
    if checkpoint['arch'] == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif checkpoint['arch'] == 'alexnet':
        model = models.alexnet(pretrained=True)
    
    for param in model.parameters():
        param.requires_grad = False

    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
       
    return model, checkpoint

def process_image(imagepath):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    img_transforms = transforms.Compose([transforms.Resize(256), 
        transforms.CenterCrop(224), 
        transforms.ToTensor()])
    
    input_img = Image.open(imagepath)
    pil_img = img_transforms(input_img).float()
    
    np_img = pil_img.numpy()
    
    mean = np.array([0.485, 0.456, 0.406]) #provided mean
    std = np.array([0.229, 0.224, 0.225]) #provided std
    img = (np.transpose(np_img, (1, 2, 0)) - mean)/std
    
    # Move color channels to first dimension as expected by PyTorch
    img = img.transpose((2, 0, 1))
    
    return img

