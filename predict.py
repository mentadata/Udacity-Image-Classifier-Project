import argparse
import json
import torch
from PIL import Image
from utils import process_image, load_checkpoint
from collections import OrderedDict
import numpy as np


parser = argparse.ArgumentParser(description='Predict flower type')
parser.add_argument('--checkpoint', type=str, help='Path to checkpoint' , default='checkpoint.pth')
parser.add_argument('--image_path', type=str, help='Path to file' , default='flowers/test/28/image_05230.jpg')
parser.add_argument('--gpu', type=bool, default=True, help='Use gpu or cpu?')
parser.add_argument('--topk', type=int, help='Number of k to predict' , default=0)
parser.add_argument('--cat_to_name_json', type=str, help='Json file to load for class labels to flower name mapping' , default='cat_to_name.json')
args = parser.parse_args()

image_path = args.image_path
with open(args.cat_to_name_json, 'r') as f:
    cat_to_name = json.load(f)

# : Load Checkpoint
model, checkpoint = load_checkpoint(args.checkpoint)

def predict(image_path, model, topk=5,device='cuda'):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    
    
    # Process image
    img = process_image(image_path)
    
    # Need to convert to tensor to push through the model
    img_tensor = torch.from_numpy(img).type(torch.FloatTensor)
    # This is needed to mimic batch size as required by VGG
    img_model = img_tensor.unsqueeze(0)
    img_model = img_model.to(device).float()
    model.to(device)
    # Model return LogSoftmax, so converting back to probabilities
    probs = torch.exp(model.forward(img_model))
    
    # Get Top K probabilities and labels
    results = probs.topk(topk)
    probs = results[0].to('cpu')
    labels = results[1].to('cpu')
    topk_probs = probs.detach().numpy().tolist()[0] 
    topk_labels = labels.detach().numpy().tolist()[0]
    
    # Convert indices to classes
    idx_to_class = {val: key for key, val in    
                                      model.class_to_idx.items()}
    top_labels = [idx_to_class[lab] for lab in topk_labels]
    top_flowers = [cat_to_name[idx_to_class[lab]] for lab in topk_labels]
    return topk_probs, top_flowers

# : Predict type and print
if args.topk:
    probs, flower_names = predict(image_path, model, args.topk,'cuda' if args.gpu else 'cpu')
    print('Probabilities of top {} flowers:'.format(args.topk))
    for i in range(args.topk):
        print('{} : {:.2f}'.format(flower_names[i],probs[i]))
else:
    probs, flower_names = predict(image_path, model)
    print('Flower is predicted to be {} with {:.2f} probability'.format(flower_names[0], probs[0]))