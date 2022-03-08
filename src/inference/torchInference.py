#!/usr/bin/env python
import os
import torch
import cv2
import time
import argparse
from collections import OrderedDict
from pathlib import Path

import matplotlib.pyplot as plt
from training.model import createDeepLabv3

parser = argparse.ArgumentParser()
parser.add_argument(
    "model_path", help='Specify the dataset directory path')
parser.add_argument(
    "image_path", help='Specify the experiment directory where metrics and model weights shall be stored.')

args = parser.parse_args()

model_path = args.model_path
image_path = args.image_path

# Load the trained model, which is either the entire
# model in pickle format, or just the state dict:

if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

model_info = torch.load(model_path,map_location=device)
if type(model_info) == OrderedDict:
    model = createDeepLabv3(outputchannels=3)
    model.load_state_dict(model_info)
else: 
    model = model_info

model.to(device)


# if torch.cuda.is_available():
#     model = torch.load(model_path)
# else:
#     model = torch.load(model_path, map_location=torch.device('cpu'))
#
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
# Set the model to evaluate mode
model.eval()

# Read  a sample image and mask from the data-set
originalImage = cv2.imread(image_path)

# Resize image
img = cv2.resize(originalImage, (256, 256), cv2.INTER_AREA).transpose(2,0,1)

# Uncomment above line and use the below one for inference with original image size
# img = originalImage.transpose(2,0,1)

img = img.reshape(1, 3, img.shape[1],img.shape[2])

start_time = time.time()
# Get a dict:
#    'out'  : mask-colored image [1,3,256,256]
#    'aux'  : logits [1, 21, 256, 256]
with torch.no_grad():
    if torch.cuda.is_available():
        a = model(torch.from_numpy(img).to(device).type(torch.cuda.FloatTensor)/255)
    else:
        a = model(torch.from_numpy(img).to(device).type(torch.FloatTensor)/255)
print("--- %s seconds ---" % (time.time() - start_time))

# Get the image as [3, 256, 256]:
outImage = a['out'].cpu().detach().squeeze().numpy()
# Transposet to [256, 256, 3]:
ax = plt.imshow(outImage.transpose([1,2,0]))

# Create an output file name for the colored image:
fstem   = Path(image_path).stem
fsuffix = Path(image_path).suffix
out_path = os.path.join('/tmp/', 
                        f"{fstem}_masked_img{fsuffix}")
print(f"Saving masked image to {out_path}")
ax.figure.savefig(out_path)
ax.figure.show()
key = input("Press any key to quit and close the image: ")

    

