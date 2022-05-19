import os
from glob import glob
import re 

import numpy as np
import pandas as pd
import scipy.io

from PIL import Image
from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import classification_report
from sklearn import preprocessing
import datetime

import dataset
import multiscale4_0 as multiscale

#import cam_for_multi
#import cam_multi_convnet as cam_for_multi #convnetのpath
#import cam_multi_conv1 as cam_for_multi #conv1のpath
import cam_multi_conv2 as cam_for_multi #conv2のpath

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

model_path = './model/multiscale4_0.pth'

drmodel = multiscale.Multi()

drmodel.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model = drmodel.to(device)

for i in range(1, 1000):
    if i < 10:
        path = './data/cars/sample/0000' + str(i) + '.jpg'
        cam_for_multi.cam_for_multi(model, path)
    elif 10 <= i < 100:
        path = './data/cars/sample/000' + str(i) + '.jpg'
        cam_for_multi.cam_for_multi(model, path)
    elif 100 <= i < 1000:
        path = './data/cars/sample/00' + str(i) + '.jpg'
        cam_for_multi.cam_for_multi(model, path)


