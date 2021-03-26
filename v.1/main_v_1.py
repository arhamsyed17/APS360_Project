import numpy as np
import h5py
import time
import os
import random
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms

import matplotlib.pyplot as plt # for plotting
import torch.optim as optim #for gradient descent

import astroNN
from astroNN.datasets import galaxy10
from astroNN.datasets.galaxy10 import galaxy10cls_lookup

from model_architecture_v_1 import *

transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_data = []
val_data = []
test_data = []

with h5py.File('../galaxy_split_augmented.h5', 'r') as gData:
    
    images = np.array(gData['train']['images'])
    labels = np.array(gData['train']['labels'])
    for i in range(len(images)):
        train_data.append(tuple([transform(images[i]),labels[i]]))
    
    images = np.array(gData['val']['images'])
    labels = np.array(gData['val']['labels'])
    for i in range(len(images)):
        val_data.append(tuple([transform(images[i]),labels[i]]))
    
    images = np.array(gData['test']['images'])
    labels = np.array(gData['test']['labels'])
    for i in range(len(images)):
        test_data.append(tuple([transform(images[i]),labels[i]]))


random.shuffle(train_data)
random.shuffle(val_data)
random.shuffle(test_data)

model = galaxy_model()
model.cuda()

train(model, train_data, batch_size=10, num_epochs=30, val_True = True, val_data = val_data)

test_loader = torch.utils.data.DataLoader(test_data, batch_size=10, shuffle=True)
print("Test Accuracy:", get_accuracy(model,test_loader)*100)
