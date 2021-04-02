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

from model_architecture_v_8 import *

transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_data = []
val_data = []
test_data = []

with h5py.File('../split_aug_v6_70.h5', 'r') as gData:
    
    images = np.array(gData['train']['images'])
    labels = np.array(gData['train']['labels'])
    for i in tqdm(range(len(images))):
        train_data.append(tuple([transform(images[i]),labels[i]]))
    
    images = np.array(gData['val']['images'])
    labels = np.array(gData['val']['labels'])
    for i in tqdm(range(len(images))):
        val_data.append(tuple([transform(images[i]),labels[i]]))
    
    images = np.array(gData['test']['images'])
    labels = np.array(gData['test']['labels'])
    for i in tqdm(range(len(images))):
        test_data.append(tuple([transform(images[i]),labels[i]]))

torch.manual_seed(300)
random.shuffle(train_data)
random.shuffle(val_data)
random.shuffle(test_data)

low_train = train_data[0:int(0.6*len(train_data))]
low_val = val_data[0:int(0.6*len(val_data))]

model = galaxy_model()
model.cuda()

train(model, train_data, batch_size= 20, num_epochs=50, val_True = True, val_data = val_data, lr = 0.01)

test_loader = torch.utils.data.DataLoader(test_data, batch_size=10, shuffle=True)
print("Test Accuracy:", get_accuracy(model,test_loader)*100)
