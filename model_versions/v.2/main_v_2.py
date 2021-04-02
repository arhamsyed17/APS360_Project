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

from model_architecture_v_2 import *

transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

images, labels = galaxy10.load_data()


galaxyData = []

for i in range(len(images)):
    galaxyData.append(tuple([transform(images[i]),labels[i]]))

random.Random(1000).shuffle(galaxyData)

train_data = galaxyData[:int(len(galaxyData)*0.6)]
val_data = galaxyData[int(len(galaxyData)*0.6):int(len(galaxyData)*0.8)]
test_data = galaxyData[int(len(galaxyData)*0.8):]


model = galaxy_model(lastOutput = 10)
model.cuda()

train(model, train_data, batch_size=10, num_epochs=50, val_True = True, val_data = val_data, lr = 0.01)

test_loader = torch.utils.data.DataLoader(test_data, batch_size=10, shuffle=True)
print("Test Accuracy:", get_accuracy(model,test_loader)*100)
