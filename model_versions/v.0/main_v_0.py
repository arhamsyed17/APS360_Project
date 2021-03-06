import numpy as np
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

from model_architecture import *

transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
images, labels = galaxy10.load_data()


galaxyData = []

for i in range(len(images)):
    galaxyData.append(tuple([transform(images[i]),labels[i]]))

random.Random(1000).shuffle(galaxyData)

train_data = galaxyData[:int(len(galaxyData)*0.6)]
val_data = galaxyData[int(len(galaxyData)*0.6):int(len(galaxyData)*0.8)]
test_data = galaxyData[int(len(galaxyData)*0.8):]


model = galaxy_model()
model.cuda()

train(model, train_data, batch_size=10, num_epochs=30, val_True = True, val_data = val_data)

class_5_test = [i for i in test_data if i[1] == 5]
class_1_test = [i for i in test_data if i[1] == 1]
test_loader_5 = torch.utils.data.DataLoader(class_5_test, batch_size=1, shuffle=True)
test_loader_1 = torch.utils.data.DataLoader(class_1_test, batch_size=1, shuffle=True)
print('Class 5 Accuracy: ',get_accuracy(model,test_loader_5))
print('Class 1 Accuracy: ',get_accuracy(model,test_loader_1))