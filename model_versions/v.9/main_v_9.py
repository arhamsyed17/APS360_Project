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

from model_architecture_v_9 import *
from confusion_matrix import *

transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_data = []
val_data = []
test_data = []

with h5py.File('../../splitData/newData/split_aug_v7.h5', 'r') as gData:
    
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



model = galaxy_model()
model.cuda()

train(model, train_data, batch_size= 20, num_epochs=30, val_True = True, val_data = val_data, lr = 0.01)

#state = torch.load('./checkpoints/model_galaxyNet_bs10_lr0.01_epoch20_acc81')
#model.load_state_dict(state)

test_loader = torch.utils.data.DataLoader(test_data, batch_size= 10, shuffle=True)
print("Test Accuracy:", get_accuracy(model,test_loader)*100)
tp,fp,fn, C = confusion_matrix(test_loader,model)
#plt.imshow(C)

size = 10
x_start = 0
x_end = 10.0
y_start = 0
y_end = 10.0

extent = [x_start, x_end, y_start, y_end]

# The normal figure
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111)
im = ax.imshow(C, cmap = "Blues")

# Add the text
jump_x = (x_end -10-x_start) / (2.0 * size)
jump_y = (y_end -10- y_start) / (2.0 * size)
x_positions = np.linspace(start=x_start, stop=x_end, num=size, endpoint=False)
y_positions = np.linspace(start=y_start, stop=y_end, num=size, endpoint=False)
plt.ylabel("True Labels", fontsize = 20)
plt.xlabel("Predicted Labels", fontsize = 20)
for y_index, y in enumerate(y_positions):
    for x_index, x in enumerate(x_positions):
        label = C[y_index, x_index]
        text_x = x + jump_x
        text_y = y + jump_y
        if (label>100):
            ax.text(text_x, text_y, label, color='white', ha='center', va='center', size = 'x-large')
        else:
            ax.text(text_x, text_y, label, color='black', ha='center', va='center', size = 'x-large')

plt.xticks(np.arange(10), fontsize = 15)
plt.yticks(np.arange(10), fontsize = 15)
#fig.colorbar(im)
plt.minorticks_on()
plt.show()
