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

transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
images, labels = galaxy10.load_data()


galaxyData = []

for i in range(len(images)):
    galaxyData.append(tuple([transform(images[i]),labels[i]]))

random.Random(1000).shuffle(galaxyData)

train_data = galaxyData[:int(len(galaxyData)*0.6)]
val_data = galaxyData[int(len(galaxyData)*0.6):int(len(galaxyData)*0.8)]
test_data = galaxyData[int(len(galaxyData)*0.8):]

class gesture_model(nn.Module):
  def __init__(self, img_size=69, out1_channels=5, out2_channels=15, out3_channels=30, kernal_size=5):
    super(gesture_model, self).__init__()
    self.name = "gesture_model"

    # Convolutional layers
    self.conv1= nn.Conv2d(3, out1_channels, kernal_size)
    n = self._n_calc(img_size, kernal_size) # calculate intermediate output dimensions
    self.pool1 = nn.MaxPool2d(2, 2)
    n = self._n_calc(n, 2, stride=2) # calculate intermediate output dimensions
    self.conv2= nn.Conv2d(out1_channels, out2_channels, kernal_size)
    n = self._n_calc(n, kernal_size) # calculate intermediate output dimensions
    self.pool2 = nn.MaxPool2d(2, 2)
    n = self._n_calc(n, 2, stride=2) # calculate intermediate output dimensions
    self.conv3= nn.Conv2d(out2_channels, out3_channels, kernal_size)
    n = self._n_calc(n, kernal_size) # calculate intermediate output dimensions
    self.pool3 = nn.MaxPool2d(2, 2)
    n = self._n_calc(n, 2, stride=2) # calculate intermediate output dimensions
    n = int(n)
    # Fully connected layers
    self.fc1 = nn.Linear(out3_channels * n * n, 2000)
    self.fc2 = nn.Linear(2000, 1000)
    self.fc3 = nn.Linear(1000, 10)

  def forward(self, x):
    out = self.pool1(F.relu(self.conv1(x)))
    out = self.pool2(F.relu(self.conv2(out)))
    out = self.pool3(F.relu(self.conv3(out)))
    out = out.view(out.size(0), -1) # flatten
    out = F.relu(self.fc1(out))
    out = self.fc2(out)
    out = out.squeeze(1)
    return out


  def _n_calc(self, n_prev, filter_size, padding=0, stride=1):
    """Calculate height and width dimensions of output."""
        
    n = ((n_prev + 2*padding - filter_size)/stride) + 1

    return n

def get_accuracy(model, data_loader):

    correct = 0
    total = 0
    for imgs, labels in data_loader:
        
        #############################################
        #Enable GPU Usage
        imgs = imgs.cuda()
        labels = labels.cuda()
        #############################################
        
        output = model(imgs) # pass through alexNet then our model
        
        #select index with maximum prediction score
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(labels.view_as(pred)).sum().item()
        total += imgs.shape[0]
    return correct / total


def train(model, train_data, val_True=False, batch_size=20, lr=0.001, num_epochs=1):
   
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    
    if val_True:
      val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    iters, losses, train_acc, val_acc, epochs = [], [], [], [], []

    # training
    start_time=time.time()
    for epoch in tqdm(range(num_epochs)):
        for imgs, labels in iter(train_loader):
          
            
            #############################################
            #Enable GPU Usage
            imgs = imgs.cuda()
            labels = labels.cuda()
            #############################################
            labels = labels.long()
            img = imgs.long()
            out = model(imgs)             # forward pass
            loss = criterion(out, labels) # compute the total loss
            loss.backward()               # backward pass (compute parameter updates)
            optimizer.step()              # make the updates for each parameter
            optimizer.zero_grad()         # a clean up step for PyTorch

        train_acc.append(get_accuracy(model, train_loader))   # compute training accuracy

        if val_data:
          val_acc.append(get_accuracy(model, val_loader))   # compute validation accuracy
          val_accuracy = str(round(val_acc[-1], 4))
        else:
          val_accuracy = "N/A"

        epochs.append(epoch)
        print ("Epoch %d Finished. " % epoch ,"Time per Epoch: % 6.2f s "% ((time.time()-start_time) / (epoch +1)), "Train Accuracy: ", round(train_acc[-1],4), "Validation Accuracy: " + val_accuracy)


    end_time= time.time()
    # plt.title("Training Curve")
    # plt.plot(iters, losses, label="Train")
    # plt.xlabel("Iterations")
    # plt.ylabel("Loss")
    # plt.show()
  
    plt.title("Training Curve")
    plt.plot(epochs, train_acc, label="Train")
    plt.plot(epochs, val_acc, label="Validation")
    plt.xlabel("Iterations")
    plt.ylabel("Training Accuracy")
    plt.legend(loc='best')
    plt.show()


    print("Final Training Accuracy: {}".format(train_acc[-1]))
    if val_data:
      print("Final Validation Accuracy: {}".format(val_acc[-1]))
    print ("Total time:  % 6.2f s  Time per Epoch: % 6.2f s " % ( (end_time-start_time), ((end_time-start_time) / num_epochs) ))
    

model = gesture_model()
model.cuda()

train(model, train_data, batch_size=10, num_epochs=30, val_True = True)