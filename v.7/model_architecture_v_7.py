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

def get_model_name(name, batch_size, learning_rate, epoch,valAcc):
    """ Generate a name for the model consisting of all the hyperparameter values

    Args:
        config: Configuration object containing the hyperparameters
    Returns:
        path: A string with the hyperparameter name and value concatenated
    """
    valAcc = str(round(valAcc[-1], 4))
    path = "./checkpoints/model_{0}_bs{1}_lr{2}_epoch{3}_acc{4}".format(name,
                                                   batch_size,
                                                   learning_rate,
                                                   epoch,
                                                   valAcc)
    return path


class galaxy_model(nn.Module):
  def __init__(self, img_size=125, out1_channels=20, out2_channels=30, out3_channels=100,kernal_size=3, lastOutput = 10):
    super(galaxy_model, self).__init__()
    self.name = "galaxyNet"
    self.out3_channels = out3_channels
    self.avgPool = nn.AvgPool2d(2,2)
    # Convolutional layers
    self.conv1= nn.Conv2d(3, out1_channels, kernal_size)
    n = self._n_calc(img_size, kernal_size) # calculate intermediate output dimensions
    print('\nConv1 Output Size:', np.floor(n))
    print('Number of Parameters from Conv1:', self.numParamCNN(3,out1_channels,kernal_size))
    suM = self.numParamCNN(3,out1_channels,kernal_size)
    
    self.pool1 = nn.AvgPool2d(2,2)
    n = self._n_calc(n, 2, stride=2) # calculate intermediate output dimensions
    print('\nPool1 Output Size:', np.floor(n))
    
    self.conv2= nn.Conv2d(out1_channels, out2_channels, kernal_size)
    n = self._n_calc(n, kernal_size) # calculate intermediate output dimensions
    print('\nConv2 Output Size:', np.floor(n))
    print('Number of Parameters from Conv2:', self.numParamCNN(out1_channels,out2_channels,kernal_size))
    suM +=  self.numParamCNN(out1_channels,out2_channels,kernal_size)
    
    self.pool2 = nn.AvgPool2d(2,2)
    n = self._n_calc(n, 2, stride=2) # calculate intermediate output dimensions
    n = int(n)
    print('\nPool2 Output Size:', np.floor(n))
    
    self.conv3= nn.Conv2d(out2_channels, out3_channels, kernal_size)
    n = self._n_calc(n, kernal_size) # calculate intermediate output dimensions
    print('\nConv3 Output Size:', np.floor(n))
    print('Number of Parameters from Conv2:', self.numParamCNN(out2_channels,out3_channels,kernal_size))
    suM +=  self.numParamCNN(out2_channels,out3_channels,kernal_size)
    
    self.pool3 = nn.AvgPool2d(2,2)
    n = self._n_calc(n, 2, stride=2) # calculate intermediate output dimensions
    n = int(n)
    print('\nPool3 Output Size:', np.floor(n))
    
    
    
    # Fully connected layers 
    self.fc1 = nn.Linear(out3_channels * n * n, lastOutput)
    print('\nFc1 Output Size:', lastOutput)
    print('Number of Parameters from Fc1:', self.numParamANN(out3_channels*n*n,lastOutput))
    suM +=  self.numParamANN(out3_channels*n*n,lastOutput)
    
    print('\nTotal Number of Parameters:', suM)
    
    self.n = n
    self.droput = nn.Dropout(0.8)

  def forward(self, x):
    out = self.pool1(F.relu(self.conv1(x)))
    out = self.pool2(F.relu(self.conv2(out)))
    out = self.pool3(F.relu(self.conv3(out)))
    out = out.view(out.size(0), -1) # flatten
    #out = self.droput(out)
    out = self.fc1(out)
    #out = self.droput(out)
    out = out.squeeze(1)
    return out



  def _n_calc(self, n_prev, filter_size, padding=0, stride=1):
    """Calculate height and width dimensions of output."""
        
    n = ((n_prev + 2*padding - filter_size)/stride) + 1

    return n

  def numParamCNN(self, inSize,outSize,filterSize):
    return ((inSize*(filterSize**2)+1)*outSize)
    
  def numParamANN(self,inSize,outSize):
    return (inSize+1)*outSize

def get_accuracy(model, data_loader):

    correct = 0
    total = 0
    for imgs, labels in data_loader:
        
        #############################################
        #Enable GPU Usage
        imgs = imgs.cuda()
        labels = labels.cuda()
        #############################################
        
        output = model(imgs)
        
        #select index with maximum prediction score
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(labels.view_as(pred)).sum().item()
        total += imgs.shape[0]
    return correct / total


def train(model, train_data, val_data, val_True=False,batch_size=20, lr=0.001, num_epochs=1):
   
    torch.manual_seed(200)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    
    if val_True:
      val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.8)
    #optimizer = optim.Adam(model.parameters(), lr=lr)

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

        if (epoch%3 == 0):
            train_acc.append(get_accuracy(model, train_loader))   # compute training accuracy
    
            if val_True:
                valAcc = get_accuracy(model, val_loader)
                val_acc.append(valAcc)   # compute validation accuracy
                val_accuracy = str(round(val_acc[-1], 4))
              
                if (valAcc>=0.8):
                    model_path = get_model_name(model.name, batch_size, lr, epoch, valAcc)
                    torch.save(model.state_dict(), model_path)
            else:
                val_accuracy = "N/A"
    
            epochs.append(epoch)
            print ("Epoch %d Finished. " % epoch ,"Time per Epoch: % 6.2f s "% ((time.time()-start_time) / (epoch +1)), "\nTrain Accuracy: ", round(train_acc[-1],4), "Validation Accuracy: " + val_accuracy)
        else:
            print ("Epoch %d Finished. " % epoch ,"Time per Epoch: % 6.2f s "% ((time.time()-start_time) / (epoch +1)))

    end_time= time.time()
    # plt.title("Training Curve")
    # plt.plot(iters, losses, label="Train")
    # plt.xlabel("Iterations")
    # plt.ylabel("Loss")
    # plt.show()
  
    plt.title("Training Curve")
    plt.plot(epochs, train_acc, label="Train")
    plt.plot(epochs, val_acc, label="Validation")
    plt.xlabel("Epochs")
    plt.ylabel("Training Accuracy")
    plt.legend(loc='best')
    plt.show()


    print("Final Training Accuracy: {}".format(train_acc[-1]))
    if val_data:
      print("Final Validation Accuracy: {}".format(val_acc[-1]))
    print ("Total time:  % 6.2f s  Time per Epoch: % 6.2f s " % ( (end_time-start_time), ((end_time-start_time) / num_epochs) ))