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

    
with h5py.File('./splitData/split_aug_v1.h5', 'r') as gData:
    
    #images = np.array(gData['train']['images'])
    labels = np.array(gData['train']['labels'])
    labels2 = np.array(gData['val']['labels'])
    labels3 = np.array(gData['test']['labels'])
    plt.imshow(gData['train']['images'][0])
    
gData.close()

print("Total images:", len(labels) + len(labels2) + len(labels3))
count0=0
count1=0
count2=0
count3=0
count4=0
count5=0
count6=0
count7=0
count8=0
count9=0

for i in labels:
    if (i==0):
        count0+=1
    if (i==1):
        count1+=1   
    if (i==2):
        count2+=1
    if (i==3):
        count3+=1
    if (i==4):
        count4+=1
    if (i==5):
        count5+=1
    if (i==6):
        count6+=1
    if (i==7):
        count7+=1
    if (i==8):
        count8+=1
    if (i==9):
        count9+=1
        
for i in labels2:
    if (i==0):
        count0+=1
    if (i==1):
        count1+=1   
    if (i==2):
        count2+=1
    if (i==3):
        count3+=1
    if (i==4):
        count4+=1
    if (i==5):
        count5+=1
    if (i==6):
        count6+=1
    if (i==7):
        count7+=1
    if (i==8):
        count8+=1
    if (i==9):
        count9+=1        
        
for i in labels3:

    if (i==0):
        count0+=1
    if (i==1):
        count1+=1   
    if (i==2):
        count2+=1
    if (i==3):
        count3+=1
    if (i==4):
        count4+=1
    if (i==5):
        count5+=1
    if (i==6):
        count6+=1
    if (i==7):
        count7+=1
    if (i==8):
        count8+=1
    if (i==9):
        count9+=1    
        
countTot = [count0,count1,count2,count3,count4,count5,count6,count7,count8,count9]

tot = 0
for j in range(len(countTot)):
    print("Class",j,": ", countTot[j])