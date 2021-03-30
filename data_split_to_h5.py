import h5py
import numpy as np
import random as rd
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

with h5py.File('./augmented_data/new/new_aug_v5.h5', 'r') as F:
    images = np.array(F['images'])
    labels = np.array(F['labels'])

class_0 = []
class_1 = []
class_2 = []
class_3 = []
class_4 = []
class_5 = []
class_6 = []
class_7 = []
class_8 = []
class_9 = []

print(len(labels))
#split into the 10 different classes (image and label combined)
for i in tqdm(range(len(images))):
    if (labels[i] == 0):
        class_0.append([images[i],labels[i]])
        
    if (labels[i] == 1):
        class_1.append([images[i],labels[i]])
        
    if (labels[i] == 2):
        class_2.append([images[i],labels[i]])
    
    if (labels[i] == 3):
        class_3.append([images[i],labels[i]])
        
    if (labels[i] == 4):
        class_4.append([images[i],labels[i]])
        
    if (labels[i] == 5):
        class_5.append([images[i],labels[i]])
        
    if (labels[i] == 6):
        class_6.append([images[i],labels[i]])
        
    if (labels[i] == 7):
        class_7.append([images[i],labels[i]])
        
    if (labels[i] == 8):
        class_8.append([images[i],labels[i]])
        
    if (labels[i] == 9):
        class_9.append([images[i],labels[i]])
        

random.shuffle(class_0)
random.shuffle(class_1)
random.shuffle(class_2)
random.shuffle(class_3)
random.shuffle(class_4)
random.shuffle(class_5)
random.shuffle(class_6)
random.shuffle(class_7)
random.shuffle(class_8)
random.shuffle(class_9)
#combines all classes into one for sake of looping
class_Ult = [class_0, class_1, class_2, class_3, class_4,class_5, class_6, class_7, class_8, class_9]

trainList = []
valList = []
testList = []

#looping through each class
for i in class_Ult:
    #shuffling the items in each class
    rd.shuffle(i)

    iLen = len(i)
    trainSplice = int(0.8*iLen)
    valSplice = int(0.9*iLen)

    #from each class, 80% to training, 10% to validation, and 10% to testing
    trainList.append(i[0:trainSplice])
    valList.append(i[trainSplice:valSplice])
    testList.append(i[valSplice:])
    
#flattening out the lists
trainList = [val for sublist in trainList for val in sublist]
valList = [val for sublist in valList for val in sublist]
testList = [val for sublist in testList for val in sublist]

#correcting the format so that the list is as follows -> [[images], [corresponding tags]]
trainList = [[i[0] for i in trainList],[i[1] for i in trainList]]
valList = [[i[0] for i in valList],[i[1] for i in valList]]
testList = [[i[0] for i in testList],[i[1] for i in testList]]


#creating the h5 file
hf = h5py.File('split_aug_v6.h5', 'w')    

g1 = hf.create_group('train')
g2 = hf.create_group('val')
g3 = hf.create_group('test')

g1.create_dataset('images',data = trainList[0])
g1.create_dataset('labels',data = trainList[1])

g2.create_dataset('images',data = valList[0])
g2.create_dataset('labels',data = valList[1])

g3.create_dataset('images',data = testList[0])
g3.create_dataset('labels',data = testList[1])

hf.close()



