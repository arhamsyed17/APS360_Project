import h5py
import numpy as np
import random as rd
import matplotlib.pyplot as plt

with h5py.File('images_augmented.h5', 'r') as F:
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


for i in range(len(images)):
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
        
 
class_Ult = [class_0, class_1, class_2, class_3, class_4, class_5, class_6, class_7, class_8, class_9]

trainList = []
valList = []
testList = []

for i in class_Ult:
    rd.shuffle(i)

    iLen = len(i)
    trainSplice = int(0.6*iLen)
    valSplice = int(0.8*iLen)

    trainList.append(i[0:trainSplice])
    valList.append(i[trainSplice:valSplice])
    testList.append(i[valSplice:])
    

trainList = [val for sublist in trainList for val in sublist]
valList = [val for sublist in valList for val in sublist]
testList = [val for sublist in testList for val in sublist]

trainList = [[i[0] for i in trainList],[i[1] for i in trainList]]
valList = [[i[0] for i in valList],[i[1] for i in valList]]
testList = [[i[0] for i in testList],[i[1] for i in testList]]



hf = h5py.File('galaxy_split_augmented.h5', 'w')    

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

# with h5py.File('galaxy_split_augmented.h5', 'r') as P:
#     train = np.array(P['train'])

