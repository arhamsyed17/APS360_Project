# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 09:52:40 2021

@author: ibathrahman
"""
import numpy as np

num_classes = 10


# confusion matrix where the rows are true labels and
# the coloumns are the predicted label
C_matrix = np.zeros((num_classes, num_classes), dtype=int)

for img, label in iter(test_loader):
    
    out = model(img)
    pred = output.max(1, keepdim=True)[1]
    
    C_matrix[label][pred] += 1
    
    

#C_matrix = np.array([[12, 1, 0], [0, 12, 4], [0, 1, 8]])

# intialize dictionaries
dict_true_positives = {}
dict_false_positives = {}
dict_false_negatives = {}

for label in range(num_classes):
    dict_true_positives["Class " + str(label)] = 0
    dict_false_positives["Class " + str(label)] = 0
    dict_false_negatives["Class " + str(label)] = 0 
        

# Calculate metrics from matrix
for label in range(num_classes):
    
    # True positives
    dict_true_positives["Class " + str(label)] = C_matrix[label, label]
    
    # False Positives
    dict_false_positives["Class " + str(label)]= sum(C_matrix[:, label]) - C_matrix[label, label]
    
    # False negatives
    dict_false_negatives["Class " + str(label)] = sum(C_matrix[label]) - C_matrix[label, label]
    
    
    
            
            
        