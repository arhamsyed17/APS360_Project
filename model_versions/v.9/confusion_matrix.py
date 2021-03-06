# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 09:52:40 2021

@author: ibathrahman
"""
import numpy as np


# confusion matrix where the rows are true labels and
# the coloumns are the predicted label

def confusion_matrix(test_loader_data, model, num_classes = 10):


    C_matrix = np.zeros((num_classes, num_classes), dtype=int)
    
    for img, label in iter(test_loader_data):
          
        img = img.cuda()
        label = label.cuda()
        
        out = model(img)
        pred = out.max(1, keepdim=True)[1]
        
        pred = pred.cpu().data.numpy()
        label = label.cpu().data.numpy()
        
        for j in range(len(pred)):
            C_matrix[label[j]][pred[j]] += 1
       
        
        
    
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
    
    # intialize F1 Score Matrix
    dict_F1_scores = {}
    dict_Precision = {}
    dict_Recall = {}

    for label in range(num_classes):
        dict_F1_scores["Class " + str(label)] = 0
        dict_Precision["Class " + str(label)] = 0
        dict_Recall["Class " + str(label)] = 0

    for label in range(num_classes):
        tp = dict_true_positives["Class " + str(label)] 
        fp = dict_false_positives["Class " + str(label)]
        fn = dict_false_negatives["Class " + str(label)]

        precision = tp / (tp + fp)
        recall = tp / (tp + fn) 

        dict_Precision["Class " + str(label)] = round(precision, 4)
        dict_Recall["Class " + str(label)] = round(recall, 4)
        dict_F1_scores["Class " + str(label)] = round(2* precision*recall / (precision + recall), 4)
    
    return dict_true_positives, dict_false_positives, dict_false_negatives, C_matrix, dict_Precision, dict_Recall, dict_F1_scores
    
            
            
        
