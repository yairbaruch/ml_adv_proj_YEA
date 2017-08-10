# -*- coding: utf-8 -*-
"""
Created on Tue Aug 08 11:36:43 2017

@author: yair

'loss' recieves a batch of image feature tensors, prototype features for each image
respectively, a distance function, and an ambiguity cost factor.

'loss' returns the cost as calculated per image pixel, as a distance from its prototype
multiplied by an ambiguity penalty and its factor - miu.

important remarks:
    prototypes needs to be calculated according to the seediness dynamically
    batch is a an array of h images. Each image object is a matrix of the Pixel class.
    Each Pixel class has three fields:
        pred_label - the CCL predicted label of the pixel
        true_label - the true instance to which it belongs
        feature_vector - the pixel's features as mapped by the NN
"""

import numpy as np

def loss(prototypes, batch, dist_func, miu):
    
    cost = 0
    
    for h in range(0, len(batch)):
        
        length, width = batch[h].shape
        
        for i in range(0, width):
        
            for j in range(0, length):
            
                ambiguity = 0
               
                true_label = batch[h][i][j].true_label
                pred_label = batch[h][i][j].pred_label

                pred_f_vec = batch[h][i][j].feature_vector
                true_f_vec = prototypes[h][true_label]
                
                dist = dist_func(pred_f_vec, true_f_vec)

                if pred_label != true_label:
                    "if the pixel was misclassified - calculate ambiguity"
                
                    for k in range(0, len(prototypes[h])):
                    
                        false_f_vec = prototypes[h][k]
                        true_dist = dist_func(true_f_vec, pred_f_vec)
                        false_dist = dist_func(false_f_vec, pred_f_vec)
                        ambiguity += np.exp(-abs(true_dist - false_dist))
                    """
                    ambiguity accumulates the 'boudary factor' which quantifies how 
                    much the pixel has been mapped to areas of unclear decision.
                    
                    remark - a by product of this is that if the pixel has been falsly 
                    predicted, given its true protoype as k will always return the maximal
                    classification ambiguity cost. This might be just helpful.
                    """
                
                cost += dist*(1 + miu*ambiguity)
    
    return cost