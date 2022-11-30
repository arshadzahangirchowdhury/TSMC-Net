#!/usr/bin/env python3
# -*- coding: utf-8 -*-



"""
Author: M Arshad Zahangir Chowdhury
Email: arshad.zahangir.bd[at]gmail[dot]com
Definitions of various functions for plotting results for tsmc-net.
"""

import sys
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal
from ipywidgets import interactive
import seaborn as sns  #heat map
import glob # batch processing of images

if '../../' not in sys.path:
    sys.path.append('../../')

import math
from scipy import signal
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_curve 
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score

import itertools

from aimos.misc.utils import classifier_internals
from aimos.misc.utils import clf_post_processor


from aimos.misc.aperture import publication_fig



def concentrations_to_one_hot_encode(y):
    '''
    args:
    y, a vector containing the concentration values for each component.

    return vector containing binary one hot encoding
    '''


    y_ohe = y
    for i in range(y.shape[0]):

        for j in range(y.shape[1]):
            if y[i][j] > 0:
                y_ohe[i][j]=1    

    return y_ohe

def create_mixture_names(ohe_label,n_mixture_component_max,reduced_labels):

    '''
    args:
    ohe_label: a one-hot encoded label corresponding to a mixture with components in it contain.
    n_mixture_component_max: maximum number of pure components in the mixture.
    reduced_labels: label list containing the names of only the pure mixtures.


    '''


    capture_labels=[]
    for _ in range(n_mixture_component_max):
        if ohe_label[_]==1:
            capture_labels.append(reduced_labels[_])


    mixture_name_label=""
    for i in capture_labels:
        mixture_name_label+= i +"+"




    return mixture_name_label.strip(mixture_name_label[-1])



def simple_spectrum_fig(frequencies, absorbances):
    
    spectrum_plot = plt.plot(frequencies, absorbances/max(absorbances), linewidth = 0.5, color = 'black')
    plt.xlabel('Frequency ($cm^{-1}$)')
    plt.ylabel('Norm. Abs.')
    plt.xlim([frequencies[0], frequencies[-1]])

def simple_plot_raw_scores(i, predictions_array, true_label,all_unique_labels):
    
    true_label = true_label[i]
    plt.grid(False)
    plt.yticks(range(255),all_unique_labels)

    scoreplot = plt.barh(range(255), predictions_array[i], color="#777777")
    
    plt.xlim([0, 1])
    predicted_label = np.argmax(predictions_array[i])
    plt.yticks(fontsize = 7);
    plt.tick_params(axis = 'y', direction = 'out') # , pad =-335
    
    plt.ylabel('label')
    plt.xlabel('softmax score')
    scoreplot[predicted_label].set_color('red')
    scoreplot[true_label].set_color('blue')
    


def plot_spectrum_with_scores(x,y, predictions, frequencies, all_unique_labels, start=0,dpi=300):
    '''Plot a spectrum, its predicted labels, and the true labels.
Color correct predictions in blue and incorrect predictions in red.'''
    num_rows = 1
    num_cols = 1
    
    fig = plt.figure(figsize=(16, 40),dpi=dpi)

    if start<0:
        start=0

    
    plt.subplot(2, 1, 1)
    simple_spectrum_fig(frequencies, x[int(start)])
    plt.subplot(2, 1, 2)
    simple_plot_raw_scores(int(start), predictions, y,all_unique_labels)
    plt.tight_layout()
#     plt.show()
    
    return fig



def multiclass_roc_auc_score(y_test, y_pred, target, average="macro", figsize = (12, 8), dpi=300 ):
    '''function for scoring roc auc score for multi-class'''
    
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)
    
    fig, c_ax = plt.subplots(1,1, figsize = figsize, dpi=dpi)

    for (idx, c_label) in enumerate(target):
        fpr, tpr, thresholds = roc_curve(y_test[:,idx].astype(int), y_pred[:,idx])
        c_ax.plot(fpr, tpr, label = '%s (AUC:%0.2f)'  % (c_label, auc(fpr, tpr)))
    
    c_ax.plot(fpr, fpr, 'b-', label = 'Random Guessing')    
    c_ax.legend(loc=4,prop={'size': 6})
    
    c_ax.set_xlabel('False Positive Rate')
    c_ax.set_ylabel('True Positive Rate')

    plt.close()
    
    return roc_auc_score(y_test, y_pred, average=average), fig, c_ax


def multiclass_sensitivity_specificity_score(y_test, y_pred, target, average="macro", figsize = (12, 8), dpi=300 ):
    '''function for scoring roc auc score for multi-class'''
    
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)
    
    fig, c_ax = plt.subplots(1,1, figsize = figsize, dpi=dpi)

    for (idx, c_label) in enumerate(target):
        fpr, tpr, thresholds = roc_curve(y_test[:,idx].astype(int), y_pred[:,idx])
        c_ax.plot(1-fpr, tpr, label = '%s (AUC:%0.2f)'  % (c_label, auc(1-fpr, tpr)))
    
#     c_ax.plot(fpr, fpr, 'b-', label = 'Random Guessing')    
    c_ax.legend(loc=3,prop={'size': 6})
    
    c_ax.set_xlabel('Specificity (1 - False Positive Rate)')
    c_ax.set_ylabel('Sensitivity (True Positive Rate)')

    plt.close()
    
    return roc_auc_score(y_test, y_pred, average=average), fig, c_ax

def multiclass_sensitivity_threshold_score(y_test, y_pred, target, average="macro", figsize = (12, 8), dpi=300 ):
    '''function for scoring roc auc score for multi-class'''
    
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)
    
    fig, c_ax = plt.subplots(1,1, figsize = figsize, dpi=dpi)

    for (idx, c_label) in enumerate(target):
        fpr, tpr, thresholds = roc_curve(y_test[:,idx].astype(int), y_pred[:,idx])
        c_ax.plot(thresholds, tpr, label = '%s (AUC:%0.2f)'  % (c_label, auc(fpr, tpr)))
    
#     c_ax.plot(fpr, fpr, 'b-', label = 'Random Guessing')    
    c_ax.legend(loc=3,prop={'size': 6})
    
    c_ax.set_xlabel('Threshold')
    c_ax.set_ylabel('Sensitivity (True Positive Rate)')

    plt.close()
    
    return roc_auc_score(y_test, y_pred, average=average), fig, c_ax



def mixture_names_to_one_hot_encode(mixture_array_named, reduced_labels, verbosity = False):
    '''
    args:
    -----
    mixture_array_named:str, list of mixture names with + strings indicating multiple components
    reduced_labels:str, list of unique components (molecules) considered
    
    return:
    -------
    one hot encoded label
    
    '''
    
    pred_y_ohe = []
    
    for named_value in mixture_array_named:
        species_list =  named_value.split('+')
        n_species = len(species_list)
        if verbosity == True:
            print("n_species: ", n_species)
            print(species_list)
        output_hot_encode = [0,0,0,0,0,0,0,0]
        for _ in range(0, n_species):
            if verbosity == True:
                print(species_list[_])
            sp_index = reduced_labels.index(species_list[_])
            if verbosity == True:
                print(sp_index)
            output_hot_encode[sp_index] = 1

        if verbosity == True:
            print(output_hot_encode)  
        pred_y_ohe.append(output_hot_encode)
        
    return np.array(pred_y_ohe)


def plot_scores_bars(x,y, predictions, frequencies, all_unique_labels, start=0,dpi=300):
    '''Plot bar plots of softmax scores'''
    
    fig = plt.figure(figsize=(16, 40),dpi=dpi)

    if start<0:
        start=0

    plt.plot
    simple_plot_raw_scores(int(start), predictions, y,all_unique_labels)
    plt.tight_layout()
#     plt.show()
    
    return fig
        
    