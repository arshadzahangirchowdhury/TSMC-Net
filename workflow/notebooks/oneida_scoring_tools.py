#!/usr/bin/env python3
# -*- coding: utf-8 -*-



"""
Author: M Arshad Zahangir Chowdhury
Email: arshad.zahangir.bd[at]gmail[dot]com
Definitions of various functions for scoring and evaluating the models. (obsolete)
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



def calc_AMCAS(y_test_named,pred_y_named):
    '''
    args
    ----
    y_test_named:str, list of mixtures used for testing.
    pred_y_named:str, list of mixtures predicted by the model. 
    
    returns
    -------
    AMCAS:float, 0-100, A seperately defined accuracy score for mixture, see paper for definition.
    correct_compounds_only:str, list of compounds correctly detected in a mixture by the model.
    
    '''
    

    numerator = 0
    correct_compounds_only = []
    for _ in range(0,y_test_named.shape[0]):

    #         print('actual ', y_test_named[_].split('+', -1))
    #         print('predicted ', pred_y_named[_].split('+', -1))


        n_comp_in_mixture = len(y_test_named[_].split('+', -1))
    #         print("Number of components (actual): ", n_comp_in_mixture)

        correct_compounds = np.intersect1d(y_test_named[_].split('+', -1), pred_y_named[_].split('+', -1))
        correct_compounds_only.append(correct_compounds.tolist())

    #         print("correctly detected: ", len(correct_compounds))

        numerator = numerator +   len(correct_compounds)/(n_comp_in_mixture)

    #         print('individual mixture classification acc. :', len(correct_compounds)/(n_comp_in_mixture))

    print("Adjusted Mixture Classification Accuracy Score (AMCAS): ", 100*numerator/y_test_named.shape[0])    
    return 100*numerator/y_test_named.shape[0], correct_compounds_only


def is_cui_present(cui, y_test_named, correct_compounds_only, verbosity=False):
    '''
    Function to check if compound of interest is present in the detected mixtures and calculate the percentage of detection of such compounds.
    
    args:
    ----
    cui: str, compound of interest
    y_test_named:str, list of mixtures used for testing.
    correct_compounds_only:str, list of compounds correctly detected in a mixture by the model.
    
    returns:
    percent detectection of cui in all mixtures
    
    '''

    cui_present = 0 #cui present in mixture
    cui_not_present = 0 #cui not present in mixture
    cui_pre_det = 0 #cui was present and dectected
    cui_pre_not_det = 0 #cui was present and not detected

    for _ in range(0,y_test_named.shape[0]):

        if cui in y_test_named[_]:
            cui_present+=1
            if cui in correct_compounds_only[_]:
    #             print('yes')
                cui_pre_det+=1
            else:
    #             print('no')
                cui_pre_not_det+=1
        else:
            cui_not_present+=1

    if verbosity==True:
        print(cui)
        print(cui_present)
        print(cui_not_present )
        print(cui_pre_det)
        print(cui_pre_not_det)
        print("% detected:", 100*cui_pre_det/cui_present)
        print("% not detected:", 100*cui_pre_not_det/cui_present)
    
    return 100*cui_pre_det/cui_present

def is_cui_present_in_mult(cui, y_test_named,correct_compounds_only, n_mixture_component = 1,  verbosity=False):
    '''
    Function to chechk if compound of interest is present in the detected mixtures and calculate the percentage of detection of such compounds.
    The function will only look into mixtures with a certain number of components, i.e., n_mixture_component.
    
    args:
    ----
    cui: str, compound of interest
    y_test_named:str, list of mixtures used for testing.
    correct_compounds_only:str, list of compounds correctly detected in a mixture by the model.
    n_mixture_component:int, default = 5. Looks for the cui in mixtures with this amount of components.
    
    returns:
    a list
    percent detectection of cui in terms of all possible mixtures in the set. Not only the mixtures with n_mixture_component.
    percent detectection of cui in terms of mixtures with components in the amount of n_mixture_component. 
    '''

    cui_present = 0 #cui present in mixture
    cui_not_present = 0 #cui not present in mixture
    cui_pre_det = 0 #cui was present and dectected
    cui_pre_not_det = 0 #cui was present and not detected

    for _ in range(0,y_test_named.shape[0]):

        if cui in y_test_named[_]:
#             print(len(y_test_named[_].split('+', -1)))
            if len(y_test_named[_].split('+', -1)) == n_mixture_component:
                cui_present+=1
                if cui in correct_compounds_only[_]:
        #             print('yes')
                    cui_pre_det+=1
                else:
        #             print('no')
                    cui_pre_not_det+=1
        else:
            cui_not_present+=1

    if verbosity==True:
        print(cui)
        print(cui_present)
        print(cui_not_present )
        print(cui_pre_det)
        print(cui_pre_not_det)
        print("% detected:", 100*cui_pre_det/cui_present)
        print("% not detected:", 100*cui_pre_not_det/cui_present)
    
    return [100*cui_pre_det/(y_test_named.shape[0]-cui_not_present), 100*cui_pre_det/cui_present]






