#!/usr/bin/env python3
# -*- coding: utf-8 -*-



"""
Author: M Arshad Zahangir Chowdhury
Email: arshad.zahangir.bd[at]gmail[dot]com
Plot grad-cam results for all validation spectra."""

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from matplotlib.ticker import MaxNLocator
import numpy as np
from numpy import asarray
import pandas as pd
import math
import seaborn as sns  #heat map
import glob # batch processing of images


import matplotlib.font_manager as fm
import random
import sys
import os

from sklearn.datasets import make_regression
import tensorflow as tf

from sklearn.metrics import confusion_matrix    #confusion matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Collect all the font names available to matplotlib
font_names = [f.name for f in fm.fontManager.ttflist]
# print(font_names)

from scipy import signal
from scipy import interpolate

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_curve 
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score


from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.gaussian_process.kernels import RBF

#Sklearn model saving and loading
from joblib import dump, load

if '../../' not in sys.path:
    sys.path.append('../../')

from aimos.spectral_datasets.THz_datasets import THz_data

from aimos.misc.utils import simple_plotter


#Set random seed
os.environ['PYTHONHASHSEED'] = str(42)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
tf.random.set_global_generator(42)
tf.random.set_seed(42)  
# tf.random.get_global_generator()
np.random.seed(42)
random.seed(42)

from oneida import THz_mixture_data
from oneida_utils import concentrations_to_one_hot_encode, create_mixture_names
from oneida_utils import simple_spectrum_fig, simple_plot_raw_scores, plot_spectrum_with_scores, multiclass_roc_auc_score, multiclass_sensitivity_specificity_score, multiclass_sensitivity_threshold_score
from oneida_scoring_tools import calc_AMCAS, is_cui_present, is_cui_present_in_mult
from aimos.misc.utils import classifier_internals
from aimos.misc.utils import clf_post_processor
from oneida_utils import mixture_names_to_one_hot_encode
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from oneida_grad_cam import grad_cam

from stats import stats
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from oneida_model import get_callbacks, get_optimizer, compile_and_fit, TSMCN_12_L_229
#Sklearn model saving and loading
from joblib import dump, load

stats(n_compounds=8)

# initialize
TAAT = 0.001 
ASAT=0.01
RSAT=0.01

m = THz_mixture_data(resolution=0.016, pressure='1 Torr', verbosity=False)
m.initiate_THz_mixture_data(TAAT = TAAT, 
                               ASAT=ASAT, 
                               RSAT=RSAT)

reduced_labels = m.labels
reduced_labels.remove('')
reduced_labels.remove(' ')
reduced_labels.remove('Diluent')
print('reduced_labels', reduced_labels)


# data_filename = "datasets/TSMCN-5-L-229_DV_04-09-2022_time_22-26-37.pkl"
data_filename = "datasets/TSMCN-8-L-229_DV__TAAT_0.001_ASAT_0.01_RSAT_0.01_20-10-2022_time_23-16-29_class_cnt_90.pkl"
DV = pd.read_pickle(data_filename)
y = DV['y'].to_numpy()
mixture_names = DV['mixture_names'].to_numpy()
y_concentrations = DV[['y_c0', 'y_c1', 'y_c2','y_c3', 'y_c4', 'y_c5', 'y_c6', 'y_c7']].to_numpy()
X = DV.drop(['y','mixture_names', 'y_c0', 'y_c1', 'y_c2','y_c3', 'y_c4', 'y_c5', 'y_c6', 'y_c7'],axis=1).to_numpy()

final_neuron_number = np.unique(y, axis=0).shape[0]
print('Number of neurons in the final layer :', final_neuron_number)

print('labels from class:', m.labels)



le = preprocessing.LabelEncoder()
le.fit(mixture_names)

mixture_types=le.classes_
# print(mixture_types)

#split intro train and validation set

#seeds used 123,237, 786


global_indices=range(0, X.shape[0])
print(global_indices)

# (np.expand_dims(X,-1)
TRAIN_SIZE=0.60
VAL_SIZE=1-TRAIN_SIZE

x_train, x_val, y_train, y_val, train_indices, val_indices = train_test_split(np.expand_dims(X, axis=-1), y, global_indices, train_size=TRAIN_SIZE,
                                                   test_size=VAL_SIZE,
                                                   random_state=786,
                                                    stratify=y

                                                   )

print('X_train shape:', x_train.shape)
print('y_ohe_train shape:', y_train.shape)

print('X_val shape:', x_val.shape)
print('y_ohe_val shape:', y_val.shape)


print("All:", np.bincount(y) / float(len(y))*100  )
print("Training:", np.bincount(y_train) / float(len(y_train))*100  )
print("Validation:", np.bincount(y_val) / float(len(y_val))*100  )


model_name = data_filename.split('.pkl')[0].split('/')[1]
print(model_name)



model = tf.keras.models.load_model('model/TSMCN-8-L-229_DV__TAAT_0.001_ASAT_0.01_RSAT_0.01_20-10-2022_time_23-16-29_class_cnt_90_20-10-2022_time_23-30-09.hdf5', compile=False)


probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])
predictions = probability_model.predict(x_val)
pred_y=np.argmax(model.predict(x_val), axis=-1)

# grad-cam
layer_name = 'C5'


print(f'layer {layer_name} class activation maps')

font = {'family': 'serif',
        'color':  'black',
        'weight': 'bold',
        'size': 16,
        }

freq_GHz = 29.9792458*m.frequencies


count = 0





for i,j,k in zip(x_val,y_val,pred_y):
    print('count:', count)
    
    data = np.expand_dims(i,0)
    heatmap = grad_cam(layer_name,data, model)
#     heatmap = normalize(grad_cam(layer_name,data, model), axis=0)
    normalized_hm = [(x-np.min(heatmap[0]))/(np.max(heatmap[0])-np.min(heatmap[0])) for x in heatmap[0]]
    heatmap = np.expand_dims(normalized_hm,0)
#     import pdb; pdb.set_trace()

    
    
    # raw map
    fig = plt.figure(figsize=(30,4),dpi=300)
    plt.imshow(np.expand_dims(heatmap,axis=2),cmap='inferno', aspect="auto", 
               interpolation='nearest',extent=[0,229,i.min(),i.max()], alpha=0.8)
    
    ticklist = range(0,229)
#     plt.xticks(ticklist[::30], np.round(m.frequencies.tolist()[::30], decimals=1) ) # tick every 40th frequency
    plt.xticks(ticklist[::20], np.round(freq_GHz[::20], decimals=1) ) # tick every 40th frequency
    plt.plot(i,'k',linewidth=3)
    
    if mixture_types[j] != mixture_types[k]:
        plt.title(f'actual:{mixture_types[j]}, predicted:{mixture_types[k]}', color='red', fontdict=font)
    else:
        plt.title(f'actual:{mixture_types[j]}, predicted:{mixture_types[k]}', color='black', fontdict=font)
    plt.colorbar()
    plt.clim(np.min(heatmap),np.max(heatmap))
    plt.close()
    
    
    fig.savefig(r'RESULTS/grad_cam_multi_class/CAM_C5_val' + str(count) + '.png', bbox_inches='tight')
    count = count + 1
    