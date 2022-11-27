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
tf.random.set_seed(42)  
tf.random.get_global_generator().reset_from_seed(42)
np.random.seed(42)
random.seed(42)

from oneida import THz_mixture_data
from oneida_utils import concentrations_to_one_hot_encode, create_mixture_names

from stats import stats
stats(n_compounds=8)


# initialize
TAAT = 0.001 
ASAT=0.01
RSAT=0.05

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
data_filename = "datasets/TSMCN-8-L-229_DV__TAAT_0.001_ASAT_0.01_RSAT_0.05_21-09-2022_time_18-53-02.pkl"
DV = pd.read_pickle(data_filename)
y = DV['y'].to_numpy()
mixture_names = DV['mixture_names'].to_numpy()
y_concentrations = DV[['y_c0', 'y_c1', 'y_c2','y_c3', 'y_c4', 'y_c5', 'y_c6', 'y_c7']].to_numpy()
X = DV.drop(['y','mixture_names', 'y_c0', 'y_c1', 'y_c2','y_c3', 'y_c4', 'y_c5', 'y_c6', 'y_c7'],axis=1).to_numpy()

final_neuron_number = np.unique(y, axis=0).shape[0]
print('Number of neurons in the final layer :', final_neuron_number)

print('labels from class:', m.labels)


from sklearn import preprocessing

le = preprocessing.LabelEncoder()
le.fit(mixture_names)

mixture_types=le.classes_
print(mixture_types)

#split intro train and validation set

#seeds used 123,237, 786
from sklearn.model_selection import train_test_split

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

from oneida_model import get_callbacks, get_optimizer, compile_and_fit, TSMCN_12_L_229

model = TSMCN_12_L_229()

# tf.keras.utils.plot_model(model, to_file="RESULTS/TSMCN_8_L_229.png", show_shapes=True, rankdir='TB', dpi=150)

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath='checkpoints/',
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

model_name = data_filename.split('.pkl')[0].split('/')[1]
print(model_name)

with tf.device('/CPU:0'):
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='SparseCatCrossentropy', patience=5)
    history = model.fit(x_train, y_train, epochs=30, validation_data=(x_val, y_val), 
                        callbacks=[model_checkpoint_callback, stop_early])
    
probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])
predictions = probability_model.predict(x_val)

model_name = data_filename.split('.pkl')[0].split('/')[1].replace('TSMCN', 'TSMMLCN')
print(model_name)

y_ohe = concentrations_to_one_hot_encode(y_concentrations).astype('int64')
y_train_ohe = y_ohe[train_indices]
y_val_ohe = y_ohe[val_indices]
y_train_ohe_tensor = tf.convert_to_tensor(y_train_ohe, np.int64)
y_val_ohe_tensor = tf.convert_to_tensor(y_val_ohe, np.int64)


plt.figure(dpi=150)
plt.scatter(history.epoch,history.history['accuracy'], color = 'red', label = 'training')
plt.scatter(history.epoch,history.history['val_accuracy'], color = 'blue', label = 'validation')
plt.legend(loc=4)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.savefig(r'RESULTS/results_figures/' + model_name + '_accuracies.png', bbox_inches='tight')

plt.figure(dpi=150)
plt.scatter(history.epoch,history.history['SparseCatCrossentropy'], color = 'red', label = 'training')
plt.scatter(history.epoch,history.history['val_SparseCatCrossentropy'],color = 'blue', label = 'validation')
plt.legend(loc=1)
plt.xlabel('Epoch')
plt.ylabel('Sparse categorical crossentropy loss')
plt.savefig(r'RESULTS/results_figures/'+ model_name + '_sparse_cat_losses.png', bbox_inches='tight')



exps = ['2 Comp-mix_ 30 % CH3Cl - 70% CH3CN/Mix 50% Dilute CM-ACN.xlsx',
'2 Comp-mix_ 30 % CH3Cl - 70% CH3CN/Pure Mix CM-ACN.xlsx',
'2 Comp-mix_ 30 % CH3Cl - 70% CH3CN/Mix 90% Dilute CM-ACN.xlsx',
'3 Comp-mix_ 90+% CH3OH + 5-% CH3CN + 5-% CH3CL/0.9 CH3OH + 0.05 CH3CN + 0.05 CH3Cl - 1.xlsx',
'3 Comp-mix_ 90+% CH3OH + 5-% CH3CN + 5-% CH3CL/0.9 CH3OH + 0.05 CH3CN + 0.05 CH3Cl - 2.xlsx',
'4 Comp-mix_ 67% CH3OH + 30% CH3CHO + 2% CH3Cl + 1% CH3CN/0.67 CH3OH + 0.3 CH3CHO + 0.02 CH3Cl + 0.01 CH3CN - v2.xlsx']


exp_path = '../../data/Mixture_exp_data/'
exp_filepath = '4 Comp-mix_ 67% CH3OH + 30% CH3CHO + 2% CH3Cl + 1% CH3CN/0.67 CH3OH + 0.3 CH3CHO + 0.02 CH3Cl + 0.01 CH3CN - v2.xlsx'


def classify_exp(exp_path,exp_filepath,mixture_types):

    df_exp1 = pd.read_excel(exp_path + exp_filepath)


    freq_exp1 = df_exp1[df_exp1.columns[0]].to_numpy()
    abs_exp1 = df_exp1[df_exp1.columns[1]].to_numpy()


    dfy_resampled= signal.resample(abs_exp1, len(m.frequencies))
    dfx_resampled= signal.resample(freq_exp1, len(m.frequencies))
    expanded_abs = np.expand_dims(np.expand_dims(dfy_resampled, axis=-1), axis=0)
    pred_exp_label = np.argmax(model.predict(expanded_abs), axis=-1)[0]

    print('Experiment name: ',exp_filepath.split('/')[0])
    print('predicted index ', pred_exp_label)
    print('predicted label', mixture_types[pred_exp_label])
    
    return pred_exp_label

for experiment in exps:
    classify_exp(exp_path,experiment,mixture_types)
    
from aimos.misc.utils import classifier_internals
from aimos.misc.utils import clf_post_processor
classifier_internals(pred_y, y_val, y_train, 'val-voc-net-style')

y_val_named = mixture_types[y_val] # array consisting of string names of validation mixtures
y_train_named = mixture_types[y_train]
pred_y_named = mixture_types[pred_y] # array consisting of string names of predicted mixtures
y_val_ohe = concentrations_to_one_hot_encode(y_concentrations[val_indices]).astype('int64')

print(y_val_ohe)
print(pred_y[0])
print(mixture_types[1])
print(mixture_names[3])
print(mixture_types)

from oneida_utils import mixture_names_to_one_hot_encode
pred_y_ohe = mixture_names_to_one_hot_encode(pred_y_named, reduced_labels, verbosity = False)

    
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print(multilabel_confusion_matrix(y_val_ohe, pred_y_ohe))



f, axes = plt.subplots(2, 4, figsize=(25, 15), dpi=300)

axes = axes.ravel()
for i in range(8):
    acc_scr = accuracy_score(y_val_ohe[:, i], pred_y_ohe[:, i])
    prc_scr = precision_score(y_val_ohe[:, i], pred_y_ohe[:, i])
    rec_scr = recall_score(y_val_ohe[:, i], pred_y_ohe[:, i])
    f1_scr = f1_score(y_val_ohe[:, i], pred_y_ohe[:, i])
    disp = ConfusionMatrixDisplay(confusion_matrix(y_val_ohe[:, i],
                                                   pred_y_ohe[:, i]),
                                  display_labels=['N', 'P'])

#     disp = ConfusionMatrixDisplay.from_predictions(y_val_ohe[:, i],
#                                                    y_pred_ohe[:, i], cmap = 'Blues',
#                                   display_labels=['N', 'P'])
    
    disp.plot(ax=axes[i], cmap='cividis',)
#     disp.plot(ax=axes[i], values_format='.4g'
    disp.ax_.set_title(f'{reduced_labels[i]} \n acc_scr = {acc_scr:.2f} \n prc_scr = {prc_scr:.2f} \n rec_scr = {rec_scr:.2f} \n f1_scr = {f1_scr:.2f}')
    if i<10:
        disp.ax_.set_xlabel('Predicted label')
    if i%5!=0:
        disp.ax_.set_ylabel('')
    disp.im_.colorbar.remove()
    
plt.subplots_adjust(wspace=1.10, hspace=0.1)
f.colorbar(disp.im_, ax=axes, orientation = 'horizontal')
plt.show()
f.savefig(r'RESULTS/results_figures/' + model_name + '_cm_bin_val.png', bbox_inches='tight')