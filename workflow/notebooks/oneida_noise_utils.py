#!/usr/bin/env python3
# -*- coding: utf-8 -*-



"""
Author: M Arshad Zahangir Chowdhury
Email: arshad.zahangir.bd[at]gmail[dot]com
Definitions of functions to add and process noisy spectra.
"""

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



def add_noise(spectrum, SNR, verbosity=False):
    '''
    Additive white Gaussian Noise.
    args:
    -----
    Spectrum: numpy array, shape (features, 1), input spectrum
    SNR:float, signal-to-noise ratio
    returns:
    --------
    y_volts: noisy spectrum.
    
    '''
    target_snr_db = SNR
    if verbosity == True:
        print(f'SNR = {target_snr_db}')

    x_volts=spectrum
    if verbosity == True:
        print('x_volts.shape',x_volts.shape)
    x_volts=x_volts.reshape(spectrum.shape[0])
    

    x_watts = x_volts ** 2

    x_db = 10 * np.log10(x_watts)

    # Calculate signal power and convert to dB 
    sig_avg_watts = np.mean(x_watts)
    sig_avg_db = 10 * np.log10(sig_avg_watts)
    # Calculate noise according to [2] then convert to watts
    noise_avg_db = sig_avg_db - target_snr_db
    noise_avg_watts = 10 ** (noise_avg_db / 10)
    # Generate an sample of white noise
    mean_noise = 0
    noise_volts = np.random.normal(mean_noise, np.sqrt(noise_avg_watts), len(x_watts))
    if verbosity == True:
        print('noise_volts.shape',noise_volts.shape)
    # Noise up the original signal
    noise_volts = np.amax(x_volts)*noise_volts/(target_snr_db*np.amax(noise_volts))
    if verbosity == True:
        print('noise_volts.shape',noise_volts.shape)
    y_volts = x_volts + noise_volts
    if verbosity == True:
        print('y_volts.shape',y_volts.shape)
        
    return y_volts


def plot_noisy_spectrum(spectrum, noisy_spectrum, frequencies, SNR):
    fig = plt.figure(figsize=(8,2),dpi=150)
    ax_comparison = fig.add_axes([0, 0, 1, 1])

    ax_comparison.plot(frequencies,spectrum,label="Mixture spectrum");

    ax_comparison.plot(frequencies,noisy_spectrum,label= f"Noise-added specturm, SNR = {SNR} ");

    ax_comparison.set_xlim(7.33, 11.0)
    ax_comparison.set_ylabel(r'Absorbance', labelpad=1, fontsize = '12',  fontweight='bold')
    ax_comparison.set_xlabel(r'Frequency, $\nu$ ($cm^{-1}$)', labelpad=1,fontsize = '12',  fontweight='bold')

    plt.legend(loc=2, prop={'size': 6})
