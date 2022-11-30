#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Author: M Arshad Zahangir Chowdhury
Email: arshad.zahangir.bd[at]gmail[dot]com
Some Filters for processing data.
"""


import numpy as np
import matplotlib.pyplot as plt


def FFT_filter(dt, f, PSD_cutoff = 100, make_plots= True):
    '''
    FFT filter. Returs filtered signal and figure
    Args:
    -----
    dt, int timestep, samples per second.
    f, float, noisy signal.
    PSD_cutoff, float, Default is 100. Cutoff power spectral density. Peaks above this will be retained in the filtered signal.
    make_plots, boolean, True for making the plots.
    
    
    '''
    
    t = np.arange(0, len(f), dt) # Fourier time parameter
    
    n = len(t)
    fhat = np.fft.fft(f,n) # compute fft
    PSD = fhat*np.conj(fhat)/n # Power spectrum, power per frequency

    freq = (1/(dt*n))*np.arange(n) # frequency values along x-axis
    L = np.arange(1, np.floor(n/2), dtype='int') #plot fist half of frequencies only
    
    ### Use the Power Spectral Density to filter out noise
    indices = PSD > PSD_cutoff # Find all freqs with larger
    PSDclean = PSD * indices # zero out all others
    fhat = indices * fhat # zero out smaller fourier coefficients in Y

    ### Inverse Fourier Transform
    ffilt= np.fft.ifft(fhat) #inverse FFT for filtered time signal
    
    if make_plots:
        fig, axs = plt.subplots(3,1, figsize=(16,12), dpi=150)
        plt.sca(axs[0])
        plt.plot(t, f, color = 'red', label='Noisy')
        plt.xlim(t[0], t[-1])
        plt.xlabel('Sampling points')
        plt.ylabel('Absorbance')
        plt.legend()
        plt.tight_layout()

#         plt.sca(axs[1])
#         plt.plot(freq[L], PSD[L], color = 'red', label='Noisy')
#         plt.axhline(y=PSD_cutoff, color='green', lw=3, linestyle='-')
#         plt.xlim(freq[L[0]], freq[L[-1]])
#         plt.xlabel('Hz')
#         plt.ylabel('Power in each frequency \n (Power Spectral Density)')
#         plt.legend()
#         plt.tight_layout()
        
        plt.sca(axs[1])
        plt.plot(freq[L], PSD[L], color = 'red', label='Noisy')
        plt.axhline(y=PSD_cutoff, color='green', lw=3, linestyle='-')
        plt.xlim(freq[L[0]], 0.001)
        plt.xlabel('Hz')
        plt.ylabel('Power in each frequency \n (Power Spectral Density)')
        plt.legend()
        plt.tight_layout()

        plt.sca(axs[2])
        plt.plot(t, f, color = 'black', label='Original')
        plt.plot(t, ffilt, color = 'blue', label='Filtered')
        plt.xlabel('Sampling points')
        plt.ylabel('Absorbance')
        plt.xlim(t[0], t[-1])
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        
        
    return ffilt, fig