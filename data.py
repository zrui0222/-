#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 29 10:21:01 2024

@author: liuzhiyan
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import io as scio

batch_size = 128
Nc = 32  # The number of subcarriers
Nt = 32  # The number of transmit antennas
N_channel = 2  # Real, Imaginary
encoded_dim = 32  # dimension of the codeword



class dataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, i):
        return self.data[i]

def channel_visualization(image):
    fig, ax = plt.subplots()
    plot = ax.imshow(image, cmap=plt.cm.gray, interpolation='nearest', origin='upper')
    plt.colorbar(plot)
    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_position(('outward', 10))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.show()

def load_data(file_path_train,file_path_test):
    train_data = scio.loadmat(file_path_train) ['HT'] # array 'data/DATA_Htrainin.mat'
    test_data = scio.loadmat(file_path_test)['HT']  #'data/DATA_Htestin.mat'
    
    x_train  = train_data .astype('float32')
    x_test = test_data.astype('float32')
    
    x_train = np.reshape(x_train , (len(x_train ), 2, 32, 32))  
    x_test = np.reshape(x_test , (len(x_test ), 2, 32, 32)) 
    #channel_visualization(x_train[0][0])

    train_dataset = dataset(x_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                               shuffle=True, num_workers=0,
                                               pin_memory=True, drop_last=True)

    test_dataset = dataset(x_test)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                              shuffle=False, num_workers=0,
                                              pin_memory=True, drop_last=True)
    return train_loader, test_loader






