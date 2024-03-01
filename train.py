#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 17:16:53 2024

@author: liuzhiyan
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 16:56:47 2024

@author: liuzhiyan
"""

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from data import load_data
from model import Encoder, Decoder, SeriesAdditionalBlock


import random
import os
import time
import sys
import argparse

import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from pathlib import Path
from os.path import join


from tempbalance import Tempbalance
from sgdsnr import SGDSNR
from adamp import SGDP, AdamP
import config as cf
import torch_optimizer
from lars_optim import LARS, LAMB
from utils import train, test, getNetwork, save_args_to_file

from scipy import io as scio




gpu_list = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list


start_epoch=1


Nc = 32  # The number of subcarriers 子载波数量
Nt = 32  # The number of transmit antennas 发射天线数量
N_channel = 2  # Real, Imaginary 实部虚部

# encoded_dim = 32  # dimension of the codeword
# revise 20230525: erase useless constant variable 'encoded_dim' declared in this file.
img_total = Nc*Nt*N_channel
# network params
batch_siza = 128    # 原始为128
epochs = 1000
print_freq= 100    # 每100个数据输出一次
residual_num = 2
encoded_dim = 32  #compress rate=1/4->dim.=512, compress rate=1/16->dim.=128, compress rate=1/32->dim.=64, compress rate=1/64->dim.=32
# encoded_dim = 32  # dimension of the codeword
# revise 20230525: erase useless constant variable 'encoded_dim' declared in this file.
#NUM_FEEDBACK_BITS = 128   # 反馈数据比特数






#损失函数
def NMSE(x, x_hat):
    x_test_real = torch.reshape(x[:, 0, :, :], (len(x), -1))
    x_test_imag = torch.reshape(x[:, 1, :, :], (len(x), -1))
    x_hat_real = torch.reshape(x_hat[:, 0, :, :], (len(x_hat), -1))
    x_hat_imag = torch.reshape(x_hat[:, 1, :, :], (len(x_hat), -1))
    power = torch.sum(x_test_real ** 2 + x_test_imag ** 2, axis=1)
    mse = torch.sum((x_test_real - x_hat_real) ** 2 + (x_test_imag - x_hat_imag) ** 2, axis=1)
    nmse = torch.mean(mse / power)
    # print(power)
    # print(mse)
    return nmse

#设置初始随机值
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

#通道可视化
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


#训练模块
class train(nn.Module):
    def __init__(self,
                 file_path_train,
                 file_path_test,
                 epochs,
                 encoded_dim,
                 print_freq,
                 device,
                 encoder=None,
                 decoder=None,
                 ):
        super().__init__()
        self.epochs = epochs
        self.encoded_dim = encoded_dim
        self.print_freq = print_freq
        

        #### 1.load data ####        
        #数据加载        
        self.train_loader,self.test_loader= load_data(file_path_train,file_path_test)
       
        self.encoder_ue = Encoder(encoded_dim).to(device)
        self.decoder_bs = Decoder(encoded_dim).to(device)
        self.additional_block = SeriesAdditionalBlock().to(device)
        if encoder is not None:
            self.encoder_ue.load_state_dict(encoder)  #加载与保存权重
        if decoder is not None:
            self.decoder_bs.load_state_dict(decoder)

        self.criterion = nn.MSELoss()
        
        #优化器选择
        #ue优化器
        self.tb_scheduler_ue = Tempbalance(net=self.encoder_ue, 
                        pl_fitting='median',
                        xmin_pos=2, 
                        filter_zeros=False,
                        remove_first_layer=True,
                        remove_last_layer=True,
                        esd_metric_for_tb='alpha',
                        assign_func='tb_linear_map',
                        lr_min_ratio=0.5,
                        lr_max_ratio=1.5,
                        batchnorm=True,
                        batchnorm_type='name'
                        )
        self.tb_param_group_ue, _ = \
            self.tb_scheduler_ue.build_optimizer_param_group(untuned_lr=0.01, initialize=True)
        self.optimizer_ue = optim.SGD(self.tb_param_group_ue,momentum=0.9, weight_decay=1e-4)
        
        #bs优化器
        self.tb_scheduler_bs = Tempbalance(net=self.decoder_bs, 
                        pl_fitting='median',
                        xmin_pos=2, 
                        filter_zeros=False,
                        remove_first_layer=True,
                        remove_last_layer=True,
                        esd_metric_for_tb='alpha',
                        assign_func='tb_linear_map',
                        lr_min_ratio=0.5,
                        lr_max_ratio=1.5,
                        batchnorm=True,
                        batchnorm_type='name'
                        )
        self.tb_param_group_bs, _ = \
            self.tb_scheduler_bs.build_optimizer_param_group(untuned_lr=0.01, initialize=True)
        self.optimizer_bs = optim.SGD(self.tb_param_group_bs,momentum=0.9, weight_decay=1e-4)
        
        #ad优化器
        self.tb_scheduler_ad = Tempbalance(net=self.additional_block, 
                        pl_fitting='median',
                        xmin_pos=2, 
                        filter_zeros=False,
                        remove_first_layer=True,
                        remove_last_layer=True,
                        esd_metric_for_tb='alpha',
                        assign_func='tb_linear_map',
                        lr_min_ratio=0.5,
                        lr_max_ratio=1.5,
                        batchnorm=True,
                        batchnorm_type='name'
                        )
        self.tb_param_group_ad, _ = \
            self.tb_scheduler_ad.build_optimizer_param_group(untuned_lr=0.01, initialize=True)
        self.optimizer_ad = optim.SGD(self.tb_param_group_ad,momentum=0.9, weight_decay=1e-4)


        SEED = 42
        seed_everything(SEED)

    def train_epoch(self):

        self.encoder_ue.train()
        self.decoder_bs.train()
        
        #### 2. train_epoch ####
        for epoch in range(self.epochs):
            

#             __________                           __________
#             |         \                         /         |
#             |          |                       |          |
# input ----> |encoder_ue| ----> codeword ---->  |decoder_bs| ----> output ----------
# (2,Nc,Nt)   |          |     (encoded_dim)     |          |      (2,Nc,Nt)        |
#   |         |_________/                         \_________|                       v
#   |                                                                              MSE
#   |                                                                               ^
#   |_______________________________________________________________________________|

            
            for i, input in enumerate(self.train_loader):
                input = input.cuda()
                codeword = self.encoder_ue(input)
                output = self.decoder_bs(codeword)

                loss = self.criterion(output, input)
                
                self.optimizer_ue.zero_grad()
                self.optimizer_bs.zero_grad()
                
                loss.backward()
                
                self.optimizer_ue.step()
                self.optimizer_bs.step()
                

                if i % self.print_freq == 0:
                    print('Epoch: [{0}][{1}/{2}]\t'
                          'Loss {loss:.4f}\t'.format(
                        epoch, i, len(self.train_loader), loss=loss.item()))
                    
            self.untuned_lr = cf.cosine_decay(0.01, epoch, self.epochs)
            self.tb_scheduler_ue.step(self.optimizer_ue, self.untuned_lr)
            self.tb_scheduler_bs.step(self.optimizer_bs, self.untuned_lr)

            #### 3. validate ####
        self.encoder_ue.eval()
        self.decoder_bs.eval()

        total_loss = 0
        start = time.time()
        with torch.no_grad():
            for i, input in enumerate(self.test_loader):
                input = input.cuda()
                codeword = self.encoder_ue(input, test=True)
                output = self.decoder_bs(codeword, test=True)
                #total_loss += self.criterion(output_additional, input).item()
                total_loss += NMSE(output, input) 

            end = time.time()
            t = end - start
            average_loss = total_loss / len(list(enumerate(self.test_loader)))
            print('NMSE %.6ftime %.3f' % (average_loss, t))

        channel_visualization(input.detach().cpu().numpy()[12][0])
        channel_visualization(output.detach().cpu().numpy()[12][0])

        torch.save(self.encoder_ue.state_dict(), './trained_models/encoder_ue_pretrain.pt')
        torch.save(self.decoder_bs.state_dict(), './trained_models/decoder_bs_pretrain.pt')

        return self.encoder_ue.state_dict(), self.decoder_bs.state_dict()

    def train_online_epoch(self):

        self.encoder_ue.eval()
        self.decoder_bs.train()
        self.additional_block.train()

        #### 2. train_epoch ####
        for epoch in range(self.epochs):

#           __________                       __________                ________                    __________
#           |         \                     /         |                |      |                    |         \
#           |          |                   |          |                |addi  |       output_      |          |
# input --> |encoder_ue| --> codeword -->  |decoder_bs| --> output  -->|tional| --> additional --> |encoder_ue| --> estimated_codeword
# (2,Nc,Nt) |          |   (encoded_dim)   |          |    (2,Nc,Nt)   |block |       (2,Nc,Nt)    |          |          (encoded_dim)
#           |_________/         |           \_________|                |______|                    |_________/               |
#            cannot be          |             can be                    can be                      cannot be                V
#             trained           |            trained                   trained                       trained                MSE
#                               |                                                                                            ^
#                               |____________________________________________________________________________________________|

            for i, input in enumerate(self.train_loader):
                input = input.cuda()
                codeword = self.encoder_ue(input)
                output = self.decoder_bs(codeword)
                output_additional = self.additional_block(output)
                estimated_codeword = self.encoder_ue(output_additional)

                loss = self.criterion(estimated_codeword, codeword) + 0.5 * self.criterion(output_additional, output)
                
                self.optimizer_bs.zero_grad()
                self.optimizer_ad.zero_grad()
                
                loss.backward()                                
                
                self.optimizer_bs.step()
                self.optimizer_ad.step()
                

                if i % self.print_freq == 0:
                    print('Epoch: [{0}][{1}/{2}]\t'
                          'Loss {loss:.4f}\t'.format(
                        epoch, i, len(self.train_loader), loss=loss.item()))
                    
            self.untuned_lr = cf.cosine_decay(0.01, epoch, self.epochs)
            self.tb_scheduler_bs.step(self.optimizer_bs, self.untuned_lr)
            self.tb_scheduler_ad.step(self.optimizer_ad, self.untuned_lr)

            #### 3. validate ####
        self.encoder_ue.eval()
        self.decoder_bs.eval()
        self.additional_block.eval()

        total_loss = 0
        start = time.time()
        with torch.no_grad():
            for i, input in enumerate(self.test_loader):
                input = input.cuda()
                codeword = self.encoder_ue(input, test=True)
                output = self.decoder_bs(codeword, test=True)
                output_additional = self.additional_block(output, test=True)
                #total_loss += self.criterion(output_additional, input).item()
                total_loss += NMSE(output_additional, input) 
                
            end = time.time()
            t = end - start
            average_loss = total_loss / len(list(enumerate(self.test_loader)))
            print('NMSE %.6ftime %.3f' % (average_loss, t))

        channel_visualization(input.detach().cpu().numpy()[12][0])
        channel_visualization(output_additional.detach().cpu().numpy()[12][0])

        torch.save(self.encoder_ue.state_dict(), './trained_models/encoder_ue_pretrain.pt')
        torch.save(self.decoder_bs.state_dict(), './trained_models/decoder_bs_pretrain.pt')

        return self.encoder_ue.state_dict(), self.decoder_bs.state_dict()