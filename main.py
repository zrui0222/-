#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 21:50:54 2024

@author: liuzhiyan
"""

from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
from data import load_data
from model import Encoder, Decoder, SeriesAdditionalBlock
from train import channel_visualization,seed_everything
import os
import time
from tempbalance import Tempbalance
import config as cf


#Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
gpu_list = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list


# Parameters Setting for Data
Nc = 32  # The number of subcarriers 子载波数量
Nt = 32  # The number of transmit antennas 发射天线数量
N_channel = 2  # Real, Imaginary 实部虚部


# Network Params
SEED = 42
seed_everything(SEED)
batch_siza = 128    # 原始为128
epochs = 1000
print_freq= 100    # 每100个数据输出一次
residual_num = 2
encoded_dim = 32  #compress rate=1/4->dim.=512, compress rate=1/16->dim.=128, compress rate=1/32->dim.=64, compress rate=1/64->dim.=32
                  # dimension of the codeword    #NUM_FEEDBACK_BITS = 128   # 反馈数据比特数
                  
# Data Loading
file_path_train = 'data/DATA_Htrainin.mat'
file_path_test = 'data/DATA_Htestin.mat'
train_loader,test_loader= load_data(file_path_train,file_path_test)


# Model Constrcuting
encoder_ue = Encoder(encoded_dim).to(device)
decoder_bs = Decoder(encoded_dim).to(device)
additional_block = SeriesAdditionalBlock().to(device)
encoder=None
decoder=None
if encoder is not None:
    encoder_ue.load_state_dict(encoder)  #加载与保存权重
if decoder is not None:
    decoder_bs.load_state_dict(decoder)    
criterion = nn.MSELoss()


# 优化器
tb_scheduler_ue = Tempbalance(net=encoder_ue, 
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
tb_param_group_ue, _ = tb_scheduler_ue.build_optimizer_param_group(untuned_lr=0.01, initialize=True)
optimizer_ue = optim.SGD(tb_param_group_ue,momentum=0.9, weight_decay=1e-4)
 
tb_scheduler_bs = Tempbalance(net=decoder_bs, 
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
tb_param_group_bs, _ = tb_scheduler_bs.build_optimizer_param_group(untuned_lr=0.01, initialize=True)
optimizer_bs = optim.SGD(tb_param_group_bs,momentum=0.9, weight_decay=1e-4)
 
tb_scheduler_ad = Tempbalance(net=additional_block, 
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
tb_param_group_ad, _ = tb_scheduler_ad.build_optimizer_param_group(untuned_lr=0.01, initialize=True)
optimizer_ad = optim.SGD(tb_param_group_ad,momentum=0.9, weight_decay=1e-4)

                  
# Model Training and Saving
bestLoss = 1  # 最大损失，小于最大损失才会保存模型
for epoch in range(epochs):
    encoder_ue.eval()()
    decoder_bs.train()
    additional_block.train()
    for i, input in enumerate(train_loader):
        input = input.cuda()
        codeword = encoder_ue(input)
        output = decoder_bs(codeword)
        output_additional = additional_block(output)
        estimated_codeword = encoder_ue(output_additional)

        loss = criterion(estimated_codeword, codeword) + 0.5 * criterion(output_additional, output)
        
        optimizer_ue.zero_grad()
        optimizer_bs.zero_grad()
        optimizer_ad.zero_grad()
                
        loss.backward()                                
        
        optimizer_ue.step()
        optimizer_bs.step()
        optimizer_ad.step()

        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t''Loss {loss:.4f}\t'.format(
                        epoch, i, len(train_loader), loss=loss.item()))
                    
    untuned_lr = cf.cosine_decay(0.01, epoch, epochs)
    tb_scheduler_ue.step(optimizer_ue, untuned_lr)
    tb_scheduler_bs.step(optimizer_bs, untuned_lr)
    tb_scheduler_ad.step(optimizer_ad, untuned_lr)
    
    # Model Evaluating
    encoder_ue.eval()
    decoder_bs.eval()
    additional_block.eval()   
                
                  
    total_loss = 0
    start = time.time()
    with torch.no_grad():
        for i, input in enumerate(test_loader):
            input = input.cuda()
            codeword = encoder_ue(input, test=True)
            output = decoder_bs(codeword, test=True)
            output_additional = additional_block(output, test=True)
            total_loss += criterion(output_additional, input).item()

        end = time.time()
        t = end - start
        average_loss = total_loss / len(list(enumerate(test_loader)))
        if average_loss < bestLoss:   # 平均损失如果小于1才会保存模型
            channel_visualization(input.detach().cpu().numpy()[12][0])
            channel_visualization(output_additional.detach().cpu().numpy()[12][0])
            # Model saving
            torch.save(encoder_ue.state_dict(), './trained_models/encoder_ue_pretrain.pt')
            torch.save(decoder_bs.state_dict(), './trained_models/decoder_bs_pretrain.pt')
            torch.save(additional_block.state_dict(), './trained_models/addtional_block_pretrain.pt')
            print("Model saved")
            bestLoss = average_loss   # 更新最大损失，使损失小于该值是才保存模型
            print('NMSE %.6ftime %.3f' % (average_loss, t))                 
                  
                  
                  
                  
                  
                  
                  
                  
                  
                  
                  
                  
                  
                  
                  
                  
                  
                  
                  
                  
                  