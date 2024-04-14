from __future__ import print_function

# Set logging level to WARNING to disable logger print
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import config as cf
import numpy as np
import random
import pandas as pd
import os
import sys
import time
import argparse
from pathlib import Path
from os.path import join
from dataset.cost2100 import Cost2100DataLoader
from train_utils_tbr import train, test, getNetwork, net_esd_estimator, get_layer_temps
import json
from sgdsnr import SGDSNR

parser = argparse.ArgumentParser(description='CsiNet PyTorch Training')
parser.add_argument('--lr', type=float, default=0.002, help='learning_rate')
parser.add_argument('--net-type', type=str, default='CsiNet', help='model')
parser.add_argument('--num-epochs', type=int, default=500, help='number of epochs')
parser.add_argument('--warmup-epochs', type=int, default=0)
parser.add_argument('--lr-sche', type=str, default='warmup_cosine', choices=['step', 'cosine', 'warmup_cosine'])
parser.add_argument('--weight-decay', type=float, default=1e-4)  # 5e-4
parser.add_argument('--print-tofile', type=str, default='False')
parser.add_argument('--ckpt-path', type=str, default='CLNet Result')
parser.add_argument('--cr', metavar='N', type=int, default=4,
                    help='compression ratio')
parser.add_argument('--batch-size', type=int, default=200)  # 5e-4
parser.add_argument('--data-dir', type=str, default='data', help='the path of dataset.')
parser.add_argument('--scenario', type=str, default='in', help="the channel scenario")
parser.add_argument('-j', '--workers', type=int, default=0, metavar='N', help='number of data loading workers')
parser.add_argument('--optim-type', type=str, default='Adam', help='type of optimizer')
parser.add_argument('--resume', type=str, default='', help='resume from checkpoint')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--ww-interval', type=int, default=1)
parser.add_argument('--epochs-to-save', type=int, nargs='+', default=[i*10 for i in range(51)])
parser.add_argument('--fix-fingers', type=str, default=None, help="xmin_peak")
parser.add_argument('--pl-package', type=str, default='powerlaw')

# temperature balance related
parser.add_argument('--remove-last-layer', type=str, default='True', help='if remove the last layer')
parser.add_argument('--remove-first-layer', type=str, default='True', help='if remove the last layer')
parser.add_argument('--metric', type=str, default='alpha', help='ww metric')
parser.add_argument('--temp-balance-lr', type=str, default='tb_linear_map', help='use tempbalance for learning rate')
parser.add_argument('--batchnorm', type=str, default='False')
parser.add_argument('--lr-min-ratio', type=float, default=0.5)
parser.add_argument('--lr-slope', type=float, default=1.0)
parser.add_argument('--xmin-pos', type=float, default=2, help='xmin_index = size of eigs // xmin_pos')
parser.add_argument('--lr-min-ratio-stage2', type=float, default=1)
# spectral regularization related
parser.add_argument('--sg', type=float, default=0.01, help='spectrum regularization')
parser.add_argument('--stage-epoch', type=int, default=0, help='stage_epoch')
parser.add_argument('--filter-zeros', type=str, default='False')

args = parser.parse_args()

print(args)
print("--------------------> TB or TB + SNR <----------------------")


def set_seed(seed=42):
    print(f"=====> Set the random seed as {seed}")
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


def save_args_to_file(args, output_file_path):
    with open(output_file_path, "w") as output_file:
        json.dump(vars(args), output_file, indent=4)


# Hyper Parameter settings
use_cuda = torch.cuda.is_available()
if use_cuda:
    pin_memory = True
else:
    pin_memory = False
best_loss = 0
start_epoch = cf.start_epoch
set_seed(args.seed)

# Data Upload
train_loader, val_loader, test_loader = Cost2100DataLoader(
        root=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=pin_memory,
        scenario=args.scenario)()


Path(args.ckpt_path).mkdir(parents=True, exist_ok=True)

if args.print_tofile == 'True':
    # Open files for stdout and stderr redirection
    stdout_file = open(os.path.join(args.ckpt_path, 'stdout.log'), 'w')
    stderr_file = open(os.path.join(args.ckpt_path, 'stderr.log'), 'w')
    # Redirect stdout and stderr to the files
    sys.stdout = stdout_file
    sys.stderr = stderr_file

# Save the arguments to a file
save_args_to_file(args, join(args.ckpt_path, 'args.json'))

# Model
print('\n[Phase 2] : Model setup')
if args.resume:
    # Load checkpoint
    print('| Resuming from checkpoint...')
    net, file_name = getNetwork(args)
    checkpoint = torch.load(args.resume, map_location='cpu')
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    net.load_state_dict(checkpoint['net'])
    best_loss = checkpoint['test_loss']
    start_epoch = checkpoint['epoch']
    print(f"Loaded Epoch: {start_epoch} \n Test loss: {best_loss:.3f} Train Acc: {checkpoint['train_loss']:.3f}")
else:
    print('| Building net type [' + args.net_type + ']...')
    net, file_name = getNetwork(args)
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    best_acc = 0

if use_cuda:
    net.cuda()
    cudnn.benchmark = True

criterion = nn.MSELoss().cuda()

print('\n[Phase 3] : Training model')
print('| Training Epochs = ' + str(args.num_epochs))
print('| Initial Learning Rate = ' + str(args.lr))
print('| Optimizer = ' + str(args.optim_type))

test_loss, test_rho, test_nmse = test(epoch=0, net=net, data_loader=test_loader, criterion=criterion)
print(f"Reevaluated: Test Loss: {test_loss:.3f}, Test Rho: {test_rho:.3f}, Test nmse: {test_nmse:.3f}")

#######################ESD analysis###############################
##################################################################
print("####################Start ESD analysis###################")
Path(os.path.join(args.ckpt_path, 'stats')).mkdir(parents=True, exist_ok=True)
esd_start_time = time.time()
metrics = net_esd_estimator(net,
                            EVALS_THRESH=0.00001,
                            bins=100,
                            fix_fingers=args.fix_fingers,
                            xmin_pos=args.xmin_pos,
                            filter_zeros=args.filter_zeros == 'True')

estimated_time = time.time() - esd_start_time
print(f"-----> ESD estimation time: {estimated_time:.3f}")
# summary and submit to wandb


# save metrics to disk and ESD
layer_stats = pd.DataFrame({key: metrics[key] for key in metrics if key != 'eigs'})
layer_stats_origin = layer_stats.copy()
layer_stats_origin.to_csv(os.path.join(args.ckpt_path, 'stats', f"origin_layer_stats_epoch_{0}.csv"))
np.save(os.path.join(args.ckpt_path, 'stats', 'esd_epoch_{0}.npy'), metrics)
##################################################################

######################  TBR scheduling ##########################
##################################################################
if args.temp_balance_lr != 'None':
    print("################## Enable temp balance ##############")

    if args.remove_first_layer == 'True':
        print("remove first layer of alpha<---------------------")
        layer_stats = layer_stats.drop(labels=0, axis=0)
        # index must be reset otherwise may delete the wrong row
        layer_stats.index = list(range(len(layer_stats[args.metric])))
    if args.remove_last_layer == 'True':
        print("remove last layer of alpha<---------------------")
        layer_stats = layer_stats.drop(labels=len(layer_stats) - 1, axis=0)
        # index must be reset otherwise may delete the wrong row
        layer_stats.index = list(range(len(layer_stats[args.metric])))

    metric_scores = np.array(layer_stats[args.metric])
    # args, temp_balance, n_alphas, epoch_val
    scheduled_lr = get_layer_temps(args, temp_balance=args.temp_balance_lr, n_alphas=metric_scores, epoch_val=args.lr)
    layer_stats['scheduled_lr'] = scheduled_lr

    # these params should be tuned
    layer_name_to_tune = list(layer_stats['longname'])
    all_params = []
    params_to_tune_ids = []

    # these params should be tuned
    for name, module in net.named_modules():
        # these are the conv layers analyzed by the weightwatcher
        if name in layer_name_to_tune:
            params_to_tune_ids += list(map(id, module.parameters()))
            scheduled_lr = layer_stats[layer_stats['longname'] == name]['scheduled_lr'].item()
            all_params.append({'params': module.parameters(), 'lr': scheduled_lr})
        # decide should we tune the batch norm accordingly,  is this layer batchnorm and does its corresponding conv in layer_name_to_tune
        elif args.batchnorm == 'True' \
                and isinstance(module, nn.BatchNorm2d) \
                and name.replace('bn', 'conv') in layer_name_to_tune:
            params_to_tune_ids += list(map(id, module.parameters()))
            scheduled_lr = layer_stats[layer_stats['longname'] == name.replace('bn', 'conv')]['scheduled_lr'].item()
            all_params.append({'params': module.parameters(), 'lr': scheduled_lr})
        # another way is to add a else here and append params with args.lr

    # those params are untuned
    untuned_params = filter(lambda p: id(p) not in params_to_tune_ids, net.parameters())
    all_params.append({'params': untuned_params, 'lr': args.lr})
    # create optimizer
    optimizer = optim.Adam(all_params)

else:
    print("-------------> Disable temp balance")
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
##################################################################


# save scheduled learning rate
layer_stats.to_csv(os.path.join(args.ckpt_path, 'stats', f"layer_stats_with_lr_epoch_{0}.csv"))

#########################################################################

if args.lr_sche == 'step':
    lr_schedule = cf.stepwise_decay
elif args.lr_sche == 'cosine':
    lr_schedule = cf.cosine_decay
elif args.lr_sche == 'warmup_cosine':
    lr_schedule = cf.warmup_cosine_decay
else:
    raise NotImplementedError

elapsed_time = 0
training_stats = \
    {'test_loss': [test_loss],
     'test_rho': [test_rho],
     'test_nmse': [test_nmse],
     'train_loss': [],

     'schedule_next_lr': []
     }

untuned_lr = args.lr
is_current_best = False
for epoch in range(start_epoch, start_epoch + args.num_epochs):
    epoch_start_time = time.time()
    # consider use another (maybe bigger) minimum learning rate in tbr
    if args.stage_epoch > 0 and epoch >= args.stage_epoch:
        print("------> Enter the second stage!!!!!!!!!!")
        args.lr_min_ratio = args.lr_min_ratio_stage2
    else:
        pass

    # this is current LR


    # train and test
    train_loss = train(args, epoch, net, args.num_epochs, train_loader, criterion, optimizer,lr_schedule)
    print("\n| Train Epoch #%d\t\t\tLoss: %.7f" % (epoch, train_loss))
    test_loss, test_rho, test_nmse = test(epoch, net, test_loader, criterion)
    print("\n| Validation Epoch #%d\t\t\tLoss: %.7f rho: %.4f  nmse: %.4f" % (epoch, test_loss, test_rho, test_nmse))

    # save in interval
    if epoch in args.epochs_to_save:
        state = {
            'net': net.state_dict(),
            'test_loss': test_loss,
            'test_rho': test_rho,
            'test_nmse': test_nmse,
            'train_loss': train_loss,
            'epoch': epoch
        }
        torch.save(state, join(args.ckpt_path, f'epoch_{epoch}.ckpt'))
    # save best
    if test_loss < best_loss:
        print('| Saving Best model')
        state = {
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'test_loss': test_loss,
            'best_loss': best_loss,
            'test_rho': test_rho,
            'test_nmse': test_nmse,
            'train_loss': train_loss,
            'epoch': epoch
        }
        best_loss = test_loss
        is_current_best = True
        torch.save(state, join(args.ckpt_path, f'epoch_best.ckpt'))
    else:
        is_current_best = False

    #######################ESD analysis###############################
    ##################################################################
    if epoch == 1 or epoch % 1 == 0:
        print("################ Start ESD analysis#############")
        esd_start_time = time.time()
        metrics = net_esd_estimator(net,
                                    EVALS_THRESH=0.00001,
                                    bins=100,
                                    fix_fingers=args.fix_fingers,
                                    xmin_pos=args.xmin_pos,
                                    filter_zeros=args.filter_zeros == 'True')

        

        layer_stats = pd.DataFrame({key: metrics[key] for key in metrics if key != 'eigs'})
        # save metrics to disk and ESD
        layer_stats_origin = layer_stats.copy()
        layer_stats_origin.to_csv(os.path.join(args.ckpt_path, 'stats', f"origin_layer_stats_epoch_{epoch}.csv"))
        np.save(os.path.join(args.ckpt_path, 'stats', f'esd_epoch_{epoch}.npy'), metrics)
        if is_current_best:
            np.save(os.path.join(args.ckpt_path, f'esd_best.npy'), metrics)

        esd_estimated_time = time.time() - esd_start_time
        print(f"-----> ESD estimation time: {esd_estimated_time:.3f}")

    training_stats['test_loss'].append(test_loss)
    training_stats['test_rho'].append(test_rho)
    training_stats['test_nmse'].append(test_nmse)
    training_stats['train_loss'].append(train_loss)

    training_stats['schedule_next_lr'].append(untuned_lr)

    np.save(join(args.ckpt_path, "training_stats.npy"), training_stats)
    epoch_time = time.time() - epoch_start_time
    elapsed_time += epoch_time
    print('| Elapsed time : %d:%02d:%02d' % (cf.get_hms(elapsed_time)))
    print('--------------------> <--------------------')

if args.print_tofile == 'True':
    # Close the files to flush the output
    stdout_file.close()
    stderr_file.close()