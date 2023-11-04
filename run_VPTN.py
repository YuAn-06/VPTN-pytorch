"""
Copyright (C) 2023
@ Name: main_VPTN.py
@ Time: 2023/11/2 16:27
@ Author: YuAn_L
@ Software: PyCharm
"""



import torch
import argparse
import random
import numpy as np

fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

from exp.exp_VPTN import Exp_VPTN

parser = argparse.ArgumentParser(description='test')
parser.add_argument('--model', type=str, default='VPTN', help='model name')
parser.add_argument('--data_name', type=str, default='deb', help='dataset name')
parser.add_argument('--root_path', type=str, default='./data/', help='root path of data')
parser.add_argument('--data_path', type=str, help='path of data')

parser.add_argument('--if_ts', type=int, default=0, help='if time series dataset')
parser.add_argument('--if_re', type=int, default=0, help='if reconstruction')
parser.add_argument('--task', type=str, default='forecast', help='forecast, imputation')


parser.add_argument('--batch_size', type=int, default=40, help='batch size of training input data')
parser.add_argument('--learning_rate', type=int, default=0.01, help='batch size of training input data')
parser.add_argument('--epoch', type=int, default=400, help='training epoch')
parser.add_argument('--scaler',type=str,default='Standard',help='type of Preprocessing data: Minax, Standard.')

parser.add_argument('--use_cuda', type=bool, default=False, help='use gpu switch')
parser.add_argument('--device', type=str, default='cpu', help='cpu or gpu')
parser.add_argument('--seed', type=int, default=1024, help='random seed')

args = parser.parse_args()

if args.use_cuda:
    args.device = 'gpu'


## set dataset path

# args.data_path = 'xxx'

print(args)

setting = '{}_{}_bt{}_lr{}_sl{}_ep{}'.format(
                args.data_name,
                args.model,

                args.batch_size,
                args.learning_rate,
                args.scaler,
                args.epoch,
                )

exp = Exp_VPTN(args=args)
exp.train()
exp.test(setting)
