# Copyright (C) 2021 #
# @Time    : 2023/6/26 10:37
# @File    : exp_basic.py
# @Software: PyCharm


import os
import torch
import numpy as np

class Exp_basic(object):
    def __init__(self,args):

        self.args = args
        self.device = self._acquire_deivce()
        self.X_train, self.y_train, self.X_test, self.y_test  = self._get_data()
        self.model = self._build_model().to(self.device)
        _, self.C_in = self.X_train[0].shape

    def _build_model(self):

        raise  NotImplementedError

        return None

    def _acquire_deivce(self):

        if self.args.use_cuda:

            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu)
            device = torch.device('cuda:{}'.format(self.args.gpu))

        else:
            device = torch.device('cpu')
            print('====use gpu=====')

        return device


    def _get_data(self):
        pass



    def train(self):
        pass

    def test(self):
        pass

    def anomaly_detect(self):
        pass
