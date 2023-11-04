# Copyright (C) 2021 #
# @Time    : 2023/6/28 11:03
# @File    : data_loader.py
# @Software: PyCharm


from torch.utils.data import Dataset
import pandas as pd
import numpy as np


class Dataset_Custom(Dataset):
    def __init__(self,data,label,mode):

        self.data = data
        self.label = label
        self.mode = mode

    def __len__(self):
        if self.mode == '2D':
            return self.data.shape[0]
        elif self.mode =='3D':
            return self. data.shape[1]

    def __getitem__(self, item):
        if self.mode == '2D':
            return self.data[item, :], self.label[item, :]
        elif self.mode == '3D':
            return self.data[:,item, :], self.label[item,:]


class Data_Process():

    def __init__(self,data_name,data_path,task=None):
        self.name = data_name # data name
        self.data_path = data_path
        self.task = task
        self.process_data()



    def process_data(self):

        
        if self.name == 'deb':
            if self.task == 'imputation':
                data = np.loadtxt(fname=self.data_path)
            else:
                data = pd.read_table('C:\Study\Code\自由探索\Debutanizer_Data.txt', sep='\s+')
                data = data.values
          
            x = data[:, :7]
            y = data[:, 7]
            x_temp = x
            y_temp = y



            x_train = x_temp[:1915, :]
            x_test = x_temp[1915:, :]

            y_train = y_temp[:1915]
            y_test = y_temp[1915:]

        elif self.name == 'sru':


            data = np.array(data)
            TRAIN_SIZE = 8000
            x = self.data[:, 0:len(self.data[0]) -1]
            y = self.data[:, -1]

            x_train = x[0:TRAIN_SIZE, :]
            y_train = y[0:TRAIN_SIZE]
            x_test = x[TRAIN_SIZE:, :]
            y_test = y[TRAIN_SIZE:]


        else:
            raise NotImplementedError(
                'Dataset doesnt exits, Please check if dataset name is one of pta, deb, tff, te, ccpp and sru.')

        return x_train, y_train, x_test, y_test

