"""
Copyright (C) 2023
@ Name: exp_VPTN.py
@ Time: 2023/11/2 11:55
@ Author: YuAn_L
@ Eamil: yuan_l1106@163.com
@ Software: PyCharm
"""

from exp.exp_basic import Exp_basic
from models import VPTN
from models.VPTN import VAEmodel
from torch.utils.data import Dataset, DataLoader
from data.data_loader import Dataset_Custom, Data_Process
import pandas as pd
import torch
import numpy as np
from torch import optim, nn
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from utils import metrics
from matplotlib import pyplot as plt
from itertools import chain
from sklearn.metrics import r2_score
from utils.tools import visual


class Exp_VPTN(Exp_basic):
    def __init__(self, args):
        super(Exp_VPTN, self).__init__(args)
        self.args = args

    def _build_model(self):

        C_in_list = []

        for i in range(len(self.X_train)):
            self.X_train[i] = np.array(self.X_train[i])
            print('chunk{}:'.format(i+1),self.X_train[i].shape)
            C_in_list.append(self.X_train[i].shape[1])


        self.subVAE1 = VAEmodel(C_in_list[0], C_in_list[0], self.args.device)
        print(self.subVAE1)
        self.subVAE2 = VAEmodel(C_in_list[1], C_in_list[1], self.args.device).to(self.args.device)
        print(self.subVAE2)
        self.subVAE3 = VAEmodel(C_in_list[2], C_in_list[2] , self.args.device).to(self.args.device)
        self.subVAE4 = VAEmodel(C_in_list[2], C_in_list[2] + 1, self.args.device).to(self.args.device)
        print(self.subVAE4)
        return self.subVAE1

    def _get_data(self):
        """

        :return:
        """
        D = Data_Process(self.args.data_name, self.args.data_path, self.args.task)
        X_train, y_train, X_test, y_test = D.process_data()

        self.y_scaler = self._select_scaler()
        self.X_scaler = self._select_scaler()

        X_train = self.X_scaler.fit_transform(X_train)
        X_test = self.X_scaler.transform(X_test)
        y_train = self.y_scaler.fit_transform(y_train.reshape(-1, 1))
        y_test = self.y_scaler.transform(y_test.reshape(-1, 1))

        if self.args.data_name == 'sru':
            self.interval = {
                0: 1,
                1: 2,
                2: 2,
                3: 2,
                4: 1}

        elif self.args.data_name == 'deb':
            self.interval = {
                0: 1,
                1: 1,
                2: 2,
                3: 2,
                4: 1,
                5: 1,
                6: 1,
                7: 4
            }
        value = list(set(self.interval.values()))
        index = []
        x_train_chunks = []
        y_train_chunks = []
        x_test_chunks = []
        y_test_chunks = []

        # chunk1_variable = [k for k, v in self.interval.items() if v == value[0]]
        # chunk1 = [X_train[i,chunk1_variable] for i in range(X_train.shape[0]) if all((i + 1) % j != 0 for j in value[1:])]
        # x_train_chunks.append(chunk1)
        # index.append(chunk1_variable)
        for i in range(len(value)-1):
            # set 出来是有序的
            chunk_variable = [k for k, v in self.interval.items() if v == value[i]]
            index.append(chunk_variable)

            chunk = [X_train[k, list(chain(*index))] for k in range(X_train.shape[0]) if
                          all((k + 1) % j != 0 for j in value[i+1:])]


            # else:
            #     chunk = X_train[::value[i], list(chain(*index))]



            x_train_chunks.append(chunk)

            # x_test_chunks.append(X_test[::value[i], list(chain(*index))])


        # 最后一个chunks 需要单独设定一下，训练的时候要拼接一下
        x_train_chunks.append(X_train[::value[-1], :])
        y_train = y_train[::value[-1], :]
        x_test_chunks.append(X_test[::value[-1], :])
        y_test = y_test[::value[-1], :]

        return x_train_chunks, y_train, x_test_chunks, y_test

    def _select_optimizer(self):

        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self, X, X_hat, mean, var):

        re = nn.MSELoss(reduction='mean')(X, X_hat)
        KLD = -0.5 * torch.mean(1 - var.pow(2) - mean.pow(2) + torch.log(1e-8 + torch.pow(var, 2)))
        loss =   re +   0.001 *  KLD
        return loss, re, KLD

    def _select_scaler(self):

        if self.args.scaler == 'Minmax':
            scaler = MinMaxScaler()

        elif self.args.scaler == 'Standard':
            scaler = StandardScaler()
        else:
            raise NotImplementedError('Check your scaler if  it is Minmax or Standard')
        return scaler

    def train_VAE(self, train_data, model,batch_size):

        optimizer = optim.Adam(model.parameters(), lr=self.args.learning_rate)
        loss_hist = []
        model.train()
        for e in range(self.args.epoch):
            loss_hist.append(0)

            data_loader = DataLoader(train_data, batch_size=batch_size)

            for batch_X, batch_y in data_loader:
                optimizer.zero_grad()
                x_hat, mean, logvar = model(batch_X)
                loss, KLD, re = self._select_criterion(batch_y, x_hat, mean, logvar)
                loss_hist[-1] += loss.item()

                loss.backward()

                optimizer.step()

            print('Epoch:{}, Loss:{}, Re:{}, KLD:{}'.format(e + 1, loss_hist[-1], re.item(), KLD.item()))

        print('Optimization finished')

        plt.figure(figsize=(12, 8))
        plt.plot(loss_hist)
        plt.title('Training Loss')
        plt.xlabel('Iteration of Training')

        return model

    def train(self):

        # 获取第一个chunk

        X_train = self.X_train[0]
        train_data = Dataset_Custom(torch.tensor(X_train, dtype=torch.float32, device=self.device),
                                    torch.tensor(X_train, dtype=torch.float32, device=self.device), mode='2D')

        # 训练第一个 VAE

        sub_VAE = self.train_VAE(train_data, self.subVAE1,32)


        state_dict_model1 = sub_VAE.state_dict()
        state_dict_model2 = self.subVAE2.state_dict()
        for name, param in state_dict_model1.items():
            if name in state_dict_model2 and param.shape == state_dict_model2[name].shape:
                state_dict_model2[name] = param

        self.subVAE2.load_state_dict(state_dict_model2)

        # 训练第二个Chunks
        X_train = self.X_train[1]
        train_data = Dataset_Custom(torch.tensor(X_train, dtype=torch.float32, device=self.device),
                                    torch.tensor(X_train, dtype=torch.float32, device=self.device), mode='2D')
        sub_VAE = self.train_VAE(train_data, self.subVAE2,32)


        state_dict_model1 = sub_VAE.state_dict()
        state_dict_model2 = self.subVAE3.state_dict()
        for name, param in state_dict_model1.items():
            if name in state_dict_model2 and param.shape == state_dict_model2[name].shape:
                state_dict_model2[name] = param
        self.subVAE3.load_state_dict(state_dict_model2)

        # 训练第三个Chunks
        X_train = self.X_train[2]

        train_data = Dataset_Custom(torch.tensor(X_train, dtype=torch.float32, device=self.device),
                                    torch.tensor(X_train, dtype=torch.float32, device=self.device), mode='2D')
        sub_VAE = self.train_VAE(train_data, self.subVAE3,32)


        # 赋值
        state_dict_model1 = sub_VAE.state_dict()
        state_dict_model2 = self.subVAE4.state_dict()
        for name, param in state_dict_model1.items():
            if name in state_dict_model2 and param.shape == state_dict_model2[name].shape:
                state_dict_model2[name] = param
        self.subVAE4.load_state_dict(state_dict_model2)

        # 训练第四个Chunks
        X_train = self.X_train[2]
        x_y_train = np.concatenate((self.X_train[2], self.y_train), axis=1)
        train_data = Dataset_Custom(torch.tensor(X_train, dtype=torch.float32, device=self.device),
                                    torch.tensor(x_y_train, dtype=torch.float32, device=self.device), mode='2D')
        sub_VAE = self.train_VAE(train_data, self.subVAE4,32)
        self.model = sub_VAE
    def test(self, setting):
        x_test = self.X_test[0]
        y_test = self.y_test
        test_data = Dataset_Custom(torch.tensor(x_test, dtype=torch.float32, device=self.device),
                                    torch.tensor(y_test, dtype=torch.float32, device=self.device), mode='2D')

        preds_list = []
        trues_list = []

        self.model.eval()
        with torch.no_grad():
            # 这里设置batch为1就行了
            data_loader = DataLoader(test_data, batch_size=1)
            for i, (batch_x, batch_y) in enumerate(data_loader):
                batch_y = batch_y[:, -1]
                preds, mean, logvar  = self.model(batch_x)
                preds = preds[:, -1]  # 这里的-1可以换成pred_len
                trues = batch_y
                preds = preds.detach().cpu().numpy()
                trues = trues.detach().cpu().numpy()
                preds_list.append(preds)
                trues_list.append(trues)
        preds = np.array(preds_list)
        trues = np.array(trues_list)

        preds = preds.reshape(-1, 1)
        trues = trues.reshape(-1, 1)
        preds = self.y_scaler.inverse_transform(preds)
        trues = self.y_scaler.inverse_transform(trues)
        print("test shape:", preds.shape, trues.shape)
        mae, mse, rmse, mape, mspe = metrics.metric(preds, trues)
        r2 = r2_score(preds, trues)

        print("==== Test Metrics ====")
        print("====== RMSE: {}, MAE: {}, R2: {} ======".format(rmse, mae, r2))

        if self.args.if_ts:
            preds = preds[:, -1, :]
            trues = trues[:, -1, :]

        visual(preds, trues, self.args.if_re)
