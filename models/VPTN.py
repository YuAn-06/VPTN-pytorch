"""
Copyright (C) 2023
@ Name: VPTN.py
@ Time: 2023/11/2 9:56

@ Software: PyCharm
"""

import torch
import torch.nn as nn


class Encoder(nn.Module):

    def __init__(self, C_in):
        super(Encoder, self).__init__()

        self.input_layer = nn.Linear(C_in, 24, )
        self.encoder_layer1 = nn.Linear(24, 5)
        self.encoder_layer2 = nn.Linear(5, 3)

        self.activation = nn.PReLU()
        # self.LayerNorm = nn.LayerNorm()
        self.Dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.activation(self.input_layer(x))
        x = self.activation(self.encoder_layer1(x))
        x = self.encoder_layer2(x)

        return x


class Decoder(nn.Module):

    def __init__(self, C_in):
        super(Decoder, self).__init__()

        self.input_layer = nn.Linear(2, 3 )
        self.decoder_layer1 = nn.Linear(3, 5)
        self.decoder_layer2 = nn.Linear(5, 24)
        self.decoder_layer3 = nn.Linear(24, C_in)

        self.activation = nn.ReLU()
        # self.LayerNorm = nn.LayerNorm()
        self.Dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.activation(self.input_layer(x))
        # x = self.Dropout(x)
        x = self.activation(self.decoder_layer1(x))
        x = self.activation(self.decoder_layer2(x))
        x = self.decoder_layer3(x)
        return x


class VAEmodel(nn.Module):

    def __init__(self, C_in, C_out,device):
        super(VAEmodel, self).__init__()

        self.C_in = C_in
        self.C_out = C_out
        self.device = device
        self.hidden_num1 = 3
        self.hidden_num2 = 2
        self.encoder = Encoder(C_in=self.C_in,)
        self.decoder = Decoder(C_in=self.C_out)
        self.mean_layer = nn.Linear(self.hidden_num1, self.hidden_num2)
        self.logvar_layer = nn.Linear(self.hidden_num1, self.hidden_num2)
        self.activation = nn.PReLU()
    def reparamterization(self, mean, var):
        # epsilon = torch.randn_like(var).to(self.device)
        epsilon = torch.normal(0, 0.1 , size=var.shape).to(self.device)
        z = mean + var * epsilon
        return z

    def forward(self, x):
        feature = self.encoder(x)

        mean = self.mean_layer(feature)
        var = self.logvar_layer(feature)
        z = self.reparamterization(mean, var)

        x_hat = self.decoder(z)

        return x_hat, mean, var
