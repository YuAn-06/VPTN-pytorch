"""
Copyright (C) 2023
@ Name: tools.py
@ Time: 2023/10/31 22:48
@ Author: YuAn_L
@ Eamil: yuan_l1106@163.com
@ Software: PyCharm
"""

import math

import torch
import numpy as np
from matplotlib import pyplot as plt

def visual(preds, trues, if_re):
    """

    :param preds: predition values
    :param trues: trues value
    :param if_re: if reconstruction task?
    :return:
    """
    _, D = preds.shape
    print(if_re)
    if if_re != 0:
        plt.figure()

        for i in range(D):

            plt.subplot(4, 5, i + 1)
            plt.plot(preds[:, i], label="Prediction", linewidth=2)
            plt.plot(trues[:, i], label="GroundTruth", linewidth=2)

    else:
        plt.figure(figsize=(20,10))
        plt.plot(trues, label="GroundTruth", linewidth=2)
        # if preds is not None:
        plt.plot(preds, label="Prediction", linewidth=2)
    plt.legend()
    plt.tight_layout()
    plt.show()
