import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import os

from torch.nn.utils import rnn
from torch.utils.tensorboard import SummaryWriter
from model.actor import Actor

curr_path = os.path.dirname(os.path.abspath(__file__))  # 当前文件所在绝对路径


def smooth(x, degree):  # degree为平滑等级，0-9，0最低，9最高
    ovr = 0.1 * degree  # 旧值比例
    nvr = 1 - 0.1 * degree  # 新值比例
    x_smooth = []
    for e in x:
        if x_smooth:
            x_smooth.append(ovr * x_smooth[-1] + nvr * e)
        else:
            x_smooth.append(e)
    return x_smooth


def to_pandas_array(x):
    nx = np.array(x)
    data = []
    for i in range(nx.shape[1]):
        for j in range(nx.shape[0]):
            data.append([i+1, nx[j][i]])
    data_df = pd.DataFrame(data, columns=['c1', 'c2'])
    return data_df


if __name__ == '__main__':
    rewards1 = np.load('outputs/UE4 and Airsim/20221127-121730/results/train_rewards.npy')
    ma_rewards1 = np.load('outputs/UE4 and Airsim/20221127-121730/results/train_ma_rewards.npy')
    rewards2 = np.load('outputs/UE4 and Airsim/20221102-100606/results/train_rewards.npy')[:3000]
    ma_rewards2 = np.load('outputs/UE4 and Airsim/20221102-100606/results/train_ma_rewards.npy')[:3000]
    sns.set()
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.title("learning curve")
    plt.xlabel('train episode')
    plt.ylabel('average reward')
    plt.ylim(-200, 100)
    plt.plot(ma_rewards1, label='ma rewards1', color='b')
    plt.plot(ma_rewards2, label='ma rewards2', color='r')
    plt.plot(rewards1, color='b', alpha=0.2)
    plt.plot(rewards2, color='r', alpha=0.2)
    plt.legend(labels=['ma rewards1', 'ma rewards2'])
    plt.savefig(curr_path + "/rewards_curve")
    plt.show()
