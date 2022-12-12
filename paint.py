import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import os

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


def reward():
    rewards = np.load('outputs/UE4 and Airsim/20221202-111003/results/train_rewards.npy')
    ma_rewards = np.load('outputs/UE4 and Airsim/20221202-111003/results/train_ma_rewards.npy')
    sns.set(style='whitegrid')
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.title("learning curve")
    plt.xlabel('train episode')
    plt.ylabel('average reward')
    plt.ylim(-200, 100)
    plt.plot(ma_rewards, label='ma rewards', color='b')
    plt.plot(rewards, color='b', alpha=0.2)
    plt.legend(labels=['ma rewards1'])
    # plt.savefig(curr_path + "/rewards_curve")
    plt.show()


def als():
    als = np.load('outputs/UE4 and Airsim/20221202-111003/results/train_als.npy', allow_pickle=True)
    sns.set(style='whitegrid')
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.title("Actor Loss")
    plt.xlabel('train episode')
    plt.ylabel('loss')
    plt.plot(als, label='AGA', color='b')
    plt.legend(labels=['AGA'])
    # plt.savefig(curr_path + "/rewards_curve")
    plt.show()


def success():
    suc = np.load('outputs/UE4 and Airsim/20221202-111003/results/train_success.npy')
    sns.set(style='whitegrid')
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.title("Success Rate")
    plt.xlabel('train episode')
    plt.ylabel('success rate')
    plt.plot(suc, label='AGA', color='b')
    plt.legend(labels=['AGA'])
    # plt.savefig(curr_path + "/rewards_curve")
    plt.show()


def collision():
    col = np.load('outputs/UE4 and Airsim/20221202-111003/results/train_collision.npy')
    sns.set(style='whitegrid')
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.title("Collision Rate")
    plt.xlabel('train episode')
    plt.ylabel('collision rate')
    plt.plot(col, label='AGA', color='b')
    plt.legend(labels=['AGA'])
    # plt.savefig(curr_path + "/rewards_curve")
    plt.show()


if __name__ == '__main__':
    reward()
