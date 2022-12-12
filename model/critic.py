import torch
import math
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils import rnn


class Critic(nn.Module):
    def __init__(self, n_states, n_actions, init_w=3e-3):
        super().__init__()
        # Q1 architecture
        self.lstm_1 = nn.LSTM(n_states + n_actions, 128, 2, batch_first=True)  # 处理历史轨迹(input_size, hidden_size, num_layers)
        self.la_1 = nn.Linear(128, 256)  # 处理Attention，ht最外层有两层，合并之后为(256, 256, 1)
        self.l1_1 = nn.Linear(128 + n_states + n_actions, 256)  # 处理当前的状态+动作数据
        self.l2_1 = nn.Linear(256, 128)  # 处理当前的状态数据
        self.l3_1 = nn.Linear(128, 1)

        # nn.init.uniform_(self.l3_1.weight.detach(), a=-init_w, b=init_w)
        # nn.init.uniform_(self.l3_1.bias.detach(), a=-init_w, b=init_w)
        nn.init.uniform_(self.la_1.weight.detach(), a=-.1, b=.1)
        nn.init.uniform_(self.la_1.bias.detach(), a=-.1, b=.1)

        # Q2 architecture
        self.lstm_2 = nn.LSTM(n_states + n_actions, 128, 2, batch_first=True)  # 处理历史轨迹(input_size, hidden_size, num_layers)
        self.la_2 = nn.Linear(128, 256)  # 处理Attention，ht最外层有两层，合并之后为(256, 256, 1)
        self.l1_2 = nn.Linear(128 + n_states + n_actions, 256)  # 处理当前的状态+动作数据
        self.l2_2 = nn.Linear(256, 128)  # 处理当前的状态数据
        self.l3_2 = nn.Linear(128, 1)

        # nn.init.uniform_(self.l3_2.weight.detach(), a=-init_w, b=init_w)
        # nn.init.uniform_(self.l3_2.bias.detach(), a=-init_w, b=init_w)
        nn.init.uniform_(self.la_2.weight.detach(), a=-.1, b=.1)
        nn.init.uniform_(self.la_2.bias.detach(), a=-.1, b=.1)

    def forward(self, history, state, action):
        self.lstm_1.flatten_parameters()  # 提高显存的利用率和效率
        x1_1, (ht_1, ct_1) = self.lstm_1(history)  # output(batch_size, time_step, hidden_size)
        x1_1, _ = rnn.pad_packed_sequence(x1_1, batch_first=True)  # 由packedSequence数据转换成tensor

        # Attention
        u_1 = torch.tanh(self.la_1(x1_1))
        d_1 = u_1.shape[0]
        k_1 = torch.cat((ht_1[0], ht_1[1]), dim=1).unsqueeze(-1)
        att_1 = torch.matmul(u_1, k_1) / math.sqrt(d_1)  # 每个时间步的数据都对应一个权重
        att_score_1 = F.softmax(att_1, dim=1)
        scored_x1_1 = x1_1 * att_score_1
        feat_1 = torch.sum(scored_x1_1, dim=1)

        sa_1 = torch.cat([feat_1, state, action], 1)
        x2_1 = F.relu(self.l1_1(sa_1))
        x2_1 = F.relu(self.l2_1(x2_1))
        # q_1 = torch.tanh(self.l3_1(x2_1))  # torch.tanh与F.tanh没有区别
        q_1 = self.l3_1(x2_1)  # torch.tanh与F.tanh没有区别

        self.lstm_2.flatten_parameters()  # 提高显存的利用率和效率
        x1_2, (ht_2, ct_2) = self.lstm_2(history)  # output(batch_size, time_step, hidden_size)
        x1_2, _ = rnn.pad_packed_sequence(x1_2, batch_first=True)  # 由packedSequence数据转换成tensor

        # Attention
        u_2 = torch.tanh(self.la_2(x1_2))
        d_2 = u_2.shape[0]
        k_2 = torch.cat((ht_2[0], ht_2[1]), dim=1).unsqueeze(-1)
        att_2 = torch.matmul(u_2, k_2) / math.sqrt(d_2)  # 每个时间步的数据都对应一个权重
        att_score_2 = F.softmax(att_2, dim=1)
        scored_x1_2 = x1_2 * att_score_2
        feat_2 = torch.sum(scored_x1_2, dim=1)

        sa_2 = torch.cat([feat_2, state, action], 1)
        x2_2 = F.relu(self.l1_2(sa_2))
        x2_2 = F.relu(self.l2_2(x2_2))
        # q_2 = torch.tanh(self.l3_2(x2_2))  # torch.tanh与F.tanh没有区别
        q_2 = self.l3_2(x2_2)  # torch.tanh与F.tanh没有区别

        return q_1, q_2

    def q1(self, history, state, action):
        self.lstm_1.flatten_parameters()  # 提高显存的利用率和效率
        x1_1, (ht_1, ct_1) = self.lstm_1(history)  # output(batch_size, time_step, hidden_size)
        x1_1, _ = rnn.pad_packed_sequence(x1_1, batch_first=True)  # 由packedSequence数据转换成tensor

        # Attention
        u_1 = torch.tanh(self.la_1(x1_1))
        d_1 = u_1.shape[0]
        k_1 = torch.cat((ht_1[0], ht_1[1]), dim=1).unsqueeze(-1)
        att_1 = torch.matmul(u_1, k_1) / math.sqrt(d_1)  # 每个时间步的数据都对应一个权重
        att_score_1 = F.softmax(att_1, dim=1)
        scored_x1_1 = x1_1 * att_score_1
        feat_1 = torch.sum(scored_x1_1, dim=1)

        sa_1 = torch.cat([feat_1, state, action], 1)
        x2_1 = F.relu(self.l1_1(sa_1))
        x2_1 = F.relu(self.l2_1(x2_1))
        # q_1 = torch.tanh(self.l3_1(x2_1))  # torch.tanh与F.tanh没有区别
        q_1 = self.l3_1(x2_1)  # torch.tanh与F.tanh没有区别

        return q_1


if __name__ == '__main__':
    lstm = nn.LSTM(24, 128, 2, batch_first=True)  # 处理历史轨迹(input_size, hidden_size, num_layers)
    la = nn.Linear(128, 256)  # 处理Attention，ht最外层有两层，合并之后为(256, 256, 1)
    critic = Critic(n_states=3 + 1 + 3 + 1 + 13, n_actions=3)
    print(sum(p.numel() for p in critic.parameters() if p.requires_grad))
