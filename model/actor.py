import torch
import math
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils import rnn


class Actor(nn.Module):
    """
    input_dim: 输入维度，这里等于n_states
    output_dim: 输出维度，这里等于n_actions
    max_action: action的最大值
    """

    def __init__(self, n_states, n_actions, max_action, device, n_move, n_sensor, init_w=3e-3):
        super(Actor, self).__init__()
        self.device = device
        self.n_move = n_move
        self.n_sensor = n_sensor

        self.lstm = nn.LSTM(n_states + n_actions, 128, 2,
                            batch_first=True)  # 处理历史轨迹(input_size, hidden_size, num_layers)
        self.la = nn.Linear(128, 256)  # 处理Attention，ht最外层有两层，合并之后为(256, 256, 1)
        self.lga1 = nn.Linear(1, n_states)  # 处理Guide_Attention
        self.lga2 = nn.Linear(n_states, 128)  # 处理Guide_Attention
        self.lga3 = nn.Linear(128, n_states)  # 处理Guide_Attention
        self.l1 = nn.Linear(128 + n_states, 256)  # 处理当前的状态数据
        self.l2 = nn.Linear(256, 128)  # 处理当前的状态数据
        self.l3 = nn.Linear(128, n_actions)
        self.max_action = max_action

        nn.init.uniform_(self.l3.weight.detach(), a=-init_w, b=init_w)
        nn.init.uniform_(self.l3.bias.detach(), a=-init_w, b=init_w)
        nn.init.uniform_(self.la.weight.detach(), a=-.1, b=.1)
        nn.init.uniform_(self.la.bias.detach(), a=-.1, b=.1)
        nn.init.uniform_(self.lga3.weight.detach(), a=-.1, b=.1)
        nn.init.uniform_(self.lga3.bias.detach(), a=-.1, b=.1)

    def forward(self, history, state):
        self.lstm.flatten_parameters()  # 提高显存的利用率和效率
        x1, (ht, ct) = self.lstm(history)  # output(batch_size, time_step, hidden_size)
        x1, _ = rnn.pad_packed_sequence(x1, batch_first=True)  # 由packedSequence数据转换成tensor

        # Attention
        k = torch.cat((ht[0], ht[1]), dim=1).unsqueeze(-1)
        u = torch.tanh(self.la(x1))
        d = u.shape[0]
        att = torch.bmm(u, k) / math.sqrt(d)  # 每个时间步的数据都对应一个权重
        att_score = F.softmax(att, dim=1)
        scored_x1 = x1 * att_score
        feat = torch.sum(scored_x1, dim=1)  # 将经过注意力处理的数据相加

        # Guide Attention
        attn_state = self.guide_attention(state)

        x2 = torch.cat([feat, attn_state], 1)
        x2 = F.relu(self.l1(x2))
        x2 = F.relu(self.l2(x2))
        action = torch.tanh(self.l3(x2))  # torch.tanh与F.tanh没有区别

        return self.max_action * action

    def guide_attention(self, state):
        s = state.unsqueeze(-1)  # 增加一维的state
        q = F.relu(self.lga1(s))
        q = F.relu(self.lga2(q))
        q = torch.tanh(self.lga3(q))
        batch_size = q.shape[0]
        mask_move = torch.zeros(batch_size, self.n_move)
        mask_sensor = torch.ones(batch_size, self.n_sensor)
        mask = torch.concat((mask_move, mask_sensor), 1).to(self.device)
        bias_move = torch.ones(batch_size, self.n_move) * 0.25
        bias_sensor = torch.zeros(batch_size, self.n_sensor)
        bias = torch.concat((bias_move, bias_sensor), 1).to(self.device)
        k1 = state * mask + bias
        k1 = k1.unsqueeze(-1)
        # d1 = k1.shape[0]
        attn = torch.bmm(q, k1) / math.sqrt(batch_size)  # 每个时间步的数据都对应一个权重
        attn_score = F.softmax(attn, dim=1)
        attn_state = attn_score * s
        attn_state = attn_state.squeeze(-1)

        return attn_state


if __name__ == '__main__':
    lstm = nn.LSTM(24, 128, 2, batch_first=True)  # 处理历史轨迹(input_size, hidden_size, num_layers)
    la = nn.Linear(128, 256)  # 处理Attention，ht最外层有两层，合并之后为(256, 256, 1)
    actor = Actor(n_states=3 + 1 + 3 + 1 + 13, n_actions=3, max_action=1, device=torch.device('cuda:0'), n_move=8, n_sensor=13)
    print(sum(p.numel() for p in actor.parameters() if p.requires_grad))
