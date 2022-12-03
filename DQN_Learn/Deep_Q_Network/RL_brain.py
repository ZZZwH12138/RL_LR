import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

import copy

np.random.seed(1)
# 生成固定的随机数
torch.manual_seed(1)


# 在神经网络中，参数默认是进行随机初始化的，这里是确定生成随机数种子，固定神经网络初始化的参数。


# define the network architecture
class Net(nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.el = nn.Linear(n_feature, n_hidden)
        self.q = nn.Linear(n_hidden, n_output)
        # 总共两层,都是全连接网络

    def forward(self, x):
        x = self.el(x)
        x = F.relu(x)
        x = self.q(x)
        return x
        # 输入特征,经过全连接层后用relu激活非线性,最后输出


# 以下就是RL_brain的主要内容了
class DeepQNetwork:
    def __init__(self, n_actions, n_features, n_hidden=20, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9,
                 replace_target_iter=200, memory_size=500, batch_size=32, e_greedy_increment=None,
                 ):
        # 每200轮更新一个target_network的参数
        # experience_replay存储500条
        self.n_actions = n_actions
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        # 采取贪婪策略的概率
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        #
        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))
        # 状态s由n_features组成，存状态就要把n_features存进去，s与s_，所以*2，最后+2是指a与r
        self.loss_func = nn.MSELoss()
        # mean squared error，均方误差
        self.cost_his = []
        # 存放损失
        self._build_net()

    def _build_net(self):
        self.q_eval = Net(self.n_features, self.n_hidden, self.n_actions)
        # 建立一个q网络，用于估计
        self.q_target = Net(self.n_features, self.n_hidden, self.n_actions)
        # 建立一个目标网络，用于计算td_target
        self.optimizer = torch.optim.RMSprop(self.q_eval.parameters(), lr=self.lr)
        # 初始化优化器，使用RMSprop，指定其更新的参数 为q_eval的参数，即只通过优化器更新估值网络的参数

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        # 如果没有memory_counter就初始化memory_counter
        transition = np.hstack((s, [a, r], s_))
        # np.hstack可以把数组组合在一起
        # [a, r]是将a和r合并成一个数组
        # 这一行代码如果不懂可以参考illustration

        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        # 给新来的transition附一个index索引
        self.memory[index, :] = transition
        # 将index索引下的内容（ : ）,替换为transition
        self.memory_counter += 1
        # 存储量+1

    def choose_action(self, observation):
        observation = torch.Tensor(observation[np.newaxis, :])
        # observation[np.newaxis, :]这个是给矩阵增加一个维度的操作
        # 不理解可以参考illustration，可能是转化成张量前需要这么做？
        # 先转化成tensor才可以输入到神经网络中
        if np.random.uniform() < self.epsilon:
            # 0-1之间随机取一个数
            actions_value = self.q_eval(observation)
            # 神经网络的输出是各个动作的价值，是张量
            action = np.argmax(actions_value.data.numpy())
            # actions_value.data.numpy()先转化为numpy
            # 选择其中动作价值最大的动作（argmmax()返回的是编号）
        else:
            action = np.random.randint(0, self.n_actions)
            # np.random.randint随机返回一个整形，限定的范围为0-self.actions，从0-self.n_actions之间返回一个整形数
            # 每个动作都有自己的编号，所以返回动作的编号就好了
            # 想详细了解np.random.randint可以参考illustration中的解释
        return action

    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.q_target.load_state_dict(self.q_eval.state_dict())
            # 每200轮，将估值网络的参数赋予target网络
            print("\ntarget params replaced\n")

        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
            #   从0-memory_size之间选取数字组成一个一位数组，数组大小为batch_size
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
            #   如果还没存满，那只能从目前存的里面随机取
        batch_memory = self.memory[sample_index, :]
        # 拿出来用于训练的是从memory中根据抽取的索引取出来的transtion
        # q_next is used for getting which action would be choosed by target network in state s_(t+1)
        q_next = self.q_target(torch.Tensor(batch_memory[:, -self.n_features:]))
        q_eval = self.q_eval(torch.Tensor(batch_memory[:, :self.n_features]))

        # 注意这里不要搞错了，self.q_eval是定义的神经网络，q_eval则是新定义的一个数组，存放的是神经网络的输出，是各个动作的价值
        # 输出的是一个二维数组，[batch_size,n_actions]
        # batch_memory[:, :self.n_features],:self.n_features的含义是从最左边开始，把数组中前self.n_features个数取出来，实际上这几个变量代表着当前的状态s
        # batch_memory[:, -self.n_features:]，-self.n_features：的含义是从最右边开始把数组中后面的self.n_features个数取出来，这几个则代表着下一时刻的状态s_t+1

        # used for calculating y, we need to copy for q_eval because this operation could keep the Q_value that has not been selected unchanged,
        # so when we do q_target - q_eval, these Q_value become zero and wouldn't affect the calculation of the loss
        q_target = torch.Tensor(q_eval.data.numpy().copy())
        # 这个是TD_target，我们的目标是想让q_eval尽可能接近q_target
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        # 生成数组0、1、2、3......31
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        # 预测的当前动作a_t的编号
        reward = torch.Tensor(batch_memory[:, self.n_features + 1])
        q_target[batch_index, eval_act_index] = reward + self.gamma * torch.max(q_next, 1)[0]
        # batch_index代表的是这是第几条数据，eval_act_index代表的是执行的第几个动作，等式右边是对该状态下执行该动作价值的评判，是由真实的环境给的reward和由target网络得到的该状态下的最优动作价值组成
        # torch.max(a,1)返回每一行中最大值的那个元素，且返回其索引（返回最大元素在这一行的列索引），详细可见illustration
        # [0]是因为torch.max(q_next, 1)不仅返回了每一组的最大动作的价值，还返回了它在那一组的编号，所以[0]，就把价值取出来了

        # 这里我们会遇到一个很有意思的问题，其实每次神经网络都会输出各个动作的价值，而每一个transion中其实就只针对一个动作做更新，我们希望的是eval网络对于该动作价值的评估更接近真实
        # 所以实际上loss的产生就是在这些特定动作上的差值，其他的动作我们通过q_target = torch.Tensor(q_eval.data.numpy().copy())就可以保证差值为零，对loss无影响
        loss = self.loss_func(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # increase epsilon
        self.cost_his.append(loss)
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        # 贪婪策略，慢慢增加贪心的概率，这样在一开始贪心概率较低的时候，可以让智能体尽可能多地尝试各种动作
        self.learn_step_counter += 1

    def plot_cost(self):
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()
