"""
This part of code is the Q learning brain, which is a brain of the agent.
All decisions are made in here.
"""

import numpy as np
import pandas as pd

class QLearningTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
            self.action = actions
            self.lr = learning_rate
            self.gamma = reward_decay
            # 衰减因子
            self.epsilon = e_greedy
            # 90%概率做贪婪的选择
            self.q_table = pd.DataFrame(colunms=self.actions)
            # 初始化一个空的表格，且列索引为动作

            # 也许是加了self后这些变量在类内的所有函数都会统一为同一个且可以调用?
    def choose_action(self, observation):
        # 根据观测值，即state选择动作
        self.check_state_exit(observation)
        # observation，对当前环境的观测，也就是当前所处的状态state
        # 我们要先检查一下我们的qtable中是否已经包含了这个state(也就意味着智能体有没有经历过该状态)

        if np.random.uniform() < self.epsilon:
        # 0-1之间随机抽取一个数
        # 选择最优动作
            state_action = self.q_table.loc[observation, : ]
            # loc是从DataFrame中获取元素的一种方式
            # loc[标签]，iloc[行列数]
            # 详情见：https://blog.csdn.net/sinat_29675423/article/details/87975489
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
            # 选取该状态下价值最大的动作的索引，即上下左右中的一个state_action[state_action == np.max(state_action)].index
            # 当有些动作价值相等时，随机从中选择一个动作np.random.choice
            # 如果看不懂这行很长的代码请参考illustration.ipynb
        else:
            action = np.random.choice(self.actions)

        return action

    def learn(self, s, a, r, s_):
        # 参数更新部分，即根据当前的状态、动作、、环境回馈的奖励、下一时刻的状态
        # 四个元素组成的transtion，对网络/表格进行更新
        self.check_state_exist(s_)
        # 检验新的s_是否存在
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()  # next state is not terminal
        else:
            q_target = r  # next state is terminal
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)  # update

    def check_state_exist(self, state):
        # 一开始不知道有多少个state，所以有一个探索的过程，发现新的Qtable会加1
        # 一开始Qtable是空的
        if state not in self.q_table.index:
            # 没有在索引中找到name为state的series，则添加一条
            #一开始是没有任何state的，从环境得到的观测值会逐渐增加智能体所拥有的state记忆
            # append new state to q table
            self.q_table = self.q_table.append(
                # dataframe.append是往dataframe中增加一些东西，这里是增加了一个series
                pd.Series(
                    [0]*len(self.actions),
                    # 意思是添加0，[0]*number,添加所有动作总个数的0
                    index = self.q_table.columns,
                    # 索引就是上下左右
                    name = state,
                    # 因为还没有这一个状态，所以新添加一个，即name=state=observation
                )
                # series可以理解为一条，他的索引就是name和index,index又等于原本定义的actions，即上下左右
            )