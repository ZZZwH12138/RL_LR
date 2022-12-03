"""
This part of code is the Q learning brain, which is a brain of the agent.
All decisions are made in here.
"""

import numpy as np
import pandas as pd


class RL(object):
    def __init__(self, action_space, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = action_space  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy

        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0] * len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )

    def choose_action(self, observation):
        self.check_state_exist(observation)
        # action selection
        if np.random.rand() < self.epsilon:
            # choose best action
            state_action = self.q_table.loc[observation, :]
            # some actions may have the same value, randomly choose on in these actions
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        else:
            # choose random action
            action = np.random.choice(self.actions)
        return action

    # def learn(self, *args):
    #     pass


class SarsaLambdaTable(RL):

    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9, trace_decay=0.9):
        super(SarsaLambdaTable, self).__init__(actions, learning_rate, reward_decay, e_greedy)

        self.lambda_ = trace_decay
        self.eligibility_trace = self.q_table.copy()

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            to_be_append = pd.Series(
                [0] * len(self.actions),
                index=self.q_table.columns,
                name=state,
                )
            self.q_table = self.q_table.append(to_be_append)

            self.eligibility_trace = self.eligibility_trace.append(to_be_append)

    def learn(self, s, a, r, s_, a_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, a_]  # next state is not terminal
        #     区别就在这里，Q_learningtable会直接选择预测中的最大的
        else:
            q_target = r  # next state is terminal
        error = q_target - q_predict

        self.eligibility_trace.loc[s, :] *= 0
        self.eligibility_trace.loc[s, a] = 1
        # 这句话意味着当我回访状态s的时候，将进行动作a的重要性化为1，其他的化为0
        # 这意味着在该状态下进行该动作地“不可或缺性”

        # Q update
        self.q_table += self.lr * error * self.eligibility_trace

        # decay衰减 eligibility trace after update
        self.eligibility_trace *= self.gamma*self.lambda_

# 理解一下他的思想就算了，他的意思就是说，我们在最后一步的时候，不仅更新前一步，还更新前面所有经历的步
# 用另一个表记录路径，即我从开始到结束，必须经历的步就写1(对应在状态s时，哪一步可以通往胜利就写1)，其他的赋予0。
# 这样就可以保证在最后更新时，我一次性为必经动作都在增加了价值，让智能体更想去经历这些动作。
