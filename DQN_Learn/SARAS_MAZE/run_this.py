"""
Sarsa is a online updating method for Reinforcement learning.
Unlike Q learning which is a offline updating method, Sarsa is updating while in the current trajectory.
You will see the sarsa is more coward when punishment is close because it cares about all behaviours,
while q learning is more brave because it only cares about maximum behaviour.
"""

from maze_env import Maze
from RL_brain import SarsaTable


def update():
    for episode in range(100):
        # initial observation
        observation = env.reset()

        # RL choose action based on observation
        action = RL.choose_action(str(observation))
        # 相当于每一轮的初始化，选初始化第一个动作

        while True:
            # fresh env
            env.render()

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)

            # RL choose action based on next observation
            action_ = RL.choose_action(str(observation_))
            # 这是与Qlearning所不同的地方，直接利用现在的网络得到a_t+1
            # 也就是说在更新参数之前，就已经确定了现在、将来的动作
            # RL learn from this transition (s, a, r, s, a) ==> Sarsa
            RL.learn(str(observation), action, reward, str(observation_), action_)
            # Saras的transition比Qlearing多一个a_t+1， s、a、r、s、a
            # 说实话我觉得两个算法的差别真的不大，Qtable就是少用了一次choose_action而已，直接从状态表里面选最大的动作
            # 这里也就是直接用了choose_action，没有考虑直接最大化而已，而实际上choose_action函数本身就有90%的可能选择价值最大的动作
            # 只能说相对来讲，有10%的概率，saras算法比q_learning算法更保守，更注重安全
            # swap observation and action
            observation = observation_
            action = action_

            # break while loop when end of this episode
            if done:
                break

    # end of game
    print('game over')
    env.destroy()

if __name__ == "__main__":
    env = Maze()
    RL = SarsaTable(actions=list(range(env.n_actions)))

    env.after(100, update)
    env.mainloop()