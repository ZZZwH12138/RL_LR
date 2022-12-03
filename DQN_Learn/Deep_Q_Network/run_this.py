from maze_env import Maze
from RL_brain import DeepQNetwork

def run_maze():
	step = 0
	for episode in range(300):
		print("episode: {}".format(episode))
		observation = env.reset()
		# 初始化
		while True:
			print("step: {}".format(step))
			env.render()
			action = RL.choose_action(observation)
			observation_, reward, done = env.step(action)
			RL.store_transition(observation, action, reward, observation_)
			if (step>200) and (step%5==0):
				RL.learn()
			# 先存够200个再开始有训练，之后每五步更新一次
			# 所以其实只训练了2000次(大概玩300轮游戏需要10000步)
			observation = observation_
			if done:
				break
			step += 1
	print('game over')
	env.destroy()

if __name__ == '__main__':
	env = Maze()
	RL = DeepQNetwork(env.n_actions, env.n_features,
					learning_rate=0.01,
					reward_decay=0.9,
					e_greedy=0.9,
					replace_target_iter=200,
					memory_size=2000
					)
	env.after(100, run_maze)
	env.mainloop()
	RL.plot_cost()