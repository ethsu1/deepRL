import gym
from ddpg import *
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
import math
import tensorflow as tf
class DDPGAI:
	def __init__(self, model, env, gamma, batch_size):
		self.env = gym.make(env)
		print(self.env.action_space.high[0])
		self.action_space = self.env.action_space.shape[0]
		self.state_space = self.env.observation_space.shape[0]
		self.scaled_action = self.env.action_space.high[0]
		self.model = model(gamma=gamma, states=self.state_space, actions=self.action_space, batch_size=batch_size, scaled_action=self.scaled_action)
		self.batch_size = batch_size


	#train the model
	def train(self, num_iterations,title):
		x_list = []
		scores_list = []
		score_window = deque(maxlen=100)
		episode = 0
		losslist1 = []
		losslist2 = []
		self.model.copy_target()
		while((np.nanmean(score_window) < 225.0 or math.isnan(np.mean(score_window)))):
			#solvable steps
			state = self.env.reset()
			state = np.reshape(state, (1,self.state_space))
			score = 0
			loss1 = 0
			loss2 = 0
			iterations = []
			steps = 0
			done = False
			while not done and steps < self.env._max_episode_steps:
				self.env.render()
				action = self.model.take_action(state)
				next_state, reward, done, _ = self.env.step(action[0])
				score += reward
				next_state = np.reshape(next_state, (1,self.state_space))
				self.model.add_to_memory((state, next_state, action, reward, done))
				#update target values every few time steps
				#if(i % 5 == 0):
				self.model.update_target()
				if(done):
					break
				#train the model
				if(len(self.model.memory) > self.batch_size):
					temp1, temp2 = self.model.train()
					loss1 += temp1
					loss2 += temp2
				state = next_state
				steps += 1
			x_list.append(episode)
			scores_list.append(score)
			score_window.append(score)
			losslist1.append(loss1)
			losslist2.append(loss2)
			episode += 1
			print("Episode: " + str(episode) + " Score: "+ str(score))
			if(np.mean(score_window) >= 225.0):
				print("Solved at Episode: {} Avg Reward: {}".format(episode, np.mean(score_window)))
				break
		self.env.close()
		self.graph(x_list, losslist1, "iterations", "loss", title+" loss (critic)")
		self.graph(x_list, losslist2, "iterations", "loss", title+" loss (actor)")
		self.graph(x_list, scores_list, "episodes", "scores", title)

		self.model.save()

	def graph(self, x, y, x_label, y_label, title):
		plt.figure()
		plt.plot(x,y)
		plt.xlabel(x_label)
		plt.ylabel(y_label)
		plt.title(title)
		temp = title.lower()
		temp = temp.replace(" ", "_")
		plt.savefig("./images/" + title+'.png')


	def watch(self):
		self.model.load()
		for i in range(10):
			state = self.env.reset()
			score = 0
			for j in range(self.env._max_episode_steps):
				#state = np.asarray(state, dtype=np.float64)
				state = np.reshape(state, (1,self.state_space))
				action = self.model.action(state)
				self.env.render()
				state, reward, done, _ = self.env.step(action[0])
				score += reward
				if done:
					break
			print("Testing: Score: "+ str(score))
		self.env.close()

ddpg = DDPGAI(DDPG, 'LunarLanderContinuous-v2', gamma=0.99, batch_size=64)
#ddpg.train(800, "Deep Deterministic Policy Gradient LunarLanderContinuous")
ddpg.watch()