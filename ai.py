import gym
from linear_regression import *
from neural_network import *
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
import math
import torch
class AI:
	def __init__(self, model, env, epsilon, alpha, gamma, batch_size):
		self.env = gym.make(env)
		self.action_space = self.env.action_space.n
		self.state_space = self.env.observation_space.shape[0]
		self.model = model(epsilon, alpha, gamma, self.state_space, self.action_space, batch_size)
		self.batch_size = batch_size


	#train the model
	def train(self, num_iterations,title):
		x_list = []
		scores_list = []
		score_window = deque(maxlen=100)
		episode = 0
		while np.mean(score_window) < 225 or math.isnan(np.mean(score_window) or episode < 2000):
			#solvable steps
			state = self.env.reset()
			state = np.reshape(state, (1,self.state_space))
			#state = torch.from_numpy(state).float()
			score = 0
			losslist = []
			iterations = []
			for i in range(num_iterations):
				self.env.render()
				action = self.model.pick_action(state)
				next_state, reward, done, _ = self.env.step(action)
				score += reward
				next_state = np.reshape(next_state, (1,self.state_space))
				self.model.add_to_memory((state, next_state, action, reward, done))
				
				self.model.epsilon = max(self.model.epsilon_min, self.model.epsilon*self.model.epsilon_decay)
				#update target values every few time steps
				if(i % 50 == 0):
					self.model.update_target()
				if(done):
					break
				#train the model
				if(len(self.model.memory) > self.batch_size):
					loss = self.model.train()
					losslist.append(loss)
					iterations.append(i)
				state = next_state
			x_list.append(episode)
			scores_list.append(score)
			score_window.append(score)
			episode += 1
			print("Episode: " + str(episode) + " Score: "+ str(score))
			if(np.mean(score_window) >= 225.0):
				print("Solved at Episode: {} Avg Reward: {}".format(episode, np.mean(score_window)))
				break
		self.env.close()
		#self.graph(iterations, losslist, "iterations", "loss")
		self.graph(x_list, scores_list, "episodes", "scores", title)
		self.model.save()

	def graph(self, x, y, x_label, y_label, title):
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
			for j in range(400):
				#state = np.asarray(state, dtype=np.float64)
				state = np.reshape(state, (1,self.state_space))
				action = self.model.action(state)
				self.env.render()
				state, reward, done, _ = self.env.step(action)
				if done:
					break
		self.env.close()

ai1 = AI(NeuralNetwork, 'LunarLander-v2', epsilon=1.0, alpha=5e-4,gamma=0.99, batch_size=64)
ai1.train(800, "Neural Network Q Learning")
#ai1.watch()
ai2 = AI(LinearRegression, 'CartPole-v0', epsilon=1.0, alpha=5e-4,gamma=0.99, batch_size=64)
ai2.train(200, "Linear Regression Q Learning")
#ai2.watch()

