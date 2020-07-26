import random
import numpy as np
from collections import deque
import torch
import tensorflow as tf
from tensorflow.keras import Model
'''class NN(torch.nn.Module):
	def __init__(self,  input_dim, output_dim):
		super(NN, self).__init__()
		self.layer_1 = torch.nn.Linear(input_dim, 256)
		self.layer_2 = torch.nn.Linear(256, 128)
		self.output_layer = torch.nn.Linear(128, output_dim)

	def forward(self, x):
		layer1 = torch.nn.functional.relu(self.layer_1(x))
		layer2 = torch.nn.functional.relu(self.layer_2(layer1))
		return self.output_layer(layer2)'''
class NN(Model):
	def __init__(self, input_dim, output_dim, batch_size):
		super(NN,self).__init__()
		self.input_layer = tf.keras.layers.InputLayer(input_shape=(batch_size, input_dim))
		self.layer_1 = tf.keras.layers.Dense(128, activation='relu')
		self.layer_2 = tf.keras.layers.Dense(128, activation='relu')
		self.output_layer = tf.keras.layers.Dense(output_dim)

	def call(self,x):
		x = self.input_layer(x)
		x = self.layer_1(x)
		x = self.layer_2(x)
		return self.output_layer(x)
		


class NeuralNetwork:
	def __init__(self,epsilon, alpha, gamma, states, actions, batch_size):
		self.epsilon = epsilon
		self.epsilon_min = 0.01
		self.epsilon_decay = 0.995
		self.gamma = gamma
		self.alpha = alpha
		self.actions = actions
		self.states = states
		self.memory_len = int(1e5)
		self.memory = deque(maxlen=self.memory_len)
		self.batch_size = batch_size
		self.model = NN(self.states, self.actions, batch_size)
		self.targetmodel = NN(self.states, self.actions, batch_size)
		#self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.alpha)
		self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.alpha)
		self.loss = tf.keras.losses.MeanSquaredError()

	def update_target(self):
		'''for target_parameters, q_parameters in zip(self.targetmodel.parameters(), self.model.parameters()):
			target_parameters.data.copy_((1e-3)*q_parameters.data + (1.0-(1e-3))*q_parameters.data)'''
		#self.targetmodel.set_weights(self.model.get_weights())
		for a, b in zip(self.targetmodel.variables, self.model.variables):
			a.assign(b)
	def train(self):
		batch = random.sample(self.memory, self.batch_size)
		state, next_state, action, reward, done = zip(*batch)
	
		'''state_matrix = torch.cat(state)
		next_state_matrix = torch.cat(next_state)
		action_matrix = torch.cat(action)
		reward_matrix = torch.cat(reward)
		done_matrix = torch.cat(done)'''
		state_matrix = tf.concat(state, 0)
		next_state_matrix = tf.concat(next_state,0)
		action_matrix = tf.concat(action, 0)
		reward_matrix = tf.concat(reward, 0)
		done_matrix = tf.concat(done, 0)

		'''
		#get only the q-values of selected action
		#predicted = torch.gather(self.model(state_matrix), 1, action_matrix)
		predicted = tf.gather_nd(self.model.predict(state_matrix), action_matrix)

		#no gradient will be backpropped on this variable
		#target_values = self.targetmodel(next_state_matrix).detach()
		target_values = tf.stop_gradient(self.targetmodel(next_state_matrix))
		

		#get only associated q value for max action (batch_size,)
		#temp, _ = torch.max(target_values,dim=1)
		temp = tf.keras.backend.max(target_values, dim=1)
		#value = torch.unsqueeze(self.targetmodel(next_state_matrix).detach().max(1)[0], 1)
		#reshape to (batch_size,1)
		#target = torch.reshape(temp, (temp.shape[0], 1))
		#target = reward_matrix + self.gamma*(target) *(1-done_matrix)

		#value = reward_matrix + self.gamma*(value) *(1-done_matrix)
		
		self.optimizer.zero_grad()
		#predicted = (64x1) and target = (64x1), we only need to fit 
		#Q(S,A) to R + gamma*max_a Q(s', a), so no need to be 64x4 anymore (waste of computation)
		loss = torch.nn.functional.mse_loss(predicted, target)
		loss.backward()
		self.optimizer.step()
		#self.update_target()
		return loss.item()
		'''
		#print(action_matrix)
		row_indices = tf.range(tf.shape(action_matrix)[0])
		full_indices = tf.stack([row_indices, action_matrix], axis=1)
		with tf.GradientTape() as tape:
			#get only the q-values of selected action
			predicted = self.model(state_matrix)
			predicted = tf.gather_nd(predicted, full_indices)
			predicted = tf.reshape(predicted, (predicted.shape[0],1))
			#no gradient will be backpropped on this variable
			target_values = tf.stop_gradient(self.targetmodel(next_state_matrix))
			#get only associated q value for max action (batch_size,)
			temp = tf.keras.backend.max(target_values, axis=1)
			#reshape to (batch_size,1)
			target = tf.reshape(temp,(temp.shape[0],1))
			#predicted = (64x1) and target = (64x1), we only need to fit 
			#Q(S,A) to R + gamma*max_a Q(s', a), so no need to be 64x4 anymore (waste of computation)
			ones = tf.ones((done_matrix.shape[0], done_matrix.shape[1]))
			#print(done_matrix)
			subtract = tf.math.subtract(ones,done_matrix)
			discounted = tf.math.scalar_mul(tf.constant(self.gamma), target)
			product = tf.math.multiply(discounted, subtract)

			target_values = tf.math.add(reward_matrix, product)
			#target = reward_matrix + self.gamma*target*(1-done_matrix)
			#print(tf.math.equal(target_values, target))
			

			loss = self.loss(target_values, predicted)
		gradients = tape.gradient(loss, self.model.trainable_variables)
		self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
		return tf.keras.backend.get_value(loss)


	#epsilon greedy
	def pick_action(self, state):
		#exploration
		if(random.random() < self.epsilon):
			return random.randrange(self.actions)
		else:
			#exploitation
			#state = torch.from_numpy(state)
			#q_values = self.model(state)
			#return torch.argmax(q_values).item()
			state = tf.convert_to_tensor(state)
			q_values = self.model(state)
			return tf.keras.backend.get_value(tf.keras.backend.argmax(q_values))[0]

	def action(self, state):
		#state = torch.from_numpy(state)
		#q_values = self.model(state)
		#return torch.argmax(q_values).item()
		state = tf.convert_to_tensor(state)
		q_values = self.model(state)
		return tf.keras.backend.get_value(tf.keras.backend.argmax(q_values))[0]

	def add_to_memory(self, data):
		state, next_state, action, reward, done = data
		'''state = torch.from_numpy(state)
		next_state = torch.from_numpy(next_state)
		action = torch.tensor([[action]])
		reward = torch.tensor([[reward]],dtype=torch.float)
		done = torch.tensor([[done]], dtype=torch.float)'''
		state = tf.convert_to_tensor(state)
		next_state = tf.convert_to_tensor(next_state)
		action = tf.convert_to_tensor([action])
		reward = tf.convert_to_tensor([[reward]], dtype=tf.float32)
		done = tf.convert_to_tensor([[done]], dtype=tf.float32)

		select_memory = (state, next_state,action,reward,done)
		self.memory.append(select_memory)


	def save(self):
		self.model.save_weights('./neural_network.pth')

	def load(self):
		self.model.load_weights('./neural_network.pth')

