import tensorflow as tf
from tensorflow.keras import Model
from collections import deque
import random
class Actor(Model):
	def __init__(self, input_dim, output_dim, batch_size, scaled_action):
		super(Actor,self).__init__()
		self.input_layer = tf.keras.layers.InputLayer(input_shape=(batch_size, input_dim))
		self.layer1 = tf.keras.layers.Dense(400, activation='relu')
		self.layer2 = tf.keras.layers.Dense(300, activation='relu')
		output_init = tf.keras.initializers.RandomUniform(minval=-0.003, maxval=0.003)
		self.output_layer = tf.keras.layers.Dense(output_dim,activation='tanh', kernel_initializer=output_init)
		#self.batch_normalize = tf.keras.layers.BatchNormalization()
		self.scaled_action = scaled_action

	def call(self,x):
		x = self.input_layer(x)
		x = self.layer1(x)
		#x = self.batch_normalize(x)
		x = self.layer2(x)
		#x = self.batch_normalize(x)
		x = self.output_layer(x)
		#scale the action
		actions = tf.scalar_mul(tf.constant(self.scaled_action), x)
		return actions


class Critic(Model):
	def __init__(self, state_dim, action_dim, batch_size):
		super(Critic,self).__init__()
		self.state_input = tf.keras.layers.InputLayer(input_shape=(batch_size, state_dim))
		self.state_layer1 = tf.keras.layers.Dense(400, activation='relu')
		#self.state_layer2 = tf.keras.layers.Dense(300, activation='relu')
		#self.batch_normalize = tf.keras.layers.BatchNormalization()
		self.action_input = tf.keras.layers.InputLayer(input_shape=(batch_size, action_dim))
		#self.action_layer1 = tf.keras.layers.Dense(300, activation='relu')

		self.concat = tf.keras.layers.Concatenate()
		self.concat_layer1 = tf.keras.layers.Dense(300, activation='relu')
		output_init = tf.keras.initializers.RandomUniform(minval=-0.0003, maxval=0.0003)
		self.output_layer = tf.keras.layers.Dense(1,kernel_initializer=output_init)

	#input is a list contain state and action
	#output will be a batch_sizex1 vector where each element is a Q value for a state-action input pair
	def call(self, x):
		states, actions = x
		states = self.state_input(states)
		states = self.state_layer1(states)
		#states = self.batch_normalize(states)
		#states = self.state_layer2(states)
		#states = self.batch_normalize(states)

		actions = self.action_input(actions)
		#actions = self.action_layer1(actions)
		#actions = self.batch_normalize(actions)
		temp = self.concat([states,actions])
		temp = self.concat_layer1(temp)
		#5.temp = self.batch_normalize(temp)
		return self.output_layer(temp)


class DDPG:
	def __init__(self, gamma, states, actions, batch_size, scaled_action):
		self.states = states
		self.actions = actions
		self.gamma = gamma
		self.batch_size = batch_size
		self.critic = Critic(states,actions, batch_size)
		self.actor = Actor(states, actions, batch_size, scaled_action)
		self.target_critic = Critic(states,actions, batch_size)
		self.target_actor = Actor(states, actions, batch_size, scaled_action)
		self.memory_len = int(1e6)
		self.memory = deque(maxlen=self.memory_len)
		self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
		self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
		self.critic_loss = tf.keras.losses.MeanSquaredError()

	def update_target(self):
		tau = 0.001
		for a, b in zip(self.target_critic.variables, self.critic.variables):
			temp = a * (1-tau) + b * tau
			a.assign(temp)
		for c,d in zip(self.target_actor.variables, self.actor.variables):
			temp = c * (1-tau) + d * tau
			c.assign(temp)

	def copy_target(self):
		print("copying")
		for a, b in zip(self.target_critic.variables, self.critic.variables):
			a.assign(b)
		for c,d in zip(self.target_actor.variables, self.actor.variables):
			c.assign(d)

	def train(self):
		#get a batch from memory
		batch = random.sample(self.memory, self.batch_size)
		state, next_state, action, reward, done = zip(*batch)
		state_matrix = tf.concat(state, 0)
		next_state_matrix = tf.concat(next_state,0)
		action_matrix = tf.concat(action, 0)
		reward_matrix = tf.concat(reward, 0)
		done_matrix = tf.concat(done, 0)
		reward_matrix = tf.reshape(reward_matrix, (reward_matrix.shape[0],1))
		done_matrix = tf.reshape(done_matrix, (done_matrix.shape[0],1))

		#training critic
		with tf.GradientTape() as tape:
			next_state_actions = self.target_actor(next_state_matrix)
			next_state_q = self.target_critic([next_state_matrix, next_state_actions])
			ones = tf.ones((done_matrix.shape[0], done_matrix.shape[1]))
			subtract = tf.math.subtract(ones, done_matrix)
			discount = tf.scalar_mul(tf.constant(self.gamma), next_state_q)
			product = tf.math.multiply(discount, subtract)
			#R(s) + gamma * Q(s', a')
			target_q = tf.math.add(reward_matrix, product)
			#Q(s,a)
			predicted_q = self.critic([state_matrix, action_matrix])

			#calculate loss for critic
			criticloss = self.critic_loss(target_q, predicted_q)
		critic_gradients = tape.gradient(criticloss, self.critic.trainable_variables)
		self.critic_optimizer.apply_gradients(zip(critic_gradients, self.critic.trainable_variables))

		#training actor
		with tf.GradientTape() as tape:
			actions = self.actor(state_matrix)

			#find the quality of actor's output
			q_value = self.critic([state_matrix, actions])
			#calculate loss for actor
			actorloss = -tf.math.reduce_mean(q_value)
		actor_gradients = tape.gradient(actorloss, self.actor.trainable_variables)
		self.actor_optimizer.apply_gradients(zip(actor_gradients, self.actor.trainable_variables))
		return tf.keras.backend.get_value(criticloss), tf.keras.backend.get_value(actorloss)

	#action during training to add noise to help with exploration of continuous action sapce
	#noise is sampled from zero-mean normal distribution (as opposed to OU Noise), simpler and works jsut as well according
	#to OpenAI's Spinning Up https://spinningup.openai.com/en/latest/algorithms/ddpg.html
	def take_action(self, state):
		state = tf.convert_to_tensor(state)
		actions = self.actor(state)
		noise_vec = self.noise()
		actions = tf.math.add(actions, noise_vec)
		return actions

	def action(self, state):
		state = tf.convert_to_tensor(state)
		actions = self.actor(state)
		return actions


	def add_to_memory(self,data):
		state, next_state, action, reward, done = data
		state = tf.convert_to_tensor(state, dtype=tf.float32)
		next_state = tf.convert_to_tensor(next_state, dtype=tf.float32)
		action = tf.convert_to_tensor(action, dtype=tf.float32)
		reward = tf.convert_to_tensor(reward, dtype=tf.float32)
		done = tf.convert_to_tensor(done, dtype=tf.float32)
		select_memory = (state, next_state,action,reward,done)
		self.memory.append(select_memory)

	#add noise to help with exploration of continuous action space during training
	def noise(self):
		return tf.random.normal(shape=(1,self.actions),mean=0.0, stddev=0.1)

	def save(self):
		self.actor.save_weights('./bipedal/ddpg_actor_lunar.pth')
		self.critic.save_weights('./bipedal/ddpg_critic_lunar.pth')

	def load(self):
		self.actor.load_weights('./bipedal/ddpg_actor_lunar.pth')
		self.critic.load_weights('./bipedal/ddpg_critic_lunar.pth')
