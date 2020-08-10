# What is this page about?
I've always been interested in machine learning and reinforcement learning. So I wanted
to document this project experimenting with machine learning and reinforcement learning.
I wanted to note down any breakthroughs/insights in my understanding to strengthen my own learning 
as well as potentially help someone else's understanding.


# How does one utilize machine learning models in reinforcement learning algorithms?
Something that always troubled me when trying to understand how machine learning
could be used in the reinforcement learning framework was that in machine learning, 
(at least for supervised learning), there is a label. A label defines a ground-truth output, meaning
for a data in training set, the input data had a corresponding output label (classification) 
or output value (regression). So it made sense of how you could training a machine learning model
on the data. But with reinforcement learning, there is no associated ground-truth label/value. 
I did not understand howw I could manipulate the reinforcement learning framework to satisfy 
the general training framework of machine learning. That's when I learned that the machine learning
model is trying to approximate the Q value table. In classical (strictly) Q learning implementations,
you have a table/dictionary to map state-action pairs to Q-values. In deep Q learning, the machine learning
model is trying to approximate the underlying function of the Q-table. So the 'true values'
would be the temporal difference updates aka the new Q value at a particular state ![equation](https://latex.codecogs.com/gif.latex?R%28s%29%20&plus;%20%5Cgamma%20*%20max_a%20Q%28s%27%2Ca%29). The model will output the particular Q value for state and action Q(s,a)
and the 'true value' will be ![equation](https://latex.codecogs.com/gif.latex?R%28s%29%20&plus;%20%5Cgamma%20*%20max_a%20Q%28s%27%2Ca%29). 



# Why use deep Q Learning or Q learning with ML function approximators?
The goal of Q learning is to find the optimal policy by finding the optimal Q-values (aka the max expected future reward of state-action pairs).
In the basic case, the environment contains discrete states and discrete actions. Therefore, one can do Q learning
via values in a lookup table. But when there are continuous-valued states, the lookup table will be infinitely big.
This is where Q learning with some sort of function approximator comes in because we can feed in continuous-valued
states. Now our function approximator represents the lookup table in the basic case.

# Why and when to have experience replay?
In deep reinforcement learning, many models utilize a replay buffer, which is just a data structure
holding tuples of state, next state, action, and reward. These tuples are just relevant information for
sampling the environment. Experience replay helps the model learn more quickly because now it is not learning strictly
from consecutive samples. The model will sample from its memory and use that sample as input into the
output Q function before calculating the loss between the output Q value and target Q value.
Experience replay will be used once the memory data structure has been filled to whatever batch size specified.
This is when the actual training part of the RL model will occur. So that means in my case,
the very first 64 steps the model will be following the epsilon greedy policy based on the randomized
weights of the machine learning model (neural network, decision tree, etc), so there is technically no training.
After the first 64 steps, there would be training (computing and minimizing loss) after every iteration.

# Why copy weights from the prediction model to the target model?
If we use the same network for target Q-values and prediction Q-values, target Q-values and prediction Q-values
will continue moving with each iterations. It makes it more difficult to get our prediction Q-values to
approximate the target Q-values if the target Q-values keep moving. Hence we have a separate model for
target Q-values and we copy the weights from our prediction model to our target model every so often. This
way the target Q-values are not moving (more stable).

# ML Models Used
I wanted to experiment with using different machine learning models to estimate the Q function. The complexity of the models
ranged from something as simple as linear regression to something more powerful like a neural network. 
It seemed very likely that the environment could not be solved with a linear model, considering when I used
a linear regression model, the agent was training on more episodes of the game and still producing very lackluster 
results. However, it seemed as if the agent was able to learn using linear regression, but it wouldn't have 
created a result that "solved the environment" (LunarLander: have an average score of 200 points over the last 100 episodes). The states of the environment
and respective "good" actions don't necessarily have a linear relationship with the Q-value, which makes sense that Q learning with
a linear regression model would not work well to solve the environment compared to a more complicated model, 
like a neural network (which has nonlinear capabilities). For example, a larger value for a right velocity (state) and adding more
right boost (action) won't necessarily lead to higher expected future reward. After trying to model the Q function
with linear regression and a quasi SVM regression, the end results were not too promising. Watching the agent
in the environment after training of 2000 episodes, you can tell the agent learned something but obviously 
it did not learn the optimal polciy. The rewards for each episode did not increase and the loss did not go down per episode.

Linear regression unable to learn the nonlinear patterns of landing (as seen with its barrel roll attempts)
![Image](https://github.com/ethsu1/deepRL/blob/master/images/Linear_Regression_Q_Learning.png?raw=true "linear regression q learning")
![Image](https://github.com/ethsu1/deepRL/blob/master/images/Linear_Regression_Q_Learning_loss.png?raw=true "linear regression q learning loss")

![alt text](https://github.com/ethsu1/deepRL/blob/master/images/linear_regression.gif?raw=true)


Quasi SVM regression worked to some extent but could not solve the environment.
![Image](https://github.com/ethsu1/deepRL/blob/master/images/SVM_Regression_Q_Learning.png?raw=true "svm regression q learning")
![Image](https://github.com/ethsu1/deepRL/blob/master/images/SVM_Regression_Q_Learning_loss.png?raw=true "svm regression q learning loss")

![alt text](https://github.com/ethsu1/deepRL/blob/master/images/svm.gif?raw=true)


On the other hand, the neural network was able to find an optimal policy reasonably well, considering it finished training in ~425 episodes. 
![Image](https://github.com/ethsu1/deepRL/blob/master/images/Neural_Network_Q_Learning.png?raw=true "neural network q learning")
![Image](https://github.com/ethsu1/deepRL/blob/master/images/Neural_Network_Q_Learning_loss.png?raw=true "neural network q learning loss")

![alt text](https://github.com/ethsu1/deepRL/blob/master/images/dqn.gif?raw=true)
Now that I had my fun messing around with different ML models 
with Q learning, I wanted to recreate models that would "solve" the environment using different reinforcement learning algorithms

# What's the difference between SARSA and Q Learning?
SARSA is an on-policy algorithm while Q learning is off-policy. So this means for SARSA, when updating the Q-values, it follows
the epsilon greedy policy, which is the same policy it uses to interact with the environment. In Q learning, it
takes the absolute greedy policy (aka the action that yields that largest expected reward) but uses the epsilon greedy policy
to interact with environment (hence why Q learning is off-policy). 

Q learning update policy: ![equation](https://latex.codecogs.com/gif.latex?max_a%20Q%28s%27%2Ca%29)
Q learning behavior policy: ![equation](https://latex.codecogs.com/gif.latex?action%20%5Cleft%5C%7B%5Cbegin%7Bmatrix%7D%5E%7B%7D%20random_a%20%3A%20P%28x%29%20%3C%20%5Cvarepsilon%20%5C%5C%20max_aQ%28s%27%2Ca%29%20%3A%20P%28x%29%20%5Cgeq%20%5Cvarepsilon%20%5Cend%7Bmatrix%7D%5Cright.)

SARSA update policy: ![equation](https://latex.codecogs.com/gif.latex?action%20%5Cleft%5C%7B%5Cbegin%7Bmatrix%7D%5E%7B%7D%20random_a%20%3A%20P%28x%29%20%3C%20%5Cvarepsilon%20%5C%5C%20max_aQ%28s%27%2Ca%29%20%3A%20P%28x%29%20%5Cgeq%20%5Cvarepsilon%20%5Cend%7Bmatrix%7D%5Cright.)
SARSA behavior policy: ![equation](https://latex.codecogs.com/gif.latex?action%20%5Cleft%5C%7B%5Cbegin%7Bmatrix%7D%5E%7B%7D%20random_a%20%3A%20P%28x%29%20%3C%20%5Cvarepsilon%20%5C%5C%20max_aQ%28s%27%2Ca%29%20%3A%20P%28x%29%20%5Cgeq%20%5Cvarepsilon%20%5Cend%7Bmatrix%7D%5Cright.)

# How did deep SARSA do compared to deep Q learning?
SARSA is basically more conservative in a sense because in the earlier episodes, its more likely to take random actions
due to a higher epsilon value. Once the epsilon decays, it begins to take the more optimal actions. In the earlier episodes,
I could tell SARSA was being more conservative because it seemed to hover, seemingly worried to drop down too much. This seems
to make sense because crashing would lead to extremely negative rewards.

![alt text](https://github.com/ethsu1/deepRL/blob/master/images/sarsa_beginning.gif?raw=true)

But it eventually "solved" the environment as seen below (in ~550 episodes)
![Image](https://github.com/ethsu1/deepRL/blob/master/images/SARSA_Neural_Network.png?raw=true "sarsa")
![Image](https://github.com/ethsu1/deepRL/blob/master/images/SARSA_Neural_Network_loss.png?raw=true "sarsa loss")
![alt text](https://github.com/ethsu1/deepRL/blob/master/images/sarsa.gif?raw=true)

# When does Q Learning and SARSA fail?
Q learning (even with its ML variants) doesn't led well to real world scenarios because the action space isn't discrete.
You can no longer find the action that leads to max Q value because there infinite potential actions. For example,
one of the interesting applications of reinforcement learning is robotic control. Take a robotic hand with 5 fingers. For it learn
to pick up an object, there are infinite number of actions (the angle of each finger has infinite number of values, the force applied by each finger has infinite number of values, etc.). So Q learning wouldn't work well in these scenarios. You might be able
to discretize the actions but it probably wouldn't work that well or solve whatever task you are trying to solve.

# What to do for continuous action spaces?
Since you can't use Q learning and SARSA for continuous action spaces, we look towards policy gradients. 
Q learning and SARSA are value estimation algorithms where you are essentially choosing an action via Q-values. This where policy gradients
come into play because they work for continuous action spaces. Now because our actions are continuous, whatever functions we
are trying to estimate are differentiable with respect to actions. 
INSERT EQUATION
So now we can directly learn a policy function rather indirectly
learning it via a value function (like Q learning). We can directly learn a policy function because the goal of a policy gradient
is to maximize expected future rewards. Usually a neural network learns the policy function and basically learns a mean and standard deviation 
for each dimension of the action space. For example, there can be 4 actions, like jump, crouch, move left, and move right. But how would we determine
how much to do each action for a given state? For a stochastic policy, we use the neural network and its learned parameters of mean and standard deviation to determine how much of each action do. The mean and standard deviation can be used to specify a Gaussian distribution and we sample 
from that distribution to know how much of each action to do. A policy gradient wants to increase the probabilities of  actions that lead to higher rewards and decreease the probabilities of actions that lead to lower rewards. Hence we sample from those specificed probability distributions 
in order to know how to select actions in a continuous action space.


# Okay, so how do policy gradients related to Deterministic Deep Policy Gradients?
DDPG follows an actor-critic framework where the actor learns a deterministic policy and the critic evaluates the actions the actor
outputs via learning the Q value function. DDPG is like an extension of deep Q learning but for continuous action spaces.
But notice that DDPG learns a deterministic policy. That means it won't be learning the mean and standard deviation of each action dimension as
that would require us to sample from probability distribution to know how much of each action to take, making it a stochastic policy.
So what does the actor neural network output then? The actor network would just ouput the mean of each action dimension and utilize those means
as the actions the agent would take. So basically DDPG learns only the mean rather than the mean and standard deviation.

# How to handle exploration in continuous action spaces?
In deep Q learning, we followed an epsilon greedy policy that allowed the agent to explore and exploit best actions based on a decaying 
epsilon value. It was easy to explore via random actions because we had finite discrete actions. But what about for continuous action
spaces. How would the agent explore the possible action space? In the DDPG paper, the authors sampled a vector from time-correleated OU Noise,
which to my knowledge is some sort of distribution. But it turns out simple mean-zero Gaussian noise works just as well. So when implementing 
my DDPG model, I sampled from the Gaussian distribution that had zero mean and 0.1 standard deviation. I wanted to keep everything relatively 
simple so that I could actually learn all the aspects of the DDPG algorithm/model.

# DDPG Architecture
I more or less followed the architecture laid out in the DDPG paper, having my actor and critic models being a two layer neural network
with hidden dimensions of 400 and 300 respectively. I utilized soft updates to update the target network parameters with the model network parameters.
I specified the learning rates according to the paper as well, with actor's learning rate being 1e-4 and the critic's learning rate being 1e-3.

# Results of DDPG
Training the model took alot longer than DQN because we are exploring continuous actions. I trained the model on LunarLanderContinuous-v2 environment.
The results are below.
![Image](https://github.com/ethsu1/deepRL/blob/master/images/Deep_Deterministic_Policy_Gradient_LunarLanderContinuous_loss_(critic).png?raw=true "Lunar Lander critic loss")
![Image](https://github.com/ethsu1/deepRL/blob/master/images/Deep_Deterministic_Policy_Gradient_LunarLanderContinuous_loss_(actor).png?raw=true "Lunar Lander critic actor")
![Image](https://github.com/ethsu1/deepRL/blob/master/images/Deep_Deterministic_Policy_Gradient_LunarLanderContinuous.png?raw=true "Lunar Lander reward")
![alt text](https://github.com/ethsu1/deepRL/blob/master/images/ddpg_lunar.gif?raw=true)



According the github repo, the BipedalWalker-V3 environment is considered solved after an average score of 300 over the past 100 episodes. However,
it turns out this environment terminates when it gets very close to the edge of the cliff. So it is highly unlikely to even get a score of 300. Therefore, I lowered the "solved" criteria. And the agent still is able to learn to walk and reach the end of the trrain.

![Image](https://github.com/ethsu1/deepRL/blob/master/images/Deep_Deterministic_Policy_Gradient_loss_(critic).png?raw=true "Bipedal critic loss")
![Image](https://github.com/ethsu1/deepRL/blob/master/images/Deep_Deterministic_Policy_Gradient_loss_(actor).png?raw=true "Bipedal critic actor")
![Image](https://github.com/ethsu1/deepRL/blob/master/images/Deep_Deterministic_Policy_Gradient.png?raw=true "Bipedal reward")
![alt text](https://github.com/ethsu1/deepRL/blob/master/images/ddpg_bipedal.gif?raw=true)

As you can tell with the BipedalWalker-v3 environment, DDPG is sort of unstable. It gets close to solving the environment (perhaps with a suboptimal policy as seen with the reward graph) and then sort of regresses before it finds a good policy again and "solves" the environment. But overall the agent does reasonably well, considering it was tripping in the earlier episodes before learning to walk and traverse the terrain in the later episodes.

# Conclusion
Overall I strengthened my knowledge about deep reinforcement learning quite a bit and was able to understand the basics of policy gradient methods.
I now understand the use cases for several different reinforcement learning algorithms. Deep reinforcement learning has become more clear and I think
I have a better foundation to learn about all the other advancement algorithms that have been developed in recent years, such as TRPO and PPO.



