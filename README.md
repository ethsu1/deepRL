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
would be the temporal difference updates aka the new Q value at a particular state
(R(s) + discount * max_a Q(s',a)). The model will output the particular Q value for state and action Q(s,a)
and the 'true value' will be (R(s) + discount * max_a Q(s',a))

# Why use deep Q Learning or Q learning with ML function approximators?
The goal of Q learning is to find the optimal policy by finding the optimal Q-values (aka the max expected future reward of state-action pairs).
In the basic case, the environment contains discrete states and discrete actions. Therefore, one can do Q learning
via values in a lookup table. But when there are continuous-valued states, the lookup table will be infinitely big.
This is where Q learning with some sort of function approximator comes in because we can feed in continuous-valued
states. Now our function approximator represents the lookup table in the basic case.

# Why and when to have experience replay?
Experience replay helps the model learn more quickly because now it is not learning strictly
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
![Image](https://github.com/ethsu1/deepRL/blob/master/images/Linear_Regression_Q_Learning.png?raw=true "linear regression q learning")
![Image](https://github.com/ethsu1/deepRL/blob/master/images/SVM_Regression_Q_Learning.png?raw=true "svm regression q learning")
![Image](https://github.com/ethsu1/deepRL/blob/master/images/Linear_Regression_Q_Learning_loss.png?raw=true "linear regression q learning loss")
![Image](https://github.com/ethsu1/deepRL/blob/master/images/SVM_Regression_Q_Learning_loss.png?raw=true "svm regression q learning loss")
PLACE GIF
PLACE GIF


On the other hand, the neural network was able to find an optimal policy reasonably 
well, considering it finished training in ~425 episodes. 
![Image](https://github.com/ethsu1/deepRL/blob/master/images/Neural_Network_Q_Learning.png?raw=true "neural network q learning")
![Image](https://github.com/ethsu1/deepRL/blob/master/images/Neural_Network_Q_Learning_loss.png?raw=true "neural network q learning loss")

![Alt Text](https://github.com/ethsu1/deepRL/blob/master/images/dqn.gif)
Now that I had my fun messing around with different ML models 
with Q learning, I wanted to recreate models that would "solve" the environment using different reinforcement learning algorithms

# What's the difference between SARSA and Q Learning?
SARSA is an on-policy algorithm while Q learning is off-policy. So this means for SARSA, when updating the Q-values, it follows
the epsilon greedy policy, which is the same policy it uses to interact with the environment. In Q learning, it
takes the absolute greedy policy (aka the action that yields that largest expected reward) but uses the epsilon greedy policy
to interact with environment (hence why Q learning is off-policy). 

# How did deep SARSA do compared to deep Q learning?
SARSA is basically more conservative in a sense because in the earlier episodes, its more likely to take random actions
due to a higher epsilon value. Once the epsilon decays, it begins to take the more optimal actions. In the earlier episodes,
I could tell SARSA was being more conservative because it seemed to hover, seemingly worried to drop down too much. This seems
to make sense because crashing would lead to extremely negative rewards.

![Alt Text](https://github.com/ethsu1/deepRL/blob/master/images/sarsa_beginning.gif)

But it eventually "solved" the environment as seen below (in ~550 episodes)
![Image](https://github.com/ethsu1/deepRL/blob/master/images/SARSA_Neural_Network.png?raw=true "sarsa")
![Image](https://github.com/ethsu1/deepRL/blob/master/images/SARSA_Neural_Network_loss.png?raw=true "sarsa loss")
![Alt Text](https://github.com/ethsu1/deepRL/blob/master/images/sarsa.gif)

# When does Q Learning and SARSA fail?
Q learning (even with its ML variants) doesn't led well to real world scenarios because the action space isn't discrete.
You can no longer find the action that leads to max Q value because there infinite potential actions. For example,
one of the interesting applications of reinforcement learning is robotic control. Take a robotic hand with 5 fingers. For it learn
to pick up an object, there are infinite number of actions (the angle of each finger has infinite number of values, the force applied by each finger has infinite number of values, etc.). So Q learning wouldn't work well in these scenarios. You might be able
to discretize the actions but it probably wouldn't work that well or solve whatever task you are trying to solve.

# What to do for continuous action spaces?


